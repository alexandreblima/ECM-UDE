"""
Paper 1 — Inference-time SoC noise sensitivity on UDDS 25°C
============================================================

Purpose
-------
Quantify how sensitive the three voltage predictors are to uncertainty
in the SoC input channel at inference time. This experiment addresses
the practical limitation that a real BMS would provide SoC with finite
estimation error rather than the reference trajectory derived from the
dataset.

Protocol
--------
- Source setting: UDDS at 25°C, same temporal validation split as Exp. 1
- No retraining
- ECM-1RC uses the identified parameters from Exp. 1
- LSTM and ECM-UDE use the best checkpoints selected in Exp. 1
- Add zero-mean Gaussian noise to the normalized SoC input channel
  before inference, after denormalizing to physical SoC units
- Clip noisy SoC to [0, 1] and renormalize with the source-domain stats
- Repeat each non-zero noise level over multiple Monte Carlo draws

Noise levels
------------
Standard deviation of additive SoC noise in absolute SoC fraction:
  sigma = 0.00, 0.01, 0.02, 0.05
corresponding to 0%, 1%, 2%, and 5% SoC uncertainty.

Output
------
  results/p1_exp1_soc_noise/summary.json
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch

try:
    from wno_battery.data import WindowConfig, make_dataloaders
    from wno_battery.models import build_model, count_parameters
    from wno_battery.training import predict
    from wno_battery.ecm import predict_ecm, ECMParams, _denorm_batch, _norm_voltage, _simulate_ecm
    from wno_battery.ude import build_ecm_ude
    from wno_battery.utils import pick_device
except ModuleNotFoundError as exc:
    if exc.name != "wno_battery":
        raise
    from data import WindowConfig, make_dataloaders
    from models import build_model, count_parameters
    from training import predict
    from ecm import predict_ecm, ECMParams, _denorm_batch, _norm_voltage, _simulate_ecm
    from ude import build_ecm_ude
    from utils import pick_device


MAT_PATH = "datasets/25degC_UDDS_Pan18650PF.mat"
EXP1_DIR = Path("results/p1_exp1")
OUTPUT_DIR = Path("results/p1_exp1_soc_noise")

WINDOW_CFG = WindowConfig(length=1024, stride=512, normalize=True)
BATCH_SIZE = 32
VAL_FRACTION = 0.2
DEVICE = pick_device()

LSTM_KWARGS = dict(
    n_features=3,
    hidden_size=32,
    num_layers=2,
    dropout=0.0,
)

NOISE_STD_LEVELS = [0.00, 0.01, 0.02, 0.05]
N_MONTE_CARLO = 5
RNG_SEED = 20260420


def extended_metrics(
    preds: torch.Tensor,
    targs: torch.Tensor,
    denorm_std: float,
    denorm_mean: float,
) -> dict[str, float]:
    """MAE, RMSE, p95, p99, max_err — all in volts."""
    p = preds * denorm_std + denorm_mean
    t = targs * denorm_std + denorm_mean
    err = (p - t).flatten()
    abs_err = err.abs()
    return {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((err ** 2).mean()).item()),
        "p95_err": float(torch.quantile(abs_err, 0.95).item()),
        "p99_err": float(torch.quantile(abs_err, 0.99).item()),
        "max_err": float(abs_err.max().item()),
    }


def aggregate_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float] | list[float]]:
    """Aggregate mean/std over Monte Carlo draws for each metric."""
    keys = metric_dicts[0].keys()
    out = {}
    for key in keys:
        vals = [m[key] for m in metric_dicts]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[key] = {
            "mean": mean,
            "std": std,
            "all": vals,
        }
    return out


def reconstruct_validation_series(
    preds: torch.Tensor,
    full_dataset,
    window_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average overlapping validation windows back onto the raw timeline."""
    preds = torch.as_tensor(preds)
    signal_len = int(full_dataset.V.shape[0])
    length = int(full_dataset.cfg.length)
    pred_sum = torch.zeros(signal_len, dtype=preds.dtype)
    counts = torch.zeros(signal_len, dtype=preds.dtype)

    for pred_win, win_idx in zip(preds, window_indices):
        start = int(full_dataset.starts[int(win_idx)])
        pred_sum[start : start + length] += pred_win
        counts[start : start + length] += 1

    mask = counts > 0
    return pred_sum[mask] / counts[mask], full_dataset.V[mask]


def perturb_soc_channel(
    X_win: torch.Tensor,
    stats: dict[str, float],
    sigma_soc: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Inject additive Gaussian noise in physical SoC units and renormalize."""
    if sigma_soc <= 0.0:
        return X_win.clone()

    X_noisy = X_win.clone()
    soc_mean = float(stats["SoC_mean"])
    soc_std = float(stats["SoC_std"])
    soc_phys = X_noisy[..., 2] * soc_std + soc_mean
    noise = torch.randn(
        soc_phys.shape,
        generator=generator,
        dtype=soc_phys.dtype,
        device=soc_phys.device,
    ) * sigma_soc
    soc_noisy = (soc_phys + noise).clamp_(0.0, 1.0)
    X_noisy[..., 2] = (soc_noisy - soc_mean) / soc_std
    return X_noisy


@torch.no_grad()
def predict_model_with_soc_noise(
    model,
    loader,
    stats: dict[str, float],
    sigma_soc: float,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference helper for trainable models under noisy SoC input."""
    model = model.to(device).eval()
    preds, targs = [], []
    for X_win, V_win in loader:
        X_noisy = perturb_soc_channel(X_win, stats, sigma_soc, generator=generator)
        preds.append(model(X_noisy.to(device)).cpu())
        targs.append(V_win)
    return torch.cat(preds, dim=0), torch.cat(targs, dim=0)


def predict_ecm_with_soc_noise(
    params: ECMParams,
    loader,
    stats: dict[str, float],
    sigma_soc: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference helper for ECM-1RC under noisy SoC input."""
    if sigma_soc <= 0.0:
        return predict_ecm(params, loader, stats)

    preds_all, targs_all = [], []
    soc_clip = (params.soc_train_min, params.soc_train_max)

    for X_win, V_win in loader:
        X_noisy = perturb_soc_channel(X_win, stats, sigma_soc, generator=generator)
        I_b, soc_b = _denorm_batch(X_noisy, stats)
        V_hat = _simulate_ecm(
            I_b,
            soc_b,
            params.ocv_coeffs,
            params.R0,
            params.R1,
            params.C1,
            soc_clip=soc_clip,
        )
        preds_all.append(_norm_voltage(V_hat, stats))
        targs_all.append(V_win.detach().cpu().numpy().astype(np.float64))

    preds = np.concatenate(preds_all, axis=0)
    targs = np.concatenate(targs_all, axis=0)
    return torch.from_numpy(preds).float(), torch.from_numpy(targs).float()


def evaluate_predictions(
    preds: torch.Tensor,
    targs: torch.Tensor,
    full_dataset,
    val_indices: list[int],
) -> dict[str, float]:
    pred_series, targ_series = reconstruct_validation_series(preds, full_dataset, val_indices)
    return extended_metrics(
        pred_series,
        targ_series,
        full_dataset.stats["V_std"],
        full_dataset.stats["V_mean"],
    )


def load_ecm_params() -> ECMParams:
    with open(EXP1_DIR / "ecm_params_cache.json") as f:
        return ECMParams.from_dict(json.load(f))


def build_eval_context():
    _, val_loader, full_dataset = make_dataloaders(
        MAT_PATH,
        window_cfg=WINDOW_CFG,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=RNG_SEED,
        split_mode="temporal",
    )
    val_indices = [int(i) for i in val_loader.dataset.indices]
    return val_loader, full_dataset, val_indices


def evaluate_ecm(
    ecm_params: ECMParams,
    val_loader,
    full_dataset,
    val_indices: list[int],
) -> dict:
    model_results = {"n_params": 9, "noise_levels": {}}
    for sigma_soc in NOISE_STD_LEVELS:
        key = f"{sigma_soc:.2f}"
        draw_metrics = []
        draws = 1 if sigma_soc == 0.0 else N_MONTE_CARLO
        for draw_idx in range(draws):
            gen = torch.Generator().manual_seed(RNG_SEED + 1000 * int(round(sigma_soc * 100)) + draw_idx)
            preds, targs = predict_ecm_with_soc_noise(
                ecm_params,
                val_loader,
                full_dataset.stats,
                sigma_soc=sigma_soc,
                generator=gen,
            )
            draw_metrics.append(evaluate_predictions(preds, targs, full_dataset, val_indices))

        agg = aggregate_metrics(draw_metrics)
        if sigma_soc == 0.0:
            baseline_mae = agg["mae"]["mean"]
        agg["mae_increase_pct"] = {
            "mean": (agg["mae"]["mean"] / baseline_mae - 1.0) * 100.0,
            "std": 0.0 if sigma_soc == 0.0 else statistics.stdev(
                [(m["mae"] / baseline_mae - 1.0) * 100.0 for m in draw_metrics]
            ),
            "all": [(m["mae"] / baseline_mae - 1.0) * 100.0 for m in draw_metrics],
        }
        model_results["noise_levels"][key] = agg
    return model_results


def evaluate_lstm(
    val_loader,
    full_dataset,
    val_indices: list[int],
) -> dict:
    model = build_model("lstm", **LSTM_KWARGS)
    model.load_state_dict(
        torch.load(EXP1_DIR / "lstm_best.pt", map_location="cpu", weights_only=True)
    )

    model_results = {"n_params": count_parameters(model), "checkpoint": "lstm_best.pt", "noise_levels": {}}
    for sigma_soc in NOISE_STD_LEVELS:
        key = f"{sigma_soc:.2f}"
        draw_metrics = []
        draws = 1 if sigma_soc == 0.0 else N_MONTE_CARLO
        for draw_idx in range(draws):
            gen = torch.Generator().manual_seed(RNG_SEED + 2000 * int(round(sigma_soc * 100)) + draw_idx)
            preds, targs = predict_model_with_soc_noise(
                model,
                val_loader,
                full_dataset.stats,
                sigma_soc=sigma_soc,
                generator=gen,
                device=DEVICE,
            )
            draw_metrics.append(evaluate_predictions(preds, targs, full_dataset, val_indices))

        agg = aggregate_metrics(draw_metrics)
        if sigma_soc == 0.0:
            baseline_mae = agg["mae"]["mean"]
        agg["mae_increase_pct"] = {
            "mean": (agg["mae"]["mean"] / baseline_mae - 1.0) * 100.0,
            "std": 0.0 if sigma_soc == 0.0 else statistics.stdev(
                [(m["mae"] / baseline_mae - 1.0) * 100.0 for m in draw_metrics]
            ),
            "all": [(m["mae"] / baseline_mae - 1.0) * 100.0 for m in draw_metrics],
        }
        model_results["noise_levels"][key] = agg
    return model_results


def evaluate_ude(
    ecm_params: ECMParams,
    val_loader,
    full_dataset,
    val_indices: list[int],
) -> dict:
    model = build_ecm_ude(
        stats=full_dataset.stats,
        ecm_params=ecm_params,
        solver="rk4",
    )
    model.load_state_dict(
        torch.load(EXP1_DIR / "ecm_ude_best.pt", map_location="cpu", weights_only=True)
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_results = {"n_params": n_params, "checkpoint": "ecm_ude_best.pt", "noise_levels": {}}
    for sigma_soc in NOISE_STD_LEVELS:
        key = f"{sigma_soc:.2f}"
        draw_metrics = []
        draws = 1 if sigma_soc == 0.0 else N_MONTE_CARLO
        for draw_idx in range(draws):
            gen = torch.Generator().manual_seed(RNG_SEED + 3000 * int(round(sigma_soc * 100)) + draw_idx)
            preds, targs = predict_model_with_soc_noise(
                model,
                val_loader,
                full_dataset.stats,
                sigma_soc=sigma_soc,
                generator=gen,
                device=DEVICE,
            )
            draw_metrics.append(evaluate_predictions(preds, targs, full_dataset, val_indices))

        agg = aggregate_metrics(draw_metrics)
        if sigma_soc == 0.0:
            baseline_mae = agg["mae"]["mean"]
        agg["mae_increase_pct"] = {
            "mean": (agg["mae"]["mean"] / baseline_mae - 1.0) * 100.0,
            "std": 0.0 if sigma_soc == 0.0 else statistics.stdev(
                [(m["mae"] / baseline_mae - 1.0) * 100.0 for m in draw_metrics]
            ),
            "all": [(m["mae"] / baseline_mae - 1.0) * 100.0 for m in draw_metrics],
        }
        model_results["noise_levels"][key] = agg
    return model_results


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 104)
    print("PAPER 1 — SOC NOISE SENSITIVITY (UDDS 25°C, inference-time perturbation)")
    print("=" * 104)
    print(
        f"{'model':12}  {'sigma':>7}  {'MAE mean':>10}  {'MAE std':>9}  "
        f"{'RMSE mean':>10}  {'MAE inc.':>10}"
    )
    print(
        f"{'':12}  {'(SoC)':>7}  {'(mV)':>10}  {'(mV)':>9}  "
        f"{'(mV)':>10}  {'(%)':>10}"
    )
    print("-" * 104)
    for model_name in ["ecm_1rc", "lstm", "ecm_ude"]:
        for sigma_key, stats in summary[model_name]["noise_levels"].items():
            print(
                f"{model_name:12}  {float(sigma_key):>7.2f}  "
                f"{stats['mae']['mean']*1000:>10.2f}  "
                f"{stats['mae']['std']*1000:>9.2f}  "
                f"{stats['rmse']['mean']*1000:>10.2f}  "
                f"{stats['mae_increase_pct']['mean']:>10.1f}"
            )
        print("-" * 104)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Paper 1 — SoC noise sensitivity")
    print(f"Device         : {DEVICE}")
    print(f"Dataset        : {MAT_PATH}")
    print(f"Noise std SoC  : {NOISE_STD_LEVELS}")
    print(f"Monte Carlo    : {N_MONTE_CARLO}")

    val_loader, full_dataset, val_indices = build_eval_context()
    ecm_params = load_ecm_params()

    summary = {
        "config": {
            "mat_path": MAT_PATH,
            "noise_std_levels_soc": NOISE_STD_LEVELS,
            "n_monte_carlo": N_MONTE_CARLO,
            "noise_mode": "gaussian_iid_per_timestep_in_physical_soc_units",
            "split_mode": "temporal",
            "rng_seed": RNG_SEED,
        }
    }

    print("\nEvaluating ECM-1RC ...")
    summary["ecm_1rc"] = evaluate_ecm(ecm_params, val_loader, full_dataset, val_indices)

    print("\nEvaluating LSTM best checkpoint ...")
    summary["lstm"] = evaluate_lstm(val_loader, full_dataset, val_indices)

    print("\nEvaluating ECM-UDE best checkpoint ...")
    summary["ecm_ude"] = evaluate_ude(ecm_params, val_loader, full_dataset, val_indices)

    out_path = OUTPUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print_summary(summary)
    print(f"\nSaved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
