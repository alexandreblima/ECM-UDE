"""
Paper 1 — Experiment 4: Wavelet family comparison for HybridECMNO
==================================================================

Purpose
-------
Compare the effect of different wavelet families on the WNO backbone
of the HybridECMNO model.  The experiment holds all hyperparameters
fixed (width, n_layers, n_levels, paraunitary=True) and varies only
the wavelet used in the DWT/IDWT decomposition.

The four families are chosen to span a progression in regularity and
vanishing moments:
  haar   1 vanishing moment, compact support (1 sample), piecewise
         constant approximation.  Maximum temporal localisation,
         minimum smoothness.
  db4    4 vanishing moments, support 7 samples.  The reference
         wavelet from Experiments 1-3, balancing localisation and
         smoothness.
  sym4   4 vanishing moments, near-symmetric (symlet).  Same number
         of vanishing moments as db4 but with improved phase linearity,
         which may better preserve the shape of voltage transients.
  coif1  2 vanishing moments, support 5 samples.  The coiflet family
         has vanishing moments on both the wavelet AND the scaling
         function, which gives better interpolation properties at the
         cost of larger support per vanishing moment.

Physical hypothesis
-------------------
The voltage residual r(t) = V_true(t) - V_ECM(t) has two spectral
components:
  - Low-frequency: OCV model error, slowly varying with SoC.
  - High-frequency: unmodelled RC dynamics, current-dependent spikes.

Wavelets with more vanishing moments (db4, sym4) should better separate
these components, leading to sparser residual representations and more
effective learning.  Haar, despite its simplicity, may perform well on
the sharp transients of drive cycles due to its maximal temporal
localisation.

Metrics
-------
For each wavelet family (5 seeds each):
  1. Global MAE and RMSE in physical volts.
  2. Per-band MAE decomposition (approximation + 4 detail levels)
     using the SAME wavelet as was used for training, so the
     decomposition matches the model's internal representation.
  3. Sparsity index: percentage of wavelet coefficients needed to
     retain 99% of the energy of the learned residual signal.

Dataset
-------
Panasonic 18650PF, UDDS 25°C.  Same split as Exp. 1.

Output
------
  results/p1_exp4/wavelet_comparison_summary.json
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from wno_battery.data import WindowConfig, make_dataloaders
    from wno_battery.models import (
        HybridECMNO, build_model, count_parameters,
    )
    from wno_battery.training import TrainConfig, train, predict
    from wno_battery.ecm import fit_ecm_1rc
    from wno_battery.metrics import band_mae_decomposition
    from wno_battery.utils import pick_device, set_seed
except ModuleNotFoundError as exc:
    if exc.name != "wno_battery":
        raise
    from data import WindowConfig, make_dataloaders
    from models import HybridECMNO, build_model, count_parameters
    from training import TrainConfig, train, predict
    from ecm import fit_ecm_1rc
    from metrics import band_mae_decomposition
    from utils import pick_device, set_seed


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

MAT_PATH   = "datasets/25degC_UDDS_Pan18650PF.mat"
OUTPUT_DIR = Path("results/p1_exp4")

SEED_MASTER: int = 20260418
N_SEEDS: int = 5

def _generate_seeds(master: int, n: int) -> list[int]:
    rng = np.random.default_rng(master)
    draws = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    return sorted(int(s) for s in draws)

SEEDS: list[int] = _generate_seeds(SEED_MASTER, N_SEEDS)

WINDOW_CFG   = WindowConfig(length=1024, stride=512, normalize=True)
BATCH_SIZE   = 16
VAL_FRACTION = 0.2
N_FEATURES   = 3
DEVICE       = pick_device()

# Wavelet families to compare
WAVELET_FAMILIES: dict[str, str] = {
    "haar": "haar",
    "db4":  "db4",
    "sym4": "sym4",
    "coif1": "coif1",
}

# Shared WNO hyperparameters (only wavelet varies)
WNO_SHARED_KWARGS = dict(
    n_features=N_FEATURES,
    backbone="wno",
    include_ecm_input=False,
    ecm_trainable_params=(),
    use_residual_loss=False,
    residual_space="normalized",
    width=8,
    n_layers=4,
    n_levels=4,
    length=WINDOW_CFG.length,
    paraunitary=True,
)

# Training config for hybrid models (WNO schedule)
TRAIN_CFG = TrainConfig(
    n_epochs=150,
    lr=5e-4,
    weight_decay=1e-5,
    grad_clip=1.0,
    patience=30,
    log_every=10,
    device=DEVICE,
    use_lr_schedule=True,
    warmup_epochs=10,
    min_lr_factor=0.01,
)

# Number of DWT levels for band MAE decomposition
N_DECOMP_LEVELS = 4

# Energy retention threshold for sparsity calculation
ENERGY_THRESHOLD = 0.99


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def extended_metrics(
    preds: torch.Tensor, targs: torch.Tensor,
    denorm_std: float, denorm_mean: float,
) -> dict[str, float]:
    """MAE, RMSE, p95, p99, max_err — all in volts."""
    p = preds * denorm_std + denorm_mean
    t = targs * denorm_std + denorm_mean
    err     = (p - t).flatten()
    abs_err = err.abs()
    return {
        "mae":     float(abs_err.mean().item()),
        "rmse":    float(torch.sqrt((err ** 2).mean()).item()),
        "p95_err": float(torch.quantile(abs_err, 0.95).item()),
        "p99_err": float(torch.quantile(abs_err, 0.99).item()),
        "max_err": float(abs_err.max().item()),
    }


def aggregate_metrics(seed_results: list[dict]) -> dict:
    """Compute mean, std, CV for each metric across seeds."""
    if not seed_results:
        return {}
    keys = seed_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        if isinstance(vals[0], dict):
            # Nested dict (band decomposition) — recurse
            sub_keys = vals[0].keys()
            sub_agg = {}
            for sk in sub_keys:
                sub_vals = [v[sk] for v in vals]
                mean = statistics.mean(sub_vals)
                std  = statistics.stdev(sub_vals) if len(sub_vals) > 1 else 0.0
                sub_agg[sk] = {"mean": mean, "std": std}
            agg[k] = sub_agg
        else:
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            agg[k] = {
                "mean": mean, "std": std,
                "cv_pct": (std / mean * 100) if mean > 0 else 0.0,
                "all": vals,
            }
    return agg


# ─────────────────────────────────────────────────────────────────────
# SPARSITY CALCULATION
# ─────────────────────────────────────────────────────────────────────

def calculate_sparsity(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    wavelet: str = "db4",
    n_levels: int = 4,
    threshold: float = 0.99,
) -> float:
    """Calculate the wavelet sparsity of the learned residual signal.

    For each validation window, the model produces a residual
    r(t) = V_pred(t) - V_ECM(t).  We decompose r(t) into wavelet
    coefficients, sort them by squared magnitude, and find what
    fraction of coefficients is needed to retain `threshold` (99%)
    of the total energy.

    A lower retention rate means a sparser representation — the model
    concentrates its correction into fewer wavelet coefficients.

    Parameters
    ----------
    model     : HybridECMNO model (must support forward_components).
    loader    : DataLoader yielding (X, V) batches.
    device    : Device string.
    wavelet   : Wavelet family for decomposition (should match model).
    n_levels  : DWT decomposition levels.
    threshold : Energy fraction to retain (default 0.99).

    Returns
    -------
    retention_rate : float in [0, 1].  Fraction of coefficients needed
                     to retain `threshold` energy.
    """
    from pytorch_wavelets import DWT1DForward

    dwt = DWT1DForward(J=n_levels, wave=wavelet, mode="symmetric")

    all_coeffs_sq: list[torch.Tensor] = []

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    with torch.no_grad():
        for X_win, V_win in loader:
            X_win = X_win.to(dev)

            # Extract the learned residual
            components = model.forward_components(X_win)
            residual = components["res_norm"]  # (B, L)

            # DWT decomposition: input needs (B, C, L)
            r_3d = residual.unsqueeze(1).float().cpu()
            yl, yh = dwt(r_3d)

            # Collect squared coefficients from all subbands
            all_coeffs_sq.append(yl.flatten() ** 2)
            for detail in yh:
                all_coeffs_sq.append(detail.flatten() ** 2)

    # Sort all squared coefficients in descending order
    all_sq = torch.cat(all_coeffs_sq)
    sorted_sq, _ = torch.sort(all_sq, descending=True)

    # Cumulative energy
    total_energy = sorted_sq.sum().item()
    if total_energy < 1e-12:
        return 1.0  # degenerate: residual is essentially zero

    cumsum = torch.cumsum(sorted_sq, dim=0)
    target = threshold * total_energy

    # Find how many coefficients are needed
    n_needed = int((cumsum >= target).float().argmax().item()) + 1
    n_total  = len(sorted_sq)

    return n_needed / n_total


# ─────────────────────────────────────────────────────────────────────
# RUN ONE SEED
# ─────────────────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    wavelet_name: str,
    wavelet_family: str,
    ecm_params,
    train_stats: dict,
    output_dir: Path,
) -> dict:
    """Train and evaluate one HybridECMNO for one seed and wavelet."""
    set_seed(seed)

    train_loader, val_loader, full_dataset = make_dataloaders(
        MAT_PATH, window_cfg=WINDOW_CFG, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, seed=seed, split_mode="temporal",
    )
    V_std  = full_dataset.stats["V_std"]
    V_mean = full_dataset.stats["V_mean"]

    # Build model with the specified wavelet
    model_kwargs = dict(WNO_SHARED_KWARGS)
    model_kwargs["wavelet"] = wavelet_family
    model_kwargs["train_stats"] = full_dataset.stats
    model_kwargs["ecm_params"] = ecm_params

    model = HybridECMNO(**model_kwargs)
    n_params = count_parameters(model)

    ckpt_name = f"hybrid_wno_{wavelet_name}_seed{seed}_best.pt"
    ckpt_path = str(output_dir / ckpt_name)
    cfg = TrainConfig(**{**TRAIN_CFG.__dict__, "ckpt_path": ckpt_path})

    # Train
    history = train(model, train_loader, val_loader, cfg)

    # Predict on validation set
    preds, targs = predict(model, val_loader, device=DEVICE)

    # Global metrics
    metrics = extended_metrics(preds, targs, V_std, V_mean)

    # Band MAE decomposition (using the same wavelet as the model)
    band_mae = band_mae_decomposition(
        preds, targs,
        n_levels=N_DECOMP_LEVELS,
        wavelet=wavelet_family,
    )
    # Scale to physical volts
    band_mae_phys = {
        band: val * V_std for band, val in band_mae.items()
    }

    # Sparsity calculation
    sparsity = calculate_sparsity(
        model, val_loader, device=DEVICE,
        wavelet=wavelet_family,
        n_levels=N_DECOMP_LEVELS,
        threshold=ENERGY_THRESHOLD,
    )

    return {
        "seed": seed,
        "wavelet": wavelet_name,
        "n_params": n_params,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "metrics_V": metrics,
        "band_mae_V": band_mae_phys,
        "sparsity_99pct": sparsity,
    }


# ─────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────

def _summary_path() -> Path:
    return OUTPUT_DIR / "wavelet_comparison_summary.json"


def _load_or_init_summary() -> dict:
    path = _summary_path()
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_summary(summary: dict) -> None:
    path = _summary_path()
    tmp  = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Paper 1 — Experiment 4: Wavelet Family Comparison")
    print(f"Device        : {DEVICE}")
    print(f"Wavelets      : {list(WAVELET_FAMILIES.keys())}")
    print(f"Seeds         : {N_SEEDS} ({SEEDS[0]}..{SEEDS[-1]})")
    print(f"Paraunitary   : True")
    print(f"Dataset       : {MAT_PATH}")

    # ── Build reference dataset and fit shared ECM ──
    train_loader_ref, val_loader_ref, full_dataset_ref = make_dataloaders(
        MAT_PATH, window_cfg=WINDOW_CFG, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, seed=SEEDS[0], split_mode="temporal",
    )
    n_val = len(val_loader_ref.dataset)
    print(
        f"Windows       : {len(full_dataset_ref)} total "
        f"({len(full_dataset_ref) - n_val} train, {n_val} val)"
    )
    print(
        f"V norm        : mean={full_dataset_ref.stats['V_mean']:.4f} V, "
        f"std={full_dataset_ref.stats['V_std']:.4f} V"
    )

    print("\nFitting shared ECM-1RC anchor ...")
    train_loader_fit = torch.utils.data.DataLoader(
        train_loader_ref.dataset, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False,
    )
    ecm_params = fit_ecm_1rc(
        train_loader_fit, full_dataset_ref.stats, verbose=True,
    )

    # ── Parameter count per wavelet ──
    print("\nModel parameter counts:")
    for wname, wfamily in WAVELET_FAMILIES.items():
        kw = dict(WNO_SHARED_KWARGS)
        kw["wavelet"] = wfamily
        kw["train_stats"] = full_dataset_ref.stats
        kw["ecm_params"] = ecm_params
        m = HybridECMNO(**kw)
        print(f"  hybrid_wno_{wname:5s}: {count_parameters(m):>9,} trainable")
        del m

    # ── Resume ──
    summary = _load_or_init_summary()
    done_keys: set[str] = set()
    for wname, wdata in summary.items():
        for entry in wdata.get("per_seed", []):
            done_keys.add(f"{wname}_{entry['seed']}")

    total_planned = N_SEEDS * len(WAVELET_FAMILIES)
    total_done = len(done_keys)
    if total_done > 0:
        print(f"\nResuming: {total_done}/{total_planned} runs already done.")
    else:
        print(f"\nStarting fresh: {total_planned} total runs.")

    # ── Main loop: wavelet-major, seed-minor ──
    for wname, wfamily in WAVELET_FAMILIES.items():
        summary.setdefault(wname, {"per_seed": []})

        print(f"\n{'=' * 70}")
        print(f"WAVELET: {wname} ({wfamily})")
        print("=" * 70)

        for seed_idx, seed in enumerate(SEEDS, start=1):
            key = f"{wname}_{seed}"
            if key in done_keys:
                print(f"  seed {seed_idx}/{N_SEEDS} (seed={seed})  (done, skipping)")
                continue

            print(f"  seed {seed_idx}/{N_SEEDS} (seed={seed})  training ...",
                  end=" ", flush=True)

            try:
                result = run_one_seed(
                    seed=seed,
                    wavelet_name=wname,
                    wavelet_family=wfamily,
                    ecm_params=ecm_params,
                    train_stats=full_dataset_ref.stats,
                    output_dir=OUTPUT_DIR,
                )
            except Exception as err:
                print(f"FAILED: {type(err).__name__}: {err}")
                continue

            summary[wname]["per_seed"].append(result)
            done_keys.add(key)
            _save_summary(summary)

            m = result["metrics_V"]
            print(
                f"MAE={m['mae']*1000:7.2f} mV  "
                f"RMSE={m['rmse']*1000:7.2f} mV  "
                f"sparsity={result['sparsity_99pct']:.1%}  "
                f"ep={result['best_epoch']:3d}"
            )

    # ── Post-processing: aggregate ──
    print(f"\n{'=' * 70}")
    print("POST-PROCESSING")
    print("=" * 70)

    for wname in WAVELET_FAMILIES:
        wdata = summary.get(wname, {})
        per_seed = wdata.get("per_seed", [])
        if not per_seed:
            continue

        # Aggregate global metrics
        wdata["aggregate_metrics"] = aggregate_metrics(
            [r["metrics_V"] for r in per_seed]
        )

        # Aggregate band MAE
        wdata["aggregate_band_mae"] = aggregate_metrics(
            [r["band_mae_V"] for r in per_seed]
        )

        # Aggregate sparsity
        sparsity_vals = [r["sparsity_99pct"] for r in per_seed]
        sp_mean = statistics.mean(sparsity_vals)
        sp_std  = statistics.stdev(sparsity_vals) if len(sparsity_vals) > 1 else 0.0
        wdata["aggregate_sparsity"] = {
            "mean": sp_mean,
            "std": sp_std,
            "all": sparsity_vals,
        }

        # Best seed
        best = min(per_seed, key=lambda r: r["metrics_V"]["mae"])
        wdata["best_seed"] = best["seed"]
        wdata["best_mae_mV"] = best["metrics_V"]["mae"] * 1000

        print(
            f"  {wname:5s}: MAE={wdata['aggregate_metrics']['mae']['mean']*1000:.2f}"
            f" ± {wdata['aggregate_metrics']['mae']['std']*1000:.2f} mV  "
            f"sparsity={sp_mean:.1%} ± {sp_std:.1%}  "
            f"best_seed={best['seed']}"
        )

    _save_summary(summary)

    # ── Print comparison tables ──
    _print_summary_tables(summary)

    print(f"\nResults -> {_summary_path()}")


# ─────────────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────────────

def _print_summary_tables(summary: dict) -> None:
    """Print formatted comparison tables."""

    wavelets = [w for w in WAVELET_FAMILIES if w in summary
                and summary[w].get("aggregate_metrics")]

    if not wavelets:
        print("\nNo completed results to display.")
        return

    # ── Table 1: Global metrics ──
    print(f"\n{'=' * 80}")
    print(f"GLOBAL METRICS  ({N_SEEDS} seeds, paraunitary=True)")
    print("=" * 80)
    print(
        f"{'wavelet':8}  {'MAE mean':>10}  {'MAE std':>9}  "
        f"{'CV(MAE)':>8}  {'RMSE mean':>10}  "
        f"{'sparsity':>10}  {'best':>6}"
    )
    print(
        f"{'':8}  {'(mV)':>10}  {'(mV)':>9}  {'(%)':>8}  "
        f"{'(mV)':>10}  {'(99% E)':>10}  {'seed':>6}"
    )
    print("-" * 80)
    for w in wavelets:
        d = summary[w]
        agg = d["aggregate_metrics"]
        sp  = d["aggregate_sparsity"]
        print(
            f"{w:8}  "
            f"{agg['mae']['mean']*1000:>10.2f}  "
            f"{agg['mae']['std']*1000:>9.2f}  "
            f"{agg['mae']['cv_pct']:>8.1f}  "
            f"{agg['rmse']['mean']*1000:>10.2f}  "
            f"{sp['mean']:>9.1%}  "
            f"{d['best_seed']:>6}"
        )
    print("=" * 80)

    # ── Table 2: Band MAE decomposition ──
    # Determine band names from first available result
    sample_bands = summary[wavelets[0]]["aggregate_band_mae"]
    band_names = list(sample_bands.keys())

    print(f"\n{'=' * 80}")
    print("BAND MAE (mV)  — per-band mean absolute error in physical volts")
    print("=" * 80)
    header = f"{'wavelet':8}" + "".join(f"  {b:>10}" for b in band_names)
    print(header)
    print("-" * len(header))
    for w in wavelets:
        bands = summary[w]["aggregate_band_mae"]
        row = f"{w:8}"
        for b in band_names:
            val = bands[b]["mean"] * 1000  # to mV
            row += f"  {val:>10.3f}"
        print(row)
    print("=" * 80)

    # ── Table 3: Sparsity comparison ──
    print(f"\n{'=' * 50}")
    print("SPARSITY INDEX  — fraction of coefficients for 99% energy")
    print("=" * 50)
    print(f"{'wavelet':8}  {'mean':>10}  {'std':>10}")
    print("-" * 50)
    for w in wavelets:
        sp = summary[w]["aggregate_sparsity"]
        print(f"{w:8}  {sp['mean']:>10.1%}  {sp['std']:>10.1%}")
    print("=" * 50)

    print("\nInterpretation:")
    print("  Lower sparsity = sparser residual = more efficient representation.")
    print("  Lower band MAE in detail bands = better high-frequency modelling.")


if __name__ == "__main__":
    main()
