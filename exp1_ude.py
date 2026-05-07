"""
Paper 1 — Experiment 1: In-distribution comparison (ECM vs LSTM vs ECM-UDE)
============================================================================

Purpose
-------
Compare three voltage estimation approaches on the UDDS 25°C drive cycle:
  1. ECM-1RC (physics-only baseline, identified by least squares)
  2. LSTM (data-driven baseline, no physics)
  3. ECM-UDE (physics-informed: ECM-1RC + neural ODE correction)

The ECM-UDE is warm-started from the ECM-1RC parameters and trained
end-to-end via adjoint sensitivity (torchdiffeq).

This script follows the same structure as the Paper 2 exp1_in_distribution.py:
  - 30 seeds, deterministic generation from master seed
  - Seed-major loop order (resumable)
  - Incremental summary.json with separate history files
  - Reconstructed-trajectory metrics (overlap-averaged windows)
  - Post-processing pass for aggregation and best-seed selection

Dataset
-------
Panasonic 18650PF (NCA, 2.9 Ah), UDDS at 25°C.
Windows: L=1024 (~102 s at 10 Hz), stride=512 (50% overlap).
Temporal split 80/20 with guard band.

Training protocol
-----------------
  ECM-1RC : No gradient training; identified by scipy least_squares.
  LSTM    : Adam, lr=1e-3, warmup 5 epochs, cosine to 1e-5, patience 30.
  ECM-UDE : Adam, lr=3e-4 (lower for ODE stability), warmup 10 epochs,
            cosine to 3e-6, patience 40 (slower convergence expected),
            grad_clip=0.5 (tighter for adjoint gradients).

Output
------
  results/p1_exp1/summary.json
  results/p1_exp1/history_{model}_seed{seed}.json
  results/p1_exp1/{model}_seed{seed}_best.pt
  results/p1_exp1/{model}_best.pt
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from wno_battery.data import WindowConfig, make_dataloaders
    from wno_battery.models import build_model, count_parameters, LSTMBaseline
    from wno_battery.training import TrainConfig, TrainHistory, predict, train
    from wno_battery.ecm import fit_ecm_1rc, predict_ecm
    from wno_battery.ude import build_ecm_ude, ECMUDE
    from wno_battery.utils import pick_device, set_seed
except ModuleNotFoundError as exc:
    if exc.name != "wno_battery":
        raise
    from data import WindowConfig, make_dataloaders
    from models import build_model, count_parameters, LSTMBaseline
    from training import TrainConfig, TrainHistory, predict, train
    from ecm import fit_ecm_1rc, predict_ecm
    from ude import build_ecm_ude, ECMUDE
    from utils import pick_device, set_seed


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

MAT_PATH = "datasets/25degC_UDDS_Pan18650PF.mat"

OUTPUT_DIR = Path("results/p1_exp1")

SEED_MASTER: int = 20260418
N_SEEDS: int = 30

def _generate_seeds(master: int, n: int) -> list[int]:
    rng = np.random.default_rng(master)
    draws = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    return sorted(int(s) for s in draws)

SEEDS: list[int] = _generate_seeds(SEED_MASTER, N_SEEDS)

WINDOW_CFG = WindowConfig(length=1024, stride=512, normalize=True)
BATCH_SIZE   = 16
VAL_FRACTION = 0.2
N_FEATURES   = 3

DEVICE = pick_device()

# ── Model configs ──

MODEL_NAMES = ["ecm_1rc", "lstm", "ecm_ude"]

LSTM_KWARGS = dict(
    n_features=N_FEATURES,
    hidden_size=32,
    num_layers=2,
    dropout=0.0,
)

# ── Training configs per model ──

TRAIN_CFG_LSTM = TrainConfig(
    n_epochs=150,
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=1.0,
    patience=30,
    log_every=10,
    device=DEVICE,
    use_lr_schedule=True,
    warmup_epochs=5,
    min_lr_factor=0.01,
)

TRAIN_CFG_UDE = TrainConfig(
    n_epochs=30,            # UDE reaches best epochs very early on this split
    lr=2e-4,                # Conservative LR to preserve warm-start performance
    weight_decay=1e-5,
    grad_clip=0.5,          # Tighter clipping for adjoint gradients
    patience=8,             # Stop quickly once warm-start optimum is not improving
    log_every=10,
    device=DEVICE,
    use_lr_schedule=True,
    warmup_epochs=3,
    min_lr_factor=0.01,
)


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def extended_metrics(
    preds: torch.Tensor,
    targs: torch.Tensor,
    denorm_std: float,
    denorm_mean: float,
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


def aggregate_metrics(seed_metrics: list[dict[str, float]]) -> dict:
    if not seed_metrics:
        return {}
    keys = seed_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in seed_metrics]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        vals_t = torch.tensor(vals, dtype=torch.float64)
        q25, q75 = torch.quantile(
            vals_t, torch.tensor([0.25, 0.75], dtype=torch.float64)
        ).tolist()
        agg[k] = {
            "mean": mean, "std": std,
            "median": statistics.median(vals),
            "iqr": q75 - q25,
            "min": min(vals), "max": max(vals),
            "cv_pct": (std / mean * 100) if mean > 0 else 0.0,
            "all": vals,
        }
    return agg


def reconstruct_validation_series(
    preds: torch.Tensor,
    full_dataset,
    window_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average overlapping validation windows back onto the raw timeline."""
    preds = torch.as_tensor(preds)
    signal_len = int(full_dataset.V.shape[0])
    length     = int(full_dataset.cfg.length)
    pred_sum   = torch.zeros(signal_len, dtype=preds.dtype)
    counts     = torch.zeros(signal_len, dtype=preds.dtype)

    for pred_win, win_idx in zip(preds, window_indices):
        start = int(full_dataset.starts[int(win_idx)])
        pred_sum[start : start + length] += pred_win
        counts  [start : start + length] += 1

    mask = counts > 0
    return pred_sum[mask] / counts[mask], full_dataset.V[mask]


# ─────────────────────────────────────────────────────────────────────
# RUN FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def run_ecm_seed(
    seed: int,
    ecm_params,
    output_dir: Path,
) -> tuple[dict, dict]:
    """Run ECM-1RC evaluation (no training, just predict)."""
    set_seed(seed)

    train_loader, val_loader, full_dataset = make_dataloaders(
        MAT_PATH, window_cfg=WINDOW_CFG, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, seed=seed, split_mode="temporal",
    )
    V_std  = full_dataset.stats["V_std"]
    V_mean = full_dataset.stats["V_mean"]

    preds, targs = predict_ecm(ecm_params, val_loader, full_dataset.stats)
    val_indices = [int(i) for i in val_loader.dataset.indices]
    pred_series, targ_series = reconstruct_validation_series(
        preds, full_dataset, val_indices,
    )
    metrics = extended_metrics(pred_series, targ_series, V_std, V_mean)

    seed_entry = {
        "seed": seed,
        "n_params": 9,  # 6 OCV + R0 + R1 + C1
        "best_epoch": 0,
        "best_val_loss": 0.0,
        "metrics_V": metrics,
    }
    return seed_entry, {}


def run_lstm_seed(
    seed: int,
    output_dir: Path,
) -> tuple[dict, dict]:
    """Train and evaluate LSTM for one seed."""
    set_seed(seed)

    train_loader, val_loader, full_dataset = make_dataloaders(
        MAT_PATH, window_cfg=WINDOW_CFG, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, seed=seed, split_mode="temporal",
    )
    V_std  = full_dataset.stats["V_std"]
    V_mean = full_dataset.stats["V_mean"]

    model = build_model("lstm", **LSTM_KWARGS)
    ckpt  = str(output_dir / f"lstm_seed{seed}_best.pt")
    cfg   = TrainConfig(**{**TRAIN_CFG_LSTM.__dict__, "ckpt_path": ckpt})

    history = train(model, train_loader, val_loader, cfg)

    preds, _ = predict(model, val_loader, device=DEVICE)
    val_indices = [int(i) for i in val_loader.dataset.indices]
    pred_series, targ_series = reconstruct_validation_series(
        preds, full_dataset, val_indices,
    )
    metrics = extended_metrics(pred_series, targ_series, V_std, V_mean)

    seed_entry = {
        "seed": seed,
        "n_params": count_parameters(model),
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "metrics_V": metrics,
    }
    history_dict = {
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "lr": history.lr,
    }
    return seed_entry, history_dict


def run_ude_seed(
    seed: int,
    ecm_params,
    train_stats_ref: dict,
    output_dir: Path,
) -> tuple[dict, dict]:
    """Train and evaluate ECM-UDE for one seed."""
    set_seed(seed)

    train_loader, val_loader, full_dataset = make_dataloaders(
        MAT_PATH, window_cfg=WINDOW_CFG, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, seed=seed, split_mode="temporal",
    )
    V_std  = full_dataset.stats["V_std"]
    V_mean = full_dataset.stats["V_mean"]

    # Build UDE with warm-start from ECM
    model = build_ecm_ude(
        stats=full_dataset.stats,
        ecm_params=ecm_params,
        solver="rk4",       # Fixed-step solver is substantially faster on MPS
    )

    ckpt = str(output_dir / f"ecm_ude_seed{seed}_best.pt")
    cfg  = TrainConfig(**{**TRAIN_CFG_UDE.__dict__, "ckpt_path": ckpt})

    history = train(model, train_loader, val_loader, cfg)

    preds, _ = predict(model, val_loader, device=DEVICE)
    val_indices = [int(i) for i in val_loader.dataset.indices]
    pred_series, targ_series = reconstruct_validation_series(
        preds, full_dataset, val_indices,
    )
    metrics = extended_metrics(pred_series, targ_series, V_std, V_mean)

    # Log physical parameters after training
    phys = model.physical_params_dict()

    seed_entry = {
        "seed": seed,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "metrics_V": metrics,
        "physical_params": {
            "R0_mOhm": phys["R0_Ohm"] * 1000,
            "R1_mOhm": phys["R1_Ohm"] * 1000,
            "C1_F": phys["C1_F"],
            "tau1_s": phys["tau1_s"],
            "eta": phys["eta"],
        },
    }
    history_dict = {
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "lr": history.lr,
    }
    return seed_entry, history_dict


# ─────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────

def _summary_path() -> Path:
    return OUTPUT_DIR / "summary.json"

def _history_path(model_name: str, seed: int) -> Path:
    return OUTPUT_DIR / f"history_{model_name}_seed{seed}.json"

def _load_or_init_summary() -> dict:
    path = _summary_path()
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for name in MODEL_NAMES:
            data.setdefault(name, {"per_seed": []})
        return data
    return {name: {"per_seed": []} for name in MODEL_NAMES}

def _completed_seeds(summary: dict) -> dict[str, set[int]]:
    return {
        name: {int(r["seed"]) for r in res.get("per_seed", [])}
        for name, res in summary.items()
    }

def _save_summary(summary: dict) -> None:
    path = _summary_path()
    tmp  = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    tmp.replace(path)

def _save_history(model_name: str, seed: int, history: dict) -> None:
    if history:
        with open(_history_path(model_name, seed), "w") as f:
            json.dump(history, f, indent=2)

def _load_history(model_name: str, seed: int) -> dict | None:
    path = _history_path(model_name, seed)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────

def _postprocess(summary: dict) -> None:
    for model_name, res in summary.items():
        per_seed = res.get("per_seed", [])
        if not per_seed:
            print(f"  [postprocess] {model_name}: no seeds, skipping")
            continue

        best = min(per_seed, key=lambda r: r["metrics_V"]["mae"])
        res["best_seed"]     = best["seed"]
        res["n_params"]      = best["n_params"]
        res["best_val_loss"] = best["best_val_loss"]
        res["best_epoch"]    = best["best_epoch"]
        res["metrics_V"]     = best["metrics_V"]

        hist = _load_history(model_name, best["seed"])
        if hist is not None:
            res["history"] = hist

        res["aggregate"] = aggregate_metrics(
            [r["metrics_V"] for r in per_seed]
        )

        # Copy best checkpoint
        src = OUTPUT_DIR / f"{model_name}_seed{best['seed']}_best.pt"
        dst = OUTPUT_DIR / f"{model_name}_best.pt"
        if src.exists():
            shutil.copy(src, dst)
            print(
                f"  [postprocess] {model_name}: best seed={best['seed']} "
                f"(MAE={best['metrics_V']['mae']*1000:.2f} mV) -> {dst.name}"
            )

        # Log UDE physical params evolution
        if model_name == "ecm_ude" and "physical_params" in best:
            pp = best["physical_params"]
            print(
                f"    R0={pp['R0_mOhm']:.2f} mΩ, "
                f"R1={pp['R1_mOhm']:.2f} mΩ, "
                f"C1={pp['C1_F']:.1f} F, "
                f"τ1={pp['tau1_s']:.2f} s, "
                f"η={pp['eta']:.4f}"
            )


def _print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 90)
    print(f"PAPER 1 — EXPERIMENT 1  (UDDS 25°C, in-distribution, {N_SEEDS} seeds)")
    print("=" * 90)
    print(
        f"{'model':12}  {'params':>9}  {'n_done':>6}  "
        f"{'MAE mean':>10}  {'MAE std':>9}  {'CV(MAE)':>8}  "
        f"{'p99 mean':>10}  {'best':>6}"
    )
    print(
        f"{'':12}  {'':>9}  {'':>6}  "
        f"{'(mV)':>10}  {'(mV)':>9}  {'(%)':>8}  {'(mV)':>10}  {'seed':>6}"
    )
    print("-" * 90)
    for name in MODEL_NAMES:
        res = summary.get(name, {})
        per_seed = res.get("per_seed", [])
        n_done = len(per_seed)
        if n_done == 0 or "aggregate" not in res:
            print(f"{name:12}  {'—':>9}  {n_done:>6}  {'(no data)':>10}")
            continue
        agg = res["aggregate"]
        print(
            f"{name:12}  {res['n_params']:>9,}  {n_done:>6}  "
            f"{agg['mae']['mean']*1000:>10.2f}  "
            f"{agg['mae']['std']*1000:>9.2f}  "
            f"{agg['mae']['cv_pct']:>8.1f}  "
            f"{agg['p99_err']['mean']*1000:>10.2f}  "
            f"{res['best_seed']:>6}"
        )
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Paper 1 — Experiment 1: ECM vs LSTM vs ECM-UDE")
    print(f"Device      : {DEVICE}")
    print(f"Master seed : {SEED_MASTER}")
    print(f"N seeds     : {N_SEEDS}")
    print(f"Dataset     : {MAT_PATH}")

    # ── Reference dataset ──
    train_loader_ref, val_loader_ref, full_dataset_ref = make_dataloaders(
        MAT_PATH, window_cfg=WINDOW_CFG, batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION, seed=SEEDS[0], split_mode="temporal",
    )
    n_val = len(val_loader_ref.dataset)
    print(
        f"Windows     : {len(full_dataset_ref)} total "
        f"({len(full_dataset_ref) - n_val} train, {n_val} val)"
    )
    print(
        f"V norm      : mean={full_dataset_ref.stats['V_mean']:.4f} V, "
        f"std={full_dataset_ref.stats['V_std']:.4f} V"
    )

    # ── Fit shared ECM-1RC ──
    print("\nFitting shared ECM-1RC from training split ...")
    train_loader_fit = torch.utils.data.DataLoader(
        train_loader_ref.dataset, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False,
    )
    ecm_params = fit_ecm_1rc(
        train_loader_fit, full_dataset_ref.stats, verbose=True,
    )

    # ── Parameter counts ──
    print("\nModel parameter counts:")
    lstm_model = build_model("lstm", **LSTM_KWARGS)
    ude_model  = build_ecm_ude(stats=full_dataset_ref.stats, ecm_params=ecm_params)
    print(f"  ecm_1rc  :         9 (identified, not trained by gradient)")
    print(f"  lstm     : {count_parameters(lstm_model):>9,}")
    print(f"  ecm_ude  : {sum(p.numel() for p in ude_model.parameters() if p.requires_grad):>9,}")
    del lstm_model, ude_model

    # ── Resume ──
    summary = _load_or_init_summary()
    done    = _completed_seeds(summary)
    total_planned = N_SEEDS * len(MODEL_NAMES)
    total_done    = sum(len(s) for s in done.values())
    if total_done > 0:
        print(f"\nResuming: {total_done}/{total_planned} runs already complete.")
    else:
        print(f"\nStarting fresh: {total_planned} total runs planned.")

    # ── Main loop: seed-major, model-minor ──
    try:
        for seed_idx, seed in enumerate(SEEDS, start=1):
            print(f"\n[seed {seed_idx:>2}/{N_SEEDS}]  seed={seed}  "
                  f"{'=' * 55}")

            for model_name in MODEL_NAMES:
                if seed in done[model_name]:
                    print(f"  {model_name:12}  (done, skipping)")
                    continue

                print(f"  {model_name:12}  ", end="", flush=True)
                try:
                    if model_name == "ecm_1rc":
                        seed_entry, history = run_ecm_seed(
                            seed, ecm_params, OUTPUT_DIR)
                    elif model_name == "lstm":
                        seed_entry, history = run_lstm_seed(
                            seed, OUTPUT_DIR)
                    elif model_name == "ecm_ude":
                        seed_entry, history = run_ude_seed(
                            seed, ecm_params,
                            full_dataset_ref.stats, OUTPUT_DIR)
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                except Exception as err:
                    print(f"FAILED: {type(err).__name__}: {err}")
                    continue

                _save_history(model_name, seed, history)
                summary[model_name]["per_seed"].append(seed_entry)
                done[model_name].add(seed)
                _save_summary(summary)

                m = seed_entry["metrics_V"]
                print(
                    f"MAE={m['mae']*1000:7.2f} mV  "
                    f"RMSE={m['rmse']*1000:7.2f} mV  "
                    f"p99={m['p99_err']*1000:7.2f} mV  "
                    f"ep={seed_entry['best_epoch']:3d}"
                )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Partial results in {_summary_path()}.")
        print("Re-run to resume.")
        sys.exit(130)

    # ── Post-processing ──
    print("\n" + "=" * 90)
    print("POST-PROCESSING")
    print("=" * 90)
    _postprocess(summary)
    _save_summary(summary)

    print(f"\nResults -> {_summary_path()}")
    _print_summary_table(summary)

    print("\nNext steps:")
    print("  python exp2_ude_temperature.py   (temperature OOD)")
    print("  python exp3_ude_cycle.py         (drive cycle OOD)")


if __name__ == "__main__":
    main()
