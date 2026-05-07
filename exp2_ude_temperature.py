"""
Paper 1 — Experiment 2: Temperature out-of-distribution (ECM vs LSTM vs ECM-UDE)
==================================================================================

Purpose
-------
Evaluate the three Paper 1 models (ECM-1RC, LSTM, ECM-UDE) under
temperature OOD conditions.  No retraining: checkpoints from P1 Exp. 1
(trained on UDDS 25°C) are loaded and applied zero-shot to UDDS cycles
at {10, 0, -10, -20}°C.

This is the experiment where the ECM-UDE should demonstrate its primary
advantage: the ECM structure maintains physically correct OCV(SoC)
behaviour across temperatures, while the neural correction f_theta
adapts the dynamic response.  The LSTM, lacking physical structure,
is expected to degrade more severely at extreme temperatures.

Physical mechanism
------------------
As temperature decreases:
  - R0, R1 increase (electrolyte conductivity drops)
  - tau1 = R1*C1 increases (~15 s at 25°C to >100 s at -20°C)
  - OCV(SoC) shape changes slightly (thermodynamic effect)

The ECM-UDE's OCV polynomial provides a stable voltage anchor even
at unseen temperatures, while the neural correction (which learned
dynamic error patterns at 25°C) degrades gracefully because it was
initialised near zero and contributes only a small perturbation.

Dependencies
------------
  P1 Exp. 1 must have been run first to produce:
    results/p1_exp1/{model}_best.pt
    results/p1_exp1/summary.json

Output
------
  results/p1_exp2/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from wno_battery.data import (
        WindowConfig, BatteryWindowDataset, load_drive_cycle,
        compute_norm_stats,
    )
    from wno_battery.models import build_model, count_parameters
    from wno_battery.training import predict
    from wno_battery.ecm import fit_ecm_1rc, predict_ecm
    from wno_battery.ude import build_ecm_ude, ECMUDE
    from wno_battery.utils import pick_device, set_seed
except ModuleNotFoundError as exc:
    if exc.name != "wno_battery":
        raise
    from data import (
        WindowConfig, BatteryWindowDataset, load_drive_cycle,
        compute_norm_stats,
    )
    from models import build_model, count_parameters
    from training import predict
    from ecm import fit_ecm_1rc, predict_ecm
    from ude import build_ecm_ude, ECMUDE
    from utils import pick_device, set_seed


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("datasets")

TRAIN_MAT = DATASET_DIR / "25degC_UDDS_Pan18650PF.mat"

TEST_CYCLES: dict[str, Path] = {
    "10C":  DATASET_DIR / "10degC_UDDS_Pan18650PF.mat",
    "0C":   DATASET_DIR / "0degC_UDDS_Pan18650PF.mat",
    "-10C": DATASET_DIR / "n10degC_UDDS_Pan18650PF.mat",
    "-20C": DATASET_DIR / "n20degC_UDDS_Pan18650PF.mat",
}

EXP1_DIR   = Path("results/p1_exp1")
OUTPUT_DIR = Path("results/p1_exp2")

WINDOW_CFG = WindowConfig(length=1024, stride=512, normalize=True)
N_FEATURES = 3
DEVICE     = pick_device()
SEED       = 42
BATCH_SIZE = 32

# Model names and kwargs — must match P1 Exp. 1 exactly
MODEL_NAMES = ["ecm_1rc", "lstm", "ecm_ude"]

LSTM_KWARGS = dict(
    n_features=N_FEATURES,
    hidden_size=32,
    num_layers=2,
    dropout=0.0,
)


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def extended_metrics(
    preds: torch.Tensor, targs: torch.Tensor,
    denorm_std: float, denorm_mean: float,
) -> dict[str, float]:
    p = preds * denorm_std + denorm_mean
    t = targs * denorm_std + denorm_mean
    err = (p - t).flatten()
    abs_err = err.abs()
    return {
        "mae":     float(abs_err.mean().item()),
        "rmse":    float(torch.sqrt((err ** 2).mean()).item()),
        "p95_err": float(torch.quantile(abs_err, 0.95).item()),
        "p99_err": float(torch.quantile(abs_err, 0.99).item()),
        "max_err": float(abs_err.max().item()),
    }


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def load_train_stats(train_mat: Path) -> dict[str, float]:
    """Recompute training-split normalisation stats from the Exp. 1 cycle."""
    cycle       = load_drive_cycle(train_mat)
    cfg_no_norm = WindowConfig(
        length=WINDOW_CFG.length, stride=WINDOW_CFG.stride, normalize=False,
    )
    ds = BatteryWindowDataset(cycle, cfg_no_norm)

    n        = len(ds)
    n_val    = max(1, int(n * 0.2))
    n_train  = n - n_val
    L, S     = WINDOW_CFG.length, WINDOW_CFG.stride
    overlap  = max(0, (L + S - 1) // S - 1)
    train_end = max(0, n_train - overlap)
    train_indices = list(range(0, train_end))

    I_tr, V_tr, T_tr = ds.raw_samples_for_indices(train_indices)
    return compute_norm_stats(I_tr, V_tr, T_tr)


def make_test_loader(mat_path: Path, train_stats: dict):
    cycle   = load_drive_cycle(mat_path)
    dataset = BatteryWindowDataset(cycle, WINDOW_CFG, stats=train_stats)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )
    return loader, dataset


ECM_CACHE_PATH = EXP1_DIR / "ecm_params_cache.json"


def load_ecm_params(train_stats: dict):
    """Fit ECM-1RC from the training split (needed for ECM and UDE eval).

    Results are cached to EXP1_DIR/ecm_params_cache.json to avoid
    re-running the expensive scipy optimisation on subsequent calls.
    """
    from ecm import ECMParams
    if ECM_CACHE_PATH.is_file():
        print(f"  Loading cached ECM params from {ECM_CACHE_PATH}")
        with open(ECM_CACHE_PATH) as f:
            return ECMParams.from_dict(json.load(f))

    cycle       = load_drive_cycle(TRAIN_MAT)
    cfg_no_norm = WindowConfig(
        length=WINDOW_CFG.length, stride=WINDOW_CFG.stride, normalize=False,
    )
    ds = BatteryWindowDataset(cycle, cfg_no_norm)
    n       = len(ds)
    n_val   = max(1, int(n * 0.2))
    n_train = n - n_val
    L, S    = WINDOW_CFG.length, WINDOW_CFG.stride
    overlap = max(0, (L + S - 1) // S - 1)
    train_end = max(0, n_train - overlap)
    train_indices = list(range(0, train_end))

    ds.renormalize(train_stats)
    from torch.utils.data import DataLoader, Subset
    train_subset = Subset(ds, train_indices)
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )
    params = fit_ecm_1rc(train_loader, train_stats, verbose=True)
    with open(ECM_CACHE_PATH, "w") as f:
        json.dump(params.to_dict(), f, indent=2)
    print(f"  ECM params cached -> {ECM_CACHE_PATH}")
    return params


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Paper 1 — Experiment 2: Temperature OOD")
    print(f"Device : {DEVICE}")
    print(f"Exp. 1 checkpoints: {EXP1_DIR}")

    # ── Recover training stats ──
    print(f"\nRecovering training stats from: {TRAIN_MAT.name}")
    train_stats = load_train_stats(TRAIN_MAT)
    print(f"  V: mean={train_stats['V_mean']:.4f}, std={train_stats['V_std']:.4f}")
    print(f"  I: mean={train_stats['I_mean']:.4f}, std={train_stats['I_std']:.4f}")
    print(f"  T: mean={train_stats['T_mean']:.4f}, std={train_stats['T_std']:.4f}")

    # ── Fit shared ECM (needed for ecm_1rc and ecm_ude) ──
    print("\nFitting shared ECM-1RC ...")
    ecm_params = load_ecm_params(train_stats)

    # ── Load test cycles ──
    print(f"\nLoading {len(TEST_CYCLES)} test cycles ...")
    test_loaders: dict[str, tuple] = {}
    for temp_label, mat_path in TEST_CYCLES.items():
        if not mat_path.is_file():
            print(f"  WARNING: {mat_path.name} not found — skipping {temp_label}")
            continue
        loader, dataset = make_test_loader(mat_path, train_stats)
        test_loaders[temp_label] = (loader, dataset)
        print(f"  {temp_label:>5s}: {len(dataset):5d} windows")

    if not test_loaders:
        print("No test cycles found. Exiting.")
        return

    # ── Evaluate each model ──
    all_results: dict[str, dict] = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 65}")
        print(f"Evaluating {model_name.upper()}")

        model_results: dict[str, dict] = {}

        for temp_label, (loader, _) in test_loaders.items():
            if model_name == "ecm_1rc":
                preds, targs = predict_ecm(ecm_params, loader, train_stats)
                n_params = 9
            elif model_name == "lstm":
                ckpt_path = EXP1_DIR / "lstm_best.pt"
                if not ckpt_path.is_file():
                    print(f"  WARNING: {ckpt_path} not found — skipping")
                    break
                model = build_model("lstm", **LSTM_KWARGS)
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                )
                n_params = count_parameters(model)
                preds, targs = predict(model, loader, device=DEVICE)
            elif model_name == "ecm_ude":
                ckpt_path = EXP1_DIR / "ecm_ude_best.pt"
                if not ckpt_path.is_file():
                    print(f"  WARNING: {ckpt_path} not found — skipping")
                    break
                model = build_ecm_ude(stats=train_stats, ecm_params=ecm_params)
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                )
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                preds, targs = predict(model, loader, device=DEVICE)
            else:
                continue

            metrics = extended_metrics(
                preds, targs, train_stats["V_std"], train_stats["V_mean"],
            )
            model_results[temp_label] = metrics
            print(
                f"  {temp_label:>5s}:  "
                f"MAE={metrics['mae']*1000:7.2f} mV  "
                f"RMSE={metrics['rmse']*1000:7.2f} mV  "
                f"p99={metrics['p99_err']*1000:7.2f} mV"
            )

        all_results[model_name] = {
            "n_params": n_params if model_results else 0,
            "temperatures": model_results,
        }

    # ── Save ──
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults -> {summary_path}")

    # ── Print tables ──
    temps = list(test_loaders.keys())
    col_w = 10
    header = f"{'model':12}" + "".join(f" {t:>{col_w}}" for t in temps)
    sep = "-" * (12 + (col_w + 1) * len(temps))

    print("\n" + "=" * len(sep))
    print("MAE (mV) — trained on UDDS 25°C, zero-shot on test temperatures")
    print("=" * len(sep))
    print(header)
    print(sep)
    for name in MODEL_NAMES:
        res = all_results.get(name, {}).get("temperatures", {})
        row = f"{name:12}"
        for t in temps:
            if t in res:
                row += f" {res[t]['mae']*1000:>{col_w}.2f}"
            else:
                row += f" {'—':>{col_w}}"
        print(row)

    # Degradation factors
    exp1_summary = EXP1_DIR / "summary.json"
    if exp1_summary.is_file():
        with open(exp1_summary) as f:
            exp1 = json.load(f)

        print(f"\n{'=' * len(sep)}")
        print("Degradation factor (MAE_T / MAE_25C) — >1 means worse")
        print("=" * len(sep))
        print(header)
        print(sep)
        for name in MODEL_NAMES:
            if name not in exp1 or name not in all_results:
                continue
            # Best seed MAE from Exp. 1
            mae_25c = exp1[name]["metrics_V"]["mae"]
            res = all_results[name].get("temperatures", {})
            row = f"{name:12}"
            for t in temps:
                if t in res:
                    factor = res[t]["mae"] / mae_25c
                    row += f" {factor:>{col_w-1}.2f}x"
                else:
                    row += f" {'—':>{col_w}}"
            print(row)
            print(f"  (25°C baseline MAE = {mae_25c*1000:.2f} mV)")
    else:
        print(f"\nNote: {exp1_summary} not found — degradation factors skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
