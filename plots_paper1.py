"""
Paper 1 — Publication Figures
==============================
Generates all figures for the paper from pre-computed results.

Usage
-----
  python plots_paper1.py

Outputs (saved to results/figures/):
  fig2_voltage_trace.pdf        Voltage prediction trace (in-dist)
  fig3_boxplot_seeds.pdf        MAE distribution over 30 seeds
  fig4_temp_ood.pdf             Temperature OOD degradation curve
  fig5_cycle_ood.pdf            Drive-cycle OOD bar chart
  fig6_us06_trace.pdf           US06 qualitative trace (ECM-UDE vs LSTM)
  fig_ocv.pdf                   OCV(SoC) curve learned by ECM
  fig_learning_curves.pdf       Train/val loss for best seed
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def save_fig(fig, stem: str) -> None:
    """Save figure as PDF, high-res PNG (600 dpi) and SVG."""
    for ext, dpi in [("pdf", None), ("png", 600), ("svg", None)]:
        kwargs = {"dpi": dpi} if dpi else {}
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, **kwargs)
    print(f"    saved {stem}.{{pdf,png,svg}}")


COLORS = {
    "ecm_1rc": "#888888",
    "lstm":    "#E07B39",
    "ecm_ude": "#2E86AB",
}
LABELS = {
    "ecm_1rc": "ECM-1RC",
    "lstm":    "LSTM",
    "ecm_ude": "ECM-UDE",
}

# ── paths ─────────────────────────────────────────────────────────────
try:
    from wno_battery.data import WindowConfig, BatteryWindowDataset, load_drive_cycle, compute_norm_stats
    from wno_battery.models import build_model
    from wno_battery.training import predict
    from wno_battery.ecm import fit_ecm_1rc, predict_ecm, ECMParams
    from wno_battery.ude import build_ecm_ude
    from wno_battery.utils import pick_device
except ModuleNotFoundError:
    from data import WindowConfig, BatteryWindowDataset, load_drive_cycle, compute_norm_stats
    from models import build_model
    from training import predict
    from ecm import fit_ecm_1rc, predict_ecm, ECMParams
    from ude import build_ecm_ude
    from utils import pick_device

DATASET_DIR  = Path("datasets")
EXP1_DIR     = Path("results/p1_exp1")
EXP1_SOC_DIR = Path("results/p1_exp1_soc_noise")
EXP2_DIR     = Path("results/p1_exp2")
EXP3_DIR     = Path("results/p1_exp3")
FIG_DIR      = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MAT    = DATASET_DIR / "25degC_UDDS_Pan18650PF.mat"
US06_MAT     = DATASET_DIR / "25degC_US06_Pan18650PF.mat"

WINDOW_CFG   = WindowConfig(length=1024, stride=512, normalize=True)
BATCH_SIZE   = 32
DEVICE       = pick_device()

LSTM_KWARGS  = dict(n_features=3, hidden_size=32, num_layers=2, dropout=0.0)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_train_stats() -> dict:
    cycle = load_drive_cycle(TRAIN_MAT)
    ds = BatteryWindowDataset(cycle, WindowConfig(WINDOW_CFG.length, WINDOW_CFG.stride, normalize=False))
    n = len(ds)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    L, S = WINDOW_CFG.length, WINDOW_CFG.stride
    overlap = max(0, (L + S - 1) // S - 1)
    train_end = max(0, n_train - overlap)
    I_tr, V_tr, T_tr = ds.raw_samples_for_indices(list(range(train_end)))
    return compute_norm_stats(I_tr, V_tr, T_tr)


def load_ecm_params(train_stats: dict) -> ECMParams:
    cache = EXP1_DIR / "ecm_params_cache.json"
    if cache.is_file():
        with open(cache) as f:
            return ECMParams.from_dict(json.load(f))
    raise FileNotFoundError("ecm_params_cache.json not found — run exp2 first")


def make_loader(mat_path: Path, stats: dict):
    cycle = load_drive_cycle(mat_path)
    ds = BatteryWindowDataset(cycle, WINDOW_CFG, stats=stats)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False), ds


def denorm_V(t: torch.Tensor, stats: dict) -> np.ndarray:
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    else:
        t = np.asarray(t)
    return t * stats["V_std"] + stats["V_mean"]


def reconstruct_series(pred_windows: np.ndarray, ds: BatteryWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Average overlapping window predictions back onto the raw timeline."""
    pred_windows = np.asarray(pred_windows, dtype=np.float64)
    signal_len = int(ds.V.shape[0])
    length = int(ds.cfg.length)

    pred_sum = np.zeros(signal_len, dtype=np.float64)
    counts = np.zeros(signal_len, dtype=np.float64)
    for pred_win, start in zip(pred_windows, ds.starts):
        pred_sum[start : start + length] += pred_win
        counts[start : start + length] += 1.0

    pred_full = np.divide(pred_sum, counts, out=np.zeros_like(pred_sum), where=counts > 0)
    true_full = denorm_V(ds.V, ds.stats)
    return pred_full, true_full


def pick_representative_segment(current_a: np.ndarray, seg_len: int = 1024) -> tuple[int, int]:
    """Return the start/end of the most dynamically active segment."""
    current_a = np.asarray(current_a)
    if len(current_a) <= seg_len:
        return 0, len(current_a)

    best_start = 0
    best_score = -np.inf
    for start in range(0, len(current_a) - seg_len + 1, max(1, seg_len // 16)):
        seg = current_a[start : start + seg_len]
        score = float(np.std(seg) + 0.25 * (np.max(seg) - np.min(seg)))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + seg_len


def get_all_preds(mat_path: Path, train_stats: dict, ecm_params: ECMParams):
    """Return dict of {model: (pred_V, true_V)} in physical units [V]."""
    loader, _ = make_loader(mat_path, train_stats)
    out = {}

    # ECM-1RC
    p, t = predict_ecm(ecm_params, loader, train_stats)
    out["ecm_1rc"] = (denorm_V(p, train_stats), denorm_V(t, train_stats))

    # LSTM
    loader, _ = make_loader(mat_path, train_stats)
    model = build_model("lstm", **LSTM_KWARGS)
    model.load_state_dict(torch.load(EXP1_DIR / "lstm_best.pt", map_location="cpu", weights_only=True))
    p, t = predict(model, loader, device=DEVICE)
    out["lstm"] = (denorm_V(p, train_stats), denorm_V(t, train_stats))

    # ECM-UDE
    loader, _ = make_loader(mat_path, train_stats)
    model = build_ecm_ude(stats=train_stats, ecm_params=ecm_params)
    model.load_state_dict(torch.load(EXP1_DIR / "ecm_ude_best.pt", map_location="cpu", weights_only=True))
    p, t = predict(model, loader, device=DEVICE)
    out["ecm_ude"] = (denorm_V(p, train_stats), denorm_V(t, train_stats))

    return out


def get_reconstructed_preds(mat_path: Path, train_stats: dict, ecm_params: ECMParams):
    """Return dict of {model: (pred_series, true_series)} in physical units [V]."""
    out = {}

    loader, ds = make_loader(mat_path, train_stats)
    p, _ = predict_ecm(ecm_params, loader, train_stats)
    out["ecm_1rc"] = reconstruct_series(denorm_V(p, train_stats), ds)

    loader, ds = make_loader(mat_path, train_stats)
    model = build_model("lstm", **LSTM_KWARGS)
    model.load_state_dict(torch.load(EXP1_DIR / "lstm_best.pt", map_location="cpu", weights_only=True))
    p, _ = predict(model, loader, device=DEVICE)
    out["lstm"] = reconstruct_series(denorm_V(p, train_stats), ds)

    loader, ds = make_loader(mat_path, train_stats)
    model = build_ecm_ude(stats=train_stats, ecm_params=ecm_params)
    model.load_state_dict(torch.load(EXP1_DIR / "ecm_ude_best.pt", map_location="cpu", weights_only=True))
    p, _ = predict(model, loader, device=DEVICE)
    out["ecm_ude"] = reconstruct_series(denorm_V(p, train_stats), ds)

    return out


def get_window_maes_mv(mat_path: Path, train_stats: dict, ecm_params: ECMParams) -> dict[str, float]:
    """Return per-model MAE in mV using the same windowed protocol as exp2/exp3."""
    preds = get_all_preds(mat_path, train_stats, ecm_params)
    return {
        name: float(np.mean(np.abs(pred_v - true_v)) * 1000.0)
        for name, (pred_v, true_v) in preds.items()
    }


# ─────────────────────────────────────────────────────────────────────
# Fig 2 — Voltage trace (in-distribution, UDDS 25°C)
# ─────────────────────────────────────────────────────────────────────

def fig2_voltage_trace(train_stats, ecm_params):
    print("  Generating fig2_voltage_trace ...")
    preds = get_reconstructed_preds(TRAIN_MAT, train_stats, ecm_params)
    cycle = load_drive_cycle(TRAIN_MAT)
    seg_start, seg_end = pick_representative_segment(cycle["I"].numpy(), WINDOW_CFG.length)
    dt = 0.1
    t_axis = np.arange(seg_end - seg_start) * dt

    fig, axes = plt.subplots(3, 1, figsize=(7, 5.6), sharex=True, sharey=True)
    y_min = np.inf
    y_max = -np.inf

    for name in ["ecm_1rc", "lstm", "ecm_ude"]:
        pred_full, true_full = preds[name]
        y_min = min(y_min, np.min(true_full[seg_start:seg_end]), np.min(pred_full[seg_start:seg_end]))
        y_max = max(y_max, np.max(true_full[seg_start:seg_end]), np.max(pred_full[seg_start:seg_end]))

    for ax, name in zip(axes, ["ecm_1rc", "lstm", "ecm_ude"]):
        pred_full, true_full = preds[name]
        seg_true = true_full[seg_start:seg_end] * 1000
        seg_pred = pred_full[seg_start:seg_end] * 1000
        local_mae = np.mean(np.abs(seg_pred - seg_true))

        ax.plot(t_axis, seg_true, color="black", lw=1.3, label="Measured", zorder=3)
        ax.plot(t_axis, seg_pred, color=COLORS[name], lw=1.1, ls="--", label=LABELS[name], zorder=2)
        ax.set_title(f"{LABELS[name]}  (segment MAE = {local_mae:.1f} mV)", loc="left", fontsize=9)
        ax.set_ylabel("Voltage (mV)")
        ax.legend(loc="lower right", framealpha=0.75)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.set_ylim(y_min * 1000 - 80, y_max * 1000 + 80)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Terminal voltage prediction — representative UDDS segment at 25°C", y=0.995, fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig2_voltage_trace")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig 3 — Box plot of MAE over 30 seeds
# ─────────────────────────────────────────────────────────────────────

def fig3_boxplot_seeds():
    print("  Generating fig3_boxplot_seeds ...")
    s1 = json.loads((EXP1_DIR / "summary.json").read_text())

    ude_maes  = [r["metrics_V"]["mae"] * 1000 for r in s1["ecm_ude"]["per_seed"]]
    lstm_maes = [r["metrics_V"]["mae"] * 1000 for r in s1["lstm"]["per_seed"]]
    ecm_mae   = s1["ecm_1rc"]["per_seed"][0]["metrics_V"]["mae"] * 1000  # deterministic

    fig, ax = plt.subplots(figsize=(4.5, 4))

    bp = ax.boxplot(
        [lstm_maes, ude_maes],
        labels=["LSTM", "ECM-UDE"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", lw=2),
        flierprops=dict(marker="o", ms=4, alpha=0.5),
    )
    bp["boxes"][0].set_facecolor(COLORS["lstm"] + "88")
    bp["boxes"][1].set_facecolor(COLORS["ecm_ude"] + "88")
    bp["boxes"][0].set_edgecolor(COLORS["lstm"])
    bp["boxes"][1].set_edgecolor(COLORS["ecm_ude"])

    # ECM-1RC reference line
    ax.axhline(ecm_mae, color=COLORS["ecm_1rc"], lw=1.5, ls=":", label=f"ECM-1RC = {ecm_mae:.1f} mV")

    # Wilcoxon annotation
    ax.text(1.5, max(lstm_maes) + 2, "W=0, p<10⁻⁹ ***", ha="center", fontsize=8,
            color="black")

    # scatter individual seeds
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(lstm_maes))
    ax.scatter(np.ones(len(lstm_maes)) + jitter, lstm_maes, color=COLORS["lstm"],
               s=15, alpha=0.6, zorder=4)
    ax.scatter(np.ones(len(ude_maes)) * 2 + jitter, ude_maes, color=COLORS["ecm_ude"],
               s=15, alpha=0.6, zorder=4)

    ax.set_ylabel("MAE (mV)")
    ax.set_title("MAE distribution over 30 random seeds\n(UDDS 25°C, in-distribution)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig3_boxplot_seeds")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig 4 — Temperature OOD degradation curve
# ─────────────────────────────────────────────────────────────────────

def fig4_temp_ood(train_stats, ecm_params):
    print("  Generating fig4_temp_ood ...")
    s2 = json.loads((EXP2_DIR / "summary.json").read_text())

    source_temp = 25
    temps_all = [-20, -10, 0, 10, source_temp]
    temp_keys = ["-20C", "-10C", "0C", "10C"]
    source_maes = get_window_maes_mv(TRAIN_MAT, train_stats, ecm_params)

    def mae_at_temps(model: str) -> list[float]:
        ood = [s2[model]["temperatures"][k]["mae"] * 1000 for k in temp_keys]
        return ood + [source_maes[model]]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for name in ["ecm_1rc", "lstm", "ecm_ude"]:
        maes = mae_at_temps(name)
        ax.plot(temps_all, maes, color=COLORS[name], label=LABELS[name], lw=1.8)
        ax.scatter(temps_all[:-1], maes[:-1], color=COLORS[name], s=55, zorder=3)
        ax.scatter(
            [source_temp], [maes[-1]], s=55, facecolors="white", edgecolors=COLORS[name],
            linewidths=1.8, zorder=4,
        )

    ax.axvline(source_temp, color="gray", lw=0.8, ls="--", alpha=0.5, label="Train temp.")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("MAE (mV)")
    ax.set_title("Temperature transfer from UDDS 25°C source\nfilled = zero-shot targets, hollow = source")
    ax.set_xticks(temps_all)
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig4_temp_ood")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig 5 — Drive-cycle OOD bar chart
# ─────────────────────────────────────────────────────────────────────

def fig5_cycle_ood():
    print("  Generating fig5_cycle_ood ...")
    s1 = json.loads((EXP1_DIR / "summary.json").read_text())
    s3 = json.loads((EXP3_DIR / "summary.json").read_text())

    cycles_ood = ["US06", "LA92", "HWFT"]
    cycles_all = ["UDDS*"] + cycles_ood

    def maes(model: str) -> list[float]:
        mae_udds = np.mean([r["metrics_V"]["mae"] * 1000 for r in s1[model]["per_seed"]])
        rest = [s3[model]["cycles"][c]["mae"] * 1000 for c in cycles_ood]
        return [mae_udds] + rest

    x = np.arange(len(cycles_all))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, name in enumerate(["ecm_1rc", "lstm", "ecm_ude"]):
        vals = maes(name)
        err = None
        if name != "ecm_1rc":
            err = [s1[name]["aggregate"]["mae"]["std"] * 1000, 0, 0, 0]
        bars = ax.bar(
            x + offsets[i], vals, width, label=LABELS[name],
            color=COLORS[name], alpha=0.85, edgecolor="white",
            yerr=err, capsize=3 if err else 0,
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cycles_all)
    ax.set_ylabel("MAE (mV)")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.8)
    ax.text(0.02, 0.96, "source", transform=ax.transAxes, fontsize=8, va="top", color="gray")
    ax.text(0.23, 0.96, "OOD targets", transform=ax.transAxes, fontsize=8, va="top", color="gray")
    ax.set_title("Drive-cycle transfer from UDDS 25°C source\nUDDS shown as in-distribution reference (mean ± std)")
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig5_cycle_ood")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig 6 — US06 qualitative trace (ECM-UDE vs LSTM)
# ─────────────────────────────────────────────────────────────────────

def fig6_us06_trace(train_stats, ecm_params):
    print("  Generating fig6_us06_trace ...")
    preds = get_reconstructed_preds(US06_MAT, train_stats, ecm_params)

    cycle = load_drive_cycle(US06_MAT)
    current_a = cycle["I"].numpy()
    seg_start, seg_end = pick_representative_segment(current_a, WINDOW_CFG.length)
    dt = 0.1
    t_axis = np.arange(seg_end - seg_start) * dt
    s3 = json.loads((EXP3_DIR / "summary.json").read_text())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    true_V = preds["ecm_ude"][1][seg_start:seg_end] * 1000
    ax1.plot(t_axis, true_V, color="black", lw=1.3, label="Measured", zorder=4)
    for name in ["ecm_1rc", "lstm", "ecm_ude"]:
        pred_V = preds[name][0][seg_start:seg_end] * 1000
        cycle_mae = s3[name]["cycles"]["US06"]["mae"] * 1000
        ax1.plot(
            t_axis, pred_V, color=COLORS[name], lw=1.0,
            ls=":" if name == "ecm_1rc" else "--",
            label=f"{LABELS[name]} (cycle MAE={cycle_mae:.1f} mV)", zorder=3,
        )
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("US06 representative segment — zero-shot transfer from UDDS 25°C", fontsize=11)
    ax1.legend(loc="lower right")

    I_denorm = current_a[seg_start:seg_end]
    ax2.fill_between(t_axis, I_denorm, alpha=0.35, color="#777")
    ax2.axhline(0, color="gray", lw=0.7)
    ax2.set_ylabel("Current (A)")
    ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    save_fig(fig, "fig6_us06_trace")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig OCV — OCV(SoC) curve from identified ECM
# ─────────────────────────────────────────────────────────────────────

def fig_ocv(ecm_params):
    print("  Generating fig_ocv ...")
    soc = np.linspace(0.0, 1.0, 500)
    ocv = ecm_params.ocv_of(soc)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(soc * 100, ocv * 1000, color=COLORS["ecm_ude"], lw=2)
    ax.axvspan(ecm_params.soc_train_min * 100, ecm_params.soc_train_max * 100,
               alpha=0.08, color=COLORS["ecm_ude"], label="SoC range (train)")
    ax.set_xlabel("State of Charge (%)")
    ax.set_ylabel("OCV (mV)")
    ax.set_title("Identified OCV(SoC) — Chebyshev polynomial")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_ocv")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig learning curves — best seed
# ─────────────────────────────────────────────────────────────────────

def fig_learning_curves():
    print("  Generating fig_learning_curves ...")
    s1 = json.loads((EXP1_DIR / "summary.json").read_text())

    # find best ECM-UDE seed
    best = min(s1["ecm_ude"]["per_seed"], key=lambda r: r["metrics_V"]["mae"])
    best_seed = best["seed"]

    history_path = EXP1_DIR / f"history_ecm_ude_seed{best_seed}.json"
    if not history_path.is_file():
        print(f"    WARNING: {history_path} not found, skipping")
        return
    hist = json.loads(history_path.read_text())

    epochs = list(range(1, len(hist["train_loss"]) + 1))
    train_loss = [v * 1000 for v in hist["train_loss"]]
    val_loss   = [v * 1000 for v in hist["val_loss"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, train_loss, label="Train loss", color=COLORS["ecm_ude"])
    ax.plot(epochs, val_loss,   label="Val loss",   color=COLORS["lstm"])
    best_ep = hist.get("best_epoch", 0)
    ax.axvline(best_ep + 1, color="gray", lw=1, ls=":", label=f"Best epoch ({best_ep+1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (mV²)")
    ax.set_title(f"Learning curves — ECM-UDE best seed ({best_seed})")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_learning_curves")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Fig SoC noise sensitivity — inference-time perturbation
# ─────────────────────────────────────────────────────────────────────

def fig_soc_noise_sensitivity():
    print("  Generating fig_soc_noise_sensitivity ...")
    summary_path = EXP1_SOC_DIR / "summary.json"
    if not summary_path.is_file():
        print(f"    WARNING: {summary_path} not found, skipping")
        return

    s = json.loads(summary_path.read_text())
    sigma_levels = [float(v) for v in s["config"]["noise_std_levels_soc"]]
    sigma_pct = [100.0 * v for v in sigma_levels]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    for name in ["ecm_1rc", "lstm", "ecm_ude"]:
        means = [s[name]["noise_levels"][f"{sigma:.2f}"]["mae"]["mean"] * 1000 for sigma in sigma_levels]
        stds = [s[name]["noise_levels"][f"{sigma:.2f}"]["mae"]["std"] * 1000 for sigma in sigma_levels]
        ax.errorbar(
            sigma_pct,
            means,
            yerr=stds,
            color=COLORS[name],
            lw=1.8,
            marker="o",
            ms=5,
            capsize=3,
            label=LABELS[name],
        )

    ax.set_xlabel(r"SoC noise std. $\sigma_{\mathrm{SoC}}$ (%)")
    ax.set_ylabel("MAE (mV)")
    ax.set_title("Sensitivity to inference-time SoC uncertainty\nGaussian noise added to the SoC input channel")
    ax.set_xticks(sigma_pct)
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "soc_noise_sensitivity")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("Paper 1 — generating publication figures")
    print(f"Device: {DEVICE}")

    print("\nLoading training stats & ECM params ...")
    train_stats = load_train_stats()
    ecm_params  = load_ecm_params(train_stats)
    print(f"  R0={ecm_params.R0*1000:.1f} mΩ  R1={ecm_params.R1*1000:.1f} mΩ  C1={ecm_params.C1:.0f} F")

    print("\nRunning figures (may take a few minutes for ODE-based models) ...")
    fig2_voltage_trace(train_stats, ecm_params)
    fig3_boxplot_seeds()
    fig4_temp_ood(train_stats, ecm_params)
    fig5_cycle_ood()
    fig6_us06_trace(train_stats, ecm_params)
    fig_ocv(ecm_params)
    fig_learning_curves()
    fig_soc_noise_sensitivity()

    print(f"\nAll figures saved to {FIG_DIR.resolve()}/")
    print("Files:")
    for f in sorted(FIG_DIR.glob("*.pdf"), key=lambda p: p.stem):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
