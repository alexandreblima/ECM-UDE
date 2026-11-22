"""Generate Part A reviewer-round-two figures, summaries, and report."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import protocol as P

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "A"
VARIANTS = ("output_residual", "ecm_ude")
DATASETS = ("panasonic", "calce")
DISPLAY = {"panasonic": "NCA", "calce": "NMC"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No records to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _a4_tables() -> tuple[list[dict], list[dict]]:
    loss_summary = _read_csv(OUT / "a4_training_loss_summary.csv")
    expected = {(experiment.upper(), dataset, variant)
                for experiment in ("a4a", "a4b")
                for dataset in DATASETS for variant in VARIANTS}
    actual = [(row["experiment"], row["dataset"], row["variant"]) for row in loss_summary]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate corrected A4 loss summary")
    for row in loss_summary:
        if int(row["n"]) != 5 or row["loss_basis"] != "selected_best_validation_checkpoint; fixed train-window evaluation":
            raise ValueError("A4 loss summary is not the reconciled checkpoint evaluation")
    tost_summary = []
    for experiment in ("a4a", "a4b"):
        root = OUT / experiment
        tests = _read_csv(root / "tost_vs_nls.csv")
        for variant in VARIANTS:
            rows = [row for row in tests if row["variant"] == variant]
            if len(rows) != 18:
                raise ValueError(
                    f"Incomplete A4 TOST family: {experiment}/{variant}"
                )
            deltas = [float(row["median_mae_difference_mV"]) for row in rows]
            tost_summary.append(
                {
                    "experiment": experiment.upper(),
                    "variant": variant,
                    "conditions": len(rows),
                    "holm_equivalent_conditions": sum(
                        float(row["p_tost_holm"]) < 0.05 for row in rows
                    ),
                    "median_condition_difference_mV": statistics.median(deltas),
                    "conditions_improved_by_more_than_1mV": sum(
                        delta < -1.0 for delta in deltas
                    ),
                    "conditions_worsened_by_more_than_1mV": sum(
                        delta > 1.0 for delta in deltas
                    ),
                }
            )
    return loss_summary, tost_summary


def _plot_a4() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.3), sharey=False)
    colors = {"A4a": "0.12", "A4b": "0.52"}
    for row_index, dataset in enumerate(DATASETS):
        for col_index, variant in enumerate(VARIANTS):
            ax = axes[row_index, col_index]
            for experiment, linestyle in (("A4a", "-"), ("A4b", "--")):
                data = _read_csv(
                    OUT / experiment.lower() / "learning_curves.csv"
                )
                selected = [
                    item
                    for item in data
                    if item["dataset"] == dataset and item["variant"] == variant
                ]
                epochs = sorted({int(item["epoch"]) for item in selected})
                train_medians, train_mins, train_maxs = [], [], []
                val_medians, val_mins, val_maxs = [], [], []
                for epoch in epochs:
                    at_epoch = [
                        item for item in selected if int(item["epoch"]) == epoch
                    ]
                    train = [float(item["train_loss"]) for item in at_epoch]
                    val = [float(item["validation_loss"]) for item in at_epoch]
                    train_medians.append(statistics.median(train))
                    train_mins.append(min(train))
                    train_maxs.append(max(train))
                    val_medians.append(statistics.median(val))
                    val_mins.append(min(val))
                    val_maxs.append(max(val))
                color = colors[experiment]
                ax.plot(
                    epochs,
                    train_medians,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.1,
                    label=f"{experiment} train",
                )
                ax.fill_between(
                    epochs, train_mins, train_maxs, color=color, alpha=0.10
                )
                ax.plot(
                    epochs,
                    val_medians,
                    color=color,
                    linestyle=":" if experiment == "A4a" else "-.",
                    linewidth=1.0,
                    label=f"{experiment} validation",
                )
                ax.fill_between(epochs, val_mins, val_maxs, color=color, alpha=0.06)
            ax.set_title(f"{DISPLAY[dataset]} — {variant.replace('_', ' ')}")
            ax.set_xlabel("Epoch")
            if col_index == 0:
                ax.set_ylabel("Normalized MSE")
            ax.grid(True, color="0.88", linewidth=0.45)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(OUT / "a4_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_a3() -> None:
    data = _read_csv(OUT / "a3" / "training_validation_mae.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), sharex=True)
    for row_index, dataset in enumerate(DATASETS):
        for col_index, variant in enumerate(VARIANTS):
            ax = axes[row_index, col_index]
            selected = sorted(
                (
                    item for item in data
                    if item["dataset"] == dataset and item["variant"] == variant
                ),
                key=lambda item: int(item["epoch"]),
            )
            epochs = [int(item["epoch"]) for item in selected]
            ax.plot(
                epochs,
                [float(item["train_mae_mV"]) for item in selected],
                color="0.15",
                linewidth=1.1,
                label="Train",
            )
            ax.plot(
                epochs,
                [float(item["validation_mae_mV"]) for item in selected],
                color="0.5",
                linestyle="--",
                linewidth=1.1,
                label="Validation",
            )
            ax.set_title(f"{DISPLAY[dataset]} — {variant.replace('_', ' ')}")
            ax.set_xlabel("Epoch")
            if col_index == 0:
                ax.set_ylabel("MAE (mV)")
            ax.grid(True, color="0.88", linewidth=0.45)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(OUT / "a3_train_validation_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_report(loss_summary: list[dict], tost_summary: list[dict], destination: Path | None = None) -> None:
    a1_nls = _read_csv(OUT / "a1_nls_tost.csv")
    a1_joint = _read_csv(OUT / "a1_joint_paired.csv")
    a3 = _read_csv(OUT / "a3" / "training_validation_mae.csv")
    lines = [
        "# Part A — Ablations and optimization sensitivity",
        "",
        "## A1 — Frozen physics",
        "",
        "The experiment fitted 40 models (two variants, two cells, ten seeds). "
        "The audit confirmed physical parameters without gradients, invariant values, "
        "and identical before/after hashes for all 40 checkpoints. "
        "Each variant was compared with the first-order NLS ECM using TOST "
        "at ±1 mV, with Holm applied separately by variant to the family of 18 conditions.",
        "",
    ]
    for variant in VARIANTS:
        rows = [row for row in a1_nls if row["variant"] == variant]
        joint_rows = [row for row in a1_joint if row["variant"] == variant]
        lines.append(
            f"- `{variant}` vs. NLS: equivalence in "
            f"{sum(float(row['mae_p_tost_holm']) < 0.05 for row in rows)}/{len(rows)} "
            "conditions after Holm; median of condition differences "
            f"{statistics.median(float(row['mae_median_difference_mV']) for row in rows):.4f} mV "
            "(A1 − NLS)."
        )
        lines.append(
            f"- `{variant}` A1 vs. joint training: "
            f"{sum(float(row['mae_p_holm']) < 0.05 for row in joint_rows)}/{len(joint_rows)} "
            "conditions with a significant paired difference after Holm; median "
            f"{statistics.median(float(row['mae_median_difference_mV']) for row in joint_rows):.4f} mV "
            "(A1 − joint)."
        )
    lines.extend(
        [
            "",
            "Raw and adjusted p-values by condition and seed are in "
            "`a1_nls_tost.csv` and `a1_joint_paired.csv`; metrics and hashes by "
            "checkpoint are in `a1/metrics_by_seed.csv` and `a1/manifest.json`.",
            "",
            "## A2 — Network zeroed after joint training",
            "",
            "The 40 original checkpoints were reevaluated without retraining, and "
            "their results were exactly reproduced before zeroing the "
            "correction. A2 confirmed equivalence of the jointly trained network against the "
            "zeroed network for output residual in 18/18 conditions for all three "
            "metrics; for ECM-UDE, in 18/18 for MAE/RMSE and 14/18 for "
            "maximum error. The zeroed network was equivalent to physics-gradient "
            "in the same totals. See `a2_report.md`, A2 tables, and manifests.",
            "",
            "## A3 — Diagnostic without early selection",
            "",
            "One seed per variant and cell was trained for 30 epochs, without "
            "early stopping. The table shows training/validation MAE at the "
            "first and last epoch; complete curves are in "
            "`a3_train_validation_mae.pdf` and `a3/training_validation_mae.csv`.",
            "",
            "| Cell | Variant | Initial → final training (mV) | Initial → final validation (mV) |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for variant in VARIANTS:
            rows = sorted(
                (
                    row for row in a3
                    if row["dataset"] == dataset and row["variant"] == variant
                ),
                key=lambda row: int(row["epoch"]),
            )
            lines.append(
                f"| {DISPLAY[dataset]} | {variant} | "
                f"{float(rows[0]['train_mae_mV']):.3f} → {float(rows[-1]['train_mae_mV']):.3f} | "
                f"{float(rows[0]['validation_mae_mV']):.3f} → "
                f"{float(rows[-1]['validation_mae_mV']):.3f} |"
            )
    a4b_output = next(
        row for row in tost_summary
        if row["experiment"] == "A4B" and row["variant"] == "output_residual"
    )
    if any(float(row["median_relative_loss_reduction"]) > 0 for row in loss_summary):
        raise ValueError("A4 checkpoint losses no longer match the reported interpretation")
    if any(row["holm_equivalent_conditions"] for row in tost_summary):
        raise ValueError("A4 TOST results no longer match the reported interpretation")
    lines.extend(
        [
            "",
            "## A4 — Starting point and budget",
            "",
            "A4a uses initial gain 1,0 and the other original settings. A4b "
            "adds a learning rate of 2×10⁻³, up to 150 epochs, and patience 30. "
            'The E1 reconciliation showed that the previous table compared NLS, evaluated once over all windows, with `history["train_loss"][-1]`: '
            "the mean mini-batch loss during updates in the last epoch. This is not the selected checkpoint's loss; "
            "when training ends, the checkpoint with the lowest validation loss is restored. The table below was corrected: "
            "each saved checkpoint was reevaluated on the same training windows as NLS, with the same normalization "
            "and RC state reset per window. Negative values indicate higher MSE for the neural checkpoint.",
            "",
            "| Run | Cell | Variant | Median training MSE reduction at checkpoint | Median online reduction in last epoch |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in loss_summary:
        lines.append(
            f"| {row['experiment']} | {DISPLAY[row['dataset']]} | {row['variant']} | "
            f"{100*float(row['median_relative_loss_reduction']):.2f}% | "
            f"{100*float(row['median_last_epoch_relative_loss_reduction']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "Figure `a4_learning_curves.pdf` shows medians across seeds and "
            "minimum–maximum ranges. No condition achieved TOST equivalence "
            "to NLS after Holm correction. Raw and adjusted p-values, along with signed "
            "differences by condition, are in `a4a/tost_vs_nls.csv` and "
            "`a4b/tost_vs_nls.csv`.",
            "",
            "| Run | Variant | Equivalent after Holm | Median condition difference (mV) | Conditions with improvement >1 mV |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in tost_summary:
        lines.append(
            f"| {row['experiment']} | {row['variant']} | "
            f"{row['holm_equivalent_conditions']}/{row['conditions']} | "
            f"{row['median_condition_difference_mV']:.3f} | "
            f"{row['conditions_improved_by_more_than_1mV']} |"
        )
    checkpoint_metrics = _read_csv(ROOT / "results/reviewer_round2/E/a4_checkpoint_metrics.csv")
    nls = [row for row in checkpoint_metrics if row["experiment"] == "NLS" and row["dataset"] == "panasonic"]
    a4b_ude = [row for row in checkpoint_metrics if row["experiment"] == "A4B"
               and row["dataset"] == "panasonic" and row["variant"] == "ecm_ude"]
    if len(nls) != 1 or len(a4b_ude) != 5:
        raise ValueError("Incomplete A4 checkpoint metrics for the reconciliation example")
    train_delta = statistics.median(float(row["continuous_train_mae_mV"]) for row in a4b_ude) - float(nls[0]["continuous_train_mae_mV"])
    validation_delta = statistics.median(float(row["continuous_validation_mae_mV"]) for row in a4b_ude) - float(nls[0]["continuous_validation_mae_mV"])
    if train_delta <= 0 or validation_delta >= 0:
        raise ValueError("A4 reconciliation example no longer matches the reported directions")
    lines.extend(
        [
            "",
            "For selected checkpoints, training MSE did not decrease relative to NLS in any cell-variant group. "
            "This does not erase the continuous improvement observed on validation/targets: for example, for ECM-UDE A4b/NCA "
            f"the median continuous MAE was {train_delta:.2f} mV higher than NLS on the source training portion and "
            f"{-validation_delta:.2f} mV lower on continuous validation; windowed MSE resets the RC state for each window, "
            "whereas continuous evaluation carries the state between samples. The last epoch's online mean underestimated "
            "the difference between the saved checkpoint and NLS, especially for A4b/NCA ECM-UDE.",
            "",
            f"Output residual improved MAE by more than 1 mV in {a4b_output['conditions_improved_by_more_than_1mV']}/18 conditions (A4b), "
            "but no contrast reached TOST-Holm equivalence with NLS. Therefore, the manuscript's conclusion should be restricted "
            "to the original procedure, without claiming that no broader initialization or budget can improve target MAE. "
            "Detailed evaluation of the four results per seed (normalized training/validation MSE and continuous training/validation MAE), "
            "epochs, and hashes is in `../E/a4_checkpoint_metrics.csv` and `../E/a4_loss_reconciliation.csv`.",
            "",
            "## Files and reproducibility",
            "",
            "- A1: `a1/metrics_by_seed.csv`, `a1/learning_curves.csv`, "
            "`a1/freeze_audit.json`, `a1/manifest.json`, "
            "`a1_nls_tost.csv`, `a1_joint_paired.csv`.",
            "- A2: `a2_report.md`, `a2_manifest.json`, "
            "`a2_network_effect_tost.csv`, "
            "`a2_zeroed_vs_physics_gradient_tost.csv`.",
            "- A3: `a3/training_validation_mae.csv`, "
            "`a3/checkpoint_hashes.json`, `a3_train_validation_mae.pdf`.",
            "- A4a/A4b: `a4a/` and `a4b/`, including metrics, curves, "
            "frozen-physics audit, TOST-Holm, and hashes.",
            "- Generate figures and report: `python reviewer_round2_a_report.py`.",
        ]
    )
    (destination if destination is not None else OUT / "report.md").write_text("\n".join(lines) + "\n")


def run() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    loss_summary, tost_summary = _a4_tables()
    _write_csv(OUT / "a4_tost_summary.csv", tost_summary)
    _plot_a4()
    _plot_a3()
    _write_report(loss_summary, tost_summary)
    manifest = {
        name: P.sha256(OUT / name)
        for name in (
            "report.md",
            "a4_learning_curves.pdf",
            "a3_train_validation_mae.pdf",
            "a4_training_loss_summary.csv",
            "a4_tost_summary.csv",
        )
    }
    (OUT / "report_artifacts_sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    run()
