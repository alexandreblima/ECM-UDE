"""Summarize Part B LG HG2 models, ablations, and provenance."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp_unified as U
import protocol as P
from exp_paper_v2_seeds import shifted_wilcoxon
from reviewer_round2_b import CONDITIONS, NEURAL_VARIANTS, OUT, SEEDS

METRICS = ("mae", "rmse", "max_error", "p99")
PAIRS = (
    ("physics_gradient", "ecm_2rc_nls"),
    ("output_residual", "physics_gradient"),
    ("ecm_ude", "physics_gradient"),
    ("output_residual", "ecm_2rc_nls"),
    ("ecm_ude", "ecm_2rc_nls"),
)


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


def _base_inference(base_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    values = defaultdict(list)
    for row in base_rows:
        if row["evaluation"] not in ("continuous", "windowed"):
            continue
        metric_values = {
            metric: float(row[f"{metric}_mV"])
            for metric in METRICS
            if row.get(f"{metric}_mV") not in (None, "")
        }
        values[
            (row["model"], str(row["seed"]), row["condition"], row["evaluation"])
        ] = metric_values

    tests = []
    for model, reference in PAIRS:
        for mode in ("continuous", "windowed"):
            for metric in METRICS:
                for condition in CONDITIONS:
                    model_values = np.asarray(
                        [
                            values[(model, str(seed), condition, mode)][metric]
                            for seed in SEEDS
                        ],
                        dtype=float,
                    )
                    if reference.endswith("_nls"):
                        reference_value = values[
                            (reference, "deterministic", condition, mode)
                        ][metric]
                        reference_values = np.full(len(SEEDS), reference_value)
                    else:
                        reference_values = np.asarray(
                            [
                                values[(reference, str(seed), condition, mode)][metric]
                                for seed in SEEDS
                            ],
                            dtype=float,
                        )
                    paired = U.paired(model_values, reference_values)
                    tests.append(
                        {
                            "model": model,
                            "reference": reference,
                            "condition": condition,
                            "evaluation": mode,
                            "metric": metric,
                            "n": len(SEEDS),
                            "median_difference_mV": float(
                                np.median(model_values - reference_values)
                            ),
                            "p_raw": paired["p_raw"],
                        }
                    )
    for model, reference in PAIRS:
        for mode in ("continuous", "windowed"):
            for metric in METRICS:
                family = [
                    {"p_raw": row["p_raw"], "_row": row}
                    for row in tests
                    if row["model"] == model
                    and row["reference"] == reference
                    and row["evaluation"] == mode
                    and row["metric"] == metric
                ]
                if len(family) != len(CONDITIONS):
                    raise ValueError(
                        f"Incomplete B Holm family: {model}/{reference}/{mode}/{metric}"
                    )
                U.holm(family)
                for item in family:
                    item["_row"]["p_holm"] = item["p_holm"]

    tost = []
    for model in NEURAL_VARIANTS:
        for reference in ("physics_gradient", "ecm_2rc_nls"):
            for condition in CONDITIONS:
                model_values = np.asarray(
                    [
                        values[(model, str(seed), condition, "continuous")]["mae"]
                        for seed in SEEDS
                    ]
                )
                if reference.endswith("_nls"):
                    baseline = values[
                        (reference, "deterministic", condition, "continuous")
                    ]["mae"]
                    reference_values = np.full(len(SEEDS), baseline)
                else:
                    reference_values = np.asarray(
                        [
                            values[
                                (reference, str(seed), condition, "continuous")
                            ]["mae"]
                            for seed in SEEDS
                        ]
                    )
                difference = model_values - reference_values
                lower = shifted_wilcoxon(difference + 1.0, "greater")
                upper = shifted_wilcoxon(difference - 1.0, "less")
                tost.append(
                    {
                        "model": model,
                        "reference": reference,
                        "condition": condition,
                        "n": len(SEEDS),
                        "median_difference_mV": float(np.median(difference)),
                        "p_lower_raw": lower["p_raw"],
                        "p_upper_raw": upper["p_raw"],
                    }
                )
    for model in NEURAL_VARIANTS:
        for reference in ("physics_gradient", "ecm_2rc_nls"):
            rows = [
                row for row in tost
                if row["model"] == model and row["reference"] == reference
            ]
            for tail in ("lower", "upper"):
                family = [
                    {"p_raw": row[f"p_{tail}_raw"], "_row": row}
                    for row in rows
                ]
                if len(family) != len(CONDITIONS):
                    raise ValueError(
                        f"Incomplete B TOST family: {model}/{reference}/{tail}"
                    )
                U.holm(family)
                for item in family:
                    item["_row"][f"p_{tail}_holm"] = item["p_holm"]
    for row in tost:
        row["p_tost_raw"] = max(row["p_lower_raw"], row["p_upper_raw"])
        row["p_tost_holm"] = max(row["p_lower_holm"], row["p_upper_holm"])
    return tests, tost


def _plot_histories() -> None:
    histories = json.loads((OUT / "base" / "training_histories.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3), sharey=False)
    colors = {
        "physics_gradient": "0.12",
        "output_residual": "0.38",
        "ecm_ude": "0.62",
        "lstm": "0.82",
    }
    linestyles = {
        "physics_gradient": "-",
        "output_residual": "--",
        "ecm_ude": "-.",
        "lstm": ":",
    }
    for ax, metric in zip(axes, ("train_loss", "val_loss")):
        grouped = defaultdict(list)
        for item in histories:
            grouped[item["model"]].append(item)
        for model, runs in grouped.items():
            epochs = sorted(
                {
                    epoch
                    for run in runs
                    for epoch in range(1, len(run["history"][metric]) + 1)
                }
            )
            medians, minima, maxima = [], [], []
            for epoch in epochs:
                samples = [
                    float(run["history"][metric][epoch - 1])
                    for run in runs
                    if len(run["history"][metric]) >= epoch
                ]
                medians.append(statistics.median(samples))
                minima.append(min(samples))
                maxima.append(max(samples))
            ax.plot(
                epochs,
                medians,
                color=colors[model],
                linestyle=linestyles[model],
                linewidth=1.1,
                label=model.replace("_", " "),
            )
            ax.fill_between(epochs, minima, maxima, color=colors[model], alpha=0.10)
            selected = statistics.median(run["best_epoch"] for run in runs)
            ax.axvline(selected, color=colors[model], alpha=0.25, linewidth=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE")
        ax.set_title("Training" if metric == "train_loss" else "Validation")
        ax.grid(True, color="0.88", linewidth=0.45)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    fig.savefig(OUT / "learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_a1_curves() -> None:
    rows = _read_csv(OUT / "a1" / "learning_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3), sharey=True)
    colors = {"output_residual": "0.18", "ecm_ude": "0.55"}
    for ax, field in zip(axes, ("train_loss", "validation_loss")):
        for variant in NEURAL_VARIANTS:
            selected = [row for row in rows if row["variant"] == variant]
            epochs = sorted({int(row["epoch"]) for row in selected})
            medians, minima, maxima = [], [], []
            for epoch in epochs:
                values = [
                    float(row[field])
                    for row in selected
                    if int(row["epoch"]) == epoch
                ]
                medians.append(statistics.median(values))
                minima.append(min(values))
                maxima.append(max(values))
            ax.plot(
                epochs,
                medians,
                color=colors[variant],
                linewidth=1.1,
                label=variant.replace("_", " "),
            )
            ax.fill_between(
                epochs, minima, maxima, color=colors[variant], alpha=0.12
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE")
        ax.set_title("Training" if field == "train_loss" else "Validation")
        ax.grid(True, color="0.88", linewidth=0.45)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    fig.savefig(OUT / "a1_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_report(base_rows: list[dict], tests: list[dict], tost: list[dict]) -> None:
    continuous = [
        row for row in base_rows
        if row["evaluation"] == "continuous"
    ]
    windowed = [
        row for row in base_rows
        if row["evaluation"] == "windowed"
    ]
    lines = [
        "# Part B — LG HG2 replication",
        "",
        "Input, signs, time ranges, initial states, "
        "capacity, and OCV are audited in `data_audit.md`; hashes "
        "of the files used are in `manifest.json`. The defect in HWFET "
        "25 °C was recorded (6002 samples, 600,007 s, maximum absolute current "
        "0 A), and the condition was replaced by Mixed1 25 °C. "
        "The eight targets are Mixed1, LA92, and US06 at 25 °C and UDDS at −20, −10, "
        "0, 10, and 40 °C; the source UDDS 25 °C is evaluated only on validation.",
        "",
        "The C/20 discharge branch of `549_C20DisCh` at 25 °C initialized the "
        "OCV curve; capacity `Cap_1C` = 2,72639 Ah. Positive current "
        "means charging. MAT inputs were sanitized and resampled "
        "at 0,1 s according to the protocol.",
        "",
        "## Continuous results",
        "",
        "| Condition | Model | Mean MAE ± SD (mV) | Mean RMSE ± SD (mV) | Mean maximum ± SD (mV) |",
        "|---|---|---:|---:|---:|",
    ]
    model_order = (
        "ecm_2rc_nls",
        "physics_gradient",
        "output_residual",
        "ecm_ude",
        "lstm",
    )
    for condition in CONDITIONS:
        for model in model_order:
            rows = [
                row for row in continuous
                if row["condition"] == condition and row["model"] == model
            ]
            if not rows:
                continue
            mae = [float(row["mae_mV"]) for row in rows]
            rmse = [float(row["rmse_mV"]) for row in rows]
            maximum = [float(row["max_error_mV"]) for row in rows]
            summarize = lambda x: (
                f"{statistics.mean(x):.3f} ± "
                f"{statistics.stdev(x):.3f}" if len(x) > 1
                else f"{x[0]:.3f} (deterministic/contextual)"
            )
            lines.append(
                f"| {condition} | {model} | {summarize(mae)} | "
                f"{summarize(rmse)} | {summarize(maximum)} |"
            )
    lines.extend(
        [
            "",
            "### Contextual LSTM (windowed)",
            "",
            "| Condition | MAE (mV) | RMSE (mV) | Maximum error (mV) |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        rows = [
            row for row in windowed
            if row["condition"] == condition and row["model"] == "lstm"
        ]
        if rows:
            row = rows[0]
            lines.append(
                f"| {condition} | {float(row['mae_mV']):.3f} | "
                f"{float(row['rmse_mV']):.3f} | "
                f"{float(row['max_error_mV']):.3f} |"
            )
    a1_nls = _read_csv(OUT / "a1_nls_tost.csv")
    a1_joint = _read_csv(OUT / "a1_joint_paired.csv")
    a2_network = _read_csv(OUT / "a2_network_effect_tost.csv")
    a2_pg = _read_csv(OUT / "a2_zeroed_vs_physics_gradient_tost.csv")
    lines.extend(
        [
            "",
            "Exact paired comparisons and raw/Holm-adjusted p-values "
            "are in `pairwise_wilcoxon.csv`; TOST ±1 mV equivalence comparisons "
            "(signed differences, tails, and raw/adjusted maxima) "
            "are in `tost_vs_reference.csv`. The families "
            "cover the eight target conditions plus source validation. "
            "LSTM is contextual and has no comparable continuous simulation; "
            "therefore it appears only in windowed metrics.",
            "",
            "## A1 and A2 ablations",
            "",
            "A1 froze the first-order ECM physics; before and after hashes "
            "and zero gradients are in `a1/freeze_audit.json`. "
            "TOST against NLS and the paired test against joint training are "
            "in `a1_nls_tost.csv` and `a1_joint_paired.csv`.",
            "",
            "| Variant | A1 equivalent to NLS | Significant A1 vs. joint difference |",
            "|---|---:|---:|",
        ]
    )
    for variant in NEURAL_VARIANTS:
        nls_rows = [row for row in a1_nls if row["variant"] == variant]
        joint_rows = [row for row in a1_joint if row["variant"] == variant]
        lines.append(
            f"| {variant} | "
            f"{sum(float(row['p_tost_holm']) < 0.05 for row in nls_rows)}/9 | "
            f"{sum(float(row['p_holm']) < 0.05 for row in joint_rows)}/9 |"
        )
    lines.extend(
        [
            "",
            "A2 exactly reproduced the jointly trained checkpoint metrics "
            "before zeroing the network; checkpoint hashes are in "
            "`a2/manifest.json`. MAE/RMSE/maximum error differences and TOST-Holm "
            "network vs. zeroed and zeroed vs. physics-gradient are in "
            "`a2_network_effect_tost.csv` and "
            "`a2_zeroed_vs_physics_gradient_tost.csv`.",
            "",
            "| Variant | Metric | Network vs. zeroed equivalent | Zeroed vs. physics-gradient equivalent |",
            "|---|---|---:|---:|",
        ]
    )
    for variant in NEURAL_VARIANTS:
        for metric in ("mae", "rmse", "max_error"):
            network_rows = [
                row for row in a2_network
                if row["variant"] == variant and row["metric"] == metric
            ]
            pg_rows = [
                row for row in a2_pg
                if row["variant"] == variant and row["metric"] == metric
            ]
            lines.append(
                f"| {variant} | {metric} | "
                f"{sum(float(row['p_tost_holm']) < 0.05 for row in network_rows)}/9 | "
                f"{sum(float(row['p_tost_holm']) < 0.05 for row in pg_rows)}/9 |"
            )
    lines.extend(
        [
            "",
            "## Figures and files",
            "",
            "- `learning_curves.pdf`: median across seeds and minimum–maximum range; "
            "vertical lines indicate the median selected epoch.",
            "- `a1_learning_curves.pdf`: frozen-ablation history.",
            "- `base/metrics_by_seed.csv`, `base/training_histories.json`, "
            "`base/nls_fits.json`, `base/manifest.json`.",
            "- `a1/metrics_by_seed.csv`, `a1/freeze_audit.json`, "
            "`a1/manifest.json`; `a2/zeroed_metrics.csv`, `a2/manifest.json`.",
            "- Generate figures, tests, and report: "
            "`python reviewer_round2_b_report.py`.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


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
    base_rows = _read_csv(OUT / "base" / "metrics_by_seed.csv")
    tests, tost = _base_inference(base_rows)
    _write_csv(OUT / "pairwise_wilcoxon.csv", tests)
    _write_csv(OUT / "tost_vs_reference.csv", tost)
    _plot_histories()
    _plot_a1_curves()
    _write_report(base_rows, tests, tost)
    outputs = (
        "report.md",
        "learning_curves.pdf",
        "a1_learning_curves.pdf",
        "pairwise_wilcoxon.csv",
        "tost_vs_reference.csv",
    )
    (OUT / "report_manifest.json").write_text(
        json.dumps({name: P.sha256(OUT / name) for name in outputs}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    run()
