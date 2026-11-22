"""Reproduce reviewer-round-two analyses from immutable saved artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

import exp_unified as U
from data_lfp import ROOT
from exp_paper_v2_seeds import shifted_wilcoxon


OUT = ROOT / "results" / "reviewer_round2" / "D"
SUMMARY_PATH = ROOT / "results" / "unified_10seeds" / "summary.json"
STATS_PATH = ROOT / "results" / "unified_10seeds" / "statistics.json"
E1_TABLE_PATH = ROOT / "results" / "paper_tables" / "E1.csv"
EPOCHS_PATH = ROOT / "results" / "training_audit" / "epochs.csv"
MODELS = ("physics_gradient", "output_residual", "ecm_ude")
CELLS = ("panasonic", "calce")
SEEDS = tuple(range(42, 52))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_histories(rows: list[dict]) -> dict[tuple[str, str, int], Path]:
    candidates = sorted(
        p
        for base in (ROOT / "results" / "unified" / "checkpoints",
                     ROOT / "results" / "unified_10seeds" / "checkpoints")
        for p in base.rglob("*.history.json")
    )
    by_hash: dict[str, list[Path]] = {}
    for path in candidates:
        by_hash.setdefault(sha256(path), []).append(path)
    resolved = {}
    for row in rows:
        if row["dataset"] not in CELLS or row["model"] not in MODELS:
            continue
        matches = by_hash.get(row["history_sha256"], [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected one history with SHA256 {row['history_sha256']}, found {len(matches)}"
            )
        resolved[(row["dataset"], row["model"], int(row["seed"]))] = matches[0]
    expected = len(CELLS) * len(MODELS) * len(SEEDS)
    if len(resolved) != expected:
        raise ValueError(f"Expected {expected} training histories, resolved {len(resolved)}")
    return resolved


def _holm(records: list[dict]) -> None:
    U.holm(records)


def _walsh_pseudomedian(values: np.ndarray) -> float:
    return float(np.median([(values[i] + values[j]) / 2
                            for i in range(len(values)) for j in range(i, len(values))]))


def _tost(values: np.ndarray) -> dict:
    lower = shifted_wilcoxon(values + 1.0, "greater")
    upper = shifted_wilcoxon(values - 1.0, "less")
    paired = U.paired(values, np.zeros_like(values))
    return {"lower": lower, "upper": upper, "paired_two_sided": paired}


def d1(summary: dict) -> tuple[list[dict], dict]:
    rows = []
    families: dict[tuple[str, str, str], list[dict]] = {}
    for cell, data in summary["datasets"].items():
        for condition in data["conditions"]:
            baseline_2rc = data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][condition]["continuous"]["mae"]
            principal = not (cell == "calce" and condition in ("US06_25C", "BJDST_25C"))
            for model in MODELS:
                values = np.asarray([
                    data["runs"][model][str(seed)]["metrics"][condition]["continuous"]["mae"]
                    for seed in SEEDS
                ], dtype=float)
                for reference, baseline, family_size in (
                    ("ecm_2rc_nls", baseline_2rc, 20),
                    ("physics_gradient", None, 18),
                ):
                    if reference == "physics_gradient":
                        baseline_values = np.asarray([
                            data["runs"]["physics_gradient"][str(seed)]["metrics"][condition]["continuous"]["mae"]
                            for seed in SEEDS
                        ], dtype=float)
                        delta = values - baseline_values
                        applicable = principal and model != "physics_gradient"
                    else:
                        delta = values - baseline
                        applicable = True
                    test = _tost(delta)
                    record = {
                        "cell": cell,
                        "condition": condition,
                        "model": model,
                        "reference": reference,
                        "principal_condition": principal,
                        "holm_family_n": family_size,
                        "median_difference_mV": float(np.median(delta)),
                        "Hodges_Lehmann_pseudomedian_mV": _walsh_pseudomedian(delta),
                        "p_lower_raw": test["lower"]["p_raw"],
                        "p_upper_raw": test["upper"]["p_raw"],
                        "p_TOST_raw": max(test["lower"]["p_raw"], test["upper"]["p_raw"]),
                        "p_lower_Holm": None,
                        "p_upper_Holm": None,
                        "p_TOST_Holm": None,
                        "holm_applicable": applicable,
                        "paired_two_sided_p_raw": test["paired_two_sided"]["p_raw"],
                    }
                    rows.append(record)
                    if applicable:
                        families.setdefault((model, reference, "lower"), []).append(
                            {"p_raw": record["p_lower_raw"], "_record": record}
                        )
                        families.setdefault((model, reference, "upper"), []).append(
                            {"p_raw": record["p_upper_raw"], "_record": record}
                        )
    for (model, reference, direction), family in families.items():
        expected = 20 if reference == "ecm_2rc_nls" else 18
        if len(family) != expected:
            raise ValueError(f"{model}/{reference}/{direction}: expected {expected}, got {len(family)}")
        _holm(family)
        key = "p_lower_Holm" if direction == "lower" else "p_upper_Holm"
        for item in family:
            item["_record"][key] = item["p_holm"]
    for row in rows:
        if row["p_lower_Holm"] is not None and row["p_upper_Holm"] is not None:
            row["p_TOST_Holm"] = max(row["p_lower_Holm"], row["p_upper_Holm"])

    table_path = E1_TABLE_PATH
    with table_path.open(newline="") as stream:
        saved = {(r["dataset"], r["condition"]): r for r in csv.DictReader(stream)}
    audits = []
    for row in rows:
        if row["reference"] != "ecm_2rc_nls" or not row["principal_condition"]:
            continue
        paper = saved[(row["cell"], row["condition"])]
        label = "physics_gradient" if row["model"] == "physics_gradient" else row["model"]
        value = paper[label]
        code = value.rsplit("; ", 1)[-1]
        paper_codes = code.split("/")
        if row["model"] == "physics_gradient":
            expected_codes = ["EQ" if row["p_TOST_Holm"] < .05 else "ND", "ref"]
        else:
            direct = next(r for r in rows if r["cell"] == row["cell"] and
                          r["condition"] == row["condition"] and r["model"] == row["model"] and
                          r["reference"] == "physics_gradient")
            expected_codes = [
                "EQ" if row["p_TOST_Holm"] < .05 else "ND",
                "EQ" if direct["p_TOST_Holm"] is not None and direct["p_TOST_Holm"] < .05 else "ND",
            ]
        match = paper_codes == expected_codes
        audits.append({
            "cell": row["cell"], "condition": row["condition"], "model": row["model"],
            "paper_code": code, "recomputed_code": "/".join(expected_codes), "match": match,
        })
    discrepancies = [entry for entry in audits if not entry["match"]]
    return rows, {
        "audited_rows": len(audits),
        "matches": len(audits) - len(discrepancies),
        "divergences": discrepancies,
        "paper_table_sha256": sha256(table_path),
        "interpretation": "E maps to EQ; N maps to ND. Holm is applied separately to both one-sided tests.",
    }


def _fmt_p(value: float | None) -> str:
    return "--" if value is None else f"{value:.3g}"


def write_d1(rows: list[dict], audit: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "tost_pvalues.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped = []
    for cell in ("panasonic", "calce"):
        grouped.extend([r for r in rows if r["cell"] == cell and r["reference"] == "ecm_2rc_nls"])
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Exact paired TOST results for continuous MAE (mV). Differences are model minus reference; "
        r"$p_{\mathrm{TOST}}$ is the larger raw one-sided $p$, and $p_{\mathrm{Holm}}$ is the larger "
        r"direction-specific Holm-adjusted $p$.}",
        r"\label{tab:reviewer-round2-tost}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Cell-condition & Variant & Median & HL & vs. 2RC: $p_L/p_U/p_T/p_H$ & "
        r"vs. gradient: $p_L/p_U/p_T/p_H$ \\",
        r"\midrule",
    ]
    for cell in ("panasonic", "calce"):
        conditions = [r["condition"] for r in grouped if r["cell"] == cell and
                      r["model"] == MODELS[0]]
        for condition in conditions:
            for model in MODELS:
                base = next(r for r in grouped if r["cell"] == cell and
                            r["condition"] == condition and r["model"] == model)
                direct = next(r for r in rows if r["cell"] == cell and
                              r["condition"] == condition and r["model"] == model and
                              r["reference"] == "physics_gradient")
                def pack(record: dict) -> str:
                    return "/".join(_fmt_p(record[key]) for key in
                                    ("p_lower_raw", "p_upper_raw", "p_TOST_raw", "p_TOST_Holm"))
                label = condition.replace("_", r"\_")
                model_label = model.replace("_", r"\_")
                lines.append(
                    f"{label} & {model_label} & "
                    f"{base['median_difference_mV']:.3f} & "
                    f"{base['Hodges_Lehmann_pseudomedian_mV']:.3f} & "
                    f"{pack(base)} & {pack(direct)} \\\\"
                )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize "
        r"Families: 20 conditions versus 2RC; 18 principal conditions versus physics-gradient. "
        r"The two non-principal CALCE conditions have raw direct-comparison $p$ values but no Holm "
        r"adjustment. HL is the Walsh Hodges--Lehmann pseudomedian. "
        f"Paper-code audit: {audit['matches']}/{audit['audited_rows']} matched; "
        f"divergences: {len(audit['divergences'])}."
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ]
    (OUT / "tost_pvalues.tex").write_text("\n".join(lines))
    (OUT / "e1_code_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def _history_curve(history: dict, key: str) -> np.ndarray:
    values = np.asarray(history[key], dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {key} in saved history")
    return values


def d3() -> tuple[list[dict], list[dict], dict]:
    with EPOCHS_PATH.open(newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    histories = resolve_histories(audit_rows)
    history_lookup = {(r["dataset"], r["model"], int(r["seed"])): r
                      for r in audit_rows if r["dataset"] in CELLS and r["model"] in MODELS}
    curves = []
    table = []
    figure_path = OUT / "learning_curves.pdf"
    if figure_path.exists():
        raise FileExistsError(figure_path)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), sharey=True,
                             layout="constrained")
    styles = {
        "physics_gradient": {"color": "black", "linestyle": "-", "marker": "o"},
        "output_residual": {"color": "0.35", "linestyle": "--", "marker": "s"},
        "ecm_ude": {"color": "0.65", "linestyle": "-.", "marker": "^"},
    }
    for axis, cell in zip(axes, CELLS):
        selected_epochs = {}
        for model in MODELS:
            runs = []
            selected = []
            for seed in SEEDS:
                path = histories[(cell, model, seed)]
                record = history_lookup[(cell, model, seed)]
                if sha256(path) != record["history_sha256"]:
                    raise ValueError(f"History SHA256 mismatch: {path}")
                history = read_json(path)
                train = _history_curve(history, "train_loss")
                val = _history_curve(history, "val_loss")
                if len(train) != len(val):
                    raise ValueError(f"Train/validation history lengths differ: {path}")
                epoch = int(record["selected_epoch"])
                if history.get("best_epoch") != epoch:
                    raise ValueError(f"Selected epoch mismatch: {path}")
                runs.append((train, val))
                selected.append(epoch)
            selected_epochs[model] = selected
            style = styles[model]
            for loss_index, loss_name in enumerate(("train_loss", "val_loss")):
                max_length = max(len(run[loss_index]) for run in runs)
                matrix = np.full((len(runs), max_length), np.nan)
                for i, run in enumerate(runs):
                    matrix[i, :len(run[loss_index])] = run[loss_index]
                epochs = np.arange(1, max_length + 1)
                low = np.nanmin(matrix, axis=0)
                median = np.nanmedian(matrix, axis=0)
                high = np.nanmax(matrix, axis=0)
                line_style = style["linestyle"] if loss_name == "train_loss" else ":"
                label = f"{model.replace('_', ' ')} {'train' if loss_name == 'train_loss' else 'validation'}"
                axis.fill_between(epochs, low, high, color=style["color"],
                                  alpha=0.07 if loss_name == "train_loss" else 0.12,
                                  hatch=None if loss_name == "train_loss" else "///",
                                  linewidth=0)
                axis.plot(epochs, median, color=style["color"], linestyle=line_style,
                          marker=style["marker"] if loss_name == "val_loss" else None,
                          markevery=max(1, max_length // 7), markersize=2.5,
                          linewidth=1.0, label=label)
                for i in range(max_length):
                    curves.append({
                        "cell": cell, "model": model, "loss": loss_name,
                        "epoch": i + 1, "n_seeds": int(np.isfinite(matrix[:, i]).sum()),
                        "minimum": float(low[i]), "median": float(median[i]),
                        "maximum": float(high[i]),
                    })
            median_epoch = float(np.median(selected))
            axis.axvline(median_epoch, color=style["color"], linestyle=":", linewidth=0.8)
            axis.plot([median_epoch], [np.median([runs[i][1][min(selected[i]-1, len(runs[i][1])-1)]
                                                   for i in range(len(runs))])],
                      marker=style["marker"], color=style["color"], markersize=4,
                      linestyle="none")
            selected_epochs[model] = {
                "median": median_epoch, "minimum": min(selected), "maximum": max(selected)
            }
            selected_train = []
            last_train = []
            reduction_train = []
            reduction_val = []
            for train, val in runs:
                epoch = int(np.argmin(val)) + 1
                selected_train.append(float(train[epoch - 1]))
                last_train.append(float(train[-1]))
                reduction_train.append(float((train[0] - train[-1]) / train[0]))
                reduction_val.append(float((val[0] - val[-1]) / val[0]))
            table.append({
                "cell": cell,
                "model": model,
                "n_seeds": len(runs),
                "selected_epoch_median": median_epoch,
                "selected_epoch_min": min(selected),
                "selected_epoch_max": max(selected),
                "train_loss_selected_median": float(np.median(selected_train)),
                "train_loss_last_median": float(np.median(last_train)),
                "relative_train_loss_reduction_median": float(np.median(reduction_train)),
                "relative_val_loss_reduction_median": float(np.median(reduction_val)),
                "relative_train_loss_reduction_min": float(np.min(reduction_train)),
                "relative_train_loss_reduction_max": float(np.max(reduction_train)),
                "relative_val_loss_reduction_min": float(np.min(reduction_val)),
                "relative_val_loss_reduction_max": float(np.max(reduction_val)),
            })
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_title("NCA" if cell == "panasonic" else "NMC")
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="0.85", linewidth=0.5)
        axis.legend(frameon=False, loc="best", ncol=2)
    axes[0].set_ylabel("Normalized voltage MSE")
    fig.savefig(figure_path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    with (OUT / "learning_curve_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    with (OUT / "learning_curve_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    provenance = {
        "training_audit_sha256": sha256(EPOCHS_PATH),
        "history_path_resolution": "SHA256 match only; no files copied or modified.",
        "resolved_history_paths": {
            f"{cell}/{model}/{seed}": str(path.relative_to(ROOT))
            for (cell, model, seed), path in histories.items()
        },
        "history_sha256": {
            f"{cell}/{model}/{seed}": sha256(path)
            for (cell, model, seed), path in histories.items()
        },
        "curve_figure_sha256": sha256(figure_path),
        "figure_width_inches": 7.16,
        "minimum_font_pt": 8,
        "font_requirement_met": True,
        "no_training": True,
    }
    (OUT / "learning_curve_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return curves, table, provenance


def d2_report() -> str:
    rows = [
        ("Fig. 3", "b", "figures_ieee.py --figures fig01_dynamic_corrections",
         "Saved E1 metrics and TOST; seconds to render; full E1 rerun is multi-hour."),
        ("Fig. 4", "b", "figures_ieee.py --figures fig02_physics_vs_lstm",
         "Saved LSTM/ECM-2RC metrics; seconds to render; full training is multi-hour."),
        ("Fig. 5", "c", "figures_ieee.py --figures fig03_lfp_hysteresis; exp_lfp_gonogo.py",
         "Seconds to render; raw fit/simulation rerun is tens of minutes to hours."),
        ("Fig. 6", "c", "figures_ieee.py --figures fig04_static_oracle; exp_lfp_gonogo.py",
         "Seconds to render; target-oracle NLS rerun is tens of minutes."),
        ("Fig. 7", "c", "figures_ieee.py --figures fig05_same_temperature_transfer; exp_lfp_static.py; exp_static_replication.py",
         "Seconds to render; fits from raw profiles take tens of minutes to hours."),
        ("Fig. 8", "c", "figures_ieee.py --figures fig06_thermal_transfer; exp_lfp_static.py",
         "Seconds to render; raw thermal profiles and NLS fits take tens of minutes."),
        ("Fig. 9", "b", "figures_ieee.py --figures fig07_temperature_interpolation",
         "Seconds to render from saved parameter tables and posterior draws."),
        ("Fig. 10", "b", "figures_ieee.py --figures fig08_gp_coverage_diagnostic",
         "Seconds to render from saved GP band/sample outputs; refitting is not required."),
        ("Table 4", "b", "reviewer_saved_analysis.py --output /path/to/empty/directory",
         "Seconds to regenerate the historical criteria table from saved outcomes."),
        ("Table 5", "c", "reviewer_saved_analysis.py; exp_unified.py --dataset all",
         "Seconds to regenerate from saved summaries; multi-hour full ten-seed E1 rerun."),
        ("Table 6", "c", "reviewer_saved_analysis.py; exp_param_ablation.py",
         "Seconds to regenerate from saved fit outputs; raw NLS refits take tens of minutes to hours."),
    ]
    lines = [
        "# Reproducibility scope",
        "",
        "Classification: (a) raw data plus released scripts; (b) saved parameters, checkpoints, "
        "histories, or result summaries; (c) both. Render-only commands read the saved artifacts "
        "named in each script; raw-data commands independently rerun the analysis where supported.",
        "",
        "| Item | Scope | Script and command | Approximate time |",
        "|---|---|---|---|",
    ]
    for item, scope, command, timing in rows:
        lines.append(f"| {item} | {scope} | `{command}` | {timing} |")
    lines += [
        "",
        "Render times are approximate process times, not model-fitting times. Full E1 training "
        "covers ten seeds, two cells, and all registered trainable models; saved-result rendering "
        "does not rerun training or overwrite legacy outputs.",
        "",
    ]
    return "\n".join(lines)


def write_report(d1_rows: list[dict], code_audit: dict, d3_table: list[dict],
                 provenance: dict) -> None:
    code = "All principal E/N codes match." if not code_audit["divergences"] else (
        f"{len(code_audit['divergences'])} code divergences; see e1_code_audit.json."
    )
    lines = [
        "# Part D — saved-result analyses",
        "",
        "No model was fitted for this part. Legacy results were read only; all outputs are under "
        "`results/reviewer_round2/D/`.",
        "",
        "## D1 — adjusted p-values",
        "",
        f"Computed {len(d1_rows)} reference-condition rows (20 conditions, three trainable variants, "
        "two references). TOST uses exact one-sided Wilcoxon tests at ±1 mV; Holm families are "
        "20 conditions against ECM-2RC and 18 principal conditions against physics-gradient. "
        f"{code}",
        "",
        "- `tost_pvalues.csv`: raw lower/upper p, raw TOST max, Holm-adjusted p per direction, "
        "adjusted TOST max, median difference and Walsh HL pseudomedian.",
        "- `tost_pvalues.tex`: compact booktabs appendix table.",
        "- `e1_code_audit.json`: row-level check against the saved E1 table.",
        "",
        "## D2 — reproducibility scope",
        "",
        "See `reproducibility_scope.md` for scripts, commands, classification, and approximate runtime.",
        "",
        "## D3 — learning curves",
        "",
        "The vector figure overlays the three trainable variants by cell. Each line is the seed "
        "median; the envelope is the observed-seed minimum–maximum without forward-filling early "
        "stopping. Vertical marks denote the median selected epoch; the summary table also gives "
        "the selected-epoch range. Loss reductions are computed per seed, then summarized by median.",
        "",
        "- `learning_curves.pdf` (7.16 in wide, grayscale, vector PDF).",
        "- `learning_curve_data.csv` and `learning_curve_summary.csv`.",
        "- `learning_curve_provenance.json`: exact history paths resolved by SHA256.",
        "",
        f"History checks: {len(provenance['resolved_history_paths'])} histories matched their recorded "
        "SHA256; no history/checkpoint source was copied or altered.",
        "",
    ]
    (OUT / "reproducibility_scope.md").write_text(d2_report())
    (OUT / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_json(SUMMARY_PATH)
    stats = read_json(STATS_PATH)
    if not summary.get("complete"):
        raise ValueError("Saved ten-seed results are incomplete")
    if not stats.get("equivalence"):
        raise ValueError("Saved statistical results are incomplete")
    d1_rows, audit = d1(summary)
    write_d1(d1_rows, audit)
    _, learning_summary, provenance = d3()
    write_report(d1_rows, audit, learning_summary, provenance)
    manifest = {
        "inputs_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (SUMMARY_PATH, STATS_PATH, E1_TABLE_PATH, EPOCHS_PATH,
                         ROOT / "exp_unified.py", ROOT / "exp_paper_v2_seeds.py")
        },
        "no_training": True,
        "code_audit": audit,
        "n_learning_curve_rows": len(learning_summary),
        "n_histories": len(provenance["resolved_history_paths"]),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
