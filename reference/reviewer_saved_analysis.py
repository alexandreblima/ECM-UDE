"""Reviewer tables, paired equivalence and training audit from saved artifacts."""
from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

from data_lfp import ROOT, save_json, sha256
from exp_paper_v2_seeds import adjust_holm, shifted_wilcoxon
from exp_param_ablation import verify
from figures_ieee import inspect_pdf
import protocol as P

MODELS = ("physics_gradient", "output_residual", "ecm_ude")
TABLES = ROOT/"results/paper_tables"
TRAINING = ROOT/"results/training_audit"
TOST = ROOT/"results/variant_tost"
SOURCES: dict[str, str] = {}


def read(path: Path) -> dict:
    SOURCES[str(path)] = sha256(path)
    return json.loads(path.read_text())


def escape(value) -> str:
    text = "--" if value is None else str(value)
    return "".join({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in text)


def table(out: Path, stem: str, rows: list[dict], caption: str, headers=None, notes="") -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out/f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if stem == "epochs":
        fields = fields[:7]
        rows = [{k: f"{v:.5g}" if isinstance(v, float) else v for k, v in row.items()} for row in rows]
        headers = ["Cell", "Model", "Seed", "Selected", "Executed", "Last train", "Last validation"]
    headings = headers or fields
    columns = ("ll"+"p{0.12\\textwidth}"*(len(fields)-2) if stem == "criteria" else
               "p{0.12\\textwidth}"*len(fields) if len(fields) > 5 else "l"*len(fields))
    if stem == "E1":
        columns = r"llr p{0.18\textwidth} p{0.18\textwidth} p{0.18\textwidth}"
    if stem == "conditions":
        columns = r"llll p{0.24\textwidth} p{0.17\textwidth} l"
    if stem == "criteria":
        columns = r"p{0.08\textwidth} p{0.23\textwidth} p{0.25\textwidth} p{0.15\textwidth} p{0.18\textwidth}"
    if stem == "epochs":
        columns = r"llrrr p{0.13\textwidth} p{0.13\textwidth}"
    if stem == "comparison":
        fields = ["dataset", "condition", "model", "median_difference_mV",
                  "Hodges_Lehmann_pseudomedian_mV", "equivalent", "p_lower_Holm", "p_upper_Holm"]
        columns = r"ll p{0.15\textwidth} rrr rr"
        headings = ["Cell", "Condition", "Variant", "Median", "HL", "EQ", "Lower Holm", "Upper Holm"]
        rows = [{k: f"{v:.5g}" if isinstance(v, float) else v for k, v in row.items()} for row in rows]
    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize",
             r"\caption{"+escape(caption)+"}", r"\label{tab:reviewer-"+stem+"}",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{"+columns+"}", r"\toprule",
             " & ".join(escape(h) for h in headings)+r" \\", r"\midrule"]
    lines += [" & ".join(escape(row[k]) for k in fields)+r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines += [r"\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize "+escape(notes)+r"\end{minipage}"]
    lines += [r"\end{table*}", ""]
    (out/f"{stem}.tex").write_text("\n".join(lines))
    if stem in ("conditions", "epochs", "comparison"):
        head_end = lines.index(r"\midrule")+1
        footer_start = lines.index(r"\bottomrule")
        body = lines[head_end:footer_start]
        pages = []
        size = 12 if stem == "conditions" else 20
        for start in range(0, len(body), size):
            part = lines[:head_end]+body[start:start+size]+lines[footer_start:]
            part = [line.replace(r"\label{tab:reviewer-"+stem+"}",
                                 r"\label{tab:reviewer-"+stem+"-"+str(start//size+1)+"}") for line in part]
            pages += part
        (out/f"{stem}.tex").write_text("\n".join(pages))


def paired_tost(summary: dict) -> list[dict]:
    records = []
    for model in ("ecm_ude", "output_residual"):
        family = []
        for dataset, data in summary["datasets"].items():
            for condition in data["conditions"]:
                if dataset == "calce" and condition in ("US06_25C", "BJDST_25C"):
                    continue
                delta = np.array([data["runs"][model][str(s)]["metrics"][condition]["continuous"]["mae"]-
                                  data["runs"]["physics_gradient"][str(s)]["metrics"][condition]["continuous"]["mae"]
                                  for s in range(42, 52)])
                walsh = [(delta[i]+delta[j])/2 for i in range(len(delta)) for j in range(i, len(delta))]
                record = {"dataset": dataset, "condition": condition, "model": model,
                          "reference": "physics_gradient", "differences_mV": delta.tolist(),
                          "median_difference_mV": float(np.median(delta)),
                          "Hodges_Lehmann_pseudomedian_mV": float(np.median(walsh)),
                          "lower": shifted_wilcoxon(delta+1, "greater"),
                          "upper": shifted_wilcoxon(delta-1, "less")}
                family.append(record)
            # Holm spans both datasets, not separate families.
        if len(family) != 18:
            raise ValueError("Direct-variant Holm family must contain18 conditions")
        adjust_holm([r["lower"] for r in family])
        adjust_holm([r["upper"] for r in family])
        for r in family:
            r["equivalent"] = r["lower"]["p_holm"] < .05 and r["upper"]["p_holm"] < .05
        records += family
    TOST.mkdir(parents=True, exist_ok=True)
    info = {"records": records, "estimand": "Wilcoxon signed-rank location; Hodges-Lehmann Walsh pseudomedian; "
            "under symmetry equals the difference-distribution center/median. Descriptive sample median "
            "is reported separately and is not the signed-rank statistic.",
            "statistic": "Wplus=sum positive signed absolute ranks of shifted differences",
            "rounding": "10decimal shifted differences, exactzeros removed before ranking; allzeros p=1",
            "ties": "scipy.stats.rankdata average ties on nonzero absolute shifted differences",
            "exact_method": "Exhaust all2^n assignments of signs; count tail incl equality; no normal approximation "
                            "and no scipy.wilcoxon(method=exact) shortcut with ties",
            "margin_mV": 1, "Holm": "18conditions per pair per unilateral direction",
            "library_versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
            "implementation": str(ROOT/"exp_paper_v2_seeds.py"),
            "implementation_sha256": sha256(ROOT/"exp_paper_v2_seeds.py"),
            "LSTM_bootstrap": {"unit": "paired seed MAE_LSTM minus deterministic2RC MAE, windowed",
                               "n_seeds": 10, "n_resamples": 20000, "replacement": True,
                               "statistic": "samplemedian", "interval": "percentile2.5/97.5",
                               "numpy_quantile_method": "linear(default)", "RNG": "default_rng(20261101)",
                               "not_temporal_bootstrap": True}}
    save_json(TOST/"summary.json", info)
    flat = [{k: r[k] for k in ("dataset", "condition", "model", "median_difference_mV",
                              "Hodges_Lehmann_pseudomedian_mV", "equivalent")} |
            {"p_lower_Holm": r["lower"]["p_holm"], "p_upper_Holm": r["upper"]["p_holm"],
             "Wplus_lower": r["lower"]["statistic"], "Wplus_upper": r["upper"]["statistic"],
             "n_nonzero_lower": r["lower"]["n_nonzero"], "n_nonzero_upper": r["upper"]["n_nonzero"],
             "p_lower_raw": r["lower"]["p_raw"], "p_upper_raw": r["upper"]["p_raw"]}
            for r in records]
    table(TOST, "comparison", flat, "Direct paired variant equivalence, continuous MAE.")
    return records


def training_audit(summary: dict) -> dict:
    TRAINING.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "savefig.bbox": None,
                         "axes.spines.top": False, "axes.spines.right": False})
    for dataset, data in summary["datasets"].items():
        for model in (*MODELS, "lstm"):
            histories = []
            for seed in range(42, 52):
                run = data["runs"][model][str(seed)]
                history = read((ROOT/run["history"]).resolve())
                checkpoint = (ROOT/run["checkpoint"]).resolve()
                if sha256(checkpoint) != run["sha256"]:
                    raise ValueError("Saved checkpoint changed")
                histories.append(history)
                n = len(history["train_loss"])
                best = run["best_epoch"]
                if history["best_epoch"] != best or len(history["val_loss"]) != n:
                    raise ValueError("Training history/checkpoint selection mismatch")
                rows.append({"dataset": dataset, "model": model, "seed": seed, "selected_epoch": best,
                             "executed_epochs": n, "final_train_loss": history["train_loss"][-1],
                             "final_val_loss": history["val_loss"][-1],
                             "selected_epoch_train_loss": history["train_loss"][best-1],
                             "selected_epoch_val_loss": history["val_loss"][best-1],
                             "history_path": run["history"], "history_sha256": sha256(ROOT/run["history"]),
                             "checkpoint_sha256": run["sha256"]})
            maximum = max(len(h["train_loss"]) for h in histories)
            aggregate = []
            fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
            for key, color, style, name in (("train_loss", "#0072B2", "-", "Train"),
                                          ("val_loss", "#333333", "--", "Validation")):
                matrix = np.full((10, maximum), np.nan)
                for i, h in enumerate(histories):
                    matrix[i, :len(h[key])] = h[key]
                low, median, high = (f(matrix, axis=0) for f in (np.nanmin, np.nanmedian, np.nanmax))
                epochs = np.arange(1, maximum+1)
                ax.fill_between(epochs, low, high, color=color, alpha=.12,
                                hatch="///" if key == "train_loss" else "...")
                ax.plot(epochs, median, color=color, linestyle=style, linewidth=1, label=name)
                for epoch in range(maximum):
                    aggregate.append({"dataset": dataset, "model": model, "loss": key, "epoch": epoch+1,
                                      "n_observed_seeds": int(np.isfinite(matrix[:, epoch]).sum()),
                                      "minimum": float(low[epoch]), "median": float(median[epoch]),
                                      "maximum": float(high[epoch])})
            ax.set(xlabel="Epoch", ylabel="Normalized voltage MSE", yscale="log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.legend(frameon=False, loc="best")
            path = TRAINING/f"{dataset}_{model}_learning.pdf"
            fig.savefig(path, bbox_inches=None, metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)
            curves[f"{dataset}/{model}"] = {"file": str(path), "sha256": sha256(path),
                                           "aggregate": aggregate, **inspect_pdf(path, 3.5)}
    table(TRAINING, "epochs", rows, "Saved training and selected epochs; final losses refer to last executed epoch.")
    code = {}
    for name in ("exp_unified.py", "protocol.py", "training.py", "data.py", "unified_dynamics.py",
                 "exp_lfp_static.py", "exp_static_replication.py", "exp_a123_temperature.py"):
        code[name] = sha256(ROOT/name)
    audit = {"runs": rows, "curves": curves, "source_sha256": code,
             "NLS_source_intervals": {d: x["conditions"][next(iter(x["conditions"]))]["split"]
                                      for d, x in summary["datasets"].items()},
             "base_training_protocol": P.manifest(), "effective_seeds": list(range(42, 52)),
             "history_provenance": SOURCES.copy(),
             "no_training": True, "early_stop_tail": "No forwardfill: median/min/max only observed seeds; "
             "late epochs may have fewer than10 seeds. Aggregates include observed seed counts.",
             "loss_audit": "UnifiedECMUDE has no compute_loss; plain nn.MSELoss through training._compute_batch_loss; "
                          "LSTM also plain MSE. History residual losses are zero.",
             "source_code_commit": "e10d552 original training; ten-seed runtime verified in81ac420"}
    save_json(TRAINING/"summary.json", audit)
    return audit


def numeric_E1(summary, original_stats, paired) -> None:
    rows = []
    for dataset, data in summary["datasets"].items():
        for condition in data["conditions"]:
            if dataset == "calce" and condition in ("US06_25C", "BJDST_25C"):
                continue
            row = {"dataset": dataset, "condition": condition,
                   "ECM2RC_mV": f'{data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][condition]["continuous"]["mae"]:.3f}'}
            for model in MODELS:
                values = [data["runs"][model][str(s)]["metrics"][condition]["continuous"]["mae"]
                          for s in range(42, 52)]
                old = next(r for r in original_stats["equivalence"] if r["dataset"] == dataset and
                           r["condition"] == condition and r["mode"] == "continuous" and r["model"] == model)
                direct = ("ref" if model == "physics_gradient" else
                          "EQ" if next(r for r in paired if r["dataset"] == dataset and
                                      r["condition"] == condition and r["model"] == model)["equivalent"] else "ND")
                row[model] = (f'{np.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]; '
                              f'{"EQ" if old["equivalent"] else "ND"}/{direct}')
            rows.append(row)
    table(TABLES, "E1", rows, "Continuous MAE (mV), ten-seed median [min,max] and equivalence.",
          ["Cell", "Condition", "2RC", "Gradient", "Residual", "ECM-UDE"],
          "EQ: equivalence demonstrated; ND: not demonstrated. Codes are versus2RC/versusgradient. "
          "Original versus2RC uses Holm20, direct variants Holm18. Min/max are seed ranges, not confidence intervals. "
          "CALCE US06/BJDST25 fallback excluded, not replaced by corrected metrics.")


def conditions(summary, ablation) -> None:
    rows = []
    for dataset, data in summary["datasets"].items():
        for condition, meta in data["conditions"].items():
            profile, temp = condition.split("_")
            source = condition == ("UDDS_25C" if dataset == "panasonic" else "FUDS_25C")
            rows.append({"experiment": "E1", "cell": dataset, "profile": profile, "temperature_C": temp[:-1],
                         "z0_origin": f'{meta["z0_estimation"]["method"]}; z0={meta["z0"]:.8f}',
                         "role": "guarded80% train/20% validation; not independent test" if source else "zero-shot test",
                         "principal": not (dataset == "calce" and condition in ("US06_25C", "BJDST_25C"))})
    for cell in ("cell_A_a123", "cell_B_mendeley"):
        for experiment in ("E2", "E3"):
            for profile in ("DST", "FUDS", "US06"):
                rows.append({"experiment": experiment, "cell": cell, "profile": profile, "temperature_C": "25 nominal",
                             "z0_origin": "CCCV full charge z0=1" if cell == "cell_A_a123" else
                                          "authorized z0=1 assumption; preparation absent",
                             "role": "full source NLS; no validation split" if profile == "DST" else "heldout cycle test",
                             "principal": True})
    static = read(ROOT/"static_replication_protocol.json")["datasets"]
    for cell, cfg in static.items():
        for condition in [cfg["source"], *cfg["primary_targets"], *cfg["secondary_targets"]]:
            profile, temperature = condition.split("_")
            if cell == "calce" and condition in ablation["calce_audit"]:
                z0 = f'samefile fullcharge count z0={ablation["calce_audit"][condition]["z0"]:.8f}'
            else:
                meta = summary["datasets"][cell]["conditions"][condition]
                z0 = f'{meta["z0_estimation"]["method"]}; z0={meta["z0"]:.8f}'
            rows.append({"experiment": "E3 audited" if cell == "calce" and condition.endswith("_25C") else "E3",
                         "cell": cell, "profile": profile, "temperature_C": temperature[:-1],
                         "z0_origin": z0+"; V6 transfers fitted source z0",
                         "role": "full source NLS" if condition == cfg["source"] else
                                 "primary heldout test" if condition in cfg["primary_targets"] else "secondary thermal test",
                         "principal": condition not in cfg["secondary_targets"]})
    for temperature in (0, 10, 25, 40, 50):
        for profile in ("DST", "FUDS", "US06"):
            role = ("full DST25 source NLS" if profile == "DST" and temperature == 25 else
                    "DST local diagnostic; not transferred-fit calibration" if profile == "DST" else
                    "thermal heldout test" if temperature != 25 else "sameT reference only")
            rows.append({"experiment": "E4 thermal", "cell": "cell_A_a123", "profile": profile,
                         "temperature_C": temperature, "z0_origin": "CCCV fullcharge z0=1; fitted source z0 for V6",
                         "role": role, "principal": (profile != "DST" and temperature != 25) or
                                                   (profile == "DST" and temperature == 25)})
    for temperature in (0, 10, 25, 40, 50):
        for profile in ("DST", "FUDS", "US06"):
            if profile != "DST" and temperature not in (10, 40):
                continue
            diagnostic = profile == "DST" and temperature in (10, 40)
            rows.append({"experiment": "E4 interp", "cell": "cell_A_a123", "profile": profile,
                         "temperature_C": temperature,
                         "z0_origin": "CCCV z0=1 reference; predicted source-fit z0 forP1/P2/P3/P4",
                         "role": "local oracle only; excluded from interpolation calibration" if diagnostic else
                                 "calibration DST0/25/50 only" if profile == "DST" else "heldout interpolation test",
                         "principal": not diagnostic})
    table(TABLES, "conditions", rows, "Experimental conditions and causal initial-state provenance.",
          notes="E3 audited CALCE25 rows describe this reviewer's all-profile charge-count audit; historical "
                "FUDS/DST initial SOC and fitted V6 are unchanged in their saved results. E2 V4 is a target-fitted "
                "oracle, excluded from the hysteresis criterion. E4 thermal and interpolation roles are separate.")
    save_json(TABLES/"conditions.json", rows)


def criteria(original_stats, paired, ablation) -> None:
    thermal = read(ROOT/"results/lfp_static_temperature/cell_A_a123/summary.json")
    interpolation = read(ROOT/"results/a123_temperature_interp/summary.json")
    rows = [
        {"experiment": "E1 vs2RC", "registered_criterion": "TOST +/-1 mV; exact shifted Wilcoxon; both pHolm<.05; 20 conditions/model/direction",
         "observed": "continuous gradient17/20,residual17/20,UDE16/20; principal15/18,15/18,14/18",
         "verdict": "partial equivalence, not universal", "registration_commit": "d5cdc37"},
        {"experiment": "E1 variants", "registered_criterion": "paired seeds42-51; TOST +/-1mV; Holm18 perpair/direction",
         "observed": "; ".join(f"{m}:{sum(r['equivalent'] for r in paired if r['model']==m)}/18"
                               for m in ("ecm_ude", "output_residual")),
         "verdict": "see conditionwise EQ/ND", "registration_commit": "d14935e"},
        {"experiment": "E2", "registered_criterion": "ESC gain>=10% versusbestV1/V2 and |h|>.9 less90% time on BOTHtargets",
         "observed": "A123-1.62/-6.14%; Mendeley13.19/23.97%; saturation passes",
         "verdict": "A NO-GO; B GO; primary NO-GO", "registration_commit": "da67c21"},
        {"experiment": "E3 historical", "registered_criterion": "BestV5/V6/V7 gain>=25% on ALLsameTtargets",
         "observed": "historical fourpositive; corrected CALCE US0623.45% fails",
         "verdict": "A/B/Pan historicalpositive; correctedCALCE negative", "registration_commit": "50e85e1;78fefb3;308dc2a audit"},
        {"experiment": "E3 wide CALCE", "registered_criterion": "V6wide z0[.7,1]; >=25% vs auditedV1 on all3targets",
         "observed": "; ".join(f'{k.split("_")[0]}:{v["reduction"]:.2%}' for k, v in
                              ablation["blocks"]["calce"]["wide_decision"]["profiles"].items()),
         "verdict": ablation["blocks"]["calce"]["wide_decision"]["verdict"], "registration_commit": "d14935e"},
        {"experiment": "E3 monotone", "registered_criterion": "V7m >=25% allsameTtargets; source-only 500evaluations/3starts",
         "observed": "Mendeley4.04/-1.72%; A/Pan/CALCE sourceNLSnonconverged, Q/z0unidentifiable",
         "verdict": "Mendeley negative; others inconclusive", "registration_commit": "8a0e2e2;3c149a8 failurepolicy"},
        {"experiment": "E4 transfer", "registered_criterion": ">=25% bestV5/V6/V7 vsV1 bothprofiles in>=3/4temperatures",
         "observed": f'{thermal["decision"]["passing_temperatures"]}/4 pass; localV6diagnostic not in criterion',
         "verdict": thermal["decision"]["verdict"], "registration_commit": "78fefb3;3d0d047 localdiagnostic"},
        {"experiment": "E4 interpolation", "registered_criterion": "BestP2/P3/P4 >=50% floor-to-local recovery all4heldoutconditions",
         "observed": "P2:61.83,62.03,71.09,116.48%; P4coverage not calibrated",
         "verdict": interpolation["decision"]["verdict"], "registration_commit": "b2d945c"},
        {"experiment": "Ablation descriptive", "registered_criterion": ">=70% jointgain majority positivegain conditions; both/neither=>joint",
         "observed": "; ".join(f"{k}:{v['verdict']}" for k, v in ablation["interpretation"].items()),
         "verdict": "descriptive blockwise, not causal proof", "registration_commit": "d14935e"}]
    table(TABLES, "criteria", rows, "Registered criteria, observations and unchanged historical decisions.")


def main() -> None:
    verify()
    summary = read(ROOT/"results/unified_10seeds/summary.json")
    stats = read(ROOT/"results/unified_10seeds/statistics.json")
    ablation = read(ROOT/"results/param_ablation/summary.json")
    paired = paired_tost(summary)
    training = training_audit(summary)
    numeric_E1(summary, stats, paired)
    conditions(summary, ablation)
    criteria(stats, paired, ablation)
    for path, digest in SOURCES.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Audit source changed:{path}")
    save_json(TABLES/"manifest.json", {"protocol_sha256": sha256(ROOT/"reviewer_round_protocol.json"),
                                      "registration_commit": "d14935e", "inputs_sha256": SOURCES,
                                      "generator_sha256": sha256(Path(__file__)),
                                      "no_new_training": True, "training_runs": len(training["runs"])})
    print("Saved paired TOST36rows, training80runs/8PDFs and E1/conditions/criteria tables.")


if __name__ == "__main__":
    main()
