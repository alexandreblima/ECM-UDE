"""Render ledger evidence from saved results only, without model evaluation."""
from __future__ import annotations

import hashlib
import argparse
import csv
import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/"figures_ieee"
COLORS = ("#0072B2", "#D55E00", "#333333", "#009E73")
MARKERS = ("o", "s", "^", "D")
STYLES = ("-", "--", "-.", ":")
EXCLUDED = {"US06_25C", "BJDST_25C"}
LABELS = {
    "V1": "Reference map", "V4": "Target oracle", "V6": "Source recalibration",
    "V6-wide": "Source recalibration, wide z0 bound",
    "Transferred V6": "Transferred from 25 °C",
    "Local V6": "Local recalibration (diagnostic)",
    "P1": "Nearest temperature", "P2": "Linear", "P3": "Arrhenius", "P4": "Gaussian process",
    "Mendeley": "Replica LFP", "Panasonic": "NCA", "CALCE": "NMC",
    "calce": "NMC", "V1: fixed OCV": "Reference map", "V4: target oracle": "Target oracle",
    "P4 empirical coverage": "Gaussian-process band coverage",
    "Floor-to-local-oracle recovery (%)": "Recovered fraction of the gap (%)",
}
ARCHIVE = OUTPUT/"archive/91b3a08"
DIRECT_ARCHIVE = OUTPUT/"archive/bca878a"
NEAREST_ARCHIVE = OUTPUT/"archive/d5fa7e1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_direct_records(summary: dict, tests: list[dict]) -> list[dict]:
    records = []
    for row in tests:
        if (row["reference"] != "physics_gradient" or row["principal_condition"] != "True"
            or row["model"] not in ("output_residual", "ecm_ude")):
            continue
        dataset, condition, model = row["cell"], row["condition"], row["model"]
        runs = summary["datasets"][dataset]["runs"]
        differences = [runs[model][str(seed)]["metrics"][condition]["continuous"]["mae"] -
                       runs["physics_gradient"][str(seed)]["metrics"][condition]["continuous"]["mae"]
                       for seed in range(42, 52)]
        records.append({"dataset": dataset, "condition": condition, "model": model,
                        "reference": "physics_gradient", "differences_mV": differences,
                        "median_difference_mV": float(np.median(differences)),
                        "equivalent": float(row["p_TOST_Holm"]) < .05})
    if len(records) != 36:
        raise ValueError("Expected 36 saved paired comparisons")
    return records


class Figures:
    def __init__(self, selected: set[str] | None = None) -> None:
        self.sources: dict[str, str] = {}
        self.entries: list[dict] = []
        self.selected = selected
        self.font = font_manager.findfont("Times New Roman", fallback_to_default=False)
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
            "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": .6, "lines.linewidth": 1,
            "lines.markersize": 4, "hatch.linewidth": .5,
            "savefig.bbox": None, "figure.constrained_layout.use": True,
            "figure.constrained_layout.w_pad": .04, "figure.constrained_layout.h_pad": .04,
        })

    def read(self, relative: str, text: bool = False):
        path = ROOT/relative
        self.sources[relative] = digest(path)
        return path.read_text() if text else json.loads(path.read_text())

    def canvas(self, width: float, height: float, columns: int = 1):
        return plt.subplots(1, columns, figsize=(width, height), squeeze=False, layout="constrained")

    def save(self, fig, stem: str, claim: int, sources: list[str], caption: str, caveat: str) -> None:
        if self.selected is not None and stem not in self.selected:
            plt.close(fig)
            return
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width_px, height_px = fig.bbox.width, fig.bbox.height
        for ax in fig.axes:
            if ax.get_title() or ax.get_title(loc="left") or ax.get_title(loc="right"):
                raise ValueError(f"Internal title is prohibited: {stem}")
        unused_ticks = set()
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not lo <= tick.get_loc() <= hi:
                        unused_ticks.update((tick.label1, tick.label2))
        for artist in fig.findobj(Text):
            if artist in unused_ticks or not artist.get_visible() or not artist.get_text().strip():
                continue
            if artist.get_fontsize() < 8:
                raise ValueError(f"Undersized font: {stem}")
            if any(alias in artist.get_text() for alias in ("HWFT", "HWFTa")):
                raise ValueError("Use the canonical HWFET label")
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width_px+1 or box.y1 > height_px+1:
                raise ValueError(f"Text clipped in {stem}: {artist.get_text()}")
        path = OUTPUT/f"{stem}.pdf"
        fig.savefig(path, format="pdf", bbox_inches=None, metadata={
            "Title": caption, "Author": "ECM-UDE", "Subject": f"Claims ledger finding {claim}",
            "CreationDate": None, "ModDate": None,
        })
        validation = inspect_pdf(path, float(fig.get_figwidth()))
        self.entries.append({"file": path.name, "claim": claim, "caption": caption, "caveat": caveat,
                             "sources": sources, "sha256": digest(path), **validation})
        plt.close(fig)

    def unified(self) -> None:
        source = "results/unified_10seeds/statistics.json"
        saved = self.read(source)
        direct_source = "results/reviewer_round2/D/tost_pvalues.csv"
        summary = self.read("results/unified_10seeds/summary.json")
        self.sources[direct_source] = digest(ROOT/direct_source)
        with (ROOT/direct_source).open(newline="") as stream:
            tests = list(csv.DictReader(stream))
        direct = {"records": paired_direct_records(summary, tests)}
        models = ("physics_gradient", "output_residual", "ecm_ude")
        labels = ("Physics gradient", "Output residual", "ECM-UDE")
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 8.6), squeeze=False, layout="constrained")
        for column, dataset in enumerate(("panasonic", "calce")):
            conditions = list(dict.fromkeys(r["condition"] for r in saved["equivalence"]
                              if r["dataset"] == dataset and r["mode"] == "continuous"
                              and not r["excluded_from_main_claims"]))
            y = np.arange(len(conditions))
            for row in range(2):
                ax = axes[row, column]
                for index, (model, label) in enumerate(zip(models, labels)):
                    if row == 1 and model == "physics_gradient":
                        continue
                    pool = direct["records"] if row == 1 else saved["equivalence"]
                    for position, condition in enumerate(conditions):
                        matches = [r for r in pool if r["dataset"] == dataset
                                   and r["condition"] == condition and r["model"] == model
                                   and (r["reference"] == "physics_gradient" if row == 1
                                        else r["mode"] == "continuous")]
                        if len(matches) != 1:
                            raise ValueError(f"Expected one saved comparison: {row}/{dataset}/{condition}/{model}")
                        record = matches[0]
                        center = record["median_difference_mV" if row == 1 else "difference_median_mV"]
                        delta = record["differences_mV"]
                        if len(delta) != 10 or not np.isfinite([center, *delta]).all():
                            raise ValueError(f"Expected ten finite saved differences: {dataset}/{condition}/{model}")
                        offset = (index-1)*.20 if row == 0 else (index-1.5)*.24
                        ax.errorbar(center, position+offset,
                                    xerr=[[center-min(delta)], [max(delta)-center]],
                                    fmt=MARKERS[index], color=COLORS[index], markersize=4,
                                    markerfacecolor=COLORS[index] if record["equivalent"] else "white",
                                    capsize=2, linewidth=.7, label=label if position == 0 else None)
                ax.axvspan(-1, 1, facecolor=".94", edgecolor=".8", hatch="...", zorder=-1)
                ax.axvline(0, color="black", linestyle=":", linewidth=.8)
                ax.set(yticks=y, yticklabels=[condition_label(c) for c in conditions],
                       xlabel=f"MAE difference from {'ECM-2RC' if row == 0 else 'physics-gradient'} (mV)",
                       ylabel="Condition")
                ax.invert_yaxis()
                ax.margins(y=.15)
                panel = "abcd"[row*2+column]
                ax.text(.02, .98, f"({panel})", transform=ax.transAxes, ha="left", va="top")
        lower_limits = (min(ax.get_xlim()[0] for ax in axes[1]),
                        max(ax.get_xlim()[1] for ax in axes[1]))
        for ax in axes[1]:
            ax.set_xlim(lower_limits)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center",
                   ncol=3, frameon=False)
        self.save(fig, "fig01_dynamic_corrections", 1, [source, direct_source],
                  "Ten-seed continuous MAE differences: (a,b) against ECM-2RC, "
                  "(c,d) paired neural variants against physics-gradient; NCA left, NMC right.",
                  "Markers are median differences; bars show the ten-seed minimum/maximum, NOT confidence intervals. "
                  "Shading denotes +/-1 mV. Filled markers: both one-sided exact Wilcoxon TOST pHolm<0.05; "
                  "hollow markers: equivalence not demonstrated. Top: Holm families have 20 conditions per model/direction; "
                  "gradient/residual/UDE are equivalent in 15/18, 15/18, 14/18 principal conditions. "
                  "Bottom: saved paired differences for seeds 42--51, Holm over 18 conditions per pair/direction; "
                  "output residual and ECM-UDE each equivalent in 18/18 conditions against physics-gradient. "
                  "The two bottom panels share an x scale. No tests or model predictions were recomputed. "
                  "Median outside the margin and inconclusive corrected tests are distinct failures (see ledger). "
                  "CALCE US06/BJDST 25 C fallback conditions excluded; source scores are validation scores. "
                  "Physics gradient has no neural network; trainable physical models initialize from 1RC.")
        if self.selected is None or "fig01_dynamic_corrections" in self.selected:
            self.entries[-1]["panels"] = [
                {"panel": f"({chr(97+row*2+column)})", "cell": ("NCA", "NMC")[column],
                 "reference": "ECM-2RC" if row == 0 else "physics-gradient",
                 "xlim_mV": list(axes[row, column].get_xlim()),
                 "conditions": [t.get_text() for t in axes[row, column].get_yticklabels()],
                 "source": source if row == 0 else direct_source}
                for row in range(2) for column in range(2)
            ]
        source = "results/unified_10seeds/summary.json"
        saved = self.read(source)
        fig, axes = self.canvas(7.16, 4.6, 2)
        for ax, dataset in zip(axes[0], ("panasonic", "calce")):
            data = saved["datasets"][dataset]
            conditions = [c for c in data["conditions"] if dataset != "calce" or c not in EXCLUDED]
            y = np.arange(len(conditions))
            baseline = [data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][c]["windowed"]["mae"]
                        for c in conditions]
            values = np.array([[data["runs"]["lstm"][str(seed)]["metrics"][c]["windowed"]["mae"]
                                for seed in range(42, 52)] for c in conditions])
            ax.barh(y-.18, baseline, .34, color="white", edgecolor=COLORS[0], hatch="///", label="ECM-2RC")
            medians = np.median(values, axis=1)
            ranges = np.vstack((medians-values.min(axis=1), values.max(axis=1)-medians))
            ax.barh(y+.18, medians, .34, xerr=ranges, capsize=2,
                    color="white", edgecolor=COLORS[1], hatch="xxx", label="LSTM")
            ax.set(yticks=y, yticklabels=[condition_label(c) for c in conditions],
                   xlabel="Windowed MAE (mV)", ylabel="Condition")
            ax.invert_yaxis()
            ax.margins(y=.12)
        axes[0, 0].legend(frameon=False, loc="lower right")
        self.save(fig, "fig02_physics_vs_lstm", 2, [source],
                  "Windowed prediction errors of ECM-2RC and LSTM: Panasonic (left), CALCE (right).",
                  "Deterministic ECM-2RC; LSTM median and min/max across ten seeds 42--51, not confidence intervals. "
                  "CALCE US06/BJDST 25 C fallback excluded. No comparison to continuous LSTM is implied. "
                  "Same cells and finite training budgets do not establish universal superiority.")

    def lfp(self) -> None:
        sources = [f"results/lfp_gonogo/{cell}/summary.json" for cell in ("cell_A_a123", "cell_B_mendeley")]
        cells = [self.read(path) for path in sources]
        labels = [f"{cell}\n{profile}" for cell in ("A123", "Replica LFP") for profile in ("FUDS", "US06")]
        tests = [data["decision"]["profile_tests"][profile] for data in cells for profile in ("FUDS", "US06")]
        fig, axes = self.canvas(7.16, 2.7, 2)
        for ax, key, multiplier, ylabel, threshold in (
            (axes[0, 0], "mae_reduction_fraction", 100, "ESC MAE reduction (%)", 10),
            (axes[0, 1], "saturation_fraction", 100, "Time with |h| > 0.9 (%)", 90),
        ):
            values = [t[key]*multiplier for t in tests]
            ax.bar(np.arange(4), values, color="white", edgecolor=COLORS[0], hatch="///")
            ax.axhline(threshold, color=COLORS[2], linestyle="--", label="Registered threshold")
            ax.set(xticks=range(4), xticklabels=labels, ylabel=ylabel)
            ax.legend(frameon=False, loc="upper left")
            if threshold == 10:
                ax.set_ylim(-12, 42)
            else:
                ax.set_ylim(0, 115)
        self.save(fig, "fig03_lfp_hysteresis", 3, sources,
                  "ESC hysteresis gain and saturation diagnostic in both LFP cells.",
                  "Gain is relative to the best of V1/V2; GO needs gain >=10% and saturation <90% "
                  "in both target profiles. A123 NO-GO, Replica LFP GO; primary/global NO-GO. "
                  "Replica LFP nominal temperature/preparation and long RC time constants remain caveats; "
                  "gain is not proof of identified minor loops.")
        fig, axes = self.canvas(3.5, 3.4)
        ax = axes[0, 0]
        names = [f"{cell} {profile}" for cell in ("A123", "Replica LFP") for profile in ("DST", "FUDS", "US06")]
        y = np.arange(6)
        for index, model in enumerate(("V1", "V4")):
            values = [data["global_mae_mV"][profile][model] for data in cells for profile in ("DST", "FUDS", "US06")]
            ax.barh(y+(index-.5)*.36, values, .34, color="white", edgecolor=COLORS[index],
                    hatch=("///", "xxx")[index], label=LABELS[model])
        ax.set(yticks=y, yticklabels=names, xlabel="MAE (mV)")
        ax.invert_yaxis()
        ax.margins(y=.17)
        ax.legend(frameon=False, loc="lower right")
        self.save(fig, "fig04_static_oracle", 4, sources,
                  "Voltage-error reduction when OCV and RC parameters are reidentified on each target cycle.",
                  "Target oracle is a target-fitted degree-9 OCV plus RC oracle, not transferable prediction. "
                  "This comparison does not isolate OCV from RC or SOC error. "
                  "These are the original go/no-go V1/V4 fits, with their original RC bounds.")

    def transfer(self) -> None:
        sources = [f"results/lfp_static/{cell}/summary.json" for cell in ("cell_A_a123", "cell_B_mendeley")]
        sources += [f"results/static_replication/{cell}/summary.json" for cell in ("panasonic", "calce")]
        saved = [self.read(path) for path in sources]
        fix_source = "results/calce_z0_fix/summary.json"
        correction = self.read(fix_source)
        sources.append(fix_source)
        wide_source = "results/param_ablation/calce/summary.json"
        wide = self.read(wide_source)
        sources.append(wide_source)
        fig, axes = self.canvas(7.16, 3.5, 4)
        for ax, data, cell, profiles in zip(axes[0], saved, ("A123", "Mendeley", "Panasonic", "CALCE"),
                                       (("FUDS", "US06"), ("FUDS", "US06"),
                                        ("US06_25C", "LA92_25C", "HWFET_25C"),
                                        ("DST_25C", "US06_25C", "BJDST_25C"))):
            baseline, recalibrated = [], []
            for profile in profiles:
                values = data["global_mae_mV"][profile]
                if cell == "CALCE" and profile in correction["static"]:
                    values = {m: correction["static"][profile][m]["after_metrics_mV"]["mae"]
                              for m in ("V1", "V6")}
                reference = values["V1"]
                if cell == "CALCE":
                    reference = next(r["mae_mV"] for r in wide["metrics"]
                                     if r["target"] == profile and r["variant"] == "V1"
                                     and r["scope"] == "global")
                baseline.append(reference)
                recalibrated.append(values["V6"])
            x = np.arange(len(profiles))
            spacing, width = (.24, .22) if cell == "CALCE" else (.36, .34)
            offset = 1 if cell == "CALCE" else .5
            ax.bar(x-offset*spacing, baseline, width, color="white", edgecolor=COLORS[0],
                   hatch="///", label=LABELS["V1"])
            ax.bar(x+(1-offset)*spacing, recalibrated, width, color="white", edgecolor=COLORS[1],
                   hatch="xxx", label=LABELS["V6"])
            if cell == "CALCE":
                wide_values = [next(r["mae_mV"] for r in wide["metrics"]
                                   if r["target"] == p and r["variant"] == "V6-wide"
                                   and r["scope"] == "global") for p in profiles]
                ax.bar(x+spacing, wide_values, width, color="white", edgecolor=COLORS[2],
                       hatch="...", label=LABELS["V6-wide"])
            ax.set(xticks=x, xticklabels=[p.split("_")[0] for p in profiles],
                   xlabel=LABELS.get(cell, cell)+" (25°C)", ylabel="MAE (mV)", ylim=(0, 43))
            ax.tick_params(axis="x", labelrotation=45)
        fig.legend(*axes[0, -1].get_legend_handles_labels(), frameon=False,
                   loc="outside upper center", ncol=2)
        self.save(fig, "fig05_same_temperature_transfer", 5, sources,
                  "Same-temperature reference map versus source recalibration: A123, Replica LFP, NCA and NMC.",
                  "All original encodings and values are retained except the explicitly authorized "
                  "NMC DST reference-map correction: 14.005509340066444 to 16.281115916311858 mV, "
                  "using the saved audited ablation baseline (no new evaluation). "
                  "NMC additionally shows source recalibration with wide z0 bound from saved parameter ablation; "
                  "the original z0 bound [0.9, 1] series is retained. Wide MAE values (DST/US06/BJDST): "
                  "9.94170496204159/10.787598237869936/11.515968681930916 mV. "
                  "A123 gains 81.63/76.62%; Replica LFP 18.97/32.56%; NCA 2.90/17.59/0.51%; "
                  "NMC gains use the audited reference map; the previously reported DST gain -124.05% "
                  "used the superseded baseline. Original source recalibration passes 25% on all targets only in A123, "
                  "a descriptive restriction of the saved original best-of test, not new preregistration. "
                  "NMC US06/BJDST reference map uses audited z0 near 0.8; source recalibration keeps source-fitted z0. "
                  "Corrected original NMC best-of verdict is negative. No V5/V7/V7m principal curves. "
                  "NCA OCV comes from a dynamic checkpoint; Replica LFP temperature is nominal. "
                  "The V7m robustness failure/nonidentifiability is separate (see ledger).")
        source = "results/lfp_static_temperature/cell_A_a123/summary.json"
        thermal = self.read(source)
        fig, axes = self.canvas(7.16, 2.8, 2)
        temperature = [0, 10, 40, 50]
        for ax, profile in zip(axes[0], ("FUDS", "US06")):
            values = [thermal["temperatures"][str(t)]["global_mae_mV"][profile] for t in temperature]
            series = (("V1", [v["V1"] for v in values]),
                      ("Transferred V6", [v["V6"] for v in values]),
                      ("Local V6", [v["V6_local"] for v in values]))
            for index, (name, mae) in enumerate(series):
                ax.plot(temperature, mae, marker=MARKERS[index], linestyle=STYLES[index],
                        color=COLORS[index], label=LABELS[name])
            ax.set(xticks=temperature, xlabel="Target temperature (°C)", ylabel="MAE (mV)", ylim=(0, 118))
            ax.legend(frameon=False, loc="upper right")
        self.save(fig, "fig06_thermal_transfer", 5, [source],
                  "A123 thermal transfer and local recalibration: FUDS (left), US06 (right).",
                  "Transferred V6 is frozen from DST 25 C; no target refit. "
                  "V6 fails the >=25% gain versus V1 at all eight target conditions. "
                  "Local V6 is fitted on the same-temperature DST and is only an out-of-criterion diagnostic, "
                  "not part of the transfer decision; it recovers 67.04–87.61% of transferred V6 error. "
                  "This thermal evidence and P2 parameter interpolation do not depend on V5, V7 or V7m. "
                  "All fits retain fixed OCV25/Q25 references.")
        source = "results/a123_temperature_interp/summary.json"
        interpolation = self.read(source)
        conditions = ("10C_FUDS", "10C_US06", "40C_FUDS", "40C_US06")
        fig, axes = self.canvas(3.5, 3.0)
        ax = axes[0, 0]
        x = np.arange(4)
        for index, model in enumerate(("P2", "P3", "P4")):
            values = [100*interpolation["decision"]["conditions"][c]["fractions"][model] for c in conditions]
            ax.plot(x, values, marker=MARKERS[index], linestyle=STYLES[index], color=COLORS[index], label=LABELS[model])
        nearest_source = "results/a123_temperature_interp/table.csv"
        table = list(csv.DictReader(io.StringIO(self.read(nearest_source, text=True))))
        nearest_values = []
        for condition in conditions:
            matches = [r for r in table if r["method"] == "P1"
                       and f'{r["temperature_C"]}C_{r["profile"]}' == condition]
            if len(matches) != 1 or not matches[0]["recovered_fraction"]:
                raise ValueError(f"Expected one saved nearest-temperature fraction: {condition}")
            value = 100*float(matches[0]["recovered_fraction"])
            if not np.isfinite(value):
                raise ValueError(f"Invalid saved nearest-temperature fraction: {condition}")
            nearest_values.append(value)
        nearest_color = "#666666"
        ax.plot(x, nearest_values, marker="D", linestyle=(0, (3, 1, 1, 1, 1, 1)),
                color=nearest_color, label=LABELS["P1"])
        lower_limit = -20
        off_scale = []
        for position, value in enumerate(nearest_values):
            if value < lower_limit:
                label = f"{value:.0f}%".replace("-", "−")
                ax.plot(position, lower_limit, marker="D", color=nearest_color,
                        linestyle="None", clip_on=False)
                ax.annotate("", xy=(position, lower_limit), xytext=(position, -3),
                            arrowprops={"arrowstyle": "->", "color": nearest_color, "lw": .8})
                ax.annotate(label, xy=(position, lower_limit), xytext=(6, 10),
                            textcoords="offset points", ha="left", va="bottom", color=nearest_color)
                off_scale.append({"condition": conditions[position], "value_percent": value,
                                  "display_y_percent": lower_limit, "annotation": label})
        ax.axhline(50, color="gray", linestyle=":", label="Criterion: 50%")
        ax.set(xticks=x, xticklabels=["10°C\nFUDS", "10°C\nUS06", "40°C\nFUDS", "40°C\nUS06"],
               ylabel="Recovered fraction of the gap (%)", ylim=(lower_limit, 155))
        fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper center",
                   ncol=2, columnspacing=.8, handlelength=1.6)
        self.save(fig, "fig07_temperature_interpolation", 5, [source, nearest_source],
                  "Held-out temperature recovery by Linear, Arrhenius, Gaussian process and Nearest temperature.",
                  "Calibration only DST 0/25/50 C; held-out FUDS/US06 at 10/40 C. "
                  "Recovery=(MAE_floor-MAE_method)/(MAE_floor-MAE_local_oracle); ratios are not clipped. "
                  "P2 exceeds 50% in all four conditions. Recovery >100% is valid because the local NLS "
                  "oracle is not a mathematical bound. Nearest temperature uses saved P1 recovered fractions "
                  "from table.csv; gray diamonds and a dash-dot-dot line distinguish it in grayscale. "
                  "The y axis is limited to [-20, 155]%; the two 10 C values below -20% are shown "
                  "by boundary diamonds, downward arrows and rounded annotations (-145%, -111%). "
                  "The line retains the exact values and is clipped at the axis boundary, not rescaled. "
                  "No axis break; all original series values are unchanged. This is not a new fit.")
        if self.selected is None or "fig07_temperature_interpolation" in self.selected:
            self.entries[-1]["nearest_temperature"] = {
                "source": nearest_source, "method": "P1", "column": "recovered_fraction",
                "conditions": list(conditions), "values_percent": nearest_values,
                "ylim_percent": [lower_limit, 155], "off_scale": off_scale,
                "representation": "Exact line clipped; boundary diamonds, downward arrows and rounded labels",
                "legend_order": ["Linear", "Arrhenius", "Gaussian process", LABELS["P1"], "Criterion: 50%"],
            }
        fig, axes = self.canvas(3.5, 2.6)
        ax = axes[0, 0]
        rows = [next(r for r in interpolation["metrics"]
                     if r["method"] == "P4" and f'{r["temperature_C"]}C_{r["profile"]}' == c) for c in conditions]
        ax.axhspan(85, 99, facecolor=".9", hatch="...", edgecolor=".65", label="Diagnostic range: 85–99%")
        ax.axhline(95, color="gray", linestyle=":", linewidth=.8, label="Nominal coverage: 95%")
        ax.plot(x, [100*r["coverage_95"] for r in rows], color=COLORS[0], marker="o", linestyle="--",
                label="Gaussian-process band coverage")
        ax.set(xticks=x, xticklabels=["10°C\nFUDS", "10°C\nUS06", "40°C\nFUDS", "40°C\nUS06"],
               ylabel="Coverage of nominal 95% band (%)", ylim=(45, 110))
        ax.legend(frameon=False, loc="lower right", handlelength=1.5)
        self.save(fig, "fig08_gp_coverage_diagnostic", 5, [source],
                  "Gaussian-process band coverage of the nominal 95% voltage band on held-out conditions.",
                  "Diagnostic outside the main interpolation criterion. Coverage is not calibrated in all "
                  "conditions: both 10 C targets fail the 85–99% range. "
                  "Bands use 200 bounded posterior parameter samples, without measurement-noise, "
                  "NLS-calibration or fitted-mean uncertainty.")

    def finish(self) -> None:
        for relative, expected in self.sources.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Source changed while rendering: {relative}")
        manifest = {"source_sha256": self.sources, "generator_sha256": digest(Path(__file__)),
                    "font_family": "Times New Roman", "font_sha256": digest(Path(self.font)),
                    "matplotlib_version": matplotlib.__version__, "figures": self.entries,
                    "no_model_fitting_or_evaluation": True}
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def lstm_ecm_ratios(saved: dict) -> dict:
    conditions, all_values = [], []
    for dataset, cell in (("panasonic", "NCA"), ("calce", "NMC")):
        data = saved["datasets"][dataset]
        for condition in data["conditions"]:
            if dataset == "calce" and condition in EXCLUDED:
                continue
            baseline = data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][condition]["windowed"]["mae"]
            if baseline <= 0 or not np.isfinite(baseline):
                raise ValueError(f"Invalid saved ECM windowed MAE for {dataset}/{condition}")
            values = [data["runs"]["lstm"][str(seed)]["metrics"][condition]["windowed"]["mae"]/baseline
                      for seed in range(42, 52)]
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid saved windowed MAE for {dataset}/{condition}")
            conditions.append({"cell": cell, "condition": condition, "ratios": values,
                               "median": float(np.median(values))})
            all_values.extend(values)
    if len(conditions) != 18 or len(all_values) != 180:
        raise ValueError("Expected 18 principal conditions and ten seeds")
    return {"source": "results/unified_10seeds/summary.json", "mode": "windowed",
            "seeds": list(range(42, 52)), "count": len(all_values),
            "minimum": min(all_values), "maximum": max(all_values),
            "median": float(np.median(all_values)), "conditions": conditions}


def condition_label(name: str) -> str:
    profile, temperature = name.rsplit("_", 1)
    profile = {"HWFT": "HWFET", "HWFTa": "HWFET"}.get(profile, profile)
    return f"{profile} {temperature[:-1]}°C"


def inspect_pdf(path: Path, expected_width: float) -> dict:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError("Publication figure must be a single page")
    page = reader.pages[0]
    width, height = float(page.mediabox.width)/72, float(page.mediabox.height)/72
    if expected_width not in (3.5, 6.0, 7.16) or abs(width-expected_width) > 1e-6:
        raise ValueError(f"IEEE column width mismatch: {path}")
    fonts, images = [], []

    def resources(objects) -> None:
        for key, ref in objects.get("/XObject", {}).items():
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Image":
                images.append(str(key))
            elif obj.get("/Subtype") == "/Form":
                resources(obj.get("/Resources", {}))
        for ref in objects.get("/Font", {}).values():
            font = ref.get_object()
            descendants = font.get("/DescendantFonts", [font])
            for child in descendants:
                child = child.get_object()
                descriptor = child.get("/FontDescriptor")
                if descriptor is None or "/FontFile2" not in descriptor.get_object():
                    raise ValueError(f"TrueType font not embedded in {path}")
                fonts.append(str(child["/BaseFont"]))

    resources(page["/Resources"])
    sizes = []

    def text_visitor(text, cm, tm, font, size):
        if text.strip():
            sizes.append(float(size))

    text = page.extract_text(visitor_text=text_visitor)
    if images or not sizes or min(sizes) < 8 or not fonts:
        raise ValueError(f"Invalid vector/font properties: {path}")
    if any(alias in text for alias in ("HWFT", "HWFTa")):
        raise ValueError(f"Uncorrected drive-cycle label in {path}")
    return {"width_in": width, "height_in": height, "minimum_font_size_pt": min(sizes),
            "embedded_fonts": sorted(set(fonts)), "raster_image_count": len(images), "page_count": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", nargs="+", help="Render only the specified figure stems")
    args = parser.parse_args()
    groups = {
        "unified": {"fig01_dynamic_corrections", "fig02_physics_vs_lstm"},
        "lfp": {"fig03_lfp_hysteresis", "fig04_static_oracle"},
        "transfer": {"fig05_same_temperature_transfer", "fig06_thermal_transfer",
                     "fig07_temperature_interpolation", "fig08_gp_coverage_diagnostic"},
    }
    known = set().union(*groups.values())
    selected = set(args.figures) if args.figures else known
    if not selected <= known:
        raise ValueError(f"Unknown figures: {selected - known}")
    OUTPUT.mkdir(exist_ok=True)
    figures = Figures(selected)
    for group, stems in groups.items():
        if stems & selected:
            getattr(figures, group)()
    if {entry["file"][:-4] for entry in figures.entries} != selected:
        raise ValueError("Not all requested figures were rendered")
    figures.finish()
    print(f"Generated and verified {len(figures.entries)} vector IEEE PDFs in {OUTPUT}")


if __name__ == "__main__":
    main()
