"""Render ledger evidence from saved results only, without model evaluation."""
from __future__ import annotations

import hashlib
import json
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Figures:
    def __init__(self) -> None:
        self.sources: dict[str, str] = {}
        self.entries: list[dict] = []
        self.read("claims_ledger.md", text=True)
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
        source = "results/unified/summary.json"
        saved = self.read(source)
        models = ("physics_gradient", "output_residual", "ecm_ude")
        labels = ("Physics gradient", "Output residual", "ECM-UDE")
        fig, axes = self.canvas(7.16, 4.6, 2)
        for ax, dataset in zip(axes[0], ("panasonic", "calce")):
            data = saved["datasets"][dataset]
            conditions = [c for c in data["conditions"] if dataset != "calce" or c not in EXCLUDED]
            y = np.arange(len(conditions))
            for index, (model, label) in enumerate(zip(models, labels)):
                delta = np.array([[data["runs"][model][str(seed)]["metrics"][c]["continuous"]["mae"]-
                                   data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][c]["continuous"]["mae"]
                                   for seed in range(42, 47)] for c in conditions])
                ax.errorbar(delta.mean(axis=1), y+(index-1)*.20, xerr=delta.std(axis=1, ddof=1),
                            fmt=MARKERS[index], color=COLORS[index], markersize=3.5,
                            capsize=2, linewidth=.8, label=label)
            ax.axvline(0, color="black", linestyle=":", linewidth=.8)
            ax.set(yticks=y, yticklabels=[condition_label(c) for c in conditions],
                   xlabel="MAE difference from ECM-2RC (mV)", ylabel="Condition")
            ax.invert_yaxis()
            ax.margins(y=.15)
        axes[0, 0].legend(loc="lower right", frameon=False)
        self.save(fig, "fig01_dynamic_corrections", 1, [source],
                  "Dynamic and output corrections relative to ECM-2RC: Panasonic (left), CALCE (right).",
                  "Continuous MAE, original seeds 42--46; markers show mean and bars sample SD across seeds. "
                  "Negative differences indicate nominal improvement, not statistical significance. "
                  "CALCE US06/BJDST 25 C fallback conditions excluded; source scores are validation scores. "
                  "Physics gradient has no neural network; trainable physical models initialize from 1RC.")
        fig, axes = self.canvas(7.16, 4.6, 2)
        for ax, dataset in zip(axes[0], ("panasonic", "calce")):
            data = saved["datasets"][dataset]
            conditions = [c for c in data["conditions"] if dataset != "calce" or c not in EXCLUDED]
            y = np.arange(len(conditions))
            baseline = [data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][c]["windowed"]["mae"]
                        for c in conditions]
            values = np.array([[data["runs"]["lstm"][str(seed)]["metrics"][c]["windowed"]["mae"]
                                for seed in range(42, 47)] for c in conditions])
            ax.barh(y-.18, baseline, .34, color="white", edgecolor=COLORS[0], hatch="///", label="ECM-2RC")
            ax.barh(y+.18, values.mean(axis=1), .34, xerr=values.std(axis=1, ddof=1), capsize=2,
                    color="white", edgecolor=COLORS[1], hatch="xxx", label="LSTM")
            ax.set(yticks=y, yticklabels=[condition_label(c) for c in conditions],
                   xlabel="Windowed MAE (mV)", ylabel="Condition")
            ax.invert_yaxis()
            ax.margins(y=.12)
        axes[0, 0].legend(frameon=False, loc="lower right")
        self.save(fig, "fig02_physics_vs_lstm", 2, [source],
                  "Windowed prediction errors of ECM-2RC and LSTM: Panasonic (left), CALCE (right).",
                  "Deterministic ECM-2RC; LSTM mean and sample SD for seeds 42--46. "
                  "CALCE US06/BJDST 25 C fallback excluded. No comparison to continuous LSTM is implied. "
                  "Same cells and finite training budgets do not establish universal superiority.")

    def lfp(self) -> None:
        sources = [f"results/lfp_gonogo/{cell}/summary.json" for cell in ("cell_A_a123", "cell_B_mendeley")]
        cells = [self.read(path) for path in sources]
        labels = [f"{cell}\n{profile}" for cell in ("A123", "Mendeley") for profile in ("FUDS", "US06")]
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
                  "in both target profiles. A123 NO-GO, Mendeley GO; primary/global NO-GO. "
                  "Mendeley nominal temperature/preparation and long RC time constants remain caveats; "
                  "gain is not proof of identified minor loops.")
        fig, axes = self.canvas(3.5, 3.4)
        ax = axes[0, 0]
        names = [f"{cell} {profile}" for cell in ("A123", "Mendeley") for profile in ("DST", "FUDS", "US06")]
        y = np.arange(6)
        for index, model in enumerate(("V1", "V4")):
            values = [data["global_mae_mV"][profile][model] for data in cells for profile in ("DST", "FUDS", "US06")]
            ax.barh(y+(index-.5)*.36, values, .34, color="white", edgecolor=COLORS[index],
                    hatch=("///", "xxx")[index], label=("V1: fixed OCV", "V4: target oracle")[index])
        ax.set(yticks=y, yticklabels=names, xlabel="MAE (mV)")
        ax.invert_yaxis()
        ax.margins(y=.17)
        ax.legend(frameon=False, loc="lower right")
        self.save(fig, "fig04_static_oracle", 4, sources,
                  "Voltage-error reduction when OCV and RC parameters are reidentified on each target cycle.",
                  "V4 is a target-fitted degree-9 OCV plus RC oracle, not transferable prediction. "
                  "This comparison does not isolate OCV from RC or SOC error. "
                  "These are the original go/no-go V1/V4 fits, with their original RC bounds.")

    def transfer(self) -> None:
        sources = [f"results/lfp_static/{cell}/summary.json" for cell in ("cell_A_a123", "cell_B_mendeley")]
        sources += [f"results/static_replication/{cell}/summary.json" for cell in ("panasonic", "calce")]
        saved = [self.read(path) for path in sources]
        labels, gains = [], []
        for data, cell, profiles in zip(saved, ("A123", "Mendeley", "Panasonic", "CALCE"),
                                       (("FUDS", "US06"), ("FUDS", "US06"),
                                        ("US06_25C", "LA92_25C", "HWFET_25C"), ("DST_25C",))):
            for profile in profiles:
                values = data["global_mae_mV"][profile]
                labels.append(f"{cell}\n{profile.split('_')[0]}")
                gains.append(100*(1-min(values[m] for m in ("V5", "V6", "V7"))/values["V1"]))
        fig, axes = self.canvas(7.16, 2.6)
        ax = axes[0, 0]
        ax.bar(np.arange(len(gains)), gains, color="white", edgecolor=COLORS[0], hatch="///")
        ax.axhline(25, color=COLORS[2], linestyle="--", label="Registered threshold")
        ax.set(xticks=range(len(gains)), xticklabels=labels, ylabel="Source-static MAE reduction (%)", ylim=(0, 105))
        ax.legend(frameon=False, loc="upper right")
        self.save(fig, "fig05_same_temperature_transfer", 5, sources,
                  "Source-only static recalibration transfers to held-out cycles at nominally matched temperature.",
                  "Best of V5/V6/V7 may differ by target; threshold is >=25%. "
                  "CALCE DST is an eligible individual target, NOT evidence that the dataset passes all targets. "
                  "Old US06/BJDST 25 C fallback results and nonphysical V5 extrapolations are excluded. "
                  "The later corrected CALCE dataset verdict is negative. Panasonic V1 uses dynamic checkpoint OCV, "
                  "not independent low-current OCV. Mendeley temperature is nominal.")
        source = "results/lfp_static_temperature/cell_A_a123/summary.json"
        thermal = self.read(source)
        fig, axes = self.canvas(7.16, 2.8, 2)
        temperature = [0, 10, 40, 50]
        for ax, profile in zip(axes[0], ("FUDS", "US06")):
            values = [thermal["temperatures"][str(t)]["global_mae_mV"][profile] for t in temperature]
            series = (("V1", [v["V1"] for v in values]),
                      ("Best transfer", [min(v[m] for m in ("V5", "V6", "V7")) for v in values]),
                      ("Local V6", [v["V6_local"] for v in values]))
            for index, (name, mae) in enumerate(series):
                ax.plot(temperature, mae, marker=MARKERS[index], linestyle=STYLES[index],
                        color=COLORS[index], label=name)
            ax.set(xticks=temperature, xlabel="Target temperature (°C)", ylabel="MAE (mV)", ylim=(0, 118))
            ax.legend(frameon=False, loc="upper right")
        self.save(fig, "fig06_thermal_transfer", 5, [source],
                  "A123 thermal transfer and local recalibration: FUDS (left), US06 (right).",
                  "Transferred fits are frozen from DST 25 C; no target refit. "
                  "Best transferred V5/V6/V7 fails the >=25% criterion in both profiles at every temperature. "
                  "Local V6 is fitted on the same-temperature DST and is only an out-of-criterion diagnostic, "
                  "not part of the transfer decision. All fits retain fixed OCV25/Q25 references.")
        source = "results/a123_temperature_interp/summary.json"
        interpolation = self.read(source)
        conditions = ("10C_FUDS", "10C_US06", "40C_FUDS", "40C_US06")
        fig, axes = self.canvas(3.5, 3.0)
        ax = axes[0, 0]
        x = np.arange(4)
        for index, model in enumerate(("P2", "P3", "P4")):
            values = [100*interpolation["decision"]["conditions"][c]["fractions"][model] for c in conditions]
            ax.plot(x, values, marker=MARKERS[index], linestyle=STYLES[index], color=COLORS[index], label=model)
        ax.axhline(50, color="gray", linestyle=":", label="Criterion: 50%")
        ax.set(xticks=x, xticklabels=["10°C\nFUDS", "10°C\nUS06", "40°C\nFUDS", "40°C\nUS06"],
               ylabel="Floor-to-local-oracle recovery (%)", ylim=(0, 155))
        ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=.8, handlelength=1.6)
        self.save(fig, "fig07_temperature_interpolation", 5, [source],
                  "Held-out temperature recovery by linear interpolation (P2), parametric fit (P3) and GP (P4).",
                  "Calibration only DST 0/25/50 C; held-out FUDS/US06 at 10/40 C. "
                  "Recovery=(MAE_floor-MAE_method)/(MAE_floor-MAE_local_oracle); ratios are not clipped. "
                  "P2 exceeds 50% in all four conditions. Recovery >100% is valid because the local NLS "
                  "oracle is not a mathematical bound. This is not a new fit.")
        fig, axes = self.canvas(3.5, 2.6)
        ax = axes[0, 0]
        rows = [next(r for r in interpolation["metrics"]
                     if r["method"] == "P4" and f'{r["temperature_C"]}C_{r["profile"]}' == c) for c in conditions]
        ax.axhspan(85, 99, facecolor=".9", hatch="...", edgecolor=".65", label="Diagnostic range: 85–99%")
        ax.axhline(95, color="gray", linestyle=":", linewidth=.8)
        ax.plot(x, [100*r["coverage_95"] for r in rows], color=COLORS[0], marker="o", linestyle="--",
                label="P4 empirical coverage")
        ax.set(xticks=x, xticklabels=["10°C\nFUDS", "10°C\nUS06", "40°C\nFUDS", "40°C\nUS06"],
               ylabel="Coverage of nominal 95% band (%)", ylim=(45, 110))
        ax.legend(frameon=False, loc="lower right", handlelength=1.5)
        self.save(fig, "fig08_gp_coverage_diagnostic", 5, [source],
                  "Empirical coverage of the P4 nominal 95% voltage band on held-out conditions.",
                  "Diagnostic outside the main interpolation criterion. Coverage is not calibrated in all "
                  "conditions: both 10 C targets fail the 85–99% range. "
                  "Bands use 200 bounded posterior parameter samples, without measurement-noise, "
                  "NLS-calibration or fitted-mean uncertainty.")

    def finish(self) -> None:
        for relative, expected in self.sources.items():
            if digest(ROOT/relative) != expected:
                raise RuntimeError(f"Source changed while rendering: {relative}")
        manifest = {"source_sha256": self.sources, "generator_sha256": digest(Path(__file__)),
                    "font": self.font, "font_sha256": digest(Path(self.font)),
                    "matplotlib_version": matplotlib.__version__, "figures": self.entries,
                    "no_model_fitting_or_evaluation": True, "minimum_font_size_pt": 8,
                    "allowed_widths_in": [3.5, 7.16], "raster_images_allowed": False}
        (OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
        lines = ["# Figuras finais IEEE", "",
                 "PDFs vetoriais; largura exata 3,5 in ou 7,16 in; Times New Roman incorporada, minimo 8 pt.",
                 "Sem titulos internos. Cores acompanhadas de marcadores, tipos de linha ou hachuras distintos;",
                 "as series continuam identificaveis sem cor. Rotulo canonico: HWFET.", "",
                 "Dados exclusivamente de resultados salvos citados pelo registro de afirmacoes; nenhum retreino,",
                 "reajuste ou nova simulacao. Os numeros de cinco seeds nao foram substituidos pelos de dez seeds.", "",
                 "| Figura | Achado | Largura (in) | Arquivos de dados de origem |",
                 "|---|---|---:|---|"]
        for entry in self.entries:
            links = ", ".join(f"[{p}](../{p})" for p in entry["sources"])
            lines.append(f'| [{entry["file"]}]({entry["file"]}) | {entry["claim"]} | '
                         f'{entry["width_in"]:.2f} | {links} |')
        lines += ["", "## Legendas propostas e ressalvas", ""]
        for entry in self.entries:
            lines += [f'### {entry["file"]}', "", entry["caption"], "", entry["caveat"], ""]
        lines += ["## Validacao", "", "[manifest.json](manifest.json) registra SHA-256 dos dados e PDFs,",
                  "dimensoes, fontes incorporadas, minimo tipografico e ausencia de imagens raster.",
                  "Extensao de texto e dimensoes foram verificadas antes e depois da exportacao.",
                  "O gerador verifica os hashes das fontes ao terminar; resultados anteriores nao sao sobrescritos.", "",
                  "Para reproduzir: execute [figures_ieee.py](../figures_ieee.py) com o ambiente cientifico",
                  "existente e pypdf. [test_figures_ieee.py](../test_figures_ieee.py) valida os artefatos.", ""]
        (OUTPUT/"figures_index.md").write_text("\n".join(lines))


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
    if expected_width not in (3.5, 7.16) or abs(width-expected_width) > 1e-6:
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
    OUTPUT.mkdir(exist_ok=True)
    figures = Figures()
    figures.unified()
    figures.lfp()
    figures.transfer()
    figures.finish()
    print(f"Generated and verified {len(figures.entries)} vector IEEE PDFs in {OUTPUT}")


if __name__ == "__main__":
    main()
