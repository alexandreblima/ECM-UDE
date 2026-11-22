"""Validate delivered IEEE PDF dimensions, fonts, provenance and ledger scope."""
from __future__ import annotations

import json
import csv
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader

import figures_ieee
from figures_ieee import ARCHIVE, DIRECT_ARCHIVE, NEAREST_ARCHIVE, COLORS, LABELS, MARKERS, OUTPUT, ROOT, STYLES, condition_label, digest, inspect_pdf
from public_integrity import verify as verify_public


def render_public():
    temporary = tempfile.TemporaryDirectory(prefix="public-figures-")
    output = Path(temporary.name)
    with patch.object(figures_ieee, "OUTPUT", output):
        renderer = figures_ieee.Figures()
        renderer.unified()
        renderer.lfp()
        renderer.transfer()
        renderer.finish()
    return temporary, output


def capture_figures(module):
    captured = {}

    def capture(self, fig, stem, *args):
        captured[stem] = fig
        self.entries.append({"file": stem+".pdf"})

    with patch.object(module.Figures, "save", capture):
        renderer = module.Figures()
        renderer.unified()
        renderer.lfp()
        renderer.transfer()
    return captured


class IEEEFiguresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global OUTPUT
        cls.temporary, OUTPUT = render_public()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.manifest = json.loads((OUTPUT/"manifest.json").read_text())

    def test_all_findings_have_final_vector_pdfs(self):
        entries = self.manifest["figures"]
        self.assertEqual(len(entries), 8)
        self.assertEqual({e["claim"] for e in entries}, set(range(1, 6)))
        self.assertEqual({p.name for p in OUTPUT.glob("*.pdf")}, {e["file"] for e in entries})
        for entry in entries:
            with self.subTest(figure=entry["file"]):
                path = OUTPUT/entry["file"]
                self.assertEqual(digest(path), entry["sha256"])
                actual = inspect_pdf(path, entry["width_in"])
                for key, value in actual.items():
                    self.assertEqual(entry[key], value)
                self.assertGreaterEqual(actual["minimum_font_size_pt"], 8)
                self.assertEqual(actual["raster_image_count"], 0)
                self.assertTrue(all("TimesNewRoman" in name for name in actual["embedded_fonts"]))
                self.assertEqual(float(PdfReader(path).pages[0].cropbox.width)/72, entry["width_in"])

    def test_source_hashes_unchanged_and_index_resolves(self):
        self.assertEqual(self.manifest["generator_sha256"], digest(ROOT/"figures_ieee.py"))
        self.assertTrue(self.manifest["no_model_fitting_or_evaluation"])
        for source, expected in self.manifest["source_sha256"].items():
            self.assertEqual(digest(ROOT/source), expected)
            verify_public(ROOT/source)
        for entry in self.manifest["figures"]:
            self.assertTrue(entry["sources"])
            self.assertTrue(all(source in self.manifest["source_sha256"] for source in entry["sources"]))

    def test_baseline_results_manuscript_and_report_unchanged(self):
        for path in ("results/unified_10seeds/summary.json",
                     "results/unified_10seeds/statistics.json",
                     "results/reviewer_round2/D/tost_pvalues.csv"):
            verify_public(ROOT/path)

    def test_canonical_labels_and_noncolor_encodings(self):
        self.assertEqual(condition_label("HWFTa_25C"), "HWFET 25°C")
        self.assertEqual(condition_label("HWFET_25C"), "HWFET 25°C")
        self.assertEqual(len(set(MARKERS)), len(COLORS))
        self.assertEqual(len(set(STYLES)), len(COLORS))
        text = PdfReader(OUTPUT/"fig05_same_temperature_transfer.pdf").pages[0].extract_text()
        self.assertIn("HWFET", text)
        self.assertNotIn("HWFT", text)
        thermal = next(e for e in self.manifest["figures"] if e["file"] == "fig06_thermal_transfer.pdf")
        self.assertIn("out-of-criterion", thermal["caveat"])
        oracle = next(e for e in self.manifest["figures"] if e["claim"] == 4)
        self.assertIn("not transferable", oracle["caveat"])

    def test_descriptive_labels_and_unaffected_pdf_hashes(self):
        forbidden = r"\b(?:V1|V4|V6(?:-wide)?|P1|P2|P3|P4|Mendeley|Panasonic|CALCE|calce)\b"
        for entry in self.manifest["figures"]:
            text = PdfReader(OUTPUT/entry["file"]).pages[0].extract_text()
            self.assertIsNone(re.search(forbidden, text), entry["file"])
            self.assertNotIn("Floor-to-local-oracle recovery", text)
        expected = {
            "fig03_lfp_hysteresis": ("Replica LFP",),
            "fig04_static_oracle": ("Reference map", "Target oracle", "Replica LFP"),
            "fig05_same_temperature_transfer": ("Reference map", "A123", "Replica LFP", "NCA", "NMC"),
            "fig07_temperature_interpolation": ("Linear", "Arrhenius", "Gaussian process", "Nearest temperature"),
            "fig08_gp_coverage_diagnostic": ("Gaussian-process band coverage", "Nominal coverage: 95%"),
        }
        for stem, labels in expected.items():
            text = " ".join(PdfReader(OUTPUT/(stem+".pdf")).pages[0].extract_text().split())
            for label in labels:
                self.assertIn(label, text)

    def test_presentation_archive_and_preserved_sources(self):
        for relative in self.manifest["source_sha256"]:
            verify_public(ROOT/relative)

    def test_ten_seed_windowed_ratios(self):
        report = figures_ieee.lstm_ecm_ratios(json.loads((ROOT/"results/unified_10seeds/summary.json").read_text()))
        saved = json.loads((ROOT/report["source"]).read_text())
        self.assertEqual(report["seeds"], list(range(42, 52)))
        self.assertEqual(report["count"], 180)
        self.assertEqual(len(report["conditions"]), 18)
        values = []
        expected_conditions = {
            (cell, c) for dataset, cell in (("panasonic", "NCA"), ("calce", "NMC"))
            for c in saved["datasets"][dataset]["conditions"]
            if dataset != "calce" or c not in {"US06_25C", "BJDST_25C"}
        }
        self.assertEqual({(r["cell"], r["condition"]) for r in report["conditions"]}, expected_conditions)
        for row in report["conditions"]:
            data = saved["datasets"][{"NCA": "panasonic", "NMC": "calce"}[row["cell"]]]["runs"]
            c = row["condition"]
            baseline = data["ecm_2rc_nls"]["deterministic"]["metrics"][c]["windowed"]["mae"]
            ratios = [data["lstm"][str(seed)]["metrics"][c]["windowed"]["mae"]/baseline for seed in range(42, 52)]
            np.testing.assert_allclose(row["ratios"], ratios, rtol=0, atol=1e-9)
            self.assertAlmostEqual(row["median"], np.median(ratios), delta=1e-9)
            values.extend(ratios)
        for key, expected in (("minimum", min(values)), ("maximum", max(values)), ("median", np.median(values))):
            self.assertAlmostEqual(report[key], expected, delta=1e-9)


class PlottedValuesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global OUTPUT
        cls.temporary, OUTPUT = render_public()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.after = capture_figures(figures_ieee)

    @classmethod
    def tearDownClass(cls):
        for fig in cls.after.values():
            plt.close(fig)

    def test_existing_plotted_values_and_encodings_unchanged(self):
        self.assertEqual(len(self.after), 8)
        self.assertEqual(set(self.after), {entry["file"][:-4] for entry in
                         json.loads((OUTPUT/"manifest.json").read_text())["figures"]})

    def test_nearest_temperature_exact_values_and_off_scale_annotations(self):
        fig = self.after["fig07_temperature_interpolation"]
        ax = fig.axes[0]
        with (ROOT/"results/a123_temperature_interp/table.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        conditions = ("10C_FUDS", "10C_US06", "40C_FUDS", "40C_US06")
        expected = [100*float(next(r["recovered_fraction"] for r in rows
                                  if r["method"] == "P1"
                                  and f'{r["temperature_C"]}C_{r["profile"]}' == c)) for c in conditions]
        np.testing.assert_allclose(np.array(expected)/100, [-1.4470, -1.1105, .9352, .9340],
                                   rtol=0, atol=.00005)
        nearest = next(line for line in ax.lines if line.get_label() == "Nearest temperature")
        np.testing.assert_allclose(nearest.get_xdata(orig=False), range(4), rtol=0, atol=1e-9)
        np.testing.assert_allclose(nearest.get_ydata(orig=False), expected, rtol=0, atol=1e-9)
        self.assertTrue(nearest.get_clip_on())
        self.assertEqual(nearest.get_marker(), "D")
        np.testing.assert_allclose(ax.get_ylim(), [-20, 155], rtol=0, atol=1e-9)
        boundary = [line for line in ax.lines if line != nearest and line.get_marker() == "D"]
        self.assertEqual(len(boundary), 2)
        for position, line in enumerate(boundary):
            np.testing.assert_allclose(line.get_xdata(orig=False), [position], rtol=0, atol=1e-9)
            np.testing.assert_allclose(line.get_ydata(orig=False), [-20], rtol=0, atol=1e-9)
        labels = [text for text in ax.texts if text.get_text()]
        arrows = [text for text in ax.texts if not text.get_text() and text.arrow_patch is not None]
        self.assertEqual([t.get_text() for t in labels], ["−145%", "−111%"])
        self.assertEqual(len(arrows), 2)
        for position, arrow in enumerate(arrows):
            self.assertEqual(arrow.xy, (position, -20))
            self.assertGreater(arrow.get_position()[1], arrow.xy[1])
        order = ["Linear", "Arrhenius", "Gaussian process", "Nearest temperature", "Criterion: 50%"]
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], order)
        for method, label in zip(("P2", "P3", "P4"), order[:3]):
            actual = next(line for line in ax.lines if line.get_label() == label)
            expected_values = [100*float(next(r["recovered_fraction"] for r in rows
                                             if r["method"] == method
                                             and f'{r["temperature_C"]}C_{r["profile"]}' == c))
                               for c in conditions]
            np.testing.assert_allclose(actual.get_ydata(orig=False), expected_values, rtol=0, atol=1e-9)
        entry = next(e for e in json.loads((OUTPUT/"manifest.json").read_text())["figures"]
                     if e["file"] == "fig07_temperature_interpolation.pdf")
        np.testing.assert_allclose(entry["nearest_temperature"]["values_percent"], expected, rtol=0, atol=1e-9)
        for position, record in enumerate(entry["nearest_temperature"]["off_scale"]):
            self.assertAlmostEqual(record["value_percent"], expected[position], delta=1e-9)
            self.assertEqual(record["annotation"], labels[position].get_text())
        self.assertEqual(entry["width_in"], 3.5)
        self.assertGreaterEqual(entry["minimum_font_size_pt"], 8)
        self.assertEqual(entry["raster_image_count"], 0)
        pdf_text = PdfReader(OUTPUT/entry["file"]).pages[0].extract_text()
        for label in ("Nearest temperature", "−145%", "−111%"):
            self.assertIn(label, pdf_text)

    def test_four_dynamic_panels_match_saved_records(self):
        fig = self.after["fig01_dynamic_corrections"]
        self.assertEqual(len(fig.axes), 4)
        top = json.loads((ROOT/"results/unified_10seeds/statistics.json").read_text())
        with (ROOT/"results/reviewer_round2/D/tost_pvalues.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary = json.loads((ROOT/"results/unified_10seeds/summary.json").read_text())
        bottom = {"records": figures_ieee.paired_direct_records(summary, rows)}
        counts = {"output_residual": 0, "ecm_ude": 0}
        filled = dict.fromkeys(counts, 0)
        for column, dataset in enumerate(("panasonic", "calce")):
            conditions = list(dict.fromkeys(r["condition"] for r in top["equivalence"]
                              if r["dataset"] == dataset and r["mode"] == "continuous"
                              and not r["excluded_from_main_claims"]))
            self.assertEqual(len(conditions), 8 if column == 0 else 10)
            for row in range(2):
                ax = fig.axes[row*2+column]
                models = ("physics_gradient", "output_residual", "ecm_ude") if row == 0 else ("output_residual", "ecm_ude")
                bars = [c for c in ax.containers if hasattr(c, "has_xerr")]
                self.assertEqual(len(bars), len(models)*len(conditions))
                self.assertEqual([t.get_text() for t in ax.get_yticklabels()],
                                 [condition_label(c) for c in conditions])
                self.assertIn(f"({'abcd'[row*2+column]})", [t.get_text() for t in ax.texts])
                self.assertEqual(ax.get_xlabel(), f"MAE difference from {'ECM-2RC' if row == 0 else 'physics-gradient'} (mV)")
                self.assertEqual(len(ax.patches), 1)
                span = ax.patches[0]
                vertices = span.get_patch_transform().transform(span.get_path().vertices)
                np.testing.assert_allclose([vertices[:, 0].min(), vertices[:, 0].max()],
                                           [-1, 1], rtol=0, atol=1e-9)
                np.testing.assert_allclose(ax.lines[-1].get_xdata(orig=False), [0, 0], rtol=0, atol=1e-9)
                for model_index, model in enumerate(models):
                    style = ("physics_gradient", "output_residual", "ecm_ude").index(model)
                    for position, condition in enumerate(conditions):
                        pool = top["equivalence"] if row == 0 else bottom["records"]
                        matches = [r for r in pool if r["dataset"] == dataset and r["condition"] == condition
                                   and r["model"] == model and (r["mode"] == "continuous" if row == 0
                                                               else r["reference"] == "physics_gradient")]
                        self.assertEqual(len(matches), 1)
                        record = matches[0]
                        self.assertEqual(len(record["differences_mV"]), 10)
                        center = record["difference_median_mV" if row == 0 else "median_difference_mV"]
                        container = bars[model_index*len(conditions)+position]
                        marker, caps, collections = container.lines
                        np.testing.assert_allclose(marker.get_xdata(orig=False), [center], rtol=0, atol=1e-9)
                        offset = (style-1)*.20 if row == 0 else (style-1.5)*.24
                        np.testing.assert_allclose(marker.get_ydata(orig=False), [position+offset], rtol=0, atol=1e-9)
                        low, high = min(record["differences_mV"]), max(record["differences_mV"])
                        np.testing.assert_allclose(collections[0].get_segments()[0],
                                                   [[low, position+offset], [high, position+offset]], rtol=0, atol=1e-9)
                        for cap, expected in zip(caps, (low, high)):
                            np.testing.assert_allclose(cap.get_xdata(orig=False), [expected], rtol=0, atol=1e-9)
                        self.assertEqual(marker.get_marker(), MARKERS[style])
                        self.assertEqual(marker.get_color(), COLORS[style])
                        self.assertEqual(marker.get_markerfacecolor(), COLORS[style] if record["equivalent"] else "white")
                        if row == 1:
                            counts[model] += 1
                            filled[model] += int(record["equivalent"])
        self.assertEqual(counts, {"output_residual": 18, "ecm_ude": 18})
        self.assertEqual(filled, counts)
        np.testing.assert_allclose(fig.axes[2].get_xlim(), fig.axes[3].get_xlim(), rtol=0, atol=1e-9)
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                         ["Physics gradient", "Output residual", "ECM-UDE"])
        manifest = json.loads((OUTPUT/"manifest.json").read_text())
        entry = next(e for e in manifest["figures"] if e["file"] == "fig01_dynamic_corrections.pdf")
        self.assertEqual(entry["width_in"], 7.16)
        self.assertGreaterEqual(entry["minimum_font_size_pt"], 8)
        self.assertEqual(entry["raster_image_count"], 0)
        self.assertEqual(len(entry["panels"]), 4)
        for ax, panel in zip(fig.axes, entry["panels"]):
            np.testing.assert_allclose(ax.get_xlim(), panel["xlim_mV"], rtol=0, atol=1e-9)
        self.assertIn("results/reviewer_round2/D/tost_pvalues.csv", entry["sources"])
        text = PdfReader(OUTPUT/entry["file"]).pages[0].extract_text()
        for label in ("(a)", "(b)", "(c)", "(d)", "MAE difference from physics-gradient (mV)"):
            self.assertIn(label, text)

    def test_wide_series_only_nmc_and_exact_saved_mae(self):
        fig = self.after["fig05_same_temperature_transfer"]
        saved = json.loads((ROOT/"results/param_ablation/calce/summary.json").read_text())
        for index, ax in enumerate(fig.axes):
            matches = [c for c in ax.containers if c.get_label() == LABELS["V6-wide"]]
            self.assertEqual(len(matches), int(index == 3))
        ax = fig.axes[3]
        self.assertEqual(ax.get_xlabel(), "NMC (25°C)")
        wide = ax.containers[2]
        profiles = ("DST_25C", "US06_25C", "BJDST_25C")
        expected = [next(r["mae_mV"] for r in saved["metrics"] if r["target"] == p
                         and r["variant"] == "V6-wide" and r["scope"] == "global") for p in profiles]
        np.testing.assert_allclose(wide.datavalues, expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(expected, [9.942, 10.788, 11.516], rtol=0, atol=.0005)
        baseline = [next(r["mae_mV"] for r in saved["metrics"] if r["target"] == p
                         and r["variant"] == "V1" and r["scope"] == "global") for p in profiles]
        np.testing.assert_allclose(ax.containers[0].datavalues, baseline, rtol=0, atol=1e-9)
        self.assertEqual([c.patches[0].get_hatch() for c in ax.containers], ["///", "xxx", "..."])

    def test_no_internal_titles_and_no_text_overlap(self):
        for stem, fig in self.after.items():
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                for loc in ("left", "center", "right"):
                    self.assertFalse(ax.get_title(loc=loc), stem)
                self.assertFalse(fig._suptitle, stem)
            for legend in fig.legends:
                box = legend.get_window_extent(renderer)
                for ax in fig.axes:
                    self.assertFalse(box.overlaps(ax.get_window_extent(renderer)), stem)
                texts = [t.get_window_extent(renderer) for t in legend.get_texts()]
                for i, a in enumerate(texts):
                    for b in texts[i+1:]:
                        self.assertFalse(a.overlaps(b), stem)


if __name__ == "__main__":
    unittest.main()
