"""Validate delivered IEEE PDF dimensions, fonts, provenance and ledger scope."""
from __future__ import annotations

import json
import re
import unittest

from pypdf import PdfReader

from figures_ieee import COLORS, MARKERS, OUTPUT, ROOT, STYLES, condition_label, digest, inspect_pdf


class IEEEFiguresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        index = (OUTPUT/"figures_index.md").read_text()
        for target in re.findall(r"\]\(([^)]+)\)", index):
            self.assertTrue((OUTPUT/target).is_file(), target)
        for entry in self.manifest["figures"]:
            self.assertIn(entry["file"], index)
            self.assertTrue(entry["sources"])
            self.assertTrue(all(s in self.manifest["source_sha256"] for s in entry["sources"]))

    def test_previous_results_manuscript_and_report_unchanged(self):
        audit = json.loads((ROOT/"results/paper_v2_audit/preservation.json").read_text())
        expected = {**audit["sha256"], **audit["new_artifact_sha256"],
                    str(ROOT/"report_paper_v2.md"): audit["report_sha256"]}
        for source, sha in expected.items():
            self.assertEqual(digest(ROOT/source), sha, source)

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


if __name__ == "__main__":
    unittest.main()
