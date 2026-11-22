import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import figures_ieee as figures
import reviewer_saved_analysis as saved_analysis


ROOT = Path(__file__).resolve().parent


class PublicFigureTests(unittest.TestCase):
    def test_paired_comparisons_match_independently_saved_tost(self):
        summary = json.loads((ROOT / "results/unified_10seeds/summary.json").read_text())
        with (ROOT / "results/reviewer_round2/D/tost_pvalues.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        comparisons = figures.paired_direct_records(summary, rows)
        for item in comparisons:
            matching = [r for r in rows if r["cell"] == item["dataset"]
                        and r["condition"] == item["condition"] and r["model"] == item["model"]
                        and r["reference"] == item["reference"]]
            self.assertEqual(len(matching), 1)
            self.assertAlmostEqual(item["median_difference_mV"],
                                   float(matching[0]["median_difference_mV"]), places=12)
            self.assertEqual(item["equivalent"], float(matching[0]["p_TOST_Holm"]) < .05)

    def test_missing_comparison_is_an_error(self):
        summary = json.loads((ROOT / "results/unified_10seeds/summary.json").read_text())
        with (ROOT / "results/reviewer_round2/D/tost_pvalues.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = [r for r in rows if not (r["cell"] == "panasonic" and r["condition"] == "UDDS_25C"
                                        and r["model"] == "ecm_ude" and r["reference"] == "physics_gradient")]
        with self.assertRaisesRegex(ValueError, "36 saved paired comparisons"):
            figures.paired_direct_records(summary, rows)

    def test_rebuilt_paired_audit_matches_saved_tests(self):
        summary = json.loads((ROOT / "results/unified_10seeds/summary.json").read_text())
        with (ROOT / "results/reviewer_round2/D/tost_pvalues.csv").open(newline="") as stream:
            expected = {(r["cell"], r["condition"], r["model"]): r
                        for r in csv.DictReader(stream)
                        if r["reference"] == "physics_gradient" and r["holm_applicable"] == "True"}
        with tempfile.TemporaryDirectory() as output, patch.object(saved_analysis, "TOST", Path(output)):
            records = saved_analysis.paired_tost(summary)
        self.assertEqual(len(records), 36)
        for record in records:
            row = expected[record["dataset"], record["condition"], record["model"]]
            self.assertAlmostEqual(record["lower"]["p_holm"], float(row["p_lower_Holm"]))
            self.assertAlmostEqual(record["upper"]["p_holm"], float(row["p_upper_Holm"]))

    def test_audit_cannot_overwrite_saved_results(self):
        with patch("sys.argv", ["reviewer_saved_analysis.py", "--output", str(ROOT / "results")]):
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                saved_analysis.main()
