"""Replay frozen E1 inference, check manuscript numbers and prove preservation."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import analyze_e1_error_structure as A


EXPECTED_CLAIMS = {
    "peak_min": (436, 0), "peak_max": (1032, 0), "late_peaks": (17, 0),
    "late_voltage_min": (2.41, 2), "late_voltage_max": (2.78, 2), "early_peak_percent": (2, 0),
    "trimmed_calce_min": (39, 0), "trimmed_calce_max": (66, 0),
    "trimmed_panasonic_min": (231, 0), "trimmed_panasonic_max": (436, 0),
    "rmse_neural": (.42, 2), "rmse_nca": (3.9, 1), "rmse_nmc": (.3, 1),
    "p99_neural": (.73, 2), "p99_nca": (12.6, 1), "p99_nmc": (2.2, 1),
    "max_error_neural": (.89, 2), "max_error_nca": (42.6, 1), "max_error_nmc": (2.2, 1),
    "calce_DST_25C_mid": (2.3, 1), "calce_DST_25C_low": (15.1, 1),
    "panasonic_HWFET_25C_mid": (20.5, 1), "panasonic_HWFET_25C_low": (133.5, 1),
    "soc_neural": (.62, 2), "soc_nca": (4.3, 1), "soc_nmc": (.5, 1),
    "ocv_panasonic_median": (.13, 2), "ocv_panasonic_maximum": (.18, 2),
    "ocv_calce_median": (.19, 2), "ocv_calce_maximum": (.35, 2),
    "ocv_all_min": (.1, 1), "ocv_all_max": (.4, 1),
}


class AnalysisGuardTests(unittest.TestCase):
    def test_mae_gate_rejects_mismatch_and_nonfinite_data(self):
        truth = np.zeros(3)
        mask = np.array([False, True, True])
        errors, check = A.validate_mae(np.array([9., .001, -.003]), truth, mask, 2., "synthetic")
        np.testing.assert_array_equal(errors, [1., -3.])
        self.assertEqual(check["reproduced_mae_mV"], 2.)
        with self.assertRaisesRegex(ValueError, "MAE mismatch"):
            A.validate_mae(np.full(3, .001), truth, mask, 0., "synthetic")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            A.validate_mae(np.full(3, np.nan), truth, mask, 0., "synthetic")
        with self.assertRaisesRegex(ValueError, "shape"):
            A.validate_mae(np.zeros(2), truth, mask, 0., "synthetic")
        with self.assertRaisesRegex(ValueError, "Empty"):
            A.validate_mae(truth, truth, np.zeros(3, dtype=bool), 0., "synthetic")

    def test_global_mae_gate_precedes_statistics_and_output(self):
        def mismatch(*args):
            A.validate_mae(np.ones(2), np.zeros(2), np.ones(2, dtype=bool), 0., "injected")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"must_not_be_created"
            with patch.object(A, "collect_validated_errors", side_effect=mismatch), \
                    patch.object(A, "summarize_records") as statistics, patch.object(A, "ocv_slope") as slopes:
                with self.assertRaisesRegex(ValueError, "MAE mismatch"):
                    A.reproduce(output)
                statistics.assert_not_called()
                slopes.assert_not_called()
                self.assertFalse(output.exists())

    def test_no_overwriting_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output/"existing.txt"
            sentinel.write_text("untouched")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                A.reproduce(output)
            self.assertEqual(sentinel.read_text(), "untouched")

    def test_checkpoint_routing_and_soc_boundary_partition(self):
        for seed in range(42, 52):
            group = "unified" if seed <= 46 else "unified_10seeds"
            self.assertEqual(A.checkpoint_directory("panasonic", seed),
                             A.ROOT/"results"/group/"checkpoints/panasonic")
        masks = A.band_masks(np.array([0., .2, np.nextafter(.8, 0), .8, 1.]))
        np.testing.assert_array_equal(masks["low"], [True, False, False, False, False])
        np.testing.assert_array_equal(masks["mid"], [False, True, True, False, False])
        np.testing.assert_array_equal(masks["high"], [False, False, False, True, True])
        np.testing.assert_array_equal(sum(masks.values()), np.ones(5))
        with self.assertRaises(ValueError):
            A.band_masks(np.array([1.01]))


@unittest.skipUnless(A.U.P.DATA_ROOT.is_dir(), "Raw datasets unavailable: set ECM_UDE_DATA")
class FrozenCheckpointReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = A.checkpoint_snapshot()
        cls.saved = json.loads((A.OUTPUT/"analysis.json").read_text())
        cls.saved_validation = json.loads((A.OUTPUT/"validation.json").read_text())
        cls.saved_preservation = json.loads((A.OUTPUT/"preservation.json").read_text())
        cls.summary = json.loads(A.SUMMARY.read_text())
        cls.temporary = tempfile.TemporaryDirectory(prefix="test-e1-replay-")
        cls.addClassCleanup(cls.temporary.cleanup)
        output = Path(cls.temporary.name)/"fresh"
        with patch.object(A.U, "train", side_effect=AssertionError("Training is forbidden")), \
                patch.object(A.U, "fit_nls", side_effect=AssertionError("Fitting is forbidden")), \
                patch.object(A.U, "least_squares", side_effect=AssertionError("Reidentification is forbidden")), \
                contextlib.redirect_stdout(io.StringIO()):
            cls.actual = A.reproduce(output)
        cls.validation = json.loads((output/"validation.json").read_text())
        cls.preservation = json.loads((output/"preservation.json").read_text())
        cls.after = A.checkpoint_snapshot()

    def test_all_continuous_maes_reproduced_before_statistics(self):
        validation = self.validation
        self.assertEqual(validation["count"], 576)
        self.assertEqual(len(validation["records"]), 576)
        self.assertTrue(validation["all_passed"])
        self.assertEqual(validation["tolerance_mV"], 1e-6)
        self.assertLessEqual(validation["maximum_absolute_difference_mV"], 1e-6)
        expected = {
            (dataset, condition, model, None if model.endswith("_nls") else seed)
            for dataset, data in self.summary["datasets"].items()
            for condition in data["conditions"] if dataset != "calce" or condition not in A.EXCLUDED
            for model in ("ecm_1rc_nls", "ecm_2rc_nls", *A.TRAINED)
            for seed in ([None] if model.endswith("_nls") else A.SEEDS)
        }
        self.assertEqual({(r["dataset"], r["condition"], r["model"], r["seed"])
                          for r in validation["records"]}, expected)
        for report in (validation, self.saved_validation):
            for row in report["records"]:
                with self.subTest(dataset=row["dataset"], condition=row["condition"],
                                  model=row["model"], seed=row["seed"]):
                    seed = "deterministic" if row["seed"] is None else str(row["seed"])
                    expected_mae = self.summary["datasets"][row["dataset"]]["runs"][row["model"]][seed][
                        "metrics"][row["condition"]]["continuous"]["mae"]
                    self.assertEqual(row["validation"]["saved_mae_mV"], expected_mae)
                    self.assertAlmostEqual(row["validation"]["reproduced_mae_mV"], expected_mae, delta=1e-6)
                    self.assertTrue(row["validation"]["passed"])

    def test_all_manuscript_numbers_and_conservative_bound_reported(self):
        claims = {r["id"]: r for r in self.actual["claims"]}
        self.assertEqual(set(claims), set(EXPECTED_CLAIMS))
        for identifier, (article, places) in EXPECTED_CLAIMS.items():
            row = claims[identifier]
            with self.subTest(claim=identifier):
                self.assertEqual(row["article"], article)
                self.assertEqual(row["decimal_places"], places)
                self.assertEqual(row["rounded_reproduced"], round(row["reproduced"], places))
                self.assertTrue(row["passed"])
                if identifier == "max_error_nmc":
                    self.assertEqual(row["rounded_reproduced"], 2.1)
                    self.assertFalse(row["same_rounded_number"])
                    self.assertLess(row["reproduced"], article)
                else:
                    self.assertEqual(round(row["reproduced"], places), article)
                    self.assertTrue(row["same_rounded_number"])
        self.assertEqual([r["id"] for r in self.actual["claims"] if not r["same_rounded_number"]],
                         ["max_error_nmc"])
        # A rounded upper bound is not an unrounded mathematical threshold.
        self.assertGreater(claims["soc_nmc"]["reproduced"], .5)
        self.assertEqual(round(claims["soc_nmc"]["reproduced"], 1), .5)

    def test_tail_metrics_and_seed_medians_independently_match_summary(self):
        for row in self.actual["records"]:
            seed = "deterministic" if row["seed"] is None else str(row["seed"])
            expected = self.summary["datasets"][row["dataset"]]["runs"][row["model"]][seed][
                "metrics"][row["condition"]]["continuous"]
            for metric, actual in row["metrics_mV"].items():
                self.assertAlmostEqual(actual, expected[metric], delta=1e-6)
        for row in self.actual["condition_medians"]:
            for model in A.TRAINED:
                for metric in ("mae", *A.TAIL_METRICS):
                    data = self.summary["datasets"][row["dataset"]]["runs"][model]
                    expected = np.median([data[str(seed)]["metrics"][row["condition"]]["continuous"][metric]
                                          for seed in A.SEEDS])
                    self.assertAlmostEqual(row["trained_medians"][model]["metrics_mV"][metric],
                                           expected, delta=1e-6)

    def test_soc_counts_empty_bands_and_ten_seed_medians(self):
        conditions = {(r["dataset"], r["condition"]): r for r in self.actual["conditions"]}
        for row in self.actual["records"]:
            counts = row["soc_bands"]
            self.assertEqual(sum(b["n_samples"] for b in counts.values()),
                             conditions[row["dataset"], row["condition"]]["evaluated_samples"])
            for band in counts.values():
                self.assertEqual(band["empty"], band["n_samples"] == 0)
                self.assertEqual(band["mae_mV"] is None, band["empty"])
        for row in self.actual["condition_medians"]:
            for model in A.TRAINED:
                selected = [r for r in self.actual["records"] if r["dataset"] == row["dataset"]
                            and r["condition"] == row["condition"] and r["model"] == model]
                self.assertEqual(sorted(r["seed"] for r in selected), list(A.SEEDS))
                for band in A.BANDS:
                    values = [r["soc_bands"][band]["mae_mV"] for r in selected]
                    median = row["trained_medians"][model]["soc_band_mae_mV"][band]
                    if values[0] is None:
                        self.assertTrue(all(v is None for v in values))
                        self.assertIsNone(median)
                    else:
                        self.assertAlmostEqual(median, np.median(values), delta=1e-12)
                    self.assertTrue(all(r["soc_bands"][band]["n_samples"]
                                        == row["baseline"]["soc_bands"][band]["n_samples"] for r in selected))
            if row["dataset"] == "panasonic" and row["condition"] in ("UDDS_-10C", "UDDS_-20C"):
                self.assertTrue(row["baseline"]["soc_bands"]["low"]["empty"])
        source = conditions["panasonic", "UDDS_25C"]
        self.assertTrue(source["source"])
        self.assertLess(source["evaluated_samples"], .25*source["total_samples"])
        for row in conditions.values():
            if not row["source"]:
                self.assertEqual(row["evaluated_samples"], row["total_samples"])

    def test_peak_positions_and_ocv_chain_rule(self):
        peaks = self.actual["peaks_2rc"]
        self.assertEqual(len(peaks), 18)
        early = [p for p in peaks if p["sample_fraction"] <= .89]
        self.assertEqual([(p["dataset"], p["condition"]) for p in early], [("panasonic", "UDDS_-20C")])
        for row in peaks:
            self.assertGreaterEqual(row["causal_soc"], 0)
            self.assertLessEqual(row["causal_soc"], 1)
            self.assertAlmostEqual(abs(row["predicted_voltage_V"]-row["measured_voltage_V"])*1000,
                                   row["max_error_mV"], delta=1e-9)
        for slope in self.actual["ocv_slopes"]:
            polynomial = np.polynomial.Chebyshev(slope["ocv_coeffs"], domain=[0, 1])
            soc = np.linspace(.2, .8, 10001)
            derivative = polynomial.deriv()(soc)
            self.assertTrue((derivative > 0).all())
            np.testing.assert_allclose(derivative, 2*np.polynomial.chebyshev.chebval(
                2*soc-1, slope["derivative_chebyshev_coefficients_in_x"]), rtol=0, atol=1e-12)
            conversion = .001/derivative*100
            for name, value in (("minimum", min(conversion)), ("median", np.median(conversion)),
                                ("maximum", max(conversion))):
                self.assertAlmostEqual(slope["soc_percent_per_1mV"][name], value, delta=1e-12)

    def test_persisted_artifacts_reproduce_and_checkpoint_hashes_never_change(self):
        self.assertEqual(self.before, self.after)
        for audit in (self.preservation, self.saved_preservation):
            self.assertTrue(audit["no_training_fitting_or_reidentification"])
            self.assertEqual(audit["checkpoint_sha256_before"], audit["checkpoint_sha256_after"])
            self.assertEqual(audit["checkpoint_sha256_before"], self.before)
            self.assertEqual(audit["input_sha256_before"], audit["input_sha256_after"])
        A.verify_hashes(self.preservation["input_sha256_before"])
        for relative, sha in self.preservation["raw_input_sha256"].items():
            self.assertEqual(A.digest(A.ROOT/relative), sha)
        for relative, sha in self.saved_preservation["raw_input_sha256"].items():
            path = Path(relative)
            if path.parts[0] == "datasets":
                raw = A.U.P.DATA_ROOT/"Pan18650"/path.name
            else:
                raw = A.U.P.DATA_ROOT/path.relative_to("../datasets")
            self.assertEqual(A.digest(raw), sha)
        from public_integrity import verify
        verify(A.SUMMARY)
        self.assertEqual(self.actual["conditions"], self.saved["conditions"])
        self.assertEqual(len(self.actual["records"]), 558)
        self.assertEqual(len(self.actual["records"]), len(self.saved["records"]))
        for actual, saved in zip(self.actual["records"], self.saved["records"]):
            self.assertEqual({k: actual[k] for k in ("dataset", "condition", "model", "seed")},
                             {k: saved[k] for k in ("dataset", "condition", "model", "seed")})
            for metric in ("mae", *A.TAIL_METRICS):
                self.assertAlmostEqual(actual["metrics_mV"][metric], saved["metrics_mV"][metric], delta=1e-6)
            for band in A.BANDS:
                a, b = actual["soc_bands"][band], saved["soc_bands"][band]
                self.assertEqual(a["n_samples"], b["n_samples"])
                self.assertEqual(a["empty"], b["empty"])
                if not a["empty"]:
                    self.assertAlmostEqual(a["mae_mV"], b["mae_mV"], delta=1e-6)
        for actual, saved in zip(self.actual["claims"], self.saved["claims"]):
            self.assertEqual(actual["id"], saved["id"])
            self.assertAlmostEqual(actual["reproduced"], saved["reproduced"], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
