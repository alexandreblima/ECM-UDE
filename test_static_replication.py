"""Exact decisions, source-only fitting and independent persisted trajectories."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.chebyshev import chebval

from data_lfp import sha256
from exp_lfp_static import bounds_starts, predict
from exp_static_replication import (MODELS, OUTPUT, PROTOCOL_PATH, THERMAL, CheckpointOCV,
                                    decide, fit_source, previous_snapshot, protocol,
                                    calce_ocv, unified_comparisons, verify_registration)
from lfp_ocv import OCVCurves
from test_lfp_static import synthetic


class ReplicationTests(unittest.TestCase):
    def test_registration_and_exact_all_target_criterion(self):
        verify_registration()
        maes = {c: {"V1": 20., "V5": 15., "V6": 16., "V7": 17.}
                for c in ("a", "b", "c")}
        self.assertEqual(decide(maes, list(maes))["verdict"], "POSITIVA")
        maes["c"]["V5"] = 15.0000001
        self.assertEqual(decide(maes, list(maes))["verdict"], "NEGATIVA")
        maes["c"]["V7"] = 14.
        self.assertEqual(decide(maes, list(maes))["verdict"], "POSITIVA")
        self.assertEqual(decide(maes, list(maes))["profiles"]["c"]["best_model"], "V7")
        self.assertIsNone(decide(maes, list(maes))["profiles"]["a"]["ceiling_recovered_fractions"]["V5"])
        with self.assertRaises(ValueError):
            decide(maes, [])
        maes["a"]["V1"] = 0.
        with self.assertRaises(ValueError):
            decide(maes, ["a"])

    def test_exact_checkpoint_ocv_not_interpolated(self):
        coeff = [3.7, .4, -.03, .02, .01, -.05]
        curves = CheckpointOCV(coeff)
        z = np.array([-.5, .001234567, .51234567, 1.3])
        np.testing.assert_array_equal(curves.evaluate(z)[2], chebval(2*np.clip(z, 0, 1)-1, coeff))

    def test_transfer_ignores_target_voltage_and_uses_causal_counter(self):
        cell, cycle, curves = synthetic()
        cycle.net_Ah = np.linspace(0, 1.5, len(cycle.t))
        cycle.soc = .82-cycle.net_Ah/cell.qnom_Ah
        cycle.z0 = .82
        for model in MODELS:
            _, _, starts = bounds_starts(cell, cycle, curves, model)
            before, soc = predict(cycle, cell, curves, starts[1], model)
            expected_z0 = .95 if model in ("V6", "V7") else .82
            np.testing.assert_allclose(soc, expected_z0-cycle.net_Ah/cell.qnom_Ah)
            original = cycle.voltage.copy()
            cycle.voltage[:] = 100.
            after, _ = predict(cycle, cell, curves, starts[1], model)
            np.testing.assert_array_equal(before, after)
            cycle.voltage[:] = original

    def test_fitting_reads_only_registered_source(self):
        cell, cycle, curves = synthetic()
        cell.name = "panasonic"
        cell.cycles = {"UDDS_25C": cycle, "target": replace(
            cycle, name="target", voltage=cycle.voltage+100)}
        seen = []

        def fake_predict(source, *args):
            seen.append(source)
            return source.voltage.copy(), source.soc.copy()

        with patch("exp_static_replication.predict", side_effect=fake_predict):
            info = fit_source(cell, curves, "V1")
        self.assertTrue(seen)
        self.assertTrue(all(c is cell.cycles["UDDS_25C"] for c in seen))
        self.assertEqual(info["identified_on"], "UDDS_25C")
        self.assertFalse(info["target_refit"])

    def test_duplicate_low_current_timestamps_preserve_zero_duration(self):
        t = np.arange(0., 72001., 100.)
        z = 1-t/72000
        a = np.column_stack((t, np.full(len(t), 3.), (3.2+.8*z)*1000, np.full(len(t), -100.)))
        rows = np.insert(a, 20, a[20], axis=0).tolist()
        wb = SimpleNamespace(active=SimpleNamespace(values=rows), close=lambda: None)
        reference = {"datasets": {"calce": {"ocv_calibration": {
            "path": "../datasets/synthetic.xlsx", "sha256": "digest"}}}}
        with patch("exp_static_replication.sha256", return_value="digest"), patch(
                "exp_static_replication.openpyxl.load_workbook", return_value=wb):
            curves, meta = calce_ocv(reference)
        self.assertEqual(meta["duplicate_timestamp_rows"], 1)
        self.assertAlmostEqual(meta["measured_discharge_Ah"], 2.)
        np.testing.assert_allclose(curves.evaluate(np.array([.2, .5, .8]))[2],
                                   [3.36, 3.60, 3.84], atol=.003)

    def test_missing_neural_comparison_is_an_error(self):
        with self.assertRaises(ValueError):
            unified_comparisons("panasonic", "US06_25C", [])


class PersistedTests(unittest.TestCase):
    def test_saved_metrics_fit_constraints_and_decisions(self):
        from public_integrity import verify
        for name, cfg in protocol()["datasets"].items():
            path = OUTPUT/name/"summary.json"
            verify(path)
            summary = json.loads(path.read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["manifest"]["protocol_sha256"], sha256(PROTOCOL_PATH))
            for model, fit in summary["fits"].items():
                self.assertEqual(fit["identified_on"], cfg["source"])
                self.assertFalse(fit["target_refit"])
                self.assertLessEqual(fit["parameters"]["tau2_s"], 3600.000001)
            for profile, values in summary["global_mae_mV"].items():
                for model, value in values.items():
                    row, = [r for r in summary["metrics"] if r["profile"] == profile
                            and r["model"] == model and r["scope"] == "global"]
                    self.assertEqual(row["mae_mV"], value)
            self.assertEqual(decide(summary["global_mae_mV"], cfg["primary_targets"]), summary["decision"])
            self.assertEqual(len(summary["metrics"]), len(summary["cycles"])*32)


if __name__ == "__main__":
    unittest.main()
