"""Frozen thresholds, causal transfer and persisted static-experiment invariants."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial.chebyshev import chebval

from data_lfp import CellData, _cycle, integrate_Ah, sha256
from exp_lfp_gonogo import metrics, rc_voltage
from exp_lfp_static import (OUTPUT, PROTOCOL_PATH, bounds_starts, decide, parameter_names,
                            predict, snapshot_previous, unpack)
from lfp_ocv import OCVCurves


def synthetic() -> tuple:
    t = np.arange(1000, dtype=float)
    current = np.sin(t/30) + 0.5
    raw = np.column_stack((t, current, np.full(len(t), 3.3), np.full(len(t), 25)))
    cycle = _cycle("DST", raw, 2.5, {"z0": 1, "h0": 1})
    z = np.array([0., 0.5, 1.])
    curves = OCVCurves(z, np.array([3., 3.3, 3.6]), z, np.array([2.9, 3.2, 3.5]))
    cell = CellData("synthetic", 2.5, z, curves.charge_V, z, curves.discharge_V,
                    {"DST": cycle}, {})
    return cell, cycle, curves


class StaticMathTests(unittest.TestCase):
    def test_exact_25_percent_threshold(self):
        data = {profile: {"V1": 20., "V5": 15., "V6": 16., "V7": 17., "V4": 10.}
                for profile in ("FUDS", "US06")}
        self.assertTrue(decide(data)["positive"])
        self.assertEqual(decide(data)["attribution"], "OCV")
        self.assertEqual(decide(data)["profiles"]["FUDS"]["ceiling_recovered_fractions"]["V5"], 0.5)
        data["US06"]["V5"] = 15.00001
        self.assertFalse(decide(data)["positive"])

    def test_mixed_attribution_and_unclipped_fractions(self):
        values = {"FUDS": {"V1": 20., "V5": 10., "V6": 15., "V7": 9., "V4": 11.},
                  "US06": {"V1": 20., "V5": 25., "V6": 10., "V7": 12., "V4": 9.}}
        result = decide(values)
        self.assertTrue(result["positive"])
        self.assertEqual(result["attribution"], "MIXED/INCONCLUSIVE")
        self.assertGreater(result["profiles"]["FUDS"]["ceiling_recovered_fractions"]["V7"], 1)
        self.assertLess(result["profiles"]["US06"]["ceiling_recovered_fractions"]["V5"], 0)

    def test_nonpositive_ceiling_explicitly_unavailable(self):
        values = {name: {"V1": 10., "V5": 7., "V6": 8., "V7": 8., "V4": 11.}
                  for name in ("FUDS", "US06")}
        with self.assertLogs("exp_lfp_static", level="WARNING"):
            result = decide(values)
        self.assertIsNone(result["profiles"]["FUDS"]["ceiling_recovered_fractions"]["V5"])
        self.assertEqual(result["attribution"], "UNAVAILABLE")

    def test_parameter_bounds_and_transfer(self):
        cell, cycle, curves = synthetic()
        for model in ("V1", "V5", "V6", "V7"):
            lo, hi, starts = bounds_starts(cell, cycle, curves, model)
            self.assertAlmostEqual(np.exp(hi[3]), 3600.)
            self.assertAlmostEqual(np.exp(hi[4]), 3600.)
            self.assertEqual(len(lo), len(parameter_names(model)))
            for start in starts:
                self.assertTrue(np.all(start >= lo))
                self.assertTrue(np.all(start <= hi))
            prediction, z = predict(cycle, cell, curves, starts[1], model)
            p = unpack(starts[1], model, cell, cycle)
            np.testing.assert_allclose(z, p["z0"] - cycle.net_Ah/p["Qnom_Ah"])
            self.assertTrue(np.isfinite(prediction).all())
            if model in ("V6", "V7"):
                self.assertAlmostEqual(z[0], 0.95)

    def test_sorted_RC_labels_preserve_prediction(self):
        cell, cycle, curves = synthetic()
        theta = np.log([0.02, 0.03, 0.01, 100., 2.])
        p = unpack(theta, "V1", cell, cycle)
        self.assertLess(p["tau1_s"], p["tau2_s"])
        voltage = predict(cycle, cell, curves, theta, "V1")[0]
        mean = curves.evaluate(cycle.soc)[2]
        expected = mean - 0.02*cycle.current - rc_voltage(cycle.current, 0.03, 100.) - rc_voltage(cycle.current, 0.01, 2.)
        np.testing.assert_allclose(voltage, expected)


class PersistedTests(unittest.TestCase):
    def test_saved_metrics_and_decisions(self):
        from public_integrity import verify
        verdicts = {}
        for name in ("cell_A_a123", "cell_B_mendeley"):
            path = OUTPUT/name/"summary.json"
            verify(path)
            summary = json.loads(path.read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["manifest"]["protocol_sha256"], sha256(PROTOCOL_PATH))
            self.assertEqual(len(summary["metrics"]), 120)
            for profile, values in summary["global_mae_mV"].items():
                for model, value in values.items():
                    row, = [r for r in summary["metrics"] if r["profile"] == profile
                            and r["model"] == model and r["scope"] == "global"]
                    self.assertEqual(row["mae_mV"], value)
            self.assertEqual(decide(summary["global_mae_mV"]), summary["decision"])
            verdicts[name] = summary["decision"]["verdict"]
        self.assertEqual(json.loads((OUTPUT/"global_decision.json").read_text())["global"], verdicts["cell_A_a123"])


if __name__ == "__main__":
    unittest.main()
