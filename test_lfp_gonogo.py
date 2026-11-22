"""Focused mathematical and local-data validation for the LFP experiment."""
from pathlib import Path
import csv
import json
import tempfile
import unittest

import numpy as np

from data_lfp import (DATA_ROOT, OUTPUT, PROTOCOL_PATH, _cycle, integrate_Ah, load_a123,
                      load_mendeley, protocol, sha256)
from lfp_ocv import OCVCurves, build_curves, monotonic_points
from exp_lfp_gonogo import (decide, directions, hysteresis, masks, metrics,
                            rc_voltage, reversal_indices, simulate)


class LoaderTests(unittest.TestCase):
    def test_discharge_positive_coulomb_counting(self):
        t = np.arange(3601, dtype=float)
        ah = integrate_Ah(t, np.ones(len(t)))
        self.assertAlmostEqual(ah[-1], 1.0)
        raw = np.column_stack((t, np.ones(len(t)), np.full(len(t), 3.3),
                               np.full(len(t), 25.0)))
        cycle = _cycle("synthetic", raw, 2.0, {"z0": 1, "h0": 1})
        self.assertAlmostEqual(cycle.soc[-1], 0.5)

    def test_invalid_channels_and_gaps_raise(self):
        for times in ([0, 0, 1], [0, 1, 10]):
            raw = np.column_stack((times, [1, 1, 1], [3.3, 3.3, 3.3], [25, 25, 25]))
            with self.assertRaises(ValueError):
                _cycle("bad", raw, 2.5, {"z0": 1, "h0": 1})

    def test_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input"
            p.write_bytes(b"abc")
            self.assertEqual(sha256(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_frozen_decision(self):
        cfg = protocol()
        self.assertEqual(cfg["decision"]["minimum_mae_reduction"], 0.10)
        self.assertEqual(cfg["decision"]["maximum_saturation_fraction_exclusive"], 0.90)
        self.assertEqual(cfg["current_threshold_A"], 0.02)


@unittest.skipUnless(DATA_ROOT.is_dir(), "Raw datasets unavailable: set ECM_UDE_DATA")
class RealDataTests(unittest.TestCase):
    def test_real_cells(self):
        a, b = load_a123(), load_mendeley()
        self.assertAlmostEqual(a.qnom_Ah, 1.063561667212343, places=9)
        self.assertAlmostEqual(b.qnom_Ah, 2.4989015833333332, places=9)
        self.assertEqual(len(b.ocv_charge_z), 11)
        self.assertEqual(len(b.ocv_discharge_z), 11)
        self.assertAlmostEqual(b.ocv_charge_z[0], b.ocv_discharge_z[0])
        self.assertTrue(np.all(np.diff(b.ocv_charge_z) > 0))
        for cell in (a, b):
            for cycle in cell.cycles.values():
                np.testing.assert_array_equal(np.diff(cycle.t), 1)
                np.testing.assert_allclose(cycle.soc, 1 - cycle.net_Ah / cell.qnom_Ah)
                self.assertEqual(cycle.h0, 1)
                self.assertIn("initial_state_provenance", cycle.metadata)
        self.assertGreater(b.cycles["FUDS"].metadata["capacity_resets_ignored"], 100)
        for cell in (a, b):
            curves = build_curves(cell)
            chg, dis, mean, m = curves.evaluate(np.linspace(0.2, 0.8, 100))
            np.testing.assert_allclose(mean + m, chg)
            np.testing.assert_allclose(mean - m, dis)
            self.assertTrue(np.isfinite(chg).all())
            if cell.name == "cell_B_mendeley":
                self.assertEqual(len(curves.charge_z), 11)


class OCVTests(unittest.TestCase):
    def test_signed_M_and_boundary_hold(self):
        z = np.array([0.1, 0.5, 0.9])
        c = OCVCurves(z, np.array([3., 3.2, 3.3]), z, np.array([3., 3.3, 3.4]))
        values = c.evaluate(np.array([-1., 0.5, 2.]))
        self.assertAlmostEqual(values[3][1], -0.05)
        self.assertAlmostEqual(values[0][0], 3.)
        self.assertAlmostEqual(values[0][-1], 3.3)

    def test_monotonic_denoising(self):
        z, v = monotonic_points(np.linspace(0, 1, 5), np.array([3, 3.3, 3.2, 3.4, 3.5]))
        self.assertTrue(np.all(np.diff(v) >= 0))
        self.assertEqual(z[0], 0)
        self.assertEqual(z[-1], 1)


class ModelTests(unittest.TestCase):
    def test_RC_exact_step(self):
        current = np.ones(200)
        resistance, tau = 0.02, 10.0
        voltage = rc_voltage(current, resistance, tau)
        np.testing.assert_allclose(voltage, resistance * (1 - np.exp(-np.arange(200) / tau)),
                                   atol=1e-14)

    def test_ESC_exact_recurrence_and_rest(self):
        rng = np.random.default_rng(42)
        current = rng.uniform(-4, 4, 2001)
        current[100:200] = 0
        s = directions(current, 0.02)
        for gamma in (0, 1, 1000):
            actual = hysteresis(current, s, 1.0, gamma, 1.0)
            expected = np.empty(len(current))
            expected[0] = 1
            for k in range(len(current) - 1):
                decay = np.exp(-gamma * abs(current[k]) / 3600)
                expected[k + 1] = s[k] + (expected[k] - s[k]) * decay
            np.testing.assert_allclose(actual, expected, atol=2e-13)
            self.assertLessEqual(np.abs(actual).max(), 1)
            np.testing.assert_allclose(actual[100:201], actual[100], atol=1e-14)

    def test_direction_and_reversal_masks(self):
        current = np.r_[np.ones(10), np.zeros(5), -np.ones(10), np.zeros(5), np.ones(70)]
        s = directions(current, 0.02)
        self.assertTrue(np.all(s[10:15] == -1))
        np.testing.assert_array_equal(reversal_indices(current, 0.02), [15, 30])
        raw = np.column_stack((np.arange(100), current, np.full(100, 3.3), np.full(100, 25)))
        cycle = _cycle("test", raw, 2.5, {"z0": 1, "h0": 1})
        mask = masks(cycle, 0.02, 60)["after_reversal_60s"]
        self.assertEqual(mask.sum(), 75)  # [15,75) union [30,90)
        self.assertFalse(mask[14])
        self.assertFalse(mask[90])

    def test_go_no_go_exact_thresholds(self):
        maes = {p: {"V1": 10., "V2": 11., "V3": 9.} for p in ("FUDS", "US06")}
        self.assertEqual(decide(maes, {"FUDS": 0.8999, "US06": 0.8999})["verdict"], "GO")
        self.assertEqual(decide(maes, {"FUDS": 0.9, "US06": 0.89})["verdict"], "NO-GO")
        maes["US06"]["V3"] = 9.0001
        self.assertEqual(decide(maes, {"FUDS": 0.1, "US06": 0.1})["verdict"], "NO-GO")

    def test_metrics_in_mV(self):
        values = metrics(np.array([-0.01, 0.01]))
        for key in ("mae", "rmse", "max_error", "p99"):
            self.assertAlmostEqual(values[key], 10.)

    def test_V3_zero_correction_matches_V1(self):
        raw = np.column_stack((np.arange(100), np.ones(100), np.full(100, 3.3), np.full(100, 25)))
        cycle = _cycle("test", raw, 2.5, {"z0": 1, "h0": 1})
        z = np.array([0., 0.5, 1.])
        curves = OCVCurves(z, np.array([3., 3.3, 3.6]), z, np.array([2.9, 3.2, 3.5]))
        theta = np.log([0.02, 0.01, 0.01, 2., 40.])
        v1 = simulate(cycle, 2.5, curves, theta, "V1")[0]
        v3 = simulate(cycle, 2.5, curves, np.r_[theta, np.log(1.), 0., 0.], "V3")[0]
        np.testing.assert_allclose(v1, v3)


class SavedResultTests(unittest.TestCase):
    def test_saved_metrics_and_decisions(self):
        from public_integrity import verify
        verdicts = {}
        for name in ("cell_A_a123", "cell_B_mendeley"):
            path = OUTPUT/name/"summary.json"
            verify(path)
            result = json.loads(path.read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["manifest"]["protocol_sha256"], sha256(PROTOCOL_PATH))
            self.assertEqual(len(result["metrics"]), 96)
            for profile, values in result["global_mae_mV"].items():
                for model, value in values.items():
                    row, = [r for r in result["metrics"] if r["profile"] == profile
                            and r["model"] == model and r["scope"] == "global"]
                    self.assertEqual(row["mae_mV"], value)
            saturation = {p: d["saturation_fraction"] for p, d in result["diagnostics"].items()}
            self.assertEqual(decide(result["global_mae_mV"], saturation), result["decision"])
            verdicts[name] = result["decision"]["verdict"]
        self.assertEqual(json.loads((OUTPUT/"global_decision.json").read_text())["global"], verdicts["cell_A_a123"])


if __name__ == "__main__":
    unittest.main()
