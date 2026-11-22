"""Independent GP, interpolation, coverage and persisted voltage checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data_lfp import DATA_ROOT, sha256
from exp_a123_temperature_interp import (
    GP_JITTER, METHODS, OUTPUT, PARAMETERS, PROTOCOL_PATH, REGISTRATION, ParameterGP,
    constrain, decision, design, fit_gp, fit_predictions, inverse, kernel,
    load_calibration, main, physical_bounds, protocol, require_heldout_trajectories,
    sample_posterior, snapshot, transform, voltage_band)
from exp_a123_temperature import OCV_SOURCE
from lfp_ocv import OCVCurves


class InterpolationTests(unittest.TestCase):
    def test_missing_heldout_trajectories_exit_before_work(self):
        cfg = {"test_temperatures_C": [10, 40], "test_profiles": ["FUDS", "US06"]}
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            with patch("exp_a123_temperature_interp.ROOT", root), \
                    patch("exp_a123_temperature_interp.THERMAL", root/"thermal"), \
                    patch("exp_a123_temperature_interp.protocol", return_value=cfg):
                with patch("exp_a123_temperature_interp.snapshot") as snapshot_call:
                    with self.assertRaisesRegex(SystemExit, "python exp_a123_temperature.py") as error:
                        main()
                    snapshot_call.assert_not_called()
                self.assertIn("thermal/10C/FUDS_trajectories.npz", str(error.exception))
                self.assertIn("thermal/40C/US06_trajectories.npz", str(error.exception))
                for temperature in cfg["test_temperatures_C"]:
                    for profile in cfg["test_profiles"]:
                        path = root/"thermal"/f"{temperature}C"/f"{profile}_trajectories.npz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                require_heldout_trajectories()

    def test_frozen_protocol_threshold_and_coverage_inclusive(self):
        self.assertEqual(sha256(PROTOCOL_PATH), "f0c6b8718383828856b3578cd96c9a2d25983d5b0cf4b48d83e4a9d25482a3ba")
        rows = [{"temperature_C": t, "profile": p, "method": m, "recovered_fraction": .5,
                 "coverage_95": .85 if t == 10 else .99}
                for t in (10, 40) for p in ("FUDS", "US06") for m in METHODS]
        d = decision(rows)
        self.assertEqual(d["verdict"], "POSITIVO")
        self.assertEqual(d["P4_uncertainty"], "calibrada")
        for row in rows:
            if row["temperature_C"] == 40 and row["profile"] == "US06" and row["method"] != "P1":
                row["recovered_fraction"] = .499999
        self.assertEqual(decision(rows)["verdict"], "NEGATIVO")
        rows[-1]["coverage_95"] = .990001
        self.assertEqual(decision(rows)["P4_uncertainty"], "nao calibrada")
        rows[-1]["recovered_fraction"] = 2.
        self.assertEqual(decision(rows)["verdict"], "POSITIVO")

    def test_transform_bounds_and_joint_branch_swap(self):
        values = np.array([1., .95, .02, .03, .04, 100., 20.])
        np.testing.assert_allclose(inverse(transform(values)), values)
        result, diag = constrain(values, 1.)
        self.assertTrue(diag["branch_swap"])
        np.testing.assert_allclose(result, [1., .95, .02, .04, .03, 20., 100.])
        values[0], values[1], values[6] = .5, 1.1, 4000.
        _, diag = constrain(values, 1.)
        self.assertEqual(set(diag["projected"]), {"Qeff_Ah", "z0", "tau2_s"})

    def test_arrhenius_and_linear_ols_exact(self):
        t = np.array([0., 25., 50.])
        for index in (0, 2):
            log = index == 2
            beta = np.array([.3, .001]) if not log else np.array([-8., 1500.])
            y = design(t, log)@beta
            fitted = fit_gp(t, y, index)
            np.testing.assert_allclose(fitted.beta, beta, atol=1e-9)
            np.testing.assert_allclose(fitted.mean(np.array([10., 40.])),
                                       design(np.array([10., 40.]), log)@beta, atol=1e-10)

    def test_gp_posterior_against_independent_scalar_matrix_formula(self):
        t = np.array([0., 25., 50.])
        beta = np.array([1., .01])
        residual = np.array([.1, -.2, .1])
        gp = ParameterGP(t, beta, False, .3, residual, .7, 20., [])
        target = np.array([10., 40.])
        mean, covariance = gp.posterior(target)
        K = np.array([[.7**2*np.exp(-.5*((a-b)/20)**2) for b in t] for a in t])+np.eye(3)*GP_JITTER
        k = np.array([[.7**2*np.exp(-.5*((a-b)/20)**2) for b in t] for a in target])
        kt = np.array([[.7**2*np.exp(-.5*((a-b)/20)**2) for b in target] for a in target])
        expected_mean = 1+.01*target+.3*k@np.linalg.solve(K, residual)
        expected_cov = .3**2*(kt-k@np.linalg.solve(K, k.T))
        np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(covariance, expected_cov, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_conditioned_samples_reproducible_and_bounded(self):
        t = np.array([0., 25., 50.])
        physical = np.array([[.95, .96, .03, .02, .05, 20., 100.],
                             [1., .97, .025, .018, .04, 15., 150.],
                             [1.05, .98, .02, .015, .03, 10., 200.]])
        y = transform(physical)
        gps = [fit_gp(t, y[:, k], k) for k in range(7)]
        target = np.array([10., 40.])
        first, metadata = sample_posterior(gps, target, 1.)
        second, _ = sample_posterior(gps, target, 1.)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (200, 2, 7))
        lo, hi = physical_bounds(1.)
        self.assertTrue(np.all(first >= lo) and np.all(first <= hi))
        self.assertTrue(np.all(first[:, :, 5] <= first[:, :, 6]))
        for name in PARAMETERS:
            self.assertEqual(metadata[name]["retained"], 200)

    @unittest.skipUnless(DATA_ROOT.is_dir(), "Raw datasets unavailable: set ECM_UDE_DATA")
    def test_calibration_load_reads_no_test_numerics_and_predictions_take_only_three_points(self):
        with patch("exp_a123_temperature_interp.json.loads", wraps=json.loads) as reads:
            calibration, q, _, provenance = load_calibration()
        self.assertEqual(set(provenance), {"0", "25", "50"})
        self.assertEqual(calibration.shape, (3, 7))
        for call in reads.call_args_list:
            text = call.args[0]
            self.assertNotIn('"source_temperature_C": 10', text)
            self.assertNotIn('"source_temperature_C": 40', text)
        predictions, _, _, _ = fit_predictions(calibration, q)
        self.assertEqual(set(predictions), {"10", "40"})
        for t, expected in (("10", calibration[0]), ("40", calibration[2])):
            np.testing.assert_allclose(list(predictions[t]["P1"]["parameters"].values()), expected)
        y = transform(calibration)
        expected = inverse(np.array([np.interp(10., [0., 25., 50.], y[:, k]) for k in range(7)]))
        np.testing.assert_allclose(list(predictions["10"]["P2"]["parameters"].values()), expected)

    def test_empirical_band_and_coverage_exact(self):
        samples = np.tile(np.arange(200.)[:, None], (1, 4))
        measured = np.array([0., 50., 100., 199.])
        low, high, coverage = voltage_band(samples, measured)
        np.testing.assert_allclose(low, np.quantile(samples, .025, axis=0))
        np.testing.assert_allclose(high, np.quantile(samples, .975, axis=0))
        self.assertEqual(coverage, .5)


class PersistedTests(unittest.TestCase):
    def test_saved_metrics_recovery_and_decision(self):
        from public_integrity import verify
        path = OUTPUT/"summary.json"
        verify(path)
        s = json.loads(path.read_text())
        self.assertTrue(s["complete"])
        self.assertEqual(s["protocol_sha256"], sha256(PROTOCOL_PATH))
        self.assertEqual(len(s["metrics"]), 16)
        for row in s["metrics"]:
            self.assertTrue(np.isfinite(row["mae_mV"]))
            floor, ceiling = row["floor_mae_mV"], row["ceiling_mae_mV"]
            if floor > ceiling:
                self.assertAlmostEqual(row["recovered_fraction"], (floor-row["mae_mV"])/(floor-ceiling))
        self.assertEqual(decision(s["metrics"]), s["decision"])


if __name__ == "__main__":
    unittest.main()
