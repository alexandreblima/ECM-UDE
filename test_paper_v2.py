"""Exact statistical thresholds, causal SOC anchors and monotone source support."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import wilcoxon

from data_lfp import ROOT, save_json, sha256
from exp_lfp_gonogo import rc_voltage
from exp_paper_v2_seeds import adjust_holm, shifted_wilcoxon, statistics
from exp_paper_v2_calce_z0 import audit_rows, hypothetical_static
from exp_paper_v2_monotone import monotone_ocv, run_cell, simulate, source_jacobian, verdict
from test_lfp_static import synthetic


class StatisticalTests(unittest.TestCase):
    def test_exact_tost_tail(self):
        d = np.arange(1., 11.)
        self.assertEqual(shifted_wilcoxon(d, "greater")["p_raw"], 1/1024)
        self.assertEqual(shifted_wilcoxon(-d, "less")["p_raw"], 1/1024)
        self.assertEqual(shifted_wilcoxon(np.zeros(10), "less")["p_raw"], 1.)
        d = np.array([1., -2., 3., -4., 5., -6., 7., 8., 9., 10.])
        for side in ("greater", "less"):
            self.assertEqual(shifted_wilcoxon(d, side)["p_raw"],
                             wilcoxon(d, alternative=side, method="exact").pvalue)

    def test_holm_twenty_and_boundary(self):
        records = [{"p_raw": 1/1024} for _ in range(20)]
        adjust_holm(records)
        self.assertTrue(all(r["p_holm"] == 20/1024 for r in records))
        delta = np.ones(10)
        self.assertEqual(shifted_wilcoxon(delta-1, "less")["p_raw"], 1.)
        self.assertLess(shifted_wilcoxon(np.full(10, .99)-1, "less")["p_raw"], .05)

    def test_calce_charge_count_priority(self):
        a = np.array([[-120., 2., 1., 4.195, 1.97, 0.],
                      [-60., 3., .2, 4.2, 1.99, 0.],
                      [0., 3., .019, 4.2, 2., 0.],
                      [600., 4., 0., 4.19, 2., 0.],
                      [601., 5., -1., 4.1, 2., .002],
                      [2040., 5., -1., 3.85, 2., .4],
                      [2041., 6., 0., 3.92, 2., .4],
                      [2042., 7., -.1, 3.92, 2., .4]])
        result = audit_rows(a, (np.array([3., 4.2]), np.array([0., 1.])))
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["z0"], .8)
        self.assertEqual(result["method"], "same_file_charge_count")

    def test_unproven_full_charge_and_noncontiguous_rest_are_excluded(self):
        a = np.array([[0., 5., .019, 4.2, 2., 0.],
                      [10., 6., 0., 3.6, 2., .4],
                      [80., 6., 0., 3.6, 2., .4],
                      [81., 7., -.1, 3.6, 2., .4]])
        with self.assertLogs("exp_paper_v2_calce_z0", level="WARNING"):
            result = audit_rows(a, (np.array([3., 4.2]), np.array([0., 1.])))
        self.assertFalse(result["eligible"])

    def test_nonphysical_candidate_not_used_for_revised_verdict(self):
        values = {name: {"V1": 20., "V5": 10., "V6": 18., "V7": 18.}
                  for name in ("DST_25C", "US06_25C", "BJDST_25C")}
        self.assertEqual(hypothetical_static(values)["verdict"], "POSITIVA")
        self.assertEqual(hypothetical_static(values, {"US06_25C": ["V5"]})["verdict"], "NEGATIVA")

    def test_calce_sixty_second_mean_or_exclude(self):
        a = np.array([[t, 6., 0., 3.6, 2., .4] for t in (0., 10., 20., 30., 40., 50., 60.)]+[
            [61., 7., -.1, 3.6, 2., .4]])
        result = audit_rows(a, (np.array([3., 4.2]), np.array([0., 1.])))
        self.assertAlmostEqual(result["z0"], .5)
        self.assertEqual(result["method"], "independent_OCV_last60s_rest_mean")
        with self.assertLogs("exp_paper_v2_calce_z0", level="WARNING"):
            result = audit_rows(a[-2:], None)
        self.assertFalse(result["eligible"])

    def test_monotone_boundary_support_is_source_only(self):
        cell, source, curves = synthetic()
        theta = np.r_[np.log([.02, .01, .03, 10., 100.]), 1., .95, 3., np.full(10, -3.)]
        knots, voltage = monotone_ocv(theta, source, cell.qnom_Ah)
        self.assertTrue(np.all(np.diff(voltage) > 0))
        grid = np.linspace(knots[0]-.5, knots[-1]+.5, 10000)
        spline = PchipInterpolator(knots, voltage)(np.clip(grid, knots[0], knots[-1]))
        self.assertTrue(np.all(np.diff(spline) >= -1e-12))
        self.assertEqual(spline[0], voltage[0])
        self.assertEqual(spline[-1], voltage[-1])
        alternative_target = replace(source, net_Ah=source.net_Ah*30, z0=.1)
        alternate, alternate_z = simulate(theta, alternative_target, source, cell.qnom_Ah)
        alternate_ocv = alternate+.02*alternative_target.current
        alternate_ocv += (rc_voltage(alternative_target.current, .01, 10.)+
                          rc_voltage(alternative_target.current, .03, 100.))
        outside = alternate_z < knots[0]
        self.assertTrue(outside.any())
        np.testing.assert_allclose(alternate_ocv[outside], voltage[0], atol=2e-12)
        target = source
        prediction, z = simulate(theta, target, source, cell.qnom_Ah)
        expected_ocv = PchipInterpolator(knots, voltage)(np.clip(z, knots[0], knots[-1]))
        rc1, rc2 = 0., 0.
        expected = []
        for k, current in enumerate(target.current):
            expected.append(expected_ocv[k]-.02*current-rc1-rc2)
            rc1 = np.exp(-.1)*rc1+.01*(-np.expm1(-.1))*current
            rc2 = np.exp(-.01)*rc2+.03*(-np.expm1(-.01))*current
        np.testing.assert_allclose(prediction, expected, atol=2e-11)

    def test_exact_monotone_robustness(self):
        self.assertTrue(verdict({"a": 15., "b": 15.}, {"a": 20., "b": 20.}, ["a", "b"])["robust"])
        self.assertFalse(verdict({"a": 15., "b": 15.00001}, {"a": 20., "b": 20.}, ["a", "b"])["robust"])

    def test_candidate_source_support_cancels_Q_and_z0_exactly(self):
        cell, source, _ = synthetic()
        theta = np.r_[np.log([.02, .01, .03, 10., 100.]), 1., .95, 3., np.full(10, -3.)]
        alternative = theta.copy()
        alternative[5:7] = [1.05, .93]
        np.testing.assert_array_equal(simulate(theta, source, source, cell.qnom_Ah)[0],
                                      simulate(alternative, source, source, cell.qnom_Ah)[0])

    def test_analytic_monotone_source_jacobian(self):
        cell, source, _ = synthetic()
        for increments in (np.linspace(-5., -2., 10), np.r_[-.3, np.full(8, -5.), -18.]):
            theta = np.r_[np.log([.02, .01, .03, 10., 100.]), 1., .95, 3., increments]
            jac = source_jacobian(theta, source, cell.qnom_Ah)
            for k in range(18):
                delta = 1e-4 if k >= 7 else 1e-5
                high, low = theta.copy(), theta.copy()
                high[k] += delta
                low[k] -= delta
                numerical = (simulate(high, source, source, cell.qnom_Ah)[0]-
                             simulate(low, source, source, cell.qnom_Ah)[0])/(2*delta)
                np.testing.assert_allclose(jac[:, k], numerical, atol=2e-8, rtol=1e-5)
            np.testing.assert_array_equal(jac[:, 5:7], 0.)

    def test_identification_failure_is_not_scored_as_transfer(self):
        cell, source, curves = synthetic()
        with TemporaryDirectory() as directory:
            out = Path(directory)
            old, failure = out/"old.json", out/"failure.json"
            save_json(old, {"global_mae_mV": {"DST": {"V1": 20.}}})
            save_json(failure, {"attempts": [{"success": False, "nfev": 500} for _ in range(3)]})
            policy = {"A123_failure_path": str(failure), "A123_failure_sha256": sha256(failure)}
            with patch("exp_paper_v2_monotone.OUTPUT", out), patch("exp_paper_v2_monotone.simulate") as score:
                with self.assertLogs("exp_paper_v2_monotone", level="ERROR"):
                    result = run_cell(cell, curves, "DST", old, ["DST"], reused_failure=policy)
            score.assert_not_called()
            self.assertEqual(result["status"], "INCONCLUSIVE_NLS")
            self.assertIsNone(result["principal_robustness"]["robust"])
            self.assertTrue(result["target_voltage_not_evaluated_with_failed_parameters"])


class PersistedTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/"results/unified_10seeds/summary.json").is_file(), "Run seeds first")
    def test_ten_seed_extension(self):
        s = json.loads((ROOT/"results/unified_10seeds/summary.json").read_text())
        if not s.get("complete"):
            self.skipTest("Ten-seed execution ongoing")
        old = json.loads((ROOT/"results/unified/summary.json").read_text())
        for dataset, data in s["datasets"].items():
            self.assertEqual(data["conditions"], old["datasets"][dataset]["conditions"])
            for model, runs in data["runs"].items():
                seeds = ["deterministic"] if model.endswith("_nls") else [str(n) for n in range(42, 52)]
                self.assertEqual(set(runs), set(seeds))
                for seed in seeds:
                    run = runs[seed]
                    self.assertEqual(sha256((ROOT/run["checkpoint"]).resolve()), run["sha256"])
                    if seed in ("deterministic", "42", "43", "44", "45", "46"):
                        self.assertEqual(run["metrics"], old["datasets"][dataset]["runs"][model][seed]["metrics"])
        computed = statistics(s)
        saved = json.loads((ROOT/"results/unified_10seeds/statistics.json").read_text())
        self.assertEqual(computed, saved)
        self.assertEqual(len(saved["equivalence"]), 120)
        self.assertEqual(len(saved["LSTM_effects"]), 20)

    @unittest.skipUnless((ROOT/"results/calce_z0_fix/summary.json").is_file(), "Run SOC audit first")
    def test_saved_soc_audit_and_unmodified_fitted_z0_models(self):
        s = json.loads((ROOT/"results/calce_z0_fix/summary.json").read_text())
        for condition, audit in s["audits"].items():
            self.assertTrue(audit["eligible"])
            self.assertAlmostEqual(audit["z0"], .8, places=3)
            self.assertEqual(audit["method"], "same_file_charge_count")
            for model in ("V6", "V7"):
                result = s["static"][condition][model]
                self.assertAlmostEqual(result["before_mae_mV"], result["after_metrics_mV"]["mae"], places=10)
            self.assertEqual(len(s["unified"][condition]["ecm_ude"]), 10)

    @unittest.skipUnless((ROOT/"results/static_monotone/summary.json").is_file(), "Run monotone stage first")
    def test_saved_monotone_shape_bounds_and_metrics(self):
        s = json.loads((ROOT/"results/static_monotone/summary.json").read_text())
        self.assertTrue(s["complete"])
        for name, data in s["datasets"].items():
            if data["status"] == "INCONCLUSIVE_NLS":
                self.assertIsNone(data["principal_robustness"]["robust"])
                self.assertTrue(all(not a["success"] and a["nfev"] == 500 for a in data["fit"]["attempts"]))
                self.assertTrue(data["target_voltage_not_evaluated_with_failed_parameters"])
                continue
            fit = data["fit"]
            p = fit["parameters"]
            self.assertLessEqual(p["tau2_s"], 3600.000001)
            knots, v = np.array(fit["OCV_knots_soc"]), np.array(fit["OCV_knots_V"])
            self.assertEqual(len(knots), 11)
            self.assertTrue(np.all(np.diff(v) > 0))
            grid = np.linspace(knots[0], knots[-1], 1001)
            self.assertTrue(np.all(np.diff(PchipInterpolator(knots, v)(grid)) >= -1e-12))
            for profile, mae in data["global_mae_mV"].items():
                self.assertTrue(np.isfinite(mae), profile)
            recalculated = verdict(data["global_mae_mV"],
                                   {k: r["V1_mae_mV"] for k, r in
                                    data["principal_robustness"]["profiles"].items()},
                                   data["primary_targets"])
            self.assertEqual(recalculated, data["principal_robustness"])


if __name__ == "__main__":
    unittest.main()
