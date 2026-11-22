"""Frozen thermal criterion and independently verified local V6 diagnostics."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.chebyshev import chebval

from data_lfp import integrate_Ah, sha256
from exp_a123_temperature import (MODELS, OUTPUT, SOURCE, OCV_SOURCE, decide_thermal,
                                  diagnostic_recovery, fit_local, protocol, snapshot,
                                  verify_protocols)
from lfp_ocv import OCVCurves
from test_lfp_static import synthetic


class ThermalTests(unittest.TestCase):
    def test_exact_three_of_four_both_profiles_and_local_exclusion(self):
        verify_protocols()
        values = {str(t): {name: {"V1": 20., "V5": 15., "V6": 16., "V7": 17., "V6_local": 0.}
                          for name in ("FUDS", "US06")} for t in (0, 10, 40, 50)}
        values["50"]["FUDS"]["V5"] = 15.00001
        d = decide_thermal(values)
        self.assertEqual(d["passing_temperatures"], 3)
        self.assertEqual(d["verdict"], "POSITIVA")
        values["40"]["US06"]["V5"] = 15.00001
        d = decide_thermal(values)
        self.assertEqual(d["passing_temperatures"], 2)
        self.assertEqual(d["verdict"], "NEGATIVA")
        values["40"]["US06"]["V7"] = 15.
        self.assertEqual(decide_thermal(values)["verdict"], "POSITIVA")
        for profiles in values.values():
            for v in profiles.values():
                v["V6_local"] = 1e12
        self.assertEqual(decide_thermal(values)["verdict"], "POSITIVA")

    def test_local_fit_reads_only_dst_and_fixed_reference(self):
        cell, source, curves = synthetic()
        cell.metadata = {"temperature_C": 40}
        cell.cycles = {"DST": source, "FUDS": replace(source, voltage=source.voltage+100)}
        seen = []

        def fake_predict(cycle, supplied_cell, supplied_curves, theta, model):
            self.assertIs(supplied_cell, cell)
            self.assertIs(supplied_curves, curves)
            self.assertEqual(model, "V6")
            seen.append(cycle)
            return cycle.voltage.copy(), cycle.soc.copy()

        with patch("exp_a123_temperature.predict", side_effect=fake_predict):
            fit = fit_local(cell, curves)
        self.assertTrue(seen)
        self.assertTrue(all(c is source for c in seen))
        self.assertTrue(fit["criterion_excluded"])
        self.assertEqual(fit["identified_on"], "DST")
        self.assertEqual(fit["source_temperature_C"], 40)
        self.assertAlmostEqual(fit["parameters"]["Qnom_Ah"], cell.qnom_Ah)

    def test_signed_recovery_and_explicit_zero_denominator(self):
        result = diagnostic_recovery({"V1": 20., "V6": 10., "V6_local": 15.})
        self.assertEqual(result["relative_to_transferred_V6"], -.5)
        self.assertEqual(result["relative_to_V1"], .25)
        with self.assertLogs("exp_a123_temperature", level="WARNING"):
            result = diagnostic_recovery({"V1": 0., "V6": 0., "V6_local": 1.})
        self.assertIsNone(result["relative_to_V1"])
        self.assertIsNone(result["relative_to_transferred_V6"])


class PersistedTests(unittest.TestCase):
    def test_saved_metrics_bounds_and_decision(self):
        from public_integrity import verify
        path = OUTPUT/"summary.json"
        verify(path)
        s = json.loads(path.read_text())
        self.assertTrue(s["complete"])
        self.assertEqual(s["frozen_DST25_fits"], json.loads((SOURCE/"summary.json").read_text())["fits"])
        maes = {}
        for temperature, result in s["temperatures"].items():
            self.assertEqual(result["loader"]["specimen"], "A1-007")
            fit = result["local_V6_fit"]
            self.assertEqual(fit["source_temperature_C"], int(temperature))
            self.assertEqual(fit["identified_on"], "DST")
            self.assertTrue(fit["criterion_excluded"])
            p = fit["parameters"]
            self.assertTrue(.9 <= p["Q_ratio"] <= 1.1 and .9 <= p["z0"] <= 1)
            self.assertLessEqual(p["tau2_s"], 3600.000001)
            self.assertIsNone(p["ocv_coefficients_V"])
            self.assertEqual(len(result["metrics"]), 3*5*8)
            for profile, values in result["global_mae_mV"].items():
                self.assertEqual(diagnostic_recovery(values), result["diagnostic_recovery"][profile])
                for model, value in values.items():
                    row, = [r for r in result["metrics"] if r["profile"] == profile
                            and r["model"] == model and r["scope"] == "global"]
                    self.assertEqual(row["mae_mV"], value)
            maes[temperature] = result["global_mae_mV"]
        self.assertEqual(decide_thermal(maes), s["decision"])


if __name__ == "__main__":
    unittest.main()
