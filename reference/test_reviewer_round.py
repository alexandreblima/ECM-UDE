"""Verify preregistration, saved-fit ablations, paired tests and saved histories."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import rankdata

from data_lfp import ROOT, sha256
from exp_param_ablation import explained, interpret, verify
from exp_paper_v2_seeds import shifted_wilcoxon
from figures_ieee import inspect_pdf


def read(relative):
    return json.loads((ROOT/relative).read_text())


class ReviewerRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ablation = read("results/param_ablation/summary.json")
        cls.paired = read("results/variant_tost/summary.json")
        cls.training = read("results/training_audit/summary.json")

    def test_protocol_and_producer_hashes(self):
        verify()
        self.assertEqual(self.ablation["registration_commit"], "d14935e")
        self.assertEqual(self.ablation["source_sha256"], sha256(ROOT/"exp_param_ablation.py"))
        for path, digest in self.ablation["manifest"]["shared_source_sha256"].items():
            self.assertEqual(sha256(ROOT/path), digest)
        for path, digest in self.ablation["manifest"]["saved_baseline_sha256"].items():
            self.assertEqual(sha256(Path(path)), digest)

    def test_fixed_coordinates_fit_bounds_and_failed_fit(self):
        failures = []
        for block_name, block in self.ablation["blocks"].items():
            for model, fit in block["fits"].items():
                if not fit["success"]:
                    failures.append((block_name, model))
                    self.assertTrue(all(not a["success"] and a["nfev"] == 500 for a in fit["attempts"]))
                    continue
                if model in ("V1", "V6"):
                    continue
                theta = np.array(fit["theta"])
                if model == "V6-Q":
                    np.testing.assert_array_equal(theta[:5], block["fits"]["V1"]["theta"][:5])
                    self.assertEqual(fit["free_coordinate_indices"], [5, 6])
                if model == "V6-RC":
                    self.assertEqual(theta[5], 1)
                    self.assertEqual(len(fit["free_coordinate_indices"]), 5)
                self.assertTrue(np.all(np.exp(theta[:3]) >= 1e-5))
                self.assertTrue(np.all(np.exp(theta[:3]) <= .5))
                self.assertTrue(np.all(np.exp(theta[3:5]) <= 3600))
                if model != "V6-RC":
                    self.assertTrue(.9 <= theta[5] <= 1.1)
                    self.assertTrue((.7 if model == "V6-wide" else .9) <= theta[6] <= 1)
                self.assertEqual(len(fit["attempts"]), 3)
                best = min(a["SSE_V2"] for a in fit["attempts"] if a["success"])
                self.assertEqual(fit["attempts"][fit["selected_start"]]["SSE_V2"], best)
        self.assertEqual(failures, [("a123_local_10C", "V6-OCVT")])
        self.assertEqual(self.ablation["blocks"]["a123_local_10C"]["interpretation"]["verdict"], "conjunto")

    def test_every_metric_matches_persisted_voltage_and_signed_fraction(self):
        n_rows = 0
        for name, block in self.ablation["blocks"].items():
            for row in block["rows"]:
                n_rows += 1
                path = ROOT/"results/param_ablation"/name/f'{row["target"]}_trajectories.npz'
                with np.load(path) as trace:
                    self.assertTrue(np.allclose(np.diff(trace["t"]), 1))
                    for model, fit in block["fits"].items():
                        if not fit["success"]:
                            self.assertIsNone(row[model+"_mae_mV"])
                            self.assertNotIn(model, trace)
                            continue
                        actual = 1000*np.mean(np.abs(trace[model]-trace["measured_voltage_V"]))
                        self.assertAlmostEqual(actual, row[model+"_mae_mV"], places=10)
                    # V6-RC and V1 must use the same per-profile measured/audited SOC.
                    np.testing.assert_array_equal(trace["V1_soc"], trace["V6-RC_soc"])
                for model in ("V6-Q", "V6-RC"):
                    self.assertAlmostEqual(row[model+"_fraction"],
                                           explained(row["V1_mae_mV"], row[model+"_mae_mV"], row["V6_mae_mV"]))
        self.assertEqual(n_rows, 26)

    def test_interpretation_strict_majority_signed_and_zero(self):
        self.assertEqual(explained(10, 12, 8), -1)
        self.assertEqual(explained(10, 9, 12), -.5)
        with self.assertLogs("exp_param_ablation", level="WARNING"):
            self.assertIsNone(explained(10, 9, 10))
        def row(q, rc, positive=True):
            return {"V1_mae_mV": 10, "V6_mae_mV": 8 if positive else 12,
                    "V6-Q_fraction": q, "V6-RC_fraction": rc}
        self.assertEqual(interpret([row(.7, 0), row(.69, 0)])["verdict"], "conjunto")
        self.assertEqual(interpret([row(.7, 0), row(.7, 0), row(.69, 0)])["verdict"], "escala de SOC")
        self.assertEqual(interpret([row(.7, .7)])["verdict"], "conjunto")
        self.assertEqual(interpret([row(.7, .7, False)])["verdict"], "INCONCLUSIVO")
        expected = {"matched_temperature": ("conjunto", 7), "thermal_transfer": ("conjunto", 1),
                    "thermal_local": ("parametros RC", 8)}
        for group, (verdict, eligible) in expected.items():
            result = self.ablation["interpretation"][group]
            self.assertEqual(result["verdict"], verdict)
            self.assertEqual(result["eligible_positive_joint_gain"], eligible)

    def test_wide_uses_all_three_audited_targets(self):
        block = self.ablation["blocks"]["calce"]
        self.assertEqual(block["wide_decision"]["verdict"], "POSITIVO")
        for record in block["wide_decision"]["profiles"].values():
            self.assertGreaterEqual(record["reduction"], .25)
        self.assertEqual(set(block["wide_decision"]["profiles"]), {"DST_25C", "US06_25C", "BJDST_25C"})
        for audit in self.ablation["calce_audit"].values():
            self.assertTrue(.79 < audit["z0"] < .81)
            self.assertEqual(audit["method"], "same_file_charge_count")

    def test_exact_wilcoxon_zeros_ties_and_tails(self):
        delta = np.array([0, 0, 1, -1, 2, 2])
        ranks = rankdata(np.abs(delta[delta != 0]), method="average")
        signed = delta[delta != 0]
        sums = np.array([sum(r for r, s in zip(ranks, signs) if s)
                         for signs in itertools.product((False, True), repeat=len(ranks))])
        w = sum(ranks[signed > 0])
        for direction in ("greater", "less"):
            result = shifted_wilcoxon(delta, direction)
            expected = np.mean(sums >= w) if direction == "greater" else np.mean(sums <= w)
            self.assertEqual(result["p_raw"], expected)
        self.assertEqual(shifted_wilcoxon(np.zeros(10), "greater")["p_raw"], 1)

    def test_paired_differences_walsh_and_holm_families(self):
        saved = read("results/unified_10seeds/summary.json")
        self.assertEqual(len(self.paired["records"]), 36)
        for model in ("ecm_ude", "output_residual"):
            records = [r for r in self.paired["records"] if r["model"] == model]
            self.assertEqual(len(records), 18)
            for r in records:
                data = saved["datasets"][r["dataset"]]["runs"]
                d = np.array([data[model][str(s)]["metrics"][r["condition"]]["continuous"]["mae"]-
                              data["physics_gradient"][str(s)]["metrics"][r["condition"]]["continuous"]["mae"]
                              for s in range(42, 52)])
                np.testing.assert_array_equal(d, r["differences_mV"])
                walsh = [(x+y)/2 for i, x in enumerate(d) for y in d[i:]]
                self.assertEqual(np.median(walsh), r["Hodges_Lehmann_pseudomedian_mV"])
                self.assertEqual(np.median(d), r["median_difference_mV"])
                for direction in ("lower", "upper"):
                    expected = shifted_wilcoxon(d+1 if direction == "lower" else d-1,
                                                "greater" if direction == "lower" else "less")
                    self.assertEqual(expected["p_raw"], r[direction]["p_raw"])
                self.assertTrue(r["equivalent"])
            for direction in ("lower", "upper"):
                order = sorted(records, key=lambda r: r[direction]["p_raw"])
                running = 0
                for i, record in enumerate(order):
                    running = max(running, min(1., (18-i)*record[direction]["p_raw"]))
                    self.assertEqual(record[direction]["p_holm"], running)

    def test_saved_histories_selection_curves_and_fonts(self):
        self.assertEqual(len(self.training["runs"]), 80)
        self.assertEqual(len(self.training["curves"]), 8)
        histories = {}
        for row in self.training["runs"]:
            history = json.loads(Path(row["history_path"]).read_text())
            self.assertEqual(row["selected_epoch"], np.argmin(history["val_loss"])+1)
            self.assertEqual(row["executed_epochs"], len(history["val_loss"]))
            self.assertEqual(row["final_train_loss"], history["train_loss"][-1])
            self.assertEqual(row["final_val_loss"], history["val_loss"][-1])
            self.assertEqual(row["history_sha256"], sha256(Path(row["history_path"])))
            histories.setdefault(f'{row["dataset"]}/{row["model"]}', []).append(history)
        for key, curve in self.training["curves"].items():
            self.assertEqual(sha256(Path(curve["file"])), curve["sha256"])
            self.assertGreaterEqual(inspect_pdf(Path(curve["file"]), 3.5)["minimum_font_size_pt"], 8)
            for point in curve["aggregate"]:
                values = [h[point["loss"]][point["epoch"]-1] for h in histories[key]
                          if len(h[point["loss"]]) >= point["epoch"]]
                self.assertEqual(point["n_observed_seeds"], len(values))
                self.assertEqual(point["median"], np.median(values))
                self.assertEqual(point["minimum"], min(values))
                self.assertEqual(point["maximum"], max(values))

    def test_existing_lstm_bootstrap_is_paired_seed_percentile(self):
        stats = read("results/unified_10seeds/statistics.json")
        rng = np.random.default_rng(20261101)
        for row in stats["LSTM_effects"]:
            delta = np.asarray(row["differences_mV"])
            boot = np.median(rng.choice(delta, (20000, 10), replace=True), axis=1)
            np.testing.assert_array_equal(np.quantile(boot, [.025, .975]), row["bootstrap95_mV"])
        self.assertEqual(self.paired["LSTM_bootstrap"]["n_resamples"], 20000)

    def test_principal_tables_and_all_previous_bytes(self):
        def csv_rows(relative):
            with (ROOT/relative).open() as stream:
                return list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows("results/paper_tables/E1.csv")), 18)
        conditions = csv_rows("results/paper_tables/conditions.csv")
        e1 = [r for r in conditions if r["experiment"] == "E1"]
        self.assertEqual(len(e1), 20)
        self.assertEqual(sum(r["principal"] == "True" for r in e1), 18)
        thermal = [r for r in conditions if r["experiment"] == "E4 thermal" and
                   r["role"] == "thermal heldout test"]
        self.assertEqual(len(thermal), 8)
        self.assertTrue(all(r["principal"] == "True" for r in thermal))
        interp = [r for r in conditions if r["experiment"] == "E4 interp" and
                  r["role"] == "calibration DST0/25/50 only"]
        self.assertEqual({r["temperature_C"] for r in interp}, {"0", "25", "50"})
        previous = read("results/reviewer_round_audit/previous_sha256.json")
        self.assertEqual(len(previous), 1131)
        for relative, expected in previous.items():
            if sha256(ROOT/relative) != expected:
                self.assertEqual(sha256(ROOT/"figures_ieee/archive/ca3ca9b"/relative), expected)
        for path in (ROOT/"figures_ieee").glob("*.pdf"):
            if path.name != "fig02_physics_vs_lstm.pdf":
                self.assertEqual(sha256(path), previous[str(path.relative_to(ROOT))])


if __name__ == "__main__":
    unittest.main()
