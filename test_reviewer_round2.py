"""Focused validation for reviewer-round-two statistics and ablations."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import exp_unified as U
import protocol as P
from ecm import ECMParams
from exp_paper_v2_seeds import shifted_wilcoxon
from reviewer_round2_a1 import FrozenCorrection
from reviewer_round2_a2 import _resolve_checkpoint
from reviewer_round2_c import _causal_z0
from unified_dynamics import UnifiedECMUDE

ROOT = Path(__file__).resolve().parent


class ReviewerRound2StatisticsTests(unittest.TestCase):
    def test_exact_tost_tails_and_holm(self):
        difference = np.arange(1.0, 11.0)
        self.assertEqual(shifted_wilcoxon(difference, "greater")["p_raw"], 1 / 1024)
        self.assertEqual(shifted_wilcoxon(-difference, "less")["p_raw"], 1 / 1024)
        self.assertEqual(shifted_wilcoxon(np.zeros(10), "less")["p_raw"], 1.0)
        family = [{"p_raw": 1 / 1024} for _ in range(18)]
        U.holm(family)
        self.assertTrue(all(row["p_holm"] == 18 / 1024 for row in family))

    def test_soc_prefix_is_independent_of_future_current(self):
        prefix = np.array([0.0, 0.1, 0.2])
        first = P.causal_soc(np.r_[prefix, 0.3], qnom_Ah=2.0, z0=0.8)
        changed_future = P.causal_soc(np.r_[prefix, -9.0, 30.0], qnom_Ah=2.0, z0=0.8)
        np.testing.assert_array_equal(first[: len(prefix)], changed_future[: len(prefix)])

    def test_m50t_initial_soc_ignores_current_after_profile_start(self):
        time = np.arange(5, dtype=float)
        current = np.array([0.0, 0.4, -0.2, 0.0, 8.0])
        changed_future = current.copy()
        changed_future[4] = -100.0
        first = _causal_z0(time, current, 0.0, 3.0, 4.8)
        second = _causal_z0(time, changed_future, 0.0, 3.0, 4.8)
        self.assertEqual(first, second)

    def test_consolidated_report_replaces_legacy_and_english_sections(self):
        import reviewer_round2_e as E

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report_reviewer_round2.md"
            path.write_text("Earlier section\n\n## E — Adendo explorat\u00f3rio antes da reescrita\n\nOld result\n")
            with patch.object(E, "ROOT", Path(directory)):
                E._update_consolidated_report("", "First result")
                E._update_consolidated_report("", "Updated result")
            result = path.read_text()
            self.assertIn("Earlier section", result)
            self.assertNotIn("Old result", result)
            self.assertNotIn("First result", result)
            self.assertEqual(result.count("## E — Exploratory addendum before rewriting"), 1)
            self.assertIn("Updated result", result)

    def test_g_checks_public_report_integrity(self):
        import reviewer_round2_g as G

        with patch.object(G, "verify", side_effect=ValueError("report hash mismatch")) as verify:
            with self.assertRaisesRegex(ValueError, "report hash mismatch"):
                G.verified_inputs()
        verify.assert_called_once_with(G.F_DIR / "report.md")

    def test_a_report_matches_reconciled_published_report(self):
        import reviewer_round2_a_report as A

        before = (A.OUT / "a4_training_loss_summary.csv").read_bytes()
        loss, tost = A._a4_tables()
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            A._write_report(loss, tost, report)
            self.assertEqual(report.read_bytes(), (A.OUT / "report.md").read_bytes())
        self.assertEqual((A.OUT / "a4_training_loss_summary.csv").read_bytes(), before)

    def test_a_report_rejects_unreconciled_loss_basis(self):
        import reviewer_round2_a_report as A

        actual = A._read_csv
        def altered(path):
            rows = actual(path)
            if path.name == "a4_training_loss_summary.csv":
                rows[0]["loss_basis"] = "online last epoch"
            return rows
        with patch.object(A, "_read_csv", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "not the reconciled"):
                A._a4_tables()


class ReviewerRound2FreezeTests(unittest.TestCase):
    def test_a1_physical_parameters_have_no_gradient_or_update(self):
        stats = {
            "I_mean": 0.0,
            "I_std": 1.0,
            "V_mean": 3.7,
            "V_std": 0.2,
            "T_mean": 25.0,
            "T_std": 10.0,
            "SoC_mean": 0.5,
            "SoC_std": 0.3,
        }
        cfg = P.DatasetProtocol(2.9, 1.0, 0.1, 8, 4)
        params = ECMParams(
            ocv_coeffs=np.array([3.7, 0.1, 0.0, 0.0, 0.0, 0.0]),
            R0=0.03,
            R1=0.02,
            C1=800.0,
        )
        base = UnifiedECMUDE(stats, params, cfg, mode=1)
        model = FrozenCorrection(base, cfg)
        before = model.physical.detach().clone()
        optimizer = torch.optim.SGD([model.correction], lr=0.01)
        output = model(torch.zeros((1, 8, 3), dtype=torch.float32))
        output.square().mean().backward()
        self.assertIsNone(model.physical.grad)
        self.assertIsNotNone(model.correction.grad)
        optimizer.step()
        self.assertTrue(torch.equal(before, model.physical.detach()))


@unittest.skipUnless(P.DATA_ROOT.is_dir(), "Raw datasets unavailable: set ECM_UDE_DATA")
class ReviewerRound2CheckpointTests(unittest.TestCase):
    def test_saved_checkpoint_reproduces_mae_before_zeroing(self):
        summary_path = ROOT / "results" / "unified_10seeds" / "summary.json"
        summary = json.loads(summary_path.read_text())
        dataset = "panasonic"
        condition = "UDDS_25C"
        seed = 42
        variant = "output_residual"
        saved = summary["datasets"][dataset]
        cfg = P.DATASETS[dataset]
        cycle, metadata = U.load_panasonic(condition, cfg)
        self.assertEqual(metadata["sha256"], saved["conditions"][condition]["sha256"])
        stats = saved["conditions"][condition]["split"]["stats"]
        context = U.dataset_context(cycle, cfg, stats=stats, source=True)
        run_info = saved["runs"][variant][str(seed)]
        checkpoint = _resolve_checkpoint(run_info, dataset, variant, seed)
        params_path = (
            ROOT
            / "results"
            / "unified"
            / "checkpoints"
            / dataset
            / "ecm_1rc_nls.json"
        )
        params = ECMParams.from_dict(json.loads(params_path.read_text())["params"])
        model = U.build_model(variant, stats, params, cfg)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        actual = U.evaluate(model, variant, context, cfg)["continuous"]["mae"]
        expected = run_info["metrics"][condition]["continuous"]["mae"]
        self.assertEqual(actual, expected)


class ReviewerRound2TransferTests(unittest.TestCase):
    def test_synthesis_m50t_ranges_are_across_cell_medians(self):
        import csv
        import reviewer_round2_e as E

        source = ROOT / "results/reviewer_round2/E"
        with (source / "e3_superiority_tests.csv").open(newline="") as stream:
            tests = list(csv.DictReader(stream))
        with (source / "m50t_differences_by_cell.csv").open(newline="") as stream:
            cells = list(csv.DictReader(stream))
        rows = E._synthesis_rows(tests, cells)
        for variant in ("output_residual", "ecm_ude"):
            result = next(r for r in rows if r["configuration"] == "Protocolo original — LG M50T"
                          and r["variant"] == variant)
            for prefix, ref in (("vs_pg", "physics_gradient"), ("vs_ecm2", "ecm_2rc_nls")):
                values = [float(r["d_c_mV"]) for r in cells
                          if r["variant"] == variant and r["reference"] == ref]
                self.assertEqual(len(values), 7)
                self.assertEqual(result[f"{prefix}_minimum_difference_mV"], min(values))
                self.assertEqual(result[f"{prefix}_maximum_difference_mV"], max(values))
                self.assertAlmostEqual(result[f"{prefix}_median_difference_mV"], np.median(values))

    def test_g_soc_mask_includes_endpoints_and_intersects_pulses_and_rest(self):
        from types import SimpleNamespace
        from reviewer_round2_g import soc_masks

        cycle = {"SoC": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)}
        masks = {"pulse": np.array([False, True, False, True, False]),
                 "rest": np.array([True, False, True, False, True])}
        p1 = SimpleNamespace(soc_train_min=0.1, soc_train_max=0.4)
        p2 = SimpleNamespace(soc_train_min=0.2, soc_train_max=0.5)
        low, high, actual = soc_masks(cycle, masks, p1, p2)
        self.assertEqual((low, high), (0.2, 0.4))
        np.testing.assert_array_equal(actual["in_range"], [False, True, True, True, False])
        np.testing.assert_array_equal(actual["pulse_in_range"], [False, True, False, True, False])
        np.testing.assert_array_equal(actual["rest_in_range"], [False, False, True, False, False])

    def test_g_aging_difference_uses_fixed_arm_and_full_episode_baseline(self):
        from reviewer_round2_g import aging_comparison

        source = [{"cell": "G1", "model": "ecm_ude", "seed": 42, "n_episodes": 2,
                   "n_samples": 10, "mae_source_full_mV": 4.0}]
        later = {"cell": "G1", "model": "ecm_ude", "seed": "42", "arm": "fixed",
                 "block_position": "1", "block_label": "N50", "n_episodes": "3",
                 "n_samples": "20", "mae_mV": "7.0"}
        result = aging_comparison(source, [later, {**later, "arm": "updated", "mae_mV": "100"}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["increase_later_minus_source_mV"], 3.0)
        self.assertEqual(result[0]["n_source_samples"], 10)

    def test_g_hppc_inference_separates_masks_and_w3_cohorts(self):
        import reviewer_round2_g as G

        rows = []
        for cell in G.C.CELLS:
            for model, seeds, error in (("ecm_2rc_nls", ["deterministic"], 10.0),
                                        ("physics_gradient", G.C.SEEDS, 10.0),
                                        ("output_residual", G.C.SEEDS, 9.5),
                                        ("ecm_ude", G.C.SEEDS, 9.0)):
                for seed in seeds:
                    rows.append({"cell": cell, "model": model, "seed": seed,
                                 **{f"mae_{mask}_mV": error for mask in G.MASKS}})
        tests, differences = G.hppc_inference(rows)
        self.assertEqual(len(tests), 24)
        self.assertEqual(len(differences), 780)
        for row in tests:
            self.assertEqual(row["n"], 6 if row["cell_cohort"] == "without_W3" else 7)
            self.assertEqual(row["holm_family_n"], 2)
            self.assertEqual(row["median_difference_mV"],
                             -0.5 if row["variant"] == "output_residual" else -1.0)

    def test_g_synthesis_preserves_unrelated_rows_and_replaces_transfer_rows(self):
        import csv
        import shutil
        import reviewer_round2_g as G

        rows = []
        for cell in G.C.CELLS:
            for model, seeds, error in (("ecm_2rc_nls", ["deterministic"], 10.0),
                                        ("physics_gradient", G.C.SEEDS, 10.0),
                                        ("output_residual", G.C.SEEDS, 9.5),
                                        ("ecm_ude", G.C.SEEDS, 9.0)):
                for seed in seeds:
                    rows.append({"cell": cell, "model": model, "seed": seed,
                                 **{f"mae_{mask}_mV": error for mask in G.MASKS}})
        tests, differences = G.hppc_inference(rows)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "E"
            out = Path(temp) / "run"
            target.mkdir()
            out.mkdir()
            for suffix in ("csv", "tex"):
                before_g = G.OUT / "run" / f"synthesis_before_G.{suffix}"
                shutil.copy2(before_g if before_g.exists() else G.E_DIR / f"synthesis.{suffix}",
                             target / f"synthesis.{suffix}")
            with (target / "synthesis.csv").open(newline="") as stream:
                before = list(csv.DictReader(stream))
            with patch.object(G, "E_DIR", target):
                G.synthesize(tests, differences, out)
            with (target / "synthesis.csv").open(newline="") as stream:
                after = list(csv.DictReader(stream))
            original = [row for row in before if not row["configuration"].startswith("M50T F")]
            self.assertEqual(after[:len(original)], original)
            self.assertEqual(len(after), len(original) + 6)
            self.assertFalse(any("updated_capacity" in row["configuration"] or "F2 HPPC" in row["configuration"]
                                 for row in after))
            text = (target / "synthesis.tex").read_text()
            self.assertIn("minimum adjusted $p=18/32>0.05$", text)
            self.assertNotIn("M50T F1 1 updated", text)
            self.assertNotIn("M50T F2 HPPC", text)

    def test_cell_inference_uses_paired_seed_medians_and_two_variant_holm(self):
        from reviewer_round2_f import inference

        rows = []
        for cell in ("a", "b", "c", "d", "e", "f", "g"):
            rows.append({"cell": cell, "model": "ecm_2rc_nls",
                         "seed": "deterministic", "mae_full_mV": 10.0})
            for seed in range(42, 47):
                for model, error in (("physics_gradient", 10.0 + seed),
                                     ("output_residual", 9.5 + seed),
                                     ("ecm_ude", 9.0 + seed)):
                    rows.append({"cell": cell, "model": model, "seed": seed,
                                 "mae_full_mV": error})
        tests, differences = inference([], rows)
        pg = [row for row in tests if row["reference"] == "physics_gradient"]
        self.assertEqual(len(pg), 2)
        self.assertEqual(len(differences), 140)
        self.assertEqual([row["n"] for row in pg], [7, 7])
        self.assertEqual([row["median_difference_mV"] for row in pg], [-0.5, -1.0])
        self.assertTrue(all(row["holm_family_n"] == 2 for row in pg))
        self.assertTrue(all(row["p_superiority_holm"] == 0.015625 for row in pg))

    def test_synthesis_adds_transfer_rows_without_losing_existing_rows(self):
        import csv
        import reviewer_round2_f as F

        hppc = []
        for cell in ("a", "b", "c", "d", "e", "f", "g"):
            hppc.append({"cell": cell, "model": "ecm_2rc_nls",
                         "seed": "deterministic", "mae_full_mV": 10.0})
            for seed in range(42, 47):
                for model, error in (("physics_gradient", 10.0),
                                     ("output_residual", 9.5), ("ecm_ude", 9.0)):
                    hppc.append({"cell": cell, "model": model, "seed": seed,
                                 "mae_full_mV": error})
        tests, differences = F.inference([], hppc)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "results/reviewer_round2/E"
            target.mkdir(parents=True)
            fields = ["configuration", "protocol_status", "variant", "analysis_unit_n"]
            suffixes = (
                "status", "n_tests", "better_holm_count", "equivalent_holm_count",
                "worse_holm_count", "unresolved_holm_count", "median_difference_mV",
                "minimum_difference_mV", "maximum_difference_mV", "p_superiority_holm",
                "p_tost_holm", "p_inferiority_holm", "cells_d_below_0",
                "cells_d_below_minus1", "cells_d_above_0", "cells_within_pm1",
            )
            fields += [f"{prefix}_{suffix}" for prefix in ("vs_pg", "vs_ecm2") for suffix in suffixes]
            with (target / "synthesis.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"configuration": "existing", "variant": "output_residual"})
            (target / "synthesis.tex").write_text("existing output_residual\n\\bottomrule\n")
            with (target / "synthesis.csv").open(newline="") as stream:
                before = list(csv.DictReader(stream))
            output = root / "provenance"
            output.mkdir()
            with patch.object(F, "ROOT", root):
                F.update_synthesis(tests, differences, output)
            with (target / "synthesis.csv").open(newline="") as stream:
                after = list(csv.DictReader(stream))
            self.assertEqual(after[:len(before)], before)
            self.assertEqual(len(after), len(before) + 2)
            self.assertEqual(after[-1]["analysis_unit_n"], "7")
            self.assertIn(r"output\_residual", (target / "synthesis.tex").read_text())


if __name__ == "__main__":
    unittest.main()
