"""Re-evaluate saved E1 neural checkpoints with their correction zeroed."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import exp_unified as U
import protocol as P
from data_calce import CALCE_FILES
from ecm import ECMParams

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "A"
SUMMARY_PATH = ROOT / "results" / "unified_10seeds" / "summary.json"
SEEDS = tuple(range(42, 52))
VARIANTS = ("output_residual", "ecm_ude")
MAIN_CONDITIONS = {
    "panasonic": tuple(P.PANASONIC_FILES),
    "calce": (
        "FUDS_25C", "DST_25C", "FUDS_0C", "DST_0C", "US06_0C",
        "BJDST_0C", "FUDS_45C", "DST_45C", "US06_45C", "BJDST_45C",
    ),
}
METRICS = ("mae", "rmse", "max_error")


def _contexts(dataset: str, cfg, saved_dataset: dict):
    source = "UDDS_25C" if dataset == "panasonic" else "FUDS_25C"
    saved_stats = saved_dataset["conditions"][source]["split"]["stats"]
    calibration, _ = U.ocv_calibration() if dataset == "calce" else (None, {})
    contexts = {}
    names = list(P.PANASONIC_FILES if dataset == "panasonic" else CALCE_FILES)
    for condition in names:
        if dataset == "panasonic":
            cycle, metadata = U.load_panasonic(condition, cfg)
        else:
            cycle, metadata = U.load_calce(condition, cfg, calibration)
        if metadata["sha256"] != saved_dataset["conditions"][condition]["sha256"]:
            raise ValueError(f"Raw input hash mismatch for {dataset}/{condition}")
        contexts[condition] = U.dataset_context(
            cycle,
            cfg,
            stats=saved_stats,
            source=condition == source,
        )
    return contexts


def _tail_test(differences: np.ndarray) -> tuple[dict, dict]:
    from exp_paper_v2_seeds import shifted_wilcoxon

    return (
        shifted_wilcoxon(differences + 1.0, "greater"),
        shifted_wilcoxon(differences - 1.0, "less"),
    )


def _adjust_tost(rows: list[dict]) -> None:
    for tail in ("lower", "upper"):
        family = [{"p_raw": row[f"p_{tail}_raw"], "_row": row} for row in rows]
        U.holm(family)
        for result in family:
            result["_row"][f"p_{tail}_holm"] = result["p_holm"]
    for row in rows:
        row["p_tost_raw"] = max(row["p_lower_raw"], row["p_upper_raw"])
        row["p_tost_holm"] = max(row["p_lower_holm"], row["p_upper_holm"])


def _test_rows(records: list[dict], difference_field: str) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault(
            (record["variant"], record["metric"]), []
        ).append(record)
    result = []
    for (variant, metric), condition_rows in groups.items():
        rows = []
        for record in condition_rows:
            values = np.asarray(record[difference_field], dtype=float)
            lower, upper = _tail_test(values)
            rows.append(
                {
                    "dataset": record["dataset"],
                    "variant": variant,
                    "condition": record["condition"],
                    "metric": metric,
                    "median_difference_mV": float(np.median(values)),
                    "n": int(len(values)),
                    "p_lower_raw": lower["p_raw"],
                    "p_upper_raw": upper["p_raw"],
                }
            )
        _adjust_tost(rows)
        result.extend(rows)
    return result


def _seed_differences(rows: list[dict], metric: str, column: str) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["dataset"], row["variant"], row["condition"]), []
        ).append(row)
    result = []
    for (dataset, variant, condition), observations in grouped.items():
        observations.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in observations] != list(SEEDS):
            raise ValueError(f"Incomplete seed family for {dataset}/{variant}/{condition}")
        result.append(
            {
                "dataset": dataset,
                "variant": variant,
                "condition": condition,
                "metric": metric,
                "difference": [row[column] for row in observations],
            }
        )
    return result


def _resolve_checkpoint(run_info: dict, dataset: str, variant: str, seed: int) -> Path:
    recorded = Path(run_info["checkpoint"])
    candidates = [recorded] if recorded.is_absolute() else [ROOT / recorded]
    for result_set in ("unified", "unified_10seeds"):
        candidates.append(
            ROOT
            / "results"
            / result_set
            / "checkpoints"
            / dataset
            / f"{variant}_seed{seed}.pt"
        )
    for candidate in candidates:
        if candidate.is_file() and P.sha256(candidate) == run_info["sha256"]:
            return candidate
    raise FileNotFoundError(
        f"No checkpoint matching the recorded SHA256 for {dataset}/{variant}/{seed}"
    )


def run(*, smoke: bool = False) -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    metrics_by_checkpoint: list[dict] = []
    checkpoint_hashes: dict[str, str] = {}
    for dataset, cfg in P.DATASETS.items():
        saved_dataset = summary["datasets"][dataset]
        contexts = _contexts(dataset, cfg, saved_dataset)
        source = "UDDS_25C" if dataset == "panasonic" else "FUDS_25C"
        nls_path = ROOT / "results" / "unified_10seeds" / "checkpoints" / dataset / "ecm_1rc_nls.json"
        nls_info = json.loads(nls_path.read_text())
        nls_params = ECMParams.from_dict(nls_info["params"])
        conditions = list(MAIN_CONDITIONS[dataset])
        seeds = SEEDS[:1] if smoke else SEEDS
        conditions = conditions[:1] if smoke else conditions

        for variant in VARIANTS:
            seed_runs = summary["datasets"][dataset]["runs"][variant]
            pg_runs = summary["datasets"][dataset]["runs"]["physics_gradient"]
            for seed in seeds:
                seed_key = str(seed)
                run_info = seed_runs[seed_key]
                checkpoint = _resolve_checkpoint(run_info, dataset, variant, seed)
                actual_hash = run_info["sha256"]
                checkpoint_hashes[str(checkpoint.relative_to(ROOT))] = actual_hash
                model = U.build_model(variant, contexts[source]["stats"], nls_params, cfg)
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                )
                original_by_condition = {}
                for condition in conditions:
                    reproduced = U.evaluate(model, variant, contexts[condition], cfg)
                    expected = run_info["metrics"][condition]["continuous"]
                    for metric in METRICS:
                        difference = reproduced["continuous"][metric] - expected[metric]
                        if difference != 0.0:
                            raise AssertionError(
                                f"Saved MAE/metric mismatch for {dataset}/{variant}/{seed}/"
                                f"{condition}/{metric}: {reproduced['continuous'][metric]} "
                                f"!= {expected[metric]}"
                            )
                    original_by_condition[condition] = reproduced["continuous"]

                with torch.no_grad():
                    model.p[10:].zero_()
                for condition in conditions:
                    zeroed = U.evaluate(model, variant, contexts[condition], cfg)["continuous"]
                    pg = pg_runs[seed_key]["metrics"][condition]["continuous"]
                    row = {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "condition": condition,
                        "checkpoint_sha256": actual_hash,
                    }
                    for metric in METRICS:
                        row[f"{metric}_network_mV"] = original_by_condition[condition][metric]
                        row[f"{metric}_zeroed_mV"] = zeroed[metric]
                        row[f"{metric}_network_minus_zeroed_mV"] = (
                            original_by_condition[condition][metric] - zeroed[metric]
                        )
                        row[f"{metric}_zeroed_minus_physics_gradient_mV"] = (
                            zeroed[metric] - pg[metric]
                        )
                    metrics_by_checkpoint.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / ("a2_smoke.csv" if smoke else "a2_zero_network_by_seed.csv")
    fields = list(metrics_by_checkpoint[0])
    with raw_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_by_checkpoint)
    (OUT / ("a2_smoke_checkpoint_hashes.json" if smoke else "a2_checkpoint_hashes.json")).write_text(
        json.dumps(checkpoint_hashes, indent=2, sort_keys=True) + "\n"
    )
    if smoke:
        return

    effect_tests = []
    equivalence_tests = []
    for metric in METRICS:
        effect_tests.extend(
            _test_rows(
                _seed_differences(
                    metrics_by_checkpoint,
                    metric,
                    f"{metric}_network_minus_zeroed_mV",
                ),
                "difference",
            )
        )
        equivalence_tests.extend(
            _test_rows(
                _seed_differences(
                    metrics_by_checkpoint,
                    metric,
                    f"{metric}_zeroed_minus_physics_gradient_mV",
                ),
                "difference",
            )
        )
    for filename, rows in (
        ("a2_network_effect_tost.csv", effect_tests),
        ("a2_zeroed_vs_physics_gradient_tost.csv", equivalence_tests),
    ):
        with (OUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
