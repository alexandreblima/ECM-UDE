"""Reproduce E1 error structure from frozen checkpoints, never fitting models.

Run with requirements-unified.txt installed:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze_e1_error_structure.py
    python -m unittest -v test_e1_error_structure

The default results/e1_error_structure directory must not already exist. Use
--output with a new directory for another run. All continuous MAEs, including
the validation-only 1RC baseline, must pass the 1e-6 mV gate before computing
peaks, tail metrics, SOC bands or OCV slopes. LSTM has no continuous estimand.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

import exp_unified as U
from ecm import ECMParams
from ecm_2rc import ECMParams2RC

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/"results/e1_error_structure"
SUMMARY = ROOT/"results/unified_10seeds/summary.json"
TRAINED = ("physics_gradient", "output_residual", "ecm_ude")
NEURAL = TRAINED[1:]
SEEDS = tuple(range(42, 52))
EXCLUDED = {"US06_25C", "BJDST_25C"}
BANDS = ("high", "mid", "low")
TAIL_METRICS = ("rmse", "p99", "max_error")
MAE_TOLERANCE_MV = 1e-6


def digest(path: Path) -> str:
    return U.P.sha256(path)


def checkpoint_directory(dataset: str, seed: int = 42) -> Path:
    group = "unified" if seed <= 46 else "unified_10seeds"
    return ROOT/"results"/group/"checkpoints"/dataset


def checkpoint_snapshot() -> dict[str, str]:
    paths = [path for group in ("unified", "unified_10seeds")
             for path in (ROOT/"results"/group/"checkpoints").rglob("*") if path.is_file()]
    if not paths:
        raise FileNotFoundError("No frozen unified checkpoints found")
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def verify_hashes(expected: dict[str, str]) -> None:
    for relative, sha in expected.items():
        if digest(ROOT/relative) != sha:
            raise RuntimeError(f"Read-only input changed: {relative}")


@contextlib.contextmanager
def repository_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def band_masks(soc: np.ndarray) -> dict[str, np.ndarray]:
    if not np.isfinite(soc).all() or np.any((soc < 0) | (soc > 1)):
        raise ValueError("Expected finite causal SOC in [0, 1]")
    return {"high": soc >= .8, "mid": (soc >= .2) & (soc < .8), "low": soc < .2}


def validate_mae(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                 expected: float, identity: str) -> tuple[np.ndarray, dict]:
    if prediction.shape != truth.shape or mask.shape != truth.shape or mask.dtype != bool:
        raise ValueError(f"Invalid prediction/truth/mask shape: {identity}")
    errors = (np.asarray(prediction, dtype=np.float64)[mask]
              - np.asarray(truth, dtype=np.float64)[mask])*1000
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(expected):
        raise ValueError(f"Empty or non-finite evaluated prediction: {identity}")
    mae = float(np.abs(errors).mean())
    difference = abs(mae-expected)
    if difference > MAE_TOLERANCE_MV:
        raise ValueError(f"MAE mismatch for {identity}: reproduced={mae:.15g} mV, "
                         f"saved={expected:.15g} mV, difference={difference:.15g} mV; "
                         "stopping before error-structure statistics")
    return errors, {"saved_mae_mV": expected, "reproduced_mae_mV": mae,
                    "absolute_difference_mV": difference, "passed": True}


def collect_validated_errors(saved: dict, scratch: Path) -> tuple[list[dict], dict, dict]:
    records, conditions, raw_inputs = [], {}, {}
    for dataset, source in (("panasonic", "UDDS_25C"), ("calce", "FUDS_25C")):
        cfg = U.P.DATASETS[dataset]
        data = saved["datasets"][dataset]
        names = [c for c in data["conditions"] if dataset != "calce" or c not in EXCLUDED]
        calibration = None
        if dataset == "calce":
            calibration, metadata = U.ocv_calibration()
            if calibration is None:
                raise FileNotFoundError("Independent CALCE OCV calibration is required")
            raw_inputs[os.path.relpath(metadata["path"], ROOT)] = metadata["sha256"]
        contexts = {}
        for condition in [source]+[c for c in names if c != source]:
            cycle, metadata = (U.load_panasonic(condition, cfg) if dataset == "panasonic"
                               else U.load_calce(condition, cfg, calibration))
            raw_inputs[os.path.relpath(metadata["path"], ROOT)] = metadata["sha256"]
            ctx = U.dataset_context(cycle, cfg, stats=contexts[source]["stats"] if condition != source else None,
                                    source=condition == source)
            contexts[condition] = ctx
            mask = ctx["mask"]
            indices = np.flatnonzero(mask)
            if condition != source and not mask.all():
                raise ValueError(f"Target evaluation must cover the entire profile: {dataset}/{condition}")
            conditions[f"{dataset}/{condition}"] = {
                "dataset": dataset, "condition": condition, "source": condition == source,
                "total_samples": len(mask), "evaluated_samples": len(indices),
                "evaluation_indices": indices, "truth_V": cycle["V"].numpy()[mask],
                "soc": cycle["SoC"].numpy()[mask], "time_s": cycle["t"].numpy()[mask],
                "profile_end_s": float(cycle["t"][-1]),
            }
        directory = checkpoint_directory(dataset)
        params = {
            "ecm_1rc_nls": ECMParams.from_dict(json.loads((directory/"ecm_1rc_nls.json").read_text())["params"]),
            "ecm_2rc_nls": ECMParams2RC.from_dict(json.loads((directory/"ecm_2rc_nls.json").read_text())["params"]),
        }

        def validate_and_store(model: str, seed: int | None, condition: str, prediction: np.ndarray) -> None:
            ctx = contexts[condition]
            key = "deterministic" if seed is None else str(seed)
            run = data["runs"][model][key]
            expected = run["metrics"][condition]["continuous"]["mae"]
            identity = f"{dataset}/{condition}/{model}/{key}"
            errors, validation = validate_mae(prediction, ctx["cycle"]["V"].numpy(), ctx["mask"],
                                              expected, identity)
            error_file = scratch/f"{len(records):04d}.npy"
            np.save(error_file, errors, allow_pickle=False)
            records.append({"dataset": dataset, "condition": condition, "model": model,
                            "seed": seed, "validation": validation, "error_file": error_file})

        for model, baseline in params.items():
            path = directory/f"{model}.json"
            if digest(path) != data["runs"][model]["deterministic"]["sha256"]:
                raise ValueError(f"Checkpoint differs from saved summary: {path}")
            for condition, ctx in contexts.items():
                validate_and_store(model, None, condition, U.nls_predict(baseline, ctx["cycle"], cfg)[0])
        for name in TRAINED:
            for seed in SEEDS:
                path = checkpoint_directory(dataset, seed)/f"{name}_seed{seed}.pt"
                if digest(path) != data["runs"][name][str(seed)]["sha256"]:
                    raise ValueError(f"Checkpoint differs from saved summary: {path}")
                model = U.build_model(name, contexts[source]["stats"], params["ecm_1rc_nls"], cfg)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                model.eval()
                for condition, ctx in contexts.items():
                    with torch.no_grad():
                        prediction = (model(ctx["ds"].X[None])[0].numpy()*ctx["stats"]["V_std"]
                                      + ctx["stats"]["V_mean"])
                    validate_and_store(name, seed, condition, prediction)
                print(f"Validated {dataset} {name} seed {seed}", flush=True)
    if len(conditions) != 18 or len(records) != 576:
        raise ValueError(f"Expected 18 conditions and 576 continuous MAE checks, got {len(conditions)}/{len(records)}")
    return records, conditions, raw_inputs


def error_metrics(errors: np.ndarray) -> dict[str, float]:
    absolute = np.abs(errors)
    return {"mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(errors*errors))),
            "p99": float(np.quantile(absolute, .99)), "max_error": float(absolute.max())}


def peak_record(errors: np.ndarray, condition: dict) -> dict:
    absolute = np.abs(errors)
    position = int(np.argmax(absolute))
    indices = condition["evaluation_indices"]
    sample = int(indices[position])
    total = condition["total_samples"]
    retained = indices < int(.95*total)
    if not retained.any():
        raise ValueError("No evaluated samples before the last 5%")
    return {
        "dataset": condition["dataset"], "condition": condition["condition"],
        "max_error_mV": float(absolute[position]), "sample_index": sample,
        "sample_fraction": sample/total,
        "elapsed_fraction": float(condition["time_s"][position])/condition["profile_end_s"],
        "measured_voltage_V": float(condition["truth_V"][position]),
        "predicted_voltage_V": float(condition["truth_V"][position]+errors[position]/1000),
        "causal_soc": float(condition["soc"][position]),
        "max_without_last_5_percent_mV": float(absolute[retained].max()),
        "evaluated_samples_without_last_5_percent": int(retained.sum()),
    }


def summarize_records(validated: list[dict], contexts: dict) -> tuple[list[dict], list[dict], list[dict]]:
    records, peaks, medians = [], [], []
    for raw in validated:
        if raw["model"] == "ecm_1rc_nls":
            continue
        condition = contexts[f'{raw["dataset"]}/{raw["condition"]}']
        errors = np.load(raw["error_file"], allow_pickle=False)
        bands = {}
        for band, mask in band_masks(condition["soc"]).items():
            count = int(mask.sum())
            bands[band] = {"n_samples": count, "empty": count == 0,
                           "mae_mV": float(np.abs(errors[mask]).mean()) if count else None}
        records.append({k: raw[k] for k in ("dataset", "condition", "model", "seed")}
                       | {"metrics_mV": error_metrics(errors), "soc_bands": bands})
        if raw["model"] == "ecm_2rc_nls":
            peaks.append(peak_record(errors, condition))
    for key, context in contexts.items():
        selected = [r for r in records if f'{r["dataset"]}/{r["condition"]}' == key]
        baseline = next(r for r in selected if r["model"] == "ecm_2rc_nls")
        models = {}
        for model in TRAINED:
            runs = [r for r in selected if r["model"] == model]
            if sorted(r["seed"] for r in runs) != list(SEEDS):
                raise ValueError(f"Missing seeds after validation: {key}/{model}")
            models[model] = {
                "metrics_mV": {metric: float(np.median([r["metrics_mV"][metric] for r in runs]))
                               for metric in ("mae", *TAIL_METRICS)},
                "soc_band_mae_mV": {
                    band: (float(np.median([r["soc_bands"][band]["mae_mV"] for r in runs]))
                           if not baseline["soc_bands"][band]["empty"] else None) for band in BANDS},
            }
        medians.append({"dataset": context["dataset"], "condition": context["condition"],
                        "baseline": baseline, "trained_medians": models})
    return records, peaks, medians


def ocv_slope(dataset: str) -> dict:
    path = checkpoint_directory(dataset)/"ecm_2rc_nls.json"
    coefficients = json.loads(path.read_text())["params"]["ocv_coeffs"]
    soc = np.linspace(.2, .8, 10001)
    derivative = np.polynomial.chebyshev.chebder(coefficients)
    slope = 2*np.polynomial.chebyshev.chebval(2*soc-1, derivative)
    if not np.isfinite(slope).all() or np.any(slope <= 0):
        raise ValueError(f"Non-positive OCV slope in the reported range: {dataset}")
    percent = .1/slope
    return {"dataset": dataset, "checkpoint": str(path.relative_to(ROOT)), "ocv_coeffs": coefficients,
            "derivative_chebyshev_coefficients_in_x": derivative.tolist(),
            "grid": {"soc_start": .2, "soc_end": .8, "points": len(soc), "endpoints_included": True},
            "chain_rule": "x=2*SOC-1; dOCV/dSOC=2*chebval(x,chebder(ocv_coeffs))",
            "slope_V_per_unit_soc": {"minimum": float(slope.min()), "median": float(np.median(slope)),
                                     "maximum": float(slope.max())},
            "soc_percent_per_1mV": {"minimum": float(percent.min()), "median": float(np.median(percent)),
                                    "maximum": float(percent.max())}}


def claim_checks(peaks: list[dict], medians: list[dict], slopes: list[dict]) -> list[dict]:
    claims = []

    def claim(identifier: str, label: str, value: float, expected: float, places: int,
              relation: str = "rounded_equal") -> None:
        rounded = round(value, places)
        matches = rounded == expected
        passed = matches if relation == "rounded_equal" else rounded <= expected
        claims.append({"id": identifier, "description": label, "reproduced": value,
                       "article": expected, "decimal_places": places, "rounded_reproduced": rounded,
                       "relation": relation, "passed": passed, "same_rounded_number": matches})

    def limits(prefix: str, label: str, values: list[float], low: float, high: float, places: int) -> None:
        claim(prefix+"_min", label+" minimum", min(values), low, places)
        claim(prefix+"_max", label+" maximum", max(values), high, places)

    limits("peak", "2RC maximum error (mV)", [p["max_error_mV"] for p in peaks], 436, 1032, 0)
    late = [p for p in peaks if p["sample_fraction"] > .89]
    claim("late_peaks", "Peaks after 89% of the profile (conditions)", len(late), 17, 0)
    limits("late_voltage", "Measured voltage at late peaks (V)",
           [p["measured_voltage_V"] for p in late], 2.41, 2.78, 2)
    early = next(p for p in peaks if p["dataset"] == "panasonic" and p["condition"] == "UDDS_-20C")
    claim("early_peak_percent", "NCA UDDS -20 C peak position (%)", 100*early["sample_fraction"], 2, 0)
    for dataset, low, high in (("calce", 39, 66), ("panasonic", 231, 436)):
        values = [p["max_without_last_5_percent_mV"] for p in peaks if p["dataset"] == dataset
                  and (p["condition"].endswith("_25C") or dataset == "calce" and p["condition"].endswith("_45C"))]
        limits("trimmed_"+dataset, f"{dataset} 2RC maximum without last 5% (mV)", values, low, high, 0)
    for metric, neural_bound, nca_bound, nmc_bound in (
        ("rmse", .42, 3.9, .3), ("p99", .73, 12.6, 2.2), ("max_error", .89, 42.6, 2.2),
    ):
        neural, differences = [], {"panasonic": [], "calce": []}
        for row in medians:
            models = row["trained_medians"]
            neural.extend(abs(models[m]["metrics_mV"][metric]-models["physics_gradient"]["metrics_mV"][metric])
                          for m in NEURAL)
            differences[row["dataset"]].extend(models[m]["metrics_mV"][metric]-row["baseline"]["metrics_mV"][metric]
                                               for m in TRAINED)
        claim(metric+"_neural", f"{metric}: max neural versus physics-gradient median difference (mV)",
              max(neural), neural_bound, 2, "rounded_upper_bound")
        claim(metric+"_nca", f"{metric}: largest trained-minus-2RC median on NCA (mV)",
              max(differences["panasonic"]), nca_bound, 1, "rounded_upper_bound")
        claim(metric+"_nmc", f"{metric}: max absolute trained-minus-2RC median on NMC (mV)",
              max(abs(x) for x in differences["calce"]), nmc_bound, 1, "rounded_upper_bound")
    for dataset, condition, mid, low in (("calce", "DST_25C", 2.3, 15.1),
                                         ("panasonic", "HWFET_25C", 20.5, 133.5)):
        row = next(r for r in medians if r["dataset"] == dataset and r["condition"] == condition)
        for band, expected in (("mid", mid), ("low", low)):
            claim(f"{dataset}_{condition}_{band}", f"{dataset} {condition} 2RC {band}-SOC MAE (mV)",
                  row["baseline"]["soc_bands"][band]["mae_mV"], expected, 1)
    neural_bands, trained_bands = [], {"panasonic": [], "calce": []}
    for row in medians:
        for band in BANDS:
            baseline = row["baseline"]["soc_bands"][band]["mae_mV"]
            if baseline is None:
                continue
            models = row["trained_medians"]
            neural_bands.extend(abs(models[m]["soc_band_mae_mV"][band]
                                    - models["physics_gradient"]["soc_band_mae_mV"][band]) for m in NEURAL)
            trained_bands[row["dataset"]].extend(models[m]["soc_band_mae_mV"][band]-baseline for m in TRAINED)
    claim("soc_neural", "All SOC bands: max neural-versus-physics-gradient median difference (mV)",
          max(neural_bands), .62, 2, "rounded_upper_bound")
    claim("soc_nca", "All SOC bands: largest trained-minus-2RC median on NCA (mV)",
          max(trained_bands["panasonic"]), 4.3, 1, "rounded_upper_bound")
    claim("soc_nmc", "All SOC bands: max absolute trained-minus-2RC median on NMC (mV)",
          max(abs(x) for x in trained_bands["calce"]), .5, 1, "rounded_upper_bound")
    for slope, median, maximum in zip(slopes, (.13, .19), (.18, .35)):
        for name, expected in (("median", median), ("maximum", maximum)):
            claim("ocv_"+slope["dataset"]+"_"+name, f'{slope["dataset"]} 1 mV in SOC percentage points ({name})',
                  slope["soc_percent_per_1mV"][name], expected, 2)
    limits("ocv_all", "1 mV in SOC percentage points (both cells)",
           [s["soc_percent_per_1mV"][k] for s in slopes for k in ("minimum", "maximum")], .1, .4, 1)
    return claims


def report_text(analysis: dict) -> str:
    lines = ["# E1 error-structure reproduction", "",
             "Read-only inference from frozen checkpoints; no training, fitting or reidentification.",
             "Run from any directory with the environment in requirements-unified.txt:", "",
             "```sh", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze_e1_error_structure.py",
             "python -m unittest -v test_e1_error_structure", "```", "",
             "Existing output directories are refused; use --output NEW_DIRECTORY for another run.",
             "analysis.json contains individual metrics, SOC counts, empty bands, seed medians, peaks,",
             "OCV derivatives and all claim checks. validation.json records all 576 continuous MAE checks",
             "(18 conditions x 2 deterministic baselines + 18 x 3 trained variants x 10 seeds).",
             "The 1RC baseline is validation-only; LSTM has no continuous prediction in E1.",
             "preservation.json records SHA-256 before/after for every file in both checkpoint trees,",
             "raw inputs, the saved summary and existing inference code. The external manuscript",
             "support scripts are not runtime dependencies of this implementation or its tests.", "",
             "All statistics are computed only after the global MAE gate passes (absolute tolerance 1e-6 mV).",
             "Source masks cover retained validation windows; target masks cover the full cycle.",
             "Peaks use the first maximum on that mask, index/N as in the reference, with elapsed-time",
             "fraction also recorded. The last-5% exclusion is index < floor(0.95*N), intersected with the mask.",
             "SOC bands match the reference: high [0.8,1], mid [0.2,0.8), low [0,0.2).",
             "Errors use float64 subtraction, as protocol.metrics_mV; no rounding precedes aggregation.",
             "Tail and band comparisons are differences of ten-seed medians, not medians of differences.",
             "OCV uses 10001 equally spaced points on [0.2,0.8], both endpoints, and the Chebyshev chain rule.",
             "1 mV / slope is reported in SOC percentage points, not relative percent SOC.", "",
             "## Manuscript checks", "",
             "Upper bounds are checked at the article's precision. A smaller rounded maximum is",
             "listed as a conservative bound, not silently equated to the article number.", "",
             "| ID | Quantity | Article | Reproduced | At article precision | Status |",
             "|---|---|---:|---:|---:|---|"]
    for c in analysis["claims"]:
        status = "matches" if c["same_rounded_number"] else "conservative bound" if c["passed"] else "DIVERGENCE"
        lines.append(f'| {c["id"]} | {c["description"]} | {c["article"]} | {c["reproduced"]:.12g} | '
                     f'{c["rounded_reproduced"]} | {status} |')
    return "\n".join(lines)+"\n"


def reproduce(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    checkpoints_before = checkpoint_snapshot()
    input_paths = [SUMMARY, Path(__file__), ROOT/"requirements.txt",
                   *[ROOT/name for name in ("exp_unified.py", "protocol.py", "unified_dynamics.py", "data.py",
                                            "data_calce.py", "ecm.py", "ecm_2rc.py", "ude.py", "models.py")]]
    inputs_before = {str(p.relative_to(ROOT)): digest(p) for p in input_paths}
    saved = json.loads(SUMMARY.read_text())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with repository_directory(), tempfile.TemporaryDirectory(prefix="e1-errors-") as directory:
            validated, contexts, raw_inputs = collect_validated_errors(saved, Path(directory))
            validation = {
                "tolerance_mV": MAE_TOLERANCE_MV, "all_passed": True, "count": len(validated),
                "maximum_absolute_difference_mV": max(r["validation"]["absolute_difference_mV"] for r in validated),
                "records": [{k: v for k, v in r.items() if k != "error_file"} for r in validated],
            }
            records, peaks, medians = summarize_records(validated, contexts)
            slopes = [ocv_slope(dataset) for dataset in ("panasonic", "calce")]
            conditions = [{k: c[k] for k in ("dataset", "condition", "source", "total_samples", "evaluated_samples")}
                          for c in contexts.values()]
            analysis = {
                "schema_version": 1, "no_training_fitting_or_reidentification": True,
                "datasets": {"panasonic": "NCA", "calce": "NMC"}, "seeds": list(SEEDS),
                "excluded_conditions": ["calce/"+c for c in sorted(EXCLUDED)],
                "conditions": conditions, "records": records, "peaks_2rc": peaks,
                "condition_medians": medians, "ocv_slopes": slopes,
                "claims": claim_checks(peaks, medians, slopes),
                "runtime": {name: importlib.metadata.version(name) for name in
                            ("numpy", "scipy", "torch", "numba", "llvmlite", "torchdiffeq", "openpyxl")},
                "python": platform.python_version(), "platform": platform.platform(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
    finally:
        torch.set_num_threads(threads)
        verify_hashes(checkpoints_before)
        verify_hashes(inputs_before)
        if checkpoint_snapshot() != checkpoints_before:
            raise RuntimeError("Checkpoint tree changed during read-only inference")
    for path, sha in raw_inputs.items():
        if digest(ROOT/path) != sha:
            raise RuntimeError(f"Raw dataset input changed: {path}")
    preservation = {
        "checkpoint_sha256_before": checkpoints_before, "checkpoint_sha256_after": checkpoint_snapshot(),
        "input_sha256_before": inputs_before, "input_sha256_after": {p: digest(ROOT/p) for p in inputs_before},
        "raw_input_sha256": raw_inputs, "no_training_fitting_or_reidentification": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("analysis.json", analysis), ("validation.json", validation),
                            ("preservation.json", preservation)):
        with (output/filename).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    with (output/"README.md").open("x") as stream:
        stream.write(report_text(analysis))
    print(f"Validated {validation['count']} MAEs; wrote new artifacts to {output}", flush=True)
    for c in analysis["claims"]:
        if not c["same_rounded_number"]:
            print(f'CLAIM DIFFERENCE {c["id"]}: article={c["article"]}, '
                  f'reproduced={c["rounded_reproduced"]}, bound_passed={c["passed"]}', flush=True)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    reproduce(args.output)


if __name__ == "__main__":
    main()
