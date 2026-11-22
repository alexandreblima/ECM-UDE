"""Exploratory follow-up analyses for the reviewer round-two results."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import exp_unified as U
import protocol as P
from ecm import ECMParams
from ecm_2rc import ECMParams2RC
from exp_paper_v2_seeds import shifted_wilcoxon
from reviewer_round2_a1 import FrozenCorrection, _nls_params
from reviewer_round2_a2 import MAIN_CONDITIONS, _contexts
from reviewer_round2_b import (
    CFG as HG2_CFG,
    CONDITIONS as HG2_CONDITIONS,
    SOURCE as HG2_SOURCE,
    TARGETS as HG2_TARGETS,
    _fit_nls as _fit_hg2_nls,
    _load_contexts as _load_hg2_contexts,
)
from reviewer_round2_b_audit import DATA_ROOT as HG2_DATA_ROOT, _read_mat as _read_hg2_mat

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "E"
A_OUT = ROOT / "results" / "reviewer_round2" / "A"
B_OUT = ROOT / "results" / "reviewer_round2" / "B"
C_OUT = ROOT / "results" / "reviewer_round2" / "C"
VARIANTS = ("output_residual", "ecm_ude")
EXPERIMENTS = ("a4a", "a4b")
SEEDS = (42, 43, 44, 45, 46)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No records to write to {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_window_mse(model, context: dict, indices: np.ndarray, cfg) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for features, target in U.loader(context, cfg, indices):
            prediction = model(features)
            squared_error += float(torch.sum((prediction - target).double().square()))
            count += target.numel()
    if count == 0:
        raise ValueError("Cannot calculate MSE for an empty source split")
    return squared_error / count


def _nls_window_mse(params: ECMParams, context: dict, indices: np.ndarray, cfg) -> float:
    dataset = context["ds"]
    starts = context["starts"][indices]
    predicted_v = U.nls_predict(params, context["cycle"], cfg, starts)
    length = cfg.window_length
    truth_v = np.stack(
        [dataset._V_raw[int(start) : int(start) + length].numpy() for start in starts]
    ).astype(np.float64)
    stats = context["stats"]
    residual_norm = (predicted_v - truth_v) / stats["V_std"]
    return float(np.mean(np.square(residual_norm)))


def _covered_sample_mask(context: dict, indices: np.ndarray, cfg) -> np.ndarray:
    mask = np.zeros(len(context["cycle"]["V"]), dtype=bool)
    for index in indices:
        start = int(context["starts"][index])
        mask[start : start + cfg.window_length] = True
    return mask


def _continuous_prediction_v(model, context: dict) -> np.ndarray:
    dataset = context["ds"]
    stats = context["stats"]
    model.eval()
    with torch.no_grad():
        normalized = model(dataset.X[None])[0].cpu().numpy()
    return normalized.astype(np.float64) * stats["V_std"] + stats["V_mean"]


def _continuous_nls_prediction_v(params: ECMParams, context: dict, cfg) -> np.ndarray:
    return np.asarray(U.nls_predict(params, context["cycle"], cfg)[0], dtype=np.float64)


def _continuous_mae_mV(prediction_v: np.ndarray, truth_v: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        raise ValueError("Cannot calculate continuous MAE on an empty sample mask")
    return float(np.mean(np.abs(prediction_v[mask] - truth_v[mask])) * 1000.0)


def _run_e1() -> tuple[list[dict], list[dict], dict]:
    summary_path = ROOT / "results" / "unified_10seeds" / "summary.json"
    summary = json.loads(summary_path.read_text())
    metric_rows, reconciliation_rows = [], []
    input_hashes = {"results/unified_10seeds/summary.json": _sha256(summary_path)}

    for dataset, cfg in P.DATASETS.items():
        saved = summary["datasets"][dataset]
        source_name = "UDDS_25C" if dataset == "panasonic" else "FUDS_25C"
        contexts = _contexts(dataset, cfg, saved)
        context = contexts[source_name]
        params, nls_path, nls_hash = _nls_params(dataset, saved)
        input_hashes[str(nls_path.relative_to(ROOT))] = nls_hash
        train_indices = np.asarray(context["tr"], dtype=int)
        validation_indices = np.asarray(context["eval_indices"], dtype=int)
        stats = context["stats"]
        truth_v = context["ds"]._V_raw.numpy().astype(np.float64)
        train_mask = _covered_sample_mask(context, train_indices, cfg)
        validation_mask = _covered_sample_mask(context, validation_indices, cfg)
        nls_train_mse = _nls_window_mse(params, context, train_indices, cfg)
        nls_validation_mse = _nls_window_mse(
            params, context, validation_indices, cfg
        )
        nls_continuous = _continuous_nls_prediction_v(params, context, cfg)
        metric_rows.append(
            {
                "experiment": "NLS",
                "dataset": dataset,
                "variant": "ecm_1rc_nls",
                "seed": "deterministic",
                "selected_epoch": "",
                "train_window_mse_normalized": nls_train_mse,
                "validation_window_mse_normalized": nls_validation_mse,
                "continuous_train_mae_mV": _continuous_mae_mV(
                    nls_continuous, truth_v, train_mask
                ),
                "continuous_validation_mae_mV": _continuous_mae_mV(
                    nls_continuous, truth_v, validation_mask
                ),
                "train_window_count": len(train_indices),
                "validation_window_count": len(validation_indices),
                "train_sample_count": int(train_mask.sum()),
                "validation_sample_count": int(validation_mask.sum()),
                "normalization_V_mean": stats["V_mean"],
                "normalization_V_std": stats["V_std"],
                "checkpoint": str(nls_path.relative_to(ROOT)),
                "checkpoint_sha256": nls_hash,
            }
        )

        for experiment in EXPERIMENTS:
            a4_root = ROOT / "results" / "reviewer_round2" / "A" / experiment
            a4_manifest = json.loads((a4_root / "manifest.json").read_text())
            freeze_audit = json.loads((a4_root / "freeze_audit.json").read_text())
            input_hashes[str((a4_root / "manifest.json").relative_to(ROOT))] = _sha256(
                a4_root / "manifest.json"
            )
            for input_name, record in a4_manifest.get("raw_inputs", {}).items():
                input_hashes[f"{experiment}:raw:{input_name}"] = record["sha256"]
            for input_name, record in a4_manifest.get("nls_warm_starts", {}).items():
                input_hashes[f"{experiment}:nls:{input_name}"] = record["sha256"]
            input_hashes[str((a4_root / "freeze_audit.json").relative_to(ROOT))] = _sha256(
                a4_root / "freeze_audit.json"
            )
            for variant in VARIANTS:
                for seed in SEEDS:
                    checkpoint = (
                        a4_root
                        / "checkpoints"
                        / dataset
                        / f"{variant}_seed{seed}.pt"
                    )
                    relative_checkpoint = str(checkpoint.relative_to(ROOT))
                    checkpoint_hash = _sha256(checkpoint)
                    if a4_manifest["checkpoint_sha256"].get(relative_checkpoint) != checkpoint_hash:
                        raise RuntimeError(f"A4 checkpoint hash mismatch: {relative_checkpoint}")
                    audit = next(
                        row
                        for row in freeze_audit
                        if row["dataset"] == dataset
                        and row["variant"] == variant
                        and int(row["seed"]) == seed
                    )
                    model = FrozenCorrection(
                        U.build_model(variant, stats, params, cfg), cfg
                    )
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=True)
                    model.eval()

                    train_mse = _source_window_mse(
                        model, context, train_indices, cfg
                    )
                    validation_mse = _source_window_mse(
                        model, context, validation_indices, cfg
                    )
                    logged_best_validation = float(
                        audit["history"]["val_loss"][int(audit["best_epoch"]) - 1]
                    )
                    if not np.isclose(
                        validation_mse,
                        logged_best_validation,
                        rtol=1e-5,
                        atol=1e-7,
                    ):
                        raise RuntimeError(
                            "Selected checkpoint validation MSE does not match "
                            f"its logged best epoch: {relative_checkpoint}"
                        )
                    continuous = _continuous_prediction_v(model, context)
                    row = {
                        "experiment": experiment.upper(),
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "selected_epoch": int(audit["best_epoch"]),
                        "last_epoch": int(audit["epochs_completed"]),
                        "last_epoch_train_mse_normalized": float(
                            audit["history"]["train_loss"][-1]
                        ),
                        "best_epoch_train_mse_normalized": float(
                            audit["history"]["train_loss"][int(audit["best_epoch"]) - 1]
                        ),
                        "best_epoch_validation_mse_normalized": float(
                            logged_best_validation
                        ),
                        "train_window_mse_normalized": train_mse,
                        "validation_window_mse_normalized": validation_mse,
                        "continuous_train_mae_mV": _continuous_mae_mV(
                            continuous, truth_v, train_mask
                        ),
                        "continuous_validation_mae_mV": _continuous_mae_mV(
                            continuous, truth_v, validation_mask
                        ),
                        "nls_train_window_mse_normalized": nls_train_mse,
                        "nls_validation_window_mse_normalized": nls_validation_mse,
                        "train_window_count": len(train_indices),
                        "validation_window_count": len(validation_indices),
                        "train_sample_count": int(train_mask.sum()),
                        "validation_sample_count": int(validation_mask.sum()),
                        "normalization_V_mean": stats["V_mean"],
                        "normalization_V_std": stats["V_std"],
                        "checkpoint": relative_checkpoint,
                        "checkpoint_sha256": checkpoint_hash,
                    }
                    metric_rows.append(row)
                    reconciliation_rows.append(
                        {
                            "experiment": experiment.upper(),
                            "dataset": dataset,
                            "variant": variant,
                            "seed": seed,
                            "selected_epoch": row["selected_epoch"],
                            "nls_training_mse_normalized": nls_train_mse,
                            "selected_checkpoint_training_mse_normalized": train_mse,
                            "selected_checkpoint_relative_loss_reduction": (
                                nls_train_mse - train_mse
                            )
                            / nls_train_mse,
                            "last_epoch_training_mse_normalized": row[
                                "last_epoch_train_mse_normalized"
                            ],
                            "last_epoch_relative_loss_reduction": (
                                nls_train_mse
                                - row["last_epoch_train_mse_normalized"]
                            )
                            / nls_train_mse,
                            "recomputed_selected_validation_mse_normalized": validation_mse,
                            "history_best_validation_mse_normalized": row[
                                "best_epoch_validation_mse_normalized"
                            ],
                        }
                    )

    return metric_rows, reconciliation_rows, input_hashes


def run_e1() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows, reconciliation_rows, input_hashes = _run_e1()
    _write_csv(OUT / "a4_checkpoint_metrics.csv", metric_rows)
    _write_csv(OUT / "a4_loss_reconciliation.csv", reconciliation_rows)
    manifest = {
        "analysis": "E1 A4 checkpoint and NLS source-split reconciliation",
        "exploratory": True,
        "retraining": False,
        "input_sha256": input_hashes,
        "source_sha256": {
            name: _sha256(ROOT / name)
            for name in (
                "reviewer_round2_e.py",
                "reviewer_round2_a4.py",
                "reviewer_round2_a1.py",
                "reviewer_round2_a2.py",
                "exp_unified.py",
                "training.py",
                "data.py",
                "protocol.py",
                "unified_dynamics.py",
            )
        },
        "checkpoint_count": sum(row["experiment"] != "NLS" for row in metric_rows),
        "metrics_rows": len(metric_rows),
        "reconciliation_rows": len(reconciliation_rows),
    }
    manifest["output_sha256"] = {
        name: _sha256(OUT / name)
        for name in ("a4_checkpoint_metrics.csv", "a4_loss_reconciliation.csv")
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _sorted_record(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    finite = np.ones(len(arrays["Time"]), dtype=bool)
    for values in arrays.values():
        finite &= np.isfinite(values)
    clean = {key: value[finite] for key, value in arrays.items()}
    order = np.argsort(clean["Time"], kind="stable")
    sorted_arrays = {key: value[order] for key, value in clean.items()}
    _, first = np.unique(sorted_arrays["Time"], return_index=True)
    return {key: value[first] for key, value in sorted_arrays.items()}


def _monotonic_curve(soc: np.ndarray, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(soc, kind="stable")
    soc = np.asarray(soc[order], dtype=float)
    voltage = np.asarray(voltage[order], dtype=float)
    unique_soc, inverse = np.unique(soc, return_inverse=True)
    if len(unique_soc) != len(soc):
        counts = np.bincount(inverse)
        voltage = np.bincount(inverse, weights=voltage) / counts
        soc = unique_soc
    voltage = np.maximum.accumulate(voltage)
    keep = np.r_[True, np.diff(voltage) > 0]
    return soc[keep], voltage[keep]


def _binned_curve(
    soc: np.ndarray,
    voltage: np.ndarray,
    selected: np.ndarray,
    *,
    width: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    centers, medians = [], []
    for lower in np.arange(0.0, 1.0, width):
        mask = selected & (soc >= lower) & (soc < lower + width)
        if mask.any():
            centers.append(float(np.median(soc[mask])))
            medians.append(float(np.median(voltage[mask])))
    if len(centers) < 6:
        raise ValueError("Too few populated SOC bins for an OCV curve")
    return _monotonic_curve(np.asarray(centers), np.asarray(medians))


def _interpolated_curve(
    soc: np.ndarray, voltage: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    if grid.min() < soc.min() or grid.max() > soc.max():
        raise ValueError("OCV comparison grid exceeds curve coverage")
    return np.interp(grid, soc, voltage)


def _ocv_differences(
    baseline_soc: np.ndarray,
    baseline_v: np.ndarray,
    alternative_soc: np.ndarray,
    alternative_v: np.ndarray,
    label: str,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    lower = max(float(baseline_soc.min()), float(alternative_soc.min()))
    upper = min(float(baseline_soc.max()), float(alternative_soc.max()))
    grid = np.arange(max(0.001, lower), min(0.999, upper) + 1e-10, 0.002)
    base = _interpolated_curve(baseline_soc, baseline_v, grid)
    alt = _interpolated_curve(alternative_soc, alternative_v, grid)
    delta_mV = (alt - base) * 1000.0
    rows = []
    for band, mask in (
        ("SOC_0.2_to_0.8", (grid >= 0.2) & (grid <= 0.8)),
        ("SOC_below_0.2", grid < 0.2),
    ):
        values = delta_mV[mask]
        if not len(values):
            raise ValueError(f"No common OCV points in {band} for {label}")
        rows.append(
            {
                "comparison": label,
                "soc_band": band,
                "grid_points": int(len(values)),
                "mean_signed_difference_mV": float(np.mean(values)),
                "median_signed_difference_mV": float(np.median(values)),
                "mean_absolute_difference_mV": float(np.mean(np.abs(values))),
                "minimum_difference_mV": float(np.min(values)),
                "maximum_difference_mV": float(np.max(values)),
            }
        )
    return rows, grid, base, alt


def _hpcc_rest_ocv() -> tuple[np.ndarray, np.ndarray, dict]:
    matches = sorted(HG2_DATA_ROOT.glob("25degC/*549_HPPC*25degC*LGHG2.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one 25 C HPPC MAT, found {len(matches)}")
    record = _sorted_record(_read_hg2_mat(matches[0]))
    time, current, voltage = (
        record["Time"],
        record["Current"],
        record["Voltage"],
    )
    q_capacity = 2.72639
    soc = 1.0 + np.r_[
        0.0,
        np.cumsum(
            (current[1:] + current[:-1])
            * 0.5
            * np.diff(time)
            / (q_capacity * 3600.0)
        ),
    ]
    rest = np.abs(current) <= 0.02
    edges = np.flatnonzero(np.diff(np.r_[False, rest, False]))
    rest_soc, rest_voltage, rest_duration = [], [], []
    for start, end in zip(edges[::2], edges[1::2]):
        duration = float(time[end - 1] - time[start])
        if duration < 600.0:
            continue
        tail = time[start:end] >= time[end - 1] - 30.0
        rest_soc.append(float(np.median(soc[start:end][tail])))
        rest_voltage.append(float(np.median(voltage[start:end][tail])))
        rest_duration.append(duration)
    if len(rest_soc) < 6:
        raise ValueError("HPPC has too few >=600 s rest periods for an OCV estimate")
    hpcc_soc, hpcc_v = _binned_curve(
        np.asarray(rest_soc),
        np.asarray(rest_voltage),
        np.ones(len(rest_soc), dtype=bool),
    )
    return hpcc_soc, hpcc_v, {
        "path": str(matches[0].relative_to(HG2_DATA_ROOT)),
        "sha256": _sha256(matches[0]),
        "rest_threshold_A": 0.02,
        "minimum_rest_duration_s": 600.0,
        "voltage_window_s": 30.0,
        "rest_period_count": len(rest_soc),
        "rest_duration_min_s": float(min(rest_duration)),
        "rest_duration_max_s": float(max(rest_duration)),
        "soc_coverage": [float(hpcc_soc.min()), float(hpcc_soc.max())],
    }


def _c20_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    matches = sorted(HG2_DATA_ROOT.glob("25degC/*549_C20DisCh*25degC*LGHG2.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one 25 C C/20 MAT, found {len(matches)}")
    path = matches[0]
    record = _sorted_record(_read_hg2_mat(path))
    time, current, voltage = record["Time"], record["Current"], record["Voltage"]
    q_capacity = 2.72639
    soc = 1.0 + np.r_[
        0.0,
        np.cumsum(
            (current[1:] + current[:-1])
            * 0.5
            * np.diff(time)
            / (q_capacity * 3600.0)
        ),
    ]
    discharge = _binned_curve(soc, voltage, current < -0.05)
    charge = _binned_curve(soc, voltage, current > 0.05)
    common_lower = max(float(discharge[0].min()), float(charge[0].min()))
    common_upper = min(float(discharge[0].max()), float(charge[0].max()))
    grid = np.arange(common_lower, common_upper + 1e-10, 0.002)
    discharge_v = _interpolated_curve(discharge[0], discharge[1], grid)
    charge_v = _interpolated_curve(charge[0], charge[1], grid)
    mean_v = (discharge_v + charge_v) / 2.0
    info = {
        "path": str(path.relative_to(HG2_DATA_ROOT)),
        "sha256": _sha256(path),
        "samples": len(time),
        "discharge_samples": int(np.count_nonzero(current < -0.05)),
        "charge_samples": int(np.count_nonzero(current > 0.05)),
        "discharge_soc_coverage": [float(discharge[0].min()), float(discharge[0].max())],
        "charge_soc_coverage": [float(charge[0].min()), float(charge[0].max())],
        "mean_curve_soc_coverage": [float(grid.min()), float(grid.max())],
    }
    return discharge[0], discharge[1], grid, mean_v, info


def _run_e2() -> tuple[list[dict], list[dict], list[dict], dict]:
    contexts, input_hashes, audit = _load_hg2_contexts()
    discharge_soc, discharge_v, mean_soc, mean_v, c20_info = _c20_curves()
    hpcc_soc, hpcc_v, hpcc_info = _hpcc_rest_ocv()
    comparison_rows = []
    curve_rows = []

    c20_rows, mean_grid, discharge_interp, mean_interp = _ocv_differences(
        discharge_soc,
        discharge_v,
        mean_soc,
        mean_v,
        "C20_mean_charge_discharge_minus_discharge",
    )
    comparison_rows.extend(c20_rows)
    hpcc_rows, hpcc_grid, hpcc_discharge, hpcc_alt = _ocv_differences(
        discharge_soc,
        discharge_v,
        hpcc_soc,
        hpcc_v,
        "HPPC_rest_minus_C20_discharge",
    )
    comparison_rows.extend(hpcc_rows)
    for soc, discharge in zip(mean_grid, discharge_interp):
        curve_rows.append(
            {
                "comparison": "C20",
                "soc": float(soc),
                "discharge_ocv_V": float(discharge),
                "alternative_ocv_V": float(mean_interp[np.argmin(np.abs(mean_grid - soc))]),
            }
        )
    for soc, discharge, alternative in zip(hpcc_grid, hpcc_discharge, hpcc_alt):
        curve_rows.append(
            {
                "comparison": "HPPC",
                "soc": float(soc),
                "discharge_ocv_V": float(discharge),
                "alternative_ocv_V": float(alternative),
            }
        )

    source = contexts[HG2_SOURCE]
    measured = source["cycle"]["V"].numpy().astype(np.float64)
    time_s = source["cycle"]["t"].numpy().astype(np.float64)
    soc = source["cycle"]["SoC"].numpy().astype(np.float64)
    current = source["cycle"]["I"].numpy().astype(np.float64)
    capacity_ah = float(audit["capacity_ah"])
    net_discharged_ah = float(-np.trapezoid(current, time_s) / 3600.0)
    discharge_ah = float(np.trapezoid(np.maximum(-current, 0.0), time_s) / 3600.0)
    charge_ah = float(np.trapezoid(np.maximum(current, 0.0), time_s) / 3600.0)
    calculated_final_soc = float(np.clip(1.0 - net_discharged_ah / capacity_ah, 0, 1))
    charge_anchor = audit["charge_verification"][HG2_SOURCE]

    nls_info_path = B_OUT / "base" / "ecm_2rc_nls.json"
    nls_info = json.loads(nls_info_path.read_text())
    nls_params = ECMParams2RC.from_dict(nls_info["params"])
    nls_hash = _sha256(nls_info_path)
    nls_pred = _continuous_nls_prediction_v(nls_params, source, HG2_CFG)
    residual_mV = (nls_pred - measured) * 1000.0
    source_validation_mask = source["mask"]
    if not source_validation_mask.any():
        raise ValueError("HG2 source validation mask is empty")
    soc_bands = (
        ("SOC_above_0.8_initial", soc > 0.8),
        ("SOC_0.2_to_0.8_plateau", (soc >= 0.2) & (soc <= 0.8)),
        ("SOC_below_0.2_end_discharge", soc < 0.2),
    )
    error_segments = []
    for label, band in soc_bands:
        selected = band & source_validation_mask
        if selected.any():
            error_segments.append(
                {
                    "segment": label,
                    "samples": int(selected.sum()),
                    "soc_min": float(np.min(soc[selected])),
                    "soc_max": float(np.max(soc[selected])),
                    "mae_mV": float(np.mean(np.abs(residual_mV[selected]))),
                    "mean_signed_error_mV": float(np.mean(residual_mV[selected])),
                }
            )
    plot_path = OUT / "hg2_udds_calibration.pdf"
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.8), sharex=True)
    stride = max(1, len(time_s) // 6000)
    hours = time_s[::stride] / 3600.0
    axes[0].plot(hours, measured[::stride], color="black", linewidth=0.7, label="Measured")
    axes[0].plot(hours, nls_pred[::stride], color="0.45", linewidth=0.75, linestyle="--", label="ECM-2RC NLS")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].plot(hours, soc[::stride], color="0.15", linewidth=0.8)
    axes[1].set_ylabel("SOC")
    axes[2].plot(hours, residual_mV[::stride], color="0.15", linewidth=0.65)
    axes[2].axhline(0.0, color="0.5", linewidth=0.55)
    axes[2].set_ylabel("Error (mV)")
    axes[2].set_xlabel("Time (h)")
    for axis in axes:
        axis.grid(True, color="0.88", linewidth=0.45)
        axis.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    ocv_mean_params = np.polynomial.chebyshev.chebfit(
        2.0 * mean_soc - 1.0, mean_v, 5
    )
    alternate_params, alternate_fit_info = _fit_hg2_nls(
        source, 2, ocv_mean_params
    )
    alternate_fit_info["ocv_initialization"] = (
        "Mean of matched 0.002-SOC median C/20 charge/discharge curves; "
        "degree-5 Chebyshev fit"
    )
    original_params = nls_params.to_dict()
    alternative_parameter_dict = alternate_params.to_dict()
    ocv_coefficients_max_abs_change = float(
        np.max(
            np.abs(
                np.asarray(alternative_parameter_dict["ocv_coeffs"], dtype=float)
                - np.asarray(original_params["ocv_coeffs"], dtype=float)
            )
        )
    )
    physical_parameter_max_abs_change = max(
        abs(
            float(alternative_parameter_dict[key])
            - float(original_params[key])
        )
        for key in ("R0", "R1", "C1", "R2", "C2")
    )
    baseline_metrics = _read_csv(B_OUT / "base" / "metrics_by_seed.csv")
    baseline_by_model_condition = {}
    for condition in HG2_CONDITIONS:
        for model_name in (
            "ecm_2rc_nls",
            "physics_gradient",
            "output_residual",
            "ecm_ude",
        ):
            vals = [
                float(row["mae_mV"])
                for row in baseline_metrics
                if row["condition"] == condition
                and row["model"] == model_name
                and row["evaluation"] == "continuous"
            ]
            baseline_by_model_condition[(condition, model_name)] = float(
                np.median(vals)
            )

    sensitivity_rows, order_changes = [], []
    for condition in HG2_CONDITIONS:
        alt_metrics = U.evaluate(
            alternate_params, "ecm_2rc_nls", contexts[condition], HG2_CFG
        )["continuous"]
        alternate_mae = float(alt_metrics["mae"])
        baseline_order = sorted(
            (
                (name, baseline_by_model_condition[(condition, name)])
                for name in (
                    "ecm_2rc_nls",
                    "physics_gradient",
                    "output_residual",
                    "ecm_ude",
                )
            ),
            key=lambda item: item[1],
        )
        sensitivity_order = sorted(
            [
                (name, alternate_mae if name == "ecm_2rc_nls" else value)
                for name, value in baseline_order
            ],
            key=lambda item: item[1],
        )
        baseline_names = [name for name, _ in baseline_order]
        sensitivity_names = [name for name, _ in sensitivity_order]
        changed = baseline_names != sensitivity_names
        if changed:
            order_changes.append(condition)
        sensitivity_rows.append(
            {
                "condition": condition,
                "alternative_ecm_2rc_mean_ocv_mae_mV": alternate_mae,
                "original_ecm_2rc_discharge_ocv_mae_mV": baseline_by_model_condition[
                    (condition, "ecm_2rc_nls")
                ],
                "physics_gradient_median_mae_mV": baseline_by_model_condition[
                    (condition, "physics_gradient")
                ],
                "output_residual_median_mae_mV": baseline_by_model_condition[
                    (condition, "output_residual")
                ],
                "ecm_ude_median_mae_mV": baseline_by_model_condition[
                    (condition, "ecm_ude")
                ],
                "baseline_model_order": " < ".join(baseline_names),
                "mean_ocv_model_order": " < ".join(sensitivity_names),
                "order_changed": changed,
            }
        )

    info = {
        "exploratory": True,
        "retraining_neural_models": False,
        "ocv_comparison_method": (
            "C/20 branch medians in 0.002-SOC bins; monotonic upper envelope; "
            "mean OCV is the pointwise mean on common SOC support. HPPC uses "
            "median voltage and SOC in the last 30 s of each >=600 s rest "
            "with |current|<=0.02 A."
        ),
        "c20": c20_info,
        "hppc": hpcc_info,
        "capacity_ah": capacity_ah,
        "udds_source": {
            "z0": 1.0,
            "charge_anchor": charge_anchor,
            "net_discharged_ah": net_discharged_ah,
            "discharge_throughput_ah": discharge_ah,
            "charge_throughput_ah": charge_ah,
            "calculated_final_soc": calculated_final_soc,
            "soc_samples": int(len(soc)),
            "duration_s": float(time_s[-1] - time_s[0]),
            "initial_voltage_V": float(measured[0]),
            "final_voltage_V": float(measured[-1]),
            "continuous_mae_mV": float(np.mean(np.abs(residual_mV))),
            "validation_mae_mV": float(
                np.mean(np.abs(residual_mV[source_validation_mask]))
            ),
            "error_by_soc_segment_validation": error_segments,
        },
        "nls_original_checkpoint": str(nls_info_path.relative_to(ROOT)),
        "nls_original_sha256": nls_hash,
        "mean_ocv_2rc_fit": alternate_fit_info,
        "mean_ocv_2rc_params": alternate_params.to_dict(),
        "fit_parameter_comparison": {
            "maximum_absolute_ocv_coefficient_change": ocv_coefficients_max_abs_change,
            "maximum_absolute_R_C_parameter_change": physical_parameter_max_abs_change,
            "original_training_soc_min": float(nls_params.soc_train_min),
            "original_training_soc_max": float(nls_params.soc_train_max),
            "interpretation": (
                "The alternate initialization converged to practically the "
                "same fitted 2RC parameters; the NLS identification windows "
                "do not cover SOC below the recorded training minimum."
            ),
        },
        "model_order_changes": order_changes,
        "sensitivity_order_changed_conditions": len(order_changes),
        "raw_inputs": input_hashes,
    }
    (OUT / "hg2_calibration.json").write_text(json.dumps(info, indent=2) + "\n")
    return comparison_rows, curve_rows, sensitivity_rows, info


def _write_e2_outputs(
    comparison_rows: list[dict],
    curve_rows: list[dict],
    sensitivity_rows: list[dict],
) -> None:
    _write_csv(OUT / "hg2_ocv_comparison.csv", comparison_rows)
    _write_csv(OUT / "hg2_ocv_curves.csv", curve_rows)
    _write_csv(OUT / "hg2_nls_mean_ocv_sensitivity.csv", sensitivity_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True)
    for axis, comparison, label in (
        (axes[0], "C20", "C/20 charge/discharge mean"),
        (axes[1], "HPPC", "HPPC rest estimate"),
    ):
        selected = [row for row in curve_rows if row["comparison"] == comparison]
        if not selected:
            raise ValueError(f"No {comparison} OCV curve points")
        soc = np.asarray([float(row["soc"]) for row in selected])
        discharge = np.asarray(
            [float(row["discharge_ocv_V"]) for row in selected]
        )
        alternative = np.asarray(
            [float(row["alternative_ocv_V"]) for row in selected]
        )
        axis.plot(soc, discharge, color="black", linewidth=1.0, label="C/20 discharge")
        axis.plot(
            soc,
            alternative,
            color="0.45",
            linewidth=1.0,
            linestyle="--",
            label=label,
        )
        axis.set_xlabel("SOC")
        axis.grid(True, color="0.88", linewidth=0.45)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("OCV estimate (V)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "hg2_ocv_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _inferential_row(
    *,
    cohort: str,
    configuration: str,
    variant: str,
    reference: str,
    condition: str,
    differences: np.ndarray,
) -> dict:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError(f"Invalid paired differences: {cohort}/{condition}")
    superiority = shifted_wilcoxon(differences, "less")
    inferiority = shifted_wilcoxon(differences, "greater")
    lower = shifted_wilcoxon(differences + 1.0, "greater")
    upper = shifted_wilcoxon(differences - 1.0, "less")
    return {
        "cohort": cohort,
        "configuration": configuration,
        "variant": variant,
        "reference": reference,
        "condition": condition,
        "n": int(len(differences)),
        "median_difference_mV": float(np.median(differences)),
        "minimum_difference_mV": float(np.min(differences)),
        "maximum_difference_mV": float(np.max(differences)),
        "n_difference_below_0": int(np.count_nonzero(differences < 0)),
        "n_difference_below_minus1": int(np.count_nonzero(differences < -1.0)),
        "p_superiority_raw": float(superiority["p_raw"]),
        "p_inferiority_raw": float(inferiority["p_raw"]),
        "p_tost_lower_raw": float(lower["p_raw"]),
        "p_tost_upper_raw": float(upper["p_raw"]),
        "p_tost_raw": max(float(lower["p_raw"]), float(upper["p_raw"])),
        "holm_family_n": "",
        "test_level": "condition",
    }


def _adjust_family(
    rows: list[dict], family_keys: tuple[str, ...], p_field: str
) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in family_keys)
        groups.setdefault(key, []).append(row)
    adjusted_field = p_field.removesuffix("_raw") + "_holm"
    for group in groups.values():
        records = [{"p_raw": row[p_field], "_row": row} for row in group]
        U.holm(records)
        for record in records:
            record["_row"][adjusted_field] = record["p_holm"]
            record["_row"]["holm_family_n"] = len(group)


def _adjust_test_rows(rows: list[dict], family_keys: tuple[str, ...]) -> None:
    for p_field in (
        "p_superiority_raw",
        "p_inferiority_raw",
        "p_tost_lower_raw",
        "p_tost_upper_raw",
    ):
        _adjust_family(rows, family_keys, p_field)
    for row in rows:
        row["p_tost_holm"] = max(
            row["p_tost_lower_holm"], row["p_tost_upper_holm"]
        )


def _run_e3_and_original_synthesis() -> tuple[list[dict], list[dict], list[dict]]:
    test_rows: list[dict] = []
    m50t_details = []

    c_path = C_OUT / "run" / "cell_differences.csv"
    for row in _read_csv(c_path):
        differences = np.asarray(json.loads(row["seed_differences_mV"]), dtype=float)
        m50t_details.append(
            {
                "cohort": "M50T",
                "cell": row["cell"],
                "variant": row["variant"],
                "reference": row["reference"],
                "seed_differences_mV": json.dumps(differences.tolist()),
                "d_c_mV": float(np.median(differences)),
            }
        )
    for reference in ("physics_gradient", "ecm_2rc_nls"):
        for variant in VARIANTS:
            cells = [
                row
                for row in m50t_details
                if row["variant"] == variant and row["reference"] == reference
            ]
            d_cells = np.asarray([row["d_c_mV"] for row in cells], dtype=float)
            test_rows.append(
                _inferential_row(
                    cohort="M50T",
                    configuration="M50T",
                    variant=variant,
                    reference=reference,
                    condition="all_valid_cells",
                    differences=d_cells,
                )
            )
            test_rows[-1]["test_level"] = "cell"
            test_rows[-1]["n_cells_d_below_0"] = int(np.count_nonzero(d_cells < 0))
            test_rows[-1]["n_cells_d_below_minus1"] = int(
                np.count_nonzero(d_cells < -1.0)
            )
            test_rows[-1]["n_cells_d_above_0"] = int(np.count_nonzero(d_cells > 0))
            test_rows[-1]["n_cells_within_pm1"] = int(
                np.count_nonzero(np.abs(d_cells) <= 1.0)
            )
            test_rows[-1]["cell_differences_mV"] = json.dumps(d_cells.tolist())
    _adjust_test_rows(test_rows, ("reference",))

    hg2_metrics = _read_csv(B_OUT / "base" / "metrics_by_seed.csv")
    for reference in ("physics_gradient", "ecm_2rc_nls"):
        for variant in VARIANTS:
            for condition in HG2_CONDITIONS:
                neural = [
                    row
                    for row in hg2_metrics
                    if row["condition"] == condition
                    and row["model"] == variant
                    and row["evaluation"] == "continuous"
                ]
                reference_rows = [
                    row
                    for row in hg2_metrics
                    if row["condition"] == condition
                    and row["model"] == reference
                    and row["evaluation"] == "continuous"
                ]
                neural.sort(key=lambda row: int(row["seed"]))
                if reference == "physics_gradient":
                    reference_rows.sort(key=lambda row: int(row["seed"]))
                    if [int(row["seed"]) for row in neural] != [
                        int(row["seed"]) for row in reference_rows
                    ]:
                        raise ValueError(f"Unpaired HG2 seeds for {condition}")
                    ref_values = np.asarray(
                        [float(row["mae_mV"]) for row in reference_rows]
                    )
                else:
                    if len(reference_rows) != 1:
                        raise ValueError(
                            f"HG2 deterministic ECM-2RC missing for {condition}"
                        )
                    ref_values = np.repeat(
                        float(reference_rows[0]["mae_mV"]), len(neural)
                    )
                deltas = np.asarray(
                    [float(row["mae_mV"]) for row in neural]
                ) - ref_values
                test_rows.append(
                    _inferential_row(
                        cohort="HG2",
                        configuration="HG2",
                        variant=variant,
                        reference=reference,
                        condition=condition,
                        differences=deltas,
                    )
                )
    _adjust_test_rows(
        [row for row in test_rows if row["cohort"] == "HG2"],
        ("variant", "reference"),
    )

    summary = json.loads(
        (ROOT / "results" / "unified_10seeds" / "summary.json").read_text()
    )
    original_rows = []
    for dataset, cfg in P.DATASETS.items():
        saved = summary["datasets"][dataset]
        conditions = MAIN_CONDITIONS[dataset]
        for variant in VARIANTS:
            for reference in ("physics_gradient", "ecm_2rc_nls"):
                for condition in conditions:
                    neural_values = np.asarray(
                        [
                            saved["runs"][variant][str(seed)]["metrics"][condition][
                                "continuous"
                            ]["mae"]
                            for seed in range(42, 52)
                        ],
                        dtype=float,
                    )
                    if reference == "physics_gradient":
                        ref_values = np.asarray(
                            [
                                saved["runs"][reference][str(seed)]["metrics"][
                                    condition
                                ]["continuous"]["mae"]
                                for seed in range(42, 52)
                            ],
                            dtype=float,
                        )
                    else:
                        ref_value = saved["runs"][reference]["deterministic"][
                            "metrics"
                        ][condition]["continuous"]["mae"]
                        ref_values = np.repeat(float(ref_value), 10)
                    original_rows.append(
                        _inferential_row(
                            cohort="E1_original",
                            configuration=dataset,
                            variant=variant,
                            reference=reference,
                            condition=condition,
                            differences=neural_values - ref_values,
                        )
                    )
    _adjust_test_rows(
        original_rows,
        ("variant", "reference"),
    )

    a4_rows = []
    for experiment in ("a4a", "a4b"):
        a4_path = ROOT / "results" / "reviewer_round2" / "A" / experiment / "metrics_by_seed.csv"
        a4_metrics = _read_csv(a4_path)
        for dataset, conditions in MAIN_CONDITIONS.items():
            saved = summary["datasets"][dataset]
            for variant in VARIANTS:
                for reference in ("physics_gradient", "ecm_2rc_nls"):
                    for condition in conditions:
                        neural = [
                            row
                            for row in a4_metrics
                            if row["dataset"] == dataset
                            and row["variant"] == variant
                            and row["condition"] == condition
                        ]
                        neural.sort(key=lambda row: int(row["seed"]))
                        if [int(row["seed"]) for row in neural] != list(SEEDS):
                            raise ValueError(
                                f"A4 seed mismatch for {experiment}/{dataset}/{condition}"
                            )
                        neural_values = np.asarray(
                            [float(row["mae_mV"]) for row in neural]
                        )
                        if reference == "physics_gradient":
                            reference_values = np.asarray(
                                [
                                    saved["runs"][reference][str(seed)]["metrics"][
                                        condition
                                    ]["continuous"]["mae"]
                                    for seed in SEEDS
                                ],
                                dtype=float,
                            )
                        else:
                            deterministic = saved["runs"][reference]["deterministic"][
                                "metrics"
                            ][condition]["continuous"]["mae"]
                            reference_values = np.repeat(
                                float(deterministic), len(SEEDS)
                            )
                        a4_rows.append(
                            _inferential_row(
                                cohort=experiment.upper(),
                                configuration=dataset,
                                variant=variant,
                                reference=reference,
                                condition=condition,
                                differences=neural_values - reference_values,
                            )
                        )
        _adjust_test_rows(
            [row for row in a4_rows if row["cohort"] == experiment.upper()],
            ("variant", "reference"),
        )

    all_rows = test_rows + original_rows + a4_rows
    return all_rows, m50t_details, original_rows


def _synthesis_rows(test_rows: list[dict], m50t_details: list[dict]) -> list[dict]:
    specifications = [
        ("Protocolo original — NCA", "E1_original", "panasonic", "preregistered metrics; exploratory superiority"),
        ("Protocolo original — NMC Samsung", "E1_original", "calce", "preregistered metrics; exploratory superiority"),
        ("Protocolo original — LG HG2", "HG2", "HG2", "preregistered TOST; exploratory superiority"),
        ("Protocolo original — LG M50T", "M50T", "M50T", "preregistered TOST; exploratory superiority"),
        ("A4b — NCA", "A4B", "panasonic", "preregistered TOST vs ECM-2RC; exploratory vs physics-gradient"),
        ("A4b — NMC Samsung", "A4B", "calce", "preregistered TOST vs ECM-2RC; exploratory vs physics-gradient"),
    ]
    output = []
    for label, cohort, config, status in specifications:
        for variant in VARIANTS:
            row = {
                "configuration": label,
                "protocol_status": status,
                "variant": variant,
                "analysis_unit_n": 7 if cohort == "M50T" else "",
            }
            for baseline, reference in (
                ("physics_gradient", "physics_gradient"),
                ("ecm_2rc", "ecm_2rc_nls"),
            ):
                if cohort == "M50T":
                    selected = [
                        item
                        for item in test_rows
                        if item["cohort"] == "M50T"
                        and item["variant"] == variant
                        and item["reference"] == reference
                    ]
                else:
                    selected = [
                        item
                        for item in test_rows
                        if item["cohort"] == cohort
                        and item["configuration"] == config
                        and item["variant"] == variant
                        and item["reference"] == reference
                    ]
                if not selected:
                    raise ValueError(
                        f"Missing synthesis inputs: {label}/{variant}/{reference}"
                    )
                if cohort == "A4B" and reference == "physics_gradient":
                    row["vs_pg_status"] = "exploratory TOST and superiority"
                elif reference == "ecm_2rc_nls":
                    row["vs_ecm2_status"] = (
                        "preregistered TOST; exploratory superiority"
                    )
                else:
                    row["vs_pg_status"] = (
                        "preregistered TOST; exploratory superiority"
                    )
                if cohort == "M50T":
                    cell_rows = [item for item in m50t_details
                                 if item["variant"] == variant and item["reference"] == reference]
                    if len(cell_rows) != 7 or len({item["cell"] for item in cell_rows}) != 7:
                        raise ValueError(f"Expected seven M50T cell medians: {variant}/{reference}")
                    deltas = [float(item["d_c_mV"]) for item in cell_rows]
                else:
                    deltas = [float(item["median_difference_mV"]) for item in selected]
                prefix = "vs_pg" if baseline == "physics_gradient" else "vs_ecm2"
                row[f"{prefix}_n_tests"] = len(selected)
                row[f"{prefix}_better_holm_count"] = sum(
                    float(item["p_superiority_holm"]) < 0.05 for item in selected
                )
                row[f"{prefix}_equivalent_holm_count"] = sum(
                    float(item["p_tost_holm"]) < 0.05 for item in selected
                )
                row[f"{prefix}_worse_holm_count"] = sum(
                    float(item["p_inferiority_holm"]) < 0.05 for item in selected
                )
                row[f"{prefix}_unresolved_holm_count"] = sum(
                    float(item["p_superiority_holm"]) >= 0.05
                    and float(item["p_tost_holm"]) >= 0.05
                    and float(item["p_inferiority_holm"]) >= 0.05
                    for item in selected
                )
                row[f"{prefix}_median_difference_mV"] = float(np.median(deltas))
                row[f"{prefix}_minimum_difference_mV"] = float(np.min(deltas))
                row[f"{prefix}_maximum_difference_mV"] = float(np.max(deltas))
                row[f"{prefix}_p_superiority_holm"] = min(
                    float(item["p_superiority_holm"]) for item in selected
                )
                row[f"{prefix}_p_tost_holm"] = min(
                    float(item["p_tost_holm"]) for item in selected
                )
                row[f"{prefix}_p_inferiority_holm"] = min(
                    float(item["p_inferiority_holm"]) for item in selected
                )
                if cohort == "M50T":
                    row[f"{prefix}_cells_d_below_0"] = selected[0][
                        "n_cells_d_below_0"
                    ]
                    row[f"{prefix}_cells_d_below_minus1"] = selected[0][
                        "n_cells_d_below_minus1"
                    ]
                    row[f"{prefix}_cells_d_above_0"] = selected[0][
                        "n_cells_d_above_0"
                    ]
                    row[f"{prefix}_cells_within_pm1"] = selected[0][
                        "n_cells_within_pm1"
                    ]
            output.append(row)
    return output


def _corrected_a4_summary(reconciliation_rows: list[dict]) -> list[dict]:
    result = []
    for experiment in ("A4A", "A4B"):
        for dataset in P.DATASETS:
            for variant in VARIANTS:
                selected = [
                    row
                    for row in reconciliation_rows
                    if row["experiment"] == experiment
                    and row["dataset"] == dataset
                    and row["variant"] == variant
                ]
                if len(selected) != len(SEEDS):
                    raise ValueError(f"Incomplete A4 summary: {experiment}/{dataset}/{variant}")
                checkpoint_reductions = [
                    float(row["selected_checkpoint_relative_loss_reduction"])
                    for row in selected
                ]
                online_reductions = [
                    float(row["last_epoch_relative_loss_reduction"])
                    for row in selected
                ]
                result.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "variant": variant,
                        "n": len(selected),
                        "loss_basis": "selected_best_validation_checkpoint; fixed train-window evaluation",
                        "median_relative_loss_reduction": statistics.median(
                            checkpoint_reductions
                        ),
                        "min_relative_loss_reduction": min(checkpoint_reductions),
                        "max_relative_loss_reduction": max(checkpoint_reductions),
                        "median_last_epoch_relative_loss_reduction": statistics.median(
                            online_reductions
                        ),
                        "min_last_epoch_relative_loss_reduction": min(online_reductions),
                        "max_last_epoch_relative_loss_reduction": max(online_reductions),
                    }
                )
    return result


def _write_synthesis_tex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Exploratory synthesis of registered errors and one-sided superiority tests.}",
        r"\label{tab:reviewer-round2-synthesis}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Configuration & Variant & Status & vs. physics-gradient & vs. ECM-2RC \\",
        r"\midrule",
    ]
    for row in rows:
        pg_status = "EX" if row["vs_pg_status"].startswith("exploratory") else "PR+EX"
        ecm_status = "PR+EX"
        status_short = f"PG {pg_status}; NLS {ecm_status}"
        pg_counts = (
            f'{row["vs_pg_better_holm_count"]}/'
            f'{row["vs_pg_equivalent_holm_count"]}/'
            f'{row["vs_pg_worse_holm_count"]}'
        )
        ecm_counts = (
            f'{row["vs_ecm2_better_holm_count"]}/'
            f'{row["vs_ecm2_equivalent_holm_count"]}/'
            f'{row["vs_ecm2_worse_holm_count"]}'
        )
        pg_delta = (
            f'{row["vs_pg_median_difference_mV"]:.3f} '
            f'[{row["vs_pg_minimum_difference_mV"]:.3f},'
            f'{row["vs_pg_maximum_difference_mV"]:.3f}]'
        )
        ecm_delta = (
            f'{row["vs_ecm2_median_difference_mV"]:.3f} '
            f'[{row["vs_ecm2_minimum_difference_mV"]:.3f},'
            f'{row["vs_ecm2_maximum_difference_mV"]:.3f}]'
        )
        lines.append(
            f'{row["configuration"]} & {row["variant"]} & {status_short} & '
            f'{pg_counts}; {pg_delta} & {ecm_counts}; {ecm_delta} \\\\'
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize Counts are superior/equivalent/inferior tests with Holm-adjusted $p<0.05$; the tests are not mutually exclusive. Differences are medians across condition-level medians and their min--max range (mV). M50T counts refer to the single population-level contrast over seven cells; descriptive cell directions are reported separately. PR: preregistered TOST; EX: exploratory superiority.}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _report_e1(
    metric_rows: list[dict], reconciliation_rows: list[dict], summary_rows: list[dict]
) -> list[str]:
    lines = [
        "## E1 — A4 reconciliation (exploratory)",
        "",
        "No network was retrained. The old history is not a fixed checkpoint evaluation: `training.py` computes each mini-batch loss during optimizer updates, aggregates those losses over the epoch, and finally restores the checkpoint with the lowest validation loss. Thus, `history['train_loss'][-1]` describes the average trajectory of the last epoch, not the selected checkpoint. NLS was evaluated once on all the same training windows, making the recorded comparison of relative reductions incompatible.",
        "",
        "For the four recalculated metrics, each window resets RC states to zero and initializes SOC to that window's value; current, temperature, SOC, and voltage are normalized using statistics computed only over the union of training-window samples. Loss is the mean pointwise MSE between predicted and target voltages normalized by `V_std`. NLS is evaluated statically on the same windows with the same normalization. For continuous simulation, each checkpoint and NLS traverse the source once with RC state carried between samples; training and validation MAE use sample masks covered by their respective windows, excluding the guard interval.",
        "",
        "The corrected column in `A/a4_training_loss_summary.csv` uses the saved checkpoint MSE, not the mean loss during the last epoch. The previous last-epoch loss was retained in separate columns for traceability. Values by seed, selected epochs, metrics, and hashes are in `a4_checkpoint_metrics.csv` and `a4_loss_reconciliation.csv`.",
        "",
        "| Run | Cell | Variant | Median training MSE reduction at checkpoint (%) | Continuous training ΔMAE vs. NLS (mV) | Continuous validation ΔMAE vs. NLS (mV) |",
        "|---|---|---|---:|---:|---:|",
    ]
    nls = {
        row["dataset"]: row
        for row in metric_rows
        if row["experiment"] == "NLS"
    }
    for row in summary_rows:
        selected = [
            item
            for item in metric_rows
            if item["experiment"] == row["experiment"]
            and item["dataset"] == row["dataset"]
            and item["variant"] == row["variant"]
        ]
        reference = nls[row["dataset"]]
        train_delta = statistics.median(
            float(item["continuous_train_mae_mV"])
            - float(reference["continuous_train_mae_mV"])
            for item in selected
        )
        validation_delta = statistics.median(
            float(item["continuous_validation_mae_mV"])
            - float(reference["continuous_validation_mae_mV"])
            for item in selected
        )
        cell_name = "NCA" if row["dataset"] == "panasonic" else "NMC Samsung"
        lines.append(
            f"| {row['experiment']} | {cell_name} | {row['variant']} | "
            f"{100 * float(row['median_relative_loss_reduction']):.2f} | "
            f"{train_delta:.3f} | {validation_delta:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the cited continuous improvement on targets does not imply reduced MSE on training windows. For ECM-UDE A4b/NCA, the selected checkpoint has higher training MSE than NLS and also higher continuous training MAE, but lower continuous validation MAE; its higher training error is hidden when NLS is compared with the last epoch's online mean. Continuous metrics by segment and the four requested values per seed allow the mask/simulation-state difference to be separated from the selection criterion.",
            "",
        ]
    )
    return lines


def _report_e2(
    comparison_rows: list[dict],
    sensitivity_rows: list[dict],
    info: dict,
) -> list[str]:
    source = info["udds_source"]
    lines = [
        "## E2 — LG HG2 calibration (exploratory)",
        "",
        "Figure `hg2_udds_calibration.pdf` shows the full UDDS 25 °C profile; the MAE of approximately 80 mV is reproduced on temporal validation (the final part of the profile), not on the full profile. Validation error is concentrated at the end of discharge: "
        + "; ".join(
            f"{row['segment']}: MAE {row['mae_mV']:.2f} mV"
            for row in source["error_by_soc_segment_validation"]
        )
        + ".",
        "",
        "Measured and predicted voltage uses the saved ECM-2RC NLS continuous simulation; z0 is anchored at 1 after `551_Charge2` (verified rest) and SOC is counted causally. `Cap_1C` is "
        f"{info['capacity_ah']:.5f} Ah; net discharge in UDDS is {source['net_discharged_ah']:.5f} Ah "
        f"(gross discharge {source['discharge_throughput_ah']:.5f} Ah and regeneration {source['charge_throughput_ah']:.5f} Ah). "
        f"Recalculated final SOC is {source['calculated_final_soc']:.5f}; initial/final voltages are "
        f"{source['initial_voltage_V']:.5f}/{source['final_voltage_V']:.5f} V.",
        "",
        "The OCV comparison uses medians in 0,002 SOC bins and the monotonic envelope; a positive difference means alternative OCV above the C/20 discharge branch. For HPPC, the median of the last 30 s of rests of at least 600 s with |I|≤0,02 A was used as a descriptive estimate of relaxed OCV.",
        "",
        "| Comparison | SOC range | Mean Δ (mV) | Median Δ (mV) | Mean absolute error (mV) | Δ range (mV) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['comparison']} | {row['soc_band']} | "
            f"{row['mean_signed_difference_mV']:.2f} | "
            f"{row['median_signed_difference_mV']:.2f} | "
            f"{row['mean_absolute_difference_mV']:.2f} | "
            f"[{row['minimum_difference_mV']:.2f}, {row['maximum_difference_mV']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "The sensitivity analysis refits only the 2RC NLS using mean charge/discharge OCV; it retrains no networks. The order of the four models did not change in "
            f"{len(sensitivity_rows) - info['sensitivity_order_changed_conditions']}/{len(sensitivity_rows)} conditions "
            f"({info['sensitivity_order_changed_conditions']} changes). The alternative fit converged to parameters practically identical to the original: maximum absolute change in OCV coefficients {info['fit_parameter_comparison']['maximum_absolute_ocv_coefficient_change']:.2e} and R/C parameters {info['fit_parameter_comparison']['maximum_absolute_R_C_parameter_change']:.2e}; the identification window starts at SOC {info['fit_parameter_comparison']['original_training_soc_min']:.3f}. Errors by condition, before/after orderings, and parameters are in `hg2_nls_mean_ocv_sensitivity.csv` and `hg2_calibration.json`; OCV curves are in `hg2_ocv_curves.csv`/`hg2_ocv_comparison.pdf`.",
            "",
        ]
    )
    return lines


def _report_e3(test_rows: list[dict]) -> list[str]:
    m50t = [row for row in test_rows if row["cohort"] == "M50T"]
    hg2_ude = [
        row
        for row in test_rows
        if row["cohort"] == "HG2" and row["variant"] == "ecm_ude"
    ]
    a4 = [row for row in test_rows if row["cohort"] in ("A4A", "A4B")]
    lines = [
        "## E3 — One-sided superiority (exploratory)",
        "",
        "All contrasts are variant−reference differences; superiority uses one-sided exact Wilcoxon (`less`), inferiority uses `greater`, and families were Holm-adjusted as described in the CSV. TOST uses ±1 mV. E3 does not change preregistered results.",
        "",
        "### M50T — cell as unit",
        "",
        "| Variant | Reference | Median d_c (mV) | Min–max (mV) | d_c<0 | d_c<−1 | Raw superiority p | Holm superiority p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in m50t:
        lines.append(
            f"| {row['variant']} | {row['reference']} | "
            f"{row['median_difference_mV']:.3f} | "
            f"[{row['minimum_difference_mV']:.3f}, {row['maximum_difference_mV']:.3f}] | "
            f"{row['n_cells_d_below_0']} | {row['n_cells_d_below_minus1']} | "
            f"{row['p_superiority_raw']:.6g} | {row['p_superiority_holm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "### LG HG2 — ECM-UDE by condition",
            "",
            "| Condition | Reference | Median (mV) | Min–max (mV) | Raw superiority p | Holm superiority p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in hg2_ude:
        lines.append(
            f"| {row['condition']} | {row['reference']} | "
            f"{row['median_difference_mV']:.3f} | "
            f"[{row['minimum_difference_mV']:.3f}, {row['maximum_difference_mV']:.3f}] | "
            f"{row['p_superiority_raw']:.6g} | {row['p_superiority_holm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "### A4a/A4b",
            "",
            "| Run | Variant | Reference | Conditions with Holm superiority | Median of medians (mV) | Min–max of medians (mV) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    keys = sorted(
        {
            (row["cohort"], row["variant"], row["reference"])
            for row in a4
        }
    )
    for cohort, variant, reference in keys:
        selected = [
            row
            for row in a4
            if (row["cohort"], row["variant"], row["reference"])
            == (cohort, variant, reference)
        ]
        effects = [float(row["median_difference_mV"]) for row in selected]
        lines.append(
            f"| {cohort} | {variant} | {reference} | "
            f"{sum(float(row['p_superiority_holm']) < 0.05 for row in selected)}/{len(selected)} | "
            f"{np.median(effects):.3f} | [{min(effects):.3f}, {max(effects):.3f}] |"
        )
    lines.extend(
        [
            "",
            "Detailed rows by condition/seeds and raw and adjusted p-values for superiority, inferiority, and TOST are in `e3_superiority_tests.csv`; the seven d_c values and values by seed are in `m50t_differences_by_cell.csv`. The supplementary superiority analysis for HG2 variants and the original E1 appears in the file to complete the E4 synthesis.",
            "",
        ]
    )
    return lines


def _report_e4(synthesis_rows: list[dict]) -> list[str]:
    lines = [
        "## E4 — Synthesis for the paper",
        "",
        "The table combines preregistered measures with exploratory one-sided tests. `better/equivalent/worse` count superiority, TOST ±1 mV, and inferiority tests with Holm separately; categories are not mutually exclusive. For M50T, inference is one population-level test over seven d_c values, not seven independent tests per cell; the cell-level numbers are descriptive.",
        "",
        "`synthesis.csv` contains one row per configuration/variant; `synthesis.tex` uses `booktabs` and `\\footnotesize`. The status column separates preregistered TOST from exploratory superiority; for A4b against physics-gradient, both are exploratory.",
        "",
    ]
    return lines


def _write_e_report(
    metric_rows: list[dict],
    reconciliation_rows: list[dict],
    a4_summary_rows: list[dict],
    ocv_rows: list[dict],
    sensitivity_rows: list[dict],
    e2_info: dict,
    test_rows: list[dict],
    synthesis_rows: list[dict],
) -> str:
    lines = [
        "# Part E — Checks and exploratory analyses",
        "",
        "All E1–E4 items are exploratory, except the correction to the interpretation of the A4 result: no network was retrained. The E2 sensitivity analysis fits only the expressly requested alternative ECM-2RC NLS.",
        "",
    ]
    lines.extend(_report_e1(metric_rows, reconciliation_rows, a4_summary_rows))
    lines.extend(_report_e2(ocv_rows, sensitivity_rows, e2_info))
    lines.extend(_report_e3(test_rows))
    lines.extend(_report_e4(synthesis_rows))
    lines.extend(
        [
            "## Files and provenance",
            "",
            "- `a4_checkpoint_metrics.csv`, `a4_loss_reconciliation.csv`: E1 metrics by seed and epoch.",
            "- `hg2_udds_calibration.pdf`, `hg2_ocv_comparison.pdf`, `hg2_ocv_comparison.csv`, `hg2_ocv_curves.csv`, `hg2_nls_mean_ocv_sensitivity.csv`, `hg2_calibration.json`: E2.",
            "- `e3_superiority_tests.csv`, `m50t_differences_by_cell.csv`, `e1_original_comparisons.csv`: E3 and reference comparisons used in E4.",
            "- `synthesis.csv`, `synthesis.tex`: E4.",
            "- `manifest.json`: hashes of inputs, checkpoints, code, and outputs.",
            "",
        ]
    )
    report = "\n".join(lines)
    (OUT / "report.md").write_text(report)
    return report


def _update_consolidated_report(report: str, key_results: str) -> None:
    path = ROOT / "report_reviewer_round2.md"
    existing = path.read_text()
    marker = "\n## E — Exploratory addendum before rewriting\n"
    previous_marker = "\n## E — Adendo explorat\u00f3rio antes da reescrita\n"
    addition = (
        marker
        + "\n"
        + key_results.strip()
        + "\n\nDetailed report: `results/reviewer_round2/E/report.md`.\n"
    )
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + addition
    elif previous_marker in existing:
        existing = existing.split(previous_marker, 1)[0].rstrip() + addition
    else:
        existing = existing.rstrip() + "\n" + addition
    path.write_text(existing)


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows, reconciliation_rows, e1_inputs = _run_e1()
    _write_csv(OUT / "a4_checkpoint_metrics.csv", metric_rows)
    _write_csv(OUT / "a4_loss_reconciliation.csv", reconciliation_rows)
    a4_summary_rows = _corrected_a4_summary(reconciliation_rows)
    _write_csv(A_OUT / "a4_training_loss_summary.csv", a4_summary_rows)

    ocv_rows, curve_rows, sensitivity_rows, e2_info = _run_e2()
    _write_e2_outputs(ocv_rows, curve_rows, sensitivity_rows)
    test_rows, m50t_details, original_rows = _run_e3_and_original_synthesis()
    _write_csv(OUT / "e3_superiority_tests.csv", test_rows)
    _write_csv(OUT / "m50t_differences_by_cell.csv", m50t_details)
    _write_csv(OUT / "e1_original_comparisons.csv", original_rows)
    synthesis_rows = _synthesis_rows(test_rows, m50t_details)
    _write_csv(OUT / "synthesis.csv", synthesis_rows)
    _write_synthesis_tex(synthesis_rows, OUT / "synthesis.tex")

    report = _write_e_report(
        metric_rows,
        reconciliation_rows,
        a4_summary_rows,
        ocv_rows,
        sensitivity_rows,
        e2_info,
        test_rows,
        synthesis_rows,
    )
    key_results = (
        "A4 was reevaluated at the selected checkpoints: the old summary used "
        "the last epoch's online mean loss, not checkpoint MSE. The HG2 analysis "
        "located validation error primarily below SOC 0,2 and "
        "compared C/20/HPPC OCV without retraining networks. One-sided tests for "
        "M50T, HG2, and A4 are identified as exploratory; the synthesis separates "
        "these tests from preregistered TOST."
    )
    _update_consolidated_report(report, key_results)

    input_hashes = dict(e1_inputs)
    for name in (
        "results/reviewer_round2/B/manifest.json",
        "results/reviewer_round2/B/base/manifest.json",
        "results/reviewer_round2/B/base/metrics_by_seed.csv",
        "results/reviewer_round2/B/base/ecm_2rc_nls.json",
        "results/reviewer_round2/B/c20_discharge_ocv.csv",
        "results/reviewer_round2/C/run/manifest.json",
        "results/reviewer_round2/C/run/cell_differences.csv",
        "results/reviewer_round2/A/a4a/manifest.json",
        "results/reviewer_round2/A/a4b/manifest.json",
        "results/reviewer_round2/D/tost_pvalues.csv",
    ):
        path = ROOT / name
        input_hashes[name] = _sha256(path)
    manifest = {
        "exploratory": True,
        "retraining_neural_models": False,
        "amendment": "amendment_008",
        "protocol_amendment_commit": "bc0c0b7",
        "input_sha256": input_hashes,
        "source_sha256": {
            name: _sha256(ROOT / name)
            for name in (
                "reviewer_round2_e.py",
                "reviewer_round2_a4.py",
                "reviewer_round2_a1.py",
                "reviewer_round2_a2.py",
                "reviewer_round2_b.py",
                "reviewer_round2_b_audit.py",
                "reviewer_round2_c.py",
                "exp_unified.py",
                "exp_paper_v2_seeds.py",
                "training.py",
                "data.py",
                "protocol.py",
            )
        },
        "outputs": {
            str(path.relative_to(OUT)): _sha256(path)
            for path in sorted(OUT.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
        "modified_existing_outputs": {
            "results/reviewer_round2/A/a4_training_loss_summary.csv": _sha256(
                A_OUT / "a4_training_loss_summary.csv"
            ),
            "results/reviewer_round2/A/report.md": _sha256(A_OUT / "report.md"),
            "report_reviewer_round2.md": _sha256(ROOT / "report_reviewer_round2.md"),
        },
        "checkpoint_hashes_verified": {
            row["checkpoint"]: row["checkpoint_sha256"]
            for row in metric_rows
            if row["experiment"] != "NLS"
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not report:
        raise AssertionError("Exploratory report generation produced no content")


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_all()
