"""Audit, fit, and evaluate the registered M50T between-cell replication."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import torch

from scipy.optimize import least_squares

import exp_unified as U
import protocol as P
from ecm import ECMParams
from ecm_2rc import ECMParams2RC
from exp_paper_v2_seeds import shifted_wilcoxon
from training import TrainConfig
from utils import set_seed

ROOT = Path(__file__).resolve().parent
DATA_ROOT = P.DATA_ROOT / "stanford_m50t_raw"
OUT = ROOT / "results" / "reviewer_round2" / "C"
CELLS = ("G1", "V4", "V5", "W3", "W7", "W8", "W10")
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("physics_gradient", "output_residual", "ecm_ude")
NEURAL_VARIANTS = ("output_residual", "ecm_ude")
DT = 0.1
REST_STEP = 11
UDDS_STEP = 14
Q_FLOOR = 0.0
Q_CEILING = 1.0
CFG_TEMPLATE = dict(
    default_z0=0.8,
    dt_seconds=DT,
    window_length=1024,
    window_stride=512,
    batch_size=32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_data(path: Path) -> dict[str, np.ndarray]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = [name for name in workbook.sheetnames if name.startswith("Channel_")]
    if len(sheets) != 1:
        workbook.close()
        raise ValueError(f"Expected one Channel sheet in {path}, found {sheets}")
    sheet = workbook[sheets[0]]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    columns = {str(name): i for i, name in enumerate(header) if name is not None}
    required = ("Test_Time(s)", "Step_Index", "Voltage(V)", "Current(A)",
                "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)")
    missing = [name for name in required if name not in columns]
    if missing:
        workbook.close()
        raise ValueError(f"Missing columns {missing} in {path}")
    temperature_column = next(
        (i for name, i in columns.items() if name.startswith("Aux_Temperature")),
        None,
    )
    fields = {
        "time": columns["Test_Time(s)"],
        "step": columns["Step_Index"],
        "voltage": columns["Voltage(V)"],
        "current": columns["Current(A)"],
        "charge_ah": columns["Charge_Capacity(Ah)"],
        "discharge_ah": columns["Discharge_Capacity(Ah)"],
    }
    if temperature_column is not None:
        fields["temperature"] = temperature_column
    data = {key: [] for key in fields}
    for row in rows:
        try:
            time_value = float(row[fields["time"]])
            step_value = int(row[fields["step"]])
            voltage = float(row[fields["voltage"]])
            current = float(row[fields["current"]])
            charge_ah = float(row[fields["charge_ah"]] or 0.0)
            discharge_ah = float(row[fields["discharge_ah"]] or 0.0)
        except (TypeError, ValueError, IndexError):
            continue
        values = (time_value, voltage, current, charge_ah, discharge_ah)
        if not all(math.isfinite(value) for value in values):
            continue
        data["time"].append(time_value)
        data["step"].append(step_value)
        data["voltage"].append(voltage)
        data["current"].append(current)
        data["charge_ah"].append(charge_ah)
        data["discharge_ah"].append(discharge_ah)
        if "temperature" in fields:
            try:
                temperature = float(row[fields["temperature"]])
            except (TypeError, ValueError, IndexError):
                temperature = math.nan
            data["temperature"].append(temperature)
    workbook.close()
    return {name: np.asarray(values) for name, values in data.items()}


def _fragment_paths(cell: str) -> list[Path]:
    paths = sorted((DATA_ROOT / cell).glob(f"*N25*{cell}_Channel_*.xlsx"))
    if len(paths) != 10:
        raise ValueError(f"{cell}: expected ten N25 fragments, found {len(paths)}")
    return paths


def _new_event(step: int, path: Path, row_start: int,
               same_file_rest: bool, data: dict[str, np.ndarray],
               start: int, stop: int) -> dict:
    times = data["time"][start:stop]
    currents = data["current"][start:stop]
    event = {
        "step": step,
        "start_path": path,
        "start_row": row_start + start,
        "start_time_s": float(times[0]),
        "end_path": path,
        "end_row": row_start + stop - 1,
        "end_time_s": float(times[-1]),
        "fragments": [path],
        "same_file_rest_before": bool(same_file_rest),
        "n_samples": len(times),
        "current_min_A": float(currents.min()),
        "current_max_A": float(currents.max()),
        "current_sum_A": float(currents.sum()),
        "current_sum_sq_A2": float(np.dot(currents, currents)),
        "dt": np.diff(times).tolist() if step == UDDS_STEP else [],
        "boundary_gaps_s": [],
        "ah_first": float(data["charge_ah"][start] - data["discharge_ah"][start]),
        "ah_last": float(data["charge_ah"][stop - 1] - data["discharge_ah"][stop - 1]),
    }
    return event


def _append_run(event: dict, path: Path, row_start: int,
                data: dict[str, np.ndarray], start: int, stop: int) -> None:
    times = data["time"][start:stop]
    currents = data["current"][start:stop]
    gap = float(times[0] - event["end_time_s"])
    ah_first = float(data["charge_ah"][start] - data["discharge_ah"][start])
    ah_gap = ah_first - event["ah_last"]
    if not (0.0 < gap <= 1.5) or abs(ah_gap) > 0.01:
        raise ValueError(f"Fragment continuity failed at {path}, gap={gap}, Ah={ah_gap}")
    event["boundary_gaps_s"].append(gap)
    if event["step"] == UDDS_STEP:
        event["dt"].append(gap)
        event["dt"].extend(np.diff(times).tolist())
    event["end_path"] = path
    event["end_row"] = row_start + stop - 1
    event["end_time_s"] = float(times[-1])
    event["fragments"].append(path)
    event["n_samples"] += len(times)
    event["current_min_A"] = min(event["current_min_A"], float(currents.min()))
    event["current_max_A"] = max(event["current_max_A"], float(currents.max()))
    event["current_sum_A"] += float(currents.sum())
    event["current_sum_sq_A2"] += float(np.dot(currents, currents))
    event["ah_last"] = float(
        data["charge_ah"][stop - 1] - data["discharge_ah"][stop - 1]
    )


def _discover_events(cell: str) -> tuple[list[dict], list[Path]]:
    paths = _fragment_paths(cell)
    events: list[dict] = []
    current_event = None
    rest_in_this_file = False
    for path in paths:
        data = _channel_data(path)
        times, steps = data["time"], data["step"]
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"Non-monotone Test_Time in {path}")
        boundaries = np.r_[0, np.flatnonzero(np.diff(steps) != 0) + 1, len(steps)]
        rest_in_this_file = False
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            step = int(steps[start])
            if current_event is None or step != current_event["step"]:
                if current_event is not None:
                    events.append(current_event)
                current_event = _new_event(
                    step, path, 2, rest_in_this_file, data, int(start), int(stop)
                )
            else:
                gap = float(times[start] - current_event["end_time_s"])
                ah_first = float(
                    data["charge_ah"][start] - data["discharge_ah"][start]
                )
                ah_gap = ah_first - current_event["ah_last"]
                if not (0.0 < gap <= 1.5) or abs(ah_gap) > 0.01:
                    events.append(current_event)
                    current_event = _new_event(
                        step, path, 2, rest_in_this_file, data, int(start), int(stop)
                    )
                else:
                    _append_run(current_event, path, 2, data, int(start), int(stop))
            if (
                step == REST_STEP
                and times[stop - 1] - times[start] >= 1700.0
                and np.max(np.abs(data["current"][start:stop])) <= 0.02
            ):
                rest_in_this_file = True
        if len(times) and current_event is not None:
            # A fragment boundary is merged only if the next fragment continues
            # this step and satisfies both registered continuity limits.
            pass
    if current_event is not None:
        events.append(current_event)
    for index, event in enumerate(events):
        event["bounded"] = (
            index > 0
            and index + 1 < len(events)
            and events[index - 1]["step"] != event["step"]
            and events[index + 1]["step"] != event["step"]
        )
    return events, paths


def _profile_candidates(cell: str) -> tuple[list[dict], list[Path]]:
    events, paths = _discover_events(cell)
    candidates = []
    for event in events:
        if event["step"] != UDDS_STEP or not event["bounded"]:
            continue
        duration = event["end_time_s"] - event["start_time_s"]
        n = event["n_samples"]
        mean = event["current_sum_A"] / n
        variance = max(0.0, event["current_sum_sq_A2"] / n - mean * mean)
        std = math.sqrt(variance)
        median_dt = statistics.median(event["dt"]) if event["dt"] else math.inf
        if (
            duration >= 1369.0
            and 0.05 <= median_dt <= 0.2
            and max(event["dt"]) <= 1.5
            and event["current_max_A"] - event["current_min_A"] >= 2.0
            and std >= 0.5
        ):
            event["duration_s"] = duration
            event["median_dt_s"] = median_dt
            event["current_std_A"] = std
            if event["same_file_rest_before"]:
                candidates.append(event)
    if len(candidates) < 2:
        raise ValueError(f"{cell}: fewer than one source and one target profile")
    return candidates, paths


def _diagnostic_path(cell: str, kind: str) -> Path:
    matches = sorted((DATA_ROOT / cell).glob(f"*{kind}*{cell}*Channel_*.xlsx"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{cell}: expected one {kind} diagnostic, found {matches}")
    return matches[0]


def _capacity_and_ocv(cell: str) -> tuple[float, np.ndarray, dict]:
    path = _diagnostic_path(cell, "0_05C")
    data = _channel_data(path)
    mask = (data["step"] == 5) & (data["current"] < -0.02)
    if mask.sum() < 100:
        raise ValueError(f"{cell}: C/20 discharge step 5 is incomplete")
    discharged_ah = data["discharge_ah"][mask]
    voltage = data["voltage"][mask]
    capacity = float(np.max(discharged_ah))
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError(f"{cell}: invalid C/20 discharge capacity")
    soc = np.clip(
        1.0 - (discharged_ah - discharged_ah[0]) / capacity,
        Q_FLOOR,
        Q_CEILING,
    )
    edges = np.linspace(0.0, 1.0, 501)
    z_values, v_values = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (soc >= low) & (soc < high)
        if np.any(in_bin):
            z_values.append(float(np.median(soc[in_bin])))
            v_values.append(float(np.median(voltage[in_bin])))
    if len(z_values) < 6:
        raise ValueError(f"{cell}: insufficient C/20 OCV bins")
    z = np.asarray(z_values)
    v = np.maximum.accumulate(np.asarray(v_values))
    unique = np.r_[True, np.diff(v) > 0]
    coeffs = np.polynomial.chebyshev.chebfit(
        2.0 * z[unique] - 1.0, v[unique], 5
    )
    hppc = _diagnostic_path(cell, "HPPC")
    return capacity, coeffs, {
        "c20_path": str(path.relative_to(DATA_ROOT)),
        "c20_sha256": _sha256(path),
        "hppc_path": str(hppc.relative_to(DATA_ROOT)),
        "hppc_sha256": _sha256(hppc),
        "ocv_method": "C/20 step 5 discharge; 0.002-SOC median bins, monotonic cumulative maximum, degree-5 Chebyshev fit",
        "capacity_ah": capacity,
        "ocv_bins": int(unique.sum()),
    }


def _fit_nls(context: dict, cfg, branches: int, ocv_guess: np.ndarray):
    ds = context["ds"]
    starts = context["starts"][context["tr"]]
    length = cfg.window_length
    current = np.stack(
        [-ds._I_raw[start : start + length].numpy() for start in starts]
    ).astype(float)
    z0 = np.asarray([float(ds._SoC_raw[start]) for start in starts])
    soc = z0[:, None] - np.c_[
        np.zeros(len(current)),
        np.cumsum(current[:, :-1], axis=1),
    ] * cfg.dt_seconds / (cfg.nominal_capacity_Ah * 3600.0)
    truth = np.stack(
        [ds._V_raw[start : start + length].numpy() for start in starts]
    ).astype(float)
    guess = np.r_[ocv_guess, np.log10(0.03)]
    for branch in range(branches):
        guess = np.r_[
            guess,
            np.log10(0.02),
            np.log10(5.0 if branch == 0 else 40.0),
        ]
    lower = np.r_[
        np.full(6, -10.0),
        -3.0,
        np.tile([-3.0, np.log10(cfg.dt_seconds)], branches),
    ]
    upper = np.r_[
        np.full(6, 10.0),
        np.log10(0.5),
        np.tile([np.log10(0.5), 4.0], branches),
    ]
    regularization = 1e-3 * np.r_[0.0, [2.0**k for k in range(1, 6)]]

    def residual(theta):
        voltage = U.nls_voltage(theta, current, soc, cfg, branches)
        return np.r_[
            (voltage - truth).ravel(),
            regularization * (theta[:6] - ocv_guess),
        ]

    result = least_squares(
        residual,
        np.clip(guess, lower + 1e-6, upper - 1e-6),
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=500,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    if not result.success:
        raise RuntimeError(f"M50T {branches}RC NLS failed: {result.message}")
    theta = result.x
    params_dict = {
        "ocv_coeffs": theta[:6].tolist(),
        "R0": float(10.0**theta[6]),
        "R1": float(10.0**theta[7]),
        "C1": float(10.0**theta[8] / 10.0**theta[7]),
        "soc_train_min": float(soc.min()),
        "soc_train_max": float(soc.max()),
        "identification_rmse_V": float(
            np.sqrt(np.mean(result.fun[: truth.size] ** 2))
        ),
        "identification_n_windows": int(len(current)),
        "identification_success": True,
    }
    if branches == 1:
        params = ECMParams(**params_dict)
    else:
        params = ECMParams2RC(
            **params_dict,
            R2=float(10.0**theta[9]),
            C2=float(10.0**theta[10] / 10.0**theta[9]),
        )
    return params, {
        "params": params.to_dict(),
        "theta": theta.tolist(),
        "ocv_initialization": "own-cell C/20 discharge; degree-5 Chebyshev fit",
        "optimizer": "scipy least_squares TRF",
        "nfev": int(result.nfev),
        "message": result.message,
        "identification_success": bool(result.success),
    }


def _json_event(event: dict) -> dict:
    return {
        "start_path": str(event["start_path"].relative_to(DATA_ROOT)),
        "start_row": event["start_row"],
        "start_time_s": event["start_time_s"],
        "end_path": str(event["end_path"].relative_to(DATA_ROOT)),
        "end_row": event["end_row"],
        "end_time_s": event["end_time_s"],
        "fragments": [
            str(path.relative_to(DATA_ROOT)) for path in dict.fromkeys(event["fragments"])
        ],
        "fragment_sha256": {
            str(path.relative_to(DATA_ROOT)): _sha256(path)
            for path in dict.fromkeys(event["fragments"])
        },
        "n_samples": event["n_samples"],
        "duration_s": event["duration_s"],
        "median_dt_s": event["median_dt_s"],
        "current_min_A": event["current_min_A"],
        "current_max_A": event["current_max_A"],
        "current_std_A": event["current_std_A"],
        "same_file_rest_before": event["same_file_rest_before"],
    }


def audit() -> dict:
    old_audit = list(csv.DictReader((OUT / "c0_audit.csv").open()))
    decisions = {row["cell"]: row["decision"] for row in old_audit}
    selections = {}
    raw_hashes = {}
    for cell in CELLS:
        profiles, fragments = _profile_candidates(cell)
        capacity, coeffs, calibration = _capacity_and_ocv(cell)
        diagnostic = json.loads(json.dumps(calibration))
        diagnostic["first_block"] = "N25"
        diagnostic["capacity_ah"] = capacity
        diagnostic["ocv_coefficients"] = coeffs.tolist()
        diagnostic["profiles"] = {
            "source": _json_event(profiles[0]),
            "targets": [_json_event(event) for event in profiles[1:]],
        }
        diagnostic["fragment_sha256"] = {
            str(path.relative_to(DATA_ROOT)): _sha256(path) for path in fragments
        }
        selections[cell] = diagnostic
        raw_hashes.update(diagnostic["fragment_sha256"])
        raw_hashes[diagnostic["c20_path"]] = diagnostic["c20_sha256"]
        raw_hashes[diagnostic["hppc_path"]] = diagnostic["hppc_sha256"]
        if decisions.get(cell) != "include":
            raise ValueError(f"C0 eligible cell {cell} is marked {decisions.get(cell)}")
    if len(selections) < 6:
        raise RuntimeError(f"Only {len(selections)} C0 cells are eligible; stop before fitting")
    result = {
        "selection_rule": "Earliest complete bounded step-14 profile with a prior same-file rested reference is source; every later complete profile with the same rule in the same N25 block is a target.",
        "cells": selections,
        "excluded": {
            row["cell"]: row["diagnostic_and_fragment_audit"]
            for row in old_audit if row["decision"] != "include"
        },
        "raw_input_sha256": raw_hashes,
    }
    path = OUT / "c0_segments.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _profile_arrays(cell: str, record: dict, capacity: float) -> tuple[dict, dict]:
    start = record["start_time_s"]
    end = record["end_time_s"]
    raw_parts = []
    for relative in record["fragments"]:
        part = _channel_data(DATA_ROOT / relative)
        mask = (
            (part["step"] == UDDS_STEP)
            & (part["time"] >= start - 1e-8)
            & (part["time"] <= end + 1e-8)
        )
        if np.any(mask):
            raw_parts.append({key: values[mask] for key, values in part.items()})
    if not raw_parts:
        raise ValueError(f"No samples found for {cell} profile {record['start_path']}")
    names = set().union(*(part.keys() for part in raw_parts))
    raw = {}
    for name in names:
        pieces = [
            part[name]
            if name in part
            else np.full(len(part["time"]), np.nan)
            for part in raw_parts
        ]
        raw[name] = np.concatenate(pieces)
    order = np.argsort(raw["time"], kind="stable")
    raw = {name: values[order] for name, values in raw.items()}
    if np.any(np.diff(raw["time"]) <= 0):
        raise ValueError(f"{cell}: profile has duplicate/non-increasing timestamps")
    if abs(raw["time"][0] - start) > 0.11 or abs(raw["time"][-1] - end) > 0.11:
        raise ValueError(f"{cell}: selected profile bounds do not match samples")

    start_path = DATA_ROOT / record["start_path"]
    start_file = _channel_data(start_path)
    rest_runs = []
    steps = start_file["step"]
    boundaries = np.r_[0, np.flatnonzero(np.diff(steps) != 0) + 1, len(steps)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if steps[left] == REST_STEP:
            if (
                start_file["time"][right - 1] - start_file["time"][left] >= 1700
                and np.max(np.abs(start_file["current"][left:right])) <= 0.02
                and start_file["time"][right - 1] < start
            ):
                rest_runs.append((left, right))
    if not rest_runs:
        raise ValueError(f"{cell}: selected profile lacks its same-file causal rest")
    anchor_index = rest_runs[-1][1] - 1
    anchor_time = float(start_file["time"][anchor_index])
    z0 = _causal_z0(
        start_file["time"], start_file["current"], anchor_time, start, capacity
    )
    if not math.isfinite(z0):
        raise ValueError(f"{cell}: non-finite causal z0")

    relative_time = raw["time"] - raw["time"][0]
    grid = np.arange(int(np.floor(relative_time[-1] / DT)) + 1) * DT
    raw_current = np.interp(grid, relative_time, raw["current"])
    voltage = np.interp(grid, relative_time, raw["voltage"])
    if "temperature" in raw and np.isfinite(raw["temperature"]).sum() >= 2:
        finite = np.isfinite(raw["temperature"])
        observed = raw["temperature"][finite]
        if 0.0 <= float(np.median(observed)) <= 60.0 and not np.all(observed == 0):
            temperature = np.interp(grid, relative_time[finite], observed)
        else:
            temperature = np.full(len(grid), 23.0)
    else:
        temperature = np.full(len(grid), 23.0)
    discharge_positive_current = -raw_current
    net_discharged_ah = np.r_[
        0.0,
        np.cumsum(
            (discharge_positive_current[1:] + discharge_positive_current[:-1])
            * 0.5
            * DT
            / 3600.0
        ),
    ]
    cycle = U.make_cycle(
        grid, raw_current, voltage, temperature, net_discharged_ah, z0,
        _cfg(capacity),
    )
    meta = {
        "z0": z0,
        "soc_reference": 0.8,
        "reference_step": REST_STEP,
        "reference_end_time_s": anchor_time,
        "causal_integral_end_time_s": start,
        "temperature_fallback_C": (
            23.0
            if "temperature" not in raw
            or np.isfinite(raw["temperature"]).sum() < 2
            or not 0.0 <= float(np.nanmedian(raw["temperature"])) <= 60.0
            or np.all(raw["temperature"][np.isfinite(raw["temperature"])] == 0)
            else None
        ),
        "n_samples": len(grid),
        "time_start_s": float(raw["time"][0]),
        "time_end_s": float(raw["time"][-1]),
    }
    return cycle, meta


def _causal_z0(
    time: np.ndarray,
    current_charge_positive: np.ndarray,
    anchor_time_s: float,
    profile_start_s: float,
    capacity_ah: float,
    reference_soc: float = 0.8,
) -> float:
    if capacity_ah <= 0 or profile_start_s <= anchor_time_s:
        raise ValueError("Causal SOC requires positive capacity and a later profile start")
    all_time = np.asarray(time, dtype=float)
    all_current = np.asarray(current_charge_positive, dtype=float)
    if all_time.shape != all_current.shape:
        raise ValueError("Time and current arrays must have matching shapes")
    causal = (all_time >= anchor_time_s) & (all_time <= profile_start_s)
    t = all_time[causal]
    current = all_current[causal]
    if (
        len(t) < 2
        or np.any(np.diff(t) <= 0)
        or not np.isfinite(t).all()
        or not np.isfinite(current).all()
        or t[0] > anchor_time_s
        or t[-1] < profile_start_s
    ):
        raise ValueError("Invalid same-file causal SOC integration interval")
    return float(
        reference_soc
        + np.trapezoid(current, t) / (3600.0 * capacity_ah)
    )


def _cfg(capacity: float):
    return P.DatasetProtocol(nominal_capacity_Ah=capacity, **CFG_TEMPLATE)


def _context_for_profile(cycle: dict, cfg, stats=None, source=False) -> dict:
    return U.dataset_context(cycle, cfg, stats=stats, source=source)


def _continuous_vectors(model, name: str, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    if name.endswith("_nls"):
        prediction = U.nls_predict(model, ctx["cycle"], ctx["cfg"])[0]
    else:
        with torch.no_grad():
            prediction = (
                model(ctx["ds"].X[None])[0].cpu().numpy()
                * ctx["stats"]["V_std"]
                + ctx["stats"]["V_mean"]
            )
    mask = ctx["mask"]
    truth = ctx["cycle"]["V"].cpu().numpy()[mask]
    return prediction[mask], truth


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fit_cell(cell: str, selection: dict, out: Path, *, smoke: bool) -> dict:
    capacity = selection["capacity_ah"]
    cfg = _cfg(capacity)
    coeffs = np.asarray(selection["ocv_coefficients"], dtype=float)
    source_cycle, source_meta = _profile_arrays(
        cell, selection["profiles"]["source"], capacity
    )
    source_ctx = _context_for_profile(source_cycle, cfg, source=True)
    source_ctx["cfg"] = cfg
    targets = {}
    for index, record in enumerate(selection["profiles"]["targets"], start=1):
        cycle, meta = _profile_arrays(cell, record, capacity)
        ctx = _context_for_profile(cycle, cfg, stats=source_ctx["stats"])
        ctx["cfg"] = cfg
        targets[f"target_{index}"] = (ctx, meta)
    contexts = {"source_validation": (source_ctx, source_meta), **targets}
    nls_1rc, nls1_info = _fit_nls(source_ctx, cfg, 1, coeffs)
    nls_2rc, nls2_info = _fit_nls(source_ctx, cfg, 2, coeffs)
    params_json = {
        "ecm_1rc_nls": {"params": nls_1rc.to_dict(), "fit": nls1_info},
        "ecm_2rc_nls": {"params": nls_2rc.to_dict(), "fit": nls2_info},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cell}_nls_fits.json").write_text(json.dumps(params_json, indent=2) + "\n")
    rows, histories, hashes = [], [], {}
    deterministic = {"ecm_2rc_nls": nls_2rc}
    for model_name, model in deterministic.items():
        target_predictions = []
        for condition, (ctx, _) in contexts.items():
            prediction, truth = _continuous_vectors(model, model_name, ctx)
            metrics = P.metrics_mV(prediction, truth)
            rows.append(
                {
                    "cell": cell, "model": model_name, "seed": "deterministic",
                    "condition": condition, "evaluation": "continuous",
                    "n_samples": len(truth), **metrics,
                }
            )
            if condition != "source_validation":
                target_predictions.append((prediction, truth))
        if target_predictions:
            target_pred = np.concatenate([item[0] for item in target_predictions])
            target_truth = np.concatenate([item[1] for item in target_predictions])
            rows.append(
                {
                    "cell": cell, "model": model_name, "seed": "deterministic",
                    "condition": "target_aggregate", "evaluation": "continuous",
                    "n_samples": len(target_truth),
                    **P.metrics_mV(target_pred, target_truth),
                }
            )

    seeds = (42,) if smoke else SEEDS
    for variant in VARIANTS:
        for seed in seeds:
            set_seed(seed)
            model = U.build_model(variant, source_ctx["stats"], nls_1rc, cfg)
            checkpoint = out / "checkpoints" / cell / f"{variant}_seed{seed}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite {checkpoint}")
            train_config = TrainConfig(
                **{
                    **P.TRAIN,
                    "n_epochs": 2 if smoke else P.TRAIN["n_epochs"],
                    "patience": 2 if smoke else P.TRAIN["patience"],
                    "ckpt_path": str(checkpoint),
                }
            )
            history = U.train(
                model,
                U.loader(source_ctx, cfg, source_ctx["tr"], seed, True),
                U.loader(source_ctx, cfg, source_ctx["eval_indices"]),
                train_config,
            )
            checkpoint_hash = _sha256(checkpoint)
            hashes[str(checkpoint.relative_to(ROOT))] = checkpoint_hash
            histories.append(
                {
                    "cell": cell, "variant": variant, "seed": seed,
                    "best_epoch": history.best_epoch,
                    "epochs_completed": len(history.train_loss),
                    "train_loss": history.train_loss,
                    "validation_loss": history.val_loss,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
            per_target_predictions = []
            for condition, (ctx, meta) in contexts.items():
                prediction, truth = _continuous_vectors(model, variant, ctx)
                metrics = P.metrics_mV(prediction, truth)
                rows.append(
                    {
                        "cell": cell, "model": variant, "seed": seed,
                        "condition": condition, "evaluation": "continuous",
                        "n_samples": len(truth), **metrics,
                        "checkpoint_sha256": checkpoint_hash,
                        "z0": meta["z0"],
                    }
                )
                if condition != "source_validation":
                    per_target_predictions.append((prediction, truth))
            if per_target_predictions:
                aggregate_pred = np.concatenate([item[0] for item in per_target_predictions])
                aggregate_truth = np.concatenate([item[1] for item in per_target_predictions])
                rows.append(
                    {
                        "cell": cell, "model": variant, "seed": seed,
                        "condition": "target_aggregate", "evaluation": "continuous",
                        "n_samples": len(aggregate_truth),
                        **P.metrics_mV(aggregate_pred, aggregate_truth),
                        "checkpoint_sha256": checkpoint_hash,
                        "z0": "",
                    }
                )
    return {
        "rows": rows,
        "histories": histories,
        "checkpoint_sha256": hashes,
        "source_metadata": source_meta,
        "target_metadata": {
            name: meta for name, (_, meta) in contexts.items()
        },
        "nls_sha256": _sha256(out / f"{cell}_nls_fits.json"),
    }


def _statistics(rows: list[dict], smoke: bool, out: Path) -> None:
    if smoke:
        return
    aggregate = {
        (row["cell"], row["model"], str(row["seed"])): float(row["mae"])
        for row in rows if row["condition"] == "target_aggregate"
    }
    cells = sorted({key[0] for key in aggregate})
    differences = []
    tost_rows = []
    for reference in ("physics_gradient", "ecm_2rc_nls"):
        for variant in NEURAL_VARIANTS:
            values = []
            for cell in cells:
                if reference == "physics_gradient":
                    cell_seeds = [
                        aggregate[(cell, variant, str(seed))]
                        - aggregate[(cell, reference, str(seed))]
                        for seed in SEEDS
                    ]
                else:
                    baseline = next(
                        float(row["mae"])
                        for row in rows
                        if row["cell"] == cell
                        and row["model"] == reference
                        and row["condition"] == "target_aggregate"
                    )
                    cell_seeds = [
                        aggregate[(cell, variant, str(seed))] - baseline
                        for seed in SEEDS
                    ]
                median = float(np.median(cell_seeds))
                differences.append(
                    {
                        "cell": cell, "variant": variant,
                        "reference": reference,
                        "seed_differences_mV": json.dumps(cell_seeds),
                        "median_difference_mV": median,
                    }
                )
                values.append(median)
            lower = shifted_wilcoxon(np.asarray(values) + 1.0, "greater")
            upper = shifted_wilcoxon(np.asarray(values) - 1.0, "less")
            tost_rows.append(
                {
                    "reference": reference,
                    "variant": variant,
                    "n_cells": len(values),
                    "median_cell_difference_mV": float(np.median(values)),
                    "p_lower_raw": lower["p_raw"],
                    "p_upper_raw": upper["p_raw"],
                }
            )
    for reference in ("physics_gradient", "ecm_2rc_nls"):
        for tail in ("lower", "upper"):
            family = [
                {"p_raw": row[f"p_{tail}_raw"], "_row": row}
                for row in tost_rows if row["reference"] == reference
            ]
            U.holm(family)
            for result in family:
                result["_row"][f"p_{tail}_holm"] = result["p_holm"]
    for row in tost_rows:
        row["p_tost_raw"] = max(row["p_lower_raw"], row["p_upper_raw"])
        row["p_tost_holm"] = max(row["p_lower_holm"], row["p_upper_holm"])
    variance_rows = []
    for variant in NEURAL_VARIANTS:
        cell_medians = [
            next(
                row["median_difference_mV"] for row in differences
                if row["cell"] == cell and row["variant"] == variant
                and row["reference"] == "physics_gradient"
            )
            for cell in cells
        ]
        within = [
            np.asarray(json.loads(row["seed_differences_mV"]), dtype=float)
            for row in differences
            if row["variant"] == variant and row["reference"] == "physics_gradient"
        ]
        between_variance = float(np.var(cell_medians, ddof=1))
        mean_within_variance = float(np.mean([np.var(x, ddof=1) for x in within]))
        variance_rows.append(
            {
                "variant": variant,
                "between_cell_variance_mV2": between_variance,
                "mean_within_cell_seed_variance_mV2": mean_within_variance,
                "between_to_within_variance_ratio": (
                    between_variance / mean_within_variance
                    if mean_within_variance > 0 else math.inf
                ),
            }
        )
    _write_csv(out / "cell_differences.csv", differences)
    _write_csv(out / "c2_tost.csv", tost_rows)
    _write_csv(out / "variance_ratio.csv", variance_rows)


def _plot(rows: list[dict], out: Path) -> None:
    if not any(row["condition"] == "target_aggregate" for row in rows):
        return
    agg = {
        (row["cell"], row["model"], str(row["seed"])): float(row["mae"])
        for row in rows if row["condition"] == "target_aggregate"
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True)
    for ax, variant in zip(axes, NEURAL_VARIANTS):
        for index, cell in enumerate(CELLS):
            values = [
                agg[(cell, variant, str(seed))]
                - agg[(cell, "physics_gradient", str(seed))]
                for seed in SEEDS if (cell, variant, str(seed)) in agg
            ]
            if not values:
                continue
            jitter = np.linspace(-0.09, 0.09, len(values))
            ax.scatter(index + jitter, values, s=12, color="0.35", marker="o")
            median = float(np.median(values))
            ax.plot([index - 0.18, index + 0.18], [median, median],
                    color="black", linewidth=1.6)
        ax.axhspan(-1.0, 1.0, color="0.85", zorder=0)
        ax.axhline(0.0, color="0.4", linewidth=0.7)
        ax.set_xticks(range(len(CELLS)), CELLS)
        ax.set_title(variant.replace("_", " "))
        ax.set_xlabel("Cell")
        ax.grid(axis="y", color="0.9", linewidth=0.45)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"$\Delta$ MAE vs. physics-gradient (mV)")
    fig.tight_layout()
    fig.savefig(out / "cell_differences.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_report(out: Path) -> None:
    lines = [
        "# Part C — Replication across M50T cells",
        "",
        "Selection was applied to the first N25 block after the diagnostics. "
        "The source is the first complete UDDS profile with a causal SOC rest "
        "reference in the same file; all later complete profiles in the same block "
        "with an equivalent reference are targets. Exclusions W4, W5, and W9 "
        "and episode identifiers/rows are in `c0_audit.md` and "
        "`c0_segments.json`.",
        "",
        "Positive raw current corresponds to charging and was converted to the "
        "canonical positive-discharge convention. Capacity and OCV use "
        "the cell's own C/20 discharge diagnostic; z0 starts at the "
        "SOC=0,8 rest and integrates only current prior to the start of "
        "each profile. Data and checkpoint hashes are in `manifest.json`.",
        "",
        "## Primary and secondary inference",
        "",
        "Each inferential observation is the median across five seeds of the "
        "difference in continuous MAE aggregated over targets. The primary comparison "
        "is against physics-gradient; the secondary is against ECM-2RC NLS. "
        "TOST uses a ±1 mV margin and exact Wilcoxon, with Holm applied separately by "
        "reference over the two variants.",
        "",
        "| Reference | Variant | n cells | Median difference (mV) | Raw TOST p | Holm TOST p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if (out / "c2_tost.csv").exists():
        for row in _read_csv(out / "c2_tost.csv"):
            lines.append(
                f"| {row['reference']} | {row['variant']} | {row['n_cells']} | "
                f"{float(row['median_cell_difference_mV']):.3f} | "
                f"{float(row['p_tost_raw']):.6g} | "
                f"{float(row['p_tost_holm']):.6g} |"
            )
    lines.extend(["", "## Dispersion", ""])
    if (out / "variance_ratio.csv").exists():
        lines.extend(
            [
                "| Variant | Between-cell variance (mV²) | Mean between-seed variance (mV²) | Ratio |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in _read_csv(out / "variance_ratio.csv"):
            lines.append(
                f"| {row['variant']} | {float(row['between_cell_variance_mV2']):.6g} | "
                f"{float(row['mean_within_cell_seed_variance_mV2']):.6g} | "
                f"{float(row['between_to_within_variance_ratio']):.6g} |"
            )
    if (out / "cell_differences.csv").exists():
        differences = _read_csv(out / "cell_differences.csv")
        lines.extend(
            [
                "",
                "## Differences by cell",
                "",
                "| Cell | Variant | vs. physics-gradient (mV) | vs. ECM-2RC (mV) |",
                "|---|---|---:|---:|",
            ]
        )
        for cell in CELLS:
            for variant in NEURAL_VARIANTS:
                values = {
                    row["reference"]: float(row["median_difference_mV"])
                    for row in differences
                    if row["cell"] == cell and row["variant"] == variant
                }
                if values:
                    lines.append(
                        f"| {cell} | {variant} | "
                        f"{values['physics_gradient']:.3f} | "
                        f"{values['ecm_2rc_nls']:.3f} |"
                    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics_by_seed.csv`: continuous metrics by cell, variant, seed, and target.",
            "- `cell_differences.csv`: differences by seed and median by cell.",
            "- `c2_tost.csv`: raw and Holm-adjusted p-values.",
            "- `variance_ratio.csv`: between-cell and between-seed dispersion.",
            "- `cell_differences.pdf`: differences by seed and median, with a ±1 mV band.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(stage: str, *, smoke: bool = False) -> None:
    torch.set_num_threads(1)
    if stage == "audit":
        audit()
        return
    selection_path = OUT / "c0_segments.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Run `reviewer_round2_c.py audit` before any fit")
    selection = json.loads(selection_path.read_text())
    output = OUT / ("smoke_retry_1" if smoke else "run")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Part C output {output}")
    output.mkdir(parents=True)
    active_cells = CELLS[:1] if smoke else CELLS
    all_rows, all_histories, checkpoint_hashes, nls_hashes = [], [], {}, {}
    source_metadata = {}
    for cell in active_cells:
        result = _fit_cell(cell, selection["cells"][cell], output, smoke=smoke)
        all_rows.extend(result["rows"])
        all_histories.extend(result["histories"])
        checkpoint_hashes.update(result["checkpoint_sha256"])
        nls_hashes[cell] = result["nls_sha256"]
        source_metadata[cell] = {
            "source": result["source_metadata"],
            "targets": result["target_metadata"],
        }
        _write_csv(output / "metrics_by_seed.csv", all_rows)
        (output / "training_histories.json").write_text(
            json.dumps(all_histories, indent=2) + "\n"
        )
    _statistics(all_rows, smoke, output)
    _plot(all_rows, output)
    if not smoke:
        _write_report(output)
    manifest = {
        "selection_sha256": _sha256(selection_path),
        "raw_input_sha256": selection["raw_input_sha256"],
        "checkpoint_sha256": checkpoint_hashes,
        "nls_fit_sha256": nls_hashes,
        "output_sha256": {
            str(path.relative_to(output)): _sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
        "seeds": list((42,) if smoke else SEEDS),
        "training_config": {
            **P.TRAIN,
            "n_epochs": 2 if smoke else P.TRAIN["n_epochs"],
            "patience": 2 if smoke else P.TRAIN["patience"],
        },
        "source_sha256": {
            name: _sha256(ROOT / name)
            for name in (
                "reviewer_round2_c.py", "exp_unified.py", "protocol.py",
                "training.py", "data.py", "models.py", "ecm.py",
                "ecm_2rc.py", "unified_dynamics.py", "utils.py",
            )
        },
        "profile_metadata": source_metadata,
        "smoke_test": smoke,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "fit"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.stage, smoke=args.smoke)
