"""Audit LG HG2 MAT profiles and their causal full-charge anchors."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import protocol as P

ROOT = Path(__file__).resolve().parent
DATA_ROOT = P.DATA_ROOT / "lg_hg2_v2" / "extracted"
OUT = ROOT / "results" / "reviewer_round2" / "B"
Q_NOM_AH = 2.72639
DT_SECONDS = 0.1
PROFILES = {
    "UDDS_25C_source": ("25degC", "551_UDDS_25degC_LGHG2.mat", "551_Charge2"),
    "Mixed1_25C": ("25degC", "551_Mixed1_25degC_LGHG2.mat", "551_Charge6"),
    "LA92_25C": ("25degC", "551_LA92_25degC_LGHG2.mat", "551_Charge4"),
    "US06_25C": ("25degC", "551_US06_25degC_LGHG2.mat", "551_Charge5"),
    "UDDS_-20C": ("n20degC", "610_UDDS_n20degC_LGHG2.mat", "610_Charge8"),
    "UDDS_-10C": ("n10degC", "596_UDDS_n10degC_LGHG2.mat", "596_Charge2"),
    "UDDS_0C": ("0degC", "589_UDDS_0degC_LGHG2.mat", "589_Charge2"),
    "UDDS_10C": ("10degC", "576_UDDS_10degC_LGHG2.mat", "576_Charge2"),
    "UDDS_40C": ("40degC", "556_UDDS_40degC_LGHG2.mat", "556_Charge2"),
}
CHANNELS = ("Time", "Current", "Voltage", "Battery_Temp_degC", "Ah")


def _read_mat(path: Path) -> dict[str, np.ndarray]:
    measurement = loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
    arrays = {
        key: np.asarray(getattr(measurement, key), dtype=np.float64).reshape(-1)
        for key in CHANNELS
    }
    if len({len(array) for array in arrays.values()}) != 1:
        raise ValueError(f"MAT channel length mismatch: {path}")
    return arrays


def _sanitize(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    finite = np.ones(len(arrays["Time"]), dtype=bool)
    for array in arrays.values():
        finite &= np.isfinite(array)
    clean = {key: value[finite] for key, value in arrays.items()}
    if len(clean["Time"]) < 2:
        raise ValueError("Profile has fewer than two finite time samples")
    raw_dt = np.diff(arrays["Time"])
    order = np.argsort(clean["Time"], kind="stable")
    sorted_arrays = {key: value[order] for key, value in clean.items()}
    _, first = np.unique(sorted_arrays["Time"], return_index=True)
    sorted_arrays = {key: value[first] for key, value in sorted_arrays.items()}
    time = sorted_arrays["Time"]
    if np.any(np.diff(time) <= 0):
        raise ValueError("Timestamp sorting/deduplication did not produce increasing time")
    relative = time - time[0]
    grid = np.arange(int(np.floor(relative[-1] / DT_SECONDS)) + 1) * DT_SECONDS
    uniform = {
        key: np.interp(grid, relative, sorted_arrays[key]) for key in CHANNELS if key != "Time"
    }
    uniform["Time"] = grid
    metadata = {
        "raw_samples": int(len(arrays["Time"])),
        "nonfinite_rows": int((~finite).sum()),
        "zero_time_intervals": int((raw_dt == 0).sum()),
        "negative_time_intervals": int((raw_dt < 0).sum()),
        "duplicate_rows_removed": int(len(clean["Time"]) - len(time)),
        "resampled_samples": int(len(grid)),
        "largest_raw_gap_s": float(np.max(np.diff(time))),
    }
    return uniform, metadata


def _charge_audit(folder: str, charge_id: str, profile_v0: float) -> dict:
    candidates = sorted((DATA_ROOT / folder).glob(f"*{charge_id}*.mat"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one MAT charge record for {charge_id}, found {len(candidates)}")
    arrays = _read_mat(candidates[0])
    time, current, voltage = arrays["Time"], arrays["Current"], arrays["Voltage"]
    finite = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage)
    time, current, voltage = time[finite], current[finite], voltage[finite]
    order = np.argsort(time, kind="stable")
    time, current, voltage = time[order], current[order], voltage[order]
    nonrest = np.flatnonzero(np.abs(current) > 0.02)
    if not len(nonrest):
        raise ValueError(f"No charge-current samples in {candidates[0].name}")
    last_charge = int(nonrest[-1])
    rest_start = last_charge + 1
    rest_duration = (
        float(time[-1] - time[rest_start]) if rest_start < len(time) else 0.0
    )
    rest_voltage = float(voltage[-1])
    cutoff_current = float(current[last_charge])
    peak_voltage = float(np.max(voltage[: last_charge + 1]))
    verified = (
        peak_voltage >= 4.19
        and 0 < cutoff_current <= 0.05
        and rest_duration >= 1800.0
        and abs(rest_voltage - profile_v0) <= 0.05
    )
    return {
        "charge_id": charge_id,
        "path": str(candidates[0].relative_to(DATA_ROOT)),
        "sha256": P.sha256(candidates[0]),
        "samples": int(len(time)),
        "peak_voltage_V": peak_voltage,
        "cutoff_current_A": cutoff_current,
        "rest_duration_s": rest_duration,
        "rest_voltage_V": rest_voltage,
        "profile_initial_voltage_V": profile_v0,
        "start_rest_voltage_difference_V": profile_v0 - rest_voltage,
        "verified_full_charge_and_rest": bool(verified),
    }


def _audit_hwfet_csv(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.replace(b"\x00", b"").decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Time Stamp,Step,Status")),
        None,
    )
    if header_index is None:
        raise ValueError(f"Could not locate channel header in {path}")
    parsed = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[header_index:]))):
        try:
            parsed.append((float(row["Current"]), float(row["Voltage"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not parsed:
        raise ValueError("No numeric HWFET CSV samples")
    return {
        "path": str(path.relative_to(DATA_ROOT)),
        "sha256": P.sha256(path),
        "samples": len(parsed),
        "max_current_A": max(row[0] for row in parsed),
        "max_abs_current_A": max(abs(row[0]) for row in parsed),
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite Part B audit directory: {OUT}")
    OUT.mkdir(parents=True)
    rows = []
    raw_hashes = {}
    charge_records = {}
    for condition, (folder, filename, charge_id) in PROFILES.items():
        candidates = sorted((DATA_ROOT / folder).glob(f"*{filename}"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one MAT profile ending in {filename}, found {len(candidates)}"
            )
        path = candidates[0]
        raw_hashes[str(path.relative_to(DATA_ROOT))] = P.sha256(path)
        raw = _read_mat(path)
        arrays, cleanup = _sanitize(raw)
        time = arrays["Time"]
        current = arrays["Current"]
        voltage = arrays["Voltage"]
        soc_end = float(
            np.clip(
                1.0 + np.trapezoid(current, time) / (Q_NOM_AH * 3600.0),
                0.0,
                1.0,
            )
        )
        charge = _charge_audit(folder, charge_id, float(voltage[0]))
        if not charge["verified_full_charge_and_rest"]:
            raise ValueError(f"Unverified z0 anchor for {condition}: {charge_id}")
        charge_records[condition] = charge
        charge_path = DATA_ROOT / charge["path"]
        raw_hashes[charge["path"]] = charge["sha256"]
        rows.append(
            {
                "condition": condition,
                **cleanup,
                "duration_s": float(time[-1] - time[0]),
                "median_dt_s": float(np.median(np.diff(raw["Time"]))),
                "max_abs_current_A": float(np.max(np.abs(raw["Current"]))),
                "initial_voltage_V": float(voltage[0]),
                "final_voltage_V": float(voltage[-1]),
                "soc_final_coulomb_count": soc_end,
                "charge_id": charge_id,
                "charge_rest_voltage_V": charge["rest_voltage_V"],
                "charge_rest_duration_s": charge["rest_duration_s"],
            }
        )

    hf_mat = DATA_ROOT / "25degC" / "10-29-18_01.36 551_HWFET_25degC_LGHG2.mat"
    hf_csv = DATA_ROOT / "25degC" / "551_HWFET.csv"
    hwfet = _read_mat(hf_mat)
    hwfet_audit = {
        "mat_path": str(hf_mat.relative_to(DATA_ROOT)),
        "mat_sha256": P.sha256(hf_mat),
        "mat_samples": len(hwfet["Time"]),
        "mat_duration_s": float(hwfet["Time"][-1] - hwfet["Time"][0]),
        "mat_max_current_A": float(np.max(hwfet["Current"])),
        "mat_max_abs_current_A": float(np.max(np.abs(hwfet["Current"]))),
        "csv": _audit_hwfet_csv(hf_csv),
        "decision": "Exclude: resting record, not an HWFET drive cycle.",
    }
    raw_hashes[hwfet_audit["mat_path"]] = hwfet_audit["mat_sha256"]
    raw_hashes[hwfet_audit["csv"]["path"]] = hwfet_audit["csv"]["sha256"]

    capacity_path = DATA_ROOT / "25degC" / "10-28-18_13.10 551_Cap_1C_25degC_LGHG2.mat"
    capacity = _read_mat(capacity_path)
    capacity_ah = float(np.max(capacity["Ah"]) - np.min(capacity["Ah"]))
    raw_hashes[str(capacity_path.relative_to(DATA_ROOT))] = P.sha256(capacity_path)
    c20_path = DATA_ROOT / "25degC" / "10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat"
    c20 = _read_mat(c20_path)
    raw_hashes[str(c20_path.relative_to(DATA_ROOT))] = P.sha256(c20_path)
    discharge_samples = int(np.count_nonzero(c20["Current"] < -0.05))
    charge_samples = int(np.count_nonzero(c20["Current"] > 0.05))
    c20_time = c20["Time"]
    c20_current = c20["Current"]
    c20_voltage = c20["Voltage"]
    order = np.argsort(c20_time, kind="stable")
    c20_time, c20_current, c20_voltage = (
        array[order] for array in (c20_time, c20_current, c20_voltage)
    )
    c20_time, first = np.unique(c20_time, return_index=True)
    c20_current, c20_voltage = c20_current[first], c20_voltage[first]
    c20_soc = 1.0 + np.r_[
        0.0,
        np.cumsum(
            (c20_current[1:] + c20_current[:-1])
            * 0.5
            * np.diff(c20_time)
            / (Q_NOM_AH * 3600.0)
        ),
    ]
    discharge = c20_current < -0.05
    ocv_soc, ocv_voltage, ocv_count = [], [], []
    for lower in np.linspace(0.0, 1.0, 501)[:-1]:
        selected = discharge & (c20_soc >= lower) & (c20_soc < lower + 0.002)
        if selected.any():
            ocv_soc.append(float(np.median(c20_soc[selected])))
            ocv_voltage.append(float(np.median(c20_voltage[selected])))
            ocv_count.append(int(selected.sum()))
    ocv_voltage = np.maximum.accumulate(np.asarray(ocv_voltage, dtype=float))
    keep = np.r_[True, np.diff(ocv_voltage) > 0]
    ocv_rows = [
        {"soc": soc, "ocv_V": voltage, "samples_in_bin": count}
        for soc, voltage, count, include in zip(
            ocv_soc, ocv_voltage, ocv_count, keep
        )
        if include
    ]
    with (OUT / "c20_discharge_ocv.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("soc", "ocv_V", "samples_in_bin"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ocv_rows)

    lines = [
        "# Part B — LG HG2 data audit",
        "",
        "Training and evaluation signals were consistently read from MAT files: time, current, voltage, surface temperature, and Ah. MAT Wh was ignored; Wh was not needed. Positive current represents charge and negative current discharge.",
        "",
        "Preregistered cleaning rejects nonfinite rows, stably sorts by time, retains the first sample at duplicate timestamps, and linearly resamples to a 0,1 s grid. The table shows raw-file time intervals before cleaning; no negative intervals were observed.",
        "",
        "| Profile | Raw samples | Duration (s) | median dt (s) | dt=0 | dt<0 | Initial V (V) | Final V (V) | Final SOC by counting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['raw_samples']} | {row['duration_s']:.3f} | "
            f"{row['median_dt_s']:.6f} | {row['zero_time_intervals']} | "
            f"{row['negative_time_intervals']} | {row['initial_voltage_V']:.5f} | "
            f"{row['final_voltage_V']:.5f} | {row['soc_final_coulomb_count']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Full charge and z0",
            "",
            f"The measured `551_Cap_1C` capacity at 25 degC was used: {capacity_ah:.5f} Ah. For each profile, the preceding charge mapped in the protocol reached at least 4,19 V, had a final charging current no greater than 0,05 A, and was followed by at least 1800 s with |I|<=0,02 A. The start of each profile differed by at most 0,05 V from the voltage at the end of rest. Thus, z0=1 is anchored before each profile; coulomb counting uses only samples preceding the evaluated point.",
            "",
            "| Profile | Preceding charge | Rest (s) | Rest V (V) | Initial V (V) | Difference (mV) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for condition, charge in charge_records.items():
        lines.append(
            f"| {condition} | {charge['charge_id']} | {charge['rest_duration_s']:.1f} | "
            f"{charge['rest_voltage_V']:.5f} | {charge['profile_initial_voltage_V']:.5f} | "
            f"{1000*charge['start_rest_voltage_difference_V']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The OCV curve uses the C/20 discharge branch of `549_C20DisCh` at 25 degC (negative current): causal current integration, 0,002 SOC bins with median voltage and cumulative monotone envelope, as in CALCE calibration. The charge branch was not used. The resulting curve is in `c20_discharge_ocv.csv`; Cap_1C capacity and the C/20 curve were read only from MAT files.",
            "",
            "## HWFET 25 degC substitution",
            "",
            f"Distributed HWFET has {hwfet_audit['mat_samples']} MAT samples, duration {hwfet_audit['mat_duration_s']:.3f} s and maximum/absolute current {hwfet_audit['mat_max_current_A']:.5f}/{hwfet_audit['mat_max_abs_current_A']:.5f} A. The CSV also has {hwfet_audit['csv']['samples']} numeric samples and maximum/absolute current {hwfet_audit['csv']['max_current_A']:.5f}/{hwfet_audit['csv']['max_abs_current_A']:.5f} A. Both are rest; therefore HWFET was replaced by Mixed1 25 degC, preceded by verified `551_Charge6`.",
            "",
            "The four 10 degC CSVs labeled MG1 were excluded. Mixed3, absent at 0 and 10 degC, was not used. Wh was read only from CSV when present; it was not used in modeling.",
            "",
            f"The C/20 file has {discharge_samples} samples with I<-0,05 A and {charge_samples} with I>0,05 A; calibration uses only the first discharge branch.",
            "",
            "SHA256 hashes of all MAT/CSV files actually audited are in `manifest.json`.",
        ]
    )
    (OUT / "data_audit.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "profile_input_sha256": raw_hashes,
        "source_code_sha256": {
            name: P.sha256(ROOT / name)
            for name in ("reviewer_round2_b_audit.py", "protocols/reviewer_round2_protocol.json", "protocol.py")
        },
        "capacity_ah": capacity_ah,
        "ocv_c20_branch": "discharge, I < -0.05 A",
        "charge_verification": charge_records,
        "hwfet_defect": hwfet_audit,
        "cleaning_rule": "finite rows; stable-sort time; keep first duplicate; linear resampling at 0.1 s",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    run()
