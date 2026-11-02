"""Auditable A123 and Mendeley LFP readers; positive current is discharge."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import zipfile

import numpy as np
from numpy.typing import NDArray
import openpyxl
from scipy.integrate import cumulative_trapezoid
import xlrd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "datasets"
OUTPUT = ROOT / "results" / "lfp_gonogo"
B_DIR = (DATA_ROOT / "mendeley_29kw38kzwj_v1" / "extracted" / "Battery Test Data"
         / "LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh" / "25 Degree")
A_DIR = DATA_ROOT / "calce_a123_raw"
PROTOCOL_PATH = ROOT / "lfp_gonogo_protocol.json"
FloatArray = NDArray[np.float64]
LOGGER = logging.getLogger(__name__)


@dataclass
class Cycle:
    name: str
    t: FloatArray
    current: FloatArray
    voltage: FloatArray
    temperature: FloatArray
    net_Ah: FloatArray
    soc: FloatArray
    z0: float
    h0: float
    metadata: dict


@dataclass
class CellData:
    name: str
    qnom_Ah: float
    ocv_charge_z: FloatArray
    ocv_charge_V: FloatArray
    ocv_discharge_z: FloatArray
    ocv_discharge_V: FloatArray
    cycles: dict[str, Cycle]
    metadata: dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def integrate_Ah(t: FloatArray, current: FloatArray) -> FloatArray:
    return cumulative_trapezoid(current, t, initial=0.0) / 3600.0


def _cycle(name: str, raw: FloatArray, qnom: float, metadata: dict) -> Cycle:
    if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) < 3:
        raise ValueError(f"{name}: expected at least three time/I/V/T rows")
    if not np.isfinite(raw).all() or np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError(f"{name}: non-finite channels or non-increasing timestamps")
    dt = np.diff(raw[:, 0])
    if dt.max() > 2.1:
        raise ValueError(f"{name}: gap {dt.max()} s cannot be interpolated as a drive cycle")
    t_raw = raw[:, 0] - raw[0, 0]
    t = np.arange(int(np.floor(t_raw[-1])) + 1, dtype=np.float64)
    current, voltage, temp = [np.interp(t, t_raw, raw[:, k]) for k in (1, 2, 3)]
    ah = integrate_Ah(t, current)
    z0, h0 = float(metadata["z0"]), float(metadata["h0"])
    soc = z0 - ah / qnom
    metadata = {**metadata, "raw_samples": len(raw), "samples_1s": len(t),
                "raw_dt_min_median_max_s": [float(dt.min()), float(np.median(dt)),
                                          float(dt.max())],
                "net_discharge_Ah": float(ah[-1]), "soc_min": float(soc.min()),
                "soc_max": float(soc.max()), "soc_outside_0_1_samples":
                int(np.count_nonzero((soc < 0) | (soc > 1))),
                "resampling": "linear I/V/T to 1s; trapezoidal current integration; no capacity-counter resets",
                "boundary_policy": "unclipped causal SOC, no target-end capacity normalization"}
    LOGGER.info("%s metadata=%s", name, json.dumps(metadata))
    return Cycle(name, t, current, voltage, temp, ah, soc, z0, h0, metadata)


def _xlsx_numeric(path: Path, sheet: str) -> tuple[list, FloatArray]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        rows = ws.values
        header = list(next(rows))
        data = [r[:5] for r in rows if len(r) >= 5 and isinstance(r[0], (int, float))]
    finally:
        wb.close()
    values = np.asarray(data, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid numeric sheet {path}:{sheet}")
    return header, values


def _branch_points(book: xlrd.book.Book, name: str) -> FloatArray:
    sheet = book.sheet_by_name(name)
    pairs = [sheet.row_values(r)[:2] for r in range(1, sheet.nrows)
             if all(isinstance(sheet.cell_value(r, c), (int, float)) for c in (0, 1))]
    values = np.asarray(pairs, dtype=np.float64)
    if values.shape != (11, 2) or not np.isfinite(values).all():
        raise ValueError(f"{name}: expected exactly 11 finite SOC/OCV points")
    return values


def load_mendeley() -> CellData:
    capacity_file = B_DIR / "1.5C 1C 0.5C.xlsx"
    ocv_file = B_DIR / "PCPD.xls"
    dynamic_file = B_DIR / "DST FUDS UDDS HPPC WLTS US06.xlsx"
    _, capacity = _xlsx_numeric(capacity_file, "0.5C")
    if not np.all(capacity[:, 1] < 0) or capacity[-1, 2] > 2.1:
        raise ValueError("0.5C sheet is not a complete discharge to the documented cutoff")
    qnom = float(np.ptp(capacity[:, 3]) / 1000.0)
    q_integral = float(integrate_Ah(capacity[:, 0], -capacity[:, 1] / 1000.0)[-1])
    if qnom <= 0 or abs(q_integral / qnom - 1) > 0.02:
        raise ValueError("Capacity counter and current integration disagree by >2%")
    book = xlrd.open_workbook(str(ocv_file))
    charge = _branch_points(book, "OCV SOC Chr")
    discharge = _branch_points(book, "OCV SOC Dis")
    stats = book.sheet_by_name("Statis_9_1_3")
    discharge_steps = [stats.row_values(r) for r in range(1, stats.nrows)
                       if stats.cell_value(r, 3) == 15 and stats.cell_value(r, 4) == "CC_DChg"]
    charge_steps = [stats.row_values(r) for r in range(1, stats.nrows)
                    if stats.cell_value(r, 3) == 19 and stats.cell_value(r, 4) == "CC_Chg"]
    if len(discharge_steps) != 10 or len(charge_steps) != 10:
        raise ValueError("PCPD pulse sequence does not contain ten pulses in each direction")
    dis_Ah = np.r_[0.0, np.cumsum([r[9] / 1000.0 for r in discharge_steps])]
    chg_Ah = np.r_[0.0, np.cumsum([r[9] / 1000.0 for r in charge_steps])]
    # The two supplied SOC axes use different denominators; keep one continuous Ah scale.
    z_dis = 1.0 - dis_Ah / qnom
    z_chg = z_dis[-1] + chg_Ah / qnom
    if not np.allclose(charge[:, 0] / 100, chg_Ah / chg_Ah[-1], atol=1e-6):
        raise ValueError("Supplied charge SOC is inconsistent with PCPD pulse capacities")
    if not np.allclose(discharge[:, 0] / 100, 1 - dis_Ah / dis_Ah[-1], atol=1e-6):
        raise ValueError("Supplied discharge SOC is inconsistent with PCPD pulse capacities")
    cycles = {}
    checks = {}
    for name in ("DST", "FUDS", "UDDS", "HPPC", "WLTS", "US06"):
        header, values = _xlsx_numeric(dynamic_file, name)
        current = -values[:, 1] / 1000.0
        if not (current.min() < -0.1 and current.max() > 0.1):
            raise ValueError(f"{name}: sheet is not bidirectional dynamic/pulse excitation")
        resets = int(np.count_nonzero(np.diff(values[:, 3]) < 0))
        checks[name] = {"header": header, "rows": len(values),
                        "current_min_max_A": [float(current.min()), float(current.max())],
                        "capacity_resets": resets,
                        "identity": "sheet label and numerical excitation verified; no official waveform reference in local package"}
        expected_lag = {"DST": 333, "FUDS": 1442, "US06": 435, "UDDS": 1370}.get(name)
        if expected_lag is not None:
            correlation = float(np.corrcoef(current[:-expected_lag], current[expected_lag:])[0, 1])
            if correlation < 0.99:
                raise ValueError(f"{name}: expected processed waveform repetition not verified")
            checks[name]["processed_repetition_seconds"] = expected_lag
            checks[name]["repetition_correlation"] = correlation
        if name not in protocol()["evaluate_profiles"]:
            continue
        raw = np.column_stack((values[:, 0], current, values[:, 2],
                               np.full(len(values), 25.0)))
        init = protocol()["mendeley_initial_state"]
        cycles[name] = _cycle(name, raw, qnom, {
            "path": str(dynamic_file), "sheet": name, "z0": init["z0"],
            "h0": init["h0"], "s0": init["s0"], "initial_state_provenance": init["provenance"],
            "preceding_steps": "not provided in processed workbook; user-authorized full-charge assumption",
            "raw_time_start_end_s": [float(values[0, 0]), float(values[-1, 0])],
            "temperature": "25 C nominal; all three workbooks include 30D labels; no measured T",
            "profile_check": checks[name],
            "capacity_resets_ignored": resets})
    metadata = {
        "source_files": {str(p): sha256(p) for p in (capacity_file, ocv_file, dynamic_file)},
        "raw_current_sign": "positive charge; verified CC_Chg/CC_DChg in PCPD",
        "current_conversion": "-raw_mA/1000", "temperature_nominal_C": 25.0,
        "qnom_method": "0.5C discharge Capacity span in 25 Degree, mAh/1000",
        "qnom_Ah": qnom, "capacity_current_integral_Ah": q_integral,
        "label_capacity_Ah": 1.1, "capacity_label_divergence_ratio": qnom / 1.1,
        "ocv_SOC_method": "discharge starts 1, integrate PCPD pulse capacities/Qnom; charge starts at same discharge endpoint",
        "ocv_charge_supplied_SOC_percent": charge[:, 0].tolist(),
        "ocv_discharge_supplied_SOC_percent": discharge[:, 0].tolist(),
        "ocv_discharge_pulse_Ah": dis_Ah.tolist(), "ocv_charge_pulse_Ah": chg_Ah.tolist(),
        "profile_checks": checks, "initial_rc_V": [0.0, 0.0],
        "caveats": ["Manufacturer/model unconfirmed; 1.1Ah label conflicts with measured ~2.50Ah.",
                    "25C nominal versus 30D labels in capacity/dynamic/PCPD and 30 Degree internal PCPD path.",
                    "Dynamic preparation absent; z0=1,h0=+1 explicitly authorized, not measured.",
                    "OCV charge endpoint supplied from final rest; slight non-monotonic charge voltage retained.",
                    "PCPD 30min rests are approximate OCV, not certified equilibrium; no dedicated minor loops.",
                    "Processed repetition periods DST333/FUDS1442/US06435s differ from standard cycles; sheet naming is not independent official waveform certification.",
                    "M may change sign: retained without abs, clipping or target-dependent correction."]}
    LOGGER.warning("B assumptions: %s", metadata["caveats"])
    return CellData("cell_B_mendeley", qnom, z_chg, charge[:, 1], z_dis[::-1],
                    discharge[::-1, 1], cycles, metadata)


def extract_a123() -> dict:
    archives = [A_DIR / "A123_OCV25-20120905.zip",
                A_DIR / "A123_DST-US06-FUDS-25.zip"]
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(f"Mandatory A123 archive missing: {archive}")
    target = A_DIR / "extracted"
    manifest = {}
    for archive in archives:
        members = {}
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                dest = (target / info.filename).resolve()
                if not dest.is_relative_to(target.resolve()) or info.filename.startswith("/"):
                    raise ValueError(f"Unsafe ZIP path: {info.filename}")
                if info.is_dir():
                    continue
                data = bundle.read(info)
                digest = hashlib.sha256(data).hexdigest()
                if dest.exists():
                    if sha256(dest) != digest:
                        raise ValueError(f"Existing extracted file differs; refusing overwrite: {dest}")
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                members[str(dest)] = digest
        manifest[str(archive)] = {"sha256": sha256(archive), "members": members}
    LOGGER.info("A123 extraction manifest=%s", json.dumps(manifest))
    return manifest


def _arbin(path: Path) -> FloatArray:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    try:
        for sheet in wb:
            if not sheet.title.startswith("Channel"):
                continue
            for r in sheet.values:
                if len(r) >= 18 and isinstance(r[4], (int, float)):
                    rows.append([r[k] for k in (1, 4, 5, 6, 7, 8, 9, 17)])
    finally:
        wb.close()
    result = np.asarray(rows, dtype=np.float64)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"Invalid Arbin channels: {path}")
    return result


def load_a123() -> CellData:
    archives = extract_a123()
    specimen = protocol()["a123_specimen"]
    ocv_path = A_DIR / "extracted" / "OCV25-20120905" / f"{specimen}-OCV-25-20120905.xlsx"
    dynamic_path = (A_DIR / "extracted" / "DST-US06-FUDS-25"
                    / f"{specimen}-DST-US06-FUDS-25-20120827.xlsx")
    ocv = _arbin(ocv_path)
    dis, chg = ocv[ocv[:, 1] == 5], ocv[ocv[:, 1] == 7]
    if not (np.all(dis[:, 3] < 0) and np.all(chg[:, 3] > 0)
            and dis[-1, 4] < 2.1 and chg[-1, 4] > 3.5):
        raise ValueError("A123 steps 5/7 are not complete low-current discharge/charge")
    dis_ah = integrate_Ah(dis[:, 0], -dis[:, 3])
    chg_ah = integrate_Ah(chg[:, 0], chg[:, 3])
    qnom = float(dis_ah[-1])
    zdis, zchg = 1 - dis_ah / qnom, chg_ah / qnom
    dynamic = _arbin(dynamic_path)
    cycles = {}
    mapping = {"DST": (8, 9, 4, 5, 7, 360),
               "US06": (16, 17, 12, 13, 15, 600),
               "FUDS": (24, 25, 20, 21, 23, 1372)}
    for name, (step, repeated_rest, cc, cv, rest, period) in mapping.items():
        selected = dynamic[np.isin(dynamic[:, 1], (step, repeated_rest))]
        preparation = dynamic[np.isin(dynamic[:, 1], (cc, cv, rest))]
        cv_rows = dynamic[dynamic[:, 1] == cv]
        if not (np.all(cv_rows[:, 3] > 0) and cv_rows[-1, 3] < 0.06
                and cv_rows[-1, 4] > 3.59):
            raise ValueError(f"{name}: full CCCV preparation not verified")
        raw = np.column_stack((selected[:, 0], -selected[:, 3],
                               selected[:, 4], selected[:, 7]))
        cycles[name] = _cycle(name, raw, qnom, {
            "path": str(dynamic_path), "dynamic_steps_including_repetition_rest": [step, repeated_rest],
            "z0": 1.0, "h0": 1.0, "s0": 1.0,
            "initial_state_provenance": "documented CCCV to 3.6V, taper <0.06A, followed by 120s rest",
            "preceding_steps": [cc, cv, rest], "preparation_time_start_end_s":
            [float(preparation[0, 0]), float(preparation[-1, 0])],
            "raw_time_start_end_s": [float(selected[0, 0]), float(selected[-1, 0])],
            "profile_identity": f"{name}: ordered archive schedule and {period}s excitation followed by 1s repeated-rest step",
            "temperature": "25 C nominal; measured auxiliary sensor retained",
            "post_profile_rest_excluded": True})
        i = cycles[name].current
        lag = period + 1
        correlation = float(np.corrcoef(i[:-lag], i[lag:])[0, 1])
        if correlation < 0.95:
            raise ValueError(f"{name}: A123 repetition signature not verified")
        cycles[name].metadata["repetition_seconds"] = lag
        cycles[name].metadata["repetition_correlation"] = correlation
    metadata = {
        "archives": archives, "source_files": {str(p): sha256(p) for p in (ocv_path, dynamic_path)},
        "specimen": specimen, "selection": protocol()["a123_selection"],
        "raw_current_sign": "positive charge, negative discharge verified against counters/CCCV",
        "current_conversion": "-raw_A", "qnom_Ah": qnom,
        "qnom_method": "trapezoidal current integration, low-current discharge step5 only",
        "low_current_charge_Ah": float(chg_ah[-1]),
        "ocv_steps": {"discharge": 5, "charge": 7, "preceding_full_charge": [2, 3, 4]},
        "ocv_raw_samples": {"discharge": len(dis), "charge": len(chg)},
        "ocv_current_A": {"discharge_median": float(np.median(dis[:, 3])),
                          "charge_median": float(np.median(chg[:, 3]))},
        "initial_rc_V": [0.0, 0.0],
        "caveats": ["A1-007 selected deterministically; A1-008 not a pooled replicate.",
                    "Low-current OCV recorded after dynamic test date; aging between dates not measured.",
                    "Low-current voltage approximates OCV and includes polarization.",
                    "120s rest: initial RC voltages assumed zero, not estimated from target.",
                    "Capacity varies by rate; no cycle-end target capacity is used for SOC.",
                    "Charge covers slightly less Ah than discharge; boundary hold declared."]}
    return CellData("cell_A_a123", qnom, zchg, chg[:, 4], zdis[::-1],
                    dis[::-1, 4], cycles, metadata)


def setup_logging(stage: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUTPUT / f"{stage}.log"),
                                  logging.StreamHandler()], force=True)


def main() -> None:
    setup_logging("stage1_loaders")
    for loader in (load_a123, load_mendeley):
        cell = loader()
        save_json(OUTPUT / cell.name / "loader_manifest.json",
                  {**cell.metadata, "protocol": protocol(), "protocol_sha256": sha256(PROTOCOL_PATH),
                   "loader_sha256": sha256(Path(__file__).resolve()),
                   "cycles": {name: c.metadata for name, c in cell.cycles.items()}})
        LOGGER.info("Verified %s Qnom=%.9f Ah", cell.name, cell.qnom_Ah)


if __name__ == "__main__":
    main()
