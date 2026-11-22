"""Preregistered source-only static replication; prior artifacts are read-only."""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from numpy.polynomial.chebyshev import chebval
from scipy.io import loadmat
from scipy.optimize import least_squares

from data_lfp import ROOT, DATA_ROOT, CellData, Cycle, integrate_Ah, save_json, sha256
from exp_lfp_gonogo import METRICS, masks, metrics
from exp_lfp_static import MODELS, bounds_starts, parameter_names, predict, unpack
from lfp_ocv import OCVCurves

OUTPUT = ROOT / "results" / "static_replication"
THERMAL = ROOT / "results" / "lfp_static_temperature"
DATA = DATA_ROOT
PROTOCOL_PATH = ROOT / "protocols" / "static_replication_protocol.json"
REGISTRATION = "78fefb3"
UNIFIED = ROOT / "results" / "unified"
LOGGER = logging.getLogger(__name__)


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def previous_snapshot() -> dict[str, str]:
    return {str(p): sha256(p) for folder in ("lfp_gonogo", "lfp_static", "unified")
            for p in sorted((ROOT / "results" / folder).rglob("*")) if p.is_file()}


def verify_registration() -> None:
    if sha256(PROTOCOL_PATH) != "0c2979ea21d574950750d09f7cedb8d4be4b7d62d57ae91a547870a593b7b9c6":
        raise ValueError("Published protocol differs from the original registration")
    old = json.loads((ROOT / "protocols" / "lfp_static_protocol.json").read_text())
    cfg = protocol()
    for key in ("starts", "optimizer"):
        expected = old[key].copy() if key == "optimizer" else old[key]
        if key == "optimizer":
            expected.pop("selection")
        if cfg[key] != expected:
            raise ValueError(f"Shared fit settings disagree: {key}")
    for key in ("R0_R1_R2_ohm", "tau1_tau2_s", "Q_ratio", "z0"):
        if cfg["bounds"][key] != old["bounds"][key]:
            raise ValueError(f"Shared bounds disagree: {key}")


class CheckpointOCV(OCVCurves):
    """Evaluate the fixed checkpoint polynomial exactly, not via sampled PCHIP."""

    def __init__(self, coefficients: list[float]) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != (6,):
            raise ValueError("Unified Panasonic checkpoint must have degree-5 OCV")
        z = np.linspace(0, 1, 1001)
        v = chebval(2*z-1, self.coefficients)
        super().__init__(z, v, z, v)

    def evaluate(self, z):
        voltage = chebval(2*np.clip(z, self.lo, self.hi)-1, self.coefficients)
        return voltage, voltage, voltage, np.zeros_like(voltage)


def calce_ocv(reference: dict) -> tuple[OCVCurves, dict]:
    meta = reference["datasets"]["calce"]["ocv_calibration"]
    path = DATA / Path(meta["path"]).relative_to("../datasets")
    if sha256(path) != meta["sha256"]:
        raise ValueError("SP1 low-current input differs from unified reference")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = [r[:4] for r in wb.active.values
                if len(r) >= 4 and isinstance(r[0], (int, float)) and r[1] == 3]
    finally:
        wb.close()
    a = np.asarray(rows, dtype=float)
    if not len(a) or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) < 0):
        raise ValueError("Invalid SP1 low-current discharge records")
    current, voltage = -a[:, 3]/1000, a[:, 2]/1000
    if np.any(current <= 0):
        raise ValueError("SP1 step 3 is not exclusively discharge")
    ah = integrate_Ah(a[:, 0], current)
    z = np.clip(1-ah/2, 0, 1)
    pairs = [(float(np.median(z[m])), float(np.median(voltage[m])))
             for lo in np.linspace(0, 1, 501)[:-1]
             if (m := (z >= lo) & (z < lo+0.002)).any()]
    knots = np.asarray(pairs)
    vg = np.maximum.accumulate(knots[:, 1])
    keep = np.r_[True, np.diff(vg) > 0]
    zg, vg = knots[keep, 0], vg[keep]
    curves = OCVCurves(zg, vg, zg, vg)
    return curves, {**meta, "path": str(path), "measured_discharge_Ah": float(ah[-1]),
                    "duplicate_timestamp_rows": int(np.count_nonzero(np.diff(a[:, 0]) == 0)),
                    "duplicate_policy": "Preserve rows as unified calibration; zero-dt intervals integrate zero Ah.",
                    "points": curves.to_dict(), "branch": "discharge only",
                    "interpolation": "PCHIP V(SOC); hold at measured support"}


def load_dataset(name: str, reference: dict) -> tuple[CellData, OCVCurves]:
    cfg = protocol()["datasets"][name]
    ref = reference["datasets"][name]
    if name == "panasonic":
        checkpoint = UNIFIED / "checkpoints" / name / "ecm_2rc_nls.json"
        info = json.loads(checkpoint.read_text())
        expected = ref["runs"]["ecm_2rc_nls"]["deterministic"]["sha256"]
        if sha256(checkpoint) != expected:
            raise ValueError("Unified checkpoint hash mismatch")
        curves = CheckpointOCV(info["params"]["ocv_coeffs"])
        ocv_meta = {"path": str(checkpoint), "sha256": expected,
                    "coefficients": curves.coefficients.tolist(),
                    "independent": False, "method": cfg["V1_OCV"],
                    "points": curves.to_dict()}
    else:
        curves, ocv_meta = calce_ocv(reference)
    cycles, sources = {}, {ocv_meta["path"]: ocv_meta["sha256"]}
    for condition in [cfg["source"], *cfg["primary_targets"], *cfg["secondary_targets"]]:
        meta = ref["conditions"][condition]
        path = (DATA / "Pan18650" / Path(meta["path"]).name if name == "panasonic"
                else DATA / Path(meta["path"]).relative_to("../datasets"))
        digest = sha256(path)
        if digest != meta["sha256"]:
            raise ValueError(f"Raw data changed: {path}")
        sources[str(path)] = digest
        trim = 0
        if name == "panasonic":
            meas = loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
            t = np.asarray(meas.Time, dtype=float).ravel()
            channels = [np.asarray(getattr(meas, k), dtype=float).ravel()
                        for k in ("Current", "Voltage", "Battery_Temp_degC", "Ah")]
            order = np.argsort(t, kind="stable")
            t, channels = t[order], [a[order] for a in channels]
            t, ix = np.unique(t, return_index=True)
            channels = [a[ix] for a in channels]
            gaps = np.flatnonzero(np.diff(t) > 5)
            if len(gaps) and gaps[0] == 0 and np.all(np.abs(channels[0][:gaps[-1]+1]) < .05):
                trim = int(gaps[-1]+1)
                t, channels = t[trim:], [a[trim:] for a in channels]
            raw_i, raw_v, raw_temp, raw_ah = channels
            raw_net = -raw_ah
        else:
            wb = openpyxl.load_workbook(io.BytesIO(path.read_bytes()), read_only=True, data_only=True)
            try:
                sheet = wb[wb.sheetnames[1] if len(wb.sheetnames) > 1 else wb.sheetnames[0]]
                rows = [(r[1], r[6], r[7], r[9]-r[8]) for r in sheet.values
                        if len(r) >= 10 and isinstance(r[4], (int, float)) and int(r[4]) == 7]
            finally:
                wb.close()
            a = np.asarray(rows, dtype=float)
            if a.ndim != 2 or len(a) < 3:
                raise ValueError(f"No dynamic step 7 in {path}")
            a = a[np.argsort(a[:, 0], kind="stable")]
            _, ix = np.unique(a[:, 0], return_index=True)
            t, raw_i, raw_v, raw_net = a[ix].T
            raw_temp = np.full(len(t), float(condition.split("_")[1][:-1]))
        raw = np.column_stack((t, raw_i, raw_v, raw_temp, raw_net))
        if not np.isfinite(raw).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"Invalid raw channels: {condition}")
        t = t-t[0]
        grid = np.arange(int(np.floor(t[-1]))+1, dtype=float)
        current, voltage, temp, net = [np.interp(grid, t, a)
                                       for a in (-raw_i, raw_v, raw_temp, raw_net)]
        net -= net[0]
        z0 = float(meta["z0"])
        soc = z0-net/cfg["Qnom_Ah"]
        metadata = {"path": str(path), "sha256": digest, "z0": z0, "h0": 1.,
                    "z0_estimation": meta["z0_estimation"], "trimmed_samples": trim,
                    "raw_dt_min_median_max_s": [float(np.min(np.diff(t))),
                                              float(np.median(np.diff(t))), float(np.max(np.diff(t)))],
                    "raw_samples": len(t), "samples_1s": len(grid),
                    "current_sign": "negated raw charge-positive; canonical positive discharge",
                    "SOC_policy": cfg["soc"], "temperature_C": float(np.median(temp)),
                    "net_Ah_final": float(net[-1]),
                    "current_integral_Ah_final": float(integrate_Ah(grid, current)[-1]),
                    "initial_polarization_V": [0., 0.], "sampling_seconds": 1.,
                    "soc_min": float(soc.min()), "soc_max": float(soc.max())}
        cycles[condition] = Cycle(condition, grid, current, voltage, temp, net, soc, z0, 1., metadata)
        LOGGER.info("DATA %s %s %s", name, condition, json.dumps(metadata))
    return CellData(name, cfg["Qnom_Ah"], curves.charge_z, curves.charge_V,
                    curves.discharge_z, curves.discharge_V, cycles,
                    {"source_files": sources, "ocv": ocv_meta, "protocol": cfg}), curves


def fit_source(cell: CellData, curves: OCVCurves, model: str) -> dict:
    condition = protocol()["datasets"][cell.name]["source"]
    source = cell.cycles[condition]
    lo, hi, starts = bounds_starts(cell, source, curves, model)
    cfg = protocol()["optimizer"]
    attempts, converged = [], []
    begun = time.perf_counter()
    for index, start in enumerate(starts):
        LOGGER.info("FIT %s %s source=%s start=%s", cell.name, model, condition, index)
        result = least_squares(
            lambda theta: predict(source, cell, curves, theta, model)[0]-source.voltage,
            start, bounds=(lo, hi), **cfg)
        attempt = {"start": index, "success": bool(result.success), "status": int(result.status),
                   "message": str(result.message), "nfev": int(result.nfev),
                   "squared_residual_V2": float(2*result.cost), "theta": result.x.tolist()}
        attempts.append(attempt)
        LOGGER.info("FIT_RESULT %s %s %s", cell.name, model, json.dumps(attempt))
        if result.success:
            converged.append((float(result.cost), index, result))
        else:
            LOGGER.error("NLS did not converge: %s", attempt)
    if not converged:
        save_json(OUTPUT/cell.name/f"FAILED_{model}.json", {"attempts": attempts})
        raise RuntimeError(f"No converged source fit: {cell.name} {model}")
    _, index, chosen = min(converged, key=lambda item: item[0])
    finite = np.isfinite(lo) & np.isfinite(hi)
    near = np.full(len(lo), np.inf)
    near[finite] = np.minimum((chosen.x[finite]-lo[finite])/(hi[finite]-lo[finite]),
                             (hi[finite]-chosen.x[finite])/(hi[finite]-lo[finite]))
    names = parameter_names(model)
    hits = [names[k] for k in np.flatnonzero(near < protocol()["bounds"]["near_bound_normalized_distance"])]
    singular = np.linalg.svd(chosen.jac, compute_uv=False)
    info = {"model": model, "identified_on": condition, "target_refit": False,
            "theta": chosen.x.tolist(), "parameters": unpack(chosen.x, model, cell, source),
            "selected_start": index, "attempts": attempts,
            "parameter_names": names, "near_bound_parameters": hits,
            "n_parameters_near_bounds": len(hits),
            "jacobian_condition": float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
            "elapsed_seconds": time.perf_counter()-begun}
    if hits:
        LOGGER.warning("%s %s near bounds: %s", cell.name, model, hits)
    return info


def decide(maes: dict, targets: list[str]) -> dict:
    profiles = {}
    for target in targets:
        v = maes[target]
        if not all(np.isfinite(v[m]) and v[m] >= 0 for m in MODELS) or v["V1"] == 0:
            raise ValueError("Decision requires finite MAE and positive V1 denominator")
        best = min(MODELS[1:], key=lambda m: v[m])
        profiles[target] = {"best_model": best, "best_mae_mV": v[best],
                            "V1_mae_mV": v["V1"],
                            "relative_reduction": {m: (v["V1"]-v[m])/v["V1"] for m in MODELS[1:]},
                            "passes": bool(v[best] <= .75*v["V1"]),
                            "ceiling_recovered_fractions": {m: None for m in MODELS[1:]},
                            "ceiling_unavailable_reason": "No authorized target-fitted V4"}
    if not profiles:
        raise ValueError("Empty primary target set cannot yield a verdict")
    return {"verdict": "POSITIVA" if all(p["passes"] for p in profiles.values()) else "NEGATIVA",
            "profiles": profiles}


def write_tables(out: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with (out/"table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tex = [r"\begin{tabular}{"+"l"*len(fields)+"}", r"\hline",
           " & ".join(f.replace("_", r"\_") for f in fields)+r" \\", r"\hline"]
    for row in rows:
        values = ["--" if row[f] is None else
                  f"{row[f]:.4f}" if isinstance(row[f], float) else
                  str(row[f]).replace("_", r"\_").replace("%", r"\%") for f in fields]
        tex.append(" & ".join(values)+r" \\")
    tex += [r"\hline", r"\end{tabular}", "% Error metrics mV; prior unified models read only."]
    (out/"table.tex").write_text("\n".join(tex)+"\n")


def figures(out: Path, cell: CellData, curves: OCVCurves, fits: dict, predictions: dict) -> None:
    z = np.linspace(curves.lo, curves.hi, 1001)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z, curves.evaluate(z)[2], "--", label="V1 fixed OCV")
    for model in ("V5", "V7"):
        ax.plot(z, chebval(2*z-1, fits[model]["parameters"]["ocv_coefficients_V"]),
                label=model+" source only")
    ax.set(xlabel="SOC (model coordinate)", ylabel="OCV (V)", title=cell.name)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out/f"identified_ocv.{ext}", dpi=150)
    plt.close(fig)
    for condition, predicted in predictions.items():
        cycle = cell.cycles[condition]
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        hours = cycle.t/3600
        axes[0].plot(hours, cycle.voltage, "k", lw=.6, label="measured")
        for index, (model, voltage) in enumerate(predicted.items(), start=1):
            axes[0].plot(hours, voltage, lw=.5, label=model)
            axes[index].plot(hours, (voltage-cycle.voltage)*1000, lw=.5)
            axes[index].set(ylabel=model+" error (mV)")
            axes[index].axhline(0, color="black", lw=.5)
        axes[0].set(ylabel="Voltage (V)", title=cell.name+" "+condition)
        axes[0].legend(ncol=5)
        axes[-1].set(xlabel="Time (h)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{condition}_error.{ext}", dpi=150)
        plt.close(fig)


def unified_comparisons(dataset: str, condition: str, rows: list[dict]) -> dict:
    selected = {r["model"]: r for r in rows if r["dataset"] == dataset
                and r["condition"] == condition and r["evaluation"] == "continuous"
                and r["model"] in ("ecm_2rc_nls", "output_residual", "ecm_ude")}
    if len(selected) != 3:
        raise ValueError(f"Missing coincident unified model results: {dataset} {condition}")
    return {m: {"mae_mean_mV": float(r["mae_mean_mV"]), "mae_std_mV": float(r["mae_std_mV"]),
                "n_seeds": int(r["n_seeds"])} for m, r in selected.items()}


def run_dataset(name: str, reference: dict, prior_rows: list[dict], manifest: dict) -> dict:
    cell, curves = load_dataset(name, reference)
    out = OUTPUT/name
    out.mkdir(parents=True, exist_ok=True)
    fits = {m: fit_source(cell, curves, m) for m in MODELS}
    save_json(out/"source_fits.json", fits)
    cfg = protocol()["datasets"][name]
    rows, comparison, predictions, maes, diagnostics = [], [], {}, {}, {}
    for condition, cycle in cell.cycles.items():
        scope_masks = masks(cycle, .02, 60)
        predictions[condition], maes[condition], diagnostics[condition] = {}, {}, {}
        socs = {}
        for model in MODELS:
            voltage, z = predict(cycle, cell, curves, np.asarray(fits[model]["theta"]), model)
            predictions[condition][model], socs[model] = voltage, z
            for scope, mask in scope_masks.items():
                count = int(mask.sum())
                if count:
                    measured = metrics(voltage[mask]-cycle.voltage[mask])
                else:
                    LOGGER.warning("Empty scope %s %s %s %s", name, condition, model, scope)
                    measured = {m: None for m in METRICS}
                rows.append({"dataset": name, "profile": condition, "model": model, "scope": scope,
                             "n_samples": count, **{m+"_mV": measured[m] for m in METRICS}})
                if scope == "global":
                    maes[condition][model] = measured["mae"]
            p = unpack(np.asarray(fits[model]["theta"]), model, cell, cycle)
            diagnostics[condition][model] = {"z0": p["z0"], "Qnom_Ah": p["Qnom_Ah"],
                                           "soc_min": float(z.min()), "soc_max": float(z.max()),
                                           "prediction_voltage_min_max_V": [float(voltage.min()), float(voltage.max())],
                                           "ocv_boundary_samples": int(((z < curves.lo) | (z > curves.hi)).sum())}
        np.savez_compressed(out/f"{condition}_trajectories.npz", t=cycle.t, current_A=cycle.current,
                            measured_voltage_V=cycle.voltage, original_soc=cycle.soc, net_Ah=cycle.net_Ah,
                            **predictions[condition], **{m+"_soc": z for m, z in socs.items()})
        refs = unified_comparisons(name, condition, prior_rows)
        comparison.append({"profile": condition, "target_role": "primary" if condition in cfg["primary_targets"]
                           else "source" if condition == cfg["source"] else "secondary",
                           **{m+"_mae_mV": maes[condition][m] for m in MODELS},
                           **{m+"_unified_mae_mV": refs[m]["mae_mean_mV"] for m in refs},
                           **{m+"_unified_std_mV": refs[m]["mae_std_mV"] for m in refs},
                           "protocol_coincident": False})
        LOGGER.info("MAE %s %s %s", name, condition, json.dumps(maes[condition]))
    summary = {"dataset": name, "complete": True, "manifest": manifest, "loader": cell.metadata,
               "cycles": {c: x.metadata for c, x in cell.cycles.items()}, "fits": fits,
               "ocv_points": curves.to_dict(), "metrics": rows, "global_mae_mV": maes,
               "state_diagnostics": diagnostics, "decision": decide(maes, cfg["primary_targets"]),
               "secondary": {c: decide(maes, [c])["profiles"][c] for c in cfg["secondary_targets"]},
               "secondary_has_decision_criterion": False,
               "unified_comparison": comparison, "protocol_differences": protocol()["unified_comparison"],
               "caveats": ["No target refitting or neural training.",
                           "Best model chosen per target by preregistered criterion, not a deployable universal selection.",
                           "Degree-9 unconstrained OCV and Q/z0 are confounded, especially V7.",
                           "Source full-profile metrics are reference, not held-out generalization.",
                           cfg["V1_OCV"], cfg.get("z0_V6_V7", cfg["z0_V1_V5"])]}
    save_json(out/"summary.json", summary)
    write_tables(out, comparison)
    detailed = out/"stratified"
    detailed.mkdir(exist_ok=True)
    write_tables(detailed, rows)
    figures(out, cell, curves, fits, predictions)
    return summary


def blocked_thermal(manifest: dict) -> dict:
    cfg = protocol()["temperature_A123"]
    missing = [name for name in cfg["missing_archives"] if not (DATA/"calce_a123_raw"/name).is_file()]
    if not missing:
        raise RuntimeError("A123 temperature files appeared after registration; inspect before resuming blocked test")
    THERMAL.mkdir(parents=True, exist_ok=True)
    summary = {"test": "A123 thermal transfer", "status": "BLOCKED_MISSING_DATA",
               "complete": False, "verdict": None, "manifest": manifest,
               "required_directory": str(DATA/"calce_a123_raw"),
               "missing_files": [{"filename": n, "url": cfg["download_base_URL"]+n} for n in missing],
               "criterion": protocol()["decision"]["thermal"],
               "fits_performed": False, "downloads_performed": False}
    save_json(THERMAL/"summary.json", summary)
    write_tables(THERMAL, [], ["temperature_C", "profile", "model", "mae_mV", "status"])
    LOGGER.warning("A123 thermal test BLOCKED: %s", missing)
    return summary


def write_report(results: dict, thermal: dict) -> None:
    verdicts = {"POSITIVA": "POSITIVE", "NEGATIVA": "NEGATIVE"}
    lines = ["# NCA/NMC static replication and A123 temperature", "",
             f"Protocol frozen before fitting: `{REGISTRATION}`. No new neural networks.", "",
             "## Test 1: primary criterion (all targets at 25 C)", "",
             "| Dataset | Target | V1 MAE (mV) | Best | MAE (mV) | Reduction | Passes >=25% |",
             "|---|---|---:|---|---:|---:|---|"]
    for name, s in results.items():
        for c, d in s["decision"]["profiles"].items():
            lines.append(f'| {name} | {c} | {d["V1_mae_mV"]:.3f} | {d["best_model"]} | '
                         f'{d["best_mae_mV"]:.3f} | {d["relative_reduction"][d["best_model"]]:.2%} | {d["passes"]} |')
    lines += [""]+[f'- **{n}: {verdicts[s["decision"]["verdict"]]}**.' for n, s in results.items()]
    lines += ["", "No global decision across datasets. The winning model may differ between targets.", "",
              "## Recovered fractions", "",
              "Fractions below are relative MAE reductions against V1, not ceiling fractions.",
              "Ceiling fraction: **unavailable**, because no V4 is authorized. No target was fitted.", "",
              "| Dataset | Condition | V5 | V6 | V7 | Role |",
              "|---|---|---:|---:|---:|---|"]
    for name, s in results.items():
        for c, v in s["global_mae_mV"].items():
            role = next(r["target_role"] for r in s["unified_comparison"] if r["profile"] == c)
            fractions = [(v["V1"]-v[m])/v["V1"] for m in MODELS[1:]]
            lines.append(f'| {name} | {c} | '+ " | ".join(f"{x:.2%}" for x in fractions)+f" | {role} |")
    lines += ["", "Other temperatures are secondary, without a decision criterion.", "",
              "## Read-only comparison with unified experiment (MAE in mV)", "",
              "Neural models: mean +/- standard deviation of five seeds; 2RC deterministic.", "",
              "| Dataset | Condition | V1 | V5 | V6 | V7 | Unified 2RC | Output residual | ECM-UDE |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in results.items():
        for row in s["unified_comparison"]:
            vals = [f'{row[m+"_mae_mV"]:.3f}' for m in MODELS]
            vals += [f'{row[m+"_unified_mae_mV"]:.3f} +/- {row[m+"_unified_std_mV"]:.3f}'
                     for m in ("ecm_2rc_nls", "output_residual", "ecm_ude")]
            lines.append(f'| {name} | {row["profile"]} | '+ " | ".join(vals)+" |")
    lines += ["", "### Protocol differences (descriptive, uncontrolled comparison)", "",
              "- New fits: full source, degree 9 OCV (V5/V7), tau <=3600 s, exact ZOH, dt=1 s.",
              "- Unified: training time windows with guard band, dynamic degree 5 OCV, tau <=10000 s, RK4.",
              "- Unified Panasonic dt=0.1 s; new dt=1 s. V1 uses exactly the 2RC checkpoint OCV, not independent OCV.",
              "- CALCE V1/V6 use low-current SP1 at 25 C, another cell; unified fitted dynamic SP2 OCV.",
              "- Targets: full profile in both; full new source versus unified validation mask.",
              "- Loader causal SOC uses the Ah counter; unified simulates SOC internally by ZOH integration.",
              "- CALCE V1/V5 retain documented z0 per profile (~80%); V6/V7 are forced to [90%,100%].",
              "  This incompatibility was confirmed before fitting; it may limit V6 and confound V7.",
              "- Fitted Q and z0 transferred unchanged; no new OCV inversion on targets.",
              "- SP1 OCV retains the two duplicate timestamps from unified: intervals dt=0 integrate zero Ah.",
              "- Nominal/measured temperature and preparation/initial RC state are in manifests; initial RC zero is an assumption.", "",
              "## Parameters and diagnostics", "",
              "| Dataset | Model | Q (Ah) | Source z0 | tau1/tau2 (s) | Parameters at bounds | Jacobian condition |",
              "|---|---|---:|---:|---|---|---:|"]
    for name, s in results.items():
        for model, fit in s["fits"].items():
            p = fit["parameters"]
            condition = fit["jacobian_condition"]
            lines.append(f'| {name} | {model} | {p["Qnom_Ah"]:.6f} | {p["z0"]:.6f} | '
                         f'{p["tau1_s"]:.3f}/{p["tau2_s"]:.3f} | '
                         f'{fit["n_parameters_near_bounds"]}: {", ".join(fit["near_bound_parameters"])} | '
                         +("--" if condition is None else f"{condition:.3e}")+" |")
    lines += ["", "Near bounds: normalized distance <1e-4; both taus bounded to prevent label swapping.",
              "Degree 9 OCV has no monotonicity constraint. OCV/Q/z0 confounding prevents unique causal attribution.", "",
              "### OCV extrapolation and physical validity", "",
              "| Dataset | Condition | Model | Simulated SOC range | Predicted voltage range (V) |",
              "|---|---|---|---|---|"]
    for name, s in results.items():
        for c in protocol()["datasets"][name]["primary_targets"]:
            for m in ("V5", "V7"):
                d = s["state_diagnostics"][c][m]
                low, high = d["prediction_voltage_min_max_V"]
                lines.append(f'| {name} | {c} | {m} | {d["soc_min"]:.4f}--{d["soc_max"]:.4f} | {low:.3f}--{high:.3f} |')
    lines += ["", "CALCE V5 extrapolates above the initial source SOC in US06/BJDST, with strongly",
              "nonphysical voltages. These errors remain in full in metrics and figures; there was no clipping,",
              "refitting or criterion change. The positive verdict uses the best model per target and does not",
              "physically validate all OCVs. Error plots use separate axes per model to avoid hiding",
              "other errors due to the V5 scale.", "",
              "## Test 2: A123 temperature — BLOCKED", "",
              "**No thermal verdict.** Data at 0, 10, 40 and 50 C are unavailable; no fitting/download.",
              f'ZIP destination: `{thermal["required_directory"]}`.', "",
              "Download the eight official files (low-current OCV and DST-US06-FUDS):", ""]
    lines += [f'- [{x["filename"]}]({x["url"]})' for x in thermal["missing_files"]]
    lines += ["", "Frozen criterion: >=25% on both profiles in at least 3 of 4 temperatures.", "",
              "## Artifacts and validation", "",
              "- [Protocol](static_replication_protocol.json)",
              "- [Panasonic](results/static_replication/panasonic/summary.json)",
              "- [CALCE](results/static_replication/calce/summary.json)",
              "- [Blocked thermal manifest](results/lfp_static_temperature/summary.json)",
              "- Each dataset: table.csv/table.tex with new and unified MAEs; stratified/ with all metrics;",
              "  NPZ trajectories, OCV and error figures, parameters and complete SHA-256 hashes.",
              "- [Log](results/static_replication/run.log)",
              "- [Tests](results/static_replication/tests.log)",
              "- All previous files are verified by SHA-256 before/after; manuscript untouched.",
              "- Independent tests verify the exact criterion, signs/SOC, Q/z0 transfer, scalar RC recurrence,",
              "  saved metrics and integrity of previous results.", ""]
    (ROOT/"report_static_replication.md").write_text("\n".join(lines))


def main() -> None:
    verify_registration()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUTPUT/"run.log", mode="w"), logging.StreamHandler()])
    previous = previous_snapshot()
    save_json(OUTPUT/"previous_artifacts_sha256.json", previous)
    manifest = {"protocol": protocol(), "protocol_sha256": sha256(PROTOCOL_PATH),
                "preregistration_commit": REGISTRATION, "execution_git_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "openpyxl", "matplotlib")},
                "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                "source_sha256": {str(ROOT/p): sha256(ROOT/p) for p in
                                  ("exp_static_replication.py", "exp_lfp_static.py", "exp_lfp_gonogo.py",
                                   "data_lfp.py", "lfp_ocv.py", "protocols/lfp_static_protocol.json", "protocol.py")},
                "unified_summary_sha256": sha256(UNIFIED/"summary.json"),
                "unified_table_sha256": sha256(UNIFIED/"table_main.csv"),
                "previous_artifact_count": len(previous),
                "previous_snapshot_path": str(OUTPUT/"previous_artifacts_sha256.json")}
    thermal = blocked_thermal(manifest)
    reference = json.loads((UNIFIED/"summary.json").read_text())
    with (UNIFIED/"table_main.csv").open() as stream:
        prior_rows = list(csv.DictReader(stream))
    results = {name: run_dataset(name, reference, prior_rows, manifest) for name in protocol()["datasets"]}
    if previous_snapshot() != previous:
        raise RuntimeError("Previous artifacts changed during experiment")
    write_report(results, thermal)
    save_json(OUTPUT/"summary.json", {"complete": True, "manifest": manifest,
                                     "decisions": {n: s["decision"] for n, s in results.items()},
                                     "thermal_status": thermal["status"]})
    LOGGER.info("COMPLETE %s", {n: s["decision"]["verdict"] for n, s in results.items()})


if __name__ == "__main__":
    main()
