"""Frozen two-cell NLS-only 2RC/ESC experiment, with no neural-network imports."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
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
from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.optimize import least_squares
from scipy.signal import lfilter

from data_lfp import (ROOT, OUTPUT, PROTOCOL_PATH, CellData, Cycle, FloatArray,
                      load_a123, load_mendeley, protocol, save_json, setup_logging, sha256)
from lfp_ocv import OCVCurves, amplitude

LOGGER = logging.getLogger(__name__)
METRICS = ("mae", "rmse", "max_error", "p99")


def directions(current: FloatArray, threshold: float, s0: float = 1.0) -> FloatArray:
    indices = np.where(np.abs(current) > threshold, np.arange(len(current)), -1)
    last = np.maximum.accumulate(indices)
    s = np.full(len(current), s0)
    known = last >= 0
    s[known] = -np.sign(current[last[known]])
    return s


def reversal_indices(current: FloatArray, threshold: float) -> np.ndarray:
    active = np.flatnonzero(np.abs(current) > threshold)
    return active[1:][np.sign(current[active[1:]]) != np.sign(current[active[:-1]])]


def hysteresis(current: FloatArray, s: FloatArray, qnom: float, gamma: float,
               h0: float, dt: float = 1.0) -> FloatArray:
    if qnom <= 0 or gamma < 0 or abs(h0) > 1 or not np.isfinite(current).all():
        raise ValueError("Invalid ESC inputs")
    n = len(current)
    h = np.empty(n)
    h[0] = h0
    a = gamma * np.abs(current[:-1]) * dt / (3600 * qnom)
    # Exact variable-input affine recurrence, in short blocks to avoid exp overflow.
    for lo in range(0, n - 1, 256):
        hi = min(lo + 256, n - 1)
        weights = a[lo:hi]
        cumulative = np.cumsum(weights)
        if cumulative[-1] > 500:
            for k in range(lo, hi):
                h[k + 1] = s[k] + (h[k] - s[k]) * np.exp(-a[k])
        else:
            increments = s[lo:hi] * (-np.expm1(-weights)) * np.exp(cumulative)
            h[lo + 1:hi + 1] = np.exp(-cumulative) * (h[lo] + np.cumsum(increments))
    if np.max(np.abs(h)) > 1 + 1e-10:
        raise FloatingPointError("ESC exact recurrence left [-1,1]")
    return np.clip(h, -1, 1)


def rc_voltage(current: FloatArray, resistance: float, tau: float,
               initial: float = 0.0, dt: float = 1.0) -> FloatArray:
    decay = np.exp(-dt / tau)
    drive = resistance * (-np.expm1(-dt / tau))
    voltage = lfilter([0.0, drive], [1.0, -decay], current)
    return voltage + initial * np.power(decay, np.arange(len(current)))


def unpack(theta: FloatArray, model: str) -> dict:
    r0, r1, r2, tau1, tau2 = np.exp(theta[:5])
    values = dict(R0=float(r0), R1=float(r1), C1=float(tau1 / r1),
                  R2=float(r2), C2=float(tau2 / r2),
                  tau1_s=float(tau1), tau2_s=float(tau2))
    if model == "V3":
        values.update(gamma=float(np.exp(theta[5])), M0_V=float(theta[6]),
                      M_scale=float(theta[7]))
    if model == "V4":
        values["ocv_chebyshev_coefficients_V"] = theta[5:].tolist()
        values["ocv_domain"] = [0.0, 1.0]
    return values


def simulate(cycle: Cycle, qnom: float, curves: OCVCurves, theta: FloatArray,
             model: str) -> tuple[FloatArray, FloatArray | None]:
    p = unpack(theta, model)
    _, dis, mean, m = curves.evaluate(cycle.soc)
    ocv = dis if model == "V2" else mean
    h = None
    if model == "V3":
        s = directions(cycle.current, protocol()["current_threshold_A"], cycle.metadata.get("s0", 1.0))
        h = hysteresis(cycle.current, s, qnom, p["gamma"], cycle.h0)
        ocv = mean + p["M_scale"] * m * h + p["M0_V"] * s
    elif model == "V4":
        z = np.clip(cycle.soc, curves.lo, curves.hi)
        ocv = chebval(2 * z - 1, theta[5:])
    predicted = (ocv - p["R0"] * cycle.current
                 - rc_voltage(cycle.current, p["R1"], p["tau1_s"])
                 - rc_voltage(cycle.current, p["R2"], p["tau2_s"]))
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"{model}: non-finite predicted voltage")
    return predicted, h


def bounds_and_starts(cycle: Cycle, curves: OCVCurves, model: str) -> tuple[FloatArray, FloatArray, list]:
    cfg = protocol()
    b = cfg["parameter_bounds"]
    lo = list(np.log([b["R0_R1_R2_ohm"][0]] * 3 + [b["tau1_tau2_seconds"][0]] * 2))
    hi = list(np.log([b["R0_R1_R2_ohm"][1]] * 3 + [b["tau1_tau2_seconds"][1]] * 2))
    if model == "V3":
        lo += [float(np.log(b["gamma"][0])), b["M0_V"][0], b["M_scale"][0]]
        hi += [float(np.log(b["gamma"][1])), b["M0_V"][1], b["M_scale"][1]]
    elif model == "V4":
        lo += [-np.inf] * 10
        hi += [np.inf] * 10
    starts = []
    for spec in cfg["initial_starts"]:
        theta = list(np.log([spec[k] for k in ("R0", "R1", "R2", "tau1", "tau2")]))
        if model == "V3":
            theta += [float(np.log(spec["gamma"])), cfg["initial_M0_V"], cfg["initial_M_scale"]]
        elif model == "V4":
            z = np.clip(cycle.soc, curves.lo, curves.hi)
            initial_rc = (rc_voltage(cycle.current, spec["R1"], spec["tau1"])
                          + rc_voltage(cycle.current, spec["R2"], spec["tau2"]))
            coeffs = np.linalg.lstsq(chebvander(2 * z - 1, 9),
                                    cycle.voltage + spec["R0"] * cycle.current + initial_rc,
                                    rcond=None)[0]
            theta += coeffs.tolist()
        starts.append(np.asarray(theta))
    return np.asarray(lo), np.asarray(hi), starts


def fit(cycle: Cycle, cell: CellData, curves: OCVCurves, model: str) -> dict:
    cfg = protocol()["optimizer"]
    lower, upper, starts = bounds_and_starts(cycle, curves, model)
    attempts = []
    successful = []
    start_time = time.perf_counter()
    for index, initial in enumerate(starts):
        LOGGER.info("FIT %s %s %s start=%d", cell.name, cycle.name, model, index)
        result = least_squares(lambda theta: simulate(cycle, cell.qnom_Ah, curves, theta, model)[0]
                               - cycle.voltage, initial, bounds=(lower, upper),
                               method=cfg["method"], loss=cfg["loss"], x_scale=cfg["x_scale"],
                               max_nfev=cfg["max_nfev"], ftol=cfg["ftol"],
                               xtol=cfg["xtol"], gtol=cfg["gtol"])
        attempt = {"start": index, "success": bool(result.success), "status": int(result.status),
                   "message": str(result.message), "nfev": int(result.nfev),
                   "squared_residual_V2": float(2 * result.cost),
                   "rmse_mV": float(np.sqrt(np.mean(result.fun ** 2)) * 1000),
                   "optimality": float(result.optimality), "theta": result.x.tolist()}
        attempts.append(attempt)
        LOGGER.info("FIT_RESULT %s", json.dumps(attempt))
        if result.success:
            successful.append((float(result.cost), index, result))
        else:
            LOGGER.error("Optimizer did not converge: %s", attempt)
    if not successful:
        save_json(OUTPUT / cell.name / f"FAILED_fit_{model}_{cycle.name}.json", {"attempts": attempts})
        raise RuntimeError(f"No successful pre-registered start: {cell.name} {model} {cycle.name}")
    _, selected, result = min(successful, key=lambda entry: entry[0])
    finite_bounds = np.isfinite(lower) & np.isfinite(upper)
    proximity = np.full(len(result.x), np.inf)
    proximity[finite_bounds] = np.minimum(
        (result.x[finite_bounds] - lower[finite_bounds]) / (upper[finite_bounds] - lower[finite_bounds]),
        (upper[finite_bounds] - result.x[finite_bounds]) / (upper[finite_bounds] - lower[finite_bounds]))
    near_bounds = np.flatnonzero(proximity < 1e-4).tolist()
    singular = np.linalg.svd(result.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return {"model": model, "identified_on": cycle.name,
            "oracle_target_fit": model == "V4", "theta": result.x.tolist(),
            "parameters": unpack(result.x, model), "selected_start": selected,
            "attempts": attempts, "near_bound_parameter_indices": near_bounds,
            "jacobian_condition": condition,
            "elapsed_seconds": time.perf_counter() - start_time,
            "success": True}


def metrics(error_V: FloatArray) -> dict:
    if not len(error_V) or not np.isfinite(error_V).all():
        raise ValueError("Metrics require finite nonempty errors")
    err = error_V * 1000
    absolute = np.abs(err)
    return {"mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(err * err))),
            "max_error": float(absolute.max()), "p99": float(np.quantile(absolute, 0.99))}


def masks(cycle: Cycle, threshold: float, window: float) -> dict[str, np.ndarray]:
    reversals = reversal_indices(cycle.current, threshold)
    after = np.zeros(len(cycle.t), dtype=bool)
    for k in reversals:
        after[k:np.searchsorted(cycle.t, cycle.t[k] + window)] = True
    result = {"global": np.ones(len(cycle.t), dtype=bool),
              "soc_lt20": cycle.soc < 0.2,
              "soc20_80": (cycle.soc >= 0.2) & (cycle.soc <= 0.8),
              "soc_gt80": cycle.soc > 0.8,
              "after_reversal_60s": after}
    for name in ("soc_lt20", "soc20_80", "soc_gt80"):
        result["after_reversal_60s_" + name] = after & result[name]
    return result


def decide(maes: dict, saturation: dict) -> dict:
    cfg = protocol()["decision"]
    tests = {}
    for name in cfg["each_required_profile"]:
        best = min(maes[name]["V1"], maes[name]["V2"])
        reduction = (best - maes[name]["V3"]) / best
        tests[name] = {"baseline_mae_mV": best, "V3_mae_mV": maes[name]["V3"],
                       "mae_reduction_fraction": reduction,
                       "saturation_fraction": saturation[name],
                       "passes_mae": bool(reduction >= cfg["minimum_mae_reduction"]),
                       "passes_saturation": bool(saturation[name] < cfg["maximum_saturation_fraction_exclusive"])}
        tests[name]["GO"] = tests[name]["passes_mae"] and tests[name]["passes_saturation"]
    return {"verdict": "GO" if all(t["GO"] for t in tests.values()) else "NO-GO",
            "profile_tests": tests, "criterion": cfg}


def write_tables(out: Path, rows: list[dict]) -> None:
    with (out / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tex = [r"\begin{tabular}{lllrrrrr}", r"\hline",
           r"Profile & Model & Scope & N & MAE & RMSE & Max & P99 \\", r"\hline"]
    for row in rows:
        labels = [str(row[k]).replace("_", r"\_") for k in ("profile", "model", "scope")]
        values = [str(row["n_samples"])] + [
            "--" if row[m + "_mV"] is None else f'{row[m + "_mV"]:.3f}' for m in METRICS]
        tex.append(" & ".join(labels + values) + r" \\")
    tex += [r"\hline", r"\end{tabular}", "% All error metrics in mV; V4 is target-fitted oracle."]
    (out / "table.tex").write_text("\n".join(tex) + "\n")


def plot_cycle(out: Path, cycle: Cycle, predictions: dict, h: FloatArray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    hours = cycle.t / 3600
    axes[0].plot(hours, cycle.voltage, color="black", lw=0.7, label="measured")
    for name, predicted in predictions.items():
        axes[0].plot(hours, predicted, lw=0.6, label=name)
        axes[1].plot(hours, (predicted - cycle.voltage) * 1000, lw=0.5, label=name)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(ncol=5)
    axes[1].set_ylabel("Error (mV)")
    axes[2].plot(hours, h, color="tab:blue", label="h")
    axes[2].axhline(0.9, color="gray", ls="--")
    axes[2].axhline(-0.9, color="gray", ls="--")
    axes[2].set(xlabel="Time (h)", ylabel="ESC h", ylim=(-1.1, 1.1))
    twin = axes[2].twinx()
    twin.plot(hours, cycle.current, color="tab:orange", alpha=0.3, lw=0.5)
    twin.set_ylabel("Current (A, discharge positive)")
    fig.suptitle(cycle.name + " | continuous state, no resets")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{cycle.name}_voltage_error_h.{ext}", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    interval = cycle.t <= 1800
    ax.plot(cycle.t[interval], h[interval])
    ax.set(xlabel="Time (s)", ylabel="h", ylim=(-1.1, 1.1))
    twin = ax.twinx()
    twin.plot(cycle.t[interval], cycle.current[interval], alpha=0.4, color="tab:orange")
    twin.set_ylabel("I (A, discharge positive)")
    fig.suptitle(cycle.name + " | first 30 minutes; h and I")
    fig.tight_layout()
    fig.savefig(out / f"{cycle.name}_h_current_zoom.png", dpi=150)
    plt.close(fig)


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   text=True).strip()


def run_cell(cell: CellData) -> dict:
    out = OUTPUT / cell.name
    ocv_manifest = json.loads((out / "ocv_manifest.json").read_text())
    if ocv_manifest["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise ValueError("OCV artifact protocol differs; refusing mixed provenance")
    if ocv_manifest["input_sha256"] != cell.metadata["source_files"]:
        raise ValueError("OCV input hashes differ; refusing stale curves")
    curves = OCVCurves.from_dict(ocv_manifest["points"])
    source = cell.cycles["DST"]
    fits = {name: fit(source, cell, curves, name) for name in ("V1", "V2", "V3")}
    save_json(out / "DST_fits.json", fits)
    rows = []
    maes = {}
    saturation = {}
    diagnostics = {}
    oracle = {}
    plateau = amplitude(curves)
    for name in protocol()["evaluate_profiles"]:
        cycle = cell.cycles[name]
        oracle[name] = fit(cycle, cell, curves, "V4")
        conditions = masks(cycle, protocol()["current_threshold_A"],
                           protocol()["reversal_window_seconds"])
        predictions = {}
        maes[name] = {}
        h = None
        for model in protocol()["models"]:
            fitted = oracle[name] if model == "V4" else fits[model]
            prediction, state = simulate(cycle, cell.qnom_Ah, curves,
                                         np.asarray(fitted["theta"]), model)
            predictions[model] = prediction
            if state is not None:
                h = state
            for scope, mask in conditions.items():
                count = int(mask.sum())
                if count:
                    error_metrics = metrics(prediction[mask] - cycle.voltage[mask])
                else:
                    LOGGER.warning("%s %s %s empty stratum %s", cell.name, name, model, scope)
                    error_metrics = {metric: None for metric in METRICS}
                rows.append({"cell": cell.name, "profile": name, "model": model,
                             "scope": scope, "n_samples": count,
                             **{metric + "_mV": error_metrics[metric] for metric in METRICS}})
                if scope == "global":
                    maes[name][model] = error_metrics["mae"]
        if h is None:
            raise RuntimeError("V3 state diagnostics missing")
        saturation[name] = float(np.mean(np.abs(h) > 0.9))
        indices = reversal_indices(cycle.current, protocol()["current_threshold_A"])
        diag = {"saturation_fraction": saturation[name], "n_reversals": len(indices),
                "reversal_threshold_A": protocol()["current_threshold_A"],
                "reversal_window_seconds": protocol()["reversal_window_seconds"],
                "reversal_indices": indices.tolist(),
                "after_reversal_union_fraction": float(conditions["after_reversal_60s"].mean()),
                "h_min": float(h.min()), "h_max": float(h.max()),
                "soc_min": float(cycle.soc.min()), "soc_max": float(cycle.soc.max()),
                "ocv_boundary_samples": int(np.count_nonzero((cycle.soc < curves.lo) | (cycle.soc > curves.hi))),
                "amplitude_to_global_MAE_ratio": {
                    model: plateau["M_max_abs_mV"] / maes[name][model] for model in ("V1", "V2")},
                "amplitude_to_plateau_MAE_ratio": {}}
        for model in ("V1", "V2"):
            row = next(r for r in rows if r["profile"] == name and r["model"] == model
                       and r["scope"] == "soc20_80")
            diag["amplitude_to_plateau_MAE_ratio"][model] = plateau["M_max_abs_mV"] / row["mae_mV"]
        diagnostics[name] = diag
        np.savez_compressed(out / f"{name}_trajectories.npz", t=cycle.t, current_A=cycle.current,
                            measured_voltage_V=cycle.voltage, soc=cycle.soc, h=h,
                            temperature_C=cycle.temperature, **predictions)
        plot_cycle(out, cycle, predictions, h)
        LOGGER.info("METRICS %s %s mae=%s saturation=%.6f", cell.name, name, maes[name], saturation[name])
    summary = {
        "cell": cell.name, "complete": True, "qnom_Ah": cell.qnom_Ah,
        "manifest": {**cell.metadata, "protocol": protocol(), "protocol_sha256": sha256(PROTOCOL_PATH),
                     "git_commit_at_execution": git_commit(), "python": platform.python_version(),
                     "platform": platform.platform(),
                     "thread_environment": {key: os.environ.get(key) for key in
                                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                     "dependencies": {pkg: importlib.metadata.version(pkg) for pkg in
                                      ("numpy", "scipy", "openpyxl", "xlrd", "matplotlib")},
                     "source_sha256": {str(ROOT / f): sha256(ROOT / f) for f in
                                      ("data_lfp.py", "lfp_ocv.py", "exp_lfp_gonogo.py", "protocols/lfp_gonogo_protocol.json")},
                     "ocv_manifest_sha256": sha256(out / "ocv_manifest.json"),
                     "cycles": {name: cycle.metadata for name, cycle in cell.cycles.items()}},
        "ocv": ocv_manifest, "plateau_amplitude": plateau, "fits": fits,
        "oracle_fits": oracle, "global_mae_mV": maes, "diagnostics": diagnostics,
        "metrics": rows, "decision": decide(maes, saturation),
        "amplitude_ratio_caveat": "Scale estimate, not a rigorous gain ceiling: M_scale is fitted up to 3, M0 contributes separately, OCV/RC bias and SOC errors may dominate.",
        "comparison_caveats": ["DST is identification reference, not held-out test.",
                               "No fitting or model selection on FUDS/US06 except explicitly diagnostic V4.",
                               "Short polarizations and long taus can be poorly identifiable; fit bounds and Jacobian condition reported.",
                               "No neural networks, no UDE, no manuscript changes."]}
    save_json(out / "summary.json", summary)
    write_tables(out, rows)
    LOGGER.info("DECISION %s %s", cell.name, json.dumps(summary["decision"]))
    return summary


def report(summaries: dict[str, dict]) -> None:
    a, b = summaries["cell_A_a123"], summaries["cell_B_mendeley"]
    global_decision = a["decision"]["verdict"]
    text = [
        "# LFP go/no-go test: ECM-2RC and ESC, NLS only",
        "",
        "## Preregistered decision",
        "",
        f"**Cell A (A123 A1-007): {a['decision']['verdict']}.**",
        f"**Cell B (Mendeley, replica): {b['decision']['verdict']}.**",
        f"**Global decision: {global_decision}**, determined exclusively by Cell A.",
        f"The replica {'agrees' if a['decision']['verdict'] == b['decision']['verdict'] else 'does not agree'}; it does not change the global decision.",
        "",
        "GO requires, separately for FUDS and US06, at least a 10% reduction in V3 MAE",
        "against the better of V1/V2 and a fraction of time |h|>0,9 strictly below 90%.",
        "DST is the identification reference; V4 is an oracle fitted to the target cycle, outside the decision.",
        "",
        "| Cell | Profile | Best V1/V2 (mV) | V3 (mV) | MAE reduction (%) | Saturated time (%) | Result |",
        "|---|---|---:|---:|---:|---:|---|"]
    for label, result in (("A", a), ("B", b)):
        for name, test in result["decision"]["profile_tests"].items():
            text.append(f"| {label} | {name} | {test['baseline_mae_mV']:.3f} | {test['V3_mae_mV']:.3f} | "
                        f"{100*test['mae_reduction_fraction']:.2f} | {100*test['saturation_fraction']:.2f} | "
                        f"{'GO' if test['GO'] else 'NO-GO'} |")
    text += ["", "## Global metrics (mV)", "",
             "| Cell | Profile | Model | MAE | RMSE | Maximum | P99 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for label, result in (("A", a), ("B", b)):
        for row in result["metrics"]:
            if row["scope"] == "global":
                text.append(f"| {label} | {row['profile']} | {row['model']} | " +
                            " | ".join(f"{row[m + '_mV']:.3f}" for m in METRICS) + " |")
    text += ["", "## Plateau amplitude and diagnostic", "",
             "| Cell | Qnom (Ah) | M minimum–maximum (mV) | max|M| (mV) | Peak to peak (mV) |",
             "|---|---:|---:|---:|---:|"]
    for label, result in (("A", a), ("B", b)):
        m = result["plateau_amplitude"]
        text.append(f"| {label} | {result['qnom_Ah']:.9f} | {m['M_signed_min_mV']:.3f}–{m['M_signed_max_mV']:.3f} | "
                    f"{m['M_max_abs_mV']:.3f} | {m['M_peak_to_peak_mV']:.3f} |")
    text += ["", "| Cell | Profile | Reversals | Post-reversal window (%) | Saturation (%) | max|M|/MAE V1 | max|M|/MAE V2 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, result in (("A", a), ("B", b)):
        for name, diag in result["diagnostics"].items():
            ratios = diag["amplitude_to_global_MAE_ratio"]
            text.append(f"| {label} | {name} | {diag['n_reversals']} | {100*diag['after_reversal_union_fraction']:.2f} | "
                        f"{100*diag['saturation_fraction']:.2f} | {ratios['V1']:.3f} | {ratios['V2']:.3f} |")
    text += ["", "The requested ceiling is recorded as the scale ratio max|M|/MAE; it is not a strict",
             "mathematical bound on the gain: M_scale is adjustable and M0 is another term. Ratios",
             "against MAE on the plateau itself are also in summary.json.",
             "", "## NLS parameters on DST", "",
             "| Cell | Model | R0 (Ω) | R1 (Ω) | C1 (F) | R2 (Ω) | C2 (F) | γ | M0 (mV) | M_scale |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, result in (("A", a), ("B", b)):
        for name, fit_result in result["fits"].items():
            p = fit_result["parameters"]
            values = [f"{p[k]:.6g}" for k in ("R0", "R1", "C1", "R2", "C2")]
            values += [f"{p['gamma']:.6g}", f"{1000*p['M0_V']:.6g}", f"{p['M_scale']:.6g}"] if name == "V3" else ["—"] * 3
            text.append(f"| {label} | {name} | " + " | ".join(values) + " |")
            if fit_result["near_bound_parameter_indices"]:
                LOGGER.warning("Bound proximity %s %s: %s", label, name, fit_result["near_bound_parameter_indices"])
    text += ["", "### Bounds and identifiability (do not change the criterion)", "",
             "| Cell | Model | τ1 (s) | τ2 (s) | Jacobian condition | Parameter indices near bounds |",
             "|---|---|---:|---:|---:|---|"]
    for label, result in (("A", a), ("B", b)):
        for name, fitted in result["fits"].items():
            p = fitted["parameters"]
            text.append(f"| {label} | {name} | {p['tau1_s']:.3f} | {p['tau2_s']:.3f} | "
                        f"{fitted['jacobian_condition']:.2f} | {fitted['near_bound_parameter_indices']} |")
    text += ["", "Indices: 0=logR0, 1=logR1, 2=logR2, 3=logτ1, 4=logτ2;",
             "in V3: 5=logγ, 6=M0, 7=M_scale. C=τ/R.",
             "A reaches R2=0,5 Ω in V1/V2/V3; B/V3 reaches τ2=100.000 s.",
             "This limits physical interpretation/identifiability, but does not change GO/NO-GO.",
             "", "| Cell | Profile | h minimum | h maximum |",
             "|---|---|---:|---:|"]
    for label, result in (("A", a), ("B", b)):
        for name, diag in result["diagnostics"].items():
            text.append(f"| {label} | {name} | {diag['h_min']:.6f} | {diag['h_max']:.6f} |")
    text += ["", "In B, h remains positive in this run: the gain does not by itself prove",
             "that physical minor loops have been identified. The result is the",
             "preregistered operational test, conditional on the authorized initial states."]
    text += ["", "## Assumptions, sanitization, and limits", "",
             "- Positive I is discharge; active current |I|>0,02 A; s=−1 discharge and +1 charge, held at rest.",
             "- γ uses Q converted from Ah to coulombs (3600 Q) and time in seconds.",
             "- One state per profile: z0=1, h0=s0=+1; initial RC states zero. A has recorded CCCV and 120 s rest; B has no preparation and uses the user-authorized assumption.",
             "- Causal SOC by trapezoidal integration, without OCV inversion on the plateau, target-cycle capacity, or reset by window/step.",
             "- A uses A1-007 selected before fitting; A1-008 remains untouched and is not selected by performance.",
             "- B uses discharge Q at 0,5C=2,498901583 Ah, not 1,1 Ah; standalone Mendeley copy only.",
             "- B is nominally 25 °C, with 25/30 °C ambiguity in all workbooks; no measured temperature.",
             "- B uses PCPD SOC on a continuous Ah scale: charge starts at the discharge endpoint; original SOC axes had different denominators.",
             "- PCHIP of B's 11 points preserves the slight nonmonotonicity and sign of M; A uses medians in 1001 bins, isotonic regression, and PCHIP, without choosing a Chebyshev degree.",
             "- OCV is bounded to the common interval by holding at the boundary; SOC is not clipped and samples outside coverage are counted. Do not extrapolate endpoint curves to explain the plateau.",
             "- B profiles are the processed profiles named DST/FUDS/US06, but their 333/1442/435 s periods differ from standardized cycles; the package has no independent waveform certification.",
             "- Exported Ah can reset; it was not used as dynamic SOC. Qnom is the only external capacity used.",
             "- NLS on the entire DST, three initializations, and bounds frozen before results. FUDS/US06 do not select parameters. V4 deliberately fits degree9 OCV on the target.",
             "- Approximations: current and voltage linearly resampled at 1 s; exact RC/h transitions under zero-order hold. Temperature does not parameterize the ECM in this fixed nominal-condition test.",
             "- The simulator does not reuse the Euler/window reset of ecm_2rc.py: the request requires continuous 2RC at 1 s; exact transitions avoid instability and were compared with a scalar recurrence.",
             "- Quasi-OCV under low current/finite rest, unmeasured aging between A dates, unknown exact identity of B.",
             "- Saturation is the fraction of samples on the uniform grid (time); 60 s masks are the union [reversal, reversal+60 s), without double counting. The first discharge does not count as an observed reversal.",
             "- Empty strata have n=0 and missing metrics with a log warning, not zero MAE. Full tables include SOC and post-reversal windows crossed with SOC.",
             "- Bound hits, Jacobian condition, convergence, and all attempts are reported; they do not certify unique identifiability.",
             "", "## Artifacts", ""]
    for cell in (a, b):
        name = cell["cell"]
        text += [f"- [{name}: summary.json](./results/lfp_gonogo/{name}/summary.json)",
                 f"- [{name}: CSV table](./results/lfp_gonogo/{name}/table.csv)",
                 f"- [{name}: LaTeX table](./results/lfp_gonogo/{name}/table.tex)",
                 f"- [{name}: OCV and M](./results/lfp_gonogo/{name}/ocv_and_M.png)"]
        for profile in protocol()["evaluate_profiles"]:
            text.append(f"- [{name}: {profile}, V/error/h/I](./results/lfp_gonogo/{name}/{profile}_voltage_error_h.png)")
    text += ["", "Logs are in results/lfp_gonogo/stage1_*, stage2_*, and stage3_nls.log.",
             "Manifests include absolute paths and SHA256 hashes of sources/data/protocol.",
             "Reproducible run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B exp_lfp_gonogo.py`.",
             "", "**Stopped after this test. Phase 2 (ESC-UDE) was not started; authorization required.**", ""]
    (ROOT / "report_gonogo.md").write_text("\n".join(text))
    save_json(OUTPUT / "global_decision.json", {
        "primary": a["decision"]["verdict"], "replica": b["decision"]["verdict"],
        "global": global_decision, "rule": protocol()["decision"]["global"],
        "protocol_sha256": sha256(PROTOCOL_PATH)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("all", "A", "B"), default="all")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    setup_logging("stage3_nls")
    LOGGER.info("Frozen protocol SHA256=%s; execution commit=%s", sha256(PROTOCOL_PATH), git_commit())
    if args.report_only:
        summaries = {name: json.loads((OUTPUT / name / "summary.json").read_text())
                     for name in ("cell_A_a123", "cell_B_mendeley")}
        for name, value in summaries.items():
            if not value["complete"] or value["manifest"]["protocol_sha256"] != sha256(PROTOCOL_PATH):
                raise ValueError(f"{name}: incomplete result or different protocol")
    else:
        summaries = {}
        for label, loader in (("A", load_a123), ("B", load_mendeley)):
            if args.cell in ("all", label):
                cell = loader()
                summaries[cell.name] = run_cell(cell)
    if set(summaries) == {"cell_A_a123", "cell_B_mendeley"}:
        report(summaries)
    LOGGER.info("COMPLETE cells=%s", list(summaries))


if __name__ == "__main__":
    main()
