"""Calibration-only parameter interpolation and held-out A123 voltage uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
from typing import TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from data_lfp import ROOT, DATA_ROOT, CellData, Cycle, save_json, sha256
from exp_a123_temperature import load_source
from exp_lfp_gonogo import METRICS, masks, metrics
from exp_lfp_static import predict, unpack
from exp_static_replication import write_tables
from lfp_ocv import OCVCurves
from public_integrity import verify as verify_public

OUTPUT = ROOT/"results"/"a123_temperature_interp"
THERMAL = ROOT/"results"/"lfp_static_temperature"/"cell_A_a123"
SOURCE = ROOT/"results"/"lfp_static"/"cell_A_a123"
PROTOCOL_PATH = ROOT/"protocols"/"a123_temperature_interp_protocol.json"
REGISTRATION = "b2d945c"
SAVED_COMMIT = "c0bab49"
PARAMETERS = ("Qeff_Ah", "z0", "R0", "R1", "R2", "tau1_s", "tau2_s")
METHODS = ("P1", "P2", "P3", "P4")
LOG_COORDS = np.array([False, False, True, True, True, True, True])
LOGGER = logging.getLogger(__name__)
FloatArray: TypeAlias = NDArray[np.float64]
GP_JITTER = 1e-8


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def require_heldout_trajectories() -> None:
    cfg = protocol()
    paths = [THERMAL/f"{temperature}C"/f"{profile}_trajectories.npz"
             for temperature in cfg["test_temperatures_C"] for profile in cfg["test_profiles"]]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing held-out thermal trajectories:\n  " + "\n  ".join(missing)
            + "\nGenerate them from raw datasets with python exp_a123_temperature.py "
              "in a disposable copy; that script overwrites saved thermal results. "
              "See REPRODUCIBILITY.md for integrity limitations."
        )


def verify_saved(path: Path) -> None:
    verify_public(path)


def snapshot() -> dict[str, str]:
    return {str(p): sha256(p) for p in sorted((ROOT/"results").rglob("*"))
            if p.is_file() and not p.is_relative_to(OUTPUT)} | {
                str(p): sha256(p) for p in sorted(ROOT.glob("report*.md")) if p.name != "report_temperature_interp.md"}


def transform(parameters: FloatArray) -> FloatArray:
    result = np.asarray(parameters, dtype=float).copy()
    if not np.isfinite(result).all() or np.any(result[..., LOG_COORDS] <= 0):
        raise ValueError("Nonfinite/nonpositive physical interpolation parameters")
    result[..., LOG_COORDS] = np.log(result[..., LOG_COORDS])
    return result


def inverse(transformed: FloatArray) -> FloatArray:
    result = np.asarray(transformed, dtype=float).copy()
    result[..., LOG_COORDS] = np.exp(result[..., LOG_COORDS])
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite inverse-transformed parameters")
    return result


def physical_bounds(q25: float) -> tuple[FloatArray, FloatArray]:
    b = protocol()["bounds"]
    return (np.array([q25*b["Q_ratio"][0], b["z0"][0], *([b["R0_R1_R2_ohm"][0]]*3),
                      *([b["tau1_tau2_s"][0]]*2)]),
            np.array([q25*b["Q_ratio"][1], b["z0"][1], *([b["R0_R1_R2_ohm"][1]]*3),
                      *([b["tau1_tau2_s"][1]]*2)]))


def constrain(values: FloatArray, q25: float) -> tuple[FloatArray, dict]:
    if np.asarray(values).shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Expected seven finite physical parameters")
    lo, hi = physical_bounds(q25)
    constrained = np.clip(values, lo, hi)
    clips = {PARAMETERS[k]: {"original": float(values[k]), "projected": float(constrained[k])}
             for k in np.flatnonzero(constrained != values)}
    swapped = bool(constrained[5] > constrained[6])
    if swapped:
        constrained[[3, 4, 5, 6]] = constrained[[4, 3, 6, 5]]
    return constrained, {"projected": clips, "branch_swap": swapped}


def parameter_vector(fit: dict) -> FloatArray:
    p = fit["parameters"]
    return np.array([p["Qnom_Ah"], p["z0"], p["R0"], p["R1"], p["R2"], p["tau1_s"], p["tau2_s"]])


def simulation_theta(values: FloatArray, q25: float) -> FloatArray:
    q, z0, r0, r1, r2, tau1, tau2 = values
    return np.r_[np.log([r0, r1, r2, tau1, tau2]), q/q25, z0]


def design(temperatures: FloatArray, log_coordinate: bool) -> FloatArray:
    t = np.asarray(temperatures, dtype=float)
    if not np.isfinite(t).all() or np.any(t <= -273.15):
        raise ValueError("Temperatures must be finite and above absolute zero")
    x = 1/(t+273.15) if log_coordinate else t
    return np.column_stack((np.ones(len(t)), x))


def kernel(a: FloatArray, b: FloatArray, sigma: float, ell: float) -> FloatArray:
    return sigma*sigma*np.exp(-.5*((a[:, None]-b[None, :])/ell)**2)


@dataclass
class ParameterGP:
    temperatures: FloatArray
    beta: FloatArray
    log_coordinate: bool
    scale: float
    residuals: FloatArray
    sigma: float
    ell: float
    attempts: list[dict]

    def mean(self, temperatures: FloatArray) -> FloatArray:
        return design(temperatures, self.log_coordinate)@self.beta

    def posterior(self, temperatures: FloatArray) -> tuple[FloatArray, FloatArray]:
        t = np.asarray(temperatures, dtype=float)
        covariance = kernel(self.temperatures, self.temperatures, self.sigma, self.ell)
        covariance += np.eye(len(self.temperatures))*GP_JITTER
        cross = kernel(t, self.temperatures, self.sigma, self.ell)
        mean = self.mean(t)+self.scale*(cross@np.linalg.solve(covariance, self.residuals))
        posterior = (kernel(t, t, self.sigma, self.ell)
                     -cross@np.linalg.solve(covariance, cross.T))*self.scale**2
        posterior = (posterior+posterior.T)*.5
        eigenvalues, eigenvectors = np.linalg.eigh(posterior)
        tolerance = max(1., float(np.max(np.abs(posterior))))*1e-10
        if eigenvalues.min() < -tolerance:
            raise FloatingPointError("GP posterior covariance is not positive semidefinite")
        posterior = (eigenvectors*np.maximum(eigenvalues, 0))@eigenvectors.T
        return mean, posterior

    def to_dict(self) -> dict:
        return {"calibration_temperatures_C": self.temperatures.tolist(), "beta_P3": self.beta.tolist(),
                "log_coordinate": self.log_coordinate, "standardization_scale": self.scale,
                "standardized_residuals": self.residuals.tolist(),
                "sigma_f_standardized": self.sigma, "ell_C": self.ell, "attempts": self.attempts}


def fit_gp(temperatures: FloatArray, values: FloatArray, index: int) -> ParameterGP:
    cfg = protocol()["P4"]
    log_coordinate = bool(LOG_COORDS[index])
    beta = np.linalg.lstsq(design(temperatures, log_coordinate), values, rcond=None)[0]
    scale = float(max(np.std(values), 1e-6))
    residuals = (values-design(temperatures, log_coordinate)@beta)/scale
    b = cfg["hyperparameter_bounds"]
    bounds = list(zip(np.log([b["sigma_f_standardized"][0], b["ell_C"][0]]),
                      np.log([b["sigma_f_standardized"][1], b["ell_C"][1]])))

    def objective(theta):
        sigma, ell = np.exp(theta)
        covariance = kernel(temperatures, temperatures, sigma, ell)+np.eye(3)*GP_JITTER
        chol = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(chol, residuals)
        nll = .5*np.dot(alpha, alpha)+np.log(np.diag(chol)).sum()+1.5*np.log(2*np.pi)
        return float(nll+.5*((theta[1]-np.log(25.))/2.)**2)

    attempts, successful = [], []
    for index_start, start in enumerate(cfg["hyperparameter_starts"]):
        result = minimize(objective, np.log(start), method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
        attempt = {"start_index": index_start, "success": bool(result.success),
                   "message": str(result.message), "penalized_negative_log_likelihood": float(result.fun),
                   "sigma_f": float(np.exp(result.x[0])), "ell_C": float(np.exp(result.x[1])),
                   "nit": int(result.nit)}
        attempts.append(attempt)
        LOGGER.info("GP_FIT %s %s", PARAMETERS[index], json.dumps(attempt))
        if result.success:
            successful.append((float(result.fun), index_start, result))
        else:
            LOGGER.error("GP optimization failure %s: %s", PARAMETERS[index], attempt)
    if not successful:
        save_json(OUTPUT/f"FAILED_GP_{PARAMETERS[index]}.json", {"attempts": attempts})
        raise RuntimeError(f"No converged GP optimization: {PARAMETERS[index]}")
    _, _, selected = min(successful, key=lambda x: x[0])
    sigma, ell = np.exp(selected.x)
    return ParameterGP(temperatures.copy(), beta, log_coordinate, scale, residuals,
                       float(sigma), float(ell), attempts)


def sample_posterior(gps: list[ParameterGP], temperatures: FloatArray, q25: float) -> tuple[FloatArray, dict]:
    cfg = protocol()["P4"]
    count = cfg["samples"]
    rng = np.random.default_rng(cfg["seed"])
    lo, hi = physical_bounds(q25)
    lower, upper = transform(lo), transform(hi)
    samples = np.empty((count, len(temperatures), 7))
    metadata = {}
    for index, gp in enumerate(gps):
        mean, covariance = gp.posterior(temperatures)
        accepted, proposals = [], 0
        while sum(len(a) for a in accepted) < count and proposals < 1000000:
            batch = min(1024, 1000000-proposals)
            draw = rng.multivariate_normal(mean, covariance, size=batch, check_valid="raise")
            good = ((draw >= lower[index]) & (draw <= upper[index])).all(axis=1)
            accepted.append(draw[good])
            proposals += batch
        if sum(len(a) for a in accepted) < count:
            raise RuntimeError(f"Bound-conditioned posterior acceptance exhausted: {PARAMETERS[index]}")
        chosen = np.concatenate(accepted)[:count]
        samples[:, :, index] = np.exp(chosen) if LOG_COORDS[index] else chosen
        metadata[PARAMETERS[index]] = {"proposed": proposals,
                                      "accepted_in_proposals": sum(len(a) for a in accepted),
                                      "retained": count, "posterior_mean_transformed": mean.tolist(),
                                      "posterior_covariance_transformed": covariance.tolist()}
    swap_count = 0
    for k in range(count):
        for j in range(len(temperatures)):
            if samples[k, j, 5] > samples[k, j, 6]:
                samples[k, j, [3, 4, 5, 6]] = samples[k, j, [4, 3, 6, 5]]
                swap_count += 1
    metadata["branch_swaps"] = swap_count
    return samples, metadata


def load_calibration() -> tuple[FloatArray, float, OCVCurves, dict]:
    for path in (ROOT/"exp_a123_temperature.py", ROOT/"protocols"/"a123_temperature_diagnostic.json",
                 ROOT/"protocols"/"lfp_static_protocol.json"):
        verify_saved(path)
    source, curves = load_source()
    q25 = source["qnom_measured_Ah"]
    fits, provenance = {}, {}
    for temperature in protocol()["calibration_temperatures_C"]:
        path = SOURCE/"summary.json" if temperature == 25 else THERMAL/f"{temperature}C"/"summary.json"
        verify_saved(path)
        saved = json.loads(path.read_text())
        fit = saved["fits"]["V6"] if temperature == 25 else saved["local_V6_fit"]
        meta = saved["manifest"]["loader"] if temperature == 25 else saved["loader"]
        for raw, digest in meta["source_files"].items():
            if sha256(DATA_ROOT/Path(raw).relative_to("datasets")) != digest:
                raise ValueError(f"Calibration source changed: {raw}")
        if fit["identified_on"] != "DST" or fit["target_refit"] or fit["parameters"]["ocv_coefficients_V"] is not None:
            raise ValueError("Calibration fit is not the registered source-only V6")
        if temperature != 25 and (meta["specimen"] != "A1-007" or meta["Qnom_reference_Ah"] != q25
                                   or fit["source_temperature_C"] != temperature):
            raise ValueError("Calibration specimen/temperature/Q reference mismatch")
        # Check theta against saved physical values and the exact shared unpacking.
        dummy = CellData("cell_A_a123", q25, curves.charge_z, curves.charge_V,
                         curves.discharge_z, curves.discharge_V, {}, {})
        cycle = Cycle("DST", np.arange(2.), np.zeros(2), np.zeros(2), np.zeros(2),
                      np.zeros(2), np.ones(2), 1., 1., {})
        recomputed = unpack(np.asarray(fit["theta"]), "V6", dummy, cycle)
        if recomputed != fit["parameters"]:
            raise ValueError("Saved calibration theta/physical parameters mismatch")
        vector = parameter_vector(fit)
        bounded, diag = constrain(vector, q25)
        if diag["projected"] or diag["branch_swap"] or not np.array_equal(vector, bounded):
            raise ValueError("Saved calibration fit violates parameter bounds/alignment")
        successful = [a for a in fit["attempts"] if a["success"]]
        if not successful or fit["parameters"]["Q_ratio"] != vector[0]/q25:
            raise ValueError("Invalid calibration optimizer manifest")
        selected = min(successful, key=lambda a: a["squared_residual_V2"])
        if selected["theta"] != fit["theta"]:
            raise ValueError("Calibration fit did not select lowest converged source squared residual")
        fits[str(temperature)] = fit
        provenance[str(temperature)] = {"path": str(path), "sha256": sha256(path),
                                        "reuse": True, "identical_protocol": True, "fit": fit,
                                        "raw_source_sha256": meta["source_files"]}
    return np.stack([parameter_vector(fits[str(t)]) for t in protocol()["calibration_temperatures_C"]]), q25, curves, provenance


def fit_predictions(calibration: FloatArray, q25: float) -> tuple[dict, list[ParameterGP], FloatArray, dict]:
    temperatures = np.asarray(protocol()["calibration_temperatures_C"], dtype=float)
    test = np.asarray(protocol()["test_temperatures_C"], dtype=float)
    y = transform(calibration)
    gps = [fit_gp(temperatures, y[:, k], k) for k in range(7)]
    predictions = {}
    for t in test:
        closest = int(np.argmin(np.abs(temperatures-t)))
        p2 = np.array([np.interp(t, temperatures, y[:, k]) for k in range(7)])
        p3 = np.array([gp.mean(np.array([t]))[0] for gp in gps])
        p4 = np.array([gp.posterior(np.array([t]))[0][0] for gp in gps])
        predictions[str(int(t))] = {}
        for method, physical in (("P1", calibration[closest]), ("P2", inverse(p2)),
                                 ("P3", inverse(p3)), ("P4", inverse(p4))):
            bounded, diagnostics = constrain(physical, q25)
            predictions[str(int(t))][method] = {"parameters": dict(zip(PARAMETERS, bounded.tolist())),
                                                "theta_V6": simulation_theta(bounded, q25).tolist(),
                                                "constraints": diagnostics}
    samples, sample_meta = sample_posterior(gps, test, q25)
    return predictions, gps, samples, sample_meta


def decision(rows: list[dict]) -> dict:
    expected = {(t, profile) for t in protocol()["test_temperatures_C"] for profile in protocol()["test_profiles"]}
    conditions = {}
    coverages = []
    for temperature, profile in sorted(expected):
        subset = {r["method"]: r for r in rows if r["temperature_C"] == temperature and r["profile"] == profile}
        if set(subset) != set(METHODS):
            raise ValueError("Decision requires exactly P1-P4 in every held-out condition")
        fractions = {m: subset[m]["recovered_fraction"] for m in ("P2", "P3", "P4")}
        available = {m: value for m, value in fractions.items() if value is not None}
        best = max(available, key=available.get) if available else None
        conditions[f"{temperature}C_{profile}"] = {"best_method": best, "fractions": fractions,
                                                   "passes": best is not None and available[best] >= .5}
        coverages.append(subset["P4"]["coverage_95"])
    calibrated = all(c is not None and .85 <= c <= .99 for c in coverages)
    return {"verdict": "POSITIVO" if all(c["passes"] for c in conditions.values()) else "NEGATIVO",
            "conditions": conditions, "P4_uncertainty": "calibrada" if calibrated else "nao calibrada",
            "coverage_diagnostic_outside_main_criterion": True}


def voltage_band(voltages: FloatArray, measured: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    if voltages.shape != (protocol()["P4"]["samples"], len(measured)) or not np.isfinite(voltages).all():
        raise ValueError("P4 bands require 200 finite voltage trajectories")
    low, high = np.quantile(voltages, [.025, .975], axis=0)
    return low, high, float(np.mean((measured >= low) & (measured <= high)))


def evaluate(predictions: dict, samples: FloatArray, q25: float, curves: OCVCurves) -> tuple[list, list, dict]:
    rows, stratified, provenance = [], [], {}
    for j, temperature in enumerate(protocol()["test_temperatures_C"]):
        prior_path = THERMAL/f"{temperature}C"/"summary.json"
        verify_saved(prior_path)
        old = json.loads(prior_path.read_text())
        for path, digest in old["loader"]["source_files"].items():
            if sha256(DATA_ROOT/Path(path).relative_to("datasets")) != digest:
                raise ValueError("Held-out raw source changed")
        provenance[str(temperature)] = {"prior_summary_path": str(prior_path), "sha256": sha256(prior_path),
                                        "raw_sha256": old["loader"]["source_files"], "trajectories": {}}
        for profile in protocol()["test_profiles"]:
            prior_trace = THERMAL/f"{temperature}C"/f"{profile}_trajectories.npz"
            verify_saved(prior_trace)
            provenance[str(temperature)]["trajectories"][profile] = {"path": str(prior_trace), "sha256": sha256(prior_trace)}
            with np.load(prior_trace) as saved:
                trace = {k: saved[k] for k in saved.files}
            cycle = Cycle(profile, trace["t"], trace["current_A"], trace["measured_voltage_V"],
                          trace["temperature_C"], trace["net_Ah"], trace["original_soc"],
                          1., 1., old["cycles"][profile])
            cell = CellData("cell_A_a123", q25, curves.charge_z, curves.charge_V,
                            curves.discharge_z, curves.discharge_V, {profile: cycle}, {})
            floor = metrics(trace["V6"]-cycle.voltage)["mae"]
            ceiling = metrics(trace["V6_local"]-cycle.voltage)["mae"]
            if abs(floor-old["global_mae_mV"][profile]["V6"]) > 1e-10 or abs(
                    ceiling-old["global_mae_mV"][profile]["V6_local"]) > 1e-10:
                raise ValueError("Saved floor/ceiling metrics mismatch")
            denominator = floor-ceiling
            if denominator <= 0:
                LOGGER.warning("Unavailable recovered fraction at %sC %s: denominator=%s",
                               temperature, profile, denominator)
            voltages = {m: predict(cycle, cell, curves,
                                   np.asarray(predictions[str(temperature)][m]["theta_V6"]), "V6")[0]
                        for m in METHODS}
            posterior_voltage = np.stack([predict(cycle, cell, curves, simulation_theta(p, q25), "V6")[0]
                                          for p in samples[:, j, :]])
            low, high, coverage = voltage_band(posterior_voltage, cycle.voltage)
            out = OUTPUT/f"{temperature}C"
            out.mkdir(exist_ok=True)
            np.savez_compressed(out/f"{profile}_trajectories.npz", t=cycle.t, current_A=cycle.current,
                                temperature_C=cycle.temperature, measured_voltage_V=cycle.voltage,
                                net_Ah=cycle.net_Ah, original_soc=cycle.soc, floor=trace["V6"],
                                ceiling=trace["V6_local"], **voltages, P4_lower95_V=low, P4_upper95_V=high,
                                P4_sample_voltage_V=posterior_voltage)
            scopes = masks(cycle, .02, 60)
            for m, voltage in voltages.items():
                values = metrics(voltage-cycle.voltage)
                fraction = (floor-values["mae"])/denominator if denominator > 0 else None
                rows.append({"temperature_C": temperature, "profile": profile, "method": m,
                             **{metric+"_mV": values[metric] for metric in METRICS},
                             "floor_mae_mV": floor, "ceiling_mae_mV": ceiling,
                             "recovered_fraction": fraction, "coverage_95": coverage if m == "P4" else None,
                             "band_mean_width_mV": float(np.mean(high-low)*1000) if m == "P4" else None})
                for scope, mask in scopes.items():
                    if mask.any():
                        value = metrics(voltage[mask]-cycle.voltage[mask])
                    else:
                        LOGGER.warning("Empty scope %s %s %s %s", temperature, profile, m, scope)
                        value = {k: None for k in METRICS}
                    stratified.append({"temperature_C": temperature, "profile": profile, "method": m,
                                       "scope": scope, "n_samples": int(mask.sum()),
                                       **{k+"_mV": value[k] for k in METRICS}})
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            hours = cycle.t/3600
            axes[0].fill_between(hours, low, high, alpha=.35, color="tab:blue", label="P4 95% parameter band")
            axes[0].plot(hours, cycle.voltage, "k", lw=.7, label="measured")
            axes[0].plot(hours, voltages["P4"], lw=.6, label="P4 central")
            for m, voltage in voltages.items():
                axes[1].plot(hours, (voltage-cycle.voltage)*1000, lw=.6, label=m)
            axes[0].set(ylabel="Voltage (V)", title=f"A123 {temperature}C {profile}; coverage={coverage:.2%}")
            axes[0].legend()
            axes[1].set(xlabel="Time (h)", ylabel="Error (mV)")
            axes[1].axhline(0, color="black", lw=.5)
            axes[1].legend()
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(out/f"{profile}_voltage_band.{suffix}", dpi=150)
            plt.close(fig)
            LOGGER.info("EVALUATED %sC %s floor=%s ceiling=%s P4coverage=%s", temperature, profile, floor, ceiling, coverage)
    return rows, stratified, provenance


def plot_parameters(calibration: FloatArray, gps: list[ParameterGP], q25: float) -> dict:
    temperatures = np.asarray(protocol()["calibration_temperatures_C"], dtype=float)
    grid = np.linspace(0, 50, 251)
    transformed = transform(calibration)
    p2 = np.column_stack([np.interp(grid, temperatures, transformed[:, k]) for k in range(7)])
    p3 = np.column_stack([gp.mean(grid) for gp in gps])
    posteriors = [gp.posterior(grid) for gp in gps]
    p4 = np.column_stack([post[0] for post in posteriors])
    sd = np.column_stack([np.sqrt(np.maximum(np.diag(post[1]), 0)) for post in posteriors])
    low, high = inverse(p4+norm.ppf(.025)*sd), inverse(p4+norm.ppf(.975)*sd)
    central = {m: np.stack([constrain(p, q25)[0] for p in inverse(y)])
               for m, y in (("P2", p2), ("P3", p3), ("P4", p4))}
    lo, hi = physical_bounds(q25)
    for k, name in enumerate(PARAMETERS):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(grid, low[:, k], high[:, k], alpha=.25, label="P4 Gaussian marginal 95% (unconditioned)")
        for m, values in central.items():
            ax.plot(grid, values[:, k], label=m+" bounded central")
        ax.scatter(temperatures, calibration[:, k], color="black", zorder=5, label="DST calibration only")
        for t in protocol()["test_temperatures_C"]:
            ax.axvline(t, color="gray", lw=.5, linestyle=":")
        ax.axhline(lo[k], color="red", lw=.5, linestyle="--")
        ax.axhline(hi[k], color="red", lw=.5, linestyle="--", label="Physical bounds")
        if LOG_COORDS[k]:
            ax.set_yscale("log")
            plot_low = min(low[:, k].min(), calibration[:, k].min())
            plot_high = max(high[:, k].max(), calibration[:, k].max())
            padding = max(np.log(plot_high/plot_low)*.1, .02)
            ax.set_ylim(plot_low*np.exp(-padding), plot_high*np.exp(padding))
        else:
            plot_low = min(low[:, k].min(), calibration[:, k].min())
            plot_high = max(high[:, k].max(), calibration[:, k].max())
            padding = max((plot_high-plot_low)*.1, 1e-6)
            ax.set_ylim(plot_low-padding, plot_high+padding)
        ax.set(xlabel="Nominal temperature (C)", ylabel=name, title="A123 "+name)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT/f"parameter_{name}.{suffix}", dpi=150)
        plt.close(fig)
    values = {"temperature_C": grid, "P2": central["P2"], "P3": central["P3"], "P4": central["P4"],
              "P4_unconditioned_lower95": low, "P4_unconditioned_upper95": high}
    np.savez_compressed(OUTPUT/"parameter_curves.npz", **values)
    return {m: {"projected_grid_coordinates": int(sum(bool(constrain(p, q25)[1]["projected"]) for p in inverse(y)))}
            for m, y in (("P2", p2), ("P3", p3), ("P4", p4))}


def write_report(summary: dict) -> None:
    d = summary["decision"]
    verdicts = {"POSITIVO": "POSITIVE", "NEGATIVO": "NEGATIVE"}
    uncertainty = {"calibrada": "calibrated", "nao calibrada": "not calibrated"}
    lines = ["# A123: temperature parameter interpolation", "",
             f"Protocol frozen at `{REGISTRATION}` before new P3/P4 fits.", "",
             f'## Verdict: **{verdicts[d["verdict"]]}**', "",
             "Criterion: the best of P2/P3/P4 recovers >=50% of the floor-to-ceiling interval in ALL four conditions.",
             "Calibration: DST0/25/50, saved and verified V6 fits. Test: FUDS/US06 at 10/40C.",
             "No local 10/40 data or fits were used for calibration or hyperparameter selection.", "",
             "| T(C) | Profile | Method | MAE(mV) | Floor(mV) | Ceiling(mV) | Fraction recovered | P4 coverage |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        fraction = "--" if row["recovered_fraction"] is None else f'{row["recovered_fraction"]:.2%}'
        coverage = "--" if row["coverage_95"] is None else f'{row["coverage_95"]:.2%}'
        lines.append(f'| {row["temperature_C"]} | {row["profile"]} | {row["method"]} | '
                     f'{row["mae_mV"]:.3f} | {row["floor_mae_mV"]:.3f} | {row["ceiling_mae_mV"]:.3f} | '
                     f'{fraction} | {coverage} |')
    lines += ["", "### Best method per condition", ""]
    for condition, value in d["conditions"].items():
        best = value["best_method"]
        fraction = "--" if best is None else f'{value["fractions"][best]:.2%}'
        lines.append(f'- {condition}: {best}, fraction {fraction}, passes={value["passes"]}.')
    lines += ["", f'## P4 uncertainty: **{uncertainty[d["P4_uncertainty"]]}** (diagnostic outside the criterion)', "",
              "Required coverage 85--99% inclusive in ALL conditions.",
              "Pointwise 95% band: quantiles of 200 simulations with posterior parameter sets.",
              "Seven independent GPs; correlation between 10/40 preserved per parameter; the same sets",
              "used for both profiles. Posterior conditioned on bounds by rejection, not clipping.",
              "Central P4 prediction: transformed posterior mean, inversion/projection, and deterministic simulation.",
              "Parameter-interpolation uncertainty only: no voltage noise, NLS uncertainty,",
              "P3 mean uncertainty, or hyperparameter uncertainty. Three temperatures do not guarantee",
              "hyperparameter identifiability or coverage calibration.", "",
              "### GP hyperparameters and rejections", "",
              "| Parameter | Standardized sigma | ell(C) | Proposed | Accepted | Retained |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, gp in summary["GPs"].items():
        sample = summary["posterior_sampling"][name]
        lines.append(f'| {name} | {gp["sigma_f_standardized"]:.6g} | {gp["ell_C"]:.4f} | '
                     f'{sample["proposed"]} | {sample["accepted_in_proposals"]} | {sample["retained"]} |')
    failures = [name for name, gp in summary["GPs"].items() if any(not a["success"] for a in gp["attempts"])]
    lines += ["", "Nonconvergent GP attempts: "+(", ".join(failures) if failures else "none")+
              ". Each parameter uses only the best CONVERGENT attempt; all starts and states are in the manifest."]
    lines += ["", "Weak prior: log ell ~ Normal(log25,2^2); penalized marginal likelihood/MAP.",
              "RBF kernel, fixed standardized jitter 1e-8; amplitude and ell fitted only on 0/25/50 with three starts.",
              "Ranges: sigma[1e-6,10], ell[1,200]C. No target MAE optimization.", "",
              "## Assumptions and constraints", "",
              "- A1-007 only; OCV_mean25 fixed and measured Q25 as reference.",
              "- Existing V6 at 0/25/50 fully reused; no new RC identification on DST.",
              "- P1 nearest; P2 linear in T (logs for R/tau); P3 Arrhenius for R/tau and linear for Q/z0.",
              "- Branch labels are fast/slow, ordered by tau; crossings swap R/tau together.",
              "- Central predictions projected to bounds, as confirmed before the commit.",
              "- Parameter figures show UNCONDITIONED Gaussian bands describing the GP;",
              "  voltage bands use conditioned samples. These are explicitly distinct distributions.",
              "- Causal SOC and continuous RC states, initial zero; 1s grid and same prior masks/thresholds.",
              "- Floor and ceiling are saved V6 references; local ceiling is an evaluation oracle only,",
              "  not a guaranteed mathematical bound on MAE. Signed fractions without clipping.",
              "- Values at bounds (Q0C and tau0/50C) are bounded NLS estimates, not exact observations",
              "  of physics; the GP treats them as exact points, an important limitation.",
              "- Best method can differ by condition; no single universal method is selected.", "",
              "## Artifacts and verification", "",
              "- [Full manifest](results/a123_temperature_interp/summary.json)",
              "- [Parameters saved BEFORE evaluation](results/a123_temperature_interp/parameter_predictions.json)",
              "- [Table](results/a123_temperature_interp/table.csv)",
              "- [Tests](results/a123_temperature_interp/tests.log)",
              "- [Protocol](a123_temperature_interp_protocol.json)",
              "- Seven parameter(T) plots, P4 voltage bands by condition, 200 parameter/voltage samples",
              "  and saved posterior mean/covariance; all previous results preserved by SHA-256.", ""]
    (ROOT/"report_temperature_interp.md").write_text("\n".join(lines))


def main() -> None:
    require_heldout_trajectories()
    if sha256(PROTOCOL_PATH) != "f0c6b8718383828856b3578cd96c9a2d25983d5b0cf4b48d83e4a9d25482a3ba":
        raise ValueError("Published interpolation protocol differs from its original registration")
    previous = snapshot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUTPUT/"run.log", mode="w"), logging.StreamHandler()])
    save_json(OUTPUT/"previous_artifacts_sha256.json", previous)
    calibration, q25, curves, provenance = load_calibration()
    predictions, gps, samples, sampling = fit_predictions(calibration, q25)
    saved_predictions = {"calibration_parameters": calibration.tolist(), "parameter_order": PARAMETERS,
                         "calibration_provenance": provenance, "Qnom_reference_Ah": q25,
                         "predictions": predictions, "GPs": {n: g.to_dict() for n, g in zip(PARAMETERS, gps)},
                         "posterior_sampling": sampling, "protocol_sha256": sha256(PROTOCOL_PATH)}
    save_json(OUTPUT/"parameter_predictions.json", saved_predictions)
    np.savez_compressed(OUTPUT/"posterior_parameter_samples.npz", parameter_samples=samples,
                        temperature_C=np.asarray(protocol()["test_temperatures_C"]), parameter_order=PARAMETERS)
    LOGGER.info("CALIBRATION_COMPLETE; parameter predictions persisted before any held-out evaluation")
    curve_diagnostics = plot_parameters(calibration, gps, q25)
    rows, stratified, heldout = evaluate(predictions, samples, q25, curves)
    write_tables(OUTPUT, rows)
    (OUTPUT/"stratified").mkdir(exist_ok=True)
    write_tables(OUTPUT/"stratified", stratified)
    summary = {**saved_predictions, "complete": True, "metrics": rows, "stratified_metrics": stratified,
               "decision": decision(rows), "parameter_curve_diagnostics": curve_diagnostics,
               "manifest": {"protocol": protocol(), "protocol_sha256": sha256(PROTOCOL_PATH),
                            "preregistration_commit": REGISTRATION,
                            "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                                        text=True).strip(),
                            "source_sha256": {str(ROOT/p): sha256(ROOT/p) for p in
                                              ("exp_a123_temperature_interp.py", "exp_a123_temperature.py",
                                               "exp_lfp_static.py", "exp_lfp_gonogo.py", "lfp_ocv.py", "data_lfp.py",
                                               "protocols/lfp_static_protocol.json")},
                            "heldout_reference_provenance": heldout, "python": platform.python_version(),
                            "platform": platform.platform(),
                            "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")},
                            "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                            "previous_artifact_count": len(previous),
                            "parameter_predictions_sha256": sha256(OUTPUT/"parameter_predictions.json"),
                            "posterior_parameter_samples_sha256": sha256(OUTPUT/"posterior_parameter_samples.npz")}}
    if snapshot() != previous:
        raise RuntimeError("Earlier artifacts changed during interpolation")
    save_json(OUTPUT/"summary.json", summary)
    write_report(summary)
    LOGGER.info("COMPLETE %s", json.dumps(summary["decision"]))


if __name__ == "__main__":
    main()
