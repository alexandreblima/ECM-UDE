"""Source-only V7 with monotonic OCV and constant source-support boundaries."""
from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
import subprocess
import importlib.metadata
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression, least_squares
from scipy.signal import lfilter
from scipy.special import expit

from data_lfp import ROOT, CellData, Cycle, load_a123, load_mendeley, save_json, sha256
from exp_lfp_static import bounds_starts, unpack
from exp_lfp_gonogo import METRICS, masks, metrics, rc_voltage
from exp_static_replication import load_dataset, write_tables
from lfp_ocv import OCVCurves
from public_integrity import verify as verify_public

OUTPUT = ROOT/"results"/"static_monotone"
PROTOCOL_PATH = ROOT/"protocols"/"paper_v2_monotone_protocol.json"
REGISTRATION = "8a0e2e2"
LOGGER = logging.getLogger(__name__)
FAILURE_POLICY = ROOT/"protocols"/"paper_v2_monotone_failure_policy.json"


class IdentificationFailure(RuntimeError):
    def __init__(self, cell: str, attempts: list[dict]) -> None:
        super().__init__(f"No converged V7m start for {cell}")
        self.attempts = attempts


def monotone_ocv(theta: np.ndarray, source: Cycle, q25: float) -> tuple[np.ndarray, np.ndarray]:
    if len(theta) != 18:
        raise ValueError("V7m requires sevenRC/Q/z0 plus11OCV parameters")
    q, z0 = q25*theta[5], theta[6]
    source_soc = z0-source.net_Ah/q
    lo, hi = float(source_soc.min()), float(source_soc.max())
    if not np.isfinite(source_soc).all() or hi <= lo:
        raise ValueError("SourceSOC must have nonzero finite support")
    z = np.linspace(lo, hi, 11)
    voltage = np.r_[theta[7], theta[7]+np.cumsum(np.logaddexp(0., theta[8:]))]
    return z, voltage


def simulate(theta: np.ndarray, cycle: Cycle, source: Cycle, qnom: float) -> tuple[np.ndarray, np.ndarray]:
    r0, r1, r2, tau1, tau2 = np.exp(theta[:5])
    z0, q = theta[6], qnom*theta[5]
    z = z0-cycle.net_Ah/q
    knots, voltage = monotone_ocv(theta, source, qnom)
    # Candidate SOC support is affine in Q/z0, which cancel exactly in this coordinate.
    ah_min, ah_max = float(source.net_Ah.min()), float(source.net_Ah.max())
    coordinate = np.clip((ah_max-cycle.net_Ah)/(ah_max-ah_min), 0., 1.)
    ocv = PchipInterpolator(np.linspace(0., 1., 11), voltage, extrapolate=False)(coordinate)
    result = ocv-r0*cycle.current-rc_voltage(cycle.current, r1, tau1)-rc_voltage(cycle.current, r2, tau2)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite monotone voltage")
    return result, z


def source_jacobian(theta: np.ndarray, source: Cycle, qnom: float) -> np.ndarray:
    """Exact positive-slope PCHIP and RC derivatives, including two zero SOC columns."""
    _, voltage = monotone_ocv(theta, source, qnom)
    h = .1
    dy = np.zeros((11, 11))
    dy[:, 0] = 1.
    for k in range(10):
        dy[k+1:, k+1] = expit(theta[k+8])
    slopes = np.diff(voltage)/h
    ds = np.diff(dy, axis=0)/h
    dd = np.zeros_like(dy)
    if 3*slopes[0]-slopes[1] > 0:
        dd[0] = (3*ds[0]-ds[1])/2
    if 3*slopes[-1]-slopes[-2] > 0:
        dd[-1] = (3*ds[-1]-ds[-2])/2
    for k in range(1, 10):
        left, right = slopes[k-1], slopes[k]
        dd[k] = 2*(right**2*ds[k-1]+left**2*ds[k])/(left+right)**2
    ah_min, ah_max = float(source.net_Ah.min()), float(source.net_Ah.max())
    x = np.clip((ah_max-source.net_Ah)/(ah_max-ah_min), 0., 1.)
    interval = np.minimum(np.floor(x/h).astype(int), 9)
    u = (x-interval*h)/h
    h00, h10 = 2*u**3-3*u**2+1, u**3-2*u**2+u
    h01, h11 = -2*u**3+3*u**2, u**3-u**2
    jac = np.zeros((len(source.t), 18))
    jac[:, 7:] = (h00[:, None]*dy[interval]+h10[:, None]*h*dd[interval]+
                  h01[:, None]*dy[interval+1]+h11[:, None]*h*dd[interval+1])
    r0, r1, r2, tau1, tau2 = np.exp(theta[:5])
    jac[:, 0] = -r0*source.current
    for amplitude_column, tau_column, r, tau in ((1, 3, r1, tau1), (2, 4, r2, tau2)):
        rc = rc_voltage(source.current, r, tau)
        decay = np.exp(-1/tau)
        jac[:, amplitude_column] = -rc
        jac[:, tau_column] = -lfilter([0., decay/tau], [1., -decay], rc-r*source.current)
    return jac


def fit(cell: CellData, curves: OCVCurves, source_name: str) -> dict:
    source = cell.cycles[source_name]
    lo, hi, base_starts = bounds_starts(cell, source, curves, "V6")
    lower, upper = np.r_[lo, 1., np.full(10, -20.)], np.r_[hi, 5., np.full(10, 1.)]
    attempts, successful = [], []
    for index, base in enumerate(base_starts):
        soc = base[6]-source.net_Ah/(cell.qnom_Ah*base[5])
        grid = np.linspace(soc.min(), soc.max(), 11)
        v = isotonic_regression(curves.evaluate(grid)[2], increasing=True).x
        increments = np.maximum(np.diff(v), 1e-5)
        initial = np.r_[base, v[0], np.log(np.expm1(increments))]
        initial = np.clip(initial, lower, upper)
        LOGGER.info("FIT_V7m %s %s start=%s", cell.name, source_name, index)
        result = least_squares(lambda th: simulate(th, source, source, cell.qnom_Ah)[0]-source.voltage,
                               initial, bounds=(lower, upper), method="trf", loss="linear", x_scale="jac",
                               jac=lambda th: source_jacobian(th, source, cell.qnom_Ah),
                               max_nfev=500, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        attempt = {"start_index": index, "success": bool(result.success), "message": str(result.message),
                   "nfev": int(result.nfev), "squared_residual_V2": float(2*result.cost),
                   "theta": result.x.tolist()}
        attempts.append(attempt)
        LOGGER.info("RESULT %s %s", cell.name, json.dumps(attempt))
        if result.success:
            successful.append((float(result.cost), index, result))
        else:
            LOGGER.error("V7m NLS failed: %s", attempt)
    if not successful:
        save_json(OUTPUT/cell.name/"FAILED.json", {"attempts": attempts})
        raise IdentificationFailure(cell.name, attempts)
    _, index, chosen = min(successful, key=lambda t: t[0])
    params = unpack(chosen.x[:7], "V6", cell, source)
    knots, voltage = monotone_ocv(chosen.x, source, cell.qnom_Ah)
    proximity = np.minimum((chosen.x-lower)/(upper-lower), (upper-chosen.x)/(upper-lower))
    names = ["log_R0", "log_R1", "log_R2", "log_tau1", "log_tau2", "Q_ratio", "z0", "OCV_v0"]+[
        f"OCV_increment{k}" for k in range(1, 11)]
    hits = [names[k] for k in np.flatnonzero(proximity < 1e-4)]
    return {"model": "V7m", "source": source_name, "target_refit": False, "theta": chosen.x.tolist(),
            "parameters": params, "OCV_knots_soc": knots.tolist(), "OCV_knots_V": voltage.tolist(),
            "source_SOC_support": [float(knots[0]), float(knots[-1])],
            "attempts": attempts, "selected_start": index, "near_bound_parameters": hits,
            "n_parameters_near_bounds": len(hits)}


def verdict(maes: dict, baselines: dict, targets: list) -> dict:
    profiles = {}
    for name in targets:
        if name not in baselines or baselines[name] <= 0:
            raise ValueError("Robustness requires every eligible primaryV1 baseline")
        profiles[name] = {"V1_mae_mV": baselines[name], "V7m_mae_mV": maes[name],
                          "reduction": (baselines[name]-maes[name])/baselines[name],
                          "passes": bool(maes[name] <= .75*baselines[name])}
    return {"robust": all(x["passes"] for x in profiles.values()), "profiles": profiles}


def run_cell(cell: CellData, curves: OCVCurves, source: str, old_path: Path, primary: list,
             reused_failure: dict | None = None) -> dict:
    old = json.loads(old_path.read_text())
    out = OUTPUT/cell.name
    out.mkdir(parents=True, exist_ok=True)
    try:
        if reused_failure is not None:
            path = ROOT/reused_failure["A123_failure_path"]
            if sha256(path) != reused_failure["A123_failure_sha256"]:
                raise ValueError("Previously executed analytic A123 failure changed")
            attempts = json.loads(path.read_text())["attempts"]
            if any(a["success"] for a in attempts):
                raise ValueError("Cannot classify converged A123 identification as failure")
            raise IdentificationFailure(cell.name, attempts)
        fitted = fit(cell, curves, source)
    except IdentificationFailure as failure:
        LOGGER.error("INCONCLUSIVE_NLS %s: %s", cell.name, failure)
        baseline = {p: old["global_mae_mV"][p]["V1"] for p in primary}
        if cell.name == "calce":
            fix = json.loads((ROOT/"results/calce_z0_fix/summary.json").read_text())
            for p in primary:
                if p in fix["audits"]:
                    baseline[p] = fix["static"][p]["V1"]["after_metrics_mV"]["mae"]
        rows = [{"dataset": cell.name, "profile": p, "scope": "global", "status": "INCONCLUSIVE_NLS",
                 "n_evaluated_samples": 0, "V1_global_mae_mV": baseline[p],
                 **{k+"_mV": None for k in METRICS}} for p in primary]
        summary = {"complete": True, "status": "INCONCLUSIVE_NLS", "cell": cell.name,
                   "fit": {"success": False, "attempts": failure.attempts, "selected_start": None},
                   "loader": cell.metadata, "source": source, "primary_targets": primary,
                   "original_static_summary_sha256": sha256(old_path),
                   "principal_robustness": {"robust": None, "verdict": "INCONCLUSIVA",
                                            "reason": str(failure), "profiles": {}},
                   "metrics": rows, "failure_policy": json.loads(FAILURE_POLICY.read_text()),
                   "historical_decision_unchanged": True,
                   "target_voltage_not_evaluated_with_failed_parameters": True}
        save_json(out/"summary.json", summary)
        write_tables(out, rows)
        (out/"report.md").write_text(
            f"# {cell.name}: INCONCLUSIVE\n\n"
            "None of the three initializations converged within the registered limit of 500 evaluations.\n"
            "Unconverged parameters were not evaluated on targets or treated as valid models.\n"
            "Historical criterion and results preserved; attempts and logs archived.\n")
        return summary
    theta = np.asarray(fitted["theta"])
    rows, maes, baselines, diagnostics = [], {}, {}, {}
    for name, cycle in cell.cycles.items():
        voltage, z = simulate(theta, cycle, cell.cycles[source], cell.qnom_Ah)
        value = metrics(voltage-cycle.voltage)
        maes[name] = value["mae"]
        baselines[name] = old["global_mae_mV"][name]["V1"]
        scopes = masks(cycle, .02, 60)
        for scope, mask in scopes.items():
            if mask.any():
                measured = metrics(voltage[mask]-cycle.voltage[mask])
            else:
                LOGGER.warning("Empty monotone scope %s %s %s", cell.name, name, scope)
                measured = {k: None for k in METRICS}
            rows.append({"dataset": cell.name, "profile": name, "scope": scope, "variant": "original",
                         "n_samples": int(mask.sum()), **{k+"_mV": measured[k] for k in METRICS},
                         "V1_global_mae_mV": baselines[name]})
        support = fitted["source_SOC_support"]
        diagnostics[name] = {"ocv_boundary_samples": int(((z < support[0]) | (z > support[1])).sum()),
                             "predicted_voltage_min_max": [float(voltage.min()), float(voltage.max())]}
        np.savez_compressed(out/f"{name}_trajectories.npz", t=cycle.t, current_A=cycle.current,
                            measured_voltage_V=cycle.voltage, net_Ah=cycle.net_Ah, original_soc=cycle.soc,
                            V7m_soc=z, V7m=voltage)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(cycle.t/3600, cycle.voltage, "k", lw=.6, label="measured")
        axes[0].plot(cycle.t/3600, voltage, lw=.6, label="V7m")
        axes[0].set(ylabel="Voltage (V)", title=cell.name+" "+name)
        axes[0].legend()
        axes[1].plot(cycle.t/3600, (voltage-cycle.voltage)*1000, lw=.6)
        axes[1].set(xlabel="Time (h)", ylabel="Error (mV)")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out/f"{name}_error.{suffix}", dpi=150)
        plt.close(fig)
    original_decision = verdict(maes, baselines, primary)
    corrected_decision = None
    if cell.name == "calce":
        fix = json.loads((ROOT/"results/calce_z0_fix/summary.json").read_text())
        corrected_baselines = baselines.copy()
        for name, a in fix["audits"].items():
            if not a["eligible"]:
                raise ValueError("Cannot claim CALCE robustness with ineligible primary conditions")
            corrected_baselines[name] = fix["static"][name]["V1"]["after_metrics_mV"]["mae"]
            c = cell.cycles[name]
            c = replace(c, z0=a["z0"], soc=a["z0"]-c.net_Ah/cell.qnom_Ah)
            voltage, _ = simulate(theta, c, cell.cycles[source], cell.qnom_Ah)
            # V7m retains fitted source z0; only baseline SOC strata differ.
            for scope, mask in masks(c, .02, 60).items():
                if mask.any():
                    measured = metrics(voltage[mask]-c.voltage[mask])
                else:
                    LOGGER.warning("Empty corrected monotone scope %s %s %s", cell.name, name, scope)
                    measured = {k: None for k in METRICS}
                rows.append({"dataset": cell.name, "profile": name, "scope": scope, "variant": "corrected_z0",
                             "n_samples": int(mask.sum()), **{k+"_mV": measured[k] for k in METRICS},
                             "V1_global_mae_mV": corrected_baselines[name]})
        corrected_decision = verdict(maes, corrected_baselines, primary)
    knots = np.asarray(fitted["OCV_knots_soc"])
    volt = np.asarray(fitted["OCV_knots_V"])
    grid = np.linspace(min(0., knots[0]), max(1., knots[-1]), 1001)
    curve = PchipInterpolator(knots, volt)(np.clip(grid, knots[0], knots[-1]))
    if np.any(np.diff(curve) < -1e-12):
        raise FloatingPointError("Monotonicity invariant failed")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, curves.evaluate(grid)[2], "--", label="fixed original OCV")
    ax.plot(grid, curve, label="source-only V7m")
    ax.scatter(knots, volt, s=15, label="11 monotonic knots")
    ax.axvline(knots[0], color="gray", linestyle=":")
    ax.axvline(knots[-1], color="gray", linestyle=":")
    ax.set(xlabel="SOC", ylabel="OCV (V)", title=cell.name)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"monotonic_OCV.{suffix}", dpi=150)
    plt.close(fig)
    summary = {"complete": True, "status": "complete", "cell": cell.name, "fit": fitted, "loader": cell.metadata,
               "structural_identifiability": {
                   "Q_z0_identifiable": False,
                   "reason": "Candidate source support cancels Q/z0 exactly in normalized PCHIP coordinates; "
                             "Q/z0 are arbitrary within bounds and affect reported SOC labels, not voltage.",
                   "coordinate_identity": "(z-z_source_min)/(z_source_max-z_source_min)"
                                          "=(Ah_source_max-Ah_target)/(Ah_source_max-Ah_source_min)",
                   "numerical_fix": "Affine-equivalent normalized Ah coordinate and analytically exact "
                                    "PCHIP/RC Jacobian, including zero Q/z0 columns; "
                                    "no changed model/bounds/starts/tolerances."},
               "original_static_summary_sha256": sha256(old_path), "source": source, "primary_targets": primary,
               "global_mae_mV": maes, "original_robustness": original_decision,
               "corrected_robustness": corrected_decision, "metrics": rows, "diagnostics": diagnostics,
               "principal_robustness": corrected_decision if corrected_decision is not None else original_decision,
               "historical_decision_unchanged": True}
    save_json(out/"summary.json", summary)
    write_tables(out, rows)
    return summary


def main() -> None:
    verify_public(PROTOCOL_PATH)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(OUTPUT/"run.log", mode="w"),
                                                     logging.StreamHandler()])
    unified = json.loads((ROOT/"results/unified/summary.json").read_text())
    policy = json.loads(FAILURE_POLICY.read_text())
    verify_public(FAILURE_POLICY)
    results = {}
    for loader in (load_a123, load_mendeley):
        cell = loader()
        path = ROOT/"results/lfp_static"/cell.name/"summary.json"
        ocv = ROOT/"results/lfp_gonogo"/cell.name/"ocv_manifest.json"
        curves = OCVCurves.from_dict(json.loads(ocv.read_text())["points"])
        results[cell.name] = run_cell(cell, curves, "DST", path, ["FUDS", "US06"],
                                     reused_failure=policy if cell.name == "cell_A_a123" else None)
    for name in ("panasonic", "calce"):
        cell, curves = load_dataset(name, unified)
        cfg = json.loads((ROOT/"protocols"/"static_replication_protocol.json").read_text())["datasets"][name]
        path = ROOT/"results/static_replication"/name/"summary.json"
        results[name] = run_cell(cell, curves, cfg["source"], path, cfg["primary_targets"])
    summary = {"complete": True, "protocol": json.loads(PROTOCOL_PATH.read_text()),
               "failure_policy": policy, "failure_policy_sha256": sha256(FAILURE_POLICY),
               "protocol_sha256": sha256(PROTOCOL_PATH), "registration_commit": REGISTRATION,
               "manifest": {"python": platform.python_version(), "platform": platform.platform(),
                            "dependencies": {p: importlib.metadata.version(p) for p in
                                             ("numpy", "scipy", "openpyxl", "matplotlib")},
                            "shared_source_sha256": {str(ROOT/p): sha256(ROOT/p) for p in
                                                    ("data_lfp.py", "exp_lfp_static.py", "exp_lfp_gonogo.py",
                                                     "exp_static_replication.py", "lfp_ocv.py")},
                            "calce_z0_summary_sha256": sha256(ROOT/"results/calce_z0_fix/summary.json"),
                            "execution_commit": subprocess.check_output(
                                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()},
               "source_sha256": sha256(Path(__file__)), "datasets": results}
    save_json(OUTPUT/"summary.json", summary)
    lines = ["# V7m: robustness with monotone OCV", "",
             "Protocol 8a0e2e2. Source only; 11 PCHIP knots with positive increments; constant hold outside source SOC.",
             "Criterion: V7m>=25% reduction versus V1 on ALL targets at the same T, without selecting other models.", "",
             "| Dataset | Profile | V1 | V7m | Reduction | Passes |", "|---|---|---:|---:|---:|---|"]
    for name, s in results.items():
        if s["status"] != "complete":
            lines.append(f'| {name} | identification failed | -- | -- | -- | INCONCLUSIVE |')
            continue
        for profile, d in s["principal_robustness"]["profiles"].items():
            lines.append(f'| {name} | {profile} | {d["V1_mae_mV"]:.3f} | {d["V7m_mae_mV"]:.3f} | '
                         f'{d["reduction"]:.2%} | {d["passes"]} |')
    lines += [""]+[f'- {name}: robust={s["principal_robustness"]["robust"]}; '
                  f'status={s["status"]}.' for name, s in results.items()]
    lines += [f'- {name}: parameters near bounds={s["fit"]["n_parameters_near_bounds"]} '
              f'{s["fit"]["near_bound_parameters"]}.' for name, s in results.items() if s["status"] == "complete"]
    if results["calce"]["status"] == "complete":
        s = results["calce"]
        lines += ["", f'CALCE with original fallback baseline: robust={s["original_robustness"]["robust"]}; '
                  f'with main audited baseline: robust={s["corrected_robustness"]["robust"]}.']
    lines += ["", "## Caveats", "",
              "- Original verdicts were not rewritten; this is a new test of a single V7m variant.",
              "- A123/Mendeley retain previous sanitization and initial-state assumptions.",
              "- Panasonic V1 OCV checkpoint degree 5; dt1s differs from unified; intra-experiment comparison.",
              "- CALCE V7m source z0 [.9,1] retained despite source ~80 SOC; main baseline V1 corrected by Ah counting.",
              "- OCV support recalculated during fitting from source SOC Q/z0, never defined by the target.",
              "- Monotone increments constrain shape, not proof of physical identifiability.",
              "- Q/z0 are not identifiable in this protocol: recalculated source SOC support cancels both exactly.",
              "  Normalized Ah coordinate eliminates a purely rounding Jacobian without changing model or hyperparameters.",
              "  Analytic PCHIP/RC Jacobian validated by central differences; failed numerical attempts preserved.",
              "- Targets and secondary temperatures evaluated only if the source fit converged; no new criterion.",
              "  NLS failures in A123/Panasonic/CALCE are inconclusive, not measured gains or losses.", ""]
    (OUTPUT/"report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
