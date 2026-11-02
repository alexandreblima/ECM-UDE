"""Preregistered source-only NLS parameter ablations; saved baselines stay read-only."""
from __future__ import annotations

from dataclasses import replace
import json
import argparse
import importlib.metadata
import platform
import logging
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import least_squares

from data_lfp import ROOT, load_a123, load_mendeley, save_json, sha256
from exp_lfp_static import bounds_starts, predict, unpack
from exp_lfp_gonogo import masks, metrics
from exp_static_replication import load_dataset, write_tables
from exp_a123_temperature import load_temperature
from exp_paper_v2_calce_z0 import audit_condition
from lfp_ocv import OCVCurves

OUTPUT = ROOT/"results/param_ablation"
PROTOCOL = ROOT/"reviewer_round_protocol.json"
REGISTRATION = "d14935e"
LOGGER = logging.getLogger(__name__)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify() -> None:
    committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REGISTRATION}:{PROTOCOL.name}"])
    if committed != PROTOCOL.read_bytes():
        raise ValueError("Reviewer protocol differs from registration")


def fit_variant(cell, curves, source_name: str, variant: str, baseline: dict) -> dict:
    cycle = cell.cycles[source_name]
    lo, hi, starts = bounds_starts(cell, cycle, curves, "V6")
    if variant == "V6-wide":
        lo[6] = .7
    active = np.array([5, 6] if variant == "V6-Q" else list(range(5)) if variant == "V6-RC"
                      else list(range(7)))
    template = np.r_[baseline["theta"][:5], 1., cycle.z0]
    if variant == "V6-Q":
        template[:5] = baseline["theta"][:5]
    attempts, candidates = [], []
    for index, initial in enumerate(starts):
        def expand(free):
            theta = template.copy()
            theta[active] = free
            return theta
        LOGGER.info("FIT %s %s source=%s start=%d", cell.name, variant, source_name, index)
        result = least_squares(
            lambda free: predict(cycle, cell, curves, expand(free), "V6")[0]-cycle.voltage,
            initial[active], bounds=(lo[active], hi[active]), method="trf", loss="linear", x_scale="jac",
            max_nfev=500, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        theta = expand(result.x)
        attempt = {"start": index, "success": bool(result.success), "nfev": int(result.nfev),
                   "message": str(result.message), "SSE_V2": float(2*result.cost), "theta": theta.tolist()}
        attempts.append(attempt)
        LOGGER.info("RESULT %s", json.dumps(attempt))
        if result.success:
            candidates.append((result.cost, index, theta, result.x))
        else:
            LOGGER.error("NLS did not converge %s %s start%d", cell.name, variant, index)
    if not candidates:
        return {"success": False, "status": "INCONCLUSIVE_NLS", "attempts": attempts}
    _, index, theta, free = min(candidates, key=lambda a: a[0])
    near = np.minimum((free-lo[active])/(hi[active]-lo[active]),
                      (hi[active]-free)/(hi[active]-lo[active]))
    return {"success": True, "theta": theta.tolist(), "selected_start": index, "attempts": attempts,
            "free_coordinate_indices": active.tolist(), "parameters": unpack(theta, "V6", cell, cycle),
            "near_bound_coordinate_indices": active[near < 1e-4].tolist(),
            "source": source_name, "variant": variant}


def explained(v1: float, variant: float, joint: float) -> float | None:
    denominator = v1-joint
    if denominator == 0:
        LOGGER.warning("Zero joint gain: explained fraction unavailable")
        return None
    return (v1-variant)/denominator


def interpret(rows: list[dict]) -> dict:
    eligible = [r for r in rows if r["V1_mae_mV"] > r["V6_mae_mV"]]
    counts = {m: sum(r[m+"_fraction"] is not None and r[m+"_fraction"] >= .7 for r in eligible)
              for m in ("V6-Q", "V6-RC")}
    passes = {m: count > len(eligible)/2 for m, count in counts.items()}
    verdict = ("INCONCLUSIVO" if not eligible else
               "escala de SOC" if passes["V6-Q"] and not passes["V6-RC"] else
               "parametros RC" if passes["V6-RC"] and not passes["V6-Q"] else "conjunto")
    return {"verdict": verdict, "eligible_positive_joint_gain": len(eligible), "total_conditions": len(rows),
            "counts_at_least_70pct": counts, "strict_majority": passes}


def evaluate_block(label, cell, curves, source, targets, old, joint, variants, ocvt=None,
                   reuse_fits=False) -> dict:
    out = OUTPUT/label
    out.mkdir(parents=True, exist_ok=True)
    fits = {"V1": {**old, "success": True, "reused": True},
            "V6": {**joint, "success": True, "reused": True}}
    for variant in variants:
        fits[variant] = (read(out/"summary.json")["fits"][variant] if reuse_fits else
                        fit_variant(cell, ocvt if variant == "V6-OCVT" else curves, source, variant, old))
    rows, stratified = [], []
    for target in targets:
        cycle = cell.cycles[target]
        maes, traces = {}, {}
        for variant, fit in fits.items():
            if not fit["success"]:
                maes[variant] = None
                continue
            selected_curves = ocvt if variant == "V6-OCVT" else curves
            model = "V1" if variant in ("V1", "V6-RC") else "V6"
            theta = np.array(fit["theta"][:5] if model == "V1" else fit["theta"])
            voltage, soc = predict(cycle, cell, selected_curves, theta, model)
            maes[variant] = metrics(voltage-cycle.voltage)["mae"]
            traces[variant], traces[variant+"_soc"] = voltage, soc
            for scope, mask in masks(cycle, .02, 60).items():
                if mask.any():
                    value = metrics(voltage[mask]-cycle.voltage[mask])
                else:
                    LOGGER.warning("Empty scope %s %s %s", label, target, scope)
                    value = {m: None for m in ("mae", "rmse", "max_error", "p99")}
                stratified.append({"block": label, "target": target, "variant": variant, "scope": scope,
                                   "n_samples": int(mask.sum()), **{m+"_mV": v for m, v in value.items()}})
        np.savez_compressed(out/f"{target}_trajectories.npz", t=cycle.t, current_A=cycle.current,
                            measured_voltage_V=cycle.voltage, net_Ah=cycle.net_Ah, **traces)
        row = {"block": label, "cell": cell.name, "target": target,
               **{m+"_mae_mV": maes.get(m) for m in fits}}
        for variant in ("V6-Q", "V6-RC"):
            row[variant+"_fraction"] = (explained(maes["V1"], maes[variant], maes["V6"])
                                        if maes.get(variant) is not None else None)
        rows.append(row)
    interpretation = interpret(rows)
    if any(not fits[m]["success"] for m in ("V6-Q", "V6-RC") if m in fits):
        interpretation["verdict"] = "INCONCLUSIVO_AJUSTE"
    wide = None
    if "V6-wide" in fits:
        wide = {"profiles": {r["target"]: {"passes": None if r["V6-wide_mae_mV"] is None else
                    r["V6-wide_mae_mV"] <= .75*r["V1_mae_mV"],
                    "reduction": None if r["V6-wide_mae_mV"] is None else
                    1-r["V6-wide_mae_mV"]/r["V1_mae_mV"]} for r in rows}}
        wide["verdict"] = ("INCONCLUSIVO" if not fits["V6-wide"]["success"] else
                           "POSITIVO" if all(p["passes"] for p in wide["profiles"].values()) else "NEGATIVO")
    result = {"complete": True, "all_requested_fits_converged": all(f["success"] for f in fits.values()),
              "label": label, "loader": cell.metadata, "source": source, "targets": targets,
              "fits": fits, "rows": rows, "metrics": stratified, "interpretation": interpretation,
              "wide_decision": wide, "fixed_OCV_points": curves.to_dict(),
              "local_OCV_points": None if ocvt is None else ocvt.to_dict(),
              "OCVT_Q_policy": "OCV branches normalized by measured local low-current capacity as existing loader; "
                               "dynamic Qreference remains Q25 +/-10%; only V6-OCVT substitutes local curve",
              "source_NLS_interval": {"first_s": float(cell.cycles[source].t[0]),
                                      "last_s": float(cell.cycles[source].t[-1]),
                                      "n_samples": len(cell.cycles[source].t), "full_profile": True}}
    save_json(out/"summary.json", result)
    write_tables(out, rows)
    return result


def main() -> None:
    verify()
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--reuse-fits", action="store_true", help="Rescore saved fits without any NLS")
    args = parser.parse_args()
    if args.report_only:
        summary = read(OUTPUT/"summary.json")
        export(summary)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(OUTPUT/"run.log"),
                                                     logging.StreamHandler()])
    reference = read(ROOT/"results/unified/summary.json")
    results, calce_audit = {}, {}
    source_info = {}
    for loader in (load_a123, load_mendeley):
        cell = loader()
        previous = ROOT/"results/lfp_static"/cell.name/"summary.json"
        saved = read(previous)
        curves = OCVCurves.from_dict(read(ROOT/"results/lfp_gonogo"/cell.name/"ocv_manifest.json")["points"])
        results[cell.name] = evaluate_block(cell.name, cell, curves, "DST", ["FUDS", "US06"],
                                            saved["fits"]["V1"], saved["fits"]["V6"], ["V6-Q", "V6-RC"],
                                            reuse_fits=args.reuse_fits)
        source_info[cell.name] = (cell, curves, saved)
    for name in ("panasonic", "calce"):
        cell, curves = load_dataset(name, reference)
        cfg = read(ROOT/"static_replication_protocol.json")["datasets"][name]
        saved = read(ROOT/"results/static_replication"/name/"summary.json")
        if name == "calce":
            # Independent curve inversion is unnecessary when charge-count anchors succeed.
            inverse = (curves.discharge_V, curves.discharge_z)
            for condition in [cfg["source"], *cfg["primary_targets"]]:
                a = audit_condition(condition, reference, inverse)
                if not a["eligible"]:
                    raise ValueError(f"CALCE condition ineligible: {condition}")
                calce_audit[condition] = a
                cycle = cell.cycles[condition]
                cell.cycles[condition] = replace(cycle, z0=a["z0"],
                                                 soc=a["z0"]-cycle.net_Ah/cell.qnom_Ah)
        variants = ["V6-Q", "V6-RC"]+(["V6-wide"] if name == "calce" else [])
        results[name] = evaluate_block(name, cell, curves, cfg["source"], cfg["primary_targets"],
                                       saved["fits"]["V1"], saved["fits"]["V6"], variants,
                                       reuse_fits=args.reuse_fits)
    cell25, curves25, saved25 = source_info["cell_A_a123"]
    thermal = read(ROOT/"results/lfp_static_temperature/cell_A_a123/summary.json")
    for temperature in (0, 10, 40, 50):
        cell, own_curves = load_temperature(temperature, thermal["manifest"]["archives"], cell25.qnom_Ah)
        for block, joint, variants in (
            ("transfer", saved25["fits"]["V6"], []),
            ("local", thermal["temperatures"][str(temperature)]["local_V6_fit"],
             ["V6-Q", "V6-RC", "V6-OCVT"])):
            label = f"a123_{block}_{temperature}C"
            if block == "transfer":
                # Reuse the newly identified DST25 ablations without fitting target temperatures.
                base = results["cell_A_a123"]
                proxy = evaluate_block(label, cell, curves25, "DST", ["FUDS", "US06"],
                                       saved25["fits"]["V1"], joint, [])
                for variant in ("V6-Q", "V6-RC"):
                    fit = base["fits"][variant]
                    proxy["fits"][variant] = fit
                    if not fit["success"]:
                        for row in proxy["rows"]:
                            row[variant+"_mae_mV"] = None
                            row[variant+"_fraction"] = None
                        continue
                    for row in proxy["rows"]:
                        cycle = cell.cycles[row["target"]]
                        model = "V1" if variant == "V6-RC" else "V6"
                        theta = np.array(fit["theta"][:5] if model == "V1" else fit["theta"])
                        voltage, z = predict(cycle, cell, curves25, theta, model)
                        value = metrics(voltage-cycle.voltage)
                        row[variant+"_mae_mV"] = value["mae"]
                        row[variant+"_fraction"] = explained(row["V1_mae_mV"], value["mae"], row["V6_mae_mV"])
                        path = OUTPUT/label/f'{row["target"]}_trajectories.npz'
                        with np.load(path) as old:
                            trace = {k: old[k] for k in old.files}
                        np.savez_compressed(path, **trace, **{variant: voltage, variant+"_soc": z})
                        for scope, mask in masks(cycle, .02, 60).items():
                            if not mask.any():
                                LOGGER.warning("Empty scope %s %s %s", label, row["target"], scope)
                            m = metrics(voltage[mask]-cycle.voltage[mask]) if mask.any() else {
                                k: None for k in ("mae", "rmse", "max_error", "p99")}
                            proxy["metrics"].append({"block": label, "target": row["target"], "variant": variant,
                                                     "scope": scope, "n_samples": int(mask.sum()),
                                                     **{k+"_mV": v for k, v in m.items()}})
                proxy["interpretation"] = interpret(proxy["rows"])
                results[label] = proxy
                save_json(OUTPUT/label/"summary.json", proxy)
                write_tables(OUTPUT/label, proxy["rows"])
            else:
                results[label] = evaluate_block(label, cell, curves25, "DST", ["FUDS", "US06"],
                                                saved25["fits"]["V1"], joint, variants, own_curves,
                                                reuse_fits=args.reuse_fits)
    groups = {"matched_temperature": [r for name in ("cell_A_a123", "cell_B_mendeley", "panasonic", "calce")
                                      for r in results[name]["rows"]],
              "thermal_transfer": [r for k, b in results.items() if k.startswith("a123_transfer") for r in b["rows"]],
              "thermal_local": [r for k, b in results.items() if k.startswith("a123_local") for r in b["rows"]]}
    summary = {"complete": True, "protocol": read(PROTOCOL), "registration_commit": REGISTRATION,
               "all_requested_fits_converged": all(b["all_requested_fits_converged"] for b in results.values()),
               "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(Path(__file__)),
               "manifest": {"python": platform.python_version(),
                            "dependencies": {m: importlib.metadata.version(m) for m in
                                             ("numpy", "scipy", "openpyxl")},
                            "shared_source_sha256": {p: sha256(ROOT/p) for p in
                                                    ("data_lfp.py", "lfp_ocv.py", "exp_lfp_static.py",
                                                     "exp_lfp_gonogo.py", "exp_static_replication.py",
                                                     "exp_a123_temperature.py", "exp_paper_v2_calce_z0.py")},
                            "saved_baseline_sha256": {str(p): sha256(p) for p in (
                                ROOT/"results/lfp_static/cell_A_a123/summary.json",
                                ROOT/"results/lfp_static/cell_B_mendeley/summary.json",
                                ROOT/"results/static_replication/panasonic/summary.json",
                                ROOT/"results/static_replication/calce/summary.json",
                                ROOT/"results/lfp_static_temperature/cell_A_a123/summary.json")},
                            "initial_execution_commit": REGISTRATION, "no_neural_training": True},
               "blocks": results, "calce_audit": calce_audit,
               "interpretation": {k: interpret(v) for k, v in groups.items()}}
    save_json(OUTPUT/"summary.json", summary)
    export(summary)


def export(summary: dict) -> None:
    rows = [r for block in summary["blocks"].values() for r in block["rows"]]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_tables(OUTPUT, [{k: row.get(k) for k in fields} for row in rows])


if __name__ == "__main__":
    main()
