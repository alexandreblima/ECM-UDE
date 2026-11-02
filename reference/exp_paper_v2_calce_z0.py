"""Same-file CALCE SOC audit and read-only checkpoint reevaluation."""
from __future__ import annotations

from dataclasses import replace
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import importlib.metadata
import platform

import numpy as np
import openpyxl
import torch

import protocol as P
import exp_unified as U
from data_lfp import ROOT, save_json, sha256
from ecm import ECMParams
from ecm_2rc import ECMParams2RC
from exp_lfp_gonogo import metrics
from exp_lfp_static import MODELS, predict
from exp_static_replication import load_dataset, write_tables

OUTPUT = ROOT/"results"/"calce_z0_fix"
PROTOCOL_PATH = ROOT/"paper_v2_calce_z0_protocol.json"
REGISTRATION = "308dc2a"
LOGGER = logging.getLogger(__name__)


def audit_rows(a: np.ndarray, calibration: tuple | None, qnom: float = 2.) -> dict:
    """Columns time, step, raw charge-positive current, voltage, chargeAh, dischargeAh."""
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("CALCE audit requires six finite raw channels")
    a = a[np.argsort(a[:, 0], kind="stable")]
    dynamics = np.flatnonzero(a[:, 1] == 7)
    if not len(dynamics):
        raise ValueError("Missing dynamic step7")
    first = int(dynamics[0])
    before = a[:first]
    # CCCV taper endpoint before dynamic profile, followed by a zero-current rest.
    anchors = np.flatnonzero((before[:, 1] == 3) & (before[:, 2] > 0) &
                            (before[:, 2] <= .02) & (before[:, 3] >= 4.19))
    for k in anchors[::-1]:
        cc = np.flatnonzero((before[:k, 1] == 2) & (before[:k, 2] > .02) &
                            (before[:k, 3] >= 4.19))
        if not len(cc) or not np.all(before[int(cc[-1])+1:k+1, 1] == 3):
            LOGGER.warning("Taper without preceding recorded CC-to-CV full charge; anchor rejected")
            continue
        if k+1 >= first or before[k+1, 1] != 4 or abs(before[k+1, 2]) > .02:
            continue
        segment = a[k:first+1]
        delta_counters = np.diff(segment[:, 4:6], axis=0)
        if np.any(delta_counters < -1e-5):
            LOGGER.warning("Counter reset after full charge; anchor rejected")
            continue
        same_step = segment[1:, 1] == segment[:-1, 1]
        contradictory = ((segment[1:, 2] > .02) & (delta_counters[:, 1] > 1e-5) |
                         (segment[1:, 2] < -.02) & (delta_counters[:, 0] > 1e-5))
        if np.any(same_step & contradictory):
            LOGGER.warning("Current sign contradicts cumulative charge/discharge counters; anchor rejected")
            continue
        delta = float((a[first, 5]-a[first, 4])-(a[k, 5]-a[k, 4]))
        z0 = 1-delta/qnom
        if not 0 <= z0 <= 1:
            LOGGER.warning("Charge-count anchor gives invalidSOC=%s", z0)
            continue
        return {"eligible": True, "method": "same_file_charge_count", "z0": z0,
                "anchor_row": a[k].tolist(), "profile_first_row": a[first].tolist(),
                "net_discharged_Ah_since_full": delta, "Qnom_Ah": qnom,
                "counter_monotonicity_verified": True, "current_sign_agrees_with_counters": True,
                "CC_to_CV_verified": True, "rest_after_CV_verified": True}
    start = first
    while start and a[start-1, 1] == 6 and abs(a[start-1, 2]) <= .02:
        start -= 1
    rest = a[start:first]
    cadence = np.diff(np.unique(rest[:, 0]))
    contiguous = len(cadence) >= 2 and cadence.max() <= 2*cadence.min()+1e-8
    if contiguous and calibration is not None:
        end = rest[-1, 0]
        window = rest[rest[:, 0] >= end-60]
        if rest[0, 0] <= end-60 and len(window) > 1:
            v = float(np.mean(window[:, 3]))
            voltages, socs = calibration
            if voltages[0] <= v <= voltages[-1]:
                return {"eligible": True, "method": "independent_OCV_last60s_rest_mean",
                        "z0": float(np.interp(v, voltages, socs)), "mean_voltage_V": v,
                        "window_start_end_s": [float(end-60), float(end)], "rest_samples": len(window),
                        "immediately_preceding_rest": True, "contiguous_rest_verified": True,
                        "rest_positive_intervals_min_max_s": [float(cadence.min()), float(cadence.max())]}
    LOGGER.warning("No valid same-file SOC anchor or60s rest; condition excluded")
    return {"eligible": False, "z0": None, "method": "excluded_no_valid_anchor"}


def audit_condition(condition: str, reference: dict, calibration: tuple | None) -> dict:
    metadata = reference["datasets"]["calce"]["conditions"][condition]
    path = (ROOT/metadata["path"]).resolve()
    if sha256(path) != metadata["sha256"]:
        raise ValueError("CALCE input changed")
    wb = openpyxl.load_workbook(io.BytesIO(path.read_bytes()), read_only=True, data_only=True)
    try:
        sheet = wb[wb.sheetnames[1] if len(wb.sheetnames) > 1 else wb.sheetnames[0]]
        rows = [[r[k] for k in (1, 4, 6, 7, 8, 9)] for r in sheet.values
                if len(r) >= 10 and isinstance(r[4], (int, float))]
    finally:
        wb.close()
    return {**audit_rows(np.asarray(rows, dtype=float), calibration),
            "raw_path": str(path), "sha256": sha256(path), "old_z0": metadata["z0"]}


def hypothetical_static(maes: dict, exclusions: dict | None = None) -> dict:
    targets = ("DST_25C", "US06_25C", "BJDST_25C")
    profiles = {}
    for c in targets:
        v = maes[c]
        candidates = [m for m in MODELS[1:] if m not in (exclusions or {}).get(c, [])]
        best = min(candidates, key=lambda m: v[m])
        profiles[c] = {"best_model": best, "reduction": (v["V1"]-v[best])/v["V1"],
                       "passes": v[best] <= .75*v["V1"],
                       "excluded_candidates": (exclusions or {}).get(c, [])}
    return {"verdict": "POSITIVA" if all(x["passes"] for x in profiles.values()) else "NEGATIVA",
            "profiles": profiles, "historical_verdict_not_modified": True}


def main() -> None:
    if subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REGISTRATION}:{PROTOCOL_PATH.name}"]) != PROTOCOL_PATH.read_bytes():
        raise ValueError("CALCE z0 protocol changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(OUTPUT/"run.log", mode="w"),
                                                     logging.StreamHandler()])
    torch.set_num_threads(1)
    original = json.loads((ROOT/"results/unified/summary.json").read_text())
    ten = json.loads((ROOT/"results/unified_10seeds/summary.json").read_text())
    if not ten.get("complete"):
        raise ValueError("Ten-seed stage must complete first")
    calibration, calmeta = U.ocv_calibration()
    if calmeta != original["datasets"]["calce"]["ocv_calibration"]:
        raise ValueError("Independent calibration differs from original unified protocol")
    audits = {c: audit_condition(c, original, calibration) for c in ("US06_25C", "BJDST_25C")}
    save_json(OUTPUT/"z0_audit.json", audits)
    data = ten["datasets"]["calce"]
    cfg = P.DATASETS["calce"]
    source_stats = data["conditions"]["FUDS_25C"]["split"]["stats"]
    params1 = ECMParams.from_dict(data["runs"]["ecm_1rc_nls"]["deterministic"]["identification"]["params"])
    params2 = ECMParams2RC.from_dict(data["runs"]["ecm_2rc_nls"]["deterministic"]["identification"]["params"])
    rows, unified = [], {}
    for condition, audit in audits.items():
        if not audit["eligible"]:
            continue
        raw, _ = U.load_calce(condition, cfg, calibration)
        net = raw["Ah"].numpy().astype(float)
        raw["SoC"] = torch.tensor(P.causal_soc(net, qnom_Ah=2., z0=audit["z0"]), dtype=torch.float32)
        ctx = U.dataset_context(raw, cfg, stats=source_stats, source=False)
        unified[condition] = {}
        for model in P.MODELS:
            selected = data["runs"][model]
            seeds = ["deterministic"] if model.endswith("_nls") else [str(s) for s in range(42, 52)]
            unified[condition][model] = {}
            for seed in seeds:
                saved = selected[seed]
                path = (ROOT/saved["checkpoint"]).resolve()
                if sha256(path) != saved["sha256"]:
                    raise ValueError("Checkpoint changed")
                if model.endswith("_nls"):
                    obj = params1 if model == "ecm_1rc_nls" else params2
                else:
                    # No training: source normalization and exact original parameter tensors.
                    U.set_seed(int(seed))
                    obj = U.build_model(model, source_stats, params1, cfg)
                    obj.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                    obj.eval()
                after = U.evaluate(obj, model, ctx, cfg)
                before = saved["metrics"][condition]
                unified[condition][model][seed] = {"before": before, "after": after,
                                                  "checkpoint": str(path), "sha256": saved["sha256"]}
                for mode, values in after.items():
                    rows.append({"condition": condition, "family": "unified", "model": model, "seed": seed,
                                 "mode": mode, "before_mae_mV": before[mode]["mae"],
                                 "after_mae_mV": values["mae"], "delta_mae_mV": values["mae"]-before[mode]["mae"]})
    cell, curves = load_dataset("calce", original)
    old_static_path = ROOT/"results/static_replication/calce/summary.json"
    old_static = json.loads(old_static_path.read_text())
    corrected_maes = json.loads(json.dumps(old_static["global_mae_mV"]))
    static = {}
    for condition, audit in audits.items():
        if not audit["eligible"]:
            corrected_maes.pop(condition, None)
            continue
        cycle = cell.cycles[condition]
        corrected = replace(cycle, z0=audit["z0"], soc=audit["z0"]-cycle.net_Ah/cell.qnom_Ah)
        static[condition] = {}
        traces = {}
        for model, fit in old_static["fits"].items():
            voltage, z = predict(corrected, cell, curves, np.asarray(fit["theta"]), model)
            value = metrics(voltage-corrected.voltage)
            before = old_static["global_mae_mV"][condition][model]
            corrected_maes[condition][model] = value["mae"]
            static[condition][model] = {"before_mae_mV": before, "after_metrics_mV": value,
                                        "z0_used": float(z[0]), "excluded_nonphysical_V5": bool(
                                            model == "V5" and (voltage.max() > 5 or voltage.min() < 1))}
            rows.append({"condition": condition, "family": "static", "model": model, "seed": "deterministic",
                         "mode": "continuous", "before_mae_mV": before, "after_mae_mV": value["mae"],
                         "delta_mae_mV": value["mae"]-before})
            traces[model], traces[model+"_soc"] = voltage, z
        np.savez_compressed(OUTPUT/f"{condition}_static_trajectories.npz", t=corrected.t,
                            current_A=corrected.current, net_Ah=corrected.net_Ah, original_soc=corrected.soc,
                            measured_voltage_V=corrected.voltage, **traces)
    exclusions = {c: ["V5"] for c, results in static.items() if results["V5"]["excluded_nonphysical_V5"]}
    revised = hypothetical_static(corrected_maes, exclusions) if all(a["eligible"] for a in audits.values()) else {
        "verdict": "EXCLUDED_CONDITIONS", "historical_verdict_not_modified": True}
    summary = {"complete": True, "protocol": json.loads(PROTOCOL_PATH.read_text()), "protocol_sha256": sha256(PROTOCOL_PATH),
               "preregistration_commit": REGISTRATION, "source_sha256": sha256(Path(__file__)),
               "manifest": {"dependencies": {k: importlib.metadata.version(k) for k in
                                             ("numpy", "scipy", "torch", "openpyxl")},
                            "python": platform.python_version(), "platform": platform.platform(),
                            "shared_source_sha256": {str(ROOT/p): sha256(ROOT/p) for p in
                                                    ("exp_unified.py", "protocol.py", "data.py", "data_calce.py",
                                                     "exp_lfp_static.py", "exp_static_replication.py")},
                            "execution_commit": subprocess.check_output(
                                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()},
               "audit_thresholds": {"CV_nominal_V": 4.2, "CV_min_recorded_V": 4.19,
                                    "charge_taper_max_A": .02, "rest_max_abs_current_A": .02,
                                    "counter_decrease_tolerance_Ah": 1e-5,
                                    "rest_min_positive_intervals": 2, "max_interval_ratio": 2,
                                    "OCV_rest_window_s": 60},
               "unified10_summary_sha256": sha256(ROOT/"results/unified_10seeds/summary.json"),
               "static_summary_sha256": sha256(old_static_path), "calibration": calmeta, "audits": audits,
               "unified": unified, "static": static, "hypothetical_static_decision": revised,
               "historical_static_decision": old_static["decision"], "table": rows}
    save_json(OUTPUT/"summary.json", summary)
    write_tables(OUTPUT, rows)
    lines = ["# CALCE: auditoria z0 e avaliacao sem retreino", "",
             "Regra congelada308dc2a. Contagem de carga tem prioridade sobre inversaoOCV. Nenhum alvo escolhe parametros.", ""]
    for c, a in audits.items():
        lines.append(f'- {c}: elegivel={a["eligible"]}, metodo={a["method"]}, z0 antigo={a["old_z0"]}, novo={a["z0"]}.')
    lines += ["", "| Condicao | Familia | Modelo | Modo | MAE antes | MAE depois |", "|---|---|---|---|---:|---:|"]
    groups = {}
    for row in rows:
        key = (row["condition"], row["family"], row["model"], row["mode"])
        groups.setdefault(key, []).append(row)
    for key, values in groups.items():
        before = np.mean([v["before_mae_mV"] for v in values])
        after = np.mean([v["after_mae_mV"] for v in values])
        lines.append("| "+" | ".join(key)+f" | {before:.3f} | {after:.3f} |")
    lines += ["", f'Veredito estatico historico: {old_static["decision"]["verdict"]}; '
              f'hipotetico com z0 corrigido: **{revised["verdict"]}**. Historico nao alterado.', "",
              "- Unificado: pesos/split/normalizacao fonte preservados; dez seeds, nenhuma atualizacao.",
              "- V1/V5 recebem z0 porperfil; V6/V7 conservam Q/z0 ajustados na fonte, portanto nao sao corrigidos retroativamente.",
              "- V5 nao fisico continua excluido dos resultados principais, mesmo se a novaSOC reduzir extrapolacao.",
              "- Mudanca do veredito e contrafactual sob novas condicoes auditadas, nao reescrita do criterio antigo.",
              "- Tabelas anteriores e dados brutos intactos; hashes dos checkpoints/ancoras no manifesto.", ""]
    (OUTPUT/"report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
