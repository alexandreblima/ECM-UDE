"""Frozen A123 thermal transfer and excluded local-DST V6 diagnostic."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from data_lfp import ROOT, A_DIR, CellData, _arbin, _cycle, integrate_Ah, save_json, sha256
from exp_lfp_gonogo import METRICS, masks, metrics
from exp_lfp_static import MODELS, bounds_starts, parameter_names, predict, unpack
from exp_static_replication import verify_registration, write_tables
from lfp_ocv import OCVCurves, build_curves

OUTPUT = ROOT/"results"/"lfp_static_temperature"/"cell_A_a123"
SOURCE = ROOT/"results"/"lfp_static"/"cell_A_a123"
OCV_SOURCE = ROOT/"results"/"lfp_gonogo"/"cell_A_a123"/"ocv_manifest.json"
PROTOCOL_PATH = ROOT/"static_replication_protocol.json"
DIAGNOSTIC_PATH = ROOT/"a123_temperature_diagnostic.json"
DIAGNOSTIC_COMMIT = "3d0d047"
LOGGER = logging.getLogger(__name__)
SCHEDULE = {"DST": (8, 9, 4, 5, 7, 361),
            "US06": (16, 17, 12, 13, 15, 601),
            "FUDS": (24, 25, 20, 21, 23, 1373)}


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def snapshot() -> dict[str, str]:
    folders = ("lfp_gonogo", "lfp_static", "unified", "static_replication", "lfp_static_temperature")
    paths = [p for name in folders for p in (ROOT/"results"/name).rglob("*")
             if p.is_file() and not p.is_relative_to(OUTPUT)]
    paths += [ROOT/"report_static_replication.md", ROOT/"report_static.md", ROOT/"report_gonogo.md"]
    return {str(p): sha256(p) for p in sorted(paths)}


def verify_protocols() -> None:
    verify_registration()
    committed = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{DIAGNOSTIC_COMMIT}:{DIAGNOSTIC_PATH.name}"])
    if committed != DIAGNOSTIC_PATH.read_bytes():
        raise ValueError("Local diagnostic differs from committed registration")
    if subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"],
                               text=True).strip() != "lfp-histerese":
        raise ValueError("Thermal experiment requires lfp-histerese branch")


def extract_archives() -> dict:
    manifest = {}
    names = protocol()["temperature_A123"]["missing_archives"]
    for name in names:
        if not (A_DIR/name).is_file():
            raise FileNotFoundError(f"Required thermal archive missing: {A_DIR/name}")
    target = (A_DIR/"extracted").resolve()
    for name in names:
        path = A_DIR/name
        members = {}
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                dest = (target/info.filename).resolve()
                if not dest.is_relative_to(target) or info.filename.startswith("/"):
                    raise ValueError(f"Unsafe ZIP member: {info.filename}")
                if info.is_dir():
                    continue
                data = archive.read(info)
                digest = hashlib.sha256(data).hexdigest()
                if dest.exists():
                    if sha256(dest) != digest:
                        raise ValueError(f"Refusing overwrite of conflicting extracted file: {dest}")
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                members[str(dest)] = digest
        manifest[str(path)] = {"sha256": sha256(path), "members": members}
    return manifest


def member_path(archives: dict, temperature: int, dynamic: bool) -> Path:
    paths = [Path(p) for name, archive in archives.items()
             if (Path(name).name == f"A123_DST-US06-FUDS-{temperature}.zip" if dynamic
                 else Path(name).name.startswith(f"A123_OCV{temperature}-"))
             for p in archive["members"] if Path(p).name.startswith("A1-007")
             and Path(p).suffix == ".xlsx"]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one A1-007 workbook at {temperature}C, dynamic={dynamic}")
    return paths[0]


def load_temperature(temperature: int, archives: dict, q25: float) -> tuple[CellData, OCVCurves]:
    dynamic_path = member_path(archives, temperature, True)
    ocv_path = member_path(archives, temperature, False)
    ocv = _arbin(ocv_path)
    dis, chg = ocv[ocv[:, 1] == 5], ocv[ocv[:, 1] == 7]
    if not (len(dis) > 2 and len(chg) > 2 and np.all(dis[:, 3] < 0)
            and np.all(chg[:, 3] > 0) and dis[-1, 4] < 2.1 and chg[-1, 4] > 3.5):
        raise ValueError(f"{temperature}C low-current branches not verified")
    dis_ah, chg_ah = integrate_Ah(dis[:, 0], -dis[:, 3]), integrate_Ah(chg[:, 0], chg[:, 3])
    qlocal = float(dis_ah[-1])
    dynamic = _arbin(dynamic_path)
    cycles = {}
    for name, (step, repeat, cc, cv, rest, lag) in SCHEDULE.items():
        selected = dynamic[np.isin(dynamic[:, 1], (step, repeat))]
        preparation = dynamic[np.isin(dynamic[:, 1], (cc, cv, rest))]
        cv_rows = dynamic[dynamic[:, 1] == cv]
        rest_rows = dynamic[dynamic[:, 1] == rest]
        if not (len(selected) > lag+2 and len(cv_rows) > 0 and len(rest_rows) > 1
                and np.all(cv_rows[:, 3] > 0) and cv_rows[-1, 3] < .06 and cv_rows[-1, 4] > 3.59):
            raise ValueError(f"{temperature}C {name} full-charge preparation not verified")
        rest_sample_span = float(rest_rows[-1, 0]-rest_rows[0, 0])
        rest_duration = rest_sample_span + float(np.median(np.diff(rest_rows[:, 0])))
        if rest_duration < 119 or np.max(np.abs(rest_rows[:, 3])) > .02:
            raise ValueError(f"{temperature}C {name} pre-profile rest differs from protocol")
        metadata = {"path": str(dynamic_path), "sha256": sha256(dynamic_path),
                    "z0": 1., "h0": 1., "s0": 1., "temperature_C_nominal": temperature,
                    "temperature": f"{temperature}C nominal; measured auxiliary retained",
                    "initial_state_provenance": "CCCV3.6V, taper<0.06A, rest>=119s; RC0 assumed",
                    "preceding_steps": [cc, cv, rest], "preceding_rest_s": rest_duration,
                    "preceding_rest_sample_span_s": rest_sample_span,
                    "rest_duration_method": "Sample span plus median interval; first 5s rest interval precedes first record",
                    "dynamic_steps_including_repetition_rest": [step, repeat],
                    "preparation_time_start_end_s": [float(preparation[0, 0]), float(preparation[-1, 0])],
                    "raw_time_start_end_s": [float(selected[0, 0]), float(selected[-1, 0])],
                    "post_profile_rest_excluded": True}
        raw = np.column_stack((selected[:, 0], -selected[:, 3], selected[:, 4], selected[:, 7]))
        cycle = _cycle(name, raw, q25, metadata)
        correlation = float(np.corrcoef(cycle.current[:-lag], cycle.current[lag:])[0, 1])
        ends = np.flatnonzero(selected[:, 1] == repeat)
        starts = np.r_[0, ends[:-1]+1]
        periods = np.diff(selected[starts, 0])
        if len(periods) < 2 or np.max(np.abs(periods-lag)) > .1 or not np.isfinite(correlation):
            raise ValueError(f"{temperature}C {name} raw profile schedule not verified")
        if correlation < .95:
            LOGGER.warning("%sC %s fixed-grid lag correlation %.4f <0.95; raw repetition timing verified",
                           temperature, name, correlation)
        cycle.metadata.update(repetition_seconds=lag, repetition_correlation=correlation,
                              raw_repetition_period_min_max_s=[float(periods.min()), float(periods.max())],
                              profile_identity="Named ordered archive schedule plus raw repeated-rest boundaries",
                              measured_temperature_min_median_max_C=[
                                  float(cycle.temperature.min()), float(np.median(cycle.temperature)),
                                  float(cycle.temperature.max())])
        cycles[name] = cycle
    cell = CellData("cell_A_a123", q25, chg_ah/qlocal, chg[:, 4], (1-dis_ah/qlocal)[::-1],
                    dis[::-1, 4], cycles,
                    {"temperature_C": temperature, "specimen": "A1-007",
                     "source_files": {str(p): sha256(p) for p in (dynamic_path, ocv_path)},
                     "raw_current_sign": "positive charge, converted to positive discharge",
                     "Qnom_reference_Ah": q25, "measured_low_current_Q_Ah": qlocal,
                     "measured_low_current_charge_Ah": float(chg_ah[-1]),
                     "target_OCV_Q_use": "Characterization only; all evaluated models use OCV25/Q25 reference"})
    local_curves = build_curves(cell)
    cell.metadata["diagnostic_OCV_points"] = local_curves.to_dict()
    LOGGER.info("LOADED %sC Q25=%s Qlocal=%s", temperature, q25, qlocal)
    return cell, local_curves


def load_source() -> tuple[dict, OCVCurves]:
    summary = json.loads((SOURCE/"summary.json").read_text())
    for path, digest in summary["manifest"]["source_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Frozen source code changed: {path}")
    for path, digest in summary["manifest"]["loader"]["source_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Frozen source data changed: {path}")
    ocv = json.loads(OCV_SOURCE.read_text())
    if sha256(OCV_SOURCE) != summary["manifest"]["old_ocv_sha256"]:
        raise ValueError("Frozen independent OCV changed")
    curves = OCVCurves.from_dict(ocv["points"])
    for model in MODELS:
        fit = summary["fits"][model]
        if fit["identified_on"] != "DST" or fit["target_refit"]:
            raise ValueError("Source fits are not frozen DST25 fits")
        p = fit["parameters"]
        if not (p["tau2_s"] <= 3600.000001 and .9 <= p["Q_ratio"] <= 1.1 and .9 <= p["z0"] <= 1):
            raise ValueError("Source parameters violate registered bounds")
    return summary, curves


def fit_local(cell: CellData, curves: OCVCurves) -> dict:
    source = cell.cycles["DST"]
    lower, upper, starts = bounds_starts(cell, source, curves, "V6")
    cfg = protocol()["optimizer"]
    attempts, converged = [], []
    begun = time.perf_counter()
    for index, start in enumerate(starts):
        LOGGER.info("FIT_LOCAL_V6 temperature=%s start=%s", cell.metadata["temperature_C"], index)
        result = least_squares(lambda theta: predict(source, cell, curves, theta, "V6")[0]-source.voltage,
                               start, bounds=(lower, upper), **cfg)
        attempt = {"start_index": index, "success": bool(result.success), "status": int(result.status),
                   "message": str(result.message), "nfev": int(result.nfev),
                   "theta": result.x.tolist(), "squared_residual_V2": float(2*result.cost)}
        attempts.append(attempt)
        LOGGER.info("LOCAL_FIT_RESULT %s", json.dumps(attempt))
        if result.success:
            converged.append((float(result.cost), index, result))
        else:
            LOGGER.error("Local V6 NLS did not converge: %s", attempt)
    if not converged:
        save_json(OUTPUT/f'{cell.metadata["temperature_C"]}C'/"FAILED_V6_local.json", {"attempts": attempts})
        raise RuntimeError("No converged local V6 source start")
    _, index, chosen = min(converged, key=lambda r: r[0])
    proximity = np.minimum((chosen.x-lower)/(upper-lower), (upper-chosen.x)/(upper-lower))
    names = parameter_names("V6")
    hits = [names[k] for k in np.flatnonzero(proximity < 1e-4)]
    singular = np.linalg.svd(chosen.jac, compute_uv=False)
    return {"model": "V6_local", "simulation_model": "V6", "identified_on": "DST",
            "source_temperature_C": cell.metadata["temperature_C"], "target_refit": False,
            "criterion_excluded": True, "theta": chosen.x.tolist(), "attempts": attempts,
            "selected_start": index, "parameters": unpack(chosen.x, "V6", cell, source),
            "n_parameters_near_bounds": len(hits), "near_bound_parameters": hits,
            "jacobian_condition": float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
            "elapsed_seconds": time.perf_counter()-begun}


def decide_thermal(maes: dict) -> dict:
    cfg = protocol()["temperature_A123"]
    temperatures = {}
    for temperature in cfg["temperatures_C"]:
        profiles = {}
        for profile in cfg["targets_per_temperature"]:
            values = maes[str(temperature)][profile]
            if not all(np.isfinite(values[m]) and values[m] >= 0 for m in MODELS) or values["V1"] <= 0:
                raise ValueError("Thermal criterion requires finite MAEs and positive V1")
            best = min(MODELS[1:], key=lambda m: values[m])
            profiles[profile] = {"best_model": best, "V1_mae_mV": values["V1"],
                                 "best_mae_mV": values[best],
                                 "best_reduction_fraction": (values["V1"]-values[best])/values["V1"],
                                 "passes": bool(values[best] <= .75*values["V1"])}
        temperatures[str(temperature)] = {"profiles": profiles,
                                        "passes_both_profiles": all(p["passes"] for p in profiles.values())}
    count = sum(x["passes_both_profiles"] for x in temperatures.values())
    return {"verdict": "POSITIVA" if count >= 3 else "NEGATIVA", "passing_temperatures": count,
            "required_temperatures": 3, "temperatures": temperatures,
            "criterion": protocol()["decision"]["thermal"], "local_V6_excluded": True}


def diagnostic_recovery(values: dict) -> dict:
    def ratio(baseline: float, label: str):
        if baseline <= 0:
            LOGGER.warning("Unavailable recovery: nonpositive %s denominator", label)
            return None
        return (baseline-values["V6_local"])/baseline
    return {"relative_to_transferred_V6": ratio(values["V6"], "V6"),
            "relative_to_V1": ratio(values["V1"], "V1"),
            "absolute_gain_vs_transferred_V6_mV": values["V6"]-values["V6_local"],
            "criterion_excluded": True}


def plot_errors(out: Path, cell: CellData, predictions: dict) -> None:
    for name, prediction in predictions.items():
        cycle = cell.cycles[name]
        fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
        hours = cycle.t/3600
        axes[0].plot(hours, cycle.voltage, "k", lw=.6, label="measured")
        for index, (model, voltage) in enumerate(prediction.items(), 1):
            axes[0].plot(hours, voltage, lw=.5, label=model)
            axes[index].plot(hours, (voltage-cycle.voltage)*1000, lw=.5)
            axes[index].axhline(0, color="black", lw=.5)
            axes[index].set_ylabel(model+" error (mV)")
        axes[0].set(ylabel="Voltage (V)", title=f'A123 {cell.metadata["temperature_C"]}C {name}')
        axes[0].legend(ncol=3)
        axes[-1].set_xlabel("Time (h)")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out/f"{name}_error.{suffix}", dpi=150)
        plt.close(fig)


def evaluate_temperature(cell: CellData, curves: OCVCurves, source_fits: dict, local_fit: dict) -> dict:
    temperature = cell.metadata["temperature_C"]
    out = OUTPUT/f"{temperature}C"
    out.mkdir(parents=True, exist_ok=True)
    rows, maes, predictions, states, diagnostics = [], {}, {}, {}, {}
    fits = {**source_fits, "V6_local": local_fit}
    for name, cycle in cell.cycles.items():
        scopes = masks(cycle, .02, 60)
        maes[name], predictions[name], states[name] = {}, {}, {}
        for model, fit in fits.items():
            simulation_model = "V6" if model == "V6_local" else model
            voltage, z = predict(cycle, cell, curves, np.asarray(fit["theta"]), simulation_model)
            predictions[name][model], states[name][model] = voltage, z
            for scope, mask in scopes.items():
                if mask.any():
                    value = metrics(voltage[mask]-cycle.voltage[mask])
                else:
                    LOGGER.warning("Empty scope: %sC %s %s %s", temperature, name, model, scope)
                    value = {m: None for m in METRICS}
                rows.append({"temperature_C": temperature, "profile": name, "model": model, "scope": scope,
                             "criterion_excluded": model == "V6_local", "n_samples": int(mask.sum()),
                             **{m+"_mV": value[m] for m in METRICS}})
                if scope == "global":
                    maes[name][model] = value["mae"]
        np.savez_compressed(out/f"{name}_trajectories.npz", t=cycle.t, current_A=cycle.current,
                            measured_voltage_V=cycle.voltage, temperature_C=cycle.temperature,
                            original_soc=cycle.soc, net_Ah=cycle.net_Ah, **predictions[name],
                            **{m+"_soc": z for m, z in states[name].items()})
        diagnostics[name] = {m: {"soc_min": float(z.min()), "soc_max": float(z.max()),
                                 "OCV_boundary_samples": int(((z < curves.lo) | (z > curves.hi)).sum())}
                             for m, z in states[name].items()}
        LOGGER.info("MAE %sC %s %s", temperature, name, json.dumps(maes[name]))
    result = {"temperature_C": temperature, "loader": cell.metadata,
              "cycles": {n: c.metadata for n, c in cell.cycles.items()}, "local_V6_fit": local_fit,
              "global_mae_mV": maes, "metrics": rows, "state_diagnostics": diagnostics,
              "diagnostic_recovery": {n: diagnostic_recovery(v) for n, v in maes.items()}}
    save_json(out/"summary.json", result)
    write_tables(out, rows)
    plot_errors(out, cell, predictions)
    return result


def write_report(results: dict, decision: dict) -> None:
    lines = ["# A123: transferencia termica e recalibracao local V6", "",
             "Celula A1-007. Protocolo primario 78fefb3 inalterado; diagnostico local registrado em 3d0d047.", "",
             f'## Veredito termico: **{decision["verdict"]}**', "",
             f'Temperaturas aprovadas nos DOIS perfis: **{decision["passing_temperatures"]}/4**; exigidas >=3.', "",
             "V1/V5/V6/V7 sao exatamente os ajustes DST25 anteriores, sem reajuste. V6_local NAO participa do criterio.", "",
             "| T (C) | Perfil | V1 | V5 | V6 | V7 | Melhor transferido | Reducao | Passa | V6 local |",
             "|---:|---|---:|---:|---:|---:|---|---:|---|---:|"]
    for temperature, result in results.items():
        for name in ("FUDS", "US06"):
            v = result["global_mae_mV"][name]
            d = decision["temperatures"][temperature]["profiles"][name]
            lines.append(f"| {temperature} | {name} | "+" | ".join(f"{v[m]:.3f}" for m in MODELS)
                         +f' | {d["best_model"]} | {d["best_reduction_fraction"]:.2%} | {d["passes"]} | {v["V6_local"]:.3f} |')
    lines += ["", "Todos os MAEs em mV. Melhor pode diferir por perfil, conforme criterio congelado.", "",
              "## Diagnostico fora do criterio", "",
              "V6_local: RC/Q/z0 ajustados SOMENTE em DST da propria temperatura; FUDS/US06 sem ajuste.",
              "OCV_mean25 continua fixa. Q limitado a +/-10% do Q medido a 25C, z0=[0.9,1], ambos taus<=3600s.",
              "OCV e capacidade a outras temperaturas foram lidas apenas para caracterizacao, nao usadas nos modelos.", "",
              "| T (C) | Perfil | V6 transferido | V6 local | Ganho (mV) | Fracao recuperada vs V6 | Reducao vs V1 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for temperature, result in results.items():
        for name in ("FUDS", "US06"):
            v, d = result["global_mae_mV"][name], result["diagnostic_recovery"][name]
            fractions = ["--" if d[k] is None else f"{d[k]:.2%}"
                         for k in ("relative_to_transferred_V6", "relative_to_V1")]
            lines.append(f'| {temperature} | {name} | {v["V6"]:.3f} | {v["V6_local"]:.3f} | '
                         f'{d["absolute_gain_vs_transferred_V6_mV"]:.3f} | '+ " | ".join(fractions)+" |")
    lines += ["", "Fracoes assinadas, nao recortadas; nao sao fracoes de teto/oraculo nem entram no veredito.", "",
              "## Capacidade medida e parametros locais", "",
              "| T (C) | Q baixa corrente (Ah, diagnostico) | Q V6 local (Ah) | z0 local | tau1/tau2 (s) | No limite |",
              "|---:|---:|---:|---:|---|---|"]
    for temperature, result in results.items():
        fit, q = result["local_V6_fit"], result["loader"]["measured_low_current_Q_Ah"]
        p = fit["parameters"]
        lines.append(f'| {temperature} | {q:.6f} | {p["Qnom_Ah"]:.6f} | {p["z0"]:.6f} | '
                     f'{p["tau1_s"]:.3f}/{p["tau2_s"]:.3f} | {fit["n_parameters_near_bounds"]}: '
                     f'{", ".join(fit["near_bound_parameters"])} |')
    lines += ["", "## Suposicoes e limites", "",
              "- Q25 e OCV25 independentes fixos; sem normalizacao da capacidade no fim de alvos ou inversao OCV.",
              "- SOC causal por integracao trapezoidal da corrente reamostrada a 1s; carga tem corrente negativa.",
              "- Preparacao CCCV3.6V, taper<0.06A e repouso ~120s verificados por perfil. z0=1 para V1/V5;",
              "  V6/V7 transferem Q/z0 ajustados em DST25; RC inicial zero e suposicao preservada.",
              "- Simulacao unica continua, ZOH exato e mesmos limiares/mascaras SOC/inversoes do teste anterior.",
              "- Identidade de perfil verificada por sequencia de etapas/nome e repeticoes brutas 361/601/1373s.",
              "  Correlacao de atraso no grid fixo pode cair abaixo de 0.95 por amostragem/deriva de fase;",
              "  valor e aviso preservados no manifesto/log, sem alterar corrente, perfis ou criterio.",
              "- OCV grau9 sem restricao de monotonicidade; Q/z0/OCV podem ser confundidos.",
              "- Capacidade e OCV baixa corrente dependem de temperatura/rate e incluem polarizacao;",
              "  datas dos ensaios OCV e dinamicos diferem, sem controle independente de envelhecimento.",
              "- Recalibracao local mede recuperabilidade operacional de V6 com OCV25 fixa, nao uma decomposicao fisica unica.",
              "- MAE local baixo nao garante erros extremos baixos: maximo/P99 e janelas pos-inversao",
              "  continuam nas tabelas estratificadas; a 0C o erro local de fim de descarga permanece grande.",
              "- Temperatura nominal por arquivo; sensor auxiliar medido retido. A1-008 nao utilizado.",
              "- Resultados bloqueados e relatorios anteriores preservados; esta e uma nova execucao concluida.", "",
              "## Artefatos", "",
              "- [Manifesto completo](results/lfp_static_temperature/cell_A_a123/summary.json)",
              "- [Tabela principal](results/lfp_static_temperature/cell_A_a123/table.csv)",
              "- [Log](results/lfp_static_temperature/cell_A_a123/run.log)",
              "- [Testes](results/lfp_static_temperature/cell_A_a123/tests.log)",
              "- Cada temperatura: metricas globais/estratificadas CSV/TEX, trajetorias NPZ e graficos de erro.",
              "- [OCV por temperatura](results/lfp_static_temperature/cell_A_a123/ocv_temperature.png)",
              "- SHA-256 dos ZIPs, membros extraidos, fontes, ajustes e resultados anteriores no manifesto.", ""]
    (ROOT/"report_static_temperature.md").write_text("\n".join(lines))


def main() -> None:
    verify_protocols()
    previous = snapshot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUTPUT/"run.log", mode="w"), logging.StreamHandler()])
    save_json(OUTPUT/"previous_artifacts_sha256.json", previous)
    source, curves = load_source()
    archives = extract_archives()
    q25 = source["qnom_measured_Ah"]
    temperatures = protocol()["temperature_A123"]["temperatures_C"]
    loaded = {str(t): load_temperature(t, archives, q25) for t in temperatures}
    results = {}
    for temperature, (cell, _) in loaded.items():
        local = fit_local(cell, curves)
        results[temperature] = evaluate_temperature(cell, curves, source["fits"], local)
    decision = decide_thermal({t: r["global_mae_mV"] for t, r in results.items()})
    table = []
    for t, r in results.items():
        for profile in ("DST", "FUDS", "US06"):
            values = r["global_mae_mV"][profile]
            d = decision["temperatures"][t]["profiles"].get(profile)
            table.append({"temperature_C": int(t), "profile": profile,
                          **{m+"_mae_mV": values[m] for m in (*MODELS, "V6_local")},
                          "best_transferred_model": None if d is None else d["best_model"],
                          "best_reduction_fraction": None if d is None else d["best_reduction_fraction"],
                          "primary_pass": None if d is None else d["passes"],
                          "V6_local_recovery_fraction": r["diagnostic_recovery"][profile]["relative_to_transferred_V6"]})
    write_tables(OUTPUT, table)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    z = np.linspace(.01, .99, 1001)
    for ax, index, title in zip(axes, (0, 1), ("Low-current charge", "Low-current discharge")):
        ax.plot(z, curves.evaluate(z)[index], "k--", label="25C source fixed")
        for t, (_, local_curves) in loaded.items():
            ax.plot(z, local_curves.evaluate(z)[index], label=t+"C diagnostic")
        ax.set(ylabel="Voltage (V)", title=title)
        ax.legend()
    axes[-1].set_xlabel("SOC (own low-current measured capacity; diagnostic only)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT/f"ocv_temperature.{suffix}", dpi=150)
    plt.close(fig)
    manifest = {"protocol": protocol(), "protocol_sha256": sha256(PROTOCOL_PATH),
                "diagnostic": json.loads(DIAGNOSTIC_PATH.read_text()),
                "diagnostic_sha256": sha256(DIAGNOSTIC_PATH), "preregistration_commit": "78fefb3",
                "diagnostic_registration_commit": DIAGNOSTIC_COMMIT,
                "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                            text=True).strip(),
                "source_sha256": {str(ROOT/p): sha256(ROOT/p) for p in
                                  ("exp_a123_temperature.py", "data_lfp.py", "lfp_ocv.py", "exp_lfp_static.py",
                                   "exp_lfp_gonogo.py", "exp_static_replication.py", "lfp_static_protocol.json")},
                "source_summary_path": str(SOURCE/"summary.json"),
                "source_summary_sha256": sha256(SOURCE/"summary.json"),
                "independent_OCV25_sha256": sha256(OCV_SOURCE), "archives": archives,
                "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "openpyxl", "matplotlib")},
                "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                "previous_artifact_count": len(previous)}
    if snapshot() != previous:
        raise RuntimeError("Previous results changed during thermal evaluation")
    save_json(OUTPUT/"summary.json", {"complete": True, "manifest": manifest, "Qnom_reference_Ah": q25,
                                     "frozen_DST25_fits": source["fits"], "temperatures": results,
                                     "decision": decision})
    write_report(results, decision)
    LOGGER.info("COMPLETE thermal %s", json.dumps(decision))


if __name__ == "__main__":
    main()
