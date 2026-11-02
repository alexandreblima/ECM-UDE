"""Source-only static 2RC transfer; previous go/no-go artifacts are read-only."""
from __future__ import annotations

import csv
from dataclasses import replace
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.optimize import least_squares

from data_lfp import (ROOT, OUTPUT as PREVIOUS, CellData, Cycle,
                      load_a123, load_mendeley, save_json, sha256)
from exp_lfp_gonogo import METRICS, masks, metrics, rc_voltage, simulate
from lfp_ocv import OCVCurves

OUTPUT = ROOT / "results" / "lfp_static"
PROTOCOL_PATH = ROOT / "lfp_static_protocol.json"
MODELS = ("V1", "V5", "V6", "V7")
LOGGER = logging.getLogger(__name__)
FloatArray: TypeAlias = NDArray[np.float64]


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def snapshot_previous() -> dict[str, str]:
    return {str(path): sha256(path) for path in sorted(PREVIOUS.rglob("*")) if path.is_file()}


def unpack(theta: FloatArray, model: str, cell: CellData, cycle: Cycle) -> dict:
    r0, r1, r2, tau1, tau2 = np.exp(theta[:5])
    q = cell.qnom_Ah
    z0 = cycle.z0
    coeffs = None
    offset = 5
    if model in ("V6", "V7"):
        q = cell.qnom_Ah * theta[5]
        z0 = theta[6]
        offset = 7
    if model in ("V5", "V7"):
        coeffs = theta[offset:]
        if len(coeffs) != 10:
            raise ValueError("Degree-9 OCV requires ten Chebyshev coefficients")
    expected = {"V1": 5, "V5": 15, "V6": 7, "V7": 17}[model]
    if len(theta) != expected:
        raise ValueError(f"{model}: wrong number of optimizer parameters")
    if tau1 > tau2:
        r1, r2, tau1, tau2 = r2, r1, tau2, tau1
    return {"R0": float(r0), "R1": float(r1), "R2": float(r2),
            "C1": float(tau1 / r1), "C2": float(tau2 / r2),
            "tau1_s": float(tau1), "tau2_s": float(tau2),
            "Qnom_Ah": float(q), "Q_ratio": float(q / cell.qnom_Ah),
            "z0": float(z0),
            "ocv_coefficients_V": None if coeffs is None else coeffs.tolist()}


def predict(cycle: Cycle, cell: CellData, curves: OCVCurves,
            theta: FloatArray, model: str) -> tuple[FloatArray, FloatArray]:
    p = unpack(theta, model, cell, cycle)
    z = p["z0"] - cycle.net_Ah / p["Qnom_Ah"]
    state_cycle = replace(cycle, soc=z, z0=p["z0"])
    rctheta = np.log([p["R0"], p["R1"], p["R2"], p["tau1_s"], p["tau2_s"]])
    if model in ("V5", "V7"):
        rctheta = np.r_[rctheta, p["ocv_coefficients_V"]]
        prediction = simulate(state_cycle, p["Qnom_Ah"], curves, rctheta, "V4")[0]
    else:
        prediction = simulate(state_cycle, p["Qnom_Ah"], curves, rctheta, "V1")[0]
    return prediction, z


def parameter_names(model: str) -> list[str]:
    names = ["log_R0", "log_R1", "log_R2", "log_tau1", "log_tau2"]
    if model in ("V6", "V7"):
        names += ["Q_ratio", "z0"]
    if model in ("V5", "V7"):
        names += [f"OCV_c{k}" for k in range(10)]
    return names


def bounds_starts(cell: CellData, cycle: Cycle, curves: OCVCurves,
                  model: str) -> tuple[FloatArray, FloatArray, list[FloatArray]]:
    cfg = protocol()
    b = cfg["bounds"]
    lower = list(np.log([b["R0_R1_R2_ohm"][0]] * 3 + [b["tau1_tau2_s"][0]] * 2))
    upper = list(np.log([b["R0_R1_R2_ohm"][1]] * 3 + [b["tau1_tau2_s"][1]] * 2))
    if model in ("V6", "V7"):
        lower += [b["Q_ratio"][0], b["z0"][0]]
        upper += [b["Q_ratio"][1], b["z0"][1]]
    if model in ("V5", "V7"):
        lower += [-np.inf] * 10
        upper += [np.inf] * 10
    starts = []
    for spec in cfg["starts"]:
        theta = list(np.log([spec[k] for k in ("R0", "R1", "R2", "tau1", "tau2")]))
        z = cycle.soc
        if model in ("V6", "V7"):
            theta += [spec["Q_ratio"], spec["z0"]]
            z = spec["z0"] - cycle.net_Ah / (cell.qnom_Ah * spec["Q_ratio"])
        if model in ("V5", "V7"):
            rc = (rc_voltage(cycle.current, spec["R1"], spec["tau1"])
                  + rc_voltage(cycle.current, spec["R2"], spec["tau2"]))
            coeffs = np.linalg.lstsq(
                chebvander(2 * np.clip(z, curves.lo, curves.hi) - 1, 9),
                cycle.voltage + spec["R0"] * cycle.current + rc, rcond=None)[0]
            theta += coeffs.tolist()
        starts.append(np.asarray(theta))
    return np.asarray(lower), np.asarray(upper), starts


def fit(cell: CellData, curves: OCVCurves, model: str) -> dict:
    source = cell.cycles["DST"]
    lo, hi, starts = bounds_starts(cell, source, curves, model)
    cfg = protocol()["optimizer"]
    attempts, results = [], []
    begun = time.perf_counter()
    for index, start in enumerate(starts):
        LOGGER.info("FIT %s %s start=%d", cell.name, model, index)
        result = least_squares(
            lambda theta: predict(source, cell, curves, theta, model)[0] - source.voltage,
            start, bounds=(lo, hi), method=cfg["method"], loss=cfg["loss"],
            x_scale=cfg["x_scale"], max_nfev=cfg["max_nfev"],
            ftol=cfg["ftol"], xtol=cfg["xtol"], gtol=cfg["gtol"])
        attempt = {"start_index": index, "success": bool(result.success),
                   "status": int(result.status), "message": str(result.message),
                   "nfev": int(result.nfev), "squared_residual_V2": float(2 * result.cost),
                   "rmse_mV": float(np.sqrt(np.mean(result.fun**2)) * 1000),
                   "optimality": float(result.optimality), "theta": result.x.tolist()}
        attempts.append(attempt)
        LOGGER.info("FIT_RESULT %s", json.dumps(attempt))
        if result.success:
            results.append((float(result.cost), index, result))
        else:
            LOGGER.error("NLS failed to converge: %s", attempt)
    if not results:
        save_json(OUTPUT / cell.name / f"FAILED_{model}.json", {"attempts": attempts})
        raise RuntimeError(f"No converged fixed start for {cell.name} {model}")
    _, index, chosen = min(results, key=lambda item: item[0])
    finite = np.isfinite(lo) & np.isfinite(hi)
    proximity = np.full(len(lo), np.inf)
    proximity[finite] = np.minimum(
        (chosen.x[finite] - lo[finite]) / (hi[finite] - lo[finite]),
        (hi[finite] - chosen.x[finite]) / (hi[finite] - lo[finite]))
    names = parameter_names(model)
    bound_names = [names[k] for k in np.flatnonzero(proximity < 1e-4)]
    singular = np.linalg.svd(chosen.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    fitted = {"model": model, "identified_on": "DST", "target_refit": False,
              "theta": chosen.x.tolist(), "parameter_names": names,
              "parameters": unpack(chosen.x, model, cell, source),
              "selected_start": index, "attempts": attempts,
              "n_parameters_near_bounds": len(bound_names), "near_bound_parameters": bound_names,
              "jacobian_condition": condition, "elapsed_seconds": time.perf_counter() - begun}
    if bound_names:
        LOGGER.warning("%s %s bounds: %s", cell.name, model, bound_names)
    return fitted


def decide(maes: dict[str, dict[str, float]]) -> dict:
    cfg = protocol()["decision"]
    profiles = {}
    for name in protocol()["targets"]:
        values = maes[name]
        best = min(("V5", "V6", "V7"), key=lambda model: values[model])
        reduction = (values["V1"] - values[best]) / values["V1"]
        denominator = values["V1"] - values["V4"]
        fractions = {}
        if denominator <= 0:
            LOGGER.warning("%s nonpositive oracle ceiling denominator=%s", name, denominator)
            fractions = {model: None for model in ("V5", "V6", "V7")}
            attribution = "UNAVAILABLE"
        else:
            fractions = {model: (values["V1"] - values[model]) / denominator
                         for model in ("V5", "V6", "V7")}
            attribution = ("SOC" if fractions["V6"] > fractions["V5"] else
                           "OCV" if fractions["V5"] > fractions["V6"] else "TIE")
        profiles[name] = {"mae_mV": values, "best_model": best,
                          "best_reduction_fraction": reduction,
                          "positive": bool(reduction >= cfg["minimum_reduction"]),
                          "ceiling_denominator_mV": denominator,
                          "ceiling_recovered_fractions": fractions,
                          "attribution": attribution}
    labels = [value["attribution"] for value in profiles.values()]
    cell_attribution = labels[0] if len(set(labels)) == 1 else "MIXED/INCONCLUSIVE"
    return {"positive": all(value["positive"] for value in profiles.values()),
            "verdict": "POSITIVA" if all(value["positive"] for value in profiles.values()) else "NEGATIVA",
            "attribution": cell_attribution, "profiles": profiles, "criterion": cfg}


def plot(out: Path, cell: CellData, curves: OCVCurves, fits: dict,
         predictions: dict[str, dict[str, FloatArray]]) -> None:
    z = np.linspace(curves.lo, curves.hi, 1001)
    fig, ax = plt.subplots(figsize=(10, 5))
    charge, dis, mean, _ = curves.evaluate(z)
    for label, y in (("low-current mean", mean), ("low-current charge", charge),
                     ("low-current discharge", dis)):
        ax.plot(z, y, label=label, ls="--")
    for model in ("V5", "V7"):
        coeffs = fits[model]["parameters"]["ocv_coefficients_V"]
        ax.plot(z, chebval(2*z-1, coeffs), label=model + " (DST only)")
    ax.set(xlabel="SOC (model coordinate)", ylabel="OCV (V)", title=cell.name)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"identified_ocv.{ext}", dpi=150)
    plt.close(fig)
    for name, values in predictions.items():
        cycle = cell.cycles[name]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        hours = cycle.t / 3600
        axes[0].plot(hours, cycle.voltage, lw=0.7, color="black", label="measured")
        for model, voltage in values.items():
            axes[0].plot(hours, voltage, lw=0.5, label=model)
            axes[1].plot(hours, (voltage-cycle.voltage)*1000, lw=0.6, label=model)
        axes[0].set_ylabel("Voltage (V)")
        axes[0].legend(ncol=6)
        axes[1].set(xlabel="Time (h)", ylabel="Error (mV)")
        fig.suptitle(cell.name + " " + name + " | V4 = previous target oracle")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}_error.{ext}", dpi=150)
        plt.close(fig)


def run_cell(cell: CellData, previous_hashes: dict[str, str]) -> dict:
    out = OUTPUT / cell.name
    out.mkdir(parents=True, exist_ok=True)
    old_summary_path = PREVIOUS / cell.name / "summary.json"
    old = json.loads(old_summary_path.read_text())
    ocv_path = PREVIOUS / cell.name / "ocv_manifest.json"
    ocv = json.loads(ocv_path.read_text())
    if cell.metadata["source_files"] != old["manifest"]["source_files"]:
        raise ValueError("Loader inputs changed from previous experiment")
    if ocv["input_sha256"] != cell.metadata["source_files"]:
        raise ValueError("Independent OCV input mismatch")
    for path, digest in old["manifest"]["source_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Previous shared source changed: {path}")
    curves = OCVCurves.from_dict(ocv["points"])
    fits = {model: fit(cell, curves, model) for model in MODELS}
    save_json(out / "DST_fits.json", fits)
    rows, predictions, maes, state_diagnostics, old_oracle = [], {}, {}, {}, {}
    for name in ("DST", "FUDS", "US06"):
        cycle = cell.cycles[name]
        scope_masks = masks(cycle, protocol()["current_threshold_A"],
                            protocol()["post_reversal_window_seconds"])
        prediction_map, soc_map = {}, {}
        for model in MODELS:
            theta = np.asarray(fits[model]["theta"])
            prediction_map[model], soc_map[model] = predict(cycle, cell, curves, theta, model)
        old_path = PREVIOUS / cell.name / f"{name}_trajectories.npz"
        with np.load(old_path) as trace:
            for key, expected in (("t", cycle.t), ("current_A", cycle.current),
                                  ("measured_voltage_V", cycle.voltage), ("soc", cycle.soc)):
                if not np.array_equal(trace[key], expected):
                    raise ValueError(f"Old/new sample mismatch: {name} {key}")
            prediction_map["V4"] = trace["V4"].copy()
        old_oracle[name] = {"trajectory_path": str(old_path), "sha256": sha256(old_path),
                            "fit": old["oracle_fits"][name],
                            "bounds_exemption": "Existing prior oracle, not reidentified; previous tau bound100000s"}
        maes[name] = {}
        state_diagnostics[name] = {}
        for model, voltage in prediction_map.items():
            for scope, mask in scope_masks.items():
                count = int(mask.sum())
                if count:
                    measured = metrics(voltage[mask]-cycle.voltage[mask])
                else:
                    LOGGER.warning("Empty scope %s %s %s %s", cell.name, name, model, scope)
                    measured = {metric: None for metric in METRICS}
                rows.append({"cell": cell.name, "profile": name, "model": model, "scope": scope,
                             "n_samples": count,
                             **{metric+"_mV": measured[metric] for metric in METRICS}})
                if scope == "global":
                    maes[name][model] = measured["mae"]
            if model != "V4":
                z = soc_map[model]
                state_diagnostics[name][model] = {
                    "soc_min": float(z.min()), "soc_max": float(z.max()),
                    "soc_outside_0_1_samples": int(np.count_nonzero((z<0)|(z>1))),
                    "ocv_boundary_samples": int(np.count_nonzero((z<curves.lo)|(z>curves.hi))),
                    "Qnom_Ah": fits[model]["parameters"]["Qnom_Ah"],
                    "z0_transferred": fits[model]["parameters"]["z0"]}
        predictions[name] = prediction_map
        np.savez_compressed(out / f"{name}_trajectories.npz", t=cycle.t,
                            current_A=cycle.current, measured_voltage_V=cycle.voltage,
                            original_soc=cycle.soc, net_Ah=cycle.net_Ah,
                            **prediction_map, **{model+"_soc": z for model, z in soc_map.items()})
        LOGGER.info("METRICS %s %s MAEs=%s", cell.name, name, maes[name])
    summary = {
        "cell": cell.name, "complete": True,
        "manifest": {"protocol": protocol(), "protocol_sha256": sha256(PROTOCOL_PATH),
                     "preregistration_commit": "50e85e1",
                     "execution_git_commit": subprocess.check_output(
                         ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                     "python": platform.python_version(), "platform": platform.platform(),
                     "dependencies": {pkg: importlib.metadata.version(pkg) for pkg in
                                      ("numpy", "scipy", "openpyxl", "xlrd", "matplotlib")},
                     "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                     "source_sha256": {str(ROOT/f): sha256(ROOT/f) for f in
                                       ("data_lfp.py", "lfp_ocv.py", "exp_lfp_gonogo.py",
                                        "exp_lfp_static.py", "lfp_static_protocol.json")},
                     "old_summary_sha256": sha256(old_summary_path),
                     "old_ocv_sha256": sha256(ocv_path),
                     "previous_snapshot_path": str(OUTPUT / "previous_artifacts_sha256.json"),
                     "previous_artifact_count": len(previous_hashes),
                     "loader": cell.metadata,
                     "cycles": {name: cycle.metadata for name, cycle in cell.cycles.items()}},
        "qnom_measured_Ah": cell.qnom_Ah, "fits": fits, "old_oracle": old_oracle,
        "global_mae_mV": maes, "state_diagnostics": state_diagnostics,
        "metrics": rows, "decision": decide(maes),
        "caveats": [
            *cell.metadata["caveats"],
            "V1 reidentified under3600s RC bound, not the previous V1 numbers.",
            "V4 prior target oracle has different RC bounds; recovered ceiling is a diagnostic comparison, not matched-bound model superiority.",
            "Degree9 OCV unconstrained like previous V4; monotonicity/physical validity not imposed.",
            "Q/z0 and OCV polynomial are confounded in V7; attribution is operational V5-vs-V6, not unique causal identification.",
            "SOC masks use original loader SOC, not model-dependent fitted SOC.",
            "Source fits all DST samples; no parameter adjustment in target cycles.",
            "A different best model per target is allowed by the pre-fixed criterion; not a claim of one universally best transferable model."]}
    save_json(out / "summary.json", summary)
    with (out/"table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tex = [r"\begin{tabular}{lllrrrrr}", r"\hline",
           r"Profile & Model & Scope & N & MAE & RMSE & Max & P99 \\", r"\hline"]
    for row in rows:
        labels = [str(row[k]).replace("_", r"\_") for k in ("profile", "model", "scope")]
        vals = [str(row["n_samples"])] + [
            "--" if row[m+"_mV"] is None else f'{row[m+"_mV"]:.3f}' for m in METRICS]
        tex.append(" & ".join(labels+vals)+r" \\")
    tex += [r"\hline", r"\end{tabular}", "% Error units mV; V4 previous target-fitted oracle."]
    (out/"table.tex").write_text("\n".join(tex)+"\n")
    plot(out, cell, curves, fits, predictions)
    LOGGER.info("DECISION %s %s", cell.name, json.dumps(summary["decision"]))
    return summary


def write_report(summaries: dict) -> None:
    a, b = summaries["cell_A_a123"], summaries["cell_B_mendeley"]
    text = ["# Transferência estática LFP (NLS ECM-2RC)", "",
            f"**A123 primária: {a['decision']['verdict']}.**",
            f"**Mendeley réplica: {b['decision']['verdict']}.**",
            f"**Decisão global: {a['decision']['verdict']}**, exclusivamente pelo A123.",
            "A réplica "+("concorda." if a["decision"]["verdict"] == b["decision"]["verdict"] else "não concorda."),
            "", "## Critério congelado",
            "Em FUDS **e** US06, o melhor V5/V6/V7 deve reduzir MAE ≥25% contra V1.",
            "O melhor pode diferir por alvo. Todos V1/V5/V6/V7 foram identificados somente em DST.",
            "V4 é o oráculo **anterior**, apenas lido; não usa o novo limite de τ≤3600s.",
            "", "| Célula | Perfil | V1 MAE (mV) | Melhor | MAE (mV) | Redução (%) | Transferência |",
            "|---|---|---:|---|---:|---:|---|"]
    for label, s in (("A", a), ("B", b)):
        for name, d in s["decision"]["profiles"].items():
            text.append(f"| {label} | {name} | {d['mae_mV']['V1']:.3f} | {d['best_model']} | "
                        f"{d['mae_mV'][d['best_model']]:.3f} | {100*d['best_reduction_fraction']:.2f} | "
                        f"{'POSITIVA' if d['positive'] else 'NEGATIVA'} |")
    text += ["", "## Fração do teto recuperada e atribuição", "",
             "Fração=(MAE_V1−MAE_Vk)/(MAE_V1−MAE_V4), sem recorte para [0,1].",
             "V6>V5: SOC; V5>V6: OCV. Discordância entre alvos: mista/inconclusiva.",
             "", "| Célula | Perfil | V4 MAE (mV) | Fração V5 | Fração V6 | Fração V7 | Atribuição |",
             "|---|---|---:|---:|---:|---:|---|"]
    for label, s in (("A", a), ("B", b)):
        for name, d in s["decision"]["profiles"].items():
            vals = ["ND" if d["ceiling_recovered_fractions"][m] is None
                    else f'{d["ceiling_recovered_fractions"][m]:.4f}' for m in ("V5", "V6", "V7")]
            text.append(f"| {label} | {name} | {d['mae_mV']['V4']:.3f} | "
                        +" | ".join(vals)+f" | {d['attribution']} |")
    for label, s in (("A", a), ("B", b)):
        text.append(f"\nAtribuição da célula {label}: **{s['decision']['attribution']}**.\n")
    text += ["## Métricas globais (mV)", "",
             "| Célula | Perfil | Modelo | MAE | RMSE | Máximo | P99 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for label, s in (("A", a), ("B", b)):
        for row in s["metrics"]:
            if row["scope"] == "global":
                text.append(f"| {label} | {row['profile']} | {row['model']} | "
                            +" | ".join(f'{row[m+"_mV"]:.3f}' for m in METRICS)+" |")
    text += ["", "## Parâmetros de SOC e limites", "",
             "| Célula | Modelo | Q (Ah) | Q/Qmedido | z0 | τrápida (s) | τlenta (s) | Nº no limite | Parâmetros no limite | Cond. Jacobiano |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for label, s in (("A", a), ("B", b)):
        for m, fitted in s["fits"].items():
            p = fitted["parameters"]
            condition = fitted["jacobian_condition"]
            text.append(f"| {label} | {m} | {p['Qnom_Ah']:.9f} | {p['Q_ratio']:.6f} | {p['z0']:.6f} | "
                        f"{p['tau1_s']:.3f} | {p['tau2_s']:.3f} | {fitted['n_parameters_near_bounds']} | "
                        f"{', '.join(fitted['near_bound_parameters']) or 'nenhum'} | "
                        f"{condition if condition is None else format(condition, '.3g')} |")
    text += ["", "Os nomes no limite referem-se às coordenadas originais do otimizador;",
             "os τ da tabela são reordenados em rápida/lenta. Assim, `log_tau1`",
             "pode corresponder à constante lenta após a troca de rótulos.",
             "V7 tem condição de Jacobiano elevada nas duas células; Q/z0 e OCV",
             "não devem ser interpretados como estimativas físicas unívocas.",
             "V7 não necessariamente supera V6 no objetivo DST: a OCV polinomial",
             "com dez coeficientes não contém exatamente a família PCHIP fixa.",
             "", "## Parâmetros RC (identificados em DST)", "",
             "| Célula | Modelo | R0 (Ω) | R1 (Ω) | C1 (F) | R2 (Ω) | C2 (F) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, s in (("A", a), ("B", b)):
        for model, fitted in s["fits"].items():
            p = fitted["parameters"]
            text.append(f"| {label} | {model} | "+" | ".join(
                f"{p[key]:.6g}" for key in ("R0", "R1", "C1", "R2", "C2"))+" |")
    text += ["", "## Verificação", "",
             "Seis testes passaram, incluindo recorrência RC escalar independente",
             "para todas as trajetórias salvas, métricas/decisões recalculadas,",
             "limiar exato25% e integridade de todos os arquivos anteriores.",
             "A execução usa o Python absoluto indicado abaixo. O diagnóstico",
             "Pylance do ambiente do editor reportou importações científicas",
             "ausentes; elas foram carregadas e executadas no ambiente de execução.",
             "Nenhum ambiente do editor ou dependência foi alterado."]
    text += ["", "## Suposições e limites da interpretação", "",
             "- Q/z0 de V6/V7 transferidos **inalterados** de DST aos dois alvos, conforme autorização anterior ao ajuste.",
             "- V1/V5 mantêm Q medido e z0 do loader; V6/V7: Q ±10%, z0∈[0,9;1,0]. RC reinicia só no início do perfil, nunca por janela.",
             "- Ambas as constantes de tempo têm teto3600s para impedir evasão por troca dos rótulos RC; relatório ordena rápida/lenta.",
             "- V1 é novo ajuste com esse teto; V4 conserva os ajustes/limites anteriores. Isso afeta a interpretação da fração de teto.",
             "- Frações negativas ou maiores que1 são mantidas. Denominador não positivo é marcado indisponível e gera aviso, não valor substituto.",
             "- A atribuição compara V5/V6 por perfil. Só acordo nos dois alvos gera atribuição única; não se muda a regra depois dos resultados.",
             "- V7 pode confundir Q/z0 e parametrização de OCV; atribuição não prova causalidade nem identificabilidade única.",
             "- PCHIP/intervalo comum e retenção de fronteira idênticos ao ensaio anterior; polinômio grau9 sem restrição de monotonicidade, como V4.",
             "- Estratos SOC e janelas de60s usam **as mesmas máscaras do loader** para todos os modelos; estados SOC ajustados também são salvos.",
             "- A123=A1-007, Q medido e preparo CCCV/repouso registrado. Mendeley=única cópia standalone, Q≈2,50Ah, não rótulo1,1Ah; z0 inicial de referência1 é hipótese autorizada, preparo ausente.",
             "- Ambiguidade25/30°C, ausência de T medida e períodos processados de B diferentes dos padrões continuam ressalvas.",
             "- Mesmas métricas mV globais/SOC/pós-inversão do go/no-go. T não é covariável ajustada, sem redes neurais.",
             "- Todas as tentativas NLS, convergência, limites, condição do Jacobiano, fontes/dados/protocolo SHA256 constam dos manifestos.",
             "- Resultados antigos verificados byte a byte antes/depois; nenhum arquivo de lfp_gonogo ou MANUSCRITO foi modificado.",
             "", "## Artefatos", ""]
    for name in summaries:
        text += [f"- [{name}: summary](./results/lfp_static/{name}/summary.json)",
                 f"- [{name}: tabela CSV](./results/lfp_static/{name}/table.csv)",
                 f"- [{name}: tabela LaTeX](./results/lfp_static/{name}/table.tex)",
                 f"- [{name}: OCV identificada](./results/lfp_static/{name}/identified_ocv.png)"]
        for profile in ("DST", "FUDS", "US06"):
            text.append(f"- [{name}: erros {profile}](./results/lfp_static/{name}/{profile}_error.png)")
    text += ["", "Protocolo: [lfp_static_protocol.json](./lfp_static_protocol.json).",
             "Log: [run.log](./results/lfp_static/run.log).",
             "Reprodução: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/torch_mps/bin/python -B "
             +str(ROOT/"exp_lfp_static.py")+"`.",
             "", "**Encerrado. Nenhuma fase com redes neurais iniciada.**", ""]
    (ROOT/"report_static.md").write_text("\n".join(text))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUTPUT/"run.log"),
                                  logging.StreamHandler()], force=True)
    before = snapshot_previous()
    snapshot_path = OUTPUT/"previous_artifacts_sha256.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != before:
        raise ValueError("Previous experiment artifacts differ from initial static-test snapshot")
    save_json(snapshot_path, before)
    LOGGER.info("PRE-REGISTERED protocol_sha256=%s; old_files=%d", sha256(PROTOCOL_PATH), len(before))
    summaries = {}
    for loader in (load_a123, load_mendeley):
        cell = loader()
        summaries[cell.name] = run_cell(cell, before)
    if snapshot_previous() != before:
        raise RuntimeError("Previous artifacts were changed; refusing successful completion")
    decision = {"primary": summaries["cell_A_a123"]["decision"],
                "replica": summaries["cell_B_mendeley"]["decision"],
                "global": summaries["cell_A_a123"]["decision"]["verdict"],
                "rule": protocol()["decision"]["global"],
                "previous_files_verified_unchanged": len(before)}
    save_json(OUTPUT/"global_decision.json", decision)
    write_report(summaries)
    LOGGER.info("COMPLETE global=%s old_files_unchanged=%d", decision["global"], len(before))


if __name__ == "__main__":
    main()
