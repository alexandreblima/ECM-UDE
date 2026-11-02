"""Ten-seed extension using unchanged unified training and exact equivalence tests."""
from __future__ import annotations

import csv
import ast
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata
import torch

import protocol as P
import exp_unified as U
from data_lfp import ROOT, save_json, sha256
from exp_static_replication import write_tables

OUTPUT = ROOT/"results"/"unified_10seeds"
ORIGINAL = ROOT/"results"/"unified"
PROTOCOL_PATH = ROOT/"paper_v2_seeds_protocol.json"
REGISTRATION = "d5cdc37"
PHYSICAL = ("physics_gradient", "output_residual", "ecm_ude")


def verify_protocol() -> dict:
    data = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REGISTRATION}:{PROTOCOL_PATH.name}"])
    if data != PROTOCOL_PATH.read_bytes():
        raise ValueError("Ten-seed protocol changed")
    return json.loads(data)


def shifted_wilcoxon(differences: np.ndarray, alternative: str) -> dict:
    d = np.round(np.asarray(differences, dtype=float), 10)
    if alternative not in ("greater", "less") or not np.isfinite(d).all():
        raise ValueError("Finite shifted differences and one-sided alternative required")
    d = d[d != 0]
    if not len(d):
        return {"p_raw": 1., "statistic": 0., "n_nonzero": 0}
    ranks = rankdata(np.abs(d))
    observed = float(ranks[d > 0].sum())
    sums = np.array([np.dot(ranks, signs) for signs in itertools.product((0, 1), repeat=len(d))])
    tail = sums >= observed-1e-12 if alternative == "greater" else sums <= observed+1e-12
    return {"p_raw": float(tail.mean()), "statistic": observed, "n_nonzero": len(d)}


def adjust_holm(records: list[dict]) -> None:
    largest = 0.
    for index, row in enumerate(sorted(records, key=lambda r: r["p_raw"])):
        largest = max(largest, min(1., (len(records)-index)*row["p_raw"]))
        row["p_holm"] = largest


def statistics(summary: dict) -> dict:
    cfg = verify_protocol()
    equivalence, effects = [], []
    rng = np.random.default_rng(20261101)
    for mode in ("continuous", "windowed"):
        for model in PHYSICAL:
            lower, upper = [], []
            for dataset, data in summary["datasets"].items():
                for condition in data["conditions"]:
                    baseline = data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][condition][mode]["mae"]
                    delta = np.array([data["runs"][model][str(s)]["metrics"][condition][mode]["mae"]-baseline
                                      for s in cfg["seeds"]])
                    record = {"dataset": dataset, "condition": condition, "mode": mode, "model": model,
                              "difference_median_mV": float(np.median(delta)), "differences_mV": delta.tolist(),
                              "excluded_from_main_claims": dataset == "calce" and condition in ("US06_25C", "BJDST_25C"),
                              "lower": shifted_wilcoxon(delta+1, "greater"),
                              "upper": shifted_wilcoxon(delta-1, "less")}
                    equivalence.append(record)
                    lower.append(record["lower"])
                    upper.append(record["upper"])
            if len(lower) != 20:
                raise ValueError("TOST Holm families must contain exactly20conditions")
            adjust_holm(lower)
            adjust_holm(upper)
    for row in equivalence:
        row["equivalent"] = row["lower"]["p_holm"] < .05 and row["upper"]["p_holm"] < .05
    for dataset, data in summary["datasets"].items():
        for condition in data["conditions"]:
            baseline = data["runs"]["ecm_2rc_nls"]["deterministic"]["metrics"][condition]["windowed"]["mae"]
            delta = np.array([data["runs"]["lstm"][str(s)]["metrics"][condition]["windowed"]["mae"]-baseline
                              for s in cfg["seeds"]])
            boot = np.median(rng.choice(delta, (20000, len(delta)), replace=True), axis=1)
            effects.append({"dataset": dataset, "condition": condition, "mode": "windowed",
                            "median_difference_mV": float(np.median(delta)),
                            "bootstrap95_mV": np.quantile(boot, [.025, .975]).tolist(),
                            "differences_mV": delta.tolist(),
                            "excluded_from_main_claims": dataset == "calce" and condition in ("US06_25C", "BJDST_25C")})
    return {"protocol": cfg, "equivalence": equivalence, "LSTM_effects": effects}


def report(summary: dict, stats: dict) -> None:
    lines = ["# Unificado: dez seeds e equivalencia", "",
             "Seeds42--51, cinco reutilizadas e cinco novas; hiperparametros e protocolo de treino inalterados.",
             "TOST exato deWilcoxon deslocado, margem+/-1mV; Holm20condicoes por modelo/modo/direcao.",
             "Dois testes devem rejeitar com pHolm<0.05; nao equivalencia nao implica melhoria.", "",
             "| Modo | Modelo | Equivalentes/20 | Equivalentes principais/18 |",
             "|---|---|---:|---:|"]
    for mode in ("continuous", "windowed"):
        for model in PHYSICAL:
            selected = [x for x in stats["equivalence"] if x["mode"] == mode and x["model"] == model]
            lines.append(f'| {mode} | {model} | {sum(x["equivalent"] for x in selected)}/20 | '
                         f'{sum(x["equivalent"] and not x["excluded_from_main_claims"] for x in selected)}/18 |')
    lines += ["", "## TOST por condicao", "",
              "| Dataset | Condicao | Modo | Modelo | Delta mediana(mV) | pHolm inferior | pHolm superior | Equivalente | Excluida principal |",
              "|---|---|---|---|---:|---:|---:|---|---|"]
    for x in stats["equivalence"]:
        lines.append(f'| {x["dataset"]} | {x["condition"]} | {x["mode"]} | {x["model"]} | '
                     f'{x["difference_median_mV"]:.4f} | {x["lower"]["p_holm"]:.6f} | '
                     f'{x["upper"]["p_holm"]:.6f} | {x["equivalent"]} | {x["excluded_from_main_claims"]} |')
    lines += ["", "## LSTM menos2RC (janelas apenas)", "",
              "| Dataset | Condicao | Mediana(mV) | Bootstrap95%(mV) |",
              "|---|---|---:|---|"]
    for x in stats["LSTM_effects"]:
        lines.append(f'| {x["dataset"]} | {x["condition"]} | {x["median_difference_mV"]:.3f} | '
                     f'{x["bootstrap95_mV"][0]:.3f}--{x["bootstrap95_mV"][1]:.3f} |')
    lines += ["", "## Ressalvas", "",
              "- FUDS25/UDDS25 fonte: held-out e validacao usada emearly stopping, nao teste independente.",
              "- Seeds nao replicam celulas;Wilcoxon usa simetria das diferencas deMAE porseed.",
              "- Gradiente e fisica aprendivel sem rede. Treinaveis fisicos partem de1RC, referencia e2RC.",
              "- CALCE US06/BJDST25 fallbackz0=1 ficam na familia20 mas excluidos dos achados principais.",
              "- NLS deterministico nao e multiplicado como dez ajustes independentes.",
              "- Bootstrap porseed, nao poramostra temporal; sem inferenciaLSTM fora de janelas.",
              "- [Manifesto](summary.json),[estatisticas](statistics.json),[tabela](table.csv).", ""]
    (OUTPUT/"report.md").write_text("\n".join(lines))


def main() -> None:
    cfg = verify_protocol()
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    old = json.loads((ORIGINAL/"summary.json").read_text())
    for data in old["datasets"].values():
        inputs = list(data["conditions"].values())
        if data["ocv_calibration"].get("available"):
            inputs.append(data["ocv_calibration"])
        for entry in inputs:
            if sha256((ROOT/entry["path"]).resolve()) != entry["sha256"]:
                raise ValueError(f"Unified raw/calibration input changed: {entry['path']}")
    runtime = ("protocol.py", "data.py", "data_calce.py", "ecm.py", "ecm_2rc.py",
               "models.py", "training.py", "unified_dynamics.py", "utils.py", "ude.py")
    for path in runtime:
        digest = old["manifest"]["source_sha256"].get(path)
        if digest is None:
            original_bytes = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f'{old["manifest"]["commit_hash"]}:{path}'])
            if (ROOT/path).read_bytes() != original_bytes:
                raise ValueError(f"Unmanifested unified dependency differs from training commit: {path}")
        elif sha256(ROOT/path) != digest:
            raise ValueError(f"Unified runtime source differs: {path}")
    original_code = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f'{old["manifest"]["commit_hash"]}:exp_unified.py'], text=True)
    original_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original_code).body
                          if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    current_functions = {n.name: ast.dump(n, include_attributes=False) for n in
                         ast.parse((ROOT/"exp_unified.py").read_text()).body
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    excluded_reporting = {"main", "provenance", "write_tables", "model_comparison"}
    for name, body in original_functions.items():
        if name not in excluded_reporting and current_functions.get(name) != body:
            raise ValueError(f"Unified training/evaluation function changed: {name}")
    P.OUTPUT = OUTPUT
    P.SEEDS = tuple(cfg["seeds"])
    path = OUTPUT/"summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
    else:
        summary = json.loads(json.dumps(old))
        summary["ten_seed_manifest"] = {"protocol": cfg, "protocol_sha256": sha256(PROTOCOL_PATH),
                                       "preregistration_commit": REGISTRATION,
                                       "original_summary_sha256": sha256(ORIGINAL/"summary.json"),
                                       "new_seeds": cfg["seeds"][5:], "reused_seeds": cfg["seeds"][:5],
                                       "source_sha256": sha256(Path(__file__)), "runtime": U.provenance()}
        summary["ten_seed_manifest"]["code_validation"] = {
            "runtime_file_hashes_match": True, "training_evaluation_AST_matches_original": True,
            "report_only_changes_in_exp_unified": "21a2a8c adds provenance/reporting; training and evaluation AST byte-equivalent"}
        for dataset, data in summary["datasets"].items():
            for model, runs in data["runs"].items():
                for seed, run in runs.items():
                    original = (ROOT/run["checkpoint"]).resolve()
                    if sha256(original) != run["sha256"]:
                        raise ValueError("Original checkpoint hash differs")
                    run["checkpoint"] = str(original)
                    if model.endswith("_nls"):
                        dest = OUTPUT/"checkpoints"/dataset/f"{model}.json"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, dest)
                    elif "history" in run:
                        run["history"] = str((ROOT/run["history"]).resolve())
            # Raw-file paths remain as unified loaders expect; all calls run at ROOT.
        save_json(path, summary)
    if "--report-only" not in sys.argv:
        for dataset in cfg["datasets"]:
            U.run_dataset(dataset, summary)
    elif not all(all(str(s) in data["runs"][model] for s in cfg["seeds"])
                 for data in summary["datasets"].values() for model in P.MODELS[2:]):
        raise ValueError("Cannot report incomplete ten-seed training")
    for dataset, data in summary["datasets"].items():
        if data["conditions"] != old["datasets"][dataset]["conditions"]:
            raise ValueError(f"Unified loader/source preprocessing differs from original: {dataset}")
    rows = []
    for dataset, data in summary["datasets"].items():
        for condition in data["conditions"]:
            for mode in ("windowed", "continuous"):
                for model in P.MODELS:
                    if model == "lstm" and mode == "continuous":
                        continue
                    runs = data["runs"][model]
                    selected = [runs["deterministic"]] if model.endswith("_nls") else [runs[str(s)] for s in cfg["seeds"]]
                    row = {"dataset": dataset, "condition": condition, "mode": mode, "model": model,
                           "n_seeds": len(selected)}
                    for metric in P.METRICS:
                        v = np.array([run["metrics"][condition][mode][metric] for run in selected])
                        row[metric+"_mean_mV"] = float(v.mean())
                        row[metric+"_std_mV"] = float(v.std(ddof=1)) if len(v) > 1 else 0.
                    rows.append(row)
    write_tables(OUTPUT, rows)
    stats = statistics(summary)
    save_json(OUTPUT/"statistics.json", stats)
    summary["complete"] = True
    summary["status"] = "complete"
    summary["ten_seed_manifest"]["final_source_sha256"] = sha256(Path(__file__))
    summary["ten_seed_manifest"]["verified_runtime_sha256"] = {p: sha256(ROOT/p) for p in runtime}
    save_json(path, summary)
    report(summary, stats)


if __name__ == "__main__":
    main()
