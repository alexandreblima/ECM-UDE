"""Reevaluate the preregistered M50T checkpoints on aging and HPPC data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import torch

import exp_unified as U
import reviewer_round2_c as C
import reviewer_round2_e as E
from ecm import ECMParams
from ecm_2rc import ECMParams2RC

ROOT = C.ROOT
OUT = ROOT / "results/reviewer_round2/F"
C_DIR = ROOT / "results/reviewer_round2/C"


def read_json(path: Path):
    return json.loads(path.read_text())


def first_timestamp(path: Path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = [name for name in workbook.sheetnames if name.startswith("Channel")]
        if len(sheets) != 1:
            raise ValueError(f"{path}: expected one Channel sheet")
        rows = workbook[sheets[0]].iter_rows(values_only=True)
        header, row = next(rows), next(rows)
        return row[header.index("Date_Time")]
    finally:
        workbook.close()


def read_channel(path: Path) -> dict[str, np.ndarray]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = [name for name in workbook.sheetnames if name.startswith("Channel")]
        if len(sheets) != 1:
            raise ValueError(f"{path}: expected one Channel sheet")
        rows = workbook[sheets[0]].iter_rows(values_only=True)
        header = next(rows)
        normalized = {
            re.sub(r"[\s_]+", "", str(name)).lower(): i
            for i, name in enumerate(header) if name is not None
        }
        fields = {
            "time": "testtime(s)", "step": "stepindex", "voltage": "voltage(v)",
            "current": "current(a)", "charge_ah": "chargecapacity(ah)",
            "discharge_ah": "dischargecapacity(ah)",
        }
        missing = [key for key, name in fields.items() if name not in normalized]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        indices = {key: normalized[name] for key, name in fields.items()}
        temperature = next(
            (i for name, i in normalized.items() if name.startswith("auxtemperature")),
            None,
        )
        if temperature is not None:
            indices["temperature"] = temperature
        data = {key: [] for key in indices}
        for row in rows:
            try:
                values = {
                    key: float(row[index])
                    for key, index in indices.items() if key != "temperature"
                }
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"{path}: invalid required sample") from exc
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}: nonfinite required sample")
            for key, value in values.items():
                data[key].append(int(value) if key == "step" else value)
            if temperature is not None:
                value = row[temperature]
                data["temperature"].append(float(value) if value is not None else math.nan)
        result = {key: np.asarray(value) for key, value in data.items()}
        if len(result["time"]) < 2 or np.any(np.diff(result["time"]) <= 0):
            raise ValueError(f"{path}: insufficient or nonmonotone samples")
        return result
    finally:
        workbook.close()


def block_families(cell: str, selection: dict) -> list[tuple[str, list[Path]]]:
    paths = sorted((C.DATA_ROOT / cell).glob("*UDDS*.xlsx"))
    groups: dict[str, list[Path]] = {}
    for path in paths:
        key = path.name.split("_Channel_")[0]
        groups.setdefault(key, []).append(path)
    source_name = Path(selection["profiles"]["source"]["start_path"]).name
    source_key = source_name.split("_Channel_")[0]
    if source_key not in groups:
        raise ValueError(f"{cell}: C0 source family not found")
    ordered = []
    for name, group in groups.items():
        by_time = sorted((first_timestamp(path), path) for path in group)
        if len({item[0] for item in by_time}) != len(by_time):
            raise ValueError(f"{cell}: ambiguous fragment chronology in {name}")
        ordered.append((by_time[0][0], name, [path for _, path in by_time]))
    ordered.sort(key=lambda item: item[0])
    index = next(i for i, (_, name, _) in enumerate(ordered) if name == source_key)
    return [(name, paths) for _, name, paths in ordered[index + 1:]]


def candidate_events(cell: str, paths: list[Path], cache: dict) -> tuple[list[dict], list[dict]]:
    def cached(path):
        if path not in cache:
            cache[path] = read_channel(path)
        return cache[path]

    with patch.object(C, "_fragment_paths", return_value=paths), patch.object(C, "_channel_data", side_effect=cached):
        events, _ = C._discover_events(cell)
    accepted, excluded = [], []
    for event in events:
        if event["step"] != C.UDDS_STEP:
            continue
        duration = event["end_time_s"] - event["start_time_s"]
        n = event["n_samples"]
        mean = event["current_sum_A"] / n
        std = math.sqrt(max(0.0, event["current_sum_sq_A2"] / n - mean * mean))
        median_dt = statistics.median(event["dt"]) if event["dt"] else math.inf
        reasons = []
        if not event["bounded"]:
            reasons.append("unbounded step")
        if duration < 1369.0:
            reasons.append("duration below 1369 s")
        if not 0.05 <= median_dt <= 0.2 or not event["dt"] or max(event["dt"]) > 1.5:
            reasons.append("sampling or continuity")
        if event["current_max_A"] - event["current_min_A"] < 2 or std < 0.5:
            reasons.append("current variability")
        if not event["same_file_rest_before"]:
            reasons.append("no preceding same-file rest")
        if reasons:
            excluded.append({"start_path": str(event["start_path"].relative_to(C.DATA_ROOT)),
                             "start_time_s": event["start_time_s"], "reasons": reasons})
            continue
        event.update(duration_s=duration, median_dt_s=median_dt, current_std_A=std)
        accepted.append(C._json_event(event))
    return accepted, excluded


def source_assets(cell: str, selection: dict, manifest: dict):
    capacity = selection["capacity_ah"]
    cycle, _ = C._profile_arrays(cell, selection["profiles"]["source"], capacity)
    context = C._context_for_profile(cycle, C._cfg(capacity), source=True)
    path = C_DIR / "run" / f"{cell}_nls_fits.json"
    if C._sha256(path) != manifest["nls_fit_sha256"][cell]:
        raise ValueError(f"{cell}: NLS fit hash mismatch")
    fits = read_json(path)
    p1 = ECMParams.from_dict(fits["ecm_1rc_nls"]["params"])
    p2 = ECMParams2RC.from_dict(fits["ecm_2rc_nls"]["params"])
    return context["stats"], p1, p2


def models_for_cell(cell: str, stats: dict, p1, p2, capacity: float, manifest: dict, seeds):
    cfg = C._cfg(capacity)
    models = [("ecm_2rc_nls", "deterministic", p2)]
    for name in C.VARIANTS:
        for seed in seeds:
            path = C_DIR / "run" / "checkpoints" / cell / f"{name}_seed{seed}.pt"
            relative = str(path.relative_to(ROOT))
            if C._sha256(path) != manifest["checkpoint_sha256"][relative]:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            model = U.build_model(name, stats, p1, cfg)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            models.append((name, seed, model))
    return models


def context(cycle: dict, stats: dict, capacity: float) -> dict:
    cfg = C._cfg(capacity)
    ctx = C._context_for_profile(cycle, cfg, stats=stats)
    ctx["cfg"] = cfg
    ctx["mask"][:] = True
    return ctx


def predict(model, name: str, ctx: dict):
    prediction, truth = C._continuous_vectors(model, name, ctx)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite prediction for {name}")
    return prediction, truth


def hppc_cycle(cell: str, selection: dict):
    path = C.DATA_ROOT / selection["hppc_path"]
    raw = read_channel(path)
    boundaries = np.r_[0, np.flatnonzero(np.diff(raw["step"]) != 0) + 1, len(raw["step"])]
    runs = list(zip(boundaries[:-1], boundaries[1:]))
    charge = [(l, r) for l, r in runs if raw["step"][l] == 3]
    rests = [(l, r) for l, r in runs if raw["step"][l] == 4]
    if len(charge) != 1 or len(rests) != 1:
        raise ValueError(f"{cell}: ambiguous HPPC charge/rest")
    cl, cr = charge[0]
    left, right = rests[0]
    if not (cr <= left and raw["voltage"][cr-1] >= 4.19
            and 0 <= raw["current"][cr-1] <= 0.051
            and raw["time"][right-1] - raw["time"][left] >= 3500
            and np.max(np.abs(raw["current"][left:right])) <= 0.02):
        raise ValueError(f"{cell}: HPPC full-charge/rest verification failed")
    time = raw["time"][left:]
    relative = time - time[0]
    grid = np.arange(int(np.floor(relative[-1] / C.DT)) + 1) * C.DT
    current = np.interp(grid, relative, raw["current"][left:])
    voltage = np.interp(grid, relative, raw["voltage"][left:])
    temp = raw.get("temperature")
    fallback = temp is None or np.isfinite(temp[left:]).sum() < 2
    if not fallback:
        finite = np.isfinite(temp[left:])
        observed = temp[left:][finite]
        fallback = not 0 <= float(np.median(observed)) <= 60 or np.all(observed == 0)
    temperature = np.full(len(grid), 23.0) if fallback else np.interp(grid, relative[finite], observed)
    net = np.r_[0.0, np.cumsum(-(current[1:] + current[:-1]) * 0.5 * C.DT / 3600)]
    cycle = U.make_cycle(grid, current, voltage, temperature, net, 1.0, C._cfg(selection["capacity_ah"]))
    pulse = np.zeros(len(grid), dtype=bool)
    for l, r in runs:
        if l < left or raw["time"][r-1] - raw["time"][l] > 15:
            continue
        if np.max(np.abs(raw["current"][l:r])) <= 0.02:
            continue
        pulse |= (grid >= raw["time"][l] - time[0]) & (grid <= raw["time"][r-1] - time[0])
    pulse &= np.abs(current) > 0.02
    soc = cycle["SoC"].numpy()
    masks = {
        "full": np.ones(len(grid), dtype=bool),
        "soc_0.2_to_0.8": (soc >= 0.2) & (soc <= 0.8),
        "soc_outside_0.2_to_0.8": (soc < 0.2) | (soc > 0.8),
        "pulse": pulse,
        "rest": np.abs(current) <= 0.02,
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"{cell}: empty HPPC evaluation mask")
    return cycle, masks, {"path": str(path.relative_to(C.DATA_ROOT)),
                          "sha256": C._sha256(path), "z0": 1.0,
                          "reference_step": 4, "temperature_fallback_C": 23.0 if fallback else None,
                          "n_samples": len(grid), "soc_min": float(soc.min()),
                          "soc_max": float(soc.max())}


def metrics(prediction, truth, mask):
    return float(np.mean(np.abs(prediction[mask] - truth[mask])) * 1000)


def run_cell(cell, selection, manifest, seeds, smoke):
    source_paths = set(selection["profiles"]["source"]["fragments"])
    source_paths.add(selection["profiles"]["source"]["start_path"])
    for relative in source_paths:
        if C._sha256(C.DATA_ROOT / relative) != manifest["raw_input_sha256"][relative]:
            raise ValueError(f"{cell}: C0 source fragment hash mismatch: {relative}")
    stats, p1, p2 = source_assets(cell, selection, manifest)
    capacity = selection["capacity_ah"]
    models = models_for_cell(cell, stats, p1, p2, capacity, manifest, seeds)
    hppc, masks, hppc_meta = hppc_cycle(cell, selection)
    expected = manifest["raw_input_sha256"][selection["hppc_path"]]
    if hppc_meta["sha256"] != expected:
        raise ValueError(f"{cell}: HPPC hash mismatch")
    ctx = context(hppc, stats, capacity)
    hppc_rows = []
    for name, seed, model in models:
        prediction, truth = predict(model, name, ctx)
        row = {"cell": cell, "model": name, "seed": seed, "capacity_ah": capacity}
        for label, mask in masks.items():
            row[f"mae_{label}_mV"] = metrics(prediction, truth, mask)
            row[f"n_{label}"] = int(mask.sum())
        hppc_rows.append(row)
    del ctx, hppc
    print(f"{cell}: HPPC complete", flush=True)

    diagnostics = list((C.DATA_ROOT / cell).glob(f"*0_05C*{cell}*Channel*.xlsx"))
    if len(diagnostics) != 1 or diagnostics[0] != C.DATA_ROOT / selection["c20_path"]:
        raise ValueError(f"{cell}: more than one or ambiguous C/20; implement latest-before-block selection")
    if C._sha256(diagnostics[0]) != manifest["raw_input_sha256"][selection["c20_path"]]:
        raise ValueError(f"{cell}: C/20 hash mismatch")
    with (C_DIR / "run/metrics_by_seed.csv").open(newline="") as stream:
        baseline = {
            (row["model"], row["seed"]): float(row["mae"])
            for row in csv.DictReader(stream)
            if row["cell"] == cell and row["condition"] == "target_aggregate"
        }
    diagnostic_time = first_timestamp(diagnostics[0])
    blocks = block_families(cell, selection)
    audit = {"hppc": hppc_meta, "blocks": [], "capacity_update_available": False}
    aging_rows = []
    for position, (label, paths) in enumerate(blocks, 1):
        if diagnostic_time >= first_timestamp(paths[0]):
            raise ValueError(f"{cell}: C/20 does not precede {label}")
        cache: dict = {}
        accepted, excluded = candidate_events(cell, paths, cache)
        block = {"position": position, "label": label,
                 "input_sha256": {str(p.relative_to(C.DATA_ROOT)): C._sha256(p) for p in paths},
                 "selected": accepted, "excluded": excluded,
                 "capacity_ah": capacity, "capacity_diagnostic": selection["c20_path"]}
        audit["blocks"].append(block)
        if smoke:
            accepted = accepted[:1]
        if not accepted:
            print(f"{cell}: block {position} has no eligible episodes", flush=True)
            continue
        accum = {(name, seed): [0.0, 0] for name, seed, _ in models}
        for episode in accepted:
            with patch.object(C, "_channel_data", side_effect=lambda path: cache[path]):
                cycle, _ = C._profile_arrays(cell, episode, capacity)
            ctx = context(cycle, stats, capacity)
            for name, seed, model in models:
                prediction, truth = predict(model, name, ctx)
                total = float(np.abs(prediction - truth).sum()) * 1000
                entry = accum[(name, seed)]
                entry[0] += total
                entry[1] += len(truth)
        for (name, seed), (error_sum, n) in accum.items():
            for arm in ("fixed", "updated_capacity"):
                aging_rows.append({"cell": cell, "block_position": position, "block_label": label,
                                   "arm": arm, "model": name, "seed": seed, "capacity_ah": capacity,
                                   "n_episodes": len(accepted), "n_samples": n, "mae_mV": error_sum / n,
                                   "source_block_target_mae_mV": baseline[(name, str(seed))],
                                   "increase_vs_source_mV": error_sum / n - baseline[(name, str(seed))]})
        print(f"{cell}: block {position} complete ({len(accepted)} episodes)", flush=True)
    return aging_rows, hppc_rows, audit


def inference(aging_rows, hppc_rows):
    grouped = {}
    for row in aging_rows:
        grouped.setdefault(("aging", str(row["block_position"]), row["arm"]), []).append(row)
    grouped[("hppc", "full", "fixed")] = [
        {**row, "mae_mV": row["mae_full_mV"]} for row in hppc_rows
    ]
    tests, cell_rows = [], []
    for (analysis, condition, arm), rows in grouped.items():
        cells = sorted({row["cell"] for row in rows})
        for reference in ("physics_gradient", "ecm_2rc_nls"):
            for variant in C.NEURAL_VARIANTS:
                deltas = []
                for cell in cells:
                    by_key = {(r["model"], str(r["seed"])): r["mae_mV"] for r in rows if r["cell"] == cell}
                    per_seed = []
                    for seed in C.SEEDS:
                        ref_seed = str(seed) if reference == "physics_gradient" else "deterministic"
                        delta = by_key[(variant, str(seed))] - by_key[(reference, ref_seed)]
                        per_seed.append(delta)
                        cell_rows.append({"analysis": analysis, "condition": condition, "arm": arm,
                                          "cell": cell, "variant": variant, "reference": reference,
                                          "seed": seed, "difference_mV": delta})
                    deltas.append(float(np.median(per_seed)))
                record = E._inferential_row(cohort="F", configuration=arm, variant=variant,
                    reference=reference, condition=f"{analysis}:{condition}", differences=np.asarray(deltas))
                record["analysis"] = analysis
                record["arm"] = arm
                record["position"] = condition
                tests.append(record)
    E._adjust_test_rows(tests, ("analysis", "position", "arm", "reference"))
    return tests, cell_rows


def plot_differences(rows, path):
    groups = sorted({(r["analysis"], r["condition"], r["arm"]) for r in rows},
                    key=lambda x: (x[0] == "hppc", x[0], x[1], x[2]))
    columns = min(3, len(groups))
    nrows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(7.16, nrows * 2.7),
                             squeeze=False, sharey=True)
    flat = axes.flatten()
    for ax, group in zip(flat, groups):
        chosen = [r for r in rows if (r["analysis"], r["condition"], r["arm"]) == group
                  and r["reference"] == "physics_gradient"]
        cells = sorted({r["cell"] for r in chosen})
        ax.axhspan(-1, 1, color="0.88")
        ax.axhline(0, color="0.4", lw=0.5)
        for i, cell in enumerate(cells):
            for offset, variant, marker in ((-0.18, "output_residual", "o"), (0.18, "ecm_ude", "s")):
                values = [r["difference_mV"] for r in chosen if r["cell"] == cell and r["variant"] == variant]
                ax.scatter(np.full(len(values), i + offset), values, s=8, color="0.65")
                ax.scatter(i + offset, np.median(values), s=25, color="black", marker=marker)
        ax.set_xticks(range(len(cells)), cells, rotation=45, fontsize=8)
        ax.set_title(f"{group[0]} {group[1]}\n{group[2]}", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in flat[len(groups):]:
        ax.set_visible(False)
    axes[0][0].set_ylabel("MAE neural − physics-gradient (mV)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path, aging, hppc, tests, audit):
    lines = ["# Part F — M50T transfer", "",
             "TOST was preregistered in amendment 009; superiority and inferiority are exploratory. "
             "No model was refitted. The inferential unit is the cell; differences use the median across five seeds.", "",
             "## F1 — Aging", "",
             "Blocks were ordered by observed chronology, not by N labels. "
             "The local copy contains only one C/20 for each eligible cell; therefore, the most recent diagnostic "
             "before each block is the source diagnostic. The two arms are numerically identical and do not measure "
             "the effect of recalibrating degraded capacity: the difference between arms is zero. Episodes and exclusions "
             "are recorded in `selection_audit.json`. The increase is the MAE of the later block minus the aggregate MAE "
             "of the C targets in the source block for the same cell, model, and seed; the C aggregate uses the window mask, "
             "whereas F uses all points of complete episodes.", "",
             "| Position | Cells | Model | Median MAE (mV) | Median increase (mV) |",
             "|---:|---:|---|---:|---:|"]
    for position in sorted({r["block_position"] for r in aging}):
        for model in ("ecm_2rc_nls", *C.VARIANTS):
            selected = [r for r in aging if r["block_position"] == position and r["model"] == model and r["arm"] == "fixed"]
            cells = {r["cell"] for r in selected}
            values = [np.median([r["mae_mV"] for r in selected if r["cell"] == cell]) for cell in cells]
            changes = [np.median([r["increase_vs_source_mV"] for r in selected if r["cell"] == cell]) for cell in cells]
            lines.append(f"| {position} | {len(cells)} | {model} | {np.median(values):.3f} | "
                         f"{np.median(changes):.3f} |")
    lines += ["", "| Cell | Later blocks observed | Blocks with eligible episodes |",
              "|---|---:|---:|"]
    for cell, details in audit.items():
        blocks = details["blocks"]
        lines.append(f"| {cell} | {len(blocks)} | {sum(bool(block['selected']) for block in blocks)} |")
    observed_positions = {block["position"] for details in audit.values() for block in details["blocks"]}
    evaluated_positions = {r["block_position"] for r in aging}
    for position in sorted(observed_positions - evaluated_positions):
        lines += ["", f"Observed position {position} has n=0: the available blocks at this position contain no "
                  "eligible episode; therefore, it has no inference or panel.", ""]
    lines += ["", "## F2 — HPPC", "",
              "The simulation starts at rest after a full CC/CV charge (SOC=1), with RC states set to zero. "
              "The pulse mask uses steps of up to 15 s with |I|>0,02 A; rest uses |I|≤0,02 A. "
              "Continuous voltage is evaluated at all resampled points; missing/invalid temperature is set to 23 °C. "
              "The range 0,2–0,8 is nominal and does not imply that all training data lie within it. "
              "For W3, HPPC precedes C/20 and uses another channel; its association with the same diagnostic could not "
              "be confirmed, although both precede the source. Joint interpretation for this cell requires caution.", "",
              "The OCV of the evaluated circuits is the source-fitted OCV, not the initial frozen C/20 curve. "
              "The table compares its extrapolation at SOC=1 with the initial C/20 calibration; the "
              "2RC identification window ends near SOC=0,60. Extrapolated values up to approximately 9 V "
              "explain a significant part of the high HPPC error and are not a measured physical OCV.", "",
              "| Cell | Maximum SOC in 2RC identification | 2RC OCV at SOC=1 (V) | Initial C/20 OCV at SOC=1 (V) |",
              "|---|---:|---:|---:|"]
    selections = read_json(C_DIR / "c0_segments.json")["cells"]
    for cell in audit:
        fit = read_json(C_DIR / "run" / f"{cell}_nls_fits.json")["ecm_2rc_nls"]["params"]
        fitted = np.polynomial.chebyshev.chebval(1.0, fit["ocv_coeffs"])
        original = np.polynomial.chebyshev.chebval(1.0, selections[cell]["ocv_coefficients"])
        lines.append(f"| {cell} | {fit['soc_train_max']:.3f} | {fitted:.3f} | {original:.3f} |")
    lines += ["",
              "| Model | Median total MAE (mV) | SOC 0,2–0,8 | Outside | Pulses | Rests |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in ("ecm_2rc_nls", *C.VARIANTS):
        selected = [r for r in hppc if r["model"] == model]
        cells = {r["cell"] for r in selected}
        vals = [np.median([np.median([r[f"mae_{mask}_mV"] for r in selected if r["cell"] == cell])
                           for cell in cells])
                for mask in ("full", "soc_0.2_to_0.8", "soc_outside_0.2_to_0.8", "pulse", "rest")]
        lines.append("| " + model + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## Inference", "",
              "| Analysis | Group | Arm | Reference | Variant | n | Median Δ (mV) | TOST p | Holm p | Raw superiority p | Holm p |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in tests:
        lines.append(f"| {r['analysis']} | {r['position']} | {r['arm']} | {r['reference']} | {r['variant']} | "
                     f"{r['n']} | {r['median_difference_mV']:.3f} | {r['p_tost_raw']:.6g} | "
                     f"{r['p_tost_holm']:.6g} | {r['p_superiority_raw']:.6g} | {r['p_superiority_holm']:.6g} |")
    lines += ["", "The tables by cell, seed, and arm are in `aging_metrics.csv`, `hppc_metrics.csv`, and "
              "`cell_differences.csv`; one-sided p-values are in `c2_style_tost.csv`; provenance is in `manifest.json`. "
              "The hashes in `E/manifest.json` record the version before the F rows were added to the synthesis; "
              "the previous and current synthesis versions are identified in the F manifest."]
    path.write_text("\n".join(lines) + "\n")


def update_synthesis(tests, differences, out):
    csv_path = ROOT / "results/reviewer_round2/E/synthesis.csv"
    tex_path = ROOT / "results/reviewer_round2/E/synthesis.tex"
    shutil.copy2(csv_path, out / "synthesis_before_F.csv")
    shutil.copy2(tex_path, out / "synthesis_before_F.tex")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        existing = list(reader)
    if fields is None:
        raise ValueError("Missing synthesis header")
    added, latex = [], []
    groups = sorted({(r["analysis"], r["position"], r["arm"]) for r in tests},
                    key=lambda x: (x[0] == "hppc", x[1], x[2]))
    for analysis, position, arm in groups:
        n_cells = next(r["n"] for r in tests if (r["analysis"], r["position"], r["arm"]) == (analysis, position, arm))
        label = (f"M50T F1 bloco {position} {arm} (n={n_cells})" if analysis == "aging"
                 else f"M50T F2 HPPC (n={n_cells})")
        for variant in C.NEURAL_VARIANTS:
            row = {field: "" for field in fields}
            row.update(configuration=label, protocol_status="registered TOST; exploratory one-sided",
                       variant=variant)
            for prefix, ref in (("vs_pg", "physics_gradient"), ("vs_ecm2", "ecm_2rc_nls")):
                match = [r for r in tests if (r["analysis"], r["position"], r["arm"]) == (analysis, position, arm)
                         and r["reference"] == ref and r["variant"] == variant]
                if len(match) != 1:
                    raise ValueError(f"Missing synthesis contrast: {label}/{variant}/{ref}")
                test = match[0]
                row["analysis_unit_n"] = test["n"]
                row[f"{prefix}_status"] = "preregistered TOST; exploratory superiority and inferiority"
                row[f"{prefix}_n_tests"] = 1
                for kind, field in (("better", "p_superiority_holm"), ("equivalent", "p_tost_holm"),
                                    ("worse", "p_inferiority_holm")):
                    row[f"{prefix}_{kind}_holm_count"] = int(test[field] < 0.05)
                row[f"{prefix}_unresolved_holm_count"] = int(
                    all(test[field] >= 0.05 for field in ("p_superiority_holm", "p_tost_holm", "p_inferiority_holm")))
                for measure in ("median_difference_mV", "minimum_difference_mV", "maximum_difference_mV",
                                "p_superiority_holm", "p_tost_holm", "p_inferiority_holm"):
                    row[f"{prefix}_{measure}"] = test[measure]
                selected = [r for r in differences if (r["analysis"], r["condition"], r["arm"]) == (analysis, position, arm)
                            and r["variant"] == variant and r["reference"] == ref]
                cells = sorted({r["cell"] for r in selected})
                medians = [np.median([r["difference_mV"] for r in selected if r["cell"] == cell]) for cell in cells]
                row[f"{prefix}_cells_d_below_0"] = sum(value < 0 for value in medians)
                row[f"{prefix}_cells_d_below_minus1"] = sum(value < -1 for value in medians)
                row[f"{prefix}_cells_d_above_0"] = sum(value > 0 for value in medians)
                row[f"{prefix}_cells_within_pm1"] = sum(-1 <= value <= 1 for value in medians)
            added.append(row)
            def format_contrast(prefix):
                counts = "/".join(str(row[f"{prefix}_{kind}_holm_count"]) for kind in ("better", "equivalent", "worse"))
                return (f"{counts}; {row[f'{prefix}_median_difference_mV']:.3f} "
                        f"[{row[f'{prefix}_minimum_difference_mV']:.3f},{row[f'{prefix}_maximum_difference_mV']:.3f}]")
            latex_label = (f"M50T F1 {position} {'fixed' if arm == 'fixed' else 'updated'} (n={n_cells})"
                           if analysis == "aging" else f"M50T F2 HPPC (n={n_cells})")
            latex_variant = variant.replace("_", r"\_")
            latex.append(f"{latex_label} & {latex_variant} & PR+EX/PR+EX & "
                         f"{format_contrast('vs_pg')} & {format_contrast('vs_ecm2')} \\\\")
    if any(row["configuration"] in {new["configuration"] for new in added} for row in existing):
        raise ValueError("F synthesis rows already exist; refusing to overwrite")
    text = tex_path.read_text()
    if text.count(r"\bottomrule") != 1:
        raise ValueError("Ambiguous synthesis LaTeX table")
    text = text.replace("output_residual", r"output\_residual").replace("ecm_ude", r"ecm\_ude")
    text = text.replace(r"\footnotesize" + "\n", r"\footnotesize" + "\n" + r"\setlength{\tabcolsep}{2pt}" + "\n", 1)
    text = text.replace("Configuration & Variant & Status &", "Configuration & Variant & Status PG/NLS &")
    text = text.replace("Protocolo original — ", "Original — ").replace("Original — NMC Samsung", "Original — Samsung NMC")
    text = text.replace("PG PR+EX; NLS PR+EX", "PR+EX/PR+EX").replace("PG EX; NLS PR+EX", "EX/PR+EX")
    text = text.replace(
        "M50T counts refer to the single population-level contrast over seven cells; descriptive cell directions are reported separately.",
        "M50T counts refer to single population-level contrasts; F ranges are across cell medians, with sample sizes shown. The F1 number is the later-block position; updated denotes updated capacity.",
    )
    text = text.replace("EX: exploratory superiority.", "EX: exploratory one-sided tests.")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + added)
    tex_path.write_text(text.replace(r"\bottomrule", "\n".join(latex) + "\n" + r"\bottomrule", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--attempt", default=None, help="Fresh output subdirectory for a retry")
    args = parser.parse_args()
    if args.attempt is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        raise ValueError("Invalid attempt name")
    out = OUT / (args.attempt or ("smoke" if args.smoke else "run"))
    out.mkdir(parents=True, exist_ok=False)
    selection = read_json(C_DIR / "c0_segments.json")
    manifest = read_json(C_DIR / "run/manifest.json")
    if C._sha256(C_DIR / "c0_segments.json") != manifest["selection_sha256"]:
        raise ValueError("C0 selection hash mismatch")
    if C._sha256(C_DIR / "run/metrics_by_seed.csv") != manifest["output_sha256"]["metrics_by_seed.csv"]:
        raise ValueError("C baseline metrics hash mismatch")
    cells = ("G1",) if args.smoke else C.CELLS
    seeds = (42,) if args.smoke else C.SEEDS
    aging, hppc, audit = [], [], {}
    for cell in cells:
        a, h, audit[cell] = run_cell(cell, selection["cells"][cell], manifest, seeds, args.smoke)
        aging.extend(a)
        hppc.extend(h)
    if aging:
        C._write_csv(out / "aging_metrics.csv", aging)
    C._write_csv(out / "hppc_metrics.csv", hppc)
    (out / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.smoke:
        return
    if not aging:
        raise ValueError("No eligible later aging blocks")
    tests, differences = inference(aging, hppc)
    C._write_csv(out / "c2_style_tost.csv", tests)
    C._write_csv(out / "cell_differences.csv", differences)
    plot_differences(differences, out / "cell_differences.pdf")
    write_report(out / "report.md", aging, hppc, tests, audit)
    update_synthesis(tests, differences, out)
    output_names = ("aging_metrics.csv", "hppc_metrics.csv", "c2_style_tost.csv",
                    "cell_differences.csv", "cell_differences.pdf", "selection_audit.json", "report.md")
    for name in output_names:
        destination = OUT / name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.copy2(out / name, destination)
    provenance = {
        "protocol_commit": "9e562a8", "training_performed": False,
        "c_manifest_sha256": C._sha256(C_DIR / "run/manifest.json"),
        "c_selection_sha256": C._sha256(C_DIR / "c0_segments.json"),
        "c_baseline_metrics_sha256": C._sha256(C_DIR / "run/metrics_by_seed.csv"),
        "c_checkpoint_sha256_verified": manifest["checkpoint_sha256"],
        "c_nls_sha256_verified": manifest["nls_fit_sha256"],
        "source_raw_sha256": manifest["raw_input_sha256"],
        "F_raw_sha256": {cell: {"hppc": data["hppc"]["sha256"],
                                  "blocks": [block["input_sha256"] for block in data["blocks"]]}
                         for cell, data in audit.items()},
        "source_sha256": {name: C._sha256(ROOT / name) for name in (
            "reviewer_round2_f.py", "reviewer_round2_c.py", "reviewer_round2_e.py",
            "exp_unified.py", "unified_dynamics.py")},
        "output_sha256": {name: C._sha256(OUT / name) for name in output_names},
        "synthesis_before_sha256": {suffix: C._sha256(out / f"synthesis_before_F.{suffix}") for suffix in ("csv", "tex")},
        "synthesis_after_sha256": {suffix: C._sha256(ROOT / f"results/reviewer_round2/E/synthesis.{suffix}")
                                   for suffix in ("csv", "tex")},
    }
    (OUT / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
