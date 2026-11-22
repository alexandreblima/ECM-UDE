"""Exploratory corrections to the saved M50T transfer evaluations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

from public_integrity import verify
import reviewer_round2_c as C
import reviewer_round2_e as E
import reviewer_round2_f as F

ROOT = C.ROOT
OUT = ROOT / "results/reviewer_round2/F2"
C_DIR = ROOT / "results/reviewer_round2/C"
F_DIR = ROOT / "results/reviewer_round2/F"
E_DIR = ROOT / "results/reviewer_round2/E"
AGING_CELLS = ("G1", "V4", "W3", "W7", "W8", "W10")
MODELS = ("ecm_2rc_nls", *C.VARIANTS)
MASKS = ("in_range", "pulse_in_range", "rest_in_range")
REFERENCES = ("physics_gradient", "ecm_2rc_nls")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verified_inputs() -> tuple[dict, dict, dict]:
    selection = F.read_json(C_DIR / "c0_segments.json")
    cm = F.read_json(C_DIR / "run/manifest.json")
    fm = F.read_json(F_DIR / "manifest.json")
    if C._sha256(C_DIR / "c0_segments.json") != cm["selection_sha256"]:
        raise ValueError("C0 selection hash mismatch")
    for name in ("aging_metrics.csv", "hppc_metrics.csv", "c2_style_tost.csv"):
        if C._sha256(F_DIR / name) != fm["output_sha256"][name]:
            raise ValueError(f"F artifact hash mismatch: {name}")
    verify(F_DIR / "report.md")
    return selection, cm, fm


def input_paths(selection: dict) -> set[str]:
    records = [selection["profiles"]["source"], *selection["profiles"]["targets"]]
    return {path for record in records for path in [record["start_path"], *record["fragments"]]}


def soc_masks(cycle: dict, original: dict, p1, p2):
    low = max(float(p1.soc_train_min), float(p2.soc_train_min))
    high = min(float(p1.soc_train_max), float(p2.soc_train_max))
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 1):
        raise ValueError(f"Invalid common SOC range: [{low}, {high}]")
    soc = cycle["SoC"].numpy()
    common = (soc >= low) & (soc <= high)
    masks = {
        "in_range": common,
        "pulse_in_range": common & original["pulse"],
        "rest_in_range": common & original["rest"],
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("No samples in a required HPPC SOC/pulse/rest mask")
    return low, high, masks


def evaluate_cell(cell: str, selection: dict, cm: dict, f_hppc: dict,
                  *, seeds: tuple[int, ...], source_targets: bool, smoke: bool):
    relevant = input_paths(selection) if source_targets else set(selection["profiles"]["source"]["fragments"])
    relevant.add(selection["profiles"]["source"]["start_path"])
    relevant.add(selection["hppc_path"])
    hashes = {}
    for relative in sorted(relevant):
        actual = C._sha256(C.DATA_ROOT / relative)
        if actual != cm["raw_input_sha256"][relative]:
            raise ValueError(f"{cell}: C input hash mismatch: {relative}")
        hashes[relative] = actual
    stats, p1, p2 = F.source_assets(cell, selection, cm)
    capacity = float(selection["capacity_ah"])
    models = F.models_for_cell(cell, stats, p1, p2, capacity, cm, seeds)
    hppc, original_masks, hppc_meta = F.hppc_cycle(cell, selection)
    low, high, masks = soc_masks(hppc, original_masks, p1, p2)
    ctx = F.context(hppc, stats, capacity)
    h_rows = []
    for name, seed, model in models:
        pred, truth = F.predict(model, name, ctx)
        original_mae = F.metrics(pred, truth, original_masks["full"])
        prior = float(f_hppc[(cell, name, str(seed))]["mae_full_mV"])
        if not math.isclose(original_mae, prior, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(f"{cell}/{name}/{seed}: F full HPPC MAE does not reproduce: {original_mae} vs {prior}")
        row = {"cell": cell, "model": name, "seed": seed,
               "soc_1rc_min": float(p1.soc_train_min), "soc_1rc_max": float(p1.soc_train_max),
               "soc_2rc_min": float(p2.soc_train_min), "soc_2rc_max": float(p2.soc_train_max),
               "soc_min": low, "soc_max": high, "hppc_n_samples": len(truth),
               "mae_full_profile_mV_not_interpretable": original_mae}
        for label, mask in masks.items():
            row[f"n_{label}"] = int(mask.sum())
            row[f"fraction_{label}_of_full"] = float(mask.mean())
            row[f"mae_{label}_mV"] = F.metrics(pred, truth, mask)
        h_rows.append(row)
    del ctx, hppc
    print(f"{cell}: HPPC full-profile reproduction and restricted masks verified", flush=True)

    source_rows = []
    if source_targets:
        targets = selection["profiles"]["targets"]
        if smoke:
            targets = targets[:1]
        cache: dict[Path, dict] = {}

        def cached(path):
            if path not in cache:
                cache[path] = F.read_channel(path)
            return cache[path]

        totals = {(name, seed): [0.0, 0] for name, seed, _ in models}
        for target in targets:
            with patch.object(C, "_channel_data", side_effect=cached):
                cycle, _ = C._profile_arrays(cell, target, capacity)
            ctx = F.context(cycle, stats, capacity)
            for name, seed, model in models:
                prediction, truth = F.predict(model, name, ctx)
                total = totals[(name, seed)]
                total[0] += float(np.abs(prediction - truth).sum()) * 1000
                total[1] += len(truth)
        for (name, seed), (error_sum, count) in totals.items():
            source_rows.append({"cell": cell, "model": name, "seed": seed,
                                "n_episodes": len(targets), "n_samples": count,
                                "mae_source_full_mV": error_sum / count})
        print(f"{cell}: {len(targets)} complete source-block targets evaluated", flush=True)
    return source_rows, h_rows, {"raw_input_sha256": hashes, "hppc": hppc_meta}


def aging_comparison(source_rows: list[dict], f_aging_rows: list[dict]) -> list[dict]:
    source = {(r["cell"], r["model"], str(r["seed"])): r for r in source_rows}
    out = []
    for row in f_aging_rows:
        if row["arm"] != "fixed":
            continue
        key = (row["cell"], row["model"], str(row["seed"]))
        prior = source[key]
        later = float(row["mae_mV"])
        baseline = float(prior["mae_source_full_mV"])
        out.append({"cell": row["cell"], "block_position": int(row["block_position"]),
                    "block_label": row["block_label"], "model": row["model"], "seed": row["seed"],
                    "n_source_episodes": int(prior["n_episodes"]), "n_source_samples": int(prior["n_samples"]),
                    "n_later_episodes": int(row["n_episodes"]), "n_later_samples": int(row["n_samples"]),
                    "mae_source_full_mV": baseline, "mae_later_full_mV": later,
                    "increase_later_minus_source_mV": later - baseline})
    if not out:
        raise ValueError("No fixed-arm aging comparisons")
    return out


def hppc_inference(rows: list[dict]):
    tests, differences = [], []
    for cohort, cells in (("all", C.CELLS), ("without_W3", tuple(c for c in C.CELLS if c != "W3"))):
        for mask in MASKS:
            for reference in REFERENCES:
                for variant in C.NEURAL_VARIANTS:
                    medians = []
                    for cell in cells:
                        lookup = {(r["model"], str(r["seed"])): r for r in rows if r["cell"] == cell}
                        values = []
                        for seed in C.SEEDS:
                            ref_seed = str(seed) if reference == "physics_gradient" else "deterministic"
                            value = float(lookup[(variant, str(seed))][f"mae_{mask}_mV"]) - float(
                                lookup[(reference, ref_seed)][f"mae_{mask}_mV"])
                            values.append(value)
                            differences.append({"cohort": cohort, "mask": mask, "cell": cell,
                                                "variant": variant, "reference": reference,
                                                "seed": seed, "difference_mV": value})
                        medians.append(float(np.median(values)))
                    result = E._inferential_row(cohort="G2", configuration=cohort, variant=variant,
                                               reference=reference, condition=mask, differences=np.asarray(medians))
                    result["mask"] = mask
                    result["cell_cohort"] = cohort
                    result["analysis_status"] = "exploratory"
                    tests.append(result)
    E._adjust_test_rows(tests, ("mask", "cell_cohort", "reference"))
    return tests, differences


def median_by_cell(rows: list[dict], column: str) -> float:
    cells = sorted({row["cell"] for row in rows})
    return float(np.median([np.median([float(row[column]) for row in rows if row["cell"] == cell])
                            for cell in cells]))


def report(path: Path, aging: list[dict], hppc: list[dict], tests: list[dict]):
    lines = [
        "# Exploratory corrections to Part F",
        "",
        "All G1–G3 analyses are exploratory; no model was retrained. "
        "Amendment 010 was registered before the recalculations. The original C and F files were preserved.",
        "",
        "## G1 — Aging on the same basis",
        "",
        "Source MAE uses all points of complete C target episodes, with RC states reset only at the start "
        "of each episode and aggregation weighted by sample count. The same rule was used for later "
        "F1 blocks. Differences are later − source, paired by seed; the table shows "
        "the median across seeds per cell, followed by the median across cells.",
        "",
        "Without a later C/20 available, the updated-capacity arm is identical to fixed calibration.",
        "",
        "| Cell | ECM-2RC (mV) | Physics-gradient (mV) | Output residual (mV) | ECM-UDE (mV) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in AGING_CELLS:
        values = []
        for model in MODELS:
            selected = [r for r in aging if r["cell"] == cell and r["model"] == model]
            values.append(float(np.median([r["increase_later_minus_source_mV"] for r in selected])))
        lines.append(f"| {cell} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    values = [median_by_cell([r for r in aging if r["model"] == model],
                             "increase_later_minus_source_mV") for model in MODELS]
    lines += ["| **Median (n=6)** | " + " | ".join(f"**{value:.3f}**" for value in values) + " |",
              "", "| Model | Median source MAE (mV) | Median later MAE (mV) | Median paired increase (mV) |",
              "|---|---:|---:|---:|"]
    for model in MODELS:
        selected = [r for r in aging if r["model"] == model]
        vals = [median_by_cell(selected, name) for name in
                ("mae_source_full_mV", "mae_later_full_mV", "increase_later_minus_source_mV")]
        lines.append(f"| {model} | " + " | ".join(f"{value:.3f}" for value in vals) + " |")
    lines += [
        "",
        "Position 1 has six cells; V5 has no later block and available position 2 has no "
        "eligible episode. `g1_source_metrics.csv` and `g1_aging_comparison.csv` contain metrics and "
        "differences by seed, with sample counts. Model contrasts in the later block "
        "did not change; the correction changes only the comparison with the source.",
        "",
        "## G2 — HPPC within the identification range",
        "",
        "The continuous simulation starts at rest after full charge at SOC=1, with initial RC states zero; "
        "the mask is applied only afterward. To ensure paired comparisons on the same samples, "
        "the mask uses nominal coulomb counting from F and the intersection of the cell's "
        "`soc_train_min`–`soc_train_max` ranges for 1RC and 2RC; the stored ranges are equal "
        "in this dataset. Neural models integrate their own SOC with learned η, which can differ "
        "slightly from this common reference. Pulses and rests use the F definitions.",
        "",
        "W3 is retained in the seven-cell analysis; HPPC precedes its C/20 and uses another channel, so "
        "association with the same diagnostic could not be confirmed. The analysis without W3 is a separate "
        "sensitivity analysis with Holm recalculated.",
        "",
        "| Cell | Minimum SOC | Maximum SOC | n in range | Fraction in range | n pulses | n rests |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in C.CELLS:
        row = next(r for r in hppc if r["cell"] == cell and r["model"] == "ecm_2rc_nls")
        lines.append(f"| {cell} | {row['soc_min']:.5f} | {row['soc_max']:.5f} | {row['n_in_range']} | "
                     f"{100 * row['fraction_in_range_of_full']:.2f}% | {row['n_pulse_in_range']} | "
                     f"{row['n_rest_in_range']} |")
    lines += ["", "| Model | Total in range (mV) | Pulses in range (mV) | Rests in range (mV) |",
              "|---|---:|---:|---:|"]
    for model in MODELS:
        selected = [r for r in hppc if r["model"] == model]
        values = [median_by_cell(selected, f"mae_{mask}_mV") for mask in MASKS]
        lines.append(f"| {model} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += [
        "",
        "Cells are the inferential unit; for each cell, the median of the five differences "
        "paired by seed is used. The tests below are exploratory: exact Wilcoxon, TOST ±1 mV, and Holm "
        "over the two variants separately for each mask, reference, and cohort.",
        "",
        "| Cohort | Mask | Reference | Variant | n | Median Δ (mV) | Raw TOST p | Holm p | Raw / Holm superiority p | Raw / Holm inferiority p |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests:
        lines.append(f"| {row['cell_cohort']} | {row['mask']} | {row['reference']} | {row['variant']} | "
                     f"{row['n']} | {row['median_difference_mV']:.3f} | {row['p_tost_raw']:.6g} | "
                     f"{row['p_tost_holm']:.6g} | {row['p_superiority_raw']:.6g} / {row['p_superiority_holm']:.6g} | "
                     f"{row['p_inferiority_raw']:.6g} / {row['p_inferiority_holm']:.6g} |")
    lines += [
        "",
        "### Full F HPPC: not interpretable because of OCV extrapolation",
        "",
        "The original full-profile numbers are retained here for traceability, not as an interpretable "
        "measure of the neural contribution: the fitted OCV is extrapolated outside the identified range.",
        "",
        "| Model | Full total (mV) | SOC 0,2–0,8 (mV) | Outside (mV) | Full pulses (mV) | Full rests (mV) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    old = read_csv(F_DIR / "hppc_metrics.csv")
    for model in MODELS:
        selected = [r for r in old if r["model"] == model]
        vals = [median_by_cell(selected, f"mae_{mask}_mV") for mask in
                ("full", "soc_0.2_to_0.8", "soc_outside_0.2_to_0.8", "pulse", "rest")]
        lines.append(f"| {model} | " + " | ".join(f"{value:.3f}" for value in vals) + " |")
    lines += [
        "",
        "## G3 — Synthesis and statistical resolution",
        "",
        "The synthesis retains the preregistered F1 model contrast, but the corrected G1 increase "
        "and G2 tests are exploratory; the HPPC row now uses the identification range. "
        "For A4a/A4b, five seeds give a minimum one-sided p of 1/32; Holm over 18 conditions gives "
        "a minimum adjusted p of 18/32=0,5625. Counts of 0/0/0 do not demonstrate absence of an effect.",
        "",
        "Detailed files: `g2_hppc_metrics.csv`, `g2_cell_differences.csv`, `g2_tests.csv`, "
        "and `manifest.json`; previous versions of the synthesis are in `run/`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def synthesize(tests: list[dict], differences: list[dict], out: Path):
    csv_path, tex_path = E_DIR / "synthesis.csv", E_DIR / "synthesis.tex"
    shutil.copy2(csv_path, out / "synthesis_before_G.csv")
    shutil.copy2(tex_path, out / "synthesis_before_G.tex")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        existing = list(reader)
    if fields is None:
        raise ValueError("Synthesis CSV header missing")
    keep = []
    for row in existing:
        name = row["configuration"]
        if name.startswith("M50T F1 bloco 1 updated_capacity") or name.startswith("M50T F2 HPPC"):
            continue
        if name.startswith("M50T F1 bloco 1 fixed"):
            row["configuration"] = "M50T G1 F1 fixed (n=6)"
            row["protocol_status"] = "preregistered F1 model contrasts; exploratory G1 aging correction"
        keep.append(row)
    if sum(r["configuration"] == "M50T G1 F1 fixed (n=6)" for r in keep) != 2:
        raise ValueError("Expected exactly two existing fixed-arm F1 synthesis rows")
    added, latex = [], []
    for cohort, count in (("all", 7), ("without_W3", 6)):
        label = f"M50T G2 HPPC {'all' if cohort == 'all' else 'without W3'} (n={count})"
        for variant in C.NEURAL_VARIANTS:
            row = {field: "" for field in fields}
            row.update(configuration=label, protocol_status="exploratory G2",
                       variant=variant, analysis_unit_n=count)
            for prefix, ref in (("vs_pg", "physics_gradient"), ("vs_ecm2", "ecm_2rc_nls")):
                found = [r for r in tests if r["cell_cohort"] == cohort and r["mask"] == "in_range"
                         and r["reference"] == ref and r["variant"] == variant]
                if len(found) != 1:
                    raise ValueError(f"Missing G2 synthesis test: {cohort}/{variant}/{ref}")
                test = found[0]
                row[f"{prefix}_status"] = "exploratory TOST and one-sided tests"
                row[f"{prefix}_n_tests"] = 1
                for kind, field in (("better", "p_superiority_holm"), ("equivalent", "p_tost_holm"),
                                    ("worse", "p_inferiority_holm")):
                    row[f"{prefix}_{kind}_holm_count"] = int(test[field] < 0.05)
                row[f"{prefix}_unresolved_holm_count"] = int(all(
                    test[field] >= 0.05 for field in ("p_superiority_holm", "p_tost_holm", "p_inferiority_holm")))
                for key in ("median_difference_mV", "minimum_difference_mV", "maximum_difference_mV",
                            "p_superiority_holm", "p_tost_holm", "p_inferiority_holm"):
                    row[f"{prefix}_{key}"] = test[key]
                selected = [r for r in differences if r["cohort"] == cohort and r["mask"] == "in_range"
                            and r["variant"] == variant and r["reference"] == ref]
                medians = [np.median([r["difference_mV"] for r in selected if r["cell"] == cell])
                           for cell in sorted({r["cell"] for r in selected})]
                row[f"{prefix}_cells_d_below_0"] = sum(x < 0 for x in medians)
                row[f"{prefix}_cells_d_below_minus1"] = sum(x < -1 for x in medians)
                row[f"{prefix}_cells_d_above_0"] = sum(x > 0 for x in medians)
                row[f"{prefix}_cells_within_pm1"] = sum(-1 <= x <= 1 for x in medians)
            added.append(row)

            def value(prefix):
                counts = "/".join(str(row[f"{prefix}_{k}_holm_count"]) for k in ("better", "equivalent", "worse"))
                return (f"{counts}; {row[f'{prefix}_median_difference_mV']:.3f} "
                        f"[{row[f'{prefix}_minimum_difference_mV']:.3f},{row[f'{prefix}_maximum_difference_mV']:.3f}]")

            short = f"M50T G2 {'all' if cohort == 'all' else 'no W3'} (n={count})"
            tex_variant = variant.replace("_", r"\_")
            latex.append(f"{short} & {tex_variant} & EX/EX & {value('vs_pg')} & {value('vs_ecm2')} \\\\")
    text = tex_path.read_text()
    lines = [line for line in text.splitlines()
             if not line.startswith("M50T F1 1 updated") and not line.startswith("M50T F2 HPPC")]
    text = "\n".join(lines) + "\n"
    text = text.replace("M50T F1 1 fixed", "M50T G1 1 fixed")
    text = text.replace(r"\bottomrule", "\n".join(latex) + "\n" + r"\bottomrule", 1)
    old_note = "The F1 number is the later-block position; updated denotes updated capacity."
    new_note = ("G1 contrasts retain preregistered F1 tests; the corrected aging change and all G2 tests are exploratory. "
                "For A4a/A4b, five seeds give minimum one-sided $p=1/32$; Holm over 18 gives "
                "minimum adjusted $p=18/32>0.05$, so 0/0/0 does not establish no effect.")
    if old_note not in text:
        raise ValueError("Expected existing F synthesis note missing")
    text = text.replace(old_note, new_note)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(keep + added)
    tex_path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--attempt", default=None)
    args = parser.parse_args()
    if args.attempt is not None and not args.attempt.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid attempt name")
    protocol = ROOT / "protocols" / "reviewer_round2_protocol.json"
    if C._sha256(protocol) != "691c9ba5df415c1b3ef0c83681294d654ab876eac9e64f3e6c9284c197d9a757":
        raise ValueError("Protocol differs from the registered amendment")
    selection, cm, fm = verified_inputs()
    for suffix in ("csv", "tex"):
        if C._sha256(E_DIR / f"synthesis.{suffix}") != fm["synthesis_after_sha256"][suffix]:
            raise ValueError(f"Synthesis {suffix} changed since F; refusing to overwrite")
    out = OUT / (args.attempt or ("smoke" if args.smoke else "run"))
    out.mkdir(parents=True, exist_ok=False)
    prior_hppc = read_csv(F_DIR / "hppc_metrics.csv")
    prior_lookup = {(row["cell"], row["model"], row["seed"]): row for row in prior_hppc}
    aging, hppc, provenance = [], [], {}
    cells = ("G1",) if args.smoke else C.CELLS
    seeds = (42,) if args.smoke else C.SEEDS
    for cell in cells:
        source, h, details = evaluate_cell(cell, selection["cells"][cell], cm, prior_lookup,
                                           seeds=seeds, source_targets=cell in AGING_CELLS, smoke=args.smoke)
        aging.extend(source)
        hppc.extend(h)
        provenance[cell] = details
        (out / f"{cell}_evaluations.json").write_text(json.dumps(
            {"source": source, "hppc": h, "provenance": details}, indent=2) + "\n")
    C._write_csv(out / "g1_source_metrics.csv", aging)
    C._write_csv(out / "g2_hppc_metrics.csv", hppc)
    if args.smoke:
        return
    if len(aging) != 96 or len(hppc) != 112:
        raise ValueError(f"Unexpected result shapes: {len(aging)} source, {len(hppc)} HPPC")
    comparison = aging_comparison(aging, read_csv(F_DIR / "aging_metrics.csv"))
    if len(comparison) != 96:
        raise ValueError(f"Expected 96 fixed-arm aging comparisons, got {len(comparison)}")
    tests, differences = hppc_inference(hppc)
    if len(tests) != 24:
        raise ValueError(f"Expected 24 inferential contrasts, got {len(tests)}")
    C._write_csv(out / "g1_aging_comparison.csv", comparison)
    C._write_csv(out / "g2_tests.csv", tests)
    C._write_csv(out / "g2_cell_differences.csv", differences)
    report(out / "report.md", comparison, hppc, tests)
    for suffix in ("csv", "tex"):
        if C._sha256(E_DIR / f"synthesis.{suffix}") != fm["synthesis_after_sha256"][suffix]:
            raise ValueError(f"Synthesis {suffix} changed during evaluation; refusing to overwrite")
    synthesize(tests, differences, out)
    output_names = ("g1_source_metrics.csv", "g1_aging_comparison.csv",
                    "g2_hppc_metrics.csv", "g2_tests.csv", "g2_cell_differences.csv", "report.md")
    for name in output_names:
        destination = OUT / name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.copy2(out / name, destination)
    manifest = {
        "protocol_commit": "1aeaf76", "analysis_status": "exploratory", "training_performed": False,
        "input_sha256": {
            "protocols/reviewer_round2_protocol.json": C._sha256(protocol),
            "results/reviewer_round2/C/c0_segments.json": C._sha256(C_DIR / "c0_segments.json"),
            "results/reviewer_round2/C/run/manifest.json": C._sha256(C_DIR / "run/manifest.json"),
            "results/reviewer_round2/F/manifest.json": C._sha256(F_DIR / "manifest.json"),
            **{f"results/reviewer_round2/F/{name}": C._sha256(F_DIR / name) for name in
               ("aging_metrics.csv", "hppc_metrics.csv", "c2_style_tost.csv", "report.md")},
            **{name: C._sha256(ROOT / name) for name in
               ("reviewer_round2_g.py", "reviewer_round2_f.py", "reviewer_round2_c.py",
                "reviewer_round2_e.py", "exp_unified.py", "unified_dynamics.py")},
        },
        "checkpoint_sha256_verified": cm["checkpoint_sha256"],
        "nls_sha256_verified": cm["nls_fit_sha256"],
        "per_cell_inputs": provenance,
        "outputs": {name: C._sha256(OUT / name) for name in output_names},
        "synthesis_before_sha256": {suffix: C._sha256(out / f"synthesis_before_G.{suffix}")
                                     for suffix in ("csv", "tex")},
        "synthesis_after_sha256": {suffix: C._sha256(E_DIR / f"synthesis.{suffix}")
                                    for suffix in ("csv", "tex")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
