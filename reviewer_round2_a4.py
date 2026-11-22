"""Fit the registered frozen-physics sensitivity experiments A4a and A4b."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import exp_unified as U
import protocol as P
from exp_paper_v2_seeds import shifted_wilcoxon
from reviewer_round2_a1 import FrozenCorrection, _hash_tensor, _nls_params
from reviewer_round2_a2 import MAIN_CONDITIONS, _contexts
from training import TrainConfig
from utils import set_seed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "A"
VARIANTS = ("output_residual", "ecm_ude")
SEEDS = (42, 43, 44, 45, 46)
MARGIN_MV = 1.0


def _nls_training_mse(context: dict, params, cfg) -> float:
    starts = context["starts"][context["tr"]]
    predicted = U.nls_predict(params, context["cycle"], cfg, starts)
    length = cfg.window_length
    truth = np.stack(
        [context["ds"]._V_raw[start : start + length].numpy() for start in starts]
    )
    stats = context["stats"]
    predicted_norm = (predicted - stats["V_mean"]) / stats["V_std"]
    truth_norm = (truth - stats["V_mean"]) / stats["V_std"]
    return float(np.mean((predicted_norm - truth_norm) ** 2))


def _run_one(dataset: str, variant: str, seed: int, contexts: dict, cfg, params,
             output: Path, label: str, *, smoke: bool) -> tuple[list[dict], list[dict], dict]:
    source = "UDDS_25C" if dataset == "panasonic" else "FUDS_25C"
    set_seed(seed)
    model = FrozenCorrection(
        U.build_model(variant, contexts[source]["stats"], params, cfg), cfg
    )
    with torch.no_grad():
        model.correction[-1] = 1.0
    gain_initial = float(model.correction[-1])
    physical_before = model.physical.detach().clone()
    physical_hash = _hash_tensor(physical_before)
    if label == "A4a":
        lr, epochs, patience = P.TRAIN["lr"], P.TRAIN["n_epochs"], P.TRAIN["patience"]
    else:
        lr, epochs, patience = 2e-3, 150, 30
    if smoke:
        epochs, patience = 2, 2
    checkpoint = output / "checkpoints" / dataset / f"{variant}_seed{seed}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite A4 checkpoint: {checkpoint}")
    train_cfg = replace(
        TrainConfig(**P.TRAIN, ckpt_path=str(checkpoint)),
        lr=lr,
        n_epochs=epochs,
        patience=patience,
    )
    train_context = contexts[source]
    history = asdict(
        U.train(
            model,
            U.loader(train_context, cfg, train_context["tr"], seed, True),
            U.loader(train_context, cfg, train_context["eval_indices"]),
            train_cfg,
        )
    )
    if not torch.equal(model.physical.detach(), physical_before):
        raise AssertionError(f"Frozen physics changed in {label}/{dataset}/{variant}/{seed}")
    if model.physical.grad is not None:
        raise AssertionError(f"Frozen physics received a gradient in {label}/{dataset}/{variant}/{seed}")

    nls_train_mse = _nls_training_mse(train_context, params, cfg)
    final_train_mse = float(history["train_loss"][-1])
    loss_row = {
        "experiment": label,
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "output_gain_initial": gain_initial,
        "nls_training_mse_normalized": nls_train_mse,
        "final_training_mse_normalized": final_train_mse,
        "relative_training_loss_reduction": (
            nls_train_mse - final_train_mse
        ) / nls_train_mse,
    }
    curve_rows = [
        {
            "experiment": label,
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": history["train_loss"][epoch],
            "validation_loss": history["val_loss"][epoch],
        }
        for epoch in range(len(history["train_loss"]))
    ]
    metric_rows = []
    conditions = [source] if smoke else MAIN_CONDITIONS[dataset]
    for condition in conditions:
        continuous = U.evaluate(model, variant, contexts[condition], cfg)["continuous"]
        metric_rows.append(
            {
                "experiment": label,
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "condition": condition,
                "mae_mV": continuous["mae"],
                "rmse_mV": continuous["rmse"],
                "max_error_mV": continuous["max_error"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": P.sha256(checkpoint),
            }
        )
    audit = {
        **loss_row,
        "physical_sha256_before": physical_hash,
        "physical_sha256_after": _hash_tensor(model.physical),
        "frozen_parameters_unchanged": torch.equal(model.physical.detach(), physical_before),
        "frozen_gradient_is_null": model.physical.grad is None,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "checkpoint_sha256": P.sha256(checkpoint),
        "history": history,
    }
    return metric_rows, curve_rows, {"loss": loss_row, "audit": audit}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No records to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_statistics(metrics_rows: list[dict], saved_summary: dict,
                      output: Path, label: str) -> None:
    rows = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in metrics_rows:
        grouped.setdefault(
            (row["dataset"], row["variant"], row["condition"]), []
        ).append(row)
    for (dataset, variant, condition), observations in grouped.items():
        observations.sort(key=lambda row: row["seed"])
        if [int(row["seed"]) for row in observations] != list(SEEDS):
            raise ValueError(f"Incomplete A4 seed family for {dataset}/{condition}/{label}")
        baseline = saved_summary["datasets"][dataset]["runs"]["ecm_1rc_nls"][
            "deterministic"
        ]["metrics"][condition]["continuous"]["mae"]
        delta = np.asarray(
            [float(row["mae_mV"]) - baseline for row in observations], dtype=float
        )
        lower = shifted_wilcoxon(delta + MARGIN_MV, "greater")
        upper = shifted_wilcoxon(delta - MARGIN_MV, "less")
        rows.append(
            {
                "experiment": label,
                "dataset": dataset,
                "variant": variant,
                "condition": condition,
                "median_mae_difference_mV": float(np.median(delta)),
                "n": len(observations),
                "p_lower_raw": lower["p_raw"],
                "p_upper_raw": upper["p_raw"],
            }
        )
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if len(variant_rows) != 18:
            raise ValueError(f"Expected 18 A4 conditions for {label}/{variant}")
        for tail in ("lower", "upper"):
            family = [{"p_raw": row[f"p_{tail}_raw"], "_row": row}
                      for row in variant_rows]
            U.holm(family)
            for item in family:
                item["_row"][f"p_{tail}_holm"] = item["p_holm"]
    for row in rows:
        row["p_tost_raw"] = max(row["p_lower_raw"], row["p_upper_raw"])
        row["p_tost_holm"] = max(row["p_lower_holm"], row["p_upper_holm"])
    _write_csv(output / "tost_vs_nls.csv", rows)


def run(label: str, *, smoke: bool = False) -> None:
    if label not in ("A4a", "A4b"):
        raise ValueError("A4 run label must be A4a or A4b")
    summary = json.loads((ROOT / "results" / "unified_10seeds" / "summary.json").read_text())
    output = OUT / (f"{label.lower()}_smoke" if smoke else label.lower())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite A4 output directory: {output}")
    output.mkdir(parents=True)
    all_metrics, all_curves, all_audits, all_losses = [], [], [], []
    for dataset, cfg in P.DATASETS.items():
        saved = summary["datasets"][dataset]
        contexts = _contexts(dataset, cfg, saved)
        params, _, _ = _nls_params(dataset, saved)
        for variant in VARIANTS:
            for seed in ((42,) if smoke else SEEDS):
                metrics, curves, records = _run_one(
                    dataset, variant, seed, contexts, cfg, params, output, label, smoke=smoke
                )
                all_metrics.extend(metrics)
                all_curves.extend(curves)
                all_losses.append(records["loss"])
                all_audits.append(records["audit"])
    _write_csv(output / "metrics_by_seed.csv", all_metrics)
    _write_csv(output / "learning_curves.csv", all_curves)
    _write_csv(output / "training_loss_reduction.csv", all_losses)
    (output / "freeze_audit.json").write_text(json.dumps(all_audits, indent=2) + "\n")
    if not smoke:
        _write_statistics(all_metrics, summary, output, label)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("A4a", "A4b"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.label, smoke=args.smoke)
