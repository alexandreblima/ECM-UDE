"""Fit the frozen-physics neural-correction ablation (A1)."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

import exp_unified as U
import protocol as P
from ecm import ECMParams
from exp_paper_v2_seeds import shifted_wilcoxon
from reviewer_round2_a2 import MAIN_CONDITIONS, SEEDS, _contexts
from training import TrainConfig
from unified_dynamics import RK4Function
from utils import set_seed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "A"
VARIANTS = ("output_residual", "ecm_ude")
MARGIN_MV = 1.0
METRICS = ("mae",)


class FrozenCorrection(nn.Module):
    def __init__(self, base: U.UnifiedECMUDE, cfg) -> None:
        super().__init__()
        self.physical = nn.Parameter(base.p[:10].detach().clone(), requires_grad=False)
        self.correction = nn.Parameter(base.p[10:].detach().clone())
        self.stats = base.stats
        self.dt = base.dt
        self.q = base.q
        self.mode = base.mode
        self.nominal_capacity_Ah = cfg.nominal_capacity_Ah

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = torch.cat((self.physical, self.correction))
        return RK4Function.apply(
            parameters, x, self.stats, self.dt, self.q, self.mode
        )


def _nls_params(dataset: str, saved_dataset: dict) -> tuple[ECMParams, Path, str]:
    run = saved_dataset["runs"]["ecm_1rc_nls"]["deterministic"]
    expected_hash = run["sha256"]
    candidates = [
        ROOT / "results" / result_set / "checkpoints" / dataset / "ecm_1rc_nls.json"
        for result_set in ("unified", "unified_10seeds")
    ]
    for path in candidates:
        if path.is_file() and P.sha256(path) == expected_hash:
            info = json.loads(path.read_text())
            return ECMParams.from_dict(info["params"]), path, expected_hash
    raise FileNotFoundError(f"No NLS warm start matches the saved SHA256 for {dataset}")


def _hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _train_one(dataset: str, variant: str, seed: int, contexts: dict, cfg,
               params: ECMParams, output: Path, *, smoke: bool) -> tuple[list[dict], list[dict], dict]:
    source = "UDDS_25C" if dataset == "panasonic" else "FUDS_25C"
    set_seed(seed)
    base = U.build_model(variant, contexts[source]["stats"], params, cfg)
    model = FrozenCorrection(base, cfg)
    physical_before = model.physical.detach().clone()
    before_hash = _hash_tensor(physical_before)
    ckpt = output / "checkpoints" / dataset / f"{variant}_seed{seed}.pt"
    if ckpt.exists():
        raise FileExistsError(f"Refusing to overwrite A1 checkpoint: {ckpt}")
    cfg_train = replace(
        TrainConfig(**P.TRAIN, ckpt_path=str(ckpt)),
        n_epochs=2 if smoke else P.TRAIN["n_epochs"],
        patience=2 if smoke else P.TRAIN["patience"],
    )
    train_ctx = contexts[source]
    history = asdict(
        U.train(
            model,
            U.loader(train_ctx, cfg, train_ctx["tr"], seed, True),
            U.loader(train_ctx, cfg, train_ctx["eval_indices"]),
            cfg_train,
        )
    )
    if not torch.equal(model.physical.detach(), physical_before):
        raise AssertionError(f"Frozen physical vector changed for {dataset}/{variant}/{seed}")
    if model.physical.grad is not None:
        raise AssertionError(f"Frozen physical vector received a gradient for {dataset}/{variant}/{seed}")
    if _hash_tensor(model.physical) != before_hash:
        raise AssertionError(f"Frozen physical SHA256 changed for {dataset}/{variant}/{seed}")
    metrics_rows = []
    evaluation_conditions = [source] if smoke else list(MAIN_CONDITIONS[dataset])
    for condition in evaluation_conditions:
        metrics = U.evaluate(model, variant, contexts[condition], cfg)
        continuous = metrics["continuous"]
        metrics_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "condition": condition,
                "mae_mV": continuous["mae"],
                "rmse_mV": continuous["rmse"],
                "max_error_mV": continuous["max_error"],
                "checkpoint_sha256": P.sha256(ckpt),
            }
        )
    curve_rows = [
        {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": history["train_loss"][epoch],
            "validation_loss": history["val_loss"][epoch],
        }
        for epoch in range(len(history["train_loss"]))
    ]
    audit = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "frozen_gradient_is_null": model.physical.grad is None,
        "frozen_parameters_unchanged": torch.equal(model.physical.detach(), physical_before),
        "physical_sha256_before": before_hash,
        "physical_sha256_after": _hash_tensor(model.physical),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "physical_parameter_count": int(model.physical.numel()),
        "best_epoch": history["best_epoch"],
        "epochs_completed": len(history["train_loss"]),
        "history": history,
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "checkpoint_sha256": P.sha256(ckpt),
    }
    return metrics_rows, curve_rows, audit


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _statistics(metrics_rows: list[dict], saved_summary: dict) -> None:
    all_tests_nls = []
    all_tests_joint = []
    for variant in VARIANTS:
        variant_rows = [row for row in metrics_rows if row["variant"] == variant]
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in variant_rows:
            grouped.setdefault((row["dataset"], row["condition"]), []).append(row)
        tost_rows = []
        joint_rows = []
        for (dataset, condition), observations in grouped.items():
            observations.sort(key=lambda row: row["seed"])
            if [row["seed"] for row in observations] != list(SEEDS):
                raise ValueError(f"Incomplete A1 seed family for {dataset}/{condition}/{variant}")
            nls = saved_summary["datasets"][dataset]["runs"]["ecm_1rc_nls"]["deterministic"][
                "metrics"
            ][condition]["continuous"]
            joint = saved_summary["datasets"][dataset]["runs"][variant]
            row_tost = {
                "dataset": dataset,
                "variant": variant,
                "condition": condition,
                "n": len(observations),
            }
            row_joint = dict(row_tost)
            for metric in METRICS:
                field = f"{metric}_mV"
                a1_values = np.asarray([row[field] for row in observations], dtype=float)
                nls_delta = a1_values - nls[metric]
                lower = shifted_wilcoxon(nls_delta + MARGIN_MV, "greater")
                upper = shifted_wilcoxon(nls_delta - MARGIN_MV, "less")
                row_tost[f"{metric}_median_difference_mV"] = float(np.median(nls_delta))
                row_tost[f"{metric}_p_lower_raw"] = lower["p_raw"]
                row_tost[f"{metric}_p_upper_raw"] = upper["p_raw"]
                row_joint[f"{metric}_median_difference_mV"] = float(
                    np.median(
                        [
                            value - joint[str(seed)]["metrics"][condition]["continuous"][metric]
                            for seed, value in zip(SEEDS, a1_values)
                        ]
                    )
                )
                paired = U.paired(
                    a1_values,
                    np.asarray(
                        [
                            joint[str(seed)]["metrics"][condition]["continuous"][metric]
                            for seed in SEEDS
                        ],
                        dtype=float,
                    ),
                )
                row_joint[f"{metric}_p_raw"] = paired["p_raw"]
                row_joint[f"{metric}_n"] = len(SEEDS)
            tost_rows.append(row_tost)
            joint_rows.append(row_joint)
        for metric in METRICS:
            for tail in ("lower", "upper"):
                family = [{"p_raw": row[f"{metric}_p_{tail}_raw"], "_row": row}
                          for row in tost_rows]
                U.holm(family)
                for item in family:
                    item["_row"][f"{metric}_p_{tail}_holm"] = item["p_holm"]
            for row in tost_rows:
                row[f"{metric}_p_tost_raw"] = max(
                    row[f"{metric}_p_lower_raw"], row[f"{metric}_p_upper_raw"]
                )
                row[f"{metric}_p_tost_holm"] = max(
                    row[f"{metric}_p_lower_holm"], row[f"{metric}_p_upper_holm"]
                )
            family = [{"p_raw": row[f"{metric}_p_raw"], "_row": row}
                      for row in joint_rows]
            U.holm(family)
            for item in family:
                item["_row"][f"{metric}_p_holm"] = item["p_holm"]
        all_tests_nls.extend(tost_rows)
        all_tests_joint.extend(joint_rows)
    _write_csv(OUT / "a1_nls_tost.csv", all_tests_nls)
    _write_csv(OUT / "a1_joint_paired.csv", all_tests_joint)


def run(*, smoke: bool = False) -> None:
    torch.set_num_threads(1)
    summary = json.loads((ROOT / "results" / "unified_10seeds" / "summary.json").read_text())
    output = OUT / ("a1_smoke" if smoke else "a1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite A1 output directory: {output}")
    output.mkdir(parents=True)
    metrics_rows, curve_rows, audit_rows = [], [], []
    for dataset, cfg in P.DATASETS.items():
        saved_dataset = summary["datasets"][dataset]
        contexts = _contexts(dataset, cfg, saved_dataset)
        params, nls_path, nls_hash = _nls_params(dataset, saved_dataset)
        seeds = (42,) if smoke else SEEDS
        for variant in VARIANTS:
            for seed in seeds:
                metrics, curves, audit = _train_one(
                    dataset, variant, seed, contexts, cfg, params, output, smoke=smoke
                )
                metrics_rows.extend(metrics)
                curve_rows.extend(curves)
                audit_rows.append(audit)
                audit_rows[-1]["nls_checkpoint_sha256"] = nls_hash
    _write_csv(output / "metrics_by_seed.csv", metrics_rows)
    _write_csv(output / "learning_curves.csv", curve_rows)
    (output / "freeze_audit.json").write_text(json.dumps(audit_rows, indent=2) + "\n")
    if not smoke:
        _statistics(metrics_rows, summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
