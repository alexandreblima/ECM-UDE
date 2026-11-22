"""Run the no-early-stopping A3 optimization diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import exp_unified as U
import protocol as P
from reviewer_round2_a1 import _nls_params
from reviewer_round2_a2 import _contexts
from training import TrainConfig, _build_lr_lambda, _run_epoch
from utils import set_seed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "A"
VARIANTS = ("output_residual", "ecm_ude")
SEED = 42


def _mae_mV(model, loader, voltage_std: float) -> float:
    predicted, target = U.predict(model, loader, device="cpu")
    return float((predicted - target).abs().mean().item() * voltage_std * 1000.0)


def _run_one(dataset: str, variant: str, contexts: dict, cfg, params,
             output: Path, *, smoke: bool) -> tuple[list[dict], dict]:
    source = "UDDS_25C" if dataset == "panasonic" else "FUDS_25C"
    seed = SEED
    set_seed(seed)
    model = U.build_model(variant, contexts[source]["stats"], params, cfg)
    epochs = 2 if smoke else P.TRAIN["n_epochs"]
    checkpoint = output / "checkpoints" / dataset / f"{variant}_seed{seed}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite A3 checkpoint: {checkpoint}")
    train_config = TrainConfig(
        **{
            **P.TRAIN,
            "n_epochs": epochs,
            "patience": epochs + 1,
            "ckpt_path": None,
        }
    )
    train_context = contexts[source]
    train_loader = U.loader(
        train_context, cfg, train_context["tr"], seed=seed, shuffle=True
    )
    train_eval_loader = U.loader(
        train_context, cfg, train_context["tr"], seed=seed, shuffle=False
    )
    validation_loader = U.loader(
        train_context, cfg, train_context["eval_indices"], seed=seed, shuffle=False
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_build_lr_lambda(train_config)
    )
    loss_fn = nn.MSELoss()
    rows = []
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(
            model, train_loader, loss_fn, torch.device("cpu"),
            optimizer=optimizer, grad_clip=train_config.grad_clip
        )
        val_loss, _, _ = _run_epoch(
            model, validation_loader, loss_fn, torch.device("cpu")
        )
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_mae_mV": _mae_mV(model, train_eval_loader, contexts[source]["stats"]["V_std"]),
                "validation_mae_mV": _mae_mV(model, validation_loader, contexts[source]["stats"]["V_std"]),
                "train_mse_normalized": train_loss,
                "validation_mse_normalized": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    audit = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "epochs_completed": epochs,
        "early_stopping": False,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    return rows, audit


def run(*, smoke: bool = False) -> None:
    torch.set_num_threads(1)
    summary = json.loads((ROOT / "results" / "unified_10seeds" / "summary.json").read_text())
    output = OUT / ("a3_smoke" if smoke else "a3")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite A3 output directory: {output}")
    output.mkdir(parents=True)
    rows, audits = [], []
    for dataset, cfg in P.DATASETS.items():
        saved = summary["datasets"][dataset]
        contexts = _contexts(dataset, cfg, saved)
        params, _, _ = _nls_params(dataset, saved)
        for variant in VARIANTS:
            run_rows, audit = _run_one(
                dataset, variant, contexts, cfg, params, output, smoke=smoke
            )
            rows.extend(run_rows)
            audits.append(audit)
    with (output / "training_validation_mae.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "checkpoint_hashes.json").write_text(json.dumps(audits, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
