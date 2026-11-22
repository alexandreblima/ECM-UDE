"""Train the registered LG HG2 NMC replication and its A1/A2 ablations."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

import exp_unified as U
import protocol as P
from ecm import ECMParams
from ecm_2rc import ECMParams2RC
from exp_paper_v2_seeds import shifted_wilcoxon
from reviewer_round2_a1 import FrozenCorrection, _hash_tensor
from reviewer_round2_b_audit import (
    DATA_ROOT,
    DT_SECONDS,
    PROFILES,
    Q_NOM_AH,
    _read_mat,
    _sanitize,
)
from training import TrainConfig
from utils import set_seed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reviewer_round2" / "B"
DATASET = "lg_hg2"
SOURCE = "UDDS_25C_source"
TARGETS = (
    "Mixed1_25C",
    "LA92_25C",
    "US06_25C",
    "UDDS_-20C",
    "UDDS_-10C",
    "UDDS_0C",
    "UDDS_10C",
    "UDDS_40C",
)
CONDITIONS = (SOURCE, *TARGETS)
VARIANTS = ("physics_gradient", "output_residual", "ecm_ude")
NEURAL_VARIANTS = ("output_residual", "ecm_ude")
SEEDS = tuple(range(42, 52))
MARGIN_MV = 1.0
CFG = P.DatasetProtocol(
    nominal_capacity_Ah=Q_NOM_AH,
    default_z0=1.0,
    dt_seconds=DT_SECONDS,
    window_length=1024,
    window_stride=512,
    batch_size=32,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No records to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _profile_path(folder: str, suffix: str) -> Path:
    matches = sorted((DATA_ROOT / folder).glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one HG2 MAT ending in {suffix}, found {len(matches)}"
        )
    return matches[0]


def _load_contexts() -> tuple[dict, dict, dict]:
    audit = json.loads((OUT / "manifest.json").read_text())
    if not audit.get("profile_input_sha256"):
        raise ValueError("Part B data audit is missing profile hashes")
    cycles, input_hashes = {}, {}
    for condition, (folder, suffix, charge_id) in PROFILES.items():
        path = _profile_path(folder, suffix)
        expected = audit["profile_input_sha256"][str(path.relative_to(DATA_ROOT))]
        if P.sha256(path) != expected:
            raise RuntimeError(f"HG2 profile SHA256 changed after audit: {path}")
        arrays, _ = _sanitize(_read_mat(path))
        time = arrays["Time"]
        current = arrays["Current"]
        net_discharged_Ah = -np.r_[
            0.0,
            np.cumsum(
                (current[1:] + current[:-1])
                * 0.5
                * np.diff(time)
                / 3600.0
            ),
        ]
        cycles[condition] = U.make_cycle(
            time,
            current,
            arrays["Voltage"],
            arrays["Battery_Temp_degC"],
            net_discharged_Ah,
            1.0,
            CFG,
        )
        input_hashes[condition] = {
            "path": str(path.relative_to(DATA_ROOT)),
            "sha256": expected,
            "charge_id": charge_id,
            "charge_sha256": audit["charge_verification"][condition]["sha256"],
        }

    contexts = {}
    source_ctx = U.dataset_context(cycles[SOURCE], CFG, source=True)
    contexts[SOURCE] = source_ctx
    for condition in TARGETS:
        contexts[condition] = U.dataset_context(
            cycles[condition], CFG, stats=source_ctx["stats"], source=False
        )
    return contexts, input_hashes, audit


def _ocv_initial_guess() -> tuple[np.ndarray, str]:
    path = OUT / "c20_discharge_ocv.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    soc = np.asarray(data["soc"], dtype=float)
    voltage = np.asarray(data["ocv_V"], dtype=float)
    if len(soc) < 6 or not np.isfinite(soc).all() or not np.isfinite(voltage).all():
        raise ValueError("C/20 discharge OCV curve is incomplete or non-finite")
    coeffs = np.polynomial.chebyshev.chebfit(2.0 * soc - 1.0, voltage, 5)
    return coeffs, P.sha256(path)


def _fit_nls(context: dict, branches: int, ocv_guess: np.ndarray):
    ds = context["ds"]
    starts = context["starts"][context["tr"]]
    length = CFG.window_length
    current = np.stack(
        [-ds._I_raw[start : start + length].numpy() for start in starts]
    ).astype(float)
    z0 = np.asarray([float(ds._SoC_raw[start]) for start in starts])
    soc = z0[:, None] - np.c_[
        np.zeros(len(current)),
        np.cumsum(current[:, :-1], axis=1),
    ] * CFG.dt_seconds / (CFG.nominal_capacity_Ah * 3600.0)
    truth = np.stack(
        [ds._V_raw[start : start + length].numpy() for start in starts]
    ).astype(float)
    guess = np.r_[ocv_guess, np.log10(0.03)]
    for branch in range(branches):
        guess = np.r_[guess, np.log10(0.02), np.log10(5.0 if branch == 0 else 40.0)]
    lower = np.r_[
        np.full(6, -10.0),
        -3.0,
        np.tile([-3.0, np.log10(CFG.dt_seconds)], branches),
    ]
    upper = np.r_[
        np.full(6, 10.0),
        np.log10(0.5),
        np.tile([np.log10(0.5), 4.0], branches),
    ]
    regularization = 1e-3 * np.r_[0.0, [2.0**k for k in range(1, 6)]]

    def residual(theta):
        voltage = U.nls_voltage(theta, current, soc, CFG, branches)
        return np.r_[
            (voltage - truth).ravel(),
            regularization * (theta[:6] - ocv_guess),
        ]

    result = least_squares(
        residual,
        np.clip(guess, lower + 1e-6, upper - 1e-6),
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=500,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    if not result.success:
        raise RuntimeError(f"HG2 {branches}RC NLS failed: {result.message}")
    theta = result.x
    params_dict = {
        "ocv_coeffs": theta[:6].tolist(),
        "R0": float(10.0**theta[6]),
        "R1": float(10.0**theta[7]),
        "C1": float(10.0**theta[8] / 10.0**theta[7]),
        "soc_train_min": float(soc.min()),
        "soc_train_max": float(soc.max()),
        "identification_rmse_V": float(np.sqrt(np.mean(result.fun[: truth.size] ** 2))),
        "identification_n_windows": int(len(current)),
        "identification_success": True,
    }
    if branches == 1:
        params = ECMParams(**params_dict)
    else:
        params = ECMParams2RC(
            **params_dict,
            R2=float(10.0**theta[9]),
            C2=float(10.0**theta[10] / 10.0**theta[9]),
        )
    info = {
        "params": params.to_dict(),
        "theta": theta.tolist(),
        "ocv_initialization": "C/20 discharge branch at 25 C; degree-5 Chebyshev fit",
        "optimizer": "scipy least_squares TRF",
        "nfev": int(result.nfev),
        "message": result.message,
        "identification_success": bool(result.success),
    }
    return params, info


def _evaluate_all(model, model_name: str, contexts: dict, *, smoke: bool) -> list[dict]:
    conditions = (SOURCE,) if smoke else CONDITIONS
    rows = []
    for condition in conditions:
        result = U.evaluate(model, model_name, contexts[condition], CFG)
        for mode, metrics in result.items():
            rows.append(
                {
                    "dataset": DATASET,
                    "model": model_name,
                    "condition": condition,
                    "evaluation": mode,
                    **{f"{key}_mV": value for key, value in metrics.items()},
                }
            )
    return rows


def _train_base(*, smoke: bool = False) -> None:
    output = OUT / ("base_smoke" if smoke else "base")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Part B output: {output}")
    output.mkdir(parents=True)
    contexts, input_hashes, audit = _load_contexts()
    ocv_guess, ocv_hash = _ocv_initial_guess()
    deterministic, metrics_rows, checkpoint_hashes = {}, [], {}
    source = contexts[SOURCE]
    for branches in (1, 2):
        name = f"ecm_{branches}rc_nls"
        params, fit_info = _fit_nls(source, branches, ocv_guess)
        ckpt = output / f"{name}.json"
        ckpt.write_text(json.dumps(fit_info, indent=2) + "\n")
        nls_hash = P.sha256(ckpt)
        checkpoint_hashes[str(ckpt.relative_to(ROOT))] = nls_hash
        deterministic[name] = fit_info
        for condition in ((SOURCE,) if smoke else CONDITIONS):
            result = U.evaluate(params, name, contexts[condition], CFG)
            for mode, values in result.items():
                metrics_rows.append(
                    {
                        "dataset": DATASET,
                        "model": name,
                        "seed": "deterministic",
                        "condition": condition,
                        "evaluation": mode,
                        "checkpoint": str(ckpt.relative_to(ROOT)),
                        "checkpoint_sha256": nls_hash,
                        **{f"{key}_mV": value for key, value in values.items()},
                    }
                )

    seeds = (42,) if smoke else SEEDS
    histories = []
    for name in VARIANTS:
        for seed in seeds:
            set_seed(seed)
            model = U.build_model(
                name, source["stats"], ECMParams.from_dict(deterministic["ecm_1rc_nls"]["params"]), CFG
            )
            ckpt = output / "checkpoints" / f"{name}_seed{seed}.pt"
            train_config = replace(
                TrainConfig(
                    **P.TRAIN,
                    ckpt_path=str(ckpt),
                ),
                n_epochs=2 if smoke else P.TRAIN["n_epochs"],
                patience=2 if smoke else P.TRAIN["patience"],
            )
            history = asdict(
                U.train(
                    model,
                    U.loader(source, CFG, source["tr"], seed, True),
                    U.loader(source, CFG, source["eval_indices"]),
                    train_config,
                )
            )
            histories.append(
                {
                    "model": name,
                    "seed": seed,
                    "best_epoch": history["best_epoch"],
                    "epochs_completed": len(history["train_loss"]),
                    "history": history,
                    "checkpoint": str(ckpt.relative_to(ROOT)),
                    "checkpoint_sha256": P.sha256(ckpt),
                }
            )
            checkpoint_hashes[str(ckpt.relative_to(ROOT))] = P.sha256(ckpt)
            for row in _evaluate_all(model, name, contexts, smoke=smoke):
                row["seed"] = seed
                row["checkpoint"] = str(ckpt.relative_to(ROOT))
                row["checkpoint_sha256"] = P.sha256(ckpt)
                metrics_rows.append(row)

    lstm_seed = 42
    set_seed(lstm_seed)
    lstm = U.build_model("lstm", source["stats"], None, CFG)
    lstm_ckpt = output / "checkpoints" / "lstm_seed42.pt"
    lstm_config = replace(
        TrainConfig(**P.LSTM_TRAIN, ckpt_path=str(lstm_ckpt)),
        n_epochs=2 if smoke else P.LSTM_TRAIN["n_epochs"],
        patience=2 if smoke else P.LSTM_TRAIN["patience"],
    )
    lstm_history = asdict(
        U.train(
            lstm,
            U.loader(source, CFG, source["tr"], lstm_seed, True),
            U.loader(source, CFG, source["eval_indices"]),
            lstm_config,
        )
    )
    histories.append(
        {
            "model": "lstm",
            "seed": lstm_seed,
            "best_epoch": lstm_history["best_epoch"],
            "epochs_completed": len(lstm_history["train_loss"]),
            "history": lstm_history,
            "checkpoint": str(lstm_ckpt.relative_to(ROOT)),
            "checkpoint_sha256": P.sha256(lstm_ckpt),
        }
    )
    checkpoint_hashes[str(lstm_ckpt.relative_to(ROOT))] = P.sha256(lstm_ckpt)
    for row in _evaluate_all(lstm, "lstm", contexts, smoke=smoke):
        row["seed"] = "contextual"
        row["checkpoint"] = str(lstm_ckpt.relative_to(ROOT))
        row["checkpoint_sha256"] = P.sha256(lstm_ckpt)
        metrics_rows.append(row)

    _write_csv(output / "metrics_by_seed.csv", metrics_rows)
    (output / "training_histories.json").write_text(json.dumps(histories, indent=2) + "\n")
    (output / "nls_fits.json").write_text(json.dumps(deterministic, indent=2) + "\n")
    manifest = {
        "data_audit_sha256": P.sha256(OUT / "manifest.json"),
        "c20_ocv_sha256": ocv_hash,
        "raw_inputs": input_hashes,
        "charge_verification_sha256": {
            condition: audit["charge_verification"][condition]["sha256"]
            for condition in CONDITIONS
        },
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {
            name: P.sha256(ROOT / name)
            for name in (
                "reviewer_round2_b.py",
                "reviewer_round2_b_audit.py",
                "exp_unified.py",
                "protocol.py",
                "training.py",
                "data.py",
                "models.py",
                "ecm.py",
                "ecm_2rc.py",
                "unified_dynamics.py",
                "utils.py",
            )
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _frozen_ablation(*, smoke: bool = False) -> None:
    base = OUT / ("base_smoke" if smoke else "base")
    output = OUT / ("a1_smoke" if smoke else "a1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Part B A1 output: {output}")
    output.mkdir(parents=True)
    contexts, input_hashes, _ = _load_contexts()
    nls_info = json.loads((base / "nls_fits.json").read_text())
    params = ECMParams.from_dict(nls_info["ecm_1rc_nls"]["params"])
    source = contexts[SOURCE]
    all_metrics, all_curves, audits = [], [], []
    for variant in NEURAL_VARIANTS:
        for seed in ((42,) if smoke else SEEDS):
            set_seed(seed)
            model = FrozenCorrection(
                U.build_model(variant, source["stats"], params, CFG), CFG
            )
            before = model.physical.detach().clone()
            before_hash = _hash_tensor(before)
            ckpt = output / "checkpoints" / f"{variant}_seed{seed}.pt"
            train_config = replace(
                TrainConfig(**P.TRAIN, ckpt_path=str(ckpt)),
                n_epochs=2 if smoke else P.TRAIN["n_epochs"],
                patience=2 if smoke else P.TRAIN["patience"],
            )
            history = asdict(
                U.train(
                    model,
                    U.loader(source, CFG, source["tr"], seed, True),
                    U.loader(source, CFG, source["eval_indices"]),
                    train_config,
                )
            )
            if not torch.equal(model.physical.detach(), before):
                raise AssertionError("B A1 frozen physical parameters changed")
            if model.physical.grad is not None:
                raise AssertionError("B A1 frozen physical parameters received gradients")
            audit = {
                "variant": variant,
                "seed": seed,
                "physical_sha256_before": before_hash,
                "physical_sha256_after": _hash_tensor(model.physical),
                "frozen_gradient_is_null": model.physical.grad is None,
                "frozen_parameters_unchanged": torch.equal(model.physical.detach(), before),
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "checkpoint_sha256": P.sha256(ckpt),
                "best_epoch": history["best_epoch"],
                "epochs_completed": len(history["train_loss"]),
                "history": history,
            }
            audits.append(audit)
            for epoch, train_loss in enumerate(history["train_loss"], 1):
                all_curves.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": history["val_loss"][epoch - 1],
                    }
                )
            for row in _evaluate_all(model, variant, contexts, smoke=smoke):
                row["seed"] = seed
                row["checkpoint_sha256"] = P.sha256(ckpt)
                all_metrics.append(row)
    _write_csv(output / "metrics_by_seed.csv", all_metrics)
    _write_csv(output / "learning_curves.csv", all_curves)
    (output / "freeze_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "base_nls_sha256": P.sha256(base / "nls_fits.json"),
                "data_audit_sha256": P.sha256(OUT / "manifest.json"),
                "raw_inputs": input_hashes,
                "checkpoint_sha256": {
                    row["checkpoint"]: row["checkpoint_sha256"] for row in audits
                },
            },
            indent=2,
        )
        + "\n"
    )
    if not smoke:
        _write_a1_statistics(all_metrics, base)


def _holm_by_variant(rows: list[dict], p_field: str, adjusted_field: str) -> None:
    for variant in NEURAL_VARIANTS:
        family = [
            {"p_raw": row[p_field], "_row": row}
            for row in rows
            if row["variant"] == variant
        ]
        if len(family) != len(CONDITIONS):
            raise ValueError(f"Incomplete Part B Holm family for {variant}/{p_field}")
        U.holm(family)
        for item in family:
            item["_row"][adjusted_field] = item["p_holm"]


def _write_a1_statistics(a1_metrics: list[dict], base: Path) -> None:
    baseline = list(csv.DictReader((base / "metrics_by_seed.csv").open()))
    nls = {
        row["condition"]: float(row["mae_mV"])
        for row in baseline
        if row["model"] == "ecm_1rc_nls"
        and row["evaluation"] == "continuous"
    }
    joint = {
        (row["model"], int(row["seed"]), row["condition"]): float(row["mae_mV"])
        for row in baseline
        if row["model"] in NEURAL_VARIANTS
        and row["evaluation"] == "continuous"
    }
    nls_rows, joint_rows = [], []
    for variant in NEURAL_VARIANTS:
        for condition in CONDITIONS:
            observations = sorted(
                (
                    row for row in a1_metrics
                    if row["model"] == variant
                    and row["condition"] == condition
                    and row["evaluation"] == "continuous"
                ),
                key=lambda row: int(row["seed"]),
            )
            if [int(row["seed"]) for row in observations] != list(SEEDS):
                raise ValueError(f"Incomplete Part B A1 seed family: {variant}/{condition}")
            values = np.asarray([float(row["mae_mV"]) for row in observations])
            delta_nls = values - nls[condition]
            lower = shifted_wilcoxon(delta_nls + MARGIN_MV, "greater")
            upper = shifted_wilcoxon(delta_nls - MARGIN_MV, "less")
            nls_rows.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "n": len(observations),
                    "median_difference_mV": float(np.median(delta_nls)),
                    "p_lower_raw": lower["p_raw"],
                    "p_upper_raw": upper["p_raw"],
                }
            )
            joint_values = np.asarray(
                [
                    joint[(variant, seed, condition)]
                    for seed in SEEDS
                ]
            )
            paired = U.paired(values, joint_values)
            joint_rows.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "n": len(SEEDS),
                    "median_difference_mV": float(np.median(values - joint_values)),
                    "p_raw": paired["p_raw"],
                }
            )
    for tail in ("lower", "upper"):
        _holm_by_variant(
            nls_rows,
            f"p_{tail}_raw",
            f"p_{tail}_holm",
        )
    for row in nls_rows:
        row["p_tost_raw"] = max(row["p_lower_raw"], row["p_upper_raw"])
        row["p_tost_holm"] = max(row["p_lower_holm"], row["p_upper_holm"])
    _holm_by_variant(joint_rows, "p_raw", "p_holm")
    _write_csv(OUT / "a1_nls_tost.csv", nls_rows)
    _write_csv(OUT / "a1_joint_paired.csv", joint_rows)


def _zero_network_ablation(*, smoke: bool = False) -> None:
    base = OUT / ("base_smoke" if smoke else "base")
    output = OUT / ("a2_smoke" if smoke else "a2")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Part B A2 output: {output}")
    output.mkdir(parents=True)
    contexts, input_hashes, _ = _load_contexts()
    saved = list(csv.DictReader((base / "metrics_by_seed.csv").open()))
    lookup = {
        (row["model"], str(row.get("seed")), row["condition"], row["evaluation"]): row
        for row in saved
    }
    nls_info = json.loads((base / "nls_fits.json").read_text())
    params = ECMParams.from_dict(nls_info["ecm_1rc_nls"]["params"])
    rows = []
    variants = NEURAL_VARIANTS
    seeds = (42,) if smoke else SEEDS
    for variant in variants:
        for seed in seeds:
            ckpt = base / "checkpoints" / f"{variant}_seed{seed}.pt"
            expected_rows = [
                row for row in saved
                if row["model"] == variant and row["seed"] == str(seed)
            ]
            if not expected_rows:
                raise ValueError(f"Missing B checkpoint baseline metrics: {variant}/{seed}")
            set_seed(seed)
            model = U.build_model(variant, contexts[SOURCE]["stats"], params, CFG)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            checkpoint_hash = P.sha256(ckpt)
            conditions_to_check = sorted({row["condition"] for row in expected_rows})
            for condition in conditions_to_check:
                original_by_mode = U.evaluate(model, variant, contexts[condition], CFG)
                for row in (item for item in expected_rows if item["condition"] == condition):
                    original = original_by_mode[row["evaluation"]]
                    for metric in ("mae", "rmse", "max_error", "p99"):
                        if original[metric] != float(row[f"{metric}_mV"]):
                            raise AssertionError(
                                f"B A2 exact baseline mismatch "
                                f"{variant}/{seed}/{condition}/{metric}"
                            )
            with torch.no_grad():
                model.p[10:].zero_()
            for condition in ((SOURCE,) if smoke else CONDITIONS):
                evaluated = U.evaluate(model, variant, contexts[condition], CFG)
                for mode, metrics in evaluated.items():
                    original = lookup[(variant, str(seed), condition, mode)]
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "condition": condition,
                                "evaluation": mode,
                                "metric": metric,
                                "original_metric_mV": float(original[f"{metric}_mV"]),
                                "zeroed_metric_mV": value,
                                "network_minus_zeroed_mV": (
                                    float(original[f"{metric}_mV"]) - value
                                ),
                                "checkpoint": str(ckpt.relative_to(ROOT)),
                                "checkpoint_sha256": checkpoint_hash,
                            }
                        )
    _write_csv(output / "zeroed_metrics.csv", rows)
    if not smoke:
        _write_a2_statistics(rows, saved)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "base_metrics_sha256": P.sha256(base / "metrics_by_seed.csv"),
                "data_audit_sha256": P.sha256(OUT / "manifest.json"),
                "raw_inputs": input_hashes,
                "checkpoint_sha256": {
                    row["checkpoint"]: row["checkpoint_sha256"]
                    for row in rows
                },
            },
            indent=2,
        )
        + "\n"
    )


def _write_a2_statistics(zeroed: list[dict], baseline: list[dict]) -> None:
    relevant_metrics = ("mae", "rmse", "max_error")
    network_rows, pg_rows = [], []
    pg = {
        (int(row["seed"]), row["condition"], row["evaluation"], metric): float(
            row[f"{metric}_mV"]
        )
        for row in baseline
        if row["model"] == "physics_gradient"
        for metric in relevant_metrics
    }
    for variant in NEURAL_VARIANTS:
        for metric in relevant_metrics:
            for condition in CONDITIONS:
                observations = [
                    row for row in zeroed
                    if row["variant"] == variant
                    and row["metric"] == metric
                    and row["condition"] == condition
                    and row["evaluation"] == "continuous"
                ]
                observations.sort(key=lambda row: int(row["seed"]))
                if [int(row["seed"]) for row in observations] != list(SEEDS):
                    raise ValueError(
                        f"Incomplete B A2 family: {variant}/{metric}/{condition}"
                    )
                network_delta = np.asarray(
                    [float(row["network_minus_zeroed_mV"]) for row in observations]
                )
                lower = shifted_wilcoxon(network_delta + MARGIN_MV, "greater")
                upper = shifted_wilcoxon(network_delta - MARGIN_MV, "less")
                network_rows.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "condition": condition,
                        "n": len(observations),
                        "median_network_minus_zeroed_mV": float(
                            np.median(network_delta)
                        ),
                        "p_lower_raw": lower["p_raw"],
                        "p_upper_raw": upper["p_raw"],
                    }
                )
                zeroed_delta = np.asarray(
                    [
                        float(row["zeroed_metric_mV"])
                        - pg[(int(row["seed"]), condition, "continuous", metric)]
                        for row in observations
                    ]
                )
                lower = shifted_wilcoxon(zeroed_delta + MARGIN_MV, "greater")
                upper = shifted_wilcoxon(zeroed_delta - MARGIN_MV, "less")
                pg_rows.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "condition": condition,
                        "n": len(observations),
                        "median_zeroed_minus_physics_gradient_mV": float(
                            np.median(zeroed_delta)
                        ),
                        "p_lower_raw": lower["p_raw"],
                        "p_upper_raw": upper["p_raw"],
                    }
                )
    for rows in (network_rows, pg_rows):
        for variant in NEURAL_VARIANTS:
            for metric in relevant_metrics:
                for tail in ("lower", "upper"):
                    family = [
                        {"p_raw": row[f"p_{tail}_raw"], "_row": row}
                        for row in rows
                        if row["variant"] == variant and row["metric"] == metric
                    ]
                    if len(family) != len(CONDITIONS):
                        raise ValueError(
                            f"Incomplete B A2 Holm family for {variant}/{metric}/{tail}"
                        )
                    U.holm(family)
                    for item in family:
                        item["_row"][f"p_{tail}_holm"] = item["p_holm"]
        for row in rows:
            row["p_tost_raw"] = max(row["p_lower_raw"], row["p_upper_raw"])
            row["p_tost_holm"] = max(row["p_lower_holm"], row["p_upper_holm"])
    _write_csv(OUT / "a2_network_effect_tost.csv", network_rows)
    _write_csv(OUT / "a2_zeroed_vs_physics_gradient_tost.csv", pg_rows)


def run(stage: str, *, smoke: bool = False) -> None:
    torch.set_num_threads(1)
    if stage == "base":
        _train_base(smoke=smoke)
    elif stage == "a1":
        _frozen_ablation(smoke=smoke)
    elif stage == "a2":
        _zero_network_ablation(smoke=smoke)
    else:
        raise ValueError(f"Unknown Part B stage: {stage}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("base", "a1", "a2"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.stage, smoke=args.smoke)
