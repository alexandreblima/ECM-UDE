"""
Unified Evaluation and Time-Series Reconstruction Module (Protocol v2)
========================================================================

Ensures consistent metric computation across ALL models and experiments:
  - Exp 1 (Nominal UDDS 25 degC)
  - Exp 2 (Temperature shift OOD)
  - Exp 3 (Drive cycle shift OOD)
  - Exp C3 (Ablation studies)
  - Exp C4 (ECM-2RC transfer)
  - Multi-dataset benchmarks (CALCE 20R, Mendeley LFP)

Key Principle (Protocol v2):
---------------------------
Metrics are ALWAYS computed on the continuous reconstructed 1-D time series
(averaging predictions on overlapping window regions), NEVER by flattening
raw window instances (which double-counts overlapped samples with differing
predictions).
"""

from __future__ import annotations

import statistics
from typing import Optional, Sequence, Union

import numpy as np
import torch


def reconstruct_time_series(
    preds: Union[torch.Tensor, np.ndarray],
    dataset,
    window_indices: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct continuous 1-D signal by averaging overlapping windows.

    Parameters
    ----------
    preds : tensor or array of shape (N_windows, window_length)
        Normalized predictions for each window.
    dataset : BatteryWindowDataset
        Dataset providing `.starts`, `.cfg.length`, and `.V` (full 1-D signal).
    window_indices : list of int, optional
        Indices of windows represented in `preds`. If None, assumes all windows
        in `dataset` are present in sequential order: range(len(dataset)).

    Returns
    -------
    pred_series : 1-D torch.Tensor
        Normalized predicted voltage along the continuous timeline (points covered).
    targ_series : 1-D torch.Tensor
        Normalized target voltage along the continuous timeline (points covered).
    """
    preds_t = torch.as_tensor(preds, dtype=torch.float32)
    signal_len = int(dataset.V.shape[0])
    length = int(dataset.cfg.length)

    if window_indices is None:
        window_indices = list(range(len(dataset)))

    pred_sum = torch.zeros(signal_len, dtype=torch.float32)
    counts = torch.zeros(signal_len, dtype=torch.float32)

    for pred_win, win_idx in zip(preds_t, window_indices):
        start = int(dataset.starts[int(win_idx)])
        pred_sum[start : start + length] += pred_win
        counts[start : start + length] += 1.0

    mask = counts > 0.0
    if not mask.any():
        raise ValueError("No windows were mapped onto the signal timeline.")

    pred_series = pred_sum[mask] / counts[mask]
    targ_series = dataset.V[mask].to(pred_series.dtype)

    return pred_series, targ_series


def compute_benchmark_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    denorm_std: float = 1.0,
    denorm_mean: float = 0.0,
) -> dict[str, float]:
    """Compute standard battery voltage metrics in physical units (Volts).

    Parameters
    ----------
    pred : 1-D tensor or array (normalized or physical)
    target : 1-D tensor or array (normalized or physical)
    denorm_std : float
        Standard deviation for denormalization (V_std).
    denorm_mean : float
        Mean for denormalization (V_mean).

    Returns
    -------
    dict with:
        mae     : Mean Absolute Error [V]
        rmse    : Root Mean Squared Error [V]
        p95_err : 95th percentile absolute error [V]
        p99_err : 99th percentile absolute error [V]
        max_err : Maximum absolute error [V]
    """
    p = torch.as_tensor(pred, dtype=torch.float64) * denorm_std + denorm_mean
    t = torch.as_tensor(target, dtype=torch.float64) * denorm_std + denorm_mean

    err = (p - t).flatten()
    abs_err = err.abs()

    return {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((err ** 2).mean()).item()),
        "p95_err": float(torch.quantile(abs_err, 0.95).item()),
        "p99_err": float(torch.quantile(abs_err, 0.99).item()),
        "max_err": float(abs_err.max().item()),
    }


def aggregate_metrics(seed_metrics: list[dict[str, float]]) -> dict:
    """Aggregate metric dictionaries across multiple random seeds."""
    if not seed_metrics:
        return {}

    keys = list(seed_metrics[0].keys())
    agg = {}
    for k in keys:
        vals = [float(m[k]) for m in seed_metrics]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        vals_t = torch.tensor(vals, dtype=torch.float64)
        q25, q75 = torch.quantile(
            vals_t, torch.tensor([0.25, 0.75], dtype=torch.float64)
        ).tolist()
        agg[k] = {
            "mean": mean,
            "std": std,
            "median": statistics.median(vals),
            "iqr": q75 - q25,
            "min": min(vals),
            "max": max(vals),
            "cv_pct": (std / mean * 100) if mean > 0 else 0.0,
            "all": vals,
        }
    return agg
