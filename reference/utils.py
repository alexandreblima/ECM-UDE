"""
Miscellaneous utilities — wno_battery.utils
============================================

Overview
--------
This module provides two pipeline-wide utilities: reproducibility
seeding and device selection.  Both are called at the top of every
experiment script before any model or data construction takes place.

Public API
----------
  set_seed(seed)          Seed all relevant RNGs for reproducibility.
  pick_device(prefer_gpu) Return the best available device as a string.

Reproducibility notes
---------------------
set_seed covers Python's random, NumPy, and PyTorch (CPU, CUDA, MPS).
It does not set torch.use_deterministic_algorithms(True) because
several operations used by pytorch_wavelets and the FNO FFT path do
not have deterministic CUDA implementations, and enabling that flag
would raise RuntimeError on GPU runs.

For the MPS backend (Apple Silicon), torch.manual_seed seeds the MPS
RNG via the shared PyTorch seed state.  There is no separate
mps.manual_seed API in current PyTorch versions.

Even with set_seed, complete bit-for-bit reproducibility is not
guaranteed across different PyTorch versions, operating systems, or
hardware because floating-point reduction order can vary with thread
scheduling.  set_seed is sufficient to produce stable aggregate
statistics across the 10-seed sweep in Exp. 1 (CV < 5% for LSTM,
< 18% for FNO) and to make single-seed results (Exp. 2-5) comparable
across runs on the same machine and PyTorch version.

Device selection notes
----------------------
pick_device returns a string ("cuda", "mps", or "cpu") rather than a
torch.device object so it can be stored in TrainConfig.device and
serialised to summary.json without extra conversion.

MPS availability requires PyTorch >= 1.12 and macOS >= 12.3.  The
hasattr guard on torch.backends.mps makes the code safe on older
installations that pre-date MPS support.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility.

    Seeds Python's random, NumPy, and PyTorch (CPU + CUDA + MPS) with
    the same integer.  Called once per training run in the experiment
    scripts, before model construction and DataLoader creation, so that
    weight initialisation and batch-shuffle order are both deterministic.

    In Exp. 1, set_seed is also called inside run_one_seed() before
    each of the 10 seeds to reset RNG state between runs and prevent
    state leakage from earlier seeds into later ones.

    Parameters
    ----------
    seed : int
        Any non-negative integer.  The experiment scripts use dispersed
        values [0, 7, 42, 137, 256, 512, 1024, 2048, 31337, 99999] to
        reduce pseudo-random correlations in PyTorch's MPS backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS RNG is seeded via torch.manual_seed in current PyTorch versions;
    # no separate mps.manual_seed API exists as of PyTorch 2.x.


def pick_device(prefer_gpu: bool = True) -> str:
    """Return the best available compute device as a string.

    Priority order when prefer_gpu=True:
      1. "cuda"   NVIDIA GPU via CUDA.
      2. "mps"    Apple Silicon GPU via Metal Performance Shaders.
                  Requires PyTorch >= 1.12 and macOS >= 12.3.
      3. "cpu"    Fallback on all other platforms.

    Returns a plain string (not a torch.device) so it can be stored in
    TrainConfig.device and serialised to summary.json.

    Parameters
    ----------
    prefer_gpu : bool
        If False, skip GPU checks and return "cpu" unconditionally.
        Useful for debugging or when running on a shared cluster node
        where GPU access should be explicitly requested.

    Returns
    -------
    str : one of "cuda", "mps", "cpu".
    """
    if not prefer_gpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"