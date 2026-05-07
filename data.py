"""
Data loading and preprocessing — wno_battery.data
==================================================

Overview
--------
This module provides all data-layer components for the WNO battery
voltage estimation pipeline.  It handles loading of Kollmeyer .mat
files, derivation of state-of-charge, normalisation, windowed dataset
construction, and train/val dataloader creation.

Public API
----------
  load_drive_cycle(mat_path)          Read one .mat file into tensors.
  compute_soc(ah_discharged)          Derive SoC from amp-hours.
  compute_norm_stats(I, V, T)         Compute normalisation statistics.
  WindowConfig                        Dataclass for window parameters.
  BatteryWindowDataset                torch.utils.data.Dataset over windows.
  make_dataloaders(mat_path, ...)     Build train/val loaders for one cycle.

Dataset format
--------------
The Kollmeyer (2018) dataset distributes drive cycles as MATLAB .mat
files.  Each file contains a struct called `meas` with the following
fields relevant to this pipeline:

  Time                  time [s]
  Current               cell current [A], discharge negative
  Voltage               terminal voltage [V]
  Battery_Temp_degC     cell temperature [degC]
  Ah                    cumulative amp-hours discharged [Ah], <= 0

`load_drive_cycle` reads these fields, verifies their lengths, derives
SoC via `compute_soc`, and returns a dict of float32 tensors.

Normalisation protocol
----------------------
All normalisation decisions are driven by a single principle: no
information from the validation or test distributions must influence the
normalisation applied to training data, and vice versa.

Current (I) and voltage (V) use empirical z-score statistics computed
from the training portion of the source cycle only.  Both vary strongly
with drive cycle and are therefore best represented relative to the
training distribution.

Temperature (T) and state-of-charge (SoC) use fixed physical constants:
  T   : mean=0, std=25  (covers the full Kollmeyer range [-20, +25 C])
  SoC : mean=0.5, std=0.3

The rationale for fixing T is critical and is described in detail in
the `compute_norm_stats` docstring.  In brief: the empirical sigma of T
within a single 25 C test cycle is ~0.4 C.  Applying that sigma to a
-20 C test cycle yields T_norm ~ -115, which is catastrophically out of
range for any model trained at 25 C.  Fixed normalisation keeps T_norm
in [-0.8, 1.1] across the full dataset.  This design decision is also
discussed in Section 4.2.1 of the paper.

Windowed dataset
----------------
`BatteryWindowDataset` segments a drive cycle into overlapping windows
of length L samples with stride S (S < L gives overlap).  Each window
yields a pair (X_win, V_win):

  X_win  float32 tensor, shape (L, 3)    features: (I, T_cell, SoC)
  V_win  float32 tensor, shape (L,)      target: normalised voltage

The class exposes three attributes used by the experiment scripts:
  .starts    list of window start indices, length == len(dataset)
  .V         normalised full voltage tensor, shape (N_samples,)
  .stats     normalisation stats dict (V_mean, V_std, I_mean, ...)

.starts is required by `reconstruct_validation_series` in Exp. 1 to
map window predictions back onto the raw timeline.

Guard band
----------
When `make_dataloaders` uses a temporal split, a guard band of
ceil(L / S) - 1 windows is dropped at the train/val boundary.  This
ensures that no raw time sample appears in both the training and
validation subsets despite the window overlap.  Without the guard band,
up to L - S raw samples at the boundary would be covered by both a
training window and a validation window, constituting a subtle form of
data leakage.  The guard band is always applied; it costs at most
(ceil(L/S) - 1) * S / L windows, which is at most one window for
stride = L // 2 (50% overlap).

Usage pattern
-------------
Typical single-cycle usage (Exp. 1):

    train_loader, val_loader, dataset = make_dataloaders(
        "25degC_UDDS_Pan18650PF.mat",
        window_cfg=WindowConfig(length=1024, stride=512, normalize=True),
        batch_size=16,
        val_fraction=0.2,
        seed=42,
        split_mode="temporal",
    )
    V_mean = dataset.stats["V_mean"]

Multi-cycle usage (Exp. 4): load each cycle with normalize=False,
collect raw training samples, call compute_norm_stats on the
concatenated signals, then call dataset.renormalize(stats) on each
dataset before building ConcatDataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset, Subset


# --------------------------------------------------------------------------
# .mat reader
# --------------------------------------------------------------------------

def load_drive_cycle(mat_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a Kollmeyer .mat file and return float32 tensors.

    Reads the `meas` MATLAB struct, extracts the five raw channels, and
    derives SoC via `compute_soc`.  All tensors are 1-D float32.

    Parameters
    ----------
    mat_path : str or Path
        Path to a Kollmeyer .mat file, e.g.
        "25degC_UDDS_Pan18650PF.mat".

    Returns
    -------
    dict with keys:
        't'      : time [s]
        'I'      : current [A], discharge negative
        'V'      : terminal voltage [V]
        'T_cell' : cell temperature [degC]
        'Ah'     : cumulative amp-hours discharged [Ah], <= 0
        'SoC'    : state-of-charge in [0, 1], derived from Ah

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    KeyError           if the file lacks a 'meas' struct or a required
                       field within it.
    ValueError         if the channel arrays have inconsistent lengths.
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"Could not find .mat file: {mat_path}")

    raw = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    if "meas" not in raw:
        raise KeyError(
            f"File {mat_path.name} does not contain a 'meas' struct. "
            f"Top-level keys: "
            f"{[k for k in raw.keys() if not k.startswith('__')]}"
        )
    meas = raw["meas"]

    def _field(name: str) -> np.ndarray:
        if not hasattr(meas, name):
            available = [a for a in dir(meas) if not a.startswith("_")]
            raise KeyError(
                f"Field '{name}' not found in meas struct. "
                f"Available fields: {available}"
            )
        return np.asarray(getattr(meas, name)).squeeze().astype(np.float32)

    t      = _field("Time")
    I      = _field("Current")
    V      = _field("Voltage")
    T_cell = _field("Battery_Temp_degC")
    Ah     = _field("Ah")

    n = len(t)
    for name, arr in [("Current", I), ("Voltage", V),
                      ("Battery_Temp_degC", T_cell), ("Ah", Ah)]:
        if len(arr) != n:
            raise ValueError(
                f"Length mismatch in {mat_path.name}: Time has {n} "
                f"samples but {name} has {len(arr)}."
            )

    soc = compute_soc(Ah)

    return {
        "t":      torch.from_numpy(t),
        "I":      torch.from_numpy(I),
        "V":      torch.from_numpy(V),
        "T_cell": torch.from_numpy(T_cell),
        "Ah":     torch.from_numpy(Ah),
        "SoC":    torch.from_numpy(soc),
    }


# --------------------------------------------------------------------------
# SoC derivation
# --------------------------------------------------------------------------

def compute_soc(ah_discharged: np.ndarray) -> np.ndarray:
    """Compute SoC in [0, 1] from the cumulative amp-hours channel.

    Assumes the minimum of `ah_discharged` corresponds to SoC = 0 and
    Ah = 0 corresponds to SoC = 1.  This is valid for Kollmeyer cycles
    that begin at or near full charge and discharge to near-empty.

    The derivation is:
        bias  = -min(Ah)          # total capacity discharged
        SoC   = (Ah + bias) / bias

    Values are clipped to [0, 1] to handle sensor noise at the
    boundaries.

    Parameters
    ----------
    ah_discharged : np.ndarray
        Cumulative amp-hours discharged, shape (N,), values <= 0.

    Returns
    -------
    np.ndarray, float32, shape (N,), values in [0, 1].
    """
    bias = -float(ah_discharged.min())
    if bias <= 0.0:
        # Flat Ah channel (e.g. constant-current soak segment):
        # return all-ones (assume fully charged).
        return np.ones_like(ah_discharged)
    soc = (ah_discharged + bias) / bias
    return np.clip(soc, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------

def _identity_stats() -> dict[str, float]:
    """Return normalisation statistics that act as a no-op (mean=0, std=1)."""
    return {
        "I_mean": 0.0, "I_std": 1.0,
        "V_mean": 0.0, "V_std": 1.0,
        "T_mean": 0.0, "T_std": 1.0,
        "SoC_mean": 0.0, "SoC_std": 1.0,
    }


def compute_norm_stats(
    I: torch.Tensor,
    V: torch.Tensor,
    T: torch.Tensor,
) -> dict[str, float]:
    """Compute normalisation statistics from raw training-split signals.

    I and V use empirical z-score (mean and std from the data).
    T and SoC use fixed physical constants (see module docstring and
    rationale below).

    Parameters
    ----------
    I, V, T : 1-D float tensors
        Raw (unnormalised) signals from the **training portion only**.
        Passing full-cycle signals introduces a statistical leak from
        the validation region into the normalisation applied at
        training time.

    Returns
    -------
    dict with keys: I_mean, I_std, V_mean, V_std, T_mean, T_std,
    SoC_mean, SoC_std.

    Normalisation details
    ---------------------
    I, V  Empirical z-score from training data.  A small epsilon (1e-8)
          is added to std to guard against zero-std signals (e.g. a
          constant-current soak segment that appears in some Kollmeyer
          low-temperature recordings).

    T     Fixed: mean=0, std=25.  Rationale: within a single 25 C test
          the cell temperature varies by only ~1.3 C, giving sigma_T ~
          0.4 C.  Dividing a -20 C test cycle by that sigma would
          produce T_norm ~ -115, collapsing neural-operator predictions
          entirely.  std=25 keeps T_norm in [-0.8, 1.1] across the
          full [-20, +25 C] dataset range and is consistent with the
          physical interpretation of 25 C as the reference temperature.

    SoC   Fixed: mean=0.5, std=0.3.  SoC is bounded in [0, 1] by
          construction; this centres and scales it to approximately
          [-1.7, 1.7], comparable to I_norm and T_norm in magnitude.
    """
    return {
        "I_mean":   float(I.mean().item()),
        "I_std":    float(I.std().item()) + 1e-8,
        "V_mean":   float(V.mean().item()),
        "V_std":    float(V.std().item()) + 1e-8,
        "T_mean":   0.0,
        "T_std":    25.0,
        "SoC_mean": 0.5,
        "SoC_std":  0.3,
    }


# --------------------------------------------------------------------------
# WindowConfig
# --------------------------------------------------------------------------

@dataclass
class WindowConfig:
    """Parameters for segmenting a drive cycle into training windows.

    Attributes
    ----------
    length : int
        Samples per window.  At 10 Hz, length=1024 is ~102 s, which
        captures the dominant RC time constant (tau_1 ~ 15 s at 25 C)
        many times over and covers most drive-cycle micro-trip durations.
    stride : int
        Step between consecutive window start positions.  stride < length
        gives overlapping windows, which amplifies dataset size at the
        cost of temporal correlation between windows.  The canonical
        setting is stride = length // 2 (50% overlap).
    normalize : bool
        If True, apply normalisation when constructing the dataset.
        Set to False when building a dataset solely to extract raw
        signals for statistics computation (Exp. 4 pipeline).
    """
    length:    int  = 1024
    stride:    int  = 512
    normalize: bool = True


# --------------------------------------------------------------------------
# BatteryWindowDataset
# --------------------------------------------------------------------------

class BatteryWindowDataset(Dataset):
    """Windowed dataset over one drive cycle.

    Each call to __getitem__(idx) returns:
        X_win : float32 tensor, shape (length, 3)   -- (I, T_cell, SoC)
        V_win : float32 tensor, shape (length,)     -- normalised voltage

    Key attributes
    --------------
    .starts  : list[int]
        Start index of each window in the raw signal.  Required by
        `reconstruct_validation_series` in Exp. 1 to map window
        predictions back onto the raw timeline.
    .V       : torch.Tensor, shape (N_raw_samples,)
        Normalised full voltage tensor.  Also used by
        `reconstruct_validation_series`.
    .stats   : dict[str, float]
        Normalisation statistics applied to this dataset.
    .n_features : int
        Number of input features per timestep (always 3).
    .feature_names : tuple[str, ...]
        Human-readable names of the input features: ('I', 'T_cell', 'SoC').

    Normalisation modes
    -------------------
    Three modes are supported, controlled by `cfg.normalize` and `stats`:

    1. cfg.normalize=False
       No normalisation.  Identity stats are stored.  Use this when
       building a dataset solely to collect raw signals for stat
       computation (Exp. 4 multi-condition pipeline).

    2. cfg.normalize=True, stats=<dict>
       Apply externally provided stats.  Correct mode for:
         (a) applying train-only stats after split (make_dataloaders)
         (b) applying source-domain stats to a target dataset in Exp. 2-3

    3. cfg.normalize=True, stats=None
       Compute stats from the full cycle passed in (fallback convenience
       mode for simple use cases).  Not recommended for any experiment
       that performs a train/val split: the stats will include validation
       signal, introducing a small statistical leak.  make_dataloaders
       always overrides this via renormalize().
    """

    def __init__(
        self,
        cycle: dict[str, torch.Tensor],
        cfg: WindowConfig,
        stats: Optional[dict[str, float]] = None,
    ) -> None:
        self.cfg = cfg

        # Store raw signals to support renormalize() and raw_samples_for_indices()
        self._I_raw   = cycle["I"].float()
        self._V_raw   = cycle["V"].float()
        self._T_raw   = cycle["T_cell"].float()
        self._SoC_raw = cycle["SoC"].float()

        # Determine normalisation stats
        if not cfg.normalize:
            self.stats = _identity_stats()
        elif stats is not None:
            self.stats = dict(stats)
        else:
            # Fallback: full-cycle stats.  Overridden by make_dataloaders.
            self.stats = compute_norm_stats(
                self._I_raw, self._V_raw, self._T_raw
            )

        self._apply_stats()

        # Precompute window start indices
        n = len(self._I_raw)
        L = cfg.length
        S = cfg.stride
        if n < L:
            raise ValueError(
                f"Cycle has {n} samples but window length is {L}. "
                f"Reduce WindowConfig.length or use a longer cycle."
            )
        self.starts = list(range(0, n - L + 1, S))

    # -- internal --

    def _apply_stats(self) -> None:
        """Normalise raw signals with self.stats and rebuild X, V tensors."""
        s = self.stats
        I   = (self._I_raw   - s["I_mean"])   / s["I_std"]
        V   = (self._V_raw   - s["V_mean"])   / s["V_std"]
        T   = (self._T_raw   - s["T_mean"])   / s["T_std"]
        SoC = (self._SoC_raw - s["SoC_mean"]) / s["SoC_std"]
        self.X = torch.stack([I, T, SoC], dim=-1)   # (N, 3)
        self.V = V                                   # (N,)
        self.n_features    = self.X.shape[-1]
        self.feature_names = ("I", "T_cell", "SoC")

    # -- Dataset interface --

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        L = self.cfg.length
        return self.X[s : s + L, :], self.V[s : s + L]

    # -- Utilities --

    def denormalize_V(self, V_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised voltage back to physical volts."""
        return V_norm * self.stats["V_std"] + self.stats["V_mean"]

    def raw_samples_for_indices(
        self,
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw (unnormalised) I, V, T samples covered by window indices.

        Collects all raw sample positions touched by the specified windows
        (union of [start, start + length) for each window index) and
        returns the corresponding raw signal values.

        Used by make_dataloaders and Exp. 4 to compute train-only
        normalisation statistics without including validation samples.

        Parameters
        ----------
        indices : list[int]
            Window indices (positions in self.starts).

        Returns
        -------
        (I_raw, V_raw, T_raw) : three 1-D float32 tensors covering the
        union of raw sample positions touched by the given windows.
        """
        L = self.cfg.length
        sample_set: set[int] = set()
        for idx in indices:
            s = self.starts[idx]
            sample_set.update(range(s, s + L))
        si = sorted(sample_set)
        return self._I_raw[si], self._V_raw[si], self._T_raw[si]

    def renormalize(self, new_stats: dict[str, float]) -> None:
        """Re-apply normalisation in-place with new statistics.

        Replaces self.stats with new_stats and rebuilds self.X and self.V.
        This is the mechanism by which make_dataloaders ensures that both
        the train and val Subsets (which share a reference to the same
        underlying dataset) are normalised with training-split-only stats.

        Also used in Exp. 4 to apply global multi-cycle stats to each
        per-cycle dataset after the global stats have been computed from
        the concatenated training portions of all pool cycles.

        Parameters
        ----------
        new_stats : dict[str, float]
            Stats dict with keys: I_mean, I_std, V_mean, V_std,
            T_mean, T_std, SoC_mean, SoC_std.
        """
        self.stats = dict(new_stats)
        self._apply_stats()


# --------------------------------------------------------------------------
# make_dataloaders
# --------------------------------------------------------------------------

def make_dataloaders(
    mat_path: str | Path,
    window_cfg: Optional[WindowConfig] = None,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
    split_mode: str = "temporal",
    stats: Optional[dict[str, float]] = None,
) -> tuple[DataLoader, DataLoader, BatteryWindowDataset]:
    """Load one drive cycle and build train/val dataloaders.

    Normalisation statistics are computed from the training split only
    (unless externally provided), then applied to the full dataset via
    renormalize().  Both train and val Subsets therefore share the same
    normalisation derived solely from training data.

    Parameters
    ----------
    mat_path : str or Path
        Path to a Kollmeyer .mat file.
    window_cfg : WindowConfig or None
        Window parameters.  Defaults to WindowConfig() (length=1024,
        stride=512, normalize=True).
    batch_size : int
        Batch size for both loaders.
    val_fraction : float
        Fraction of windows reserved for validation.
    seed : int
        Random seed for the DataLoader shuffle generator and, in
        split_mode="random", for the permutation.
    split_mode : {"temporal", "random"}
        "temporal" (default): contiguous temporal blocks with guard band
        (see module docstring).  Recommended for all experiments.
        "random": random permutation split.  Preserved for backward
        compatibility; not recommended for time-series data because it
        allows future values to appear in the training set.
    stats : dict or None
        If provided, these normalisation stats are applied directly and
        no stats are computed from the training split.  Used in
        transfer experiments (Exp. 2-3) to apply source-domain stats
        to a target-domain dataset.

    Returns
    -------
    train_loader : DataLoader  (shuffled)
    val_loader   : DataLoader  (sequential)
    full_dataset : BatteryWindowDataset
        The underlying dataset, shared by both loaders.  Exposes
        .stats, .n_features, .feature_names, .starts, and .V.
    """
    if window_cfg is None:
        window_cfg = WindowConfig()

    cycle = load_drive_cycle(mat_path)

    # Step 1: build dataset without normalisation to access raw signals
    dataset = BatteryWindowDataset(
        cycle,
        WindowConfig(
            length=window_cfg.length,
            stride=window_cfg.stride,
            normalize=False,
        ),
    )

    # Step 2: determine train/val split indices
    n     = len(dataset)
    n_val = max(1, int(n * val_fraction))

    if split_mode == "temporal":
        n_train   = n - n_val
        L         = window_cfg.length
        S         = window_cfg.stride
        overlap   = max(0, (L + S - 1) // S - 1)
        train_end = max(0, n_train - overlap)
        train_indices = list(range(0, train_end))
        val_indices   = list(range(n_train, n))

    elif split_mode == "random":
        n_train = n - n_val
        gen     = torch.Generator().manual_seed(seed)
        perm    = torch.randperm(n, generator=gen).tolist()
        train_indices = perm[:n_train]
        val_indices   = perm[n_train:]

    else:
        raise ValueError(
            f"Unknown split_mode: {split_mode!r}. "
            f"Choose 'temporal' or 'random'."
        )

    # Step 3: compute normalisation stats from training samples only
    if stats is not None:
        train_stats = dict(stats)
    elif window_cfg.normalize:
        I_tr, V_tr, T_tr = dataset.raw_samples_for_indices(train_indices)
        train_stats = compute_norm_stats(I_tr, V_tr, T_tr)
    else:
        train_stats = _identity_stats()

    # Step 4: re-normalise the full dataset with train-only stats.
    # Both Subsets view into the same underlying object, so they
    # automatically share the same normalisation.
    dataset.cfg = window_cfg      # restore normalize=True
    dataset.renormalize(train_stats)

    # Step 5: build Subsets and DataLoaders
    gen          = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size, shuffle=True,
        drop_last=False, generator=gen,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )
    return train_loader, val_loader, dataset