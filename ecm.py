"""
ECM-1RC baseline — wno_battery.ecm
====================================

Overview
--------
This module provides a deterministic physics-based baseline for terminal
voltage prediction using a first-order Thevenin equivalent circuit model
(ECM-1RC).  It is trained and evaluated under the same window-based
protocol as the LSTM, FNO, and WNO models of Experiments 1-5, enabling
direct comparison in all summary tables.

Model definition
----------------
First-order Thevenin circuit (Plett 2004 convention, I > 0 = discharge):

    V(t)   = OCV(SoC(t)) - R0 * I(t) - V1(t)
    dV1/dt = -V1(t) / (R1 * C1) + I(t) / C1

OCV(SoC) is represented as a Chebyshev polynomial of degree OCV_DEGREE
on the domain [0, 1].  Integration uses forward Euler at dt = 0.1 s
(10 Hz sampling), matching the Kollmeyer dataset rate.  V1(0) = 0 for
each window, matching the stateless window-wise evaluation of the neural
models.

ECM-1RC has 9 scalar parameters: OCV_DEGREE + 1 = 6 Chebyshev
coefficients plus R0, R1, C1.  These are identified jointly via
nonlinear least squares on the training windows.

Identification design decisions
--------------------------------
1. Chebyshev OCV basis (not monomials).
   The Vandermonde matrix of the monomial basis {1, s, s^2, ...} on
   [0, 1] is severely ill-conditioned (Runge phenomenon).  The joint
   optimizer degenerates: it drives R0, R1, C1 to their bounds and
   absorbs all residual into OCV.  Chebyshev on [0, 1] has a well-
   conditioned Gram matrix; the optimizer distributes the fit correctly
   between OCV and the RC parameters.

2. Log10 parameterisation for R0, R1, C1.
   RC parameters span several orders of magnitude across cell chemistries
   and temperatures.  Log10 parameterisation makes the search space
   approximately uniform in scale and prevents the optimizer from
   spending most iterations near zero.

3. Jacobian-based variable scaling (x_scale='jac').
   The Chebyshev coefficients (~0.1 V scale) and log10 RC parameters
   (~unit scale) have very different Jacobian sensitivities.  scipy's
   x_scale='jac' rescales each variable by its Jacobian column norm,
   making the trust-region steps well-conditioned without manual tuning.

4. Tikhonov regularisation on OCV high-order modes.
   A smoothness prior damps Runge-style oscillations in the OCV curve
   at SoC values not covered by the training data (e.g. the low-SoC
   knee).  The penalty weight grows geometrically with polynomial order
   (OCV_REG_GROWTH = 2 per order) so that the constant and linear terms
   are unpenalised and only high-frequency modes are suppressed.

5. Degeneracy guard.
   If any of R0, R1, C1 lands within BOUND_PROXIMITY_TOL of its bound
   after optimisation, fit_ecm_1rc raises RuntimeError with a
   diagnostic instead of silently returning a degenerate solution.

6. OCV SoC clipping at inference time.
   The Chebyshev polynomial is only reliable within the SoC range
   observed during identification.  predict_ecm clips the SoC argument
   to [soc_train_min, soc_train_max] before evaluating OCV.  The RC
   dynamics (V1 update) are not clipped; the model continues to respond
   correctly to I(t) at out-of-range SoC.

Sign convention
---------------
The Kollmeyer dataset and the wno_battery neural pipeline use the
convention I > 0 = charge (discharge currents are negative in the
denormalised I channel).  The ECM equations are written in the classical
electrochemistry convention I > 0 = discharge.  The conversion is done
once at the pipeline boundary in _denorm_batch(), which flips the sign
of I after denormalisation.  All ECM-internal code uses the classical
convention exclusively.

Public API
----------
  ECMParams                   Identified ECM parameter set.
  fit_ecm_1rc(loader, stats)  Identify parameters from training windows.
  predict_ecm(params, loader, stats)
                              Run the ECM over a DataLoader; returns
                              normalised tensors matching the neural
                              model output interface.

Interface compatibility
-----------------------
predict_ecm returns (preds, targs) as normalised float32 tensors in the
same space as training.predict(), so extended_metrics() in the experiment
scripts can be called without modification.

Dependencies
------------
  numpy, scipy   Chebyshev fitting and nonlinear least squares.
  torch          Tensor I/O for DataLoader compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import torch
from numpy.polynomial import Chebyshev
from scipy.optimize import least_squares


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DT_SECONDS:   float = 0.1   # Kollmeyer dataset sampling period [s]
OCV_DEGREE:   int   = 5     # Chebyshev polynomial degree for OCV(SoC)
N_OCV_COEFFS: int   = OCV_DEGREE + 1
OCV_DOMAIN          = (0.0, 1.0)

# Fixed normalisation constants from data.py (must stay in sync)
SOC_MU:    float = 0.5
SOC_SIGMA: float = 0.3

# RC parameter bounds in log10 space.
# Physical range for a Panasonic 18650PF NCA cell at -20 to 25 C.
LOG10_R0_LO, LOG10_R0_HI = np.log10(1e-3), np.log10(0.5)   # 1 mOhm .. 500 mOhm
LOG10_R1_LO, LOG10_R1_HI = np.log10(1e-3), np.log10(0.5)   # 1 mOhm .. 500 mOhm
LOG10_C1_LO, LOG10_C1_HI = np.log10(1.0),  np.log10(1e5)   # 1 F    .. 100 kF

# Distance (in log10 units) from a bound that triggers the degeneracy guard.
BOUND_PROXIMITY_TOL: float = 1e-3

# Tikhonov regularisation on OCV Chebyshev coefficients.
# Penalty per coefficient k: OCV_REG_BASE * OCV_REG_GROWTH^k.
# At typical coefficient magnitudes (~0.1 V), this contributes ~1 mV
# equivalent at k=1, well below the ~10-60 mV data residual.
OCV_REG_BASE:   float = 1e-3
OCV_REG_GROWTH: float = 2.0


# --------------------------------------------------------------------------
# ECMParams dataclass
# --------------------------------------------------------------------------

@dataclass
class ECMParams:
    """Identified parameters of the ECM-1RC model.

    Attributes
    ----------
    ocv_coeffs : np.ndarray, shape (N_OCV_COEFFS,)
        Chebyshev coefficients of OCV(SoC) on domain [0, 1].
        Evaluate with ocv_of(soc) or chebyshev_polynomial()(soc).
    R0 : float   Ohmic resistance [Ohm].
    R1 : float   Charge-transfer resistance [Ohm].
    C1 : float   Double-layer capacitance [F].
    soc_train_min, soc_train_max : float
        SoC range observed during identification.  OCV is clipped to
        this range at inference time to prevent Runge extrapolation.
    identification_rmse_V : float
        Data-only RMSE on the identification set [V].
    identification_n_windows : int
        Number of windows used for identification.
    identification_success : bool
        scipy.optimize.least_squares convergence flag.
    """
    ocv_coeffs:              np.ndarray
    R0:                      float
    R1:                      float
    C1:                      float
    soc_train_min:           float = 0.0
    soc_train_max:           float = 1.0
    identification_rmse_V:   float = 0.0
    identification_n_windows: int  = 0
    identification_success:  bool  = True

    def chebyshev_polynomial(self) -> Chebyshev:
        """Return a numpy Chebyshev object for OCV evaluation."""
        return Chebyshev(self.ocv_coeffs, domain=OCV_DOMAIN)

    def ocv_of(self, soc: np.ndarray) -> np.ndarray:
        """Evaluate OCV [V] at the given SoC values."""
        return self.chebyshev_polynomial()(soc)

    def tau1(self) -> float:
        """Dominant RC time constant [s]."""
        return self.R1 * self.C1

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (for summary.json)."""
        d = asdict(self)
        d["ocv_coeffs"] = list(self.ocv_coeffs)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ECMParams":
        """Deserialise from a dict produced by to_dict()."""
        d = dict(d)
        d["ocv_coeffs"] = np.asarray(d["ocv_coeffs"], dtype=np.float64)
        return cls(**d)


# --------------------------------------------------------------------------
# Denormalisation helpers
# --------------------------------------------------------------------------

def _denorm_batch(
    X_norm: torch.Tensor,
    train_stats: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Denormalise a window batch and convert to classical ECM sign convention.

    The wno_battery pipeline uses I > 0 = charge (discharge negative).
    The ECM equations use I > 0 = discharge.  The sign flip is applied
    once here, at the pipeline boundary.

    Parameters
    ----------
    X_norm : (B, L, 3) float tensor, normalised features (I, T_cell, SoC)
    train_stats : normalisation stats dict from data.compute_norm_stats

    Returns
    -------
    I_ecm : (B, L) float64 array, I > 0 = discharge [A]
    soc   : (B, L) float64 array, SoC in [0, 1]
    """
    X = X_norm.detach().cpu().numpy().astype(np.float64)
    I_pipeline = X[..., 0] * float(train_stats["I_std"]) + float(train_stats["I_mean"])
    I_ecm = -I_pipeline   # flip to classical convention: I > 0 = discharge
    soc   = X[..., 2] * SOC_SIGMA + SOC_MU
    return I_ecm, soc


def _denorm_voltage(
    V_norm: torch.Tensor,
    train_stats: dict[str, float],
) -> np.ndarray:
    """Denormalise voltage to physical units [V]."""
    return (
        V_norm.detach().cpu().numpy().astype(np.float64)
        * float(train_stats["V_std"])
        + float(train_stats["V_mean"])
    )


def _norm_voltage(
    V_physical: np.ndarray,
    train_stats: dict[str, float],
) -> np.ndarray:
    """Normalise physical voltage to the pipeline's normalised space."""
    return (V_physical - float(train_stats["V_mean"])) / float(train_stats["V_std"])


# --------------------------------------------------------------------------
# OCV evaluation
# --------------------------------------------------------------------------

def _eval_ocv(ocv_coeffs: np.ndarray, soc: np.ndarray) -> np.ndarray:
    """Evaluate the Chebyshev OCV polynomial.

    numpy.polynomial.chebyshev.chebval operates in the standard domain
    [-1, 1].  We apply the affine map [0, 1] -> [-1, 1] inline to avoid
    constructing a Chebyshev object inside the simulation hot loop.
    """
    return np.polynomial.chebyshev.chebval(2.0 * soc - 1.0, ocv_coeffs)


# --------------------------------------------------------------------------
# Core ECM simulation
# --------------------------------------------------------------------------

def _simulate_ecm(
    I:          np.ndarray,
    soc:        np.ndarray,
    ocv_coeffs: np.ndarray,
    R0:         float,
    R1:         float,
    C1:         float,
    dt:         float = DT_SECONDS,
    soc_clip:   tuple[float, float] | None = None,
) -> np.ndarray:
    """Forward Euler integration of ECM-1RC on a batch of windows.

    Parameters
    ----------
    I, soc : (B, L) float64 arrays.  I > 0 = discharge (classical convention).
    ocv_coeffs : Chebyshev OCV coefficients.
    R0, R1, C1 : ECM parameters.
    dt : integration timestep [s].
    soc_clip : if provided as (soc_min, soc_max), the SoC argument to
        the OCV polynomial is clipped to this range.  The V1 dynamics
        (current-driven RC response) are unaffected.

    Returns
    -------
    V_hat : (B, L) float64 array, predicted terminal voltage [V].
    """
    B, L   = I.shape
    V1     = np.zeros(B, dtype=np.float64)
    V_hat  = np.empty((B, L), dtype=np.float64)

    soc_for_ocv = np.clip(soc, *soc_clip) if soc_clip is not None else soc
    ocv = _eval_ocv(ocv_coeffs, soc_for_ocv)   # (B, L)

    inv_tau = 1.0 / (R1 * C1)
    inv_C1  = 1.0 / C1

    for k in range(L):
        V_hat[:, k] = ocv[:, k] - R0 * I[:, k] - V1
        V1 = V1 + dt * (-V1 * inv_tau + I[:, k] * inv_C1)

    return V_hat


# --------------------------------------------------------------------------
# Parameter packing / unpacking
# --------------------------------------------------------------------------
# Layout of the optimisation vector theta (length N_OCV_COEFFS + 3):
#   theta[0 : N_OCV_COEFFS]  = OCV Chebyshev coefficients
#   theta[N_OCV_COEFFS + 0]  = log10(R0)
#   theta[N_OCV_COEFFS + 1]  = log10(R1)
#   theta[N_OCV_COEFFS + 2]  = log10(C1)

def _pack(
    ocv_coeffs: np.ndarray, R0: float, R1: float, C1: float
) -> np.ndarray:
    theta = np.empty(N_OCV_COEFFS + 3, dtype=np.float64)
    theta[:N_OCV_COEFFS]      = ocv_coeffs
    theta[N_OCV_COEFFS + 0]   = np.log10(R0)
    theta[N_OCV_COEFFS + 1]   = np.log10(R1)
    theta[N_OCV_COEFFS + 2]   = np.log10(C1)
    return theta


def _unpack(theta: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    ocv = theta[:N_OCV_COEFFS]
    R0  = 10.0 ** theta[N_OCV_COEFFS + 0]
    R1  = 10.0 ** theta[N_OCV_COEFFS + 1]
    C1  = 10.0 ** theta[N_OCV_COEFFS + 2]
    return ocv, R0, R1, C1


# --------------------------------------------------------------------------
# Identification helpers
# --------------------------------------------------------------------------

def _collect_windows(
    loader:      Iterable,
    train_stats: dict[str, float],
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect and denormalise batches from a DataLoader into arrays.

    Returns (I, soc, V_true), each shape (N_windows, L), float64,
    in physical units with the classical I > 0 = discharge convention.
    """
    I_list, soc_list, V_list = [], [], []
    n_collected = 0
    for X, Y in loader:
        I_b, soc_b = _denorm_batch(X, train_stats)
        V_b = _denorm_voltage(Y, train_stats)
        I_list.append(I_b)
        soc_list.append(soc_b)
        V_list.append(V_b)
        n_collected += I_b.shape[0]
        if max_windows is not None and n_collected >= max_windows:
            break

    I   = np.concatenate(I_list,   axis=0)
    soc = np.concatenate(soc_list, axis=0)
    V   = np.concatenate(V_list,   axis=0)

    if max_windows is not None:
        I, soc, V = I[:max_windows], soc[:max_windows], V[:max_windows]
    return I, soc, V


def _initial_guess(
    I: np.ndarray, soc: np.ndarray, V: np.ndarray,
) -> np.ndarray:
    """Compute an initial parameter vector for the optimiser.

    OCV is initialised by a direct Chebyshev fit of V vs. SoC (coarse,
    folds IR drop into OCV, but well-scaled).  RC parameters are set to
    physically reasonable values for a 18650 NCA cell at 25 C.
    """
    cheb = Chebyshev.fit(
        soc.flatten(), V.flatten(), OCV_DEGREE, domain=OCV_DOMAIN,
    )
    ocv0 = np.zeros(N_OCV_COEFFS, dtype=np.float64)
    ocv0[:min(cheb.coef.size, N_OCV_COEFFS)] = cheb.coef[:N_OCV_COEFFS]
    return _pack(ocv0, R0=0.030, R1=0.020, C1=1000.0)


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(N_OCV_COEFFS + 3, -np.inf)
    hi = np.full(N_OCV_COEFFS + 3,  np.inf)
    lo[N_OCV_COEFFS + 0], hi[N_OCV_COEFFS + 0] = LOG10_R0_LO, LOG10_R0_HI
    lo[N_OCV_COEFFS + 1], hi[N_OCV_COEFFS + 1] = LOG10_R1_LO, LOG10_R1_HI
    lo[N_OCV_COEFFS + 2], hi[N_OCV_COEFFS + 2] = LOG10_C1_LO, LOG10_C1_HI
    return lo, hi


def _check_bounds(theta: np.ndarray) -> list[str]:
    """Return a list of parameters that landed on their bounds."""
    hits = []
    for name, idx, lo, hi in [
        ("R0", N_OCV_COEFFS + 0, LOG10_R0_LO, LOG10_R0_HI),
        ("R1", N_OCV_COEFFS + 1, LOG10_R1_LO, LOG10_R1_HI),
        ("C1", N_OCV_COEFFS + 2, LOG10_C1_LO, LOG10_C1_HI),
    ]:
        if abs(theta[idx] - lo) < BOUND_PROXIMITY_TOL:
            hits.append(f"{name} at lower bound ({10**lo:g})")
        if abs(theta[idx] - hi) < BOUND_PROXIMITY_TOL:
            hits.append(f"{name} at upper bound ({10**hi:g})")
    return hits


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------

def fit_ecm_1rc(
    train_loader: Iterable,
    train_stats:  dict[str, float],
    max_windows:  int | None = 4000,
    verbose:      bool = True,
) -> ECMParams:
    """Identify ECM-1RC parameters from a DataLoader of training windows.

    Minimises the sum of squared residuals between the ECM prediction
    and the observed terminal voltage across all training windows,
    subject to physical bounds on R0, R1, C1 and a Tikhonov smoothness
    prior on the OCV Chebyshev coefficients.

    Parameters
    ----------
    train_loader : DataLoader
        Yields (X_norm, V_norm) batches.  Uses at most max_windows
        windows from the start of the loader.
    train_stats : dict
        Normalisation stats dict from data.compute_norm_stats.  Must
        contain I_mean, I_std, V_mean, V_std, SoC_mean, SoC_std.
    max_windows : int or None
        Maximum number of windows to use for identification.  If None,
        all windows in the loader are used.  Default 4000 keeps the
        residual vector at ~4M elements, which scipy TRF handles well.
    verbose : bool
        If True, print identification diagnostics.

    Returns
    -------
    ECMParams with identified parameters and diagnostics.

    Raises
    ------
    RuntimeError
        If any of R0, R1, C1 lands on its bound after optimisation
        (degenerate fit; see module docstring, decision 5).
    """
    I, soc, V_true = _collect_windows(train_loader, train_stats, max_windows)
    B, L = I.shape

    if verbose:
        print(f"  identification set: {B} windows x {L} samples "
              f"= {B*L:,} residuals")
        print(f"  SoC range : [{soc.min():.3f}, {soc.max():.3f}]")
        print(f"  V   range : [{V_true.min():.3f}, {V_true.max():.3f}] V")
        print(f"  I   range : [{I.min():.3f}, {I.max():.3f}] A "
              f"(mean {I.mean():+.3f})")

    theta0  = _initial_guess(I, soc, V_true)
    lo, hi  = _bounds()

    # Per-coefficient Tikhonov weights: constant term (k=0) unpenalised;
    # higher-order coefficients penalised geometrically.
    reg_weights = OCV_REG_BASE * np.array(
        [0.0] + [OCV_REG_GROWTH ** k for k in range(1, N_OCV_COEFFS)]
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        ocv, R0, R1, C1 = _unpack(theta)
        V_hat = _simulate_ecm(I, soc, ocv, R0, R1, C1)
        return np.concatenate([
            (V_hat - V_true).flatten(),
            reg_weights * ocv,            # Tikhonov penalty residuals
        ])

    if verbose:
        r0 = residuals(theta0)
        print(f"  initial RMSE = {np.sqrt((r0**2).mean())*1000:.2f} mV "
              f"(data-only, before optimisation)")
        print("  solving nonlinear least squares (TRF, x_scale='jac') ...")

    result = least_squares(
        residuals, theta0,
        bounds=(lo, hi),
        method="trf",
        x_scale="jac",
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=500,
        verbose=1 if verbose else 0,
    )

    hits = _check_bounds(result.x)
    if hits:
        raise RuntimeError(
            "ECM identification produced a degenerate fit. "
            "Parameters at bounds:\n  " + "\n  ".join(hits) + "\n"
            "This typically means the optimizer is absorbing the signal "
            "into OCV and dropping the RC dynamics.  Consider widening "
            "the RC bounds or increasing max_windows."
        )

    ocv_fit, R0_fit, R1_fit, C1_fit = _unpack(result.x)
    # Strip Tikhonov residuals from result.fun for data-only RMSE.
    n_data       = B * L
    data_res     = result.fun[:n_data]
    rmse         = float(np.sqrt(np.mean(data_res ** 2)))

    if verbose:
        print(f"  identified parameters:")
        print(f"    R0   = {R0_fit*1000:8.3f} mOhm")
        print(f"    R1   = {R1_fit*1000:8.3f} mOhm")
        print(f"    C1   = {C1_fit:10.2f} F")
        print(f"    tau1 = {R1_fit*C1_fit:8.3f} s")
        ocv_pts = _eval_ocv(ocv_fit, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        print(f"    OCV @ SoC [0.1,0.3,0.5,0.7,0.9] = "
              f"{', '.join(f'{v:.3f}' for v in ocv_pts)} V")
        print(f"  identification RMSE = {rmse*1000:.2f} mV")
        print(f"  success = {result.success},  nfev = {result.nfev}")

    return ECMParams(
        ocv_coeffs=ocv_fit,
        R0=R0_fit, R1=R1_fit, C1=C1_fit,
        soc_train_min=float(soc.min()),
        soc_train_max=float(soc.max()),
        identification_rmse_V=rmse,
        identification_n_windows=int(B),
        identification_success=bool(result.success),
    )


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict_ecm(
    params:      ECMParams,
    loader:      Iterable,
    train_stats: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the ECM over a DataLoader and return normalised tensors.

    Output format matches training.predict() exactly: (preds, targs)
    are normalised float32 tensors of shape (N_windows, L), so
    extended_metrics() in the experiment scripts can be called without
    modification.

    SoC is clipped to the range [params.soc_train_min,
    params.soc_train_max] before evaluating OCV, preventing Chebyshev
    polynomial extrapolation from producing unphysical voltages outside
    the training SoC range (see module docstring, decision 6).

    Parameters
    ----------
    params      : ECMParams  Identified model parameters.
    loader      : DataLoader  Yields (X_norm, V_norm) batches.
    train_stats : dict  Normalisation stats from data.compute_norm_stats.

    Returns
    -------
    (preds, targs) : (N_windows, L) float32 tensors on CPU, normalised.
    """
    preds_all, targs_all = [], []
    soc_clip = (params.soc_train_min, params.soc_train_max)

    for X, Y in loader:
        I_b, soc_b = _denorm_batch(X, train_stats)
        V_hat = _simulate_ecm(
            I_b, soc_b,
            params.ocv_coeffs, params.R0, params.R1, params.C1,
            soc_clip=soc_clip,
        )
        preds_all.append(_norm_voltage(V_hat, train_stats))
        targs_all.append(Y.detach().cpu().numpy().astype(np.float64))

    preds = np.concatenate(preds_all, axis=0)
    targs = np.concatenate(targs_all, axis=0)
    return torch.from_numpy(preds).float(), torch.from_numpy(targs).float()