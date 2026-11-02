"""
ECM-2RC baseline — extension of wno_battery.ecm to a second-order Thevenin
circuit, added to answer Reviewer 3 (JES), comment #2: "does ECM-UDE
actually outperform a conventional second-order (2RC) equivalent-circuit
model, i.e. is the neural correction doing more than simply raising the
circuit order would?"

Model definition
----------------
Second-order Thevenin circuit (same conventions as ecm.py, I > 0 =
discharge):

    V(t)    = OCV(SoC(t)) - R0*I(t) - V1(t) - V2(t)
    dV1/dt  = -V1(t)/(R1*C1) + I(t)/C1      (fast polarization branch)
    dV2/dt  = -V2(t)/(R2*C2) + I(t)/C2      (slow polarization branch)

OCV(SoC): same degree-5 Chebyshev basis on [0,1] as ecm.py. Integration:
forward Euler at dt=0.1s, V1(0)=V2(0)=0 per window -- identical protocol
to ECM-1RC and ECM-UDE, so results are directly comparable.

ECM-2RC has 11 scalar parameters (6 OCV coefficients + R0 + R1,C1 + R2,C2),
two more than ECM-1RC's 9 and roughly an order of magnitude fewer than
ECM-UDE's ~1260 (which are dominated by the neural correction f_theta;
the physical branch of ECM-UDE itself only has the same 9 as ECM-1RC).

Identification: nonlinear least squares (TRF, x_scale='jac'), same
regularisation and degeneracy-guard philosophy as fit_ecm_1rc. The two
RC branches are label-switch-symmetric (swapping (R1,C1)<->(R2,C2) leaves
the circuit unchanged), so the initial guess deliberately separates the
two time constants (a fast ~2s branch and a slow ~40s branch, typical of
the charge-transfer / diffusion split reported for 18650 NCA cells) to
seed the optimiser away from that symmetry; the identified tau1, tau2 are
reported in the summary so the fitted branches' physical plausibility can
be checked post hoc.

This module deliberately mirrors ecm.py's structure and public-API shape
(ECMParams2RC / fit_ecm_2rc / predict_ecm_2rc) so callers (exp_c4_*.py)
can reuse extended_metrics/reconstruct_validation_series from exp1_ude.py
unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import torch
from numpy.polynomial import Chebyshev
from scipy.optimize import least_squares

from ecm import (
    DT_SECONDS, OCV_DEGREE, N_OCV_COEFFS, OCV_DOMAIN,
    SOC_MU, SOC_SIGMA,
    OCV_REG_BASE, OCV_REG_GROWTH,
    BOUND_PROXIMITY_TOL,
    _denorm_batch, _denorm_voltage, _norm_voltage, _eval_ocv,
    _collect_windows,
)

# --------------------------------------------------------------------------
# Bounds (same physical range as ecm.py's single branch, reused for both)
# --------------------------------------------------------------------------

LOG10_R0_LO, LOG10_R0_HI = np.log10(1e-3), np.log10(0.5)
LOG10_R1_LO, LOG10_R1_HI = np.log10(1e-3), np.log10(0.5)
LOG10_C1_LO, LOG10_C1_HI = np.log10(1.0),  np.log10(1e5)
LOG10_R2_LO, LOG10_R2_HI = np.log10(1e-3), np.log10(0.5)
LOG10_C2_LO, LOG10_C2_HI = np.log10(1.0),  np.log10(1e6)   # slow branch: allow larger C


# --------------------------------------------------------------------------
# ECMParams2RC
# --------------------------------------------------------------------------

@dataclass
class ECMParams2RC:
    ocv_coeffs:              np.ndarray
    R0:                      float
    R1:                      float
    C1:                      float
    R2:                      float
    C2:                      float
    soc_train_min:           float = 0.0
    soc_train_max:           float = 1.0
    identification_rmse_V:   float = 0.0
    identification_n_windows: int  = 0
    identification_success:  bool  = True

    def chebyshev_polynomial(self) -> Chebyshev:
        return Chebyshev(self.ocv_coeffs, domain=OCV_DOMAIN)

    def ocv_of(self, soc: np.ndarray) -> np.ndarray:
        return self.chebyshev_polynomial()(soc)

    def tau1(self) -> float:
        return self.R1 * self.C1

    def tau2(self) -> float:
        return self.R2 * self.C2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ocv_coeffs"] = list(self.ocv_coeffs)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ECMParams2RC":
        d = dict(d)
        d["ocv_coeffs"] = np.asarray(d["ocv_coeffs"], dtype=np.float64)
        return cls(**d)


# --------------------------------------------------------------------------
# Core simulation
# --------------------------------------------------------------------------

def _simulate_ecm_2rc(
    I: np.ndarray, soc: np.ndarray, ocv_coeffs: np.ndarray,
    R0: float, R1: float, C1: float, R2: float, C2: float,
    dt: float = DT_SECONDS,
    soc_clip: tuple[float, float] | None = None,
) -> np.ndarray:
    B, L = I.shape
    V1 = np.zeros(B, dtype=np.float64)
    V2 = np.zeros(B, dtype=np.float64)
    V_hat = np.empty((B, L), dtype=np.float64)

    soc_for_ocv = np.clip(soc, *soc_clip) if soc_clip is not None else soc
    ocv = _eval_ocv(ocv_coeffs, soc_for_ocv)

    inv_tau1, inv_C1 = 1.0 / (R1 * C1), 1.0 / C1
    inv_tau2, inv_C2 = 1.0 / (R2 * C2), 1.0 / C2

    for k in range(L):
        V_hat[:, k] = ocv[:, k] - R0 * I[:, k] - V1 - V2
        V1 = V1 + dt * (-V1 * inv_tau1 + I[:, k] * inv_C1)
        V2 = V2 + dt * (-V2 * inv_tau2 + I[:, k] * inv_C2)

    return V_hat


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------
# theta layout: [OCV (N_OCV_COEFFS)] [log10 R0] [log10 R1] [log10 C1]
#               [log10 R2] [log10 C2]

def _pack(ocv_coeffs, R0, R1, C1, R2, C2) -> np.ndarray:
    theta = np.empty(N_OCV_COEFFS + 5, dtype=np.float64)
    theta[:N_OCV_COEFFS] = ocv_coeffs
    theta[N_OCV_COEFFS + 0] = np.log10(R0)
    theta[N_OCV_COEFFS + 1] = np.log10(R1)
    theta[N_OCV_COEFFS + 2] = np.log10(C1)
    theta[N_OCV_COEFFS + 3] = np.log10(R2)
    theta[N_OCV_COEFFS + 4] = np.log10(C2)
    return theta


def _unpack(theta: np.ndarray):
    ocv = theta[:N_OCV_COEFFS]
    R0 = 10.0 ** theta[N_OCV_COEFFS + 0]
    R1 = 10.0 ** theta[N_OCV_COEFFS + 1]
    C1 = 10.0 ** theta[N_OCV_COEFFS + 2]
    R2 = 10.0 ** theta[N_OCV_COEFFS + 3]
    C2 = 10.0 ** theta[N_OCV_COEFFS + 4]
    return ocv, R0, R1, C1, R2, C2


def _bounds():
    lo = np.full(N_OCV_COEFFS + 5, -np.inf)
    hi = np.full(N_OCV_COEFFS + 5, np.inf)
    lo[N_OCV_COEFFS + 0], hi[N_OCV_COEFFS + 0] = LOG10_R0_LO, LOG10_R0_HI
    lo[N_OCV_COEFFS + 1], hi[N_OCV_COEFFS + 1] = LOG10_R1_LO, LOG10_R1_HI
    lo[N_OCV_COEFFS + 2], hi[N_OCV_COEFFS + 2] = LOG10_C1_LO, LOG10_C1_HI
    lo[N_OCV_COEFFS + 3], hi[N_OCV_COEFFS + 3] = LOG10_R2_LO, LOG10_R2_HI
    lo[N_OCV_COEFFS + 4], hi[N_OCV_COEFFS + 4] = LOG10_C2_LO, LOG10_C2_HI
    return lo, hi


def _check_bounds(theta: np.ndarray) -> list[str]:
    hits = []
    for name, idx, lo, hi in [
        ("R0", N_OCV_COEFFS + 0, LOG10_R0_LO, LOG10_R0_HI),
        ("R1", N_OCV_COEFFS + 1, LOG10_R1_LO, LOG10_R1_HI),
        ("C1", N_OCV_COEFFS + 2, LOG10_C1_LO, LOG10_C1_HI),
        ("R2", N_OCV_COEFFS + 3, LOG10_R2_LO, LOG10_R2_HI),
        ("C2", N_OCV_COEFFS + 4, LOG10_C2_LO, LOG10_C2_HI),
    ]:
        if abs(theta[idx] - lo) < BOUND_PROXIMITY_TOL:
            hits.append(f"{name} at lower bound ({10**lo:g})")
        if abs(theta[idx] - hi) < BOUND_PROXIMITY_TOL:
            hits.append(f"{name} at upper bound ({10**hi:g})")
    return hits


def _initial_guess(I: np.ndarray, soc: np.ndarray, V: np.ndarray) -> np.ndarray:
    cheb = Chebyshev.fit(soc.flatten(), V.flatten(), OCV_DEGREE, domain=OCV_DOMAIN)
    ocv0 = np.zeros(N_OCV_COEFFS, dtype=np.float64)
    ocv0[:min(cheb.coef.size, N_OCV_COEFFS)] = cheb.coef[:N_OCV_COEFFS]
    # Deliberately separate the two time constants (fast ~2s vs slow ~40s
    # branch) to seed the optimiser away from the R1C1<->R2C2 label-switch
    # symmetry; see module docstring.
    return _pack(ocv0, R0=0.030, R1=0.010, C1=200.0, R2=0.020, C2=2000.0)


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------

def fit_ecm_2rc(
    train_loader: Iterable,
    train_stats: dict[str, float],
    max_windows: int | None = 4000,
    verbose: bool = True,
    dt: float = DT_SECONDS,
) -> ECMParams2RC:
    I, soc, V_true = _collect_windows(train_loader, train_stats, max_windows)
    B, L = I.shape

    if verbose:
        print(f"  identification set: {B} windows x {L} samples = {B*L:,} residuals")

    theta0 = _initial_guess(I, soc, V_true)
    lo, hi = _bounds()

    reg_weights = OCV_REG_BASE * np.array(
        [0.0] + [OCV_REG_GROWTH ** k for k in range(1, N_OCV_COEFFS)]
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        ocv, R0, R1, C1, R2, C2 = _unpack(theta)
        V_hat = _simulate_ecm_2rc(I, soc, ocv, R0, R1, C1, R2, C2, dt=dt)
        return np.concatenate([
            (V_hat - V_true).flatten(),
            reg_weights * ocv,
        ])

    if verbose:
        r0 = residuals(theta0)
        print(f"  initial RMSE = {np.sqrt((r0**2).mean())*1000:.2f} mV (before optimisation)")
        print("  solving nonlinear least squares (TRF, x_scale='jac') ...")

    result = least_squares(
        residuals, theta0, bounds=(lo, hi), method="trf", x_scale="jac",
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=800,
        verbose=1 if verbose else 0,
    )

    hits = _check_bounds(result.x)
    if hits:
        raise RuntimeError(
            "ECM-2RC identification produced a degenerate fit. "
            "Parameters at bounds:\n  " + "\n  ".join(hits)
        )

    ocv_fit, R0_fit, R1_fit, C1_fit, R2_fit, C2_fit = _unpack(result.x)
    n_data = B * L
    data_res = result.fun[:n_data]
    rmse = float(np.sqrt(np.mean(data_res ** 2)))

    # Canonicalise branch order so tau1 < tau2 (fast, then slow) --
    # the fit is label-switch symmetric, so this only affects reporting.
    if R1_fit * C1_fit > R2_fit * C2_fit:
        R1_fit, C1_fit, R2_fit, C2_fit = R2_fit, C2_fit, R1_fit, C1_fit

    if verbose:
        print(f"  identified parameters:")
        print(f"    R0   = {R0_fit*1000:8.3f} mOhm")
        print(f"    R1   = {R1_fit*1000:8.3f} mOhm   C1 = {C1_fit:10.2f} F   tau1 = {R1_fit*C1_fit:8.3f} s")
        print(f"    R2   = {R2_fit*1000:8.3f} mOhm   C2 = {C2_fit:10.2f} F   tau2 = {R2_fit*C2_fit:8.3f} s")
        print(f"  identification RMSE = {rmse*1000:.2f} mV")
        print(f"  success = {result.success},  nfev = {result.nfev}")

    return ECMParams2RC(
        ocv_coeffs=ocv_fit,
        R0=R0_fit, R1=R1_fit, C1=C1_fit, R2=R2_fit, C2=C2_fit,
        soc_train_min=float(soc.min()), soc_train_max=float(soc.max()),
        identification_rmse_V=rmse, identification_n_windows=int(B),
        identification_success=bool(result.success),
    )


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict_ecm_2rc(
    params: ECMParams2RC,
    loader: Iterable,
    train_stats: dict[str, float],
    dt: float = DT_SECONDS,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds_all, targs_all = [], []
    soc_clip = (0.0, 1.0)   # same unclamped policy as ECM-1RC/ECM-UDE

    for X, Y in loader:
        I_b, soc_b = _denorm_batch(X, train_stats)
        V_hat = _simulate_ecm_2rc(
            I_b, soc_b,
            params.ocv_coeffs, params.R0, params.R1, params.C1,
            params.R2, params.C2,
            dt=dt,
            soc_clip=soc_clip,
        )
        preds_all.append(_norm_voltage(V_hat, train_stats))
        targs_all.append(Y.detach().cpu().numpy().astype(np.float64))

    preds = np.concatenate(preds_all, axis=0)
    targs = np.concatenate(targs_all, axis=0)
    return torch.from_numpy(preds).float(), torch.from_numpy(targs).float()
