"""
Universal Differential Equation for Battery Voltage Estimation
==============================================================
wno_battery.ude

Paper 1 core module.  Implements a first-order Thevenin ECM whose RC
dynamics are augmented by a learnable neural correction term:

    V(t)   = OCV(SoC(t)) - R0 * I(t) - V1(t)
    dV1/dt = -V1/(R1*C1) + I/C1 + f_theta(V1, I, SoC, T)

The neural term f_theta is a small MLP that absorbs modelling errors
(nonlinear impedance, hysteresis, thermal coupling) that the ECM alone
cannot capture.  The ODE is integrated with torchdiffeq (adaptive-step
Dormand--Prince or Tsit5) and trained end-to-end via adjoint
sensitivity.

Key design decisions
--------------------
1. SoC is NOT an input to the model -- it is an OUTPUT.  The model
   receives (I, T, V_measured) and estimates SoC via Coulomb counting
   with a learnable efficiency factor.  This is the fundamental
   difference from the Paper 2 pipeline where SoC was a given feature.

2. OCV is parameterised as a Chebyshev polynomial (reusing ecm.py's
   conventions) but with torch Parameters, so it is end-to-end
   differentiable.

3. The integration uses the adjoint method (O(1) memory in depth)
   rather than backprop-through-the-solver, which is essential for
   long windows (1024 steps at 10 Hz).

4. Physical constraints are enforced via softplus reparameterisation:
   R0, R1, C1 are stored as unconstrained reals and mapped to
   positive values via softplus + floor.

Public API
----------
  ECMUDEConfig       Dataclass of architecture hyperparameters.
  ECMUDE             nn.Module implementing the UDE.
  build_ecm_ude()    Factory with sensible defaults.

Dependencies
------------
  torch, torchdiffeq
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ECMUDEConfig:
    """Hyperparameters for the ECM-UDE model."""

    # Neural correction MLP
    nn_hidden: int = 32          # hidden units per layer
    nn_layers: int = 2           # number of hidden layers
    nn_activation: str = "gelu"  # activation function

    # OCV polynomial
    ocv_degree: int = 5          # Chebyshev degree

    # Integration
    solver: str = "dopri5"       # torchdiffeq solver
    rtol: float = 1e-5           # relative tolerance
    atol: float = 1e-6           # absolute tolerance
    dt_seconds: float = 0.1      # sampling period [s]

    # Coulomb counting
    nominal_capacity_Ah: float = 2.9  # Panasonic 18650PF

    # Physical parameter initialisation (will be learned)
    R0_init: float = 0.030       # [Ohm]
    R1_init: float = 0.020       # [Ohm]
    C1_init: float = 1000.0      # [F]

    # Minimum physical values (softplus floor)
    R_min: float = 1e-4          # [Ohm]
    C_min: float = 1.0           # [F]


# ─────────────────────────────────────────────────────────────────────
# Neural correction network
# ─────────────────────────────────────────────────────────────────────

class NeuralCorrection(nn.Module):
    """Small MLP: (V1, I, SoC, T) -> scalar correction to dV1/dt.

    The output is scaled by a learnable gain initialised near zero so
    that the UDE starts close to the pure ECM at the beginning of
    training.  This warm-start strategy avoids destabilising the ODE
    integration in early epochs.
    """

    def __init__(self, hidden: int = 32, n_layers: int = 2,
                 activation: str = "gelu"):
        super().__init__()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers = [nn.Linear(4, hidden), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Output gain initialised near zero for warm start
        self.gain = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, V1: torch.Tensor, I: torch.Tensor,
                soc: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        All inputs: (batch,) tensors.
        Returns: (batch,) correction term.
        """
        x = torch.stack([V1, I, soc, T], dim=-1)  # (B, 4)
        return self.gain * self.net(x).squeeze(-1)  # (B,)


# ─────────────────────────────────────────────────────────────────────
# Chebyshev OCV in torch (differentiable)
# ─────────────────────────────────────────────────────────────────────

class ChebyshevOCV(nn.Module):
    """Differentiable Chebyshev polynomial OCV(SoC) on [0, 1].

    Coefficients are nn.Parameters trained end-to-end.  Initialisation
    from a pre-identified ECMParams is supported via from_numpy().
    """

    def __init__(self, degree: int = 5):
        super().__init__()
        self.degree = degree
        # Initialise to a rough NCA OCV curve: ~3.0 V at SoC=0, ~4.2 V at SoC=1
        coeffs = torch.zeros(degree + 1, dtype=torch.float32)
        coeffs[0] = 3.6   # mean voltage
        coeffs[1] = 0.6   # linear trend
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, soc: torch.Tensor) -> torch.Tensor:
        """Evaluate OCV at given SoC values.

        Uses the Chebyshev recurrence in the mapped domain [-1, 1].
        """
        # Map [0, 1] -> [-1, 1]
        x = 2.0 * soc.clamp(0.0, 1.0) - 1.0

        # Clenshaw evaluation (numerically stable)
        if self.degree == 0:
            return self.coeffs[0] * torch.ones_like(x)

        b_k1 = torch.zeros_like(x)  # b_{k+1}
        b_k2 = torch.zeros_like(x)  # b_{k+2}

        for k in range(self.degree, 0, -1):
            b_new = self.coeffs[k] + 2.0 * x * b_k1 - b_k2
            b_k2 = b_k1
            b_k1 = b_new

        return self.coeffs[0] + x * b_k1 - b_k2

    def from_numpy(self, np_coeffs):
        """Initialise from numpy Chebyshev coefficients (e.g. from ecm.py)."""
        with torch.no_grad():
            n = min(len(np_coeffs), self.degree + 1)
            self.coeffs[:n] = torch.tensor(np_coeffs[:n], dtype=torch.float32)
        return self


# ─────────────────────────────────────────────────────────────────────
# Positive-constrained parameters
# ─────────────────────────────────────────────────────────────────────

def _to_unconstrained(value: float, floor: float) -> float:
    """Inverse softplus: find x such that softplus(x) + floor = value."""
    shifted = value - floor
    if shifted <= 0:
        shifted = 1e-6
    # For large shifted, log(exp(s)-1) ≈ s (avoid overflow)
    if shifted > 20.0:
        return shifted
    return math.log(math.exp(shifted) - 1.0)


def _to_positive(raw: torch.Tensor, floor: float) -> torch.Tensor:
    """Softplus + floor: guarantees output > floor."""
    return F.softplus(raw) + floor


def _coerce_stats(stats: dict) -> dict[str, float]:
    """Convert stats to Python floats to avoid float64 tensor promotion."""
    return {k: float(v) for k, v in dict(stats).items()}


# ─────────────────────────────────────────────────────────────────────
# ODE right-hand side
# ─────────────────────────────────────────────────────────────────────

class ECMUDEFunc(nn.Module):
    """ODE function for torchdiffeq.

    State vector y = [V1, SoC], shape (batch, 2).

    The current I(t) and temperature T(t) are provided as
    piecewise-constant interpolants via set_driving_signals().

    Sign convention: I > 0 = discharge (classical ECM convention).
    The sign flip from the Kollmeyer convention (I > 0 = charge) is
    done at the ECMUDE.forward() boundary.
    """

    def __init__(self, cfg: ECMUDEConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable physical parameters (unconstrained)
        self._R0_raw = nn.Parameter(torch.tensor(
            _to_unconstrained(cfg.R0_init, cfg.R_min), dtype=torch.float32))
        self._R1_raw = nn.Parameter(torch.tensor(
            _to_unconstrained(cfg.R1_init, cfg.R_min), dtype=torch.float32))
        self._C1_raw = nn.Parameter(torch.tensor(
            _to_unconstrained(cfg.C1_init, cfg.C_min), dtype=torch.float32))

        # Neural correction
        self.nn_correction = NeuralCorrection(
            hidden=cfg.nn_hidden,
            n_layers=cfg.nn_layers,
            activation=cfg.nn_activation,
        )

        # Coulomb counting efficiency (learnable, ~1.0)
        self._eta_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # sigmoid -> ~0.5, but we use 1+tanh/10

        # Driving signals (set before each forward pass)
        self._I_interp = None      # (B, L) current
        self._T_interp = None      # (B, L) temperature
        self._dt = cfg.dt_seconds
        self._L = 0

    # ── Physical parameter accessors ──

    @property
    def R0(self) -> torch.Tensor:
        return _to_positive(self._R0_raw, self.cfg.R_min)

    @property
    def R1(self) -> torch.Tensor:
        return _to_positive(self._R1_raw, self.cfg.R_min)

    @property
    def C1(self) -> torch.Tensor:
        return _to_positive(self._C1_raw, self.cfg.C_min)

    @property
    def eta(self) -> torch.Tensor:
        """Coulomb efficiency in (0.9, 1.1)."""
        return 1.0 + 0.1 * torch.tanh(self._eta_raw)

    # ── Driving signal management ──

    def set_driving_signals(self, I: torch.Tensor, T: torch.Tensor):
        """Register the current and temperature profiles for this batch.

        I, T: (batch, length) in physical units.
        I > 0 = discharge (classical convention).
        """
        self._I_interp = I
        self._T_interp = T
        self._L = I.shape[1]

    def _get_driving(self, t: torch.Tensor):
        """Piecewise-constant interpolation of I(t), T(t).

        t is a scalar (the integration time in seconds).
        Returns I, T each of shape (batch,).
        """
        k = int(t.item() / self._dt)
        k = min(k, self._L - 1)
        return self._I_interp[:, k], self._T_interp[:, k]

    # ── ODE right-hand side ──

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt = f(t, y).

        y: (batch, 2) where y[:, 0] = V1, y[:, 1] = SoC.
        Returns: (batch, 2) derivatives.
        """
        V1  = y[:, 0]
        soc = y[:, 1]

        I_t, T_t = self._get_driving(t)

        R1 = self.R1
        C1 = self.C1

        # ECM dynamics: dV1/dt = -V1/(R1*C1) + I/C1
        dV1_ecm = -V1 / (R1 * C1) + I_t / C1

        # Neural correction
        dV1_nn = self.nn_correction(V1, I_t, soc, T_t)

        # Coulomb counting: dSoC/dt = -eta * I / (Q_nom * 3600)
        Q_coulombs = self.cfg.nominal_capacity_Ah * 3600.0
        dSoC = -self.eta * I_t / Q_coulombs

        dy = torch.stack([dV1_ecm + dV1_nn, dSoC], dim=-1)
        return dy


# ─────────────────────────────────────────────────────────────────────
# Main ECMUDE model
# ─────────────────────────────────────────────────────────────────────

class ECMUDE(nn.Module):
    """ECM-1RC + Universal Differential Equation.

    Input:  (batch, length, n_features) where features = (I_norm, T_norm, SoC_norm)
            NOTE: SoC_norm is used ONLY for initialisation of the ODE state.
            During integration, SoC is propagated by Coulomb counting.

    Output: (batch, length) predicted terminal voltage in normalised space.

    The model internally:
      1. Denormalises I, T, SoC using provided stats.
      2. Flips I sign to classical convention (I > 0 = discharge).
      3. Integrates the UDE over the window.
      4. Computes V(t) = OCV(SoC(t)) - R0*I(t) - V1(t).
      5. Normalises V back to the pipeline's z-score space.
    """

    def __init__(self, cfg: Optional[ECMUDEConfig] = None,
                 stats: Optional[dict] = None):
        super().__init__()
        if cfg is None:
            cfg = ECMUDEConfig()
        self.cfg = cfg

        self.ocv = ChebyshevOCV(degree=cfg.ocv_degree)
        self.ode_func = ECMUDEFunc(cfg)

        # Normalisation stats (set before training)
        self._stats = stats or {
            "I_mean": 0.0, "I_std": 1.0,
            "V_mean": 0.0, "V_std": 1.0,
            "T_mean": 0.0, "T_std": 25.0,
            "SoC_mean": 0.5, "SoC_std": 0.3,
        }
        self._stats = _coerce_stats(self._stats)

    def set_stats(self, stats: dict):
        """Set normalisation statistics (call before training)."""
        self._stats = _coerce_stats(stats)

    @property
    def R0(self) -> torch.Tensor:
        return _to_positive(self.ode_func._R0_raw, self.cfg.R_min)

    def _denorm(self, x: torch.Tensor):
        """Denormalise input features to physical units.

        Returns I_phys (discharge positive), T_phys [degC], SoC [0,1].
        """
        s = self._stats
        I_std = x.new_tensor(s["I_std"])
        I_mean = x.new_tensor(s["I_mean"])
        T_std = x.new_tensor(s["T_std"])
        T_mean = x.new_tensor(s["T_mean"])
        soc_std = x.new_tensor(s["SoC_std"])
        soc_mean = x.new_tensor(s["SoC_mean"])

        I_pipeline = x[..., 0] * I_std + I_mean
        I_ecm = -I_pipeline   # flip: Kollmeyer charge-positive -> ECM discharge-positive
        T = x[..., 1] * T_std + T_mean
        SoC = x[..., 2] * soc_std + soc_mean
        return I_ecm, T, SoC

    def _norm_voltage(self, V_phys: torch.Tensor) -> torch.Tensor:
        """Normalise voltage to pipeline z-score space."""
        V_mean = V_phys.new_tensor(self._stats["V_mean"])
        V_std = V_phys.new_tensor(self._stats["V_std"])
        return (V_phys - V_mean) / V_std

    def _ode_options(self, x: torch.Tensor) -> dict:
        """Return solver-specific odeint options.

        Adaptive solvers benefit from explicit float32 time dtype on MPS.
        Fixed-step RK4 does not accept max_num_steps/dtype kwargs.
        """
        if self.cfg.solver.lower() == "rk4":
            return {}
        return {"max_num_steps": 4000, "dtype": x.dtype}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 3) normalised features.
        Returns: (B, L) normalised voltage prediction.
        """
        B, L, _ = x.shape
        I_phys, T_phys, SoC_phys = self._denorm(x)

        # Set driving signals for the ODE
        self.ode_func.set_driving_signals(I_phys, T_phys)

        # Initial state: V1(0) = 0, SoC(0) from first timestep
        y0 = torch.zeros(B, 2, device=x.device, dtype=x.dtype)
        y0[:, 1] = SoC_phys[:, 0].clamp(0.0, 1.0)

        # Time grid
        t_span = torch.linspace(
            0.0, (L - 1) * self.cfg.dt_seconds, L,
            device=x.device, dtype=x.dtype,
        )

        # Integrate
        # y_traj: (L, B, 2) with adjoint method
        y_traj = odeint(
            self.ode_func, y0, t_span,
            method=self.cfg.solver,
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
            options=self._ode_options(x),
        )

        V1_traj  = y_traj[:, :, 0].T     # (B, L)
        SoC_traj = y_traj[:, :, 1].T     # (B, L)

        # Terminal voltage: V = OCV(SoC) - R0*I - V1
        ocv_traj = self.ocv(SoC_traj.clamp(0.0, 1.0))
        V_phys = ocv_traj - self.R0 * I_phys - V1_traj

        return self._norm_voltage(V_phys)

    def forward_components(self, x: torch.Tensor) -> dict:
        """Return decomposed voltage components for analysis.

        Same as forward() but returns a dict with:
          pred_norm, pred_phys, ocv, V1, SoC, I_phys
        """
        B, L, _ = x.shape
        I_phys, T_phys, SoC_phys = self._denorm(x)

        self.ode_func.set_driving_signals(I_phys, T_phys)

        y0 = torch.zeros(B, 2, device=x.device, dtype=x.dtype)
        y0[:, 1] = SoC_phys[:, 0].clamp(0.0, 1.0)

        t_span = torch.linspace(
            0.0, (L - 1) * self.cfg.dt_seconds, L,
            device=x.device, dtype=x.dtype,
        )

        y_traj = odeint(
            self.ode_func, y0, t_span,
            method=self.cfg.solver,
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
            options=self._ode_options(x),
        )

        V1_traj  = y_traj[:, :, 0].T
        SoC_traj = y_traj[:, :, 1].T

        ocv_traj = self.ocv(SoC_traj.clamp(0.0, 1.0))
        V_phys = ocv_traj - self.R0 * I_phys - V1_traj
        V_norm = self._norm_voltage(V_phys)

        return {
            "pred_norm": V_norm,
            "pred_phys": V_phys,
            "ocv": ocv_traj,
            "V1": V1_traj,
            "SoC": SoC_traj,
            "I_phys": I_phys,
            "R0": self.R0.item(),
            "R1": self.ode_func.R1.item(),
            "C1": self.ode_func.C1.item(),
            "eta": self.ode_func.eta.item(),
        }

    def physical_params_dict(self) -> dict:
        """Return current physical parameters as a plain dict."""
        return {
            "R0_Ohm": self.R0.item(),
            "R1_Ohm": self.ode_func.R1.item(),
            "C1_F": self.ode_func.C1.item(),
            "tau1_s": self.ode_func.R1.item() * self.ode_func.C1.item(),
            "eta": self.ode_func.eta.item(),
            "ocv_coeffs": self.ocv.coeffs.detach().cpu().tolist(),
        }

    def init_from_ecm_params(self, ecm_params) -> "ECMUDE":
        """Warm-start from pre-identified ECM parameters (from ecm.py).

        This initialises the OCV polynomial and RC parameters to the
        values found by scipy least_squares, giving the UDE a much
        better starting point than random initialisation.
        """
        # OCV coefficients
        self.ocv.from_numpy(ecm_params.ocv_coeffs)

        # RC parameters
        with torch.no_grad():
            self.ode_func._R0_raw.fill_(
                _to_unconstrained(ecm_params.R0, self.cfg.R_min))
            self.ode_func._R1_raw.fill_(
                _to_unconstrained(ecm_params.R1, self.cfg.R_min))
            self.ode_func._C1_raw.fill_(
                _to_unconstrained(ecm_params.C1, self.cfg.C_min))

        return self


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────

def build_ecm_ude(stats: Optional[dict] = None,
                  ecm_params=None,
                  **kwargs) -> ECMUDE:
    """Build an ECMUDE model with sensible defaults.

    Parameters
    ----------
    stats : dict, optional
        Normalisation statistics from data.compute_norm_stats().
    ecm_params : ECMParams, optional
        Pre-identified ECM parameters for warm-start.
    **kwargs
        Override any ECMUDEConfig field.

    Returns
    -------
    ECMUDE model (untrained, on CPU).
    """
    cfg = ECMUDEConfig(**kwargs)
    model = ECMUDE(cfg=cfg, stats=stats)
    if ecm_params is not None:
        model.init_from_ecm_params(ecm_params)
    return model
