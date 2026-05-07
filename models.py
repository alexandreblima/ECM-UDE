"""
Model architectures — wno_battery.models
=========================================

Overview
--------
This module provides the neural, physical, and hybrid architectures
used in the WNO battery voltage estimation pipeline, plus a factory
function and parameter counters.

All models implement a unified prediction interface:

    input  : (batch, length, n_features)   float32
    output : (batch, length)               float32

The input carries three features per timestep — current I(t), cell
temperature T_cell(t), and state-of-charge SoC(t) — and the output is
the predicted terminal voltage trajectory V(t) over the window.

Why SoC is an input feature
---------------------------
Without SoC, a short-window operator has no way to distinguish high-SoC
from low-SoC operation from current alone: the same current profile at
different SoC values produces different voltages because the OCV(SoC)
curve is the dominant term in the ECM voltage equation.  Including SoC
anchors each window in the correct region of the OCV curve, which is
essential for accurate voltage estimation across full drive cycles.

Public API
----------
  LSTMBaseline    Stacked LSTM (recurrent baseline)
  FNO1D           1D Fourier Neural Operator (Li et al., 2020)
  WNO1D           1D Wavelet Neural Operator (Tripura & Chakraborty, 2022)
  ECM1RCModel     Physics-only ECM-1RC baseline in torch
  HybridECMNO     ECM-1RC + residual neural operator
  build_model(name, **kwargs)    Factory: instantiate by string name
  count_parameters(model)        Count trainable parameters

Architecture summary
--------------------
The three neural operators share the same lifting / body / projection
skeleton:
  1. Lifting    Linear projection from n_features to width channels.
  2. Body       n_layers operator blocks.  Each block adds a global
                spectral path (FNO: Fourier; WNO: wavelet DWT) to a
                local pointwise path (1x1 Conv1d), then applies GELU.
  3. Projection Two-layer MLP (width -> 128 -> 1) per timestep.

The LSTM departs from this structure: it uses a stacked recurrent
encoder followed by a linear head.  ECM1RCModel is the explicit
physics baseline, and HybridECMNO composes ECM1RCModel with an FNO or
WNO residual learner.

Architectural inductive biases
-------------------------------
  LSTMBaseline  Recurrent; no explicit spectral prior.  Extrapolates
                smoothly to unseen current amplitudes but has limited
                capacity to represent complex spectral structure.

  FNO1D         Parameterises the integral kernel in the Fourier domain.
                Efficient for signals with globally distributed frequency
                content (HWFT, smooth voltage transients) but the fixed
                mode truncation limits how much spectral diversity it can
                absorb from multi-condition training.

  WNO1D (db4)   Decomposes the signal via DWT with Daubechies-4 (4
                vanishing moments, filter support 7 samples).  The multi-
                resolution representation captures both low-frequency
                trends (OCV drift) and high-frequency transients (RC
                polarisation spikes, US06 pulses) simultaneously.
                db4 is the reference configuration used in Exp. 1-4.

  WNO1D (Haar)  Same as WNO1D but with the Haar wavelet (db1, 1 vanishing
                moment, support 1 sample).  Haar is the limiting case of
                the Daubechies family: minimal regularity, maximum
                temporal localisation.  Used as an ablation of wavelet
                regularity across all five experiments.  Particularly
                informative at short window lengths (L=256 in Exp. 5)
                where the db4 filter support of 7 samples occupies nearly
                half of the coarsest-level subband (16 samples).

build_model and WNO aliases
---------------------------
build_model("wno_haar", **kwargs) instantiates WNO1D with wavelet="haar".
This avoids duplicating the WNO1D class while allowing the experiment
scripts to treat wno_haar as a first-class model name in MODEL_CONFIGS
dicts and summary JSON files.  Likewise, build_model("wno_pu", **kwargs)
instantiates WNO1D with the default db4 wavelet and paraunitary=True.

Dependencies
------------
  torch            Core tensors and nn.
  pytorch_wavelets DWT1DForward / DWT1DInverse required by WaveletConv1d.
                   Install: pip install pytorch-wavelets
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from wno_battery.ecm import DT_SECONDS, ECMParams, N_OCV_COEFFS
except ModuleNotFoundError as exc:
    if exc.name != "wno_battery":
        raise
    from ecm import DT_SECONDS, ECMParams, N_OCV_COEFFS


# =============================================================================
# LSTM BASELINE
# =============================================================================

class LSTMBaseline(nn.Module):
    """Stacked LSTM mapping (I, T, SoC)(t) to V(t) at each timestep.

    Recurrent baseline representing the conventional data-driven paradigm
    discussed in Lima et al. (INDUSCON 2021).  The LSTM is chosen over a
    feed-forward network because it maintains a hidden state across the
    window, giving it access to the same temporal context as the neural
    operators.

    Architecture
    ------------
    LSTM (num_layers stacked, batch_first=True)
        input_size  = n_features
        hidden_size = hidden_size
    Linear head: hidden_size -> 1 (applied pointwise in time)

    Parameters
    ----------
    n_features  : int   Number of input features per timestep (default 3).
    hidden_size : int   LSTM hidden state dimension.
    num_layers  : int   Number of stacked LSTM layers.
    dropout     : float Dropout between LSTM layers (applied only when
                        num_layers > 1 to avoid a PyTorch warning).

    Input / output
    --------------
    forward(x) : x shape (B, L, n_features) -> output shape (B, L)
    """

    def __init__(
        self,
        n_features:  int   = 3,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)       # (B, L, hidden_size)
        return self.head(h).squeeze(-1)   # (B, L)


# =============================================================================
# FOURIER NEURAL OPERATOR (FNO1D)
# =============================================================================

class SpectralConv1d(nn.Module):
    """Single spectral convolution layer for 1D FNO.

    Applies a learnable linear operator in the truncated Fourier domain:
      1. rfft along the length dimension.
      2. Multiply the lowest `modes` Fourier coefficients by a learnable
         complex weight matrix (in_channels, out_channels, modes).
      3. Zero-pad the remaining coefficients (effectively truncating the
         operator to low-frequency modes).
      4. irfft to return to the time domain.

    The truncation to `modes` frequencies is the FNO's bandwidth
    constraint: it acts as a learnable low-pass filter.  Higher-frequency
    transients are handled by the parallel pointwise Conv1d path in FNO1D.

    Parameters
    ----------
    in_channels  : int  Width of the input channel dimension.
    out_channels : int  Width of the output channel dimension.
    modes        : int  Number of Fourier modes retained (<= L // 2 + 1).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        modes:        int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes

        scale = 1.0 / (in_channels * out_channels)
        # Stored as real (in, out, modes, 2) and viewed as complex at runtime
        # to avoid issues with complex-number gradient checkpointing.
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def _complex_mul(self, x_ft: torch.Tensor) -> torch.Tensor:
        """Batched complex multiplication: (B, in, modes) x (in, out, modes)."""
        w = torch.view_as_complex(self.weight)   # (in, out, modes)
        return torch.einsum("bim,iom->bom", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, L)
        B, C, L = x.shape
        x_ft  = torch.fft.rfft(x, n=L)          # (B, in, L//2+1)
        out_ft = torch.zeros(
            B, self.out_channels, L // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = self._complex_mul(x_ft[:, :, :m])
        return torch.fft.irfft(out_ft, n=L)      # (B, out_channels, L)


class FNO1D(nn.Module):
    """1D Fourier Neural Operator.

    Architecture (lifting / body / projection):
        Lifting    : Linear(n_features, width)  applied per timestep
        Body       : n_layers FourierLayer blocks
                       each = SpectralConv1d(width, width, modes)
                             + Conv1d(width, width, 1)  [pointwise/local]
                             + GELU activation
        Projection : Linear(width, 128) + GELU + Linear(128, 1)
                     applied per timestep; output squeezed to (B, L)

    The parallel pointwise path compensates for the modes truncation:
    high-frequency content discarded by the spectral path is retained
    locally.

    Reference
    ---------
    Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar.
    "Fourier Neural Operator for Parametric PDEs", ICLR 2021.

    Parameters
    ----------
    n_features : int   Input features per timestep (default 3).
    modes      : int   Fourier modes retained per SpectralConv1d layer.
    width      : int   Channel width throughout the body.
    n_layers   : int   Number of FourierLayer blocks in the body.
    """

    def __init__(
        self,
        n_features: int = 3,
        modes:      int = 16,
        width:      int = 32,
        n_layers:   int = 4,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.width      = width
        self.n_layers   = n_layers

        self.lift     = nn.Linear(n_features, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.w        = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, n_features)
        h = self.lift(x).transpose(1, 2)          # (B, width, L)
        for conv, w in zip(self.spectral, self.w):
            h = F.gelu(conv(h) + w(h))
        h = h.transpose(1, 2)                      # (B, L, width)
        return self.proj2(F.gelu(self.proj1(h))).squeeze(-1)   # (B, L)


# =============================================================================
# WAVELET NEURAL OPERATOR (WNO1D)
# =============================================================================

class WaveletConv1d(nn.Module):
    """Single wavelet spectral convolution layer for 1D WNO.

    Applies a learnable linear operator in the wavelet domain:
      1. DWT (pytorch_wavelets.DWT1DForward) with J=n_levels.
         Output: approximation coefficients yl (coarsest subband) and
         detail coefficients yh (list of J detail subbands, finest first).
      2. Multiply each subband by a separate learnable weight matrix.
         Channel mixing follows the same einsum convention as SpectralConv1d.
      3. IDWT (DWT1DInverse) to reconstruct the time-domain signal.

    Weight sizes are fixed at construction time based on the subband
    lengths computed from a dummy forward pass.  If the actual signal
    length at runtime differs (e.g. due to boundary effects), the weights
    are linearly interpolated to match via F.interpolate.

    Paraunitary constraint (optional)
    ----------------------------------
    When paraunitary=True, each weight matrix W[:,:,k] is column-
    normalised at each forward pass so that the channel-mixing linear
    map at every spectral position k is a partial isometry.  This
    enforces local energy preservation in the wavelet domain and prevents
    the learned weights from amplifying or attenuating subband energy
    without bound — the mechanism identified as the root cause of high
    inter-seed variance (CV ~30%) in WNO-haar (see Lima, Hesselbach &
    Amazonas, "MERA-Wavelet", 2026, Section 6).

    The normalisation is applied in the forward pass (not as a post-
    optimiser projection) so that gradients flow through the normalised
    weights.  This is analogous to weight normalisation (Salimans &
    Kingma, 2016) but applied per spectral position rather than per
    neuron, reflecting the structure of the wavelet domain.

    Parameters
    ----------
    in_channels  : int   Width of the input channel dimension.
    out_channels : int   Width of the output channel dimension.
    n_levels     : int   Number of DWT decomposition levels (J).
    wavelet      : str   Wavelet name accepted by pytorch_wavelets
                         (e.g. "db4", "haar").
    mode         : str   DWT boundary mode (default "symmetric").
    dummy_length : int   Nominal window length used to pre-compute
                         subband dimensions.  Must match the window
                         length used in the experiment.  See WNO1D.
    paraunitary  : bool  If True, column-normalise weights at each
                         spectral position to enforce local energy
                         preservation (default False).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_levels:     int = 4,
        wavelet:      str = "db4",
        mode:         str = "symmetric",
        dummy_length: int = 1024,
        paraunitary:  bool = False,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_levels     = n_levels
        self.paraunitary  = paraunitary

        try:
            from pytorch_wavelets import DWT1DForward, DWT1DInverse
        except ImportError as exc:
            raise ImportError(
                "WaveletConv1d requires pytorch_wavelets. "
                "Install with:  pip install pytorch-wavelets"
            ) from exc

        self.dwt  = DWT1DForward(J=n_levels, wave=wavelet, mode=mode)
        self.idwt = DWT1DInverse(wave=wavelet, mode=mode)

        # Pre-compute subband lengths with a dummy forward pass so that
        # weight tensors can be allocated at the correct sizes.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, dummy_length)
            yl, yh = self.dwt(dummy)

        scale = 1.0 / (in_channels * out_channels)
        self.weight_approx = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, yl.shape[-1])
        )
        self.weight_details = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_channels, out_channels, d.shape[-1]))
            for d in yh
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yl, yh = self.dwt(x)
        yl_out = self._mix(yl, self.weight_approx, self.paraunitary)
        yh_out = [
            self._mix(d, w, self.paraunitary)
            for d, w in zip(yh, self.weight_details)
        ]
        return self.idwt((yl_out, yh_out))

    @staticmethod
    def _mix(
        coeffs: torch.Tensor,
        weight: torch.Tensor,
        paraunitary: bool = False,
    ) -> torch.Tensor:
        """Per-subband channel mixing: (B, Cin, L) x (Cin, Cout, Lw) -> (B, Cout, L).

        If the weight's last dimension Lw does not match the coefficient
        length L (boundary effects from DWT padding), the weight is
        linearly interpolated to L before the einsum.

        When paraunitary=True, the weight matrix W[:,:,k] at each spectral
        position k is column-normalised so that each output channel receives
        a unit-norm projection of the input channels.  This enforces local
        energy preservation in the wavelet domain:

            ||output[:, :, k]||^2  <=  ||input[:, :, k]||^2

        with equality when in_channels == out_channels (square case).

        Motivation (cf. Lima, Hesselbach & Amazonas, "MERA-Wavelet", 2026):
        Without normalisation the learned weights R can amplify or attenuate
        energy arbitrarily at each spectral position, destroying the
        Parseval identity that the DWT/IDWT pair would otherwise guarantee.
        Across 4 stacked WNO layers this produces exponentially divergent
        energy scaling, creating qualitatively different loss basins
        depending on initialisation — the root cause of the 30% CV observed
        in WNO-haar Exp. 1.  Column normalisation is the minimal constraint
        that breaks the multiplicative degeneracy between layers while
        preserving full channel-mixing expressivity.
        """
        B, Cin, L = coeffs.shape
        Win, Wout, Lw = weight.shape

        if Lw != L:
            w = weight.unsqueeze(0).reshape(1, Win * Wout, Lw)
            w = F.interpolate(w, size=L, mode="linear", align_corners=False)
            weight = w.reshape(Win, Wout, L)

        if paraunitary:
            # Column-normalise: for each output channel o and spectral
            # position k, the vector weight[:, o, k] (over input channels)
            # is projected to unit norm.  This ensures that the linear map
            # Cin -> Cout at each position k is a (partial) isometry,
            # preserving energy up to the rank reduction when Cout < Cin.
            #
            # Shape: weight is (Cin, Cout, L).
            # Norm is taken over dim=0 (input channels).
            weight = weight / (weight.norm(dim=0, keepdim=True) + 1e-8)

        return torch.einsum("bil,iol->bol", coeffs, weight)


class WNO1D(nn.Module):
    """1D Wavelet Neural Operator.

    Same lifting / body / projection skeleton as FNO1D, with
    WaveletConv1d layers replacing SpectralConv1d.  The wavelet DWT
    provides a multi-resolution decomposition: coarse subbands capture
    slow OCV drift while fine subbands capture rapid RC polarisation
    transients.  This simultaneous multi-scale representation is the
    WNO's primary inductive bias advantage over the FNO's single-
    resolution Fourier truncation for drive-cycle voltage signals.

    The `wavelet` parameter selects the wavelet family:
      "db4"   Daubechies-4, 4 vanishing moments, filter support 7.
              Reference configuration, used in Exp. 1-4 and Exp. 5.
      "haar"  Haar (= db1), 1 vanishing moment, filter support 1.
              Ablation configuration; instantiated via build_model
              with name="wno_haar".

    Important: the `length` argument must equal the window length used
    at training and inference time.  WaveletConv1d pre-computes subband
    dimensions from a dummy forward pass of this length; mismatches
    cause silent weight interpolation errors in _mix.

    Reference
    ---------
    Tripura & Chakraborty. "Wavelet Neural Operator for solving
    parametric partial differential equations in computational
    mechanics problems", CMAME 2023.

    Parameters
    ----------
    n_features  : int   Input features per timestep (default 3).
    width       : int   Channel width throughout the body.
    n_layers    : int   Number of WaveletLayer blocks in the body.
    n_levels    : int   DWT decomposition levels (J in DWT1DForward).
    wavelet     : str   Wavelet name (e.g. "db4", "haar").
    length      : int   Nominal window length; must match WindowConfig.length.
    paraunitary : bool  If True, column-normalise wavelet-domain weights
                        at each spectral position to enforce local energy
                        preservation (see WaveletConv1d._mix docstring).
                        Default False for backward compatibility with
                        Exp. 1 baseline runs.
    """

    def __init__(
        self,
        n_features:  int  = 3,
        width:       int  = 32,
        n_layers:    int  = 4,
        n_levels:    int  = 4,
        wavelet:     str  = "db4",
        length:      int  = 1024,
        paraunitary: bool = False,
    ) -> None:
        super().__init__()
        self.n_features  = n_features
        self.width       = width
        self.n_layers    = n_layers
        self.paraunitary = paraunitary

        self.lift = nn.Linear(n_features, width)

        self.wavelet_layers = nn.ModuleList([
            WaveletConv1d(
                in_channels=width,
                out_channels=width,
                n_levels=n_levels,
                wavelet=wavelet,
                dummy_length=length,
                paraunitary=paraunitary,
            )
            for _ in range(n_layers)
        ])
        self.w = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, n_features)
        h = self.lift(x).transpose(1, 2)          # (B, width, L)
        for wconv, w in zip(self.wavelet_layers, self.w):
            h = F.gelu(wconv(h) + w(h))
        h = h.transpose(1, 2)                      # (B, L, width)
        return self.proj2(F.gelu(self.proj1(h))).squeeze(-1)   # (B, L)


# =============================================================================
# ECM-1RC + HYBRID OPERATOR
# =============================================================================

DEFAULT_ECM_STATS: dict[str, float] = {
    "I_mean": 0.0,
    "I_std": 1.0,
    "V_mean": 0.0,
    "V_std": 1.0,
    "SoC_mean": 0.5,
    "SoC_std": 0.3,
}


def _coerce_ecm_params(ecm_params: ECMParams | dict | None) -> ECMParams:
    """Return a fully-populated ECM parameter object."""
    if isinstance(ecm_params, ECMParams):
        return ecm_params
    if isinstance(ecm_params, dict):
        return ECMParams.from_dict(ecm_params)
    return ECMParams(
        ocv_coeffs=[3.7] + [0.0] * (N_OCV_COEFFS - 1),
        R0=0.03,
        R1=0.02,
        C1=1000.0,
        soc_train_min=0.0,
        soc_train_max=1.0,
    )


@dataclass
class HybridLossConfig:
    """Configuration of the optional residual loss term."""

    use_residual_loss: bool = False
    lambda_pred: float = 1.0
    lambda_res: float = 1.0
    residual_space: str = "normalized"

    def __post_init__(self) -> None:
        if self.residual_space not in {"normalized", "physical"}:
            raise ValueError(
                "Hybrid residual_space must be 'normalized' or 'physical'."
            )


class ECM1RCModel(nn.Module):
    """Window-wise ECM-1RC model with optional trainable physical parameters.

    The internal RC state is reset per window: V1(0)=0.  This is the
    conservative first implementation requested for the hybrid model and
    matches the existing numpy baseline in ecm.py.
    """

    def __init__(
        self,
        train_stats: dict[str, float] | None = None,
        ecm_params: ECMParams | dict | None = None,
        trainable_params: tuple[str, ...] | list[str] | None = None,
        dt: float = DT_SECONDS,
    ) -> None:
        super().__init__()
        params = _coerce_ecm_params(ecm_params)
        stats = dict(DEFAULT_ECM_STATS)
        if train_stats is not None:
            stats.update(train_stats)
        trainable = set(trainable_params or [])

        self.ocv_coeffs = nn.Parameter(
            torch.as_tensor(params.ocv_coeffs, dtype=torch.float32),
            requires_grad="ocv" in trainable,
        )
        self.log_R0 = nn.Parameter(
            torch.tensor(math.log10(float(params.R0)), dtype=torch.float32),
            requires_grad="R0" in trainable,
        )
        self.log_R1 = nn.Parameter(
            torch.tensor(math.log10(float(params.R1)), dtype=torch.float32),
            requires_grad="R1" in trainable,
        )
        self.log_C1 = nn.Parameter(
            torch.tensor(math.log10(float(params.C1)), dtype=torch.float32),
            requires_grad="C1" in trainable,
        )

        self.register_buffer("I_mean", torch.tensor(float(stats["I_mean"])))
        self.register_buffer("I_std", torch.tensor(float(stats["I_std"])))
        self.register_buffer("V_mean", torch.tensor(float(stats["V_mean"])))
        self.register_buffer("V_std", torch.tensor(float(stats["V_std"])))
        self.register_buffer("SoC_mean", torch.tensor(float(stats["SoC_mean"])))
        self.register_buffer("SoC_std", torch.tensor(float(stats["SoC_std"])))
        self.register_buffer(
            "soc_train_min", torch.tensor(float(params.soc_train_min))
        )
        self.register_buffer(
            "soc_train_max", torch.tensor(float(params.soc_train_max))
        )
        self.register_buffer("dt", torch.tensor(float(dt)))

    def iter_physical_parameters(self):
        yield self.ocv_coeffs
        yield self.log_R0
        yield self.log_R1
        yield self.log_C1

    def n_trainable_physical_parameters(self) -> int:
        return sum(
            p.numel() for p in self.iter_physical_parameters() if p.requires_grad
        )

    def model_info(self) -> dict[str, object]:
        return {
            "model_family": "ecm_1rc",
            "backbone": None,
            "include_ecm_input": False,
            "residual_space": None,
            "ecm_trainable": self.n_trainable_physical_parameters() > 0,
            "n_physical_trainable": self.n_trainable_physical_parameters(),
        }

    def normalize_voltage(self, v_phys: torch.Tensor) -> torch.Tensor:
        return (v_phys - self.V_mean) / self.V_std

    def denormalize_voltage(self, v_norm: torch.Tensor) -> torch.Tensor:
        return v_norm * self.V_std + self.V_mean

    def residual_to_normalized(self, residual_phys: torch.Tensor) -> torch.Tensor:
        return residual_phys / self.V_std

    def residual_to_physical(self, residual_norm: torch.Tensor) -> torch.Tensor:
        return residual_norm * self.V_std

    def _pipeline_to_ecm_current(self, x: torch.Tensor) -> torch.Tensor:
        I_pipeline = x[..., 0] * self.I_std + self.I_mean
        return -I_pipeline

    def _denormalize_soc(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 2] * self.SoC_std + self.SoC_mean

    def _eval_ocv(self, soc: torch.Tensor) -> torch.Tensor:
        soc = soc.clamp(self.soc_train_min, self.soc_train_max)
        x = 2.0 * soc - 1.0
        coeffs = self.ocv_coeffs.to(dtype=x.dtype)

        t0 = torch.ones_like(x)
        out = coeffs[0] * t0
        if coeffs.numel() == 1:
            return out

        t1 = x
        out = out + coeffs[1] * t1
        for idx in range(2, coeffs.numel()):
            t2 = 2.0 * x * t1 - t0
            out = out + coeffs[idx] * t2
            t0, t1 = t1, t2
        return out

    def simulate_physical(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate V_ECM in physical volts from normalized inputs."""
        if x.ndim != 3 or x.shape[-1] < 3:
            raise ValueError(
                "ECM1RCModel expects x with shape (batch, length, >=3). "
                f"Got {tuple(x.shape)}."
            )

        I = self._pipeline_to_ecm_current(x)
        soc = self._denormalize_soc(x)
        ocv = self._eval_ocv(soc)

        ten = torch.tensor(10.0, device=x.device, dtype=x.dtype)
        R0 = torch.pow(ten, self.log_R0.to(device=x.device, dtype=x.dtype))
        R1 = torch.pow(ten, self.log_R1.to(device=x.device, dtype=x.dtype))
        C1 = torch.pow(ten, self.log_C1.to(device=x.device, dtype=x.dtype))
        inv_tau = 1.0 / (R1 * C1)
        inv_C1 = 1.0 / C1
        dt = self.dt.to(device=x.device, dtype=x.dtype)

        batch_size, length = I.shape
        V1 = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        V_hat: list[torch.Tensor] = []
        for idx in range(length):
            V_k = ocv[:, idx] - R0 * I[:, idx] - V1
            V_hat.append(V_k.unsqueeze(1))
            V1 = V1 + dt * (-V1 * inv_tau + I[:, idx] * inv_C1)
        return torch.cat(V_hat, dim=1)

    def forward_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        v_ecm_phys = self.simulate_physical(x)
        v_ecm_norm = self.normalize_voltage(v_ecm_phys)
        zeros = torch.zeros_like(v_ecm_norm)
        return {
            "pred_norm": v_ecm_norm,
            "pred_phys": v_ecm_phys,
            "ecm_norm": v_ecm_norm,
            "ecm_phys": v_ecm_phys,
            "res_norm": zeros,
            "res_phys": zeros,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        pred_loss = loss_fn(self(x), target)
        zero = pred_loss.detach().new_zeros(())
        return {
            "loss": pred_loss,
            "pred_loss": pred_loss.detach(),
            "residual_loss": zero,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)["pred_norm"]


class ResidualNeuralOperator(nn.Module):
    """Backbone wrapper used by HybridECMNO."""

    def __init__(
        self,
        backbone: str = "wno",
        n_features: int = 3,
        modes: int = 24,
        width: int = 32,
        n_layers: int = 4,
        n_levels: int = 4,
        wavelet: str = "db4",
        length: int = 1024,
        paraunitary: bool = False,
    ) -> None:
        super().__init__()
        key = backbone.lower()
        self.backbone = key
        if key == "fno":
            self.net = FNO1D(
                n_features=n_features,
                modes=modes,
                width=width,
                n_layers=n_layers,
            )
        elif key == "wno":
            self.net = WNO1D(
                n_features=n_features,
                width=width,
                n_layers=n_layers,
                n_levels=n_levels,
                wavelet=wavelet,
                length=length,
                paraunitary=paraunitary,
            )
        else:
            raise ValueError(
                f"Unknown residual backbone {backbone!r}. "
                "Expected 'fno' or 'wno'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridECMNO(nn.Module):
    """Hybrid predictor with a frozen/trainable ECM anchor plus NO residual."""

    def __init__(
        self,
        n_features: int = 3,
        backbone: str = "wno",
        include_ecm_input: bool = False,
        train_stats: dict[str, float] | None = None,
        ecm_params: ECMParams | dict | None = None,
        ecm_trainable_params: tuple[str, ...] | list[str] | None = None,
        use_residual_loss: bool = False,
        lambda_pred: float = 1.0,
        lambda_res: float = 1.0,
        residual_space: str = "normalized",
        modes: int = 24,
        width: int = 32,
        n_layers: int = 4,
        n_levels: int = 4,
        wavelet: str = "db4",
        length: int = 1024,
        paraunitary: bool = False,
    ) -> None:
        super().__init__()
        if n_features < 3:
            raise ValueError(
                "HybridECMNO requires at least the base channels "
                "(I, T_cell, SoC)."
            )

        self.n_features = n_features
        self.include_ecm_input = include_ecm_input
        self.loss_cfg = HybridLossConfig(
            use_residual_loss=use_residual_loss,
            lambda_pred=lambda_pred,
            lambda_res=lambda_res,
            residual_space=residual_space,
        )
        self.ecm = ECM1RCModel(
            train_stats=train_stats,
            ecm_params=ecm_params,
            trainable_params=ecm_trainable_params,
        )
        residual_features = n_features + int(include_ecm_input)
        self.residual = ResidualNeuralOperator(
            backbone=backbone,
            n_features=residual_features,
            modes=modes,
            width=width,
            n_layers=n_layers,
            n_levels=n_levels,
            wavelet=wavelet,
            length=length,
            paraunitary=paraunitary,
        )
        self.backbone = backbone.lower()

    def iter_physical_parameters(self):
        yield from self.ecm.iter_physical_parameters()

    def n_trainable_physical_parameters(self) -> int:
        return self.ecm.n_trainable_physical_parameters()

    def model_info(self) -> dict[str, object]:
        return {
            "model_family": "hybrid_ecmno",
            "backbone": self.backbone,
            "include_ecm_input": self.include_ecm_input,
            "residual_space": self.loss_cfg.residual_space,
            "ecm_trainable": self.n_trainable_physical_parameters() > 0,
            "n_physical_trainable": self.n_trainable_physical_parameters(),
            "use_residual_loss": self.loss_cfg.use_residual_loss,
            "lambda_pred": self.loss_cfg.lambda_pred,
            "lambda_res": self.loss_cfg.lambda_res,
        }

    def _residual_input(
        self,
        x: torch.Tensor,
        v_ecm_norm: torch.Tensor,
    ) -> torch.Tensor:
        if not self.include_ecm_input:
            return x[..., : self.n_features]
        return torch.cat([x[..., : self.n_features], v_ecm_norm.unsqueeze(-1)], dim=-1)

    def forward_components(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        v_ecm_phys = self.ecm.simulate_physical(x)
        v_ecm_norm = self.ecm.normalize_voltage(v_ecm_phys)

        residual_input = self._residual_input(x, v_ecm_norm)
        residual_raw = self.residual(residual_input)

        if self.loss_cfg.residual_space == "physical":
            v_res_phys = residual_raw
            v_res_norm = self.ecm.residual_to_normalized(v_res_phys)
            v_pred_phys = v_ecm_phys + v_res_phys
            v_pred_norm = self.ecm.normalize_voltage(v_pred_phys)
        else:
            v_res_norm = residual_raw
            v_res_phys = self.ecm.residual_to_physical(v_res_norm)
            v_pred_norm = v_ecm_norm + v_res_norm
            v_pred_phys = self.ecm.denormalize_voltage(v_pred_norm)

        outputs = {
            "pred_norm": v_pred_norm,
            "pred_phys": v_pred_phys,
            "ecm_norm": v_ecm_norm,
            "ecm_phys": v_ecm_phys,
            "res_norm": v_res_norm,
            "res_phys": v_res_phys,
        }
        if target is not None:
            target_phys = self.ecm.denormalize_voltage(target)
            outputs.update(
                {
                    "target_norm": target,
                    "target_phys": target_phys,
                    "target_res_norm": target - v_ecm_norm,
                    "target_res_phys": target_phys - v_ecm_phys,
                }
            )
        return outputs

    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        comp = self.forward_components(x, target=target)
        pred_loss = loss_fn(comp["pred_norm"], target)

        if self.loss_cfg.use_residual_loss:
            if self.loss_cfg.residual_space == "physical":
                residual_loss = loss_fn(comp["res_phys"], comp["target_res_phys"])
            else:
                residual_loss = loss_fn(comp["res_norm"], comp["target_res_norm"])
        else:
            residual_loss = pred_loss.new_zeros(())

        total = (
            self.loss_cfg.lambda_pred * pred_loss
            + self.loss_cfg.lambda_res * residual_loss
        )
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "residual_loss": residual_loss.detach(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)["pred_norm"]


# =============================================================================
# FACTORY AND UTILITIES
# =============================================================================

_REGISTRY: dict[str, type] = {
    "lstm":        LSTMBaseline,
    "fno":         FNO1D,
    "wno":         WNO1D,
    "wno_pu":      WNO1D,      # same class, db4 + paraunitary=True
    "wno_db4_pu":  WNO1D,      # explicit alias for db4 + paraunitary=True
    "wno_haar":    WNO1D,      # same class, wavelet="haar" injected below
    "wno_haar_pu": WNO1D,      # same class, wavelet="haar" + paraunitary=True
    "ecm_1rc":     ECM1RCModel,
    "hybrid_ecmno": HybridECMNO,
    "hybrid_wno":  HybridECMNO,
    "hybrid_fno":  HybridECMNO,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of "lstm", "fno", "wno", "wno_pu", "wno_db4_pu",
        "wno_haar", "wno_haar_pu", "ecm_1rc", "hybrid_ecmno",
        "hybrid_wno", or "hybrid_fno".
        "wno_haar" instantiates WNO1D with wavelet="haar" injected into
        kwargs if not already present, allowing experiment scripts to use
        it as a first-class model name without a separate class.
        "wno_pu" / "wno_db4_pu" instantiate WNO1D with the default db4
        wavelet and paraunitary=True.
        "wno_haar_pu" additionally injects paraunitary=True.
        "hybrid_wno" and "hybrid_fno" inject the residual backbone.
    **kwargs
        Passed directly to the model constructor.  Must match the
        constructor signature of the chosen architecture.

    Returns
    -------
    nn.Module  (untrained, on CPU)

    Raises
    ------
    ValueError  if name is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model: {name!r}. "
            f"Available: {sorted(_REGISTRY)}."
        )
    if key == "wno_haar":
        kwargs.setdefault("wavelet", "haar")
    if key in {"wno_pu", "wno_db4_pu"}:
        kwargs.setdefault("wavelet", "db4")
        kwargs.setdefault("paraunitary", True)
    if key == "wno_haar_pu":
        kwargs.setdefault("wavelet", "haar")
        kwargs.setdefault("paraunitary", True)
    if key == "hybrid_wno":
        kwargs.setdefault("backbone", "wno")
    if key == "hybrid_fno":
        kwargs.setdefault("backbone", "fno")
    return _REGISTRY[key](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model.

    Counts only parameters with requires_grad=True.  Used for the
    parameter-count column in all experiment summary tables.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Return the total number of registered parameters."""
    return sum(p.numel() for p in model.parameters())


def count_physical_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable ECM parameters, if present."""
    if hasattr(model, "iter_physical_parameters"):
        return sum(
            p.numel()
            for p in model.iter_physical_parameters()
            if isinstance(p, nn.Parameter) and p.requires_grad
        )
    return 0
