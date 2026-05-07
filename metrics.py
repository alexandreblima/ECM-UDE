"""
Evaluation metrics for battery voltage prediction.

Includes:
    - Standard pointwise metrics (MAE, RMSE, max error)
    - Wavelet-band error decomposition (for Experiment 4)

The wavelet-band decomposition is the methodologically distinctive metric
for this project: it shows *at which scales* each model fails, which is
the kind of forensic analysis that a wavelet specialist can do and that
differentiates this work from a generic "we applied X to battery data" paper.
"""

from __future__ import annotations

import torch


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error. Works on any shape; averages over all elements."""
    return (pred - target).abs().mean().item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return ((pred - target) ** 2).mean().sqrt().item()


def compute_max_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Maximum pointwise absolute error, across all samples.

    Often more informative than MAE for battery applications, because a
    large instantaneous error near a transient can be operationally
    worse than a small average error.
    """
    return (pred - target).abs().max().item()


def report(pred: torch.Tensor, target: torch.Tensor,
           denorm_std: float = 1.0, denorm_mean: float = 0.0) -> dict[str, float]:
    """Compute the standard set of metrics and return them as a dict.

    If the predictions are in z-scored (normalized) space, pass `denorm_std`
    (and optionally `denorm_mean`) to get metrics in physical units (Volts).
    MAE and RMSE scale linearly with `denorm_std`.
    """
    pred_phys = pred * denorm_std + denorm_mean
    targ_phys = target * denorm_std + denorm_mean
    return {
        "mae": compute_mae(pred_phys, targ_phys),
        "rmse": compute_rmse(pred_phys, targ_phys),
        "max_err": compute_max_error(pred_phys, targ_phys),
    }


# --- wavelet-band error decomposition (Experiment 4) ---------------------

def band_energy_decomposition(
    residual: torch.Tensor,
    n_levels: int = 6,
    wavelet: str = "db4",
) -> dict[str, float]:
    """Decompose the energy of a residual signal across wavelet bands.

    Given a residual r(t) = V_pred(t) - V_true(t), applies a DWT and returns
    the relative energy in each frequency band (approximation + details).
    This reveals at which scales the model's errors are concentrated.

    Parameters
    ----------
    residual : tensor of shape (batch, length) or (length,)
        The prediction residual in the time domain.
    n_levels : int
        Number of DWT decomposition levels. Each level roughly corresponds
        to a dyadic scale: level 1 = fastest, level n_levels = slowest.
    wavelet : str
        Wavelet family name (e.g. "db4", "sym6", "coif2").

    Returns
    -------
    dict with keys:
        'approx'  : energy fraction in the approximation (slowest) band
        'detail_1', ..., 'detail_n' : energy fraction in each detail band,
                     where detail_1 is the finest (fastest) scale
    All values are normalized so they sum to 1.
    """
    try:
        from pytorch_wavelets import DWT1DForward
    except ImportError as e:
        raise ImportError(
            "band_energy_decomposition requires pytorch_wavelets. "
            "Install with: pip install pytorch-wavelets"
        ) from e

    if residual.ndim == 1:
        residual = residual.unsqueeze(0)
    # DWT1DForward expects (batch, channels, length)
    x = residual.unsqueeze(1).float()

    dwt = DWT1DForward(J=n_levels, wave=wavelet, mode="symmetric")
    yl, yh = dwt(x)

    # yl: (B, 1, L_a) ; yh: list of (B, 1, L_k) tensors, finest first
    e_approx = (yl ** 2).sum().item()
    e_details = [(d ** 2).sum().item() for d in yh]
    total = e_approx + sum(e_details) + 1e-12

    result = {"approx": e_approx / total}
    for k, e in enumerate(e_details, start=1):
        result[f"detail_{k}"] = e / total
    return result


def band_mae_decomposition(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_levels: int = 6,
    wavelet: str = "db4",
) -> dict[str, float]:
    """Compute MAE *after* band-pass filtering both pred and target.

    Alternative to energy decomposition: filters both signals to isolate
    each wavelet band, then computes MAE in that band. Makes the error
    decomposition directly comparable to the usual MAE metric (same units).

    This is the version I'd use for a plot of "MAE vs band" in the paper.
    """
    try:
        from pytorch_wavelets import DWT1DForward, DWT1DInverse
    except ImportError as e:
        raise ImportError(
            "band_mae_decomposition requires pytorch_wavelets."
        ) from e

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    p = pred.unsqueeze(1).float()
    t = target.unsqueeze(1).float()

    dwt = DWT1DForward(J=n_levels, wave=wavelet, mode="symmetric")
    idwt = DWT1DInverse(wave=wavelet, mode="symmetric")

    yl_p, yh_p = dwt(p)
    yl_t, yh_t = dwt(t)

    def _reconstruct_band(yl, yh, band_idx):
        """Zero out all bands except `band_idx`, then IDWT.
        band_idx == 0  -> approximation only
        band_idx == k  -> detail level k only (1-indexed)
        """
        zero_yl = torch.zeros_like(yl)
        zero_yh = [torch.zeros_like(d) for d in yh]
        if band_idx == 0:
            return idwt((yl, zero_yh))
        else:
            zero_yh[band_idx - 1] = yh[band_idx - 1]
            return idwt((zero_yl, zero_yh))

    result = {}
    for band_idx, name in enumerate(
        ["approx"] + [f"detail_{k}" for k in range(1, n_levels + 1)]
    ):
        p_band = _reconstruct_band(yl_p, yh_p, band_idx)
        t_band = _reconstruct_band(yl_t, yh_t, band_idx)
        result[name] = (p_band - t_band).abs().mean().item()

    return result
