"""
Training loop — wno_battery.training
======================================

Overview
--------
This module provides a model-agnostic training loop, a learning-rate
schedule, and a batched inference helper for the WNO battery voltage
estimation pipeline.

Any nn.Module that accepts (batch, length, n_features) and returns
(batch, length) can be trained with train() without modification.  The
same is true of predict(), which simply iterates a DataLoader and
concatenates predictions.

Public API
----------
  TrainConfig     Dataclass of all training hyperparameters.
  TrainHistory    Dataclass recording per-epoch metrics.
  train(model, train_loader, val_loader, cfg)  Full training loop.
  predict(model, loader, device)               Batched inference.

Loss function
-------------
Plain MSE in normalised V space.  MSE in normalised units is
equivalent to weighted MSE in physical units with weight 1/V_std^2,
which is constant across the dataset and does not distort the optimum.
Physical-unit metrics (MAE, RMSE, p99) are computed post-hoc in the
experiment scripts by denormalising predictions with the dataset stats.

Learning-rate schedule
----------------------
Two-phase schedule applied when TrainConfig.use_lr_schedule=True:

  Phase 1 — Linear warmup over the first warmup_epochs epochs.
    LR rises from 0 to cfg.lr.  Warmup prevents the large gradient
    updates that occur at initialisation from destabilising the
    spectral operator weight matrices, which were observed to cause
    val_loss oscillation in early FNO/WNO runs with a constant LR.
    Using (epoch + 1) / warmup rather than epoch / warmup ensures the
    first epoch has a non-zero LR.

  Phase 2 — Cosine annealing from cfg.lr down to
    min_lr_factor * cfg.lr over the remaining (n_epochs - warmup_epochs)
    epochs.  The cosine schedule decays LR smoothly without the sharp
    drop of step-decay, which improves convergence stability for the
    WNO's wavelet kernel parameters.

Set use_lr_schedule=False to recover constant-LR Adam behaviour
(used in ablation experiments that isolate the schedule contribution).

Early stopping
--------------
Training stops if val_loss has not improved for `patience` consecutive
epochs.  The best checkpoint is saved to cfg.ckpt_path whenever a new
val_loss minimum is reached.  At the end of training the best checkpoint
is reloaded into the model, so the returned model is always the best-
validation model regardless of whether early stopping was triggered.

Checkpoint behaviour
--------------------
Checkpoints are saved as PyTorch state_dicts (weights only, no
optimiser state, no normalisation statistics).  Normalisation stats
must be separately saved or recomputed by the experiment scripts;
see the Normalisation note in data.py and the load_train_stats
function in Exp. 2-3.

LR logging convention
---------------------
The LR recorded in TrainHistory.lr for epoch e is the LR used during
the forward/backward pass of epoch e, i.e. the LR *before* the
scheduler step.  This matches the convention used by PyTorch's
LambdaLR when step() is called at the end of the epoch (not the
beginning).  The summary tables in the experiment scripts print this
value for interpretability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------
# Configuration dataclasses
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters for one training run.

    Attributes
    ----------
    n_epochs : int
        Maximum number of training epochs.  May be fewer if early
        stopping triggers.
    lr : float
        Peak learning rate.  With the schedule enabled, this is the
        LR reached at the end of warmup and the starting point for
        cosine annealing.
    weight_decay : float
        L2 regularisation coefficient for Adam.
    grad_clip : float or None
        If not None, clip gradient norm to this value before each
        optimiser step.  Prevents exploding gradients in the early
        warmup phase of FNO/WNO training.
    patience : int
        Early stopping patience: stop if val_loss has not improved for
        this many consecutive epochs.
    ckpt_path : str or None
        Path for saving the best-val-loss checkpoint.  If None, no
        checkpoint is written to disk (useful for quick ablations).
    log_every : int
        Print a progress line every this many epochs.  Set higher (e.g.
        50) when running many configurations in Exp. 5 to reduce output.
    device : str
        PyTorch device string, e.g. "cpu", "cuda", "mps".
        Determined at script startup by utils.pick_device().
    use_lr_schedule : bool
        If True (default), apply the warmup + cosine annealing schedule
        described in the module docstring.  If False, use a constant LR.
    warmup_epochs : int
        Number of linear-warmup epochs (Phase 1 of the schedule).
    min_lr_factor : float
        Floor of the cosine annealing phase as a fraction of lr.
        Final LR = min_lr_factor * lr.  Default 0.01 (1% of peak LR).
    """
    n_epochs:       int            = 100
    lr:             float          = 1e-3
    weight_decay:   float          = 1e-5
    grad_clip:      Optional[float] = 1.0
    patience:       int            = 20
    ckpt_path:      Optional[str]  = None
    log_every:      int            = 5
    device:         str            = "cpu"
    use_lr_schedule: bool          = True
    warmup_epochs:  int            = 5
    min_lr_factor:  float          = 0.01


@dataclass
class TrainHistory:
    """Per-epoch training metrics, accumulated by train().

    Attributes
    ----------
    train_loss : list[float]
        Mean training MSE (normalised V units) per epoch.
    val_loss : list[float]
        Mean validation MSE (normalised V units) per epoch.
    lr : list[float]
        Learning rate used during each epoch's forward/backward pass
        (before the scheduler step; see LR logging convention in the
        module docstring).
    best_val_loss : float
        Lowest validation MSE seen during training.
    best_epoch : int
        Epoch (1-indexed) at which best_val_loss was achieved.
    """
    train_loss:          list[float] = field(default_factory=list)
    val_loss:            list[float] = field(default_factory=list)
    train_pred_loss:     list[float] = field(default_factory=list)
    val_pred_loss:       list[float] = field(default_factory=list)
    train_residual_loss: list[float] = field(default_factory=list)
    val_residual_loss:   list[float] = field(default_factory=list)
    lr:                  list[float] = field(default_factory=list)
    best_val_loss:       float       = float("inf")
    best_epoch:          int         = -1


# --------------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------------

def _build_lr_lambda(cfg: TrainConfig):
    """Return a LambdaLR-compatible callable implementing warmup + cosine.

    The returned function maps epoch (0-indexed, PyTorch LambdaLR
    convention) to a multiplicative factor applied to the base lr.

    Phase 1 (epoch < warmup_epochs): linear ramp from 1/warmup to 1.
      Using (epoch + 1) / warmup ensures the first epoch has a non-zero
      LR.  At epoch = warmup_epochs - 1 the factor reaches 1.0.

    Phase 2 (epoch >= warmup_epochs): cosine annealing.
      progress = (epoch - warmup) / (n_epochs - warmup), clipped to [0, 1].
      factor = min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + cos(pi * progress))
      At progress=0 the factor is 1.0; at progress=1 it is min_lr_factor.
    """
    warmup     = max(0, cfg.warmup_epochs)
    total      = max(1, cfg.n_epochs)
    min_factor = cfg.min_lr_factor

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))
        progress = (epoch - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return lr_lambda


# --------------------------------------------------------------------------
# Loss / epoch helpers
# --------------------------------------------------------------------------

def _compute_batch_loss(
    model: nn.Module,
    X_win: torch.Tensor,
    V_win: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total, prediction, and residual loss for one batch.

    Models may optionally implement `compute_loss(x, y, loss_fn=...)` and
    return a dict with keys:

      loss           scalar tensor used for backprop
      pred_loss      scalar tensor/logging value
      residual_loss  scalar tensor/logging value

    Plain models fall back to standard prediction MSE only.
    """
    if hasattr(model, "compute_loss"):
        loss_dict = model.compute_loss(X_win, V_win, loss_fn=loss_fn)
        loss = loss_dict["loss"]
        pred_loss = loss_dict.get("pred_loss", loss.detach())
        residual_loss = loss_dict.get(
            "residual_loss",
            loss.detach().new_zeros(()),
        )
        if not torch.is_tensor(pred_loss):
            pred_loss = loss.detach().new_tensor(float(pred_loss))
        if not torch.is_tensor(residual_loss):
            residual_loss = loss.detach().new_tensor(float(residual_loss))
        return loss, pred_loss.detach(), residual_loss.detach()

    preds = model(X_win)
    loss = loss_fn(preds, V_win)
    zero = loss.detach().new_zeros(())
    return loss, loss.detach(), zero


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> tuple[float, float, float]:
    """Run one train or validation epoch and return mean losses."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    loss_sum = 0.0
    pred_sum = 0.0
    residual_sum = 0.0
    n_items = 0

    if is_train:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()

    with context:
        for X_win, V_win in loader:
            X_win = X_win.to(device)
            V_win = V_win.to(device)

            if is_train:
                optimizer.zero_grad()

            loss, pred_loss, residual_loss = _compute_batch_loss(
                model, X_win, V_win, loss_fn
            )

            if is_train:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = X_win.size(0)
            loss_sum += float(loss.item()) * batch_size
            pred_sum += float(pred_loss.item()) * batch_size
            residual_sum += float(residual_loss.item()) * batch_size
            n_items += batch_size

    denom = max(1, n_items)
    return loss_sum / denom, pred_sum / denom, residual_sum / denom


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          TrainConfig,
) -> TrainHistory:
    """Train a model with Adam + MSE loss and optional LR schedule.

    Model-agnostic: any nn.Module accepting (B, L, F) -> (B, L) works
    without modification.

    The training loop:
      1. Forward pass on train_loader, MSE loss, backward, clip, step.
      2. Validation pass on val_loader (no gradient).
      3. Log current LR (pre-scheduler-step).
      4. Checkpoint if val_loss improved; increment patience counter
         otherwise.
      5. Step the LR scheduler (at epoch end, not batch end).
      6. Early stop if patience is exhausted.
    After the loop, reload the best checkpoint into the model so the
    caller always receives the best-validation model.

    Parameters
    ----------
    model        : nn.Module  Untrained model (moved to cfg.device).
    train_loader : DataLoader  Shuffled training windows.
    val_loader   : DataLoader  Sequential validation windows.
    cfg          : TrainConfig  All hyperparameters.

    Returns
    -------
    TrainHistory  Per-epoch train_loss, val_loss, lr, best_val_loss,
                  best_epoch.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer: Optional[torch.optim.Optimizer] = None
    if trainable_params:
        optimizer = torch.optim.Adam(
            trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    if cfg.use_lr_schedule and optimizer is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_build_lr_lambda(cfg),
        )

    loss_fn = nn.MSELoss()
    history = TrainHistory()
    epochs_since_best = 0

    if optimizer is None:
        print("  model has no trainable parameters; running evaluation-only pass")
        train_loss, train_pred_loss, train_residual_loss = _run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
        )
        val_loss, val_pred_loss, val_residual_loss = _run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_pred_loss.append(train_pred_loss)
        history.val_pred_loss.append(val_pred_loss)
        history.train_residual_loss.append(train_residual_loss)
        history.val_residual_loss.append(val_residual_loss)
        history.lr.append(0.0)
        history.best_val_loss = val_loss
        history.best_epoch = 1
        if cfg.ckpt_path is not None:
            Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.ckpt_path)
        print(
            f"  eval-only  train {train_loss:.6f}  "
            f"val {val_loss:.6f}"
        )
        return history

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss, train_pred_loss, train_residual_loss = _run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
        )

        val_loss, val_pred_loss, val_residual_loss = _run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        # Log LR before stepping the scheduler (see module docstring)
        current_lr = optimizer.param_groups[0]["lr"]
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_pred_loss.append(train_pred_loss)
        history.val_pred_loss.append(val_pred_loss)
        history.train_residual_loss.append(train_residual_loss)
        history.val_residual_loss.append(val_residual_loss)
        history.lr.append(current_lr)

        # ── Checkpoint / early stopping ───────────────────────────────────
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch    = epoch
            epochs_since_best     = 0
            if cfg.ckpt_path is not None:
                Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), cfg.ckpt_path)
        else:
            epochs_since_best += 1

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1:
            log_line = (
                f"  epoch {epoch:4d}  "
                f"train {train_loss:.6f}  "
                f"val {val_loss:.6f}  "
                f"lr {current_lr:.2e}  "
                f"(best {history.best_val_loss:.6f} @ ep {history.best_epoch})"
            )
            if train_residual_loss > 0.0 or val_residual_loss > 0.0:
                log_line += (
                    f"  pred {val_pred_loss:.6f}  "
                    f"res {val_residual_loss:.6f}"
                )
            print(log_line)

        if epochs_since_best >= cfg.patience:
            print(
                f"  early stopping at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs)"
            )
            break

        if scheduler is not None:
            scheduler.step()

    # ── Reload best checkpoint ────────────────────────────────────────────
    if cfg.ckpt_path is not None and Path(cfg.ckpt_path).is_file():
        model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=device, weights_only=True)
        )

    return history


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model:  nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a DataLoader and return concatenated tensors.

    Parameters
    ----------
    model  : nn.Module   Trained model (set to eval mode internally).
    loader : DataLoader  Any loader yielding (X_win, V_win) batches.
    device : str         Device to run inference on.

    Returns
    -------
    (preds, targs) : two float32 tensors, both on CPU, shape
        (N_total_windows, window_length).
    N_total_windows = sum of batch sizes across all batches in loader.

    Notes
    -----
    The model is moved to `device` and set to eval mode.  All output
    tensors are moved back to CPU before concatenation to keep GPU memory
    pressure low when iterating large test datasets.

    predict() does not perform any denormalization.  Callers are
    responsible for applying dataset.stats["V_mean"] / ["V_std"] to
    convert predictions to physical units before computing metrics.
    """
    model = model.to(device).eval()
    preds, targs = [], []
    for X_win, V_win in loader:
        preds.append(model(X_win.to(device)).cpu())
        targs.append(V_win)
    return torch.cat(preds, dim=0), torch.cat(targs, dim=0)


@torch.no_grad()
def predict_with_components(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Run inference and collect auxiliary components when available.

    For plain models the returned dict contains:
      preds, targs

    For HybridECMNO / ECM1RCModel it additionally contains any tensors
    returned by `forward_components`, such as:
      pred_norm, pred_phys, ecm_norm, ecm_phys, res_norm, res_phys
    """
    model = model.to(device).eval()
    collected: dict[str, list[torch.Tensor]] = {}

    for X_win, V_win in loader:
        X_dev = X_win.to(device)
        if hasattr(model, "forward_components"):
            batch = model.forward_components(X_dev)
        else:
            batch = {"pred_norm": model(X_dev)}

        for key, value in batch.items():
            if torch.is_tensor(value):
                collected.setdefault(key, []).append(value.cpu())

        collected.setdefault("target_norm", []).append(V_win.cpu())

    merged = {
        key: torch.cat(values, dim=0)
        for key, values in collected.items()
    }
    merged["preds"] = merged["pred_norm"]
    merged["targs"] = merged["target_norm"]
    return merged
