"""Single source of truth for the unified battery benchmark.

Ah is canonical *net discharged* Ah, positive on discharge. Raw Panasonic
Ah is charge-positive and must be negated; CALCE uses discharge minus charge
capacity. No cycle-end capacity or held-out voltage is used to define SOC.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import numpy as np

VERSION = "unified-1"
SEEDS = (42, 43, 44, 45, 46)
OCV_CLIP = (0.0, 1.0)
VAL_FRACTION = 0.2
METRICS = ("mae", "rmse", "max_error", "p99")
MODELS = ("ecm_1rc_nls", "ecm_2rc_nls", "physics_gradient",
          "output_residual", "ecm_ude", "lstm")
OUTPUT = Path("results/unified")
TRAIN = dict(n_epochs=30, lr=2e-4, weight_decay=1e-5, grad_clip=0.5,
             patience=8, use_lr_schedule=True, warmup_epochs=3,
             min_lr_factor=0.01, log_every=1, device="cpu")
LSTM = dict(n_features=3, hidden_size=32, num_layers=2, dropout=0.0)
LSTM_TRAIN = {**TRAIN, "n_epochs":150, "lr":1e-3, "grad_clip":1.0,
              "patience":30, "warmup_epochs":5}
REST_CURRENT_A = 0.02
REST_SECONDS = 30.0

@dataclass(frozen=True)
class DatasetProtocol:
    nominal_capacity_Ah: float
    default_z0: float
    dt_seconds: float
    window_length: int
    window_stride: int
    batch_size: int = 32

DATASETS = {
    "panasonic": DatasetProtocol(2.9, 1.0, 0.1, 1024, 512),
    "calce": DatasetProtocol(2.0, 1.0, 1.0, 256, 128),
}
PANASONIC_FILES = {
    "UDDS_25C": "25degC_UDDS_Pan18650PF.mat",
    "UDDS_10C": "10degC_UDDS_Pan18650PF.mat",
    "UDDS_0C": "0degC_UDDS_Pan18650PF.mat",
    "UDDS_-10C": "n10degC_UDDS_Pan18650PF.mat",
    "UDDS_-20C": "n20degC_UDDS_Pan18650PF.mat",
    "US06_25C": "25degC_US06_Pan18650PF.mat",
    "LA92_25C": "25degC_LA92_Pan18650PF.mat",
    "HWFET_25C": "25degC_" + "HW" + "FTa_Pan18650PF.mat",
}

def causal_soc(net_discharged_Ah, *, qnom_Ah, z0):
    ah = np.asarray(net_discharged_Ah, dtype=np.float64)
    if qnom_Ah <= 0 or not 0 <= z0 <= 1:
        raise ValueError("Positive Qnom and explicit z0 in [0,1] required")
    return np.clip(z0 - (ah - ah[0]) / qnom_Ah, *OCV_CLIP)

def temporal_split(n_samples, cfg):
    """80/20 by windows; remove overlapping training windows (guard band)."""
    starts = np.arange(0, n_samples - cfg.window_length + 1,
                       cfg.window_stride, dtype=int)
    if len(starts) and starts[-1] != n_samples-cfg.window_length:
        starts = np.append(starts,n_samples-cfg.window_length)
    if len(starts) < 3:
        raise ValueError("Cycle too short for a guarded temporal split")
    nv = max(1, int(len(starts) * VAL_FRACTION))
    cut = len(starts) - nv
    guard = max(0, (cfg.window_length + cfg.window_stride - 1)
                // cfg.window_stride - 1)
    tr = np.arange(cut - guard)
    va = np.arange(cut, len(starts))
    assert starts[tr[-1]] + cfg.window_length <= starts[va[0]]
    return starts, tr, va

def metrics_mV(pred_V, true_V):
    e = (np.asarray(pred_V, dtype=np.float64)
         - np.asarray(true_V, dtype=np.float64)) * 1000.0
    if not len(e) or not np.isfinite(e).all():
        raise ValueError("Empty or non-finite evaluated trajectory")
    a = np.abs(e)
    return dict(mae=float(a.mean()), rmse=float(np.sqrt(np.mean(e*e))),
                max_error=float(a.max()), p99=float(np.quantile(a, .99)))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest():
    return dict(version=VERSION, datasets={k: asdict(v) for k,v in DATASETS.items()},
                seeds=list(SEEDS), models=list(MODELS), train=TRAIN, lstm=LSTM,
                lstm_train=LSTM_TRAIN,
                split="temporal 80/20 windows with overlap guard band",
                normalization="training samples only; fixed T and SOC scales",
                soc="z0 - (net_discharged_Ah - Ah[0])/Qnom; physical clip [0,1]",
                z0="inverse independent OCV at last pre-cycle rested point; otherwise 1.0, logged per cycle",
                ocv_clip=list(OCV_CLIP), solver="classical fixed-step RK4, causal zero-order-held current/temperature",
                initial_state="polarization zero; windowed SOC from causal input at window start; continuous SOC initialized once at cycle start",
                continuous="carry states from cycle start; source metrics masked to held-out samples; identical covered-sample mask in both modes",
                metrics_units="mV", metrics=list(METRICS),
                selection="early stopping on source validation MSE; reported source held-out is validation, not an independent final test",
                inference="zero-shot, no target parameter fitting",
                statistics="two-sided exact paired Wilcoxon across 5 seeds; Holm across all model pairs per condition/mode/metric; NLS deterministic values repeated for pairing only")

def run_legacy(dataset="panasonic"):
    """Legacy entry points now execute/resume the canonical suite."""
    from exp_unified import main
    main(["--dataset", dataset])
