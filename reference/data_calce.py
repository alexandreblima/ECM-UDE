"""
CALCE Battery Dataset Loader: Samsung INR18650-20R (Protocol v2)
==================================================================

Cell Specification:
  - Model: Samsung INR18650-20R
  - Chemistry: LiNiMnCoO2 (NMC) / Graphite
  - Nominal Capacity: 2.0 Ah (2000 mAh)
  - Voltage Range: 2.5 V to 4.2 V
  - Form Factor: 18650 Cylindrical

Profiles Supported (SP2 cell):
  - Dynamic drive cycles: FUDS, DST, US06, BJDST
  - Operating temperatures: 0 degC, 25 degC, 45 degC
  - Step 7 in Arbin cycler logs represents the dynamic driving schedule.

Protocol v2 Specifications:
  - Sampling grid: Uniform 1.0 Hz (dt = 1.0 s) via interpolation
  - Windowing: Length = 256 s, Stride = 128 s (or configurable)
  - Automatic disk caching to results/calce_cache/ for fast repeated execution.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import openpyxl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from wno_battery.data import WindowConfig, BatteryWindowDataset, compute_norm_stats
except ModuleNotFoundError:
    from data import WindowConfig, BatteryWindowDataset, compute_norm_stats


CALCE_RAW_DIR = Path("../datasets/calce_20r_raw/extracted")
CACHE_DIR = Path("results/calce_cache")

CALCE_NOMINAL_CAPACITY_AH = 2.0  # Samsung 20R

# Mapping of canonical cycle names to (folder_name, file_name, temperature_degC)
CALCE_FILES = {
    # 25 degC
    "FUDS_25C":  ("SP2_25C_FUDS",  "11_06_2015_SP20-2_FUDS_80SOC.xls",  25.0),
    "DST_25C":   ("SP2_25C_DST",   "11_05_2015_SP20-2_DST_80SOC.xls",   25.0),
    "US06_25C":  ("SP2_25C_US06",  "11_11_2015_SP20-2_US06_80SOC.xls",  25.0),
    "BJDST_25C": ("SP2_25C_BJDST", "11_12_2015_SP20-2_BJDST_80SOC.xls", 25.0),
    # 0 degC
    "FUDS_0C":   ("SP2_0C_FUDS",   "02_25_2016_SP20-2_0C_FUDS_80SOC.xls",  0.0),
    "DST_0C":    ("SP2_0C_DST",    "02_24_2016_SP20-2_0C_DST_80SOC.xls",   0.0),
    "US06_0C":   ("SP2_0C_US06",   "02_26_2016_SP20-2_0C_US06_80SOC.xls",  0.0),
    "BJDST_0C":  ("SP2_0C_BJDST",  "02_27_2016_SP20-2_0C_BJDST_80SOC.xls", 0.0),
    # 45 degC
    "FUDS_45C":  ("SP2_45C_FUDS",  "12_15_2015_SP20-2_45C_FUDS_80SOC.xls",  45.0),
    "DST_45C":   ("SP2_45C_DST",   "12_11_2015_SP20-2_45C_DST_80SOC.xls",   45.0),
    "US06_45C":  ("SP2_45C_US06",  "12_16_2015_SP20-2_45C_US06_80SOC.xls",  45.0),
    "BJDST_45C": ("SP2_45C_BJDST", "12_17_2015_SP20-2_45C_BJDST_80SOC.xls", 45.0),
}


def load_calce_cycle(
    cycle_name: str,
    cache_dir: Path = CACHE_DIR,
    target_dt: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Load a CALCE drive cycle as float32 tensors on a uniform 1.0s grid.

    Uses .npz caching for near-instant subsequent loads.
    """
    if cycle_name not in CALCE_FILES:
        raise KeyError(
            f"Unknown CALCE cycle: '{cycle_name}'. Available: {list(CALCE_FILES.keys())}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cycle_name}_dt{target_dt:.1f}.npz"

    if cache_file.is_file():
        cached = np.load(cache_file)
        return {
            "t":      torch.from_numpy(cached["t"]),
            "I":      torch.from_numpy(cached["I"]),
            "V":      torch.from_numpy(cached["V"]),
            "T_cell": torch.from_numpy(cached["T_cell"]),
            "Ah":     torch.from_numpy(cached["Ah"]),
            "SoC":    torch.from_numpy(cached["SoC"]),
        }

    folder_name, file_name, temp_degC = CALCE_FILES[cycle_name]
    excel_path = CALCE_RAW_DIR / folder_name / file_name

    if not excel_path.is_file():
        raise FileNotFoundError(f"CALCE excel file not found: {excel_path}")

    # Read Step 7 from Arbin excel sheet
    with open(excel_path, "rb") as f:
        wb = openpyxl.load_workbook(io.BytesIO(f.read()), read_only=True)

    sheet_name = wb.sheetnames[1] if len(wb.sheetnames) > 1 else wb.sheetnames[0]
    sheet = wb[sheet_name]

    step7_rows = []
    for i, r in enumerate(sheet.iter_rows(values_only=True)):
        if i > 0 and r[4] is not None and int(r[4]) == 7:
            # Test_Time, Current, Voltage, Discharge_Capacity
            step7_rows.append((r[1], r[6], r[7], r[9]))

    if not step7_rows:
        raise RuntimeError(f"No Step 7 records found in {excel_path.name}")

    raw_arr = np.array(step7_rows, dtype=np.float64)
    t_raw = raw_arr[:, 0] - raw_arr[0, 0]
    I_raw = raw_arr[:, 1]
    V_raw = raw_arr[:, 2]
    dischg_ah_raw = raw_arr[:, 3]

    # Resample onto uniform 1.0s grid
    t_max = np.floor(t_raw[-1])
    t_grid = np.arange(0.0, t_max + target_dt, target_dt, dtype=np.float32)
    I_grid = np.interp(t_grid, t_raw, I_raw).astype(np.float32)
    V_grid = np.interp(t_grid, t_raw, V_raw).astype(np.float32)
    dischg_ah_grid = np.interp(t_grid, t_raw, dischg_ah_raw).astype(np.float32)

    # Causal Coulomb counting from 80% SoC:
    # At start of Step 7, cell was discharged 20% (0.4 Ah), so starting SoC is ~0.80.
    # Net amp-hours relative to start of Step 7:
    delta_ah = dischg_ah_grid - dischg_ah_grid[0]
    soc_grid = np.clip(0.80 - (delta_ah / CALCE_NOMINAL_CAPACITY_AH), 0.0, 1.0).astype(np.float32)

    Ah_grid = -delta_ah.astype(np.float32)
    T_cell_grid = np.full_like(t_grid, temp_degC, dtype=np.float32)

    # Save to disk cache
    np.savez_compressed(
        cache_file,
        t=t_grid,
        I=I_grid,
        V=V_grid,
        T_cell=T_cell_grid,
        Ah=Ah_grid,
        SoC=soc_grid,
    )

    return {
        "t":      torch.from_numpy(t_grid),
        "I":      torch.from_numpy(I_grid),
        "V":      torch.from_numpy(V_grid),
        "T_cell": torch.from_numpy(T_cell_grid),
        "Ah":     torch.from_numpy(Ah_grid),
        "SoC":    torch.from_numpy(soc_grid),
    }


def make_calce_dataloaders(
    cycle_name: str,
    window_cfg: Optional[WindowConfig] = None,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
    stats: Optional[dict[str, float]] = None,
) -> tuple[DataLoader, DataLoader, BatteryWindowDataset]:
    """Build train/val dataloaders for a CALCE cycle."""
    if window_cfg is None:
        window_cfg = WindowConfig(length=256, stride=128, normalize=True)

    cycle = load_calce_cycle(cycle_name)
    dataset = BatteryWindowDataset(
        cycle,
        WindowConfig(length=window_cfg.length, stride=window_cfg.stride, normalize=False),
    )

    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    L, S = window_cfg.length, window_cfg.stride
    overlap = max(0, (L + S - 1) // S - 1)
    train_end = max(0, n_train - overlap)
    train_indices = list(range(0, train_end))
    val_indices = list(range(n_train, n))

    if stats is None:
        I_tr, V_tr, T_tr = dataset.raw_samples_for_indices(train_indices)
        stats = compute_norm_stats(I_tr, V_tr, T_tr)

    dataset.renormalize(stats)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, dataset
