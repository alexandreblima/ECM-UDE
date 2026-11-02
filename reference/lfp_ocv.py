"""Independent OCV branches and signed ESC amplitude, before dynamic fitting."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression

from data_lfp import (CellData, FloatArray, OUTPUT, PROTOCOL_PATH, load_a123,
                      load_mendeley, protocol, save_json, setup_logging, sha256)

LOGGER = logging.getLogger(__name__)


def monotonic_points(z: FloatArray, voltage: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(z)
    z, voltage = z[order], voltage[order]
    labels = np.minimum((z * 1001).astype(int), 1000)
    points = [(float(np.median(z[labels == k])), float(np.median(voltage[labels == k])))
              for k in np.unique(labels)]
    result = np.asarray(points)
    # Preserve measured SOC endpoints, not an extrapolation toward voltage cutoffs.
    result[0] = (z[0], voltage[0])
    result[-1] = (z[-1], voltage[-1])
    fitted = isotonic_regression(result[:, 1], increasing=True).x
    return result[:, 0], fitted


@dataclass
class OCVCurves:
    charge_z: FloatArray
    charge_V: FloatArray
    discharge_z: FloatArray
    discharge_V: FloatArray

    def __post_init__(self) -> None:
        for z, voltage in ((self.charge_z, self.charge_V),
                           (self.discharge_z, self.discharge_V)):
            if len(z) < 2 or np.any(np.diff(z) <= 0) or not np.isfinite(voltage).all():
                raise ValueError("OCV knots must be finite with strictly increasing SOC")
        self.lo = float(max(self.charge_z[0], self.discharge_z[0]))
        self.hi = float(min(self.charge_z[-1], self.discharge_z[-1]))
        if self.lo >= self.hi or not (self.lo <= 0.2 and self.hi >= 0.8):
            raise ValueError("OCV branches must overlap and cover SOC20-80")
        self._charge = PchipInterpolator(self.charge_z, self.charge_V, extrapolate=False)
        self._discharge = PchipInterpolator(self.discharge_z, self.discharge_V, extrapolate=False)

    def evaluate(self, z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        common_z = np.clip(z, self.lo, self.hi)
        charge = np.asarray(self._charge(common_z), dtype=float)
        discharge = np.asarray(self._discharge(common_z), dtype=float)
        return charge, discharge, (charge + discharge) / 2, (charge - discharge) / 2

    def to_dict(self) -> dict:
        return {"charge_z": self.charge_z.tolist(), "charge_V": self.charge_V.tolist(),
                "discharge_z": self.discharge_z.tolist(), "discharge_V": self.discharge_V.tolist(),
                "common_SOC_interval": [self.lo, self.hi]}

    @classmethod
    def from_dict(cls, value: dict) -> OCVCurves:
        return cls(*(np.asarray(value[k], dtype=float) for k in
                     ("charge_z", "charge_V", "discharge_z", "discharge_V")))


def build_curves(cell: CellData) -> OCVCurves:
    if cell.name == "cell_A_a123":
        charge_z, charge_v = monotonic_points(cell.ocv_charge_z, cell.ocv_charge_V)
        dis_z, dis_v = monotonic_points(cell.ocv_discharge_z, cell.ocv_discharge_V)
    else:
        charge_z, charge_v = cell.ocv_charge_z, cell.ocv_charge_V
        dis_z, dis_v = cell.ocv_discharge_z, cell.ocv_discharge_V
    return OCVCurves(charge_z, charge_v, dis_z, dis_v)


def amplitude(curves: OCVCurves) -> dict:
    z = np.linspace(0.2, 0.8, 6001)
    m = curves.evaluate(z)[3] * 1000
    return {"SOC_interval": [0.2, 0.8], "grid_points": len(z),
            "M_signed_min_mV": float(m.min()), "M_signed_max_mV": float(m.max()),
            "M_max_abs_mV": float(np.abs(m).max()), "M_peak_to_peak_mV": float(np.ptp(m)),
            "M_mean_abs_mV": float(np.abs(m).mean()),
            "M_RMS_mV": float(np.sqrt(np.mean(m * m))),
            "negative_M_fraction_on_grid": float(np.mean(m < 0)),
            "definition": "M=(OCV_charge-OCV_discharge)/2; amplitude for ratios is max(abs(M)) on SOC20-80"}


def write_ocv(cell: CellData) -> None:
    curves = build_curves(cell)
    out = OUTPUT / cell.name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"cell": cell.name, "qnom_Ah": cell.qnom_Ah, "points": curves.to_dict(),
            "plateau_amplitude": amplitude(curves), "protocol": protocol(),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "ocv_source_sha256": sha256(Path(__file__).resolve()),
            "loader_sha256": sha256(Path(__file__).resolve().with_name("data_lfp.py")),
            "input_sha256": cell.metadata["source_files"],
            "method": protocol()["ocv_A" if cell.name == "cell_A_a123" else "ocv_B"],
            "caveats": cell.metadata["caveats"]}
    save_json(out / "ocv_manifest.json", meta)
    z = np.linspace(curves.lo, curves.hi, 1001)
    chg, dis, mean, m = curves.evaluate(z)
    np.savetxt(out / "ocv_curves.csv", np.column_stack((z, chg, dis, mean, m)),
               delimiter=",", header="soc,OCV_chg_V,OCV_dis_V,OCV_mean_V,M_V", comments="")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(z, chg, label="OCV charge")
    axes[0].plot(z, dis, label="OCV discharge")
    axes[0].plot(z, mean, "--", label="OCV mean")
    axes[0].scatter(curves.charge_z, curves.charge_V, s=8, alpha=0.5)
    axes[0].scatter(curves.discharge_z, curves.discharge_V, s=8, alpha=0.5)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend()
    axes[1].plot(z, m * 1000)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].axvspan(0.2, 0.8, color="gray", alpha=0.15)
    axes[1].set(xlabel="SOC (common measured-Ah scale)", ylabel="Signed M (mV)")
    fig.suptitle(cell.name + " | 25 C nominal")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"ocv_and_M.{suffix}", dpi=160)
    plt.close(fig)
    LOGGER.info("%s amplitude=%s", cell.name, meta["plateau_amplitude"])


def main() -> None:
    setup_logging("stage2_ocv")
    for loader in (load_a123, load_mendeley):
        write_ocv(loader())


if __name__ == "__main__":
    main()
