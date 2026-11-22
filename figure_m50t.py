"""Render the M50T cell-level comparison from the saved C results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figures_ieee as ieee

ROOT = Path(__file__).resolve().parent
CELLS = ("G1", "V4", "V5", "W3", "W7", "W8", "W10")


def main() -> None:
    ieee.Figures()
    source = ROOT / "results/reviewer_round2/C/run/cell_differences.csv"
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if ieee.digest(source) != manifest["output_sha256"][source.name]:
        raise ValueError("Cell differences hash mismatch")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True, layout="none")
    fig.subplots_adjust(left=.145, right=.985, bottom=.19, top=.83, wspace=.10)
    for ax, model, name, panel in zip(
        axes, ("output_residual", "ecm_ude"), ("Output residual", "ECM-UDE"), ("a", "b")
    ):
        ax.axhspan(-1, 1, color=".90", zorder=0)
        ax.axhline(0, color="black", linestyle=":", linewidth=.8)
        for index, cell in enumerate(CELLS):
            records = [r for r in rows if r["cell"] == cell and r["variant"] == model
                       and r["reference"] == "physics_gradient"]
            if len(records) != 1:
                raise ValueError(f"Expected one cell comparison: {cell}/{model}")
            record = records[0]
            seeds = np.asarray(json.loads(record["seed_differences_mV"]), dtype=float)
            median = float(record["median_difference_mV"])
            if len(seeds) != 5 or not np.isfinite(seeds).all() or not np.isclose(np.median(seeds), median, atol=1e-12):
                raise ValueError(f"Invalid saved cell differences: {cell}/{model}")
            ax.scatter(index + np.linspace(-.16, .16, 5), seeds, s=12, color=".35", zorder=2,
                       label="Seeds" if index == 0 else None)
            ax.scatter(index, median, s=32, marker="D", color="black", edgecolor="white", linewidth=.5,
                       zorder=3, label="Median" if index == 0 else None)
        ax.set_xticks(range(len(CELLS)), CELLS)
        ax.set_xlim(-.5, len(CELLS)-.5)
        ax.set_ylim(-2.6, 1.4)
        ax.text(.02, .98, f"({panel})", transform=ax.transAxes, ha="left", va="top")
        center = (ax.get_position().x0 + ax.get_position().x1) / 2
        fig.text(center, .89, name, ha="center", va="center")
    axes[0].set_ylabel("MAE difference from physics-gradient (mV)")
    axes[1].legend(loc="upper right", frameon=False, handletextpad=.3, borderpad=.1, labelspacing=.1)
    output = ROOT / "figures_ieee"
    output.mkdir(exist_ok=True)
    destination = output / "fig_m50t_cells.pdf"
    fig.savefig(destination, format="pdf", bbox_inches=None, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    ieee.inspect_pdf(destination, 6.0)
    print(f"{destination}: SHA-256 {ieee.digest(destination)}")


if __name__ == "__main__":
    main()
