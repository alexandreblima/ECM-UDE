"""Render the loss-change figure from the verified per-seed histories."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figures_ieee as ieee
from public_integrity import verify

ROOT = Path(__file__).resolve().parent
MODELS = {
    "physics_gradient": (ieee.COLORS[0], "-", "o", "Physics gradient"),
    "output_residual": (ieee.COLORS[1], "--", "s", "Output residual"),
    "ecm_ude": (ieee.COLORS[2], "-.", "^", "ECM-UDE"),
}


def main():
    ieee.Figures()
    audit = ROOT/"results/training_audit/epochs.csv"
    verify(audit)
    with audit.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    histories = {}
    for row in rows:
        if row["dataset"] not in ("panasonic", "calce") or row["model"] not in MODELS:
            continue
        path = ROOT/row["history_path"]
        if ieee.digest(path) != row["history_sha256"]:
            raise ValueError(f"History hash mismatch: {path}")
        history = json.loads(path.read_text())
        if history["best_epoch"] != int(row["selected_epoch"]):
            raise ValueError(f"Selected epoch mismatch: {path}")
        histories.setdefault((row["dataset"], row["model"]), []).append(history)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6), layout="none")
    fig.subplots_adjust(left=.12, right=.985, bottom=.12, top=.79, wspace=.26, hspace=.45)
    for row_index, (cell, label) in enumerate((("panasonic", "NCA"), ("calce", "NMC"))):
        for col, loss in enumerate(("train_loss", "val_loss")):
            ax = axes[row_index, col]
            ax.axhline(0, color=".75", linewidth=.5, zorder=0)
            for model, (color, linestyle, marker, name) in MODELS.items():
                runs = histories[cell, model]
                if len(runs) != 10:
                    raise ValueError(f"Expected ten histories for {cell}/{model}")
                curves = [np.asarray(run[loss], dtype=float) for run in runs]
                if any(not len(curve) or not np.isfinite(curve).all() or curve[0] <= 0 for curve in curves):
                    raise ValueError(f"Invalid losses for {cell}/{model}")
                matrix = np.full((len(curves), max(map(len, curves))), np.nan)
                for i, curve in enumerate(curves):
                    matrix[i, :len(curve)] = (curve/curve[0]-1)*100
                x = np.arange(1, matrix.shape[1]+1)
                ax.fill_between(x, np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0),
                                color=color, alpha=.10, linewidth=0)
                width = 1.8 if model == "physics_gradient" else 1
                ax.plot(x, np.nanmedian(matrix, axis=0), color=color, linestyle=linestyle,
                        marker=marker, markersize=3, markevery=3, linewidth=width, label=name)
                ax.axvline(np.median([run["best_epoch"] for run in runs]), color=color,
                           linestyle=linestyle, linewidth=.55, alpha=.45)
            ax.set_xlim(.5, 23.5)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(axis="y", color=".86", linewidth=.45)
            if col == 0:
                ax.set_ylabel(f"{label}\nLoss change from epoch 1 (%)")
    for col, text in enumerate(("Training loss", "Validation loss")):
        ax = axes[0, col]
        fig.text((ax.get_position().x0+ax.get_position().x1)/2, .835, text, ha="center", va="center")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .99), columnspacing=2, handlelength=2.3)
    fig.supxlabel("Epoch", y=.015, fontsize=8)
    output = ROOT/"figures_ieee"
    output.mkdir(exist_ok=True)
    destination = output/"fig_learning_curves.pdf"
    fig.savefig(destination, format="pdf", bbox_inches=None, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    ieee.inspect_pdf(destination, 7.16)
    print(f"{destination}: SHA-256 {ieee.digest(destination)}")


if __name__ == "__main__":
    main()
