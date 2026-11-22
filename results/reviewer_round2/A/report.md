# Part A — Ablations and optimization sensitivity

## A1 — Frozen physics

The experiment fitted 40 models (two variants, two cells, ten seeds). The audit confirmed physical parameters without gradients, invariant values, and identical before/after hashes for all 40 checkpoints. Each variant was compared with the first-order NLS ECM using TOST at ±1 mV, with Holm applied separately by variant to the family of 18 conditions.

- `output_residual` vs. NLS: equivalence in 18/18 conditions after Holm; median of condition differences -0.0023 mV (A1 − NLS).
- `output_residual` A1 vs. joint training: 3/18 conditions with a significant paired difference after Holm; median 0.0412 mV (A1 − joint).
- `ecm_ude` vs. NLS: equivalence in 18/18 conditions after Holm; median of condition differences 0.0769 mV (A1 − NLS).
- `ecm_ude` A1 vs. joint training: 0/18 conditions with a significant paired difference after Holm; median -0.0447 mV (A1 − joint).

Raw and adjusted p-values by condition and seed are in `a1_nls_tost.csv` and `a1_joint_paired.csv`; metrics and hashes by checkpoint are in `a1/metrics_by_seed.csv` and `a1/manifest.json`.

## A2 — Network zeroed after joint training

The 40 original checkpoints were reevaluated without retraining, and their results were exactly reproduced before zeroing the correction. A2 confirmed equivalence of the jointly trained network against the zeroed network for output residual in 18/18 conditions for all three metrics; for ECM-UDE, in 18/18 for MAE/RMSE and 14/18 for maximum error. The zeroed network was equivalent to physics-gradient in the same totals. See `a2_report.md`, A2 tables, and manifests.

## A3 — Diagnostic without early selection

One seed per variant and cell was trained for 30 epochs, without early stopping. The table shows training/validation MAE at the first and last epoch; complete curves are in `a3_train_validation_mae.pdf` and `a3/training_validation_mae.csv`.

| Cell | Variant | Initial → final training (mV) | Initial → final validation (mV) |
|---|---|---:|---:|
| NCA | output_residual | 5.509 → 5.502 | 43.356 → 43.209 |
| NCA | ecm_ude | 5.492 → 5.394 | 43.560 → 45.360 |
| NMC | output_residual | 2.525 → 2.538 | 11.442 → 11.508 |
| NMC | ecm_ude | 2.527 → 2.460 | 11.433 → 11.564 |

## A4 — Starting point and budget

A4a uses initial gain 1,0 and the other original settings. A4b adds a learning rate of 2×10⁻³, up to 150 epochs, and patience 30. The E1 reconciliation showed that the previous table compared NLS, evaluated once over all windows, with `history["train_loss"][-1]`: the mean mini-batch loss during updates in the last epoch. This is not the selected checkpoint's loss; when training ends, the checkpoint with the lowest validation loss is restored. The table below was corrected: each saved checkpoint was reevaluated on the same training windows as NLS, with the same normalization and RC state reset per window. Negative values indicate higher MSE for the neural checkpoint.

| Run | Cell | Variant | Median training MSE reduction at checkpoint | Median online reduction in last epoch |
|---|---|---|---:|---:|
| A4A | NCA | output_residual | -1.98% | -0.43% |
| A4A | NCA | ecm_ude | -39.51% | -1.08% |
| A4A | NMC | output_residual | -17.42% | -0.27% |
| A4A | NMC | ecm_ude | -126.64% | -42.73% |
| A4B | NCA | output_residual | -2.63% | -0.27% |
| A4B | NCA | ecm_ude | -363.78% | -2.25% |
| A4B | NMC | output_residual | -40.26% | -2.17% |
| A4B | NMC | ecm_ude | -85.85% | -39.02% |

Figure `a4_learning_curves.pdf` shows medians across seeds and minimum–maximum ranges. No condition achieved TOST equivalence to NLS after Holm correction. Raw and adjusted p-values, along with signed differences by condition, are in `a4a/tost_vs_nls.csv` and `a4b/tost_vs_nls.csv`.

| Run | Variant | Equivalent after Holm | Median condition difference (mV) | Conditions with improvement >1 mV |
|---|---|---:|---:|---:|
| A4A | output_residual | 0/18 | -0.557 | 3 |
| A4A | ecm_ude | 0/18 | 7.898 | 4 |
| A4B | output_residual | 0/18 | -1.061 | 10 |
| A4B | ecm_ude | 0/18 | -0.036 | 6 |

For selected checkpoints, training MSE did not decrease relative to NLS in any cell-variant group. This does not erase the continuous improvement observed on validation/targets: for example, for ECM-UDE A4b/NCA the median continuous MAE was 11.93 mV higher than NLS on the source training portion and 14.59 mV lower on continuous validation; windowed MSE resets the RC state for each window, whereas continuous evaluation carries the state between samples. The last epoch's online mean underestimated the difference between the saved checkpoint and NLS, especially for A4b/NCA ECM-UDE.

Output residual improved MAE by more than 1 mV in 10/18 conditions (A4b), but no contrast reached TOST-Holm equivalence with NLS. Therefore, the manuscript's conclusion should be restricted to the original procedure, without claiming that no broader initialization or budget can improve target MAE. Detailed evaluation of the four results per seed (normalized training/validation MSE and continuous training/validation MAE), epochs, and hashes is in `../E/a4_checkpoint_metrics.csv` and `../E/a4_loss_reconciliation.csv`.

## Files and reproducibility

- A1: `a1/metrics_by_seed.csv`, `a1/learning_curves.csv`, `a1/freeze_audit.json`, `a1/manifest.json`, `a1_nls_tost.csv`, `a1_joint_paired.csv`.
- A2: `a2_report.md`, `a2_manifest.json`, `a2_network_effect_tost.csv`, `a2_zeroed_vs_physics_gradient_tost.csv`.
- A3: `a3/training_validation_mae.csv`, `a3/checkpoint_hashes.json`, `a3_train_validation_mae.pdf`.
- A4a/A4b: `a4a/` and `a4b/`, including metrics, curves, frozen-physics audit, TOST-Holm, and hashes.
- Generate figures and report: `python reviewer_round2_a_report.py`.
