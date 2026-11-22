# Part E — Checks and exploratory analyses

All E1–E4 items are exploratory, except the correction to the interpretation of the A4 result: no network was retrained. The E2 sensitivity analysis fits only the expressly requested alternative ECM-2RC NLS.

## E1 — A4 reconciliation (exploratory)

No network was retrained. The old history is not a fixed checkpoint evaluation: `training.py` computes each mini-batch loss during optimizer updates, aggregates those losses over the epoch, and finally restores the checkpoint with the lowest validation loss. Thus, `history['train_loss'][-1]` describes the average trajectory of the last epoch, not the selected checkpoint. NLS was evaluated once on all the same training windows, making the recorded comparison of relative reductions incompatible.

For the four recalculated metrics, each window resets RC states to zero and initializes SOC to that window's value; current, temperature, SOC, and voltage are normalized using statistics computed only over the union of training-window samples. Loss is the mean pointwise MSE between predicted and target voltages normalized by `V_std`. NLS is evaluated statically on the same windows with the same normalization. For continuous simulation, each checkpoint and NLS traverse the source once with RC state carried between samples; training and validation MAE use sample masks covered by their respective windows, excluding the guard interval.

The corrected column in `A/a4_training_loss_summary.csv` uses the saved checkpoint MSE, not the mean loss during the last epoch. The previous last-epoch loss was retained in separate columns for traceability. Values by seed, selected epochs, metrics, and hashes are in `a4_checkpoint_metrics.csv` and `a4_loss_reconciliation.csv`.

| Run | Cell | Variant | Median training MSE reduction at checkpoint (%) | Continuous training ΔMAE vs. NLS (mV) | Continuous validation ΔMAE vs. NLS (mV) |
|---|---|---|---:|---:|---:|
| A4A | NCA | output_residual | -1.98 | 0.182 | -0.970 |
| A4A | NCA | ecm_ude | -39.51 | 2.087 | -4.806 |
| A4A | NMC Samsung | output_residual | -17.42 | 0.490 | -0.293 |
| A4A | NMC Samsung | ecm_ude | -126.64 | 2.303 | -0.089 |
| A4B | NCA | output_residual | -2.63 | 0.228 | -1.191 |
| A4B | NCA | ecm_ude | -363.78 | 11.933 | -14.586 |
| A4B | NMC Samsung | output_residual | -40.26 | 0.975 | -0.295 |
| A4B | NMC Samsung | ecm_ude | -85.85 | 1.683 | -0.352 |

Interpretation: the cited continuous improvement on targets does not imply reduced MSE on training windows. For ECM-UDE A4b/NCA, the selected checkpoint has higher training MSE than NLS and also higher continuous training MAE, but lower continuous validation MAE; its higher training error is hidden when NLS is compared with the last epoch's online mean. Continuous metrics by segment and the four requested values per seed allow the mask/simulation-state difference to be separated from the selection criterion.

## E2 — LG HG2 calibration (exploratory)

Figure `hg2_udds_calibration.pdf` shows the full UDDS 25 °C profile; the MAE of approximately 80 mV is reproduced on temporal validation (the final part of the profile), not on the full profile. Validation error is concentrated at the end of discharge: SOC_0.2_to_0.8_plateau: MAE 8.62 mV; SOC_below_0.2_end_discharge: MAE 95.81 mV.

Measured and predicted voltage uses the saved ECM-2RC NLS continuous simulation; z0 is anchored at 1 after `551_Charge2` (verified rest) and SOC is counted causally. `Cap_1C` is 2.72639 Ah; net discharge in UDDS is 2.59213 Ah (gross discharge 3.81572 Ah and regeneration 1.22359 Ah). Recalculated final SOC is 0.04924; initial/final voltages are 4.18464/3.20104 V.

The OCV comparison uses medians in 0,002 SOC bins and the monotonic envelope; a positive difference means alternative OCV above the C/20 discharge branch. For HPPC, the median of the last 30 s of rests of at least 600 s with |I|≤0,02 A was used as a descriptive estimate of relaxed OCV.

| Comparison | SOC range | Mean Δ (mV) | Median Δ (mV) | Mean absolute error (mV) | Δ range (mV) |
|---|---|---:|---:|---:|---:|
| C20_mean_charge_discharge_minus_discharge | SOC_0.2_to_0.8 | 8.65 | 5.57 | 11.03 | [-11.55, 35.07] |
| C20_mean_charge_discharge_minus_discharge | SOC_below_0.2 | 67.29 | 64.73 | 67.29 | [23.09, 115.99] |
| HPPC_rest_minus_C20_discharge | SOC_0.2_to_0.8 | 14.29 | 14.13 | 14.29 | [5.39, 18.56] |
| HPPC_rest_minus_C20_discharge | SOC_below_0.2 | 41.64 | 37.77 | 41.64 | [0.19, 123.70] |

The sensitivity analysis refits only the 2RC NLS using mean charge/discharge OCV; it retrains no networks. The order of the four models did not change in 9/9 conditions (0 changes). The alternative fit converged to parameters practically identical to the original: maximum absolute change in OCV coefficients 2.95e-08 and R/C parameters 1.86e-05; the identification window starts at SOC 0.230. Errors by condition, before/after orderings, and parameters are in `hg2_nls_mean_ocv_sensitivity.csv` and `hg2_calibration.json`; OCV curves are in `hg2_ocv_curves.csv`/`hg2_ocv_comparison.pdf`.

## E3 — One-sided superiority (exploratory)

All contrasts are variant−reference differences; superiority uses one-sided exact Wilcoxon (`less`), inferiority uses `greater`, and families were Holm-adjusted as described in the CSV. TOST uses ±1 mV. E3 does not change preregistered results.

### M50T — cell as unit

| Variant | Reference | Median d_c (mV) | Min–max (mV) | d_c<0 | d_c<−1 | Raw superiority p | Holm superiority p |
|---|---|---:|---:|---:|---:|---:|---:|
| output_residual | physics_gradient | -0.001 | [-0.004, -0.001] | 7 | 0 | 0.0078125 | 0.015625 |
| ecm_ude | physics_gradient | -0.482 | [-2.047, -0.400] | 7 | 1 | 0.0078125 | 0.015625 |
| output_residual | ecm_2rc_nls | -0.143 | [-0.960, -0.125] | 7 | 0 | 0.0078125 | 0.015625 |
| ecm_ude | ecm_2rc_nls | -0.643 | [-3.004, -0.534] | 7 | 1 | 0.0078125 | 0.015625 |

### LG HG2 — ECM-UDE by condition

| Condition | Reference | Median (mV) | Min–max (mV) | Raw superiority p | Holm superiority p |
|---|---|---:|---:|---:|---:|
| UDDS_25C_source | physics_gradient | -1.227 | [-2.451, -0.530] | 0.000976562 | 0.00878906 |
| Mixed1_25C | physics_gradient | -0.224 | [-0.537, -0.035] | 0.000976562 | 0.00878906 |
| LA92_25C | physics_gradient | -0.356 | [-0.704, -0.166] | 0.000976562 | 0.00878906 |
| US06_25C | physics_gradient | -0.027 | [-0.377, 0.105] | 0.1875 | 0.1875 |
| UDDS_-20C | physics_gradient | -5.126 | [-6.804, -4.333] | 0.000976562 | 0.00878906 |
| UDDS_-10C | physics_gradient | -3.547 | [-4.392, -3.091] | 0.000976562 | 0.00878906 |
| UDDS_0C | physics_gradient | -2.231 | [-2.674, -1.971] | 0.000976562 | 0.00878906 |
| UDDS_10C | physics_gradient | -0.911 | [-1.173, -0.784] | 0.000976562 | 0.00878906 |
| UDDS_40C | physics_gradient | -2.136 | [-2.637, -1.606] | 0.000976562 | 0.00878906 |
| UDDS_25C_source | ecm_2rc_nls | -0.682 | [-1.189, -0.249] | 0.000976562 | 0.00878906 |
| Mixed1_25C | ecm_2rc_nls | 0.415 | [0.262, 0.531] | 1 | 1 |
| LA92_25C | ecm_2rc_nls | 0.179 | [0.035, 0.302] | 1 | 1 |
| US06_25C | ecm_2rc_nls | 1.189 | [1.054, 1.300] | 1 | 1 |
| UDDS_-20C | ecm_2rc_nls | -4.825 | [-6.355, -3.991] | 0.000976562 | 0.00878906 |
| UDDS_-10C | ecm_2rc_nls | -3.305 | [-4.137, -2.761] | 0.000976562 | 0.00878906 |
| UDDS_0C | ecm_2rc_nls | -1.975 | [-2.479, -1.648] | 0.000976562 | 0.00878906 |
| UDDS_10C | ecm_2rc_nls | -0.605 | [-0.883, -0.498] | 0.000976562 | 0.00878906 |
| UDDS_40C | ecm_2rc_nls | -2.120 | [-2.474, -1.580] | 0.000976562 | 0.00878906 |

### A4a/A4b

| Run | Variant | Reference | Conditions with Holm superiority | Median of medians (mV) | Min–max of medians (mV) |
|---|---|---|---:|---:|---:|
| A4A | ecm_ude | ecm_2rc_nls | 0/18 | 8.191 | [-4.319, 24.567] |
| A4A | ecm_ude | physics_gradient | 0/18 | 7.983 | [-5.309, 24.689] |
| A4A | output_residual | ecm_2rc_nls | 0/18 | -0.020 | [-1.012, 2.504] |
| A4A | output_residual | physics_gradient | 0/18 | -0.273 | [-1.012, 0.349] |
| A4B | ecm_ude | ecm_2rc_nls | 0/18 | 0.050 | [-14.099, 6.609] |
| A4B | ecm_ude | physics_gradient | 0/18 | 0.005 | [-13.823, 6.793] |
| A4B | output_residual | ecm_2rc_nls | 0/18 | -0.377 | [-1.348, 2.252] |
| A4B | output_residual | physics_gradient | 0/18 | -0.726 | [-1.348, 1.010] |

Detailed rows by condition/seeds and raw and adjusted p-values for superiority, inferiority, and TOST are in `e3_superiority_tests.csv`; the seven d_c values and values by seed are in `m50t_differences_by_cell.csv`. The supplementary superiority analysis for HG2 variants and the original E1 appears in the file to complete the E4 synthesis.

## E4 — Synthesis for the paper

The table combines preregistered measures with exploratory one-sided tests. `better/equivalent/worse` count superiority, TOST ±1 mV, and inferiority tests with Holm separately; categories are not mutually exclusive. For M50T, inference is one population-level test over seven d_c values, not seven independent tests per cell; the cell-level numbers are descriptive.

`synthesis.csv` contains one row per configuration/variant; `synthesis.tex` uses `booktabs` and `\footnotesize`. The status column separates preregistered TOST from exploratory superiority; for A4b against physics-gradient, both are exploratory.

## Files and provenance

- `a4_checkpoint_metrics.csv`, `a4_loss_reconciliation.csv`: E1 metrics by seed and epoch.
- `hg2_udds_calibration.pdf`, `hg2_ocv_comparison.pdf`, `hg2_ocv_comparison.csv`, `hg2_ocv_curves.csv`, `hg2_nls_mean_ocv_sensitivity.csv`, `hg2_calibration.json`: E2.
- `e3_superiority_tests.csv`, `m50t_differences_by_cell.csv`, `e1_original_comparisons.csv`: E3 and reference comparisons used in E4.
- `synthesis.csv`, `synthesis.tex`: E4.
- `manifest.json`: hashes of inputs, checkpoints, code, and outputs.
