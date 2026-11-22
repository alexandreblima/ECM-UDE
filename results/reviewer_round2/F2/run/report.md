# Exploratory corrections to Part F

All G1–G3 analyses are exploratory; no model was retrained. Amendment 010 was registered before the recalculations. The original C and F files were preserved.

## G1 — Aging on the same basis

Source MAE uses all points of complete C target episodes, with RC states reset only at the start of each episode and aggregation weighted by sample count. The same rule was used for later F1 blocks. Differences are later − source, paired by seed; the table shows the median across seeds per cell, followed by the median across cells.

Without a later C/20 available, the updated-capacity arm is identical to fixed calibration.

| Cell | ECM-2RC (mV) | Physics-gradient (mV) | Output residual (mV) | ECM-UDE (mV) |
|---|---:|---:|---:|---:|
| G1 | -12.601 | -12.595 | -12.593 | -12.520 |
| V4 | -2.087 | -1.651 | -1.653 | -0.891 |
| W3 | -0.287 | -0.174 | -0.175 | -0.010 |
| W7 | 0.436 | 1.049 | 1.047 | 2.019 |
| W8 | -1.202 | -0.610 | -0.611 | 0.312 |
| W10 | -0.984 | -0.292 | -0.295 | 0.669 |
| **Median (n=6)** | **-1.093** | **-0.451** | **-0.453** | **0.151** |

| Model | Median source MAE (mV) | Median later MAE (mV) | Median paired increase (mV) |
|---|---:|---:|---:|
| ecm_2rc_nls | 14.723 | 14.143 | -1.093 |
| physics_gradient | 14.562 | 14.667 | -0.451 |
| output_residual | 14.561 | 14.662 | -0.453 |
| ecm_ude | 14.063 | 15.266 | 0.151 |

Position 1 has six cells; V5 has no later block and available position 2 has no eligible episode. `g1_source_metrics.csv` and `g1_aging_comparison.csv` contain metrics and differences by seed, with sample counts. Model contrasts in the later block did not change; the correction changes only the comparison with the source.

## G2 — HPPC within the identification range

The continuous simulation starts at rest after full charge at SOC=1, with initial RC states zero; the mask is applied only afterward. To ensure paired comparisons on the same samples, the mask uses nominal coulomb counting from F and the intersection of the cell's `soc_train_min`–`soc_train_max` ranges for 1RC and 2RC; the stored ranges are equal in this dataset. Neural models integrate their own SOC with learned η, which can differ slightly from this common reference. Pulses and rests use the F definitions.

W3 is retained in the seven-cell analysis; HPPC precedes its C/20 and uses another channel, so association with the same diagnostic could not be confirmed. The analysis without W3 is a separate sensitivity analysis with Holm recalculated.

| Cell | Minimum SOC | Maximum SOC | n in range | Fraction in range | n pulses | n rests |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 0.11669 | 0.60223 | 166129 | 41.15% | 811 | 147473 |
| V4 | 0.11202 | 0.60083 | 202610 | 50.16% | 900 | 183852 |
| V5 | 0.11064 | 0.60049 | 202626 | 50.17% | 900 | 183852 |
| W3 | 0.10502 | 0.59885 | 202596 | 50.26% | 900 | 183852 |
| W7 | 0.11384 | 0.60140 | 166615 | 41.27% | 900 | 147864 |
| W8 | 0.11307 | 0.60117 | 202614 | 50.16% | 901 | 183854 |
| W10 | 0.11148 | 0.60072 | 202617 | 50.17% | 900 | 183852 |

| Model | Total in range (mV) | Pulses in range (mV) | Rests in range (mV) |
|---|---:|---:|---:|
| ecm_2rc_nls | 163.525 | 158.845 | 161.164 |
| physics_gradient | 164.217 | 159.204 | 161.590 |
| output_residual | 164.216 | 159.204 | 161.590 |
| ecm_ude | 166.049 | 161.213 | 163.785 |

Cells are the inferential unit; for each cell, the median of the five differences paired by seed is used. The tests below are exploratory: exact Wilcoxon, TOST ±1 mV, and Holm over the two variants separately for each mask, reference, and cohort.

| Cohort | Mask | Reference | Variant | n | Median Δ (mV) | Raw TOST p | Holm p | Raw / Holm superiority p | Raw / Holm inferiority p |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| all | in_range | physics_gradient | output_residual | 7 | -0.001 | 0.0078125 | 0.015625 | 0.289062 / 0.578125 | 0.765625 / 0.765625 |
| all | in_range | physics_gradient | ecm_ude | 7 | 1.919 | 1 | 1 | 1 / 1 | 0.0078125 / 0.015625 |
| all | in_range | ecm_2rc_nls | output_residual | 7 | 0.712 | 0.0078125 | 0.015625 | 1 / 1 | 0.0078125 / 0.015625 |
| all | in_range | ecm_2rc_nls | ecm_ude | 7 | 2.530 | 1 | 1 | 1 / 1 | 0.0078125 / 0.015625 |
| all | pulse_in_range | physics_gradient | output_residual | 7 | -0.001 | 0.0078125 | 0.015625 | 0.234375 / 0.46875 | 0.8125 / 0.8125 |
| all | pulse_in_range | physics_gradient | ecm_ude | 7 | 2.110 | 1 | 1 | 1 / 1 | 0.0078125 / 0.015625 |
| all | pulse_in_range | ecm_2rc_nls | output_residual | 7 | 0.359 | 0.0078125 | 0.015625 | 1 / 1 | 0.0078125 / 0.015625 |
| all | pulse_in_range | ecm_2rc_nls | ecm_ude | 7 | 2.388 | 1 | 1 | 1 / 1 | 0.0078125 / 0.015625 |
| all | rest_in_range | physics_gradient | output_residual | 7 | -0.001 | 0.0078125 | 0.015625 | 0.234375 / 0.46875 | 0.8125 / 0.8125 |
| all | rest_in_range | physics_gradient | ecm_ude | 7 | 2.202 | 1 | 1 | 1 / 1 | 0.0078125 / 0.015625 |
| all | rest_in_range | ecm_2rc_nls | output_residual | 7 | 0.419 | 0.0078125 | 0.015625 | 1 / 1 | 0.0078125 / 0.015625 |
| all | rest_in_range | ecm_2rc_nls | ecm_ude | 7 | 2.496 | 1 | 1 | 1 / 1 | 0.0078125 / 0.015625 |
| without_W3 | in_range | physics_gradient | output_residual | 6 | -0.000 | 0.015625 | 0.03125 | 0.421875 / 0.84375 | 0.65625 / 0.65625 |
| without_W3 | in_range | physics_gradient | ecm_ude | 6 | 1.904 | 1 | 1 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | in_range | ecm_2rc_nls | output_residual | 6 | 0.722 | 0.015625 | 0.03125 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | in_range | ecm_2rc_nls | ecm_ude | 6 | 2.527 | 1 | 1 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | pulse_in_range | physics_gradient | output_residual | 6 | -0.001 | 0.015625 | 0.03125 | 0.34375 / 0.6875 | 0.71875 / 0.71875 |
| without_W3 | pulse_in_range | physics_gradient | ecm_ude | 6 | 2.079 | 1 | 1 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | pulse_in_range | ecm_2rc_nls | output_residual | 6 | 0.357 | 0.015625 | 0.03125 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | pulse_in_range | ecm_2rc_nls | ecm_ude | 6 | 2.378 | 1 | 1 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | rest_in_range | physics_gradient | output_residual | 6 | -0.000 | 0.015625 | 0.03125 | 0.34375 / 0.6875 | 0.71875 / 0.71875 |
| without_W3 | rest_in_range | physics_gradient | ecm_ude | 6 | 2.187 | 1 | 1 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | rest_in_range | ecm_2rc_nls | output_residual | 6 | 0.419 | 0.015625 | 0.03125 | 1 / 1 | 0.015625 / 0.03125 |
| without_W3 | rest_in_range | ecm_2rc_nls | ecm_ude | 6 | 2.474 | 1 | 1 | 1 / 1 | 0.015625 / 0.03125 |

### Full F HPPC: not interpretable because of OCV extrapolation

The original full-profile numbers are retained here for traceability, not as an interpretable measure of the neural contribution: the fitted OCV is extrapolated outside the identified range.

| Model | Full total (mV) | SOC 0,2–0,8 (mV) | Outside (mV) | Full pulses (mV) | Full rests (mV) |
|---|---:|---:|---:|---:|---:|
| ecm_2rc_nls | 816.885 | 184.727 | 1705.058 | 416.777 | 837.477 |
| physics_gradient | 817.293 | 185.382 | 1704.978 | 417.023 | 837.601 |
| output_residual | 817.302 | 185.387 | 1704.984 | 417.029 | 837.607 |
| ecm_ude | 820.938 | 187.889 | 1710.323 | 420.360 | 841.484 |

## G3 — Synthesis and statistical resolution

The synthesis retains the preregistered F1 model contrast, but the corrected G1 increase and G2 tests are exploratory; the HPPC row now uses the identification range. For A4a/A4b, five seeds give a minimum one-sided p of 1/32; Holm over 18 conditions gives a minimum adjusted p of 18/32=0,5625. Counts of 0/0/0 do not demonstrate absence of an effect.

Detailed files: `g2_hppc_metrics.csv`, `g2_cell_differences.csv`, `g2_tests.csv`, and `manifest.json`; previous versions of the synthesis are in `run/`.
