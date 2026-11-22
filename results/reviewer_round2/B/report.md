# Part B — LG HG2 replication

Input, signs, time ranges, initial states, capacity, and OCV are audited in `data_audit.md`; hashes of the files used are in `manifest.json`. The defect in HWFET 25 °C was recorded (6002 samples, 600,007 s, maximum absolute current 0 A), and the condition was replaced by Mixed1 25 °C. The eight targets are Mixed1, LA92, and US06 at 25 °C and UDDS at −20, −10, 0, 10, and 40 °C; the source UDDS 25 °C is evaluated only on validation.

The C/20 discharge branch of `549_C20DisCh` at 25 °C initialized the OCV curve; capacity `Cap_1C` = 2,72639 Ah. Positive current means charging. MAT inputs were sanitized and resampled at 0,1 s according to the protocol.

## Continuous results

| Condition | Model | Mean MAE ± SD (mV) | Mean RMSE ± SD (mV) | Mean maximum ± SD (mV) |
|---|---|---:|---:|---:|
| UDDS_25C_source | ecm_2rc_nls | 80.692 (deterministic/contextual) | 100.381 (deterministic/contextual) | 197.123 (deterministic/contextual) |
| UDDS_25C_source | physics_gradient | 81.247 ± 0.323 | 101.146 ± 0.349 | 198.784 ± 0.573 |
| UDDS_25C_source | output_residual | 81.143 ± 0.312 | 101.037 ± 0.321 | 198.607 ± 0.508 |
| UDDS_25C_source | ecm_ude | 79.988 ± 0.335 | 100.205 ± 0.332 | 197.893 ± 0.489 |
| Mixed1_25C | ecm_2rc_nls | 32.213 (deterministic/contextual) | 63.626 (deterministic/contextual) | 199.565 (deterministic/contextual) |
| Mixed1_25C | physics_gradient | 32.857 ± 0.094 | 64.207 ± 0.199 | 217.909 ± 0.544 |
| Mixed1_25C | output_residual | 32.844 ± 0.089 | 64.146 ± 0.180 | 217.742 ± 0.488 |
| Mixed1_25C | ecm_ude | 32.628 ± 0.087 | 63.783 ± 0.179 | 216.954 ± 0.546 |
| LA92_25C | ecm_2rc_nls | 31.652 (deterministic/contextual) | 63.103 (deterministic/contextual) | 210.993 (deterministic/contextual) |
| LA92_25C | physics_gradient | 32.185 ± 0.094 | 63.634 ± 0.192 | 212.658 ± 0.574 |
| LA92_25C | output_residual | 32.158 ± 0.094 | 63.575 ± 0.176 | 212.480 ± 0.509 |
| LA92_25C | ecm_ude | 31.822 ± 0.096 | 63.163 ± 0.178 | 211.736 ± 0.490 |
| US06_25C | ecm_2rc_nls | 46.448 (deterministic/contextual) | 83.797 (deterministic/contextual) | 220.008 (deterministic/contextual) |
| US06_25C | physics_gradient | 47.703 ± 0.092 | 84.572 ± 0.229 | 221.664 ± 0.572 |
| US06_25C | output_residual | 47.670 ± 0.084 | 84.500 ± 0.204 | 221.488 ± 0.508 |
| US06_25C | ecm_ude | 47.635 ± 0.072 | 84.194 ± 0.195 | 220.734 ± 0.489 |
| UDDS_-20C | ecm_2rc_nls | 207.132 (deterministic/contextual) | 257.075 (deterministic/contextual) | 816.158 (deterministic/contextual) |
| UDDS_-20C | physics_gradient | 207.476 ± 0.107 | 257.436 ± 0.103 | 815.937 ± 0.272 |
| UDDS_-20C | output_residual | 207.463 ± 0.099 | 257.425 ± 0.095 | 815.933 ± 0.276 |
| UDDS_-20C | ecm_ude | 202.063 ± 0.753 | 252.424 ± 0.696 | 810.049 ± 0.960 |
| UDDS_-10C | ecm_2rc_nls | 103.203 (deterministic/contextual) | 128.881 (deterministic/contextual) | 559.366 (deterministic/contextual) |
| UDDS_-10C | physics_gradient | 103.506 ± 0.091 | 129.182 ± 0.085 | 561.122 ± 0.286 |
| UDDS_-10C | output_residual | 103.503 ± 0.089 | 129.180 ± 0.084 | 561.115 ± 0.291 |
| UDDS_-10C | ecm_ude | 99.798 ± 0.423 | 125.675 ± 0.401 | 557.074 ± 0.596 |
| UDDS_0C | ecm_2rc_nls | 49.068 (deterministic/contextual) | 61.018 (deterministic/contextual) | 300.838 (deterministic/contextual) |
| UDDS_0C | physics_gradient | 49.363 ± 0.085 | 61.301 ± 0.084 | 302.449 ± 0.289 |
| UDDS_0C | output_residual | 49.362 ± 0.086 | 61.304 ± 0.086 | 302.442 ± 0.298 |
| UDDS_0C | ecm_ude | 47.062 ± 0.258 | 59.065 ± 0.256 | 300.056 ± 0.410 |
| UDDS_10C | ecm_2rc_nls | 21.155 (deterministic/contextual) | 26.582 (deterministic/contextual) | 142.504 (deterministic/contextual) |
| UDDS_10C | physics_gradient | 21.441 ± 0.060 | 26.890 ± 0.064 | 149.557 ± 0.329 |
| UDDS_10C | output_residual | 21.437 ± 0.066 | 26.892 ± 0.068 | 149.650 ± 0.318 |
| UDDS_10C | ecm_ude | 20.519 ± 0.128 | 25.992 ± 0.128 | 148.238 ± 0.463 |
| UDDS_40C | ecm_2rc_nls | 17.120 (deterministic/contextual) | 25.176 (deterministic/contextual) | 93.752 (deterministic/contextual) |
| UDDS_40C | physics_gradient | 17.141 ± 0.103 | 25.396 ± 0.144 | 95.045 ± 0.476 |
| UDDS_40C | output_residual | 17.114 ± 0.107 | 25.349 ± 0.140 | 94.889 ± 0.437 |
| UDDS_40C | ecm_ude | 15.054 ± 0.314 | 23.442 ± 0.301 | 91.606 ± 0.650 |

### Contextual LSTM (windowed)

| Condition | MAE (mV) | RMSE (mV) | Maximum error (mV) |
|---|---:|---:|---:|
| UDDS_25C_source | 87.286 | 122.276 | 576.187 |
| Mixed1_25C | 46.559 | 83.774 | 537.155 |
| LA92_25C | 36.859 | 70.033 | 479.581 |
| US06_25C | 71.025 | 105.047 | 730.022 |
| UDDS_-20C | 266.536 | 312.783 | 909.812 |
| UDDS_-10C | 166.825 | 190.985 | 766.138 |
| UDDS_0C | 109.178 | 123.234 | 424.553 |
| UDDS_10C | 64.069 | 78.522 | 513.351 |
| UDDS_40C | 54.334 | 67.340 | 459.798 |

Exact paired comparisons and raw/Holm-adjusted p-values are in `pairwise_wilcoxon.csv`; TOST ±1 mV equivalence comparisons (signed differences, tails, and raw/adjusted maxima) are in `tost_vs_reference.csv`. The families cover the eight target conditions plus source validation. LSTM is contextual and has no comparable continuous simulation; therefore it appears only in windowed metrics.

## A1 and A2 ablations

A1 froze the first-order ECM physics; before and after hashes and zero gradients are in `a1/freeze_audit.json`. TOST against NLS and the paired test against joint training are in `a1_nls_tost.csv` and `a1_joint_paired.csv`.

| Variant | A1 equivalent to NLS | Significant A1 vs. joint difference |
|---|---:|---:|
| output_residual | 9/9 | 0/9 |
| ecm_ude | 9/9 | 5/9 |

A2 exactly reproduced the jointly trained checkpoint metrics before zeroing the network; checkpoint hashes are in `a2/manifest.json`. MAE/RMSE/maximum error differences and TOST-Holm network vs. zeroed and zeroed vs. physics-gradient are in `a2_network_effect_tost.csv` and `a2_zeroed_vs_physics_gradient_tost.csv`.

| Variant | Metric | Network vs. zeroed equivalent | Zeroed vs. physics-gradient equivalent |
|---|---|---:|---:|
| output_residual | mae | 9/9 | 9/9 |
| output_residual | rmse | 9/9 | 9/9 |
| output_residual | max_error | 9/9 | 9/9 |
| ecm_ude | mae | 2/9 | 1/9 |
| ecm_ude | rmse | 1/9 | 0/9 |
| ecm_ude | max_error | 1/9 | 0/9 |

## Figures and files

- `learning_curves.pdf`: median across seeds and minimum–maximum range; vertical lines indicate the median selected epoch.
- `a1_learning_curves.pdf`: frozen-ablation history.
- `base/metrics_by_seed.csv`, `base/training_histories.json`, `base/nls_fits.json`, `base/manifest.json`.
- `a1/metrics_by_seed.csv`, `a1/freeze_audit.json`, `a1/manifest.json`; `a2/zeroed_metrics.csv`, `a2/manifest.json`.
- Generate figures, tests, and report: `python reviewer_round2_b_report.py`.
