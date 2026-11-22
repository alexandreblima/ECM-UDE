# Part C — Replication across M50T cells

## Selection and execution

Rule C0 selected the first N25 block after the diagnostics. Seven cells were eligible (G1, V4, V5, W3, W7, W8, and W10); W4, W5, and W9 were excluded for the reasons documented in `c0_audit.md` and `c0_segments.json`. For each cell, the first complete UDDS with a causal reference in the same file was used as the source, and eligible complete UDDS profiles in the same block were used as targets.

Training produced 105 neural checkpoints (seven cells, three variants, and five seeds per variant). The 105 history entries and 105 checkpoint hashes are in the manifest; all checkpoint hashes were verified. Positive raw current corresponds to charging and was converted to positive discharge. Capacity and OCV came from the cell's own C/20 discharge diagnostic; z0 was integrated from the SOC=0,8 rest using only current prior to the start of the profile.

## C2 inference

For each cell, the differences are the median across five seeds of the difference in continuous MAE aggregated over targets. The primary test compares against physics-gradient and the secondary test against ECM-2RC NLS. TOST uses a ±1 mV margin; exact Wilcoxon and Holm were applied separately by reference to the two variants.

| Reference | Variant | n cells | Median difference (mV) | Raw TOST p | Holm TOST p |
|---|---|---:|---:|---:|---:|
| physics-gradient | output residual | 7 | -0.001 | 0.0078125 | 0.015625 |
| physics-gradient | ECM-UDE | 7 | -0.482 | 0.148438 | 0.148438 |
| ECM-2RC NLS | output residual | 7 | -0.143 | 0.0078125 | 0.015625 |
| ECM-2RC NLS | ECM-UDE | 7 | -0.643 | 0.148438 | 0.148438 |

Output residual is equivalent to both references within the margin after Holm correction. For ECM-UDE, equivalence was not established.

| Variant | Between-cell variance (mV²) | Mean between-seed variance (mV²) | Ratio |
|---|---:|---:|---:|
| output residual | 6.72999e-07 | 7.91712e-06 | 0.0850 |
| ECM-UDE | 0.350009 | 0.0077072 | 45.4133 |

## Differences by cell (mV)

| Cell | Variant | vs. physics-gradient | vs. ECM-2RC NLS |
|---|---|---:|---:|
| G1 | output residual | -0.004 | -0.960 |
| G1 | ECM-UDE | -2.047 | -3.004 |
| V4 | output residual | -0.001 | -0.199 |
| V4 | ECM-UDE | -0.550 | -0.785 |
| V5 | output residual | -0.001 | -0.141 |
| V5 | ECM-UDE | -0.456 | -0.636 |
| W3 | output residual | -0.001 | -0.218 |
| W3 | ECM-UDE | -0.620 | -0.820 |
| W7 | output residual | -0.001 | -0.125 |
| W7 | ECM-UDE | -0.400 | -0.534 |
| W8 | output residual | -0.001 | -0.143 |
| W8 | ECM-UDE | -0.454 | -0.606 |
| W10 | output residual | -0.001 | -0.137 |
| W10 | ECM-UDE | -0.482 | -0.643 |

## Files

The complete results and hash manifest are in `run/`: `metrics_by_seed.csv`, `cell_differences.csv`, `c2_tost.csv`, `variance_ratio.csv`, `cell_differences.pdf`, `training_histories.json`, and `manifest.json`. Rule C0, including selected episodes, exclusions, and input hashes, is in `c0_segments.json`.
