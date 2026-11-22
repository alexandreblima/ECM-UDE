# Part C — Replication across M50T cells

Selection was applied to the first N25 block after the diagnostics. The source is the first complete UDDS profile with a causal SOC rest reference in the same file; all later complete profiles in the same block with an equivalent reference are targets. Exclusions W4, W5, and W9 and episode identifiers/rows are in `c0_audit.md` and `c0_segments.json`.

Positive raw current corresponds to charging and was converted to the canonical positive-discharge convention. Capacity and OCV use the cell's own C/20 discharge diagnostic; z0 starts at the SOC=0,8 rest and integrates only current prior to the start of each profile. Data and checkpoint hashes are in `manifest.json`.

## Primary and secondary inference

Each inferential observation is the median across five seeds of the difference in continuous MAE aggregated over targets. The primary comparison is against physics-gradient; the secondary is against ECM-2RC NLS. TOST uses a ±1 mV margin and exact Wilcoxon, with Holm applied separately by reference over the two variants.

| Reference | Variant | n cells | Median difference (mV) | Raw TOST p | Holm TOST p |
|---|---|---:|---:|---:|---:|
| physics_gradient | output_residual | 7 | -0.001 | 0.0078125 | 0.015625 |
| physics_gradient | ecm_ude | 7 | -0.482 | 0.148438 | 0.148438 |
| ecm_2rc_nls | output_residual | 7 | -0.143 | 0.0078125 | 0.015625 |
| ecm_2rc_nls | ecm_ude | 7 | -0.643 | 0.148438 | 0.148438 |

## Dispersion

| Variant | Between-cell variance (mV²) | Mean between-seed variance (mV²) | Ratio |
|---|---:|---:|---:|
| output_residual | 6.72999e-07 | 7.91712e-06 | 0.0850055 |
| ecm_ude | 0.350009 | 0.0077072 | 45.4133 |

## Differences by cell

| Cell | Variant | vs. physics-gradient (mV) | vs. ECM-2RC (mV) |
|---|---|---:|---:|
| G1 | output_residual | -0.004 | -0.960 |
| G1 | ecm_ude | -2.047 | -3.004 |
| V4 | output_residual | -0.001 | -0.199 |
| V4 | ecm_ude | -0.550 | -0.785 |
| V5 | output_residual | -0.001 | -0.141 |
| V5 | ecm_ude | -0.456 | -0.636 |
| W3 | output_residual | -0.001 | -0.218 |
| W3 | ecm_ude | -0.620 | -0.820 |
| W7 | output_residual | -0.001 | -0.125 |
| W7 | ecm_ude | -0.400 | -0.534 |
| W8 | output_residual | -0.001 | -0.143 |
| W8 | ecm_ude | -0.454 | -0.606 |
| W10 | output_residual | -0.001 | -0.137 |
| W10 | ecm_ude | -0.482 | -0.643 |

## Files

- `metrics_by_seed.csv`: continuous metrics by cell, variant, seed, and target.
- `cell_differences.csv`: differences by seed and median by cell.
- `c2_tost.csv`: raw and Holm-adjusted p-values.
- `variance_ratio.csv`: between-cell and between-seed dispersion.
- `cell_differences.pdf`: differences by seed and median, with a ±1 mV band.
