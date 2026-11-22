# Part F — M50T transfer

TOST was preregistered in amendment 009; superiority and inferiority are exploratory. No model was refitted. The inferential unit is the cell; differences use the median across five seeds.

## F1 — Aging

Blocks were ordered by observed chronology, not by N labels. The local copy contains only one C/20 for each eligible cell; therefore, the most recent diagnostic before each block is the source diagnostic. The two arms are numerically identical and do not measure the effect of recalibrating degraded capacity: the difference between arms is zero. Episodes and exclusions are recorded in `selection_audit.json`. The increase is the MAE of the later block minus the aggregate MAE of the C targets in the source block for the same cell, model, and seed; the C aggregate uses the window mask, whereas F uses all points of complete episodes.

| Position | Cells | Model | Median MAE (mV) | Median increase (mV) |
|---:|---:|---|---:|---:|
| 1 | 6 | ecm_2rc_nls | 14.143 | -1.093 |
| 1 | 6 | physics_gradient | 14.667 | -0.451 |
| 1 | 6 | output_residual | 14.662 | -0.453 |
| 1 | 6 | ecm_ude | 15.266 | 0.151 |

| Cell | Later blocks observed | Blocks with eligible episodes |
|---|---:|---:|
| G1 | 1 | 1 |
| V4 | 1 | 1 |
| V5 | 0 | 0 |
| W3 | 1 | 1 |
| W7 | 2 | 1 |
| W8 | 1 | 1 |
| W10 | 1 | 1 |

## F2 — HPPC

The simulation starts at rest after a full CC/CV charge (SOC=1), with RC states set to zero. The pulse mask uses steps of up to 15 s with |I|>0,02 A; rest uses |I|≤0,02 A. Continuous voltage is evaluated at all resampled points; missing/invalid temperature is set to 23 °C. The range 0,2–0,8 is nominal and does not imply that all training data lie within it. For W3, HPPC precedes C/20 and uses another channel; its association with the same diagnostic could not be confirmed, although both precede the source. Joint interpretation for this cell requires caution.

| Model | Median total MAE (mV) | SOC 0,2–0,8 | Outside | Pulses | Rests |
|---|---:|---:|---:|---:|---:|
| ecm_2rc_nls | 816.885 | 184.727 | 1705.058 | 416.777 | 837.477 |
| physics_gradient | 817.293 | 185.382 | 1704.978 | 417.023 | 837.601 |
| output_residual | 817.302 | 185.387 | 1704.984 | 417.029 | 837.607 |
| ecm_ude | 820.938 | 187.889 | 1710.323 | 420.360 | 841.484 |

## Inference

| Analysis | Group | Arm | Reference | Variant | n | Median Δ (mV) | TOST p | Holm p | Raw superiority p | Holm p |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| aging | 1 | fixed | physics_gradient | output_residual | 6 | -0.003 | 0.015625 | 0.03125 | 0.015625 | 0.03125 |
| aging | 1 | fixed | physics_gradient | ecm_ude | 6 | 0.337 | 0.046875 | 0.046875 | 0.71875 | 0.71875 |
| aging | 1 | fixed | ecm_2rc_nls | output_residual | 6 | 0.346 | 0.015625 | 0.03125 | 0.78125 | 1 |
| aging | 1 | fixed | ecm_2rc_nls | ecm_ude | 6 | 0.694 | 0.15625 | 0.15625 | 0.71875 | 1 |
| aging | 1 | updated_capacity | physics_gradient | output_residual | 6 | -0.003 | 0.015625 | 0.03125 | 0.015625 | 0.03125 |
| aging | 1 | updated_capacity | physics_gradient | ecm_ude | 6 | 0.337 | 0.046875 | 0.046875 | 0.71875 | 0.71875 |
| aging | 1 | updated_capacity | ecm_2rc_nls | output_residual | 6 | 0.346 | 0.015625 | 0.03125 | 0.78125 | 1 |
| aging | 1 | updated_capacity | ecm_2rc_nls | ecm_ude | 6 | 0.694 | 0.15625 | 0.15625 | 0.71875 | 1 |
| hppc | full | fixed | physics_gradient | output_residual | 7 | 0.005 | 0.0078125 | 0.015625 | 0.992188 | 1 |
| hppc | full | fixed | physics_gradient | ecm_ude | 7 | 3.550 | 0.992188 | 0.992188 | 0.992188 | 1 |
| hppc | full | fixed | ecm_2rc_nls | output_residual | 7 | 0.420 | 0.0078125 | 0.015625 | 1 | 1 |
| hppc | full | fixed | ecm_2rc_nls | ecm_ude | 7 | 3.903 | 0.992188 | 0.992188 | 0.992188 | 1 |

The tables by cell, seed, and arm are in `aging_metrics.csv`, `hppc_metrics.csv`, and `cell_differences.csv`; one-sided p-values are in `c2_style_tost.csv`; provenance is in `manifest.json`. The hashes in `E/manifest.json` record the version before the F rows were added to the synthesis; the previous and current synthesis versions are identified in the F manifest.
