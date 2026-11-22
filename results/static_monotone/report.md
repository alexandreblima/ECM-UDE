# V7m: robustness with monotone OCV

Protocol 8a0e2e2. Source only; 11 PCHIP knots with positive increments; constant hold outside source SOC.
Criterion: V7m>=25% reduction versus V1 on ALL targets at the same T, without selecting other models.

| Dataset | Profile | V1 | V7m | Reduction | Passes |
|---|---|---:|---:|---:|---|
| cell_A_a123 | identification failed | -- | -- | -- | INCONCLUSIVE |
| cell_B_mendeley | FUDS | 18.842 | 18.082 | 4.04% | False |
| cell_B_mendeley | US06 | 17.917 | 18.224 | -1.72% | False |
| panasonic | identification failed | -- | -- | -- | INCONCLUSIVE |
| calce | identification failed | -- | -- | -- | INCONCLUSIVE |

- cell_A_a123: robust=None; status=INCONCLUSIVE_NLS.
- cell_B_mendeley: robust=False; status=complete.
- panasonic: robust=None; status=INCONCLUSIVE_NLS.
- calce: robust=None; status=INCONCLUSIVE_NLS.
- cell_B_mendeley: parameters near bounds=0 [].

## Caveats

- Original verdicts were not rewritten; this is a new test of a single V7m variant.
- A123/Mendeley retain previous sanitization and initial-state assumptions.
- Panasonic V1 OCV checkpoint degree 5; dt1s differs from unified; intra-experiment comparison.
- CALCE V7m source z0 [.9,1] retained despite source ~80 SOC; main baseline V1 corrected by Ah counting.
- OCV support recalculated during fitting from source SOC Q/z0, never defined by the target.
- Monotone increments constrain shape, not proof of physical identifiability.
- Q/z0 are not identifiable in this protocol: recalculated source SOC support cancels both exactly.
  Normalized Ah coordinate eliminates a purely rounding Jacobian without changing model or hyperparameters.
  Analytic PCHIP/RC Jacobian validated by central differences; failed numerical attempts preserved.
- Targets and secondary temperatures evaluated only if the source fit converged; no new criterion.
  NLS failures in A123/Panasonic/CALCE are inconclusive, not measured gains or losses.
