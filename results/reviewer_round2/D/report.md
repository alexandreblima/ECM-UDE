# Part D — saved-result analyses

No model was fitted for this part. Legacy results were read only; all outputs are under `results/reviewer_round2/D/`.

## D1 — adjusted p-values

Computed 120 reference-condition rows (20 conditions, three trainable variants, two references). TOST uses exact one-sided Wilcoxon tests at ±1 mV; Holm families are 20 conditions against ECM-2RC and 18 principal conditions against physics-gradient. All principal E/N codes match.

- `tost_pvalues.csv`: raw lower/upper p, raw TOST max, Holm-adjusted p per direction, adjusted TOST max, median difference and Walsh HL pseudomedian.
- `tost_pvalues.tex`: compact booktabs appendix table.
- `e1_code_audit.json`: row-level check against the saved E1 table.

## D2 — reproducibility scope

See `reproducibility_scope.md` for scripts, commands, classification, and approximate runtime.

## D3 — learning curves

The vector figure overlays the three trainable variants by cell. Each line is the seed median; the envelope is the observed-seed minimum–maximum without forward-filling early stopping. Vertical marks denote the median selected epoch; the summary table also gives the selected-epoch range. Loss reductions are computed per seed, then summarized by median.

- `learning_curves.pdf` (7.16 in wide, grayscale, vector PDF).
- `learning_curve_data.csv` and `learning_curve_summary.csv`.
- `learning_curve_provenance.json`: exact history paths resolved by SHA256.

History checks: 60 histories matched their recorded SHA256; no history/checkpoint source was copied or altered.
