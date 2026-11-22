# E1 error-structure reproduction

Read-only inference from frozen checkpoints; no training, fitting or reidentification.
Run from any directory with the environment in requirements-unified.txt:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze_e1_error_structure.py
python -m unittest -v test_e1_error_structure
```

Existing output directories are refused; use --output NEW_DIRECTORY for another run.
analysis.json contains individual metrics, SOC counts, empty bands, seed medians, peaks,
OCV derivatives and all claim checks. validation.json records all 576 continuous MAE checks
(18 conditions x 2 deterministic baselines + 18 x 3 trained variants x 10 seeds).
The 1RC baseline is validation-only; LSTM has no continuous prediction in E1.
preservation.json records SHA-256 before/after for every file in both checkpoint trees,
raw inputs, the saved summary and existing inference code. The external manuscript
support scripts are not runtime dependencies of this implementation or its tests.

All statistics are computed only after the global MAE gate passes (absolute tolerance 1e-6 mV).
Source masks cover retained validation windows; target masks cover the full cycle.
Peaks use the first maximum on that mask, index/N as in the reference, with elapsed-time
fraction also recorded. The last-5% exclusion is index < floor(0.95*N), intersected with the mask.
SOC bands match the reference: high [0.8,1], mid [0.2,0.8), low [0,0.2).
Errors use float64 subtraction, as protocol.metrics_mV; no rounding precedes aggregation.
Tail and band comparisons are differences of ten-seed medians, not medians of differences.
OCV uses 10001 equally spaced points on [0.2,0.8], both endpoints, and the Chebyshev chain rule.
1 mV / slope is reported in SOC percentage points, not relative percent SOC.

## Manuscript checks

Upper bounds are checked at the article's precision. A smaller rounded maximum is
listed as a conservative bound, not silently equated to the article number.

| ID | Quantity | Article | Reproduced | At article precision | Status |
|---|---|---:|---:|---:|---|
| peak_min | 2RC maximum error (mV) minimum | 436 | 435.887889826 | 436.0 | matches |
| peak_max | 2RC maximum error (mV) maximum | 1032 | 1032.2939955 | 1032.0 | matches |
| late_peaks | Peaks after 89% of the profile (conditions) | 17 | 17 | 17 | matches |
| late_voltage_min | Measured voltage at late peaks (V) minimum | 2.41 | 2.4104514122 | 2.41 | matches |
| late_voltage_max | Measured voltage at late peaks (V) maximum | 2.78 | 2.77774357796 | 2.78 | matches |
| early_peak_percent | NCA UDDS -20 C peak position (%) | 2 | 2.22097964661 | 2.0 | matches |
| trimmed_calce_min | calce 2RC maximum without last 5% (mV) minimum | 39 | 39.0583885427 | 39.0 | matches |
| trimmed_calce_max | calce 2RC maximum without last 5% (mV) maximum | 66 | 65.9962442362 | 66.0 | matches |
| trimmed_panasonic_min | panasonic 2RC maximum without last 5% (mV) minimum | 231 | 230.707896897 | 231.0 | matches |
| trimmed_panasonic_max | panasonic 2RC maximum without last 5% (mV) maximum | 436 | 435.887889826 | 436.0 | matches |
| rmse_neural | rmse: max neural versus physics-gradient median difference (mV) | 0.42 | 0.423305504052 | 0.42 | matches |
| rmse_nca | rmse: largest trained-minus-2RC median on NCA (mV) | 3.9 | 3.87876852758 | 3.9 | matches |
| rmse_nmc | rmse: max absolute trained-minus-2RC median on NMC (mV) | 0.3 | 0.279705106878 | 0.3 | matches |
| p99_neural | p99: max neural versus physics-gradient median difference (mV) | 0.73 | 0.732326507568 | 0.73 | matches |
| p99_nca | p99: largest trained-minus-2RC median on NCA (mV) | 12.6 | 12.5992620131 | 12.6 | matches |
| p99_nmc | p99: max absolute trained-minus-2RC median on NMC (mV) | 2.2 | 2.20271789039 | 2.2 | matches |
| max_error_neural | max_error: max neural versus physics-gradient median difference (mV) | 0.89 | 0.892281532288 | 0.89 | matches |
| max_error_nca | max_error: largest trained-minus-2RC median on NCA (mV) | 42.6 | 42.5709302893 | 42.6 | matches |
| max_error_nmc | max_error: max absolute trained-minus-2RC median on NMC (mV) | 2.2 | 2.13130979666 | 2.1 | conservative bound |
| calce_DST_25C_mid | calce DST_25C 2RC mid-SOC MAE (mV) | 2.3 | 2.29906628964 | 2.3 | matches |
| calce_DST_25C_low | calce DST_25C 2RC low-SOC MAE (mV) | 15.1 | 15.0957229852 | 15.1 | matches |
| panasonic_HWFET_25C_mid | panasonic HWFET_25C 2RC mid-SOC MAE (mV) | 20.5 | 20.537222567 | 20.5 | matches |
| panasonic_HWFET_25C_low | panasonic HWFET_25C 2RC low-SOC MAE (mV) | 133.5 | 133.461363745 | 133.5 | matches |
| soc_neural | All SOC bands: max neural-versus-physics-gradient median difference (mV) | 0.62 | 0.618019603715 | 0.62 | matches |
| soc_nca | All SOC bands: largest trained-minus-2RC median on NCA (mV) | 4.3 | 4.28651945954 | 4.3 | matches |
| soc_nmc | All SOC bands: max absolute trained-minus-2RC median on NMC (mV) | 0.5 | 0.540338340955 | 0.5 | matches |
| ocv_panasonic_median | panasonic 1 mV in SOC percentage points (median) | 0.13 | 0.125263982025 | 0.13 | matches |
| ocv_panasonic_maximum | panasonic 1 mV in SOC percentage points (maximum) | 0.18 | 0.182199739769 | 0.18 | matches |
| ocv_calce_median | calce 1 mV in SOC percentage points (median) | 0.19 | 0.18971746472 | 0.19 | matches |
| ocv_calce_maximum | calce 1 mV in SOC percentage points (maximum) | 0.35 | 0.351437246522 | 0.35 | matches |
| ocv_all_min | 1 mV in SOC percentage points (both cells) minimum | 0.1 | 0.09215233959 | 0.1 | matches |
| ocv_all_max | 1 mV in SOC percentage points (both cells) maximum | 0.4 | 0.351437246522 | 0.4 | matches |
