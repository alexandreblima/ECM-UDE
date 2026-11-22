# Unified ECM–UDE comparison

Run completed under the unified protocol recorded in [protocol.py](../../protocol.py).

The run completed 40 training runs (four models × five seeds × two datasets) and four NLS fits. The hashes of the 44 checkpoints/parameter files were verified. The full table contains 220 rows, with MAE, RMSE, maximum error, and P99 in mV: [CSV](table_main.csv), [LaTeX](table_main.tex).

The protocol uses causal SOC based on nominal capacity (2,9/2,0 Ah), OCV bounded to [0,1], classical RK4 with input held constant per step, and an 80/20 temporal split excluding windows that overlap the boundary. Normalization uses training data only. OOD targets are evaluated in full. In continuous mode, physical models start at the beginning of the cycle and are scored only on the source held-out mask. LSTM appears only in windowed mode.

In the tables, B is the physics model with internal SOC/learnable eta, no network, fitted by gradient descent; Residual corrects the output. B, Residual, and UDE share the 30-epoch budget and optimizer. LSTM retains a budget of up to 150 epochs. The seeds are 42–46. NLS fits are deterministic, with one fit per dataset.

**Statistics:** no comparison between the winner and runner-up is significant at 5%. With five nonzero pairs, the smallest possible exact two-sided Wilcoxon p-value is 0,0625, even before Holm correction. The units are seeds; time samples are not independent replicates. Comparisons between two deterministic NLS fits are not assigned a p-value. Tests of UDE versus each baseline for all metrics are in [statistics.json](statistics.json); winners by metric are in [winners.csv](winners.csv).

**Interpretation limits:** the source held-out data also select checkpoints by early stopping; this is validation, not a third independent test set. CALCE uses independent SP1 OCV at 25 °C (a low-current approximation from a different cell), reused at other temperatures. A rest of at least 30 s with |I|≤0,02 A is required. Panasonic uses the documented z0=1 fallback because no independent local calibration is available. CALCE US06/BJDST at 25 °C also uses z0=1 for lack of a qualifying rest; the other conditions use OCV inversion. FUDS25 starts at z0≈0,82291.

**Descriptive reading:** UDE shows no overall advantage specific to the dynamic correction. ECM-2RC has the lowest MAE in the eight windowed Panasonic conditions and in 15/20 conditions in continuous mode. Output residual leads in eight windowed CALCE conditions; UDE leads only on windowed DST25, by approximately 0,020 mV over B. The two LSTM wins occur on CALCE25 targets with SOC fallback, which should be considered when interpreting them.

## MAE in mV — mean ± sample standard deviation

NLS: deterministic value. Bold: lowest mean, without implying statistical significance. All source cells (UDDS25 Panasonic and FUDS25 CALCE) use held-out data only.

| Dataset/condition | Mode | 1RC | 2RC | B | Residual | UDE | LSTM | Lowest MAE | Second | Raw p | Holm p | Significant at 5% |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| panasonic/UDDS_25C | windowed | 43.051 | **42.849** | 42.947 ± 0.147 | 42.953 ± 0.137 | 43.090 ± 0.230 | 82.381 ± 4.633 | 2RC | B | 0.3125 | 1.0000 | No |
| panasonic/UDDS_25C | continuous | 43.148 | 42.661 | **42.611 ± 0.264** | 42.617 ± 0.270 | 43.164 ± 0.258 | — | B | Residual | 0.3125 | 1.0000 | No |
| panasonic/UDDS_10C | windowed | 37.537 | **37.312** | 37.515 ± 0.106 | 37.508 ± 0.112 | 37.726 ± 0.080 | 74.525 ± 5.650 | 2RC | Residual | 0.0625 | 0.8750 | No |
| panasonic/UDDS_10C | continuous | 36.755 | **36.263** | 36.571 ± 0.193 | 36.565 ± 0.199 | 36.945 ± 0.071 | — | 2RC | Residual | 0.0625 | 0.5625 | No |
| panasonic/UDDS_0C | windowed | 96.163 | **96.107** | 96.148 ± 0.136 | 96.135 ± 0.143 | 96.394 ± 0.166 | 155.418 ± 9.283 | 2RC | Residual | 0.8125 | 1.0000 | No |
| panasonic/UDDS_0C | continuous | 94.287 | **93.816** | 94.116 ± 0.222 | 94.103 ± 0.228 | 94.525 ± 0.164 | — | 2RC | Residual | 0.0625 | 0.5625 | No |
| panasonic/UDDS_-10C | windowed | 173.957 | **173.900** | 173.940 ± 0.158 | 173.924 ± 0.163 | 174.163 ± 0.261 | 250.523 ± 12.613 | 2RC | Residual | 0.6250 | 1.0000 | No |
| panasonic/UDDS_-10C | continuous | 171.787 | **171.303** | 171.627 ± 0.239 | 171.611 ± 0.240 | 171.993 ± 0.274 | — | 2RC | Residual | 0.0625 | 0.5625 | No |
| panasonic/UDDS_-20C | windowed | 295.088 | **295.021** | 295.050 ± 0.176 | 295.032 ± 0.174 | 295.238 ± 0.360 | 381.222 ± 14.923 | 2RC | Residual | 0.8125 | 1.0000 | No |
| panasonic/UDDS_-20C | continuous | 292.906 | **292.392** | 292.733 ± 0.253 | 292.715 ± 0.248 | 293.045 ± 0.391 | — | 2RC | Residual | 0.0625 | 0.5625 | No |
| panasonic/US06_25C | windowed | 38.774 | **36.944** | 38.727 ± 0.089 | 38.731 ± 0.092 | 38.852 ± 0.074 | 58.866 ± 2.256 | 2RC | B | 0.0625 | 0.8750 | No |
| panasonic/US06_25C | continuous | 34.217 | **31.120** | 34.010 ± 0.171 | 34.014 ± 0.175 | 34.273 ± 0.094 | — | 2RC | B | 0.0625 | 0.5625 | No |
| panasonic/LA92_25C | windowed | 17.467 | **16.594** | 17.446 ± 0.082 | 17.448 ± 0.084 | 17.556 ± 0.029 | 24.263 ± 0.860 | 2RC | B | 0.0625 | 0.8750 | No |
| panasonic/LA92_25C | continuous | 15.903 | **14.592** | 15.754 ± 0.128 | 15.756 ± 0.132 | 15.969 ± 0.049 | — | 2RC | B | 0.0625 | 0.5625 | No |
| panasonic/HWFET_25C | windowed | 39.637 | **39.340** | 39.583 ± 0.118 | 39.585 ± 0.123 | 39.763 ± 0.048 | 54.847 ± 1.765 | 2RC | B | 0.0625 | 0.8750 | No |
| panasonic/HWFET_25C | continuous | 36.080 | **35.005** | 35.805 ± 0.245 | 35.808 ± 0.251 | 36.192 ± 0.073 | — | 2RC | B | 0.0625 | 0.5625 | No |
| calce/FUDS_25C | windowed | 13.218 | **13.177** | 13.187 ± 0.047 | 13.189 ± 0.049 | 13.184 ± 0.043 | 71.336 ± 3.198 | 2RC | UDE | 1.0000 | 1.0000 | No |
| calce/FUDS_25C | continuous | 13.328 | **13.258** | 13.265 ± 0.072 | 13.267 ± 0.076 | 13.261 ± 0.067 | — | 2RC | UDE | 0.8125 | 1.0000 | No |
| calce/DST_25C | windowed | 5.802 | 5.789 | 5.771 ± 0.075 | 5.774 ± 0.077 | **5.751 ± 0.071** | 29.006 ± 16.890 | UDE | B | 0.8125 | 1.0000 | No |
| calce/DST_25C | continuous | 5.203 | **5.102** | 5.162 ± 0.073 | 5.164 ± 0.075 | 5.157 ± 0.054 | — | 2RC | UDE | 0.0625 | 0.5625 | No |
| calce/US06_25C | windowed | 123.701 | 123.575 | 123.707 ± 0.075 | 123.712 ± 0.073 | 123.634 ± 0.127 | **120.748 ± 11.155** | LSTM | 2RC | 1.0000 | 1.0000 | No |
| calce/US06_25C | continuous | 122.049 | **121.807** | 122.032 ± 0.065 | 122.037 ± 0.064 | 121.965 ± 0.140 | — | 2RC | UDE | 0.1250 | 0.8750 | No |
| calce/BJDST_25C | windowed | 123.481 | 123.366 | 123.487 ± 0.075 | 123.491 ± 0.073 | 123.414 ± 0.127 | **120.469 ± 12.528** | LSTM | 2RC | 0.8125 | 1.0000 | No |
| calce/BJDST_25C | continuous | 122.911 | **122.683** | 122.894 ± 0.064 | 122.899 ± 0.064 | 122.827 ± 0.140 | — | 2RC | UDE | 0.1875 | 1.0000 | No |
| calce/FUDS_0C | windowed | 53.873 | 53.871 | 53.880 ± 0.070 | **53.865 ± 0.063** | 54.059 ± 0.163 | 99.653 ± 4.946 | Residual | 2RC | 0.8125 | 1.0000 | No |
| calce/FUDS_0C | continuous | 53.326 | **53.226** | 53.316 ± 0.058 | 53.301 ± 0.053 | 53.514 ± 0.164 | — | 2RC | Residual | 0.0625 | 0.5625 | No |
| calce/DST_0C | windowed | 55.549 | 55.560 | 55.552 ± 0.076 | **55.536 ± 0.069** | 55.744 ± 0.171 | 102.399 ± 5.186 | Residual | 1RC | 0.8125 | 1.0000 | No |
| calce/DST_0C | continuous | 54.638 | **54.542** | 54.622 ± 0.065 | 54.606 ± 0.060 | 54.835 ± 0.173 | — | 2RC | Residual | 0.1250 | 0.5625 | No |
| calce/US06_0C | windowed | 64.370 | 64.379 | 64.367 ± 0.078 | **64.351 ± 0.071** | 64.569 ± 0.179 | 111.919 ± 6.137 | Residual | B | 0.0625 | 0.8750 | No |
| calce/US06_0C | continuous | 63.790 | **63.682** | 63.766 ± 0.068 | 63.750 ± 0.063 | 63.992 ± 0.182 | — | 2RC | Residual | 0.1250 | 0.6250 | No |
| calce/BJDST_0C | windowed | 75.043 | 75.075 | 75.033 ± 0.083 | **75.017 ± 0.076** | 75.237 ± 0.180 | 123.305 ± 8.149 | Residual | B | 0.0625 | 0.8750 | No |
| calce/BJDST_0C | continuous | 74.378 | **74.297** | 74.346 ± 0.073 | 74.330 ± 0.069 | 74.576 ± 0.184 | — | 2RC | Residual | 0.4375 | 1.0000 | No |
| calce/FUDS_45C | windowed | 17.515 | 17.550 | 17.504 ± 0.055 | **17.496 ± 0.055** | 17.653 ± 0.124 | 64.566 ± 8.947 | Residual | B | 0.1875 | 0.8750 | No |
| calce/FUDS_45C | continuous | 17.477 | 17.545 | 17.464 ± 0.054 | **17.456 ± 0.054** | 17.628 ± 0.135 | — | Residual | B | 0.1875 | 0.7500 | No |
| calce/DST_45C | windowed | 17.054 | 17.158 | 17.033 ± 0.053 | **17.023 ± 0.052** | 17.222 ± 0.151 | 63.937 ± 7.539 | Residual | B | 0.1250 | 0.8750 | No |
| calce/DST_45C | continuous | 17.511 | 17.680 | 17.500 ± 0.047 | **17.486 ± 0.046** | 17.753 ± 0.206 | — | Residual | B | 0.1250 | 0.5625 | No |
| calce/US06_45C | windowed | 15.904 | 15.999 | 15.893 ± 0.066 | **15.887 ± 0.065** | 16.011 ± 0.103 | 62.420 ± 7.637 | Residual | B | 0.1875 | 0.8750 | No |
| calce/US06_45C | continuous | 16.061 | 16.184 | 16.043 ± 0.066 | **16.035 ± 0.066** | 16.208 ± 0.137 | — | Residual | B | 0.1875 | 0.7500 | No |
| calce/BJDST_45C | windowed | 18.378 | 18.487 | 18.358 ± 0.075 | **18.353 ± 0.074** | 18.480 ± 0.107 | 63.787 ± 5.526 | Residual | B | 0.1875 | 0.8750 | No |
| calce/BJDST_45C | continuous | 18.368 | 18.516 | 18.341 ± 0.076 | **18.333 ± 0.075** | 18.531 ± 0.157 | — | Residual | B | 0.1875 | 0.7500 | No |

Checks: [validation.json](validation.json). Manifest, data, and checkpoints: [summary.json](summary.json). Full logs: `logs/`. The previous backup was checked against all 270 of its files, with no changes.
