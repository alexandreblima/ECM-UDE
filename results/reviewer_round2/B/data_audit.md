# Part B — LG HG2 data audit

Training and evaluation signals were consistently read from MAT files: time, current, voltage, surface temperature, and Ah. MAT Wh was ignored; Wh was not needed. Positive current represents charge and negative current discharge.

Preregistered cleaning rejects nonfinite rows, stably sorts by time, retains the first sample at duplicate timestamps, and linearly resamples to a 0,1 s grid. The table shows raw-file time intervals before cleaning; no negative intervals were observed.

| Profile | Raw samples | Duration (s) | median dt (s) | dt=0 | dt<0 | Initial V (V) | Final V (V) | Final SOC by counting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UDDS_25C_source | 159646 | 15965.500 | 0.099999 | 0 | 0 | 4.18464 | 3.20104 | 0.0492 |
| Mixed1_25C | 77227 | 7722.400 | 0.099999 | 1 | 0 | 4.18464 | 3.20273 | 0.0491 |
| LA92_25C | 100764 | 10083.500 | 0.099999 | 1 | 0 | 4.19155 | 3.21385 | 0.0490 |
| US06_25C | 40155 | 4015.200 | 0.099999 | 1 | 0 | 4.18481 | 3.22515 | 0.0495 |
| UDDS_-20C | 81911 | 8190.800 | 0.100995 | 1 | 0 | 4.18312 | 3.59666 | 0.4181 |
| UDDS_-10C | 123602 | 12359.900 | 0.099999 | 1 | 0 | 4.18514 | 3.47732 | 0.2092 |
| UDDS_0C | 140435 | 14043.200 | 0.100995 | 2 | 0 | 4.18632 | 3.34870 | 0.1377 |
| UDDS_10C | 148898 | 14891.200 | 0.099004 | 1 | 0 | 4.18616 | 3.28566 | 0.1127 |
| UDDS_40C | 158956 | 15896.900 | 0.099999 | 1 | 0 | 4.18177 | 3.20323 | 0.0717 |

## Full charge and z0

The measured `551_Cap_1C` capacity at 25 degC was used: 2.72639 Ah. For each profile, the preceding charge mapped in the protocol reached at least 4,19 V, had a final charging current no greater than 0,05 A, and was followed by at least 1800 s with |I|<=0,02 A. The start of each profile differed by at most 0,05 V from the voltage at the end of rest. Thus, z0=1 is anchored before each profile; coulomb counting uses only samples preceding the evaluated point.

| Profile | Preceding charge | Rest (s) | Rest V (V) | Initial V (V) | Difference (mV) |
|---|---|---:|---:|---:|---:|
| UDDS_25C_source | 551_Charge2 | 3601.0 | 4.18579 | 4.18464 | -1.15 |
| Mixed1_25C | 551_Charge6 | 3600.0 | 4.18579 | 4.18464 | -1.15 |
| LA92_25C | 551_Charge4 | 3600.0 | 4.19271 | 4.19155 | -1.16 |
| US06_25C | 551_Charge5 | 3600.0 | 4.18579 | 4.18481 | -0.98 |
| UDDS_-20C | 610_Charge8 | 3600.0 | 4.18630 | 4.18312 | -3.18 |
| UDDS_-10C | 596_Charge2 | 3600.0 | 4.18748 | 4.18514 | -2.34 |
| UDDS_0C | 589_Charge2 | 3600.0 | 4.18799 | 4.18632 | -1.67 |
| UDDS_10C | 576_Charge2 | 3601.0 | 4.18748 | 4.18616 | -1.32 |
| UDDS_40C | 556_Charge2 | 3601.0 | 4.18259 | 4.18177 | -0.82 |

The OCV curve uses the C/20 discharge branch of `549_C20DisCh` at 25 degC (negative current): causal current integration, 0,002 SOC bins with median voltage and cumulative monotone envelope, as in CALCE calibration. The charge branch was not used. The resulting curve is in `c20_discharge_ocv.csv`; Cap_1C capacity and the C/20 curve were read only from MAT files.

## HWFET 25 degC substitution

Distributed HWFET has 6002 MAT samples, duration 600.007 s and maximum/absolute current 0.00000/0.00000 A. The CSV also has 6002 numeric samples and maximum/absolute current 0.00000/0.00000 A. Both are rest; therefore HWFET was replaced by Mixed1 25 degC, preceded by verified `551_Charge6`.

The four 10 degC CSVs labeled MG1 were excluded. Mixed3, absent at 0 and 10 degC, was not used. Wh was read only from CSV when present; it was not used in modeling.

The C/20 file has 1097 samples with I<-0,05 A and 1203 with I>0,05 A; calibration uses only the first discharge branch.

SHA256 hashes of all MAT/CSV files actually audited are in `manifest.json`.
