# Reproducibility scope

Classification: (a) raw data plus released scripts; (b) saved parameters, checkpoints, histories, or result summaries; (c) both. Render-only commands read the saved artifacts named in each script; raw-data commands independently rerun the analysis where supported.

| Item | Scope | Script and command | Approximate time |
|---|---|---|---|
| Fig. 3 | b | `figures_ieee.py --figures fig01_dynamic_corrections` | Saved E1 metrics and TOST; seconds to render; full E1 rerun is multi-hour. |
| Fig. 4 | b | `figures_ieee.py --figures fig02_physics_vs_lstm` | Saved LSTM/ECM-2RC metrics; seconds to render; full training is multi-hour. |
| Fig. 5 | c | `figures_ieee.py --figures fig03_lfp_hysteresis; exp_lfp_gonogo.py` | Seconds to render; raw fit/simulation rerun is tens of minutes to hours. |
| Fig. 6 | c | `figures_ieee.py --figures fig04_static_oracle; exp_lfp_gonogo.py` | Seconds to render; target-oracle NLS rerun is tens of minutes. |
| Fig. 7 | c | `figures_ieee.py --figures fig05_same_temperature_transfer; exp_lfp_static.py; exp_static_replication.py` | Seconds to render; fits from raw profiles take tens of minutes to hours. |
| Fig. 8 | c | `figures_ieee.py --figures fig06_thermal_transfer; exp_lfp_static.py` | Seconds to render; raw thermal profiles and NLS fits take tens of minutes. |
| Fig. 9 | b | `figures_ieee.py --figures fig07_temperature_interpolation` | Seconds to render from saved parameter tables and posterior draws. |
| Fig. 10 | b | `figures_ieee.py --figures fig08_gp_coverage_diagnostic` | Seconds to render from saved GP band/sample outputs; refitting is not required. |
| Table 4 | b | `python reviewer_saved_analysis.py --output /path/to/empty/directory` | Seconds to regenerate the historical criteria table from saved outcomes. |
| Table 5 | c | `reviewer_saved_analysis.py; exp_unified.py --dataset all` | Seconds to regenerate from saved summaries; multi-hour full ten-seed E1 rerun. |
| Table 6 | c | `reviewer_saved_analysis.py; exp_param_ablation.py` | Seconds to regenerate from saved fit outputs; raw NLS refits take tens of minutes to hours. |

Render times are approximate process times, not model-fitting times. Full E1 training covers ten seeds, two cells, and all registered trainable models; saved-result rendering does not rerun training or overwrite legacy outputs.
