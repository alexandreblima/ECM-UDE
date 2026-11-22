# Part A2 — Network zeroed after joint training

The ablation reevaluated the same saved output residual and ECM-UDE checkpoints for ten seeds, without retraining. The hashes of 40 checkpoints were verified. Before zeroing the correction, the evaluation exactly reproduced the saved MAEs, RMSEs, and maximum errors for 2.160 (two thousand one hundred sixty) checkpoint-condition-metric combinations, using the normalization statistics recorded in the original summary. Only the neural correction parameters and output gain were zeroed; the physical parameters remained those of the checkpoint.

The differences below are `error(with network) - error(zeroed network)`, in mV; positive values indicate greater error with the network. The medians are descriptive, aggregated over seeds and conditions for each cell.

| Cell | Variant | Median MAE | Median RMSE | Median maximum error |
|---|---|---:|---:|---:|
| NCA | Output residual | 0,0023 | 0,0017 | 0,0026 |
| NCA | ECM-UDE | 0,0725 | 0,0598 | 0,0744 |
| NMC | Output residual | -0,0008 | 0,0007 | 0,0025 |
| NMC | ECM-UDE | 0,0465 | 0,0186 | 0,0275 |

With a TOST margin of ±1 mV and Holm correction over the 18 combined main conditions, equivalence to the zeroed network was found for output residual in 18/18 conditions for MAE, RMSE, and maximum error; and for ECM-UDE in 18/18 for MAE and RMSE and 14/18 for maximum error. Equivalence of the zeroed ECM-UDE against physics-gradient was confirmed in 18/18 conditions for MAE and RMSE and 14/18 for maximum error; zeroed output residual was equivalent in 18/18 for all three metrics.

The complete tables with differences by seed and condition, raw p-values, TOST tails, Holm-adjusted p-values, and hashes are in:

- `a2_zero_network_by_seed.csv`
- `a2_network_effect_tost.csv`
- `a2_zeroed_vs_physics_gradient_tost.csv`
- `a2_checkpoint_hashes.json`
- `a2_manifest.json`

The one-seed, one-condition smoke test is also in `a2_smoke.csv`; it was not used for inference.
