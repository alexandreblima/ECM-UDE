# Paper 1 Pipeline: ECM, LSTM, and ECM-UDE

## Final IEEE evidence figures

[figures_ieee/figures_index.md](./figures_ieee/figures_index.md) maps eight
vector PDFs to the five claims and their original saved data, with proposed
captions and scientific caveats. Run the absolute path to
[figures_ieee.py](./figures_ieee.py) in the existing scientific environment
with Matplotlib and pypdf; it does not fit, evaluate or retrain models.
The original five-seed ledger evidence is not replaced by ten-seed results.

Widths are exactly 3.5 or 7.16 inches, with embedded Times New Roman at
least 8 pt, no internal titles, and markers/line styles/hatches that remain
distinguishable in grayscale. HWFET is the canonical drive-cycle label.
[test_figures_ieee.py](./test_figures_ieee.py) checks PDF dimensions, embedded
fonts, absence of raster images, canonical labels and source hashes.

## Paper-v2 robustness study

[claims_ledger.md](./claims_ledger.md) records five evidence-qualified findings
without new experiments. Three separate protocols were committed before their
experiments: [seeds](./paper_v2_seeds_protocol.json),
[CALCE SOC audit](./paper_v2_calce_z0_protocol.json), and
[monotonic static correction](./paper_v2_monotone_protocol.json).

With the existing unified/LFP environment, run the absolute script paths in
this order, using `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

1. [exp_paper_v2_seeds.py](./exp_paper_v2_seeds.py) reuses seeds 42--46 and
   deterministic checkpoints, trains only 47--51, and writes
   `results/unified_10seeds/`. `--report-only` requires all ten seeds and
   regenerates only the new tables/statistics/report.
2. [exp_paper_v2_calce_z0.py](./exp_paper_v2_calce_z0.py) audits same-file
   full-charge counters for US06/BJDST 25 C and evaluates existing weights
   without retraining, writing `results/calce_z0_fix/`.
3. [exp_paper_v2_monotone.py](./exp_paper_v2_monotone.py) fits source-only
   V7m with 11 positive-increment PCHIP knots and constant source-support
   extrapolation, writing `results/static_monotone/`.

[test_paper_v2.py](./test_paper_v2.py) checks exact TOST/Holm thresholds,
causal initialization, the scalar continuous simulator, monotonic boundaries,
and saved results. [finalize_paper_v2.py](./finalize_paper_v2.py) requires the
pre-task SHA256 snapshot stored in the session folder; it generates
[report_paper_v2.md](./report_paper_v2.md) and the preservation audit under
`results/paper_v2_audit/`. Previous outputs and `MANUSCRITO/` are not rewritten.
CALCE corrected-baseline comparisons are labeled separately from historical
fallback results. Continuous TOST is primary; windowed TOST and LSTM effect
sizes are secondary. Failed equivalence is not evidence of improvement.
The user-authorized [failure policy](./paper_v2_monotone_failure_policy.json)
preserves the registered 500-evaluation budget: nonconverged V7m source fits
are explicitly inconclusive, not scored on targets. Recalculating candidate
source SOC support makes Q/z0 structurally unidentifiable in this V7m design;
the report must not interpret their fitted values as capacity/SOC evidence.

## LFP hysteresis go/no-go (separate NLS-only experiment)

[report_gonogo.md](./report_gonogo.md) records the completed A123/Mendeley
two-cell experiment. It does not change the unified benchmark below.
[lfp_gonogo_protocol.json](./lfp_gonogo_protocol.json) freezes the decision,
initial-state assumptions, bounds and optimizer starts before dynamic fitting.
No neural networks are imported by this experiment; ESC-UDE is not authorized.

The scripts resolve input/output paths relative to their own absolute location.
With `requirements-lfp.txt` installed, run the absolute paths to
`data_lfp.py`, then `lfp_ocv.py`, then `exp_lfp_gonogo.py`, using
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for the NLS run. Logs and
per-cell manifests/tables/figures are saved in `results/lfp_gonogo/`.
`exp_lfp_gonogo.py --report-only` renders the comparison from existing results;
`test_lfp_gonogo.py` validates mathematical invariants and saved trajectories.

Only A123 ZIPs are extracted, preserving their original bytes and rejecting
conflicting existing extracted files. Mendeley uses only the standalone copy.
Its missing preparation, 1.1-versus-2.50 Ah discrepancy, 25/30 C ambiguity,
and processed waveform periods remain explicit limitations, not silent fixes.
The report lists bound-hitting and identifiability diagnostics; the global
decision is determined only by A123, never by the replica.

## LFP static source-transfer experiment (separate preregistration)

[lfp_static_protocol.json](./lfp_static_protocol.json) freezes the new
source-only test. [exp_lfp_static.py](./exp_lfp_static.py) fits V1/V5/V6/V7
only on DST, caps both RC time constants at 3600 seconds, and transfers
fitted Q/z0 unchanged to FUDS/US06. It reads the previous V4 oracle without
refitting it or modifying any previous go/no-go result.

Run the absolute path to `exp_lfp_static.py` with the same NLS Python
environment and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; outputs and logs
go to `results/lfp_static/`. [test_lfp_static.py](./test_lfp_static.py)
verifies the exact 25% criterion, saved metrics, scalar RC recurrence, SOC
transfer, and SHA256 preservation of all prior artifacts.
[report_static.md](./report_static.md) documents ceiling-recovery fractions,
OCV-versus-SOC attribution, bound hits, and limitations. This is not a
reinterpretation of the earlier hysteresis criterion.

## Unified benchmark (current protocol)

`protocol.py` is the authoritative configuration. Run
`python exp_unified.py --dataset all` (or `panasonic` / `calce`). Historical
experiment entry points delegate to this resumable suite; their old helper
functions remain only to inspect archived results. Do not combine outputs in
`results/p1_*` with the unified benchmark. The manuscript is not regenerated.

The suite reidentifies 1RC/2RC by NLS and retrains physics-only gradient,
output-residual, dynamic UDE, and LSTM models, with seeds 42--46. Physics-only,
output-residual and UDE share Adam, 30-epoch budget and early stopping; LSTM
retains its original 150-epoch/1e-3 budget. All physical models use classical
RK4 with zero-order-held inputs and internal Coulomb-counted SOC. There is
no reuse of historical trained checkpoints. `unified_dynamics.py` compiles
the existing 2x32 GELU architecture and the exact discrete RK4 derivative;
`test_unified_protocol.py` compares its outputs and gradients to PyTorch.

Ah is standardized to **net discharged** Ah before applying
`z = z0 - (Ah - Ah[0])/Qnom`. Panasonic charge-positive Ah is negated;
CALCE uses discharge capacity minus charge capacity, including regeneration.
An independent CALCE SP1 25C low-current OCV curve estimates initial SOC
from the last pre-cycle rest (at least 30 s, |I| <= 0.02 A), and is reused
unchanged across temperatures. Its different-cell/low-current limitations
are recorded. If no usable independent curve/rest exists, z0=1 is recorded
explicitly. No end-of-cycle capacity is used. Raw files are read directly;
old CALCE caches are not reused or overwritten.

Both evaluation modes use the same sample mask. On the source cycle only
the final held-out 20% of windows is scored, with overlapping training
windows removed. The continuous physical evaluation starts at the beginning
of the cycle, carries its states through the training portion using only
inputs, and scores only held-out samples. Windowed evaluation resets states
per window and averages overlaps. An extra final window covers the tail;
OOD cycles are scored in full. LSTM is reported only in windowed mode.
The source held-out set also selects checkpoints: it is a validation score,
not a separate untouched final test set.

Outputs in `results/unified/`:

- `summary.json`: protocol, source/data/checkpoint SHA256, Git commit,
  environment, seeds, initial SOC provenance, masks and per-seed metrics.
- `table_main.csv` and `.tex`: MAE/RMSE/maximum/P99 in mV, mean and sample SD.
- `statistics.json`: paired exact Wilcoxon and Holm adjustment across all
  eligible model pairs within each condition/mode/metric; UDE comparisons and
  winners. Two deterministic NLS fits have no inferential seed comparison.
- `winners.csv`: winner and runner-up by metric, including significance.
- `checkpoints/`, histories and `logs/`: one log per fit/training experiment.

NLS is deterministic and identified once per source, not five independent
fits. Its value is repeated only for pairing with the five trainable seeds.
With five nonzero paired differences, the smallest exact two-sided Wilcoxon
p-value is 0.0625; this design cannot establish p<0.05 significance. Time
samples are not treated as independent replicates.

Install `requirements-unified.txt`. The local execution uses the existing
`/opt/anaconda3/envs/torch_mps/bin/python`; Numba/llvmlite may also be loaded
from ignored `.unified_deps/`. Resume requires identical source hashes and
protocol; old results are never silently mixed after a code change.

Use `python exp_unified.py --report-only` (also `python make_results_table.py`)
to regenerate tables and statistics without retraining or reading raw datasets.
The training manifest retains its original Git/source hashes; a separate
`report_provenance` records the version that rendered the tables. Parallel
preparations, when used, are linked through `parallel_preparations.json`.

The earlier README below documents the archived protocol only.

This repository implements the experimental pipeline for Paper 1 on battery voltage estimation with three approaches:

- `ecm_1rc`: physics-only baseline based on a 1RC equivalent circuit.
- `lstm`: purely data-driven baseline.
- `ecm_ude`: hybrid model combining an ECM with a Universal Differential Equation.

The pipeline covers:

- in-distribution comparison on `UDDS @ 25C`;
- inference-time sensitivity to SoC noise;
- zero-shot generalization to unseen temperatures;
- zero-shot generalization to unseen drive cycles;
- generation of the paper figures.

## Pipeline overview

1. `exp1_ude.py`
Trains and compares `ECM-1RC`, `LSTM`, and `ECM-UDE` on `25degC_UDDS_Pan18650PF.mat`.

2. `exp1_soc_noise_sensitivity.py`
Uses the best checkpoints from Experiment 1 to measure robustness to noise in the `SoC` input channel. In the current codebase, it also depends on `results/p1_exp1/ecm_params_cache.json`.

3. `exp2_ude_temperature.py`
Evaluates zero-shot transfer on `UDDS` at `10C`, `0C`, `-10C`, and `-20C`.

4. `exp3_ude_cycle.py`
Evaluates zero-shot transfer to unseen `25C` drive cycles: `US06`, `LA92`, and `HWFT`.

5. `plots_paper1.py`
Generates the final paper figures from saved results in `results/`.

## Repository structure

- `data.py`: `.mat` loading, `SoC` derivation, normalization, and windowing.
- `training.py`: training loop, early stopping, scheduler, and batched inference.
- `ecm.py`: identification and inference for the `ECM-1RC` baseline.
- `ude.py`: `ECM-UDE` implementation based on `torchdiffeq`.
- `models.py`: neural model definitions; the main pipeline here uses `LSTM`.
- `utils.py`: device selection and reproducibility seeding.
- `exp1_ude.py`: main Paper 1 experiment.
- `exp1_soc_noise_sensitivity.py`: SoC-noise robustness experiment.
- `exp2_ude_temperature.py`: temperature OOD experiment.
- `exp3_ude_cycle.py`: drive-cycle OOD experiment.
- `plots_paper1.py`: paper figure generation.
- `datasets/`: input `.mat` files.
- `results/`: outputs, checkpoints, and generated figures.
- `anteriores/`: older scripts outside the current main pipeline.

## Expected data

The code expects Kollmeyer-format MATLAB files inside `datasets/`. Each `.mat` file must contain a `meas` struct with:

- `Time`
- `Current`
- `Voltage`
- `Battery_Temp_degC`
- `Ah`

`SoC` is derived internally from `Ah`.

Files used by the main pipeline:

- `datasets/25degC_UDDS_Pan18650PF.mat`
- `datasets/10degC_UDDS_Pan18650PF.mat`
- `datasets/0degC_UDDS_Pan18650PF.mat`
- `datasets/n10degC_UDDS_Pan18650PF.mat`
- `datasets/n20degC_UDDS_Pan18650PF.mat`
- `datasets/25degC_US06_Pan18650PF.mat`
- `datasets/25degC_LA92_Pan18650PF.mat`
- `datasets/25degC_HWFTa_Pan18650PF.mat`

## Environment

Run everything from the repository root.

The scripts do not expose command-line arguments. Paths, hyperparameters, and output directories are defined as constants near the top of each script.

There is currently no `requirements.txt`, so dependencies need to be installed manually. A minimal setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib torch torchdiffeq
```

Optional dependency:

```bash
pip install pytorch-wavelets
```

`pytorch-wavelets` is only needed if you want to use the `WNO` / wavelet-based models in `models.py` or older scripts. The main pipeline in this directory uses `LSTM`, `ECM-1RC`, and `ECM-UDE`.

## Protocol used in this pipeline

- Input features per timestep: `I`, `T_cell`, and `SoC`.
- Target: terminal voltage trajectory `V(t)`.
- Windows: `length=1024`, `stride=512`.
- Split: temporal `80/20` with a guard band to avoid train/validation leakage.
- `I` and `V` use empirical statistics from the training split.
- `T` and `SoC` use fixed physical normalization constants.
- Experiment 1 runs `30` seeds and saves results incrementally, so it can be resumed safely.

## How to run

### 1. Main experiment

```bash
python exp1_ude.py
```

Expected outputs in `results/p1_exp1/`:

- `summary.json`
- `history_{model}_seed{seed}.json`
- `{model}_seed{seed}_best.pt`
- `{model}_best.pt`

Notes:

- the script is resumable; if interrupted, run it again;
- `ECM-1RC` is identified once on the training split and reused across seeds;
- `LSTM` and `ECM-UDE` are trained seed by seed.

### 2. SoC noise sensitivity

```bash
python exp1_soc_noise_sensitivity.py
```

Dependencies:

- `results/p1_exp1/lstm_best.pt`
- `results/p1_exp1/ecm_ude_best.pt`
- `results/p1_exp1/ecm_params_cache.json`

Important:

- this cache is not generated by `exp1_ude.py`;
- run `exp2_ude_temperature.py` or `exp3_ude_cycle.py` before this step.

Output:

- `results/p1_exp1_soc_noise/summary.json`

### 3. Temperature OOD

```bash
python exp2_ude_temperature.py
```

Dependencies:

- Experiment 1 artifacts in `results/p1_exp1/`

Outputs:

- `results/p1_exp2/summary.json`
- `results/p1_exp1/ecm_params_cache.json`

### 4. Drive-cycle OOD

```bash
python exp3_ude_cycle.py
```

Dependencies:

- Experiment 1 artifacts in `results/p1_exp1/`

Outputs:

- `results/p1_exp3/summary.json`
- may reuse or create `results/p1_exp1/ecm_params_cache.json`

### 5. Paper figures

```bash
python plots_paper1.py
```

To generate the full set of figures, run first:

1. `python exp1_ude.py`
2. `python exp2_ude_temperature.py`
3. `python exp3_ude_cycle.py`
4. `python exp1_soc_noise_sensitivity.py`

Outputs in `results/figures/`:

- `fig2_voltage_trace.*`
- `fig3_boxplot_seeds.*`
- `fig4_temp_ood.*`
- `fig5_cycle_ood.*`
- `fig6_us06_trace.*`
- `fig_ocv.*`
- `fig_learning_curves.*`
- `soc_noise_sensitivity.*`

Each figure is saved as `PDF`, `PNG`, and `SVG`.

## Recommended execution order

If the goal is to reproduce the full Paper 1 pipeline:

```bash
python exp1_ude.py
python exp2_ude_temperature.py
python exp3_ude_cycle.py
python exp1_soc_noise_sensitivity.py
python plots_paper1.py
```

This order ensures that `results/p1_exp1/ecm_params_cache.json` exists before figure generation.

## Outputs and versioning

- `results/` is ignored in `.gitignore`.
- In a clean clone, it is normal for results not to exist yet.
- Checkpoints and `summary.json` files are generated on demand by the scripts.

## Quick customization

- To change datasets, edit `MAT_PATH`, `TRAIN_MAT`, or `TEST_CYCLES` in the scripts.
- To change temporal granularity, edit `WINDOW_CFG`.
- To change training hyperparameters, edit `TRAIN_CFG_*` in `exp1_ude.py`.
- To change output locations, edit `OUTPUT_DIR` in the relevant script.

## Important notes

- Imports first try `wno_battery.*` and then fall back to the local modules. In practice, running from the repository root is enough.
- `plots_paper1.py` requires `results/p1_exp1/ecm_params_cache.json`; that file is produced by the OOD scripts.
- `models.py` contains additional classes such as `FNO`, `WNO`, and hybrid variants, but they are not part of the main workflow documented here.

### Static replication and A123 temperature tests

The preregistered [static replication protocol](static_replication_protocol.json)
is separate from all previous experiments. Run
`python -B exp_static_replication.py` and `python -B test_static_replication.py`
from this repository with the scientific dependencies in `requirements-lfp.txt`.
New outputs go to `results/static_replication/` and
`results/lfp_static_temperature/`; prior unified and LFP outputs are read-only
and checked by SHA-256.

Panasonic uses UDDS25 as source and CALCE uses FUDS25. V1/V5/V6/V7 are fitted
only on the full source; the primary criterion requires at least 25% MAE
reduction against V1 on every same-temperature target. Cross-temperature
replication is secondary. Panasonic V1 fixes the old unified 2RC checkpoint
OCV (not an independent calibration); CALCE V1 uses SP1 low-current OCV.
CALCE V6/V7 retain the requested z0 bounds despite the 80SOC preparation.
No neural model is trained; coincident unified MAEs are descriptive comparisons
with explicit protocol differences. The A123 thermal test stops when the eight
required temperature archives are missing. See
[the replication report](report_static_replication.md).

The completed A123 thermal test is run separately with
`python -B exp_a123_temperature.py` and `python -B test_a123_temperature.py`.
It reuses frozen DST25 fits under the original criterion. The additionally
registered [local V6 diagnostic](a123_temperature_diagnostic.json) fits only
each temperature's DST, with OCV25 fixed, and never enters the thermal decision.
Completed artifacts are in `results/lfp_static_temperature/cell_A_a123/`;
earlier blocked manifests and all previous results remain unchanged. See
[the thermal report](report_static_temperature.md).

The preregistered A123 parameter-temperature interpolation experiment uses
only saved DST V6 fits at 0/25/50 C to predict parameters for FUDS/US06 at
10/40 C. Run `python -B exp_a123_temperature_interp.py` and
`python -B test_a123_temperature_interp.py`. See
[the frozen protocol](a123_temperature_interp_protocol.json) and
[the interpolation report](report_temperature_interp.md).
P1 is nearest-temperature transfer, P2 transformed linear interpolation,
P3 Arrhenius/linear regression, and P4 independent parameter GPs with P3 mean.
P4 propagates 200 bounded posterior parameter samples through the unchanged
continuous simulator; voltage coverage is a separate diagnostic. All earlier
results remain read-only; outputs go to `results/a123_temperature_interp/`.
