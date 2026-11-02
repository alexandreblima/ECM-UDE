# Figuras finais IEEE

PDFs vetoriais; largura exata 3,5 in ou 7,16 in; Times New Roman incorporada, minimo 8 pt.
Sem titulos internos. Cores acompanhadas de marcadores, tipos de linha ou hachuras distintos;
as series continuam identificaveis sem cor. Rotulo canonico: HWFET.

Dados exclusivamente de resultados salvos citados pelo registro de afirmacoes; nenhum retreino,
reajuste ou nova simulacao. Os numeros de cinco seeds nao foram substituidos pelos de dez seeds.

| Figura | Achado | Largura (in) | Arquivos de dados de origem |
|---|---|---:|---|
| [fig01_dynamic_corrections.pdf](fig01_dynamic_corrections.pdf) | 1 | 7.16 | [results/unified/summary.json](../results/unified/summary.json) |
| [fig02_physics_vs_lstm.pdf](fig02_physics_vs_lstm.pdf) | 2 | 7.16 | [results/unified/summary.json](../results/unified/summary.json) |
| [fig03_lfp_hysteresis.pdf](fig03_lfp_hysteresis.pdf) | 3 | 7.16 | [results/lfp_gonogo/cell_A_a123/summary.json](../results/lfp_gonogo/cell_A_a123/summary.json), [results/lfp_gonogo/cell_B_mendeley/summary.json](../results/lfp_gonogo/cell_B_mendeley/summary.json) |
| [fig04_static_oracle.pdf](fig04_static_oracle.pdf) | 4 | 3.50 | [results/lfp_gonogo/cell_A_a123/summary.json](../results/lfp_gonogo/cell_A_a123/summary.json), [results/lfp_gonogo/cell_B_mendeley/summary.json](../results/lfp_gonogo/cell_B_mendeley/summary.json) |
| [fig05_same_temperature_transfer.pdf](fig05_same_temperature_transfer.pdf) | 5 | 7.16 | [results/lfp_static/cell_A_a123/summary.json](../results/lfp_static/cell_A_a123/summary.json), [results/lfp_static/cell_B_mendeley/summary.json](../results/lfp_static/cell_B_mendeley/summary.json), [results/static_replication/panasonic/summary.json](../results/static_replication/panasonic/summary.json), [results/static_replication/calce/summary.json](../results/static_replication/calce/summary.json) |
| [fig06_thermal_transfer.pdf](fig06_thermal_transfer.pdf) | 5 | 7.16 | [results/lfp_static_temperature/cell_A_a123/summary.json](../results/lfp_static_temperature/cell_A_a123/summary.json) |
| [fig07_temperature_interpolation.pdf](fig07_temperature_interpolation.pdf) | 5 | 3.50 | [results/a123_temperature_interp/summary.json](../results/a123_temperature_interp/summary.json) |
| [fig08_gp_coverage_diagnostic.pdf](fig08_gp_coverage_diagnostic.pdf) | 5 | 3.50 | [results/a123_temperature_interp/summary.json](../results/a123_temperature_interp/summary.json) |

## Legendas propostas e ressalvas

### fig01_dynamic_corrections.pdf

Dynamic and output corrections relative to ECM-2RC: Panasonic (left), CALCE (right).

Continuous MAE, original seeds 42--46; markers show mean and bars sample SD across seeds. Negative differences indicate nominal improvement, not statistical significance. CALCE US06/BJDST 25 C fallback conditions excluded; source scores are validation scores. Physics gradient has no neural network; trainable physical models initialize from 1RC.

### fig02_physics_vs_lstm.pdf

Windowed prediction errors of ECM-2RC and LSTM: Panasonic (left), CALCE (right).

Deterministic ECM-2RC; LSTM mean and sample SD for seeds 42--46. CALCE US06/BJDST 25 C fallback excluded. No comparison to continuous LSTM is implied. Same cells and finite training budgets do not establish universal superiority.

### fig03_lfp_hysteresis.pdf

ESC hysteresis gain and saturation diagnostic in both LFP cells.

Gain is relative to the best of V1/V2; GO needs gain >=10% and saturation <90% in both target profiles. A123 NO-GO, Mendeley GO; primary/global NO-GO. Mendeley nominal temperature/preparation and long RC time constants remain caveats; gain is not proof of identified minor loops.

### fig04_static_oracle.pdf

Voltage-error reduction when OCV and RC parameters are reidentified on each target cycle.

V4 is a target-fitted degree-9 OCV plus RC oracle, not transferable prediction. This comparison does not isolate OCV from RC or SOC error. These are the original go/no-go V1/V4 fits, with their original RC bounds.

### fig05_same_temperature_transfer.pdf

Source-only static recalibration transfers to held-out cycles at nominally matched temperature.

Best of V5/V6/V7 may differ by target; threshold is >=25%. CALCE DST is an eligible individual target, NOT evidence that the dataset passes all targets. Old US06/BJDST 25 C fallback results and nonphysical V5 extrapolations are excluded. The later corrected CALCE dataset verdict is negative. Panasonic V1 uses dynamic checkpoint OCV, not independent low-current OCV. Mendeley temperature is nominal.

### fig06_thermal_transfer.pdf

A123 thermal transfer and local recalibration: FUDS (left), US06 (right).

Transferred fits are frozen from DST 25 C; no target refit. Best transferred V5/V6/V7 fails the >=25% criterion in both profiles at every temperature. Local V6 is fitted on the same-temperature DST and is only an out-of-criterion diagnostic, not part of the transfer decision. All fits retain fixed OCV25/Q25 references.

### fig07_temperature_interpolation.pdf

Held-out temperature recovery by linear interpolation (P2), parametric fit (P3) and GP (P4).

Calibration only DST 0/25/50 C; held-out FUDS/US06 at 10/40 C. Recovery=(MAE_floor-MAE_method)/(MAE_floor-MAE_local_oracle); ratios are not clipped. P2 exceeds 50% in all four conditions. Recovery >100% is valid because the local NLS oracle is not a mathematical bound. This is not a new fit.

### fig08_gp_coverage_diagnostic.pdf

Empirical coverage of the P4 nominal 95% voltage band on held-out conditions.

Diagnostic outside the main interpolation criterion. Coverage is not calibrated in all conditions: both 10 C targets fail the 85–99% range. Bands use 200 bounded posterior parameter samples, without measurement-noise, NLS-calibration or fitted-mean uncertainty.

## Validacao

[manifest.json](manifest.json) registra SHA-256 dos dados e PDFs,
dimensoes, fontes incorporadas, minimo tipografico e ausencia de imagens raster.
Extensao de texto e dimensoes foram verificadas antes e depois da exportacao.
O gerador verifica os hashes das fontes ao terminar; resultados anteriores nao sao sobrescritos.

Para reproduzir: execute [figures_ieee.py](../figures_ieee.py) com o ambiente cientifico
existente e pypdf. [test_figures_ieee.py](../test_figures_ieee.py) valida os artefatos.
