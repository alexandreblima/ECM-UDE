# Figuras finais IEEE

PDFs vetoriais; largura exata 3,5 in ou 7,16 in; Times New Roman incorporada, minimo 8 pt.
Sem titulos internos. Cores acompanhadas de marcadores, tipos de linha ou hachuras distintos;
as series continuam identificaveis sem cor. Rotulo canonico: HWFET.

Dados exclusivamente de resultados salvos citados pelo registro de afirmacoes; nenhum retreino,
reajuste ou nova simulacao. Figura1 usa TOST dez seeds; figura2 conserva os cinco seeds historicos.
Figuras5/6 usam V6 apenas; figura7 conserva a interpolacao dos parametros V6.

Revisao: apenas PDFs1,5,6 regenerados; PDFs2,3,4,7,8 preservados byte a byte.
Versoes anteriores dos arquivos revisados: [arquivo ab6cdab](archive/ab6cdab/claims_ledger.md).

## Analise das oito figuras

- Figura1 afetada: cinco seeds/desempenho nominal substituidos por TOST dez seeds com excecoes explicitas.
- Figura5 afetada: best-of V5/V6/V7 substituido por V1 versus V6 em cada celula, CALCE corrigido incluido.
- Figura6 afetada: melhor transferencia substituida por V6 transferido25 versus V6 local e V1.
- Figura2 nao afetada: comparacao historica em janelas continua valida, com exclusoes declaradas.
- Figuras3/4 nao afetadas: ESC e oraculo alvo mantem as mesmas afirmacoes qualificadas.
- Figura7 nao afetada: P2/P3/P4 interpolam parametros V6, nao OCV V5/V7/V7m.
- Figura8 nao afetada: cobertura P4 permanece diagnostico fora do resultado principal.
- [Auditoria SHA-256](../results/claims_revision_audit/preservation.json) confirma preservacao.

| Figura | Achado | Largura (in) | Arquivos de dados de origem |
|---|---|---:|---|
| [fig01_dynamic_corrections.pdf](fig01_dynamic_corrections.pdf) | 1 | 7.16 | [results/unified_10seeds/statistics.json](../results/unified_10seeds/statistics.json) |
| [fig02_physics_vs_lstm.pdf](fig02_physics_vs_lstm.pdf) | 2 | 7.16 | [results/unified/summary.json](../results/unified/summary.json) |
| [fig03_lfp_hysteresis.pdf](fig03_lfp_hysteresis.pdf) | 3 | 7.16 | [results/lfp_gonogo/cell_A_a123/summary.json](../results/lfp_gonogo/cell_A_a123/summary.json), [results/lfp_gonogo/cell_B_mendeley/summary.json](../results/lfp_gonogo/cell_B_mendeley/summary.json) |
| [fig04_static_oracle.pdf](fig04_static_oracle.pdf) | 4 | 3.50 | [results/lfp_gonogo/cell_A_a123/summary.json](../results/lfp_gonogo/cell_A_a123/summary.json), [results/lfp_gonogo/cell_B_mendeley/summary.json](../results/lfp_gonogo/cell_B_mendeley/summary.json) |
| [fig05_same_temperature_transfer.pdf](fig05_same_temperature_transfer.pdf) | 5 | 7.16 | [results/lfp_static/cell_A_a123/summary.json](../results/lfp_static/cell_A_a123/summary.json), [results/lfp_static/cell_B_mendeley/summary.json](../results/lfp_static/cell_B_mendeley/summary.json), [results/static_replication/panasonic/summary.json](../results/static_replication/panasonic/summary.json), [results/static_replication/calce/summary.json](../results/static_replication/calce/summary.json), [results/calce_z0_fix/summary.json](../results/calce_z0_fix/summary.json) |
| [fig06_thermal_transfer.pdf](fig06_thermal_transfer.pdf) | 5 | 7.16 | [results/lfp_static_temperature/cell_A_a123/summary.json](../results/lfp_static_temperature/cell_A_a123/summary.json) |
| [fig07_temperature_interpolation.pdf](fig07_temperature_interpolation.pdf) | 5 | 3.50 | [results/a123_temperature_interp/summary.json](../results/a123_temperature_interp/summary.json) |
| [fig08_gp_coverage_diagnostic.pdf](fig08_gp_coverage_diagnostic.pdf) | 5 | 3.50 | [results/a123_temperature_interp/summary.json](../results/a123_temperature_interp/summary.json) |

## Legendas propostas e ressalvas

### fig01_dynamic_corrections.pdf

Ten-seed continuous MAE equivalence to ECM-2RC: Panasonic (left), CALCE (right).

Markers are median differences; bars show the ten-seed minimum/maximum, NOT confidence intervals. Shading denotes +/-1 mV. Filled markers: both one-sided exact Wilcoxon TOST pHolm<0.05; hollow markers: equivalence not demonstrated. Holm families have 20 conditions per model/direction. Gradient/residual/UDE are equivalent in 15/18, 15/18, 14/18 principal conditions. Median outside the margin and inconclusive corrected tests are distinct failures (see ledger). CALCE US06/BJDST 25 C fallback conditions excluded; source scores are validation scores. Physics gradient has no neural network; trainable physical models initialize from 1RC.

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

Same-temperature V1 versus source-fitted V6: A123, Mendeley, Panasonic and corrected CALCE.

Only V1/V6 are principal series: fixed OCV, source-reidentified Q/z0 and RC for V6. A123 gains 81.63/76.62%; Mendeley 18.97/32.56%; Panasonic 2.90/17.59/0.51%; CALCE -124.05/-66.14/-58.78%. V6 alone passes 25% on all targets only in A123, a descriptive restriction of the saved original best-of test, not new preregistration. CALCE US06/BJDST V1 uses audited z0 near 0.8; V6 keeps source-fitted z0. Corrected original CALCE best-of verdict is negative. No V5/V7/V7m principal curves. Panasonic OCV comes from a dynamic checkpoint; Mendeley temperature is nominal. The V7m robustness failure/nonidentifiability is separate (see ledger).

### fig06_thermal_transfer.pdf

A123 thermal transfer and local recalibration: FUDS (left), US06 (right).

Transferred V6 is frozen from DST 25 C; no target refit. V6 fails the >=25% gain versus V1 at all eight target conditions. Local V6 is fitted on the same-temperature DST and is only an out-of-criterion diagnostic, not part of the transfer decision; it recovers 67.04–87.61% of transferred V6 error. This thermal evidence and P2 parameter interpolation do not depend on V5, V7 or V7m. All fits retain fixed OCV25/Q25 references.

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
