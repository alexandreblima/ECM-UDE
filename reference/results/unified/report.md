# Comparação unificada ECM–UDE

Execução concluída em 01/11/2026. Branch `protocolo-unificado`; preservação `389172b`, código de treino `e10d552`, geração do relatório `21a2a8c`. O manuscrito permaneceu intacto.

Foram concluídos 40 treinos (quatro modelos × cinco seeds × dois datasets) e quatro identificações NLS. Os 44 checkpoints/arquivos de parâmetros têm hashes conferidos. A tabela completa contém 220 linhas, com MAE, RMSE, máximo e P99 em mV: [CSV](table_main.csv), [LaTeX](table_main.tex).

O protocolo usa SOC causal por capacidade nominal (2,9/2,0 Ah), OCV limitada a [0,1], RK4 clássico com entrada mantida por passo e split temporal 80/20 com exclusão de janelas sobrepostas na fronteira. A normalização usa somente o treino. Os alvos OOD são integralmente avaliados. No modo contínuo, os modelos físicos começam no início do ciclo e são pontuados apenas na máscara held-out da origem. LSTM aparece apenas no modo de janelas.

Nas tabelas, B é a física com SOC interno/eta aprendível, sem rede, ajustada por gradiente; Residual corrige a saída. B, Residual e UDE compartilham o orçamento de 30 épocas e otimizador. A LSTM conserva o orçamento de até 150 épocas. As seeds são 42–46. Os NLS são determinísticos, com um ajuste por dataset.

**Estatística:** nenhuma comparação entre vencedor e segundo colocado apresenta significância a 5%. Com cinco pares não nulos, o menor p exato bilateral possível no Wilcoxon é 0,0625, antes mesmo de Holm. As unidades são as seeds; amostras temporais não são réplicas independentes. Comparações entre dois NLS determinísticos não recebem p-valor. Os testes UDE versus cada baseline, para todas as métricas, estão em [statistics.json](statistics.json); os vencedores por métrica estão em [winners.csv](winners.csv).

**Limites de interpretação:** o held-out da origem também seleciona os checkpoints por early stopping; é validação, não um terceiro conjunto de teste independente. O CALCE usa OCV independente SP1 a 25 °C (aproximação de baixa corrente, célula diferente), reutilizada nas demais temperaturas. Exige-se repouso de pelo menos 30 s com |I|≤0,02 A. Panasonic usa o fallback documentado z0=1 por ausência de calibração independente local. CALCE US06/BJDST a 25 °C também usa z0=1 por falta de repouso qualificado; as demais condições usam inversão OCV. O FUDS25 começa em z0≈0,82291.

**Leitura descritiva:** o UDE não apresenta vantagem geral específica da correção dinâmica. O ECM-2RC lidera em MAE nas oito condições Panasonic com janelas e em 15/20 condições no modo contínuo. O residual na saída lidera em oito condições com janelas no CALCE; o UDE lidera apenas no DST25 com janelas, por cerca de 0,020 mV sobre B. As duas vitórias da LSTM ocorrem nos alvos CALCE25 com fallback de SOC, o que deve acompanhar sua interpretação.

## MAE em mV — média ± desvio amostral

NLS: valor determinístico. Negrito: menor média, sem implicar significância estatística. Todas as células de origem (UDDS25 Panasonic e FUDS25 CALCE) usam apenas held-out.

| Dataset/condição | Modo | 1RC | 2RC | B | Residual | UDE | LSTM | Menor MAE | Segundo | p bruto | p Holm | Significativo 5% |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| panasonic/UDDS_25C | windowed | 43.051 | **42.849** | 42.947 ± 0.147 | 42.953 ± 0.137 | 43.090 ± 0.230 | 82.381 ± 4.633 | 2RC | B | 0.3125 | 1.0000 | Não |
| panasonic/UDDS_25C | continuous | 43.148 | 42.661 | **42.611 ± 0.264** | 42.617 ± 0.270 | 43.164 ± 0.258 | — | B | Residual | 0.3125 | 1.0000 | Não |
| panasonic/UDDS_10C | windowed | 37.537 | **37.312** | 37.515 ± 0.106 | 37.508 ± 0.112 | 37.726 ± 0.080 | 74.525 ± 5.650 | 2RC | Residual | 0.0625 | 0.8750 | Não |
| panasonic/UDDS_10C | continuous | 36.755 | **36.263** | 36.571 ± 0.193 | 36.565 ± 0.199 | 36.945 ± 0.071 | — | 2RC | Residual | 0.0625 | 0.5625 | Não |
| panasonic/UDDS_0C | windowed | 96.163 | **96.107** | 96.148 ± 0.136 | 96.135 ± 0.143 | 96.394 ± 0.166 | 155.418 ± 9.283 | 2RC | Residual | 0.8125 | 1.0000 | Não |
| panasonic/UDDS_0C | continuous | 94.287 | **93.816** | 94.116 ± 0.222 | 94.103 ± 0.228 | 94.525 ± 0.164 | — | 2RC | Residual | 0.0625 | 0.5625 | Não |
| panasonic/UDDS_-10C | windowed | 173.957 | **173.900** | 173.940 ± 0.158 | 173.924 ± 0.163 | 174.163 ± 0.261 | 250.523 ± 12.613 | 2RC | Residual | 0.6250 | 1.0000 | Não |
| panasonic/UDDS_-10C | continuous | 171.787 | **171.303** | 171.627 ± 0.239 | 171.611 ± 0.240 | 171.993 ± 0.274 | — | 2RC | Residual | 0.0625 | 0.5625 | Não |
| panasonic/UDDS_-20C | windowed | 295.088 | **295.021** | 295.050 ± 0.176 | 295.032 ± 0.174 | 295.238 ± 0.360 | 381.222 ± 14.923 | 2RC | Residual | 0.8125 | 1.0000 | Não |
| panasonic/UDDS_-20C | continuous | 292.906 | **292.392** | 292.733 ± 0.253 | 292.715 ± 0.248 | 293.045 ± 0.391 | — | 2RC | Residual | 0.0625 | 0.5625 | Não |
| panasonic/US06_25C | windowed | 38.774 | **36.944** | 38.727 ± 0.089 | 38.731 ± 0.092 | 38.852 ± 0.074 | 58.866 ± 2.256 | 2RC | B | 0.0625 | 0.8750 | Não |
| panasonic/US06_25C | continuous | 34.217 | **31.120** | 34.010 ± 0.171 | 34.014 ± 0.175 | 34.273 ± 0.094 | — | 2RC | B | 0.0625 | 0.5625 | Não |
| panasonic/LA92_25C | windowed | 17.467 | **16.594** | 17.446 ± 0.082 | 17.448 ± 0.084 | 17.556 ± 0.029 | 24.263 ± 0.860 | 2RC | B | 0.0625 | 0.8750 | Não |
| panasonic/LA92_25C | continuous | 15.903 | **14.592** | 15.754 ± 0.128 | 15.756 ± 0.132 | 15.969 ± 0.049 | — | 2RC | B | 0.0625 | 0.5625 | Não |
| panasonic/HWFET_25C | windowed | 39.637 | **39.340** | 39.583 ± 0.118 | 39.585 ± 0.123 | 39.763 ± 0.048 | 54.847 ± 1.765 | 2RC | B | 0.0625 | 0.8750 | Não |
| panasonic/HWFET_25C | continuous | 36.080 | **35.005** | 35.805 ± 0.245 | 35.808 ± 0.251 | 36.192 ± 0.073 | — | 2RC | B | 0.0625 | 0.5625 | Não |
| calce/FUDS_25C | windowed | 13.218 | **13.177** | 13.187 ± 0.047 | 13.189 ± 0.049 | 13.184 ± 0.043 | 71.336 ± 3.198 | 2RC | UDE | 1.0000 | 1.0000 | Não |
| calce/FUDS_25C | continuous | 13.328 | **13.258** | 13.265 ± 0.072 | 13.267 ± 0.076 | 13.261 ± 0.067 | — | 2RC | UDE | 0.8125 | 1.0000 | Não |
| calce/DST_25C | windowed | 5.802 | 5.789 | 5.771 ± 0.075 | 5.774 ± 0.077 | **5.751 ± 0.071** | 29.006 ± 16.890 | UDE | B | 0.8125 | 1.0000 | Não |
| calce/DST_25C | continuous | 5.203 | **5.102** | 5.162 ± 0.073 | 5.164 ± 0.075 | 5.157 ± 0.054 | — | 2RC | UDE | 0.0625 | 0.5625 | Não |
| calce/US06_25C | windowed | 123.701 | 123.575 | 123.707 ± 0.075 | 123.712 ± 0.073 | 123.634 ± 0.127 | **120.748 ± 11.155** | LSTM | 2RC | 1.0000 | 1.0000 | Não |
| calce/US06_25C | continuous | 122.049 | **121.807** | 122.032 ± 0.065 | 122.037 ± 0.064 | 121.965 ± 0.140 | — | 2RC | UDE | 0.1250 | 0.8750 | Não |
| calce/BJDST_25C | windowed | 123.481 | 123.366 | 123.487 ± 0.075 | 123.491 ± 0.073 | 123.414 ± 0.127 | **120.469 ± 12.528** | LSTM | 2RC | 0.8125 | 1.0000 | Não |
| calce/BJDST_25C | continuous | 122.911 | **122.683** | 122.894 ± 0.064 | 122.899 ± 0.064 | 122.827 ± 0.140 | — | 2RC | UDE | 0.1875 | 1.0000 | Não |
| calce/FUDS_0C | windowed | 53.873 | 53.871 | 53.880 ± 0.070 | **53.865 ± 0.063** | 54.059 ± 0.163 | 99.653 ± 4.946 | Residual | 2RC | 0.8125 | 1.0000 | Não |
| calce/FUDS_0C | continuous | 53.326 | **53.226** | 53.316 ± 0.058 | 53.301 ± 0.053 | 53.514 ± 0.164 | — | 2RC | Residual | 0.0625 | 0.5625 | Não |
| calce/DST_0C | windowed | 55.549 | 55.560 | 55.552 ± 0.076 | **55.536 ± 0.069** | 55.744 ± 0.171 | 102.399 ± 5.186 | Residual | 1RC | 0.8125 | 1.0000 | Não |
| calce/DST_0C | continuous | 54.638 | **54.542** | 54.622 ± 0.065 | 54.606 ± 0.060 | 54.835 ± 0.173 | — | 2RC | Residual | 0.1250 | 0.5625 | Não |
| calce/US06_0C | windowed | 64.370 | 64.379 | 64.367 ± 0.078 | **64.351 ± 0.071** | 64.569 ± 0.179 | 111.919 ± 6.137 | Residual | B | 0.0625 | 0.8750 | Não |
| calce/US06_0C | continuous | 63.790 | **63.682** | 63.766 ± 0.068 | 63.750 ± 0.063 | 63.992 ± 0.182 | — | 2RC | Residual | 0.1250 | 0.6250 | Não |
| calce/BJDST_0C | windowed | 75.043 | 75.075 | 75.033 ± 0.083 | **75.017 ± 0.076** | 75.237 ± 0.180 | 123.305 ± 8.149 | Residual | B | 0.0625 | 0.8750 | Não |
| calce/BJDST_0C | continuous | 74.378 | **74.297** | 74.346 ± 0.073 | 74.330 ± 0.069 | 74.576 ± 0.184 | — | 2RC | Residual | 0.4375 | 1.0000 | Não |
| calce/FUDS_45C | windowed | 17.515 | 17.550 | 17.504 ± 0.055 | **17.496 ± 0.055** | 17.653 ± 0.124 | 64.566 ± 8.947 | Residual | B | 0.1875 | 0.8750 | Não |
| calce/FUDS_45C | continuous | 17.477 | 17.545 | 17.464 ± 0.054 | **17.456 ± 0.054** | 17.628 ± 0.135 | — | Residual | B | 0.1875 | 0.7500 | Não |
| calce/DST_45C | windowed | 17.054 | 17.158 | 17.033 ± 0.053 | **17.023 ± 0.052** | 17.222 ± 0.151 | 63.937 ± 7.539 | Residual | B | 0.1250 | 0.8750 | Não |
| calce/DST_45C | continuous | 17.511 | 17.680 | 17.500 ± 0.047 | **17.486 ± 0.046** | 17.753 ± 0.206 | — | Residual | B | 0.1250 | 0.5625 | Não |
| calce/US06_45C | windowed | 15.904 | 15.999 | 15.893 ± 0.066 | **15.887 ± 0.065** | 16.011 ± 0.103 | 62.420 ± 7.637 | Residual | B | 0.1875 | 0.8750 | Não |
| calce/US06_45C | continuous | 16.061 | 16.184 | 16.043 ± 0.066 | **16.035 ± 0.066** | 16.208 ± 0.137 | — | Residual | B | 0.1875 | 0.7500 | Não |
| calce/BJDST_45C | windowed | 18.378 | 18.487 | 18.358 ± 0.075 | **18.353 ± 0.074** | 18.480 ± 0.107 | 63.787 ± 5.526 | Residual | B | 0.1875 | 0.8750 | Não |
| calce/BJDST_45C | continuous | 18.368 | 18.516 | 18.341 ± 0.076 | **18.333 ± 0.075** | 18.531 ± 0.157 | — | Residual | B | 0.1875 | 0.7500 | Não |

Conferências: [validation.json](validation.json). Manifesto, dados e checkpoints: [summary.json](summary.json). Logs completos: `logs/`. O backup anterior foi verificado contra todos os seus 270 arquivos, sem alterações.
