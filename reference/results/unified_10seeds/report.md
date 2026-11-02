# Unificado: dez seeds e equivalencia

Seeds42--51, cinco reutilizadas e cinco novas; hiperparametros e protocolo de treino inalterados.
TOST exato deWilcoxon deslocado, margem+/-1mV; Holm20condicoes por modelo/modo/direcao.
Dois testes devem rejeitar com pHolm<0.05; nao equivalencia nao implica melhoria.

| Modo | Modelo | Equivalentes/20 | Equivalentes principais/18 |
|---|---|---:|---:|
| continuous | physics_gradient | 17/20 | 15/18 |
| continuous | output_residual | 17/20 | 15/18 |
| continuous | ecm_ude | 16/20 | 14/18 |
| windowed | physics_gradient | 19/20 | 17/18 |
| windowed | output_residual | 19/20 | 17/18 |
| windowed | ecm_ude | 19/20 | 17/18 |

## TOST por condicao

| Dataset | Condicao | Modo | Modelo | Delta mediana(mV) | pHolm inferior | pHolm superior | Equivalente | Excluida principal |
|---|---|---|---|---:|---:|---:|---|---|
| panasonic | UDDS_25C | continuous | physics_gradient | 0.0565 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_10C | continuous | physics_gradient | 0.3951 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_0C | continuous | physics_gradient | 0.3904 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-10C | continuous | physics_gradient | 0.3973 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-20C | continuous | physics_gradient | 0.4075 | 0.019531 | 0.019531 | True | False |
| panasonic | US06_25C | continuous | physics_gradient | 2.9686 | 0.019531 | 1.000000 | False | False |
| panasonic | LA92_25C | continuous | physics_gradient | 1.2251 | 0.019531 | 1.000000 | False | False |
| panasonic | HWFET_25C | continuous | physics_gradient | 0.9106 | 0.019531 | 0.055664 | False | False |
| calce | FUDS_25C | continuous | physics_gradient | 0.0130 | 0.019531 | 0.019531 | True | False |
| calce | DST_25C | continuous | physics_gradient | 0.0913 | 0.019531 | 0.019531 | True | False |
| calce | US06_25C | continuous | physics_gradient | 0.2397 | 0.019531 | 0.019531 | True | True |
| calce | BJDST_25C | continuous | physics_gradient | 0.2266 | 0.019531 | 0.019531 | True | True |
| calce | FUDS_0C | continuous | physics_gradient | 0.1143 | 0.019531 | 0.019531 | True | False |
| calce | DST_0C | continuous | physics_gradient | 0.1060 | 0.019531 | 0.019531 | True | False |
| calce | US06_0C | continuous | physics_gradient | 0.1121 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_0C | continuous | physics_gradient | 0.0781 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_45C | continuous | physics_gradient | -0.0907 | 0.019531 | 0.019531 | True | False |
| calce | DST_45C | continuous | physics_gradient | -0.1855 | 0.019531 | 0.019531 | True | False |
| calce | US06_45C | continuous | physics_gradient | -0.1472 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_45C | continuous | physics_gradient | -0.1729 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_25C | continuous | output_residual | 0.0729 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_10C | continuous | output_residual | 0.3953 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_0C | continuous | output_residual | 0.3910 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-10C | continuous | output_residual | 0.3994 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-20C | continuous | output_residual | 0.4288 | 0.019531 | 0.019531 | True | False |
| panasonic | US06_25C | continuous | output_residual | 2.9675 | 0.019531 | 1.000000 | False | False |
| panasonic | LA92_25C | continuous | output_residual | 1.2241 | 0.019531 | 1.000000 | False | False |
| panasonic | HWFET_25C | continuous | output_residual | 0.9091 | 0.019531 | 0.055664 | False | False |
| calce | FUDS_25C | continuous | output_residual | 0.0168 | 0.019531 | 0.019531 | True | False |
| calce | DST_25C | continuous | output_residual | 0.0902 | 0.019531 | 0.019531 | True | False |
| calce | US06_25C | continuous | output_residual | 0.2395 | 0.019531 | 0.019531 | True | True |
| calce | BJDST_25C | continuous | output_residual | 0.2269 | 0.019531 | 0.019531 | True | True |
| calce | FUDS_0C | continuous | output_residual | 0.0936 | 0.019531 | 0.019531 | True | False |
| calce | DST_0C | continuous | output_residual | 0.0844 | 0.019531 | 0.019531 | True | False |
| calce | US06_0C | continuous | output_residual | 0.0897 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_0C | continuous | output_residual | 0.0560 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_45C | continuous | output_residual | -0.0911 | 0.019531 | 0.019531 | True | False |
| calce | DST_45C | continuous | output_residual | -0.1893 | 0.019531 | 0.019531 | True | False |
| calce | US06_45C | continuous | output_residual | -0.1472 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_45C | continuous | output_residual | -0.1729 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_25C | continuous | ecm_ude | 0.5851 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_10C | continuous | ecm_ude | 0.6369 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_0C | continuous | ecm_ude | 0.6162 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-10C | continuous | ecm_ude | 0.6521 | 0.019531 | 0.024414 | True | False |
| panasonic | UDDS_-20C | continuous | ecm_ude | 0.6212 | 0.019531 | 0.054688 | False | False |
| panasonic | US06_25C | continuous | ecm_ude | 3.1917 | 0.019531 | 1.000000 | False | False |
| panasonic | LA92_25C | continuous | ecm_ude | 1.3784 | 0.019531 | 1.000000 | False | False |
| panasonic | HWFET_25C | continuous | ecm_ude | 1.1945 | 0.019531 | 1.000000 | False | False |
| calce | FUDS_25C | continuous | ecm_ude | -0.0060 | 0.019531 | 0.019531 | True | False |
| calce | DST_25C | continuous | ecm_ude | 0.0526 | 0.019531 | 0.019531 | True | False |
| calce | US06_25C | continuous | ecm_ude | 0.2102 | 0.019531 | 0.019531 | True | True |
| calce | BJDST_25C | continuous | ecm_ude | 0.1972 | 0.019531 | 0.019531 | True | True |
| calce | FUDS_0C | continuous | ecm_ude | 0.2477 | 0.019531 | 0.019531 | True | False |
| calce | DST_0C | continuous | ecm_ude | 0.2503 | 0.019531 | 0.019531 | True | False |
| calce | US06_0C | continuous | ecm_ude | 0.2676 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_0C | continuous | ecm_ude | 0.2399 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_45C | continuous | ecm_ude | 0.0319 | 0.019531 | 0.019531 | True | False |
| calce | DST_45C | continuous | ecm_ude | 0.0027 | 0.019531 | 0.019531 | True | False |
| calce | US06_45C | continuous | ecm_ude | -0.0287 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_45C | continuous | ecm_ude | -0.0467 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_25C | windowed | physics_gradient | 0.1327 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_10C | windowed | physics_gradient | 0.2420 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_0C | windowed | physics_gradient | 0.0735 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-10C | windowed | physics_gradient | 0.0607 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-20C | windowed | physics_gradient | 0.0453 | 0.019531 | 0.019531 | True | False |
| panasonic | US06_25C | windowed | physics_gradient | 1.8167 | 0.019531 | 1.000000 | False | False |
| panasonic | LA92_25C | windowed | physics_gradient | 0.8700 | 0.019531 | 0.019531 | True | False |
| panasonic | HWFET_25C | windowed | physics_gradient | 0.3003 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_25C | windowed | physics_gradient | 0.0126 | 0.019531 | 0.019531 | True | False |
| calce | DST_25C | windowed | physics_gradient | 0.0207 | 0.019531 | 0.019531 | True | False |
| calce | US06_25C | windowed | physics_gradient | 0.1491 | 0.019531 | 0.019531 | True | True |
| calce | BJDST_25C | windowed | physics_gradient | 0.1387 | 0.019531 | 0.019531 | True | True |
| calce | FUDS_0C | windowed | physics_gradient | 0.0208 | 0.019531 | 0.019531 | True | False |
| calce | DST_0C | windowed | physics_gradient | 0.0092 | 0.019531 | 0.019531 | True | False |
| calce | US06_0C | windowed | physics_gradient | 0.0115 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_0C | windowed | physics_gradient | -0.0117 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_45C | windowed | physics_gradient | -0.0559 | 0.019531 | 0.019531 | True | False |
| calce | DST_45C | windowed | physics_gradient | -0.1231 | 0.019531 | 0.019531 | True | False |
| calce | US06_45C | windowed | physics_gradient | -0.1185 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_45C | windowed | physics_gradient | -0.1334 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_25C | windowed | output_residual | 0.1250 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_10C | windowed | output_residual | 0.2426 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_0C | windowed | output_residual | 0.0755 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-10C | windowed | output_residual | 0.0627 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-20C | windowed | output_residual | 0.0469 | 0.019531 | 0.019531 | True | False |
| panasonic | US06_25C | windowed | output_residual | 1.8240 | 0.019531 | 1.000000 | False | False |
| panasonic | LA92_25C | windowed | output_residual | 0.8749 | 0.019531 | 0.019531 | True | False |
| panasonic | HWFET_25C | windowed | output_residual | 0.3017 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_25C | windowed | output_residual | 0.0135 | 0.019531 | 0.019531 | True | False |
| calce | DST_25C | windowed | output_residual | 0.0173 | 0.019531 | 0.019531 | True | False |
| calce | US06_25C | windowed | output_residual | 0.1513 | 0.019531 | 0.019531 | True | True |
| calce | BJDST_25C | windowed | output_residual | 0.1408 | 0.019531 | 0.019531 | True | True |
| calce | FUDS_0C | windowed | output_residual | 0.0140 | 0.019531 | 0.019531 | True | False |
| calce | DST_0C | windowed | output_residual | 0.0020 | 0.019531 | 0.019531 | True | False |
| calce | US06_0C | windowed | output_residual | 0.0041 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_0C | windowed | output_residual | -0.0190 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_45C | windowed | output_residual | -0.0568 | 0.019531 | 0.019531 | True | False |
| calce | DST_45C | windowed | output_residual | -0.1267 | 0.019531 | 0.019531 | True | False |
| calce | US06_45C | windowed | output_residual | -0.1182 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_45C | windowed | output_residual | -0.1331 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_25C | windowed | ecm_ude | 0.3096 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_10C | windowed | ecm_ude | 0.3620 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_0C | windowed | ecm_ude | 0.2012 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-10C | windowed | ecm_ude | 0.2220 | 0.019531 | 0.019531 | True | False |
| panasonic | UDDS_-20C | windowed | ecm_ude | 0.1755 | 0.019531 | 0.019531 | True | False |
| panasonic | US06_25C | windowed | ecm_ude | 1.9382 | 0.019531 | 1.000000 | False | False |
| panasonic | LA92_25C | windowed | ecm_ude | 0.9559 | 0.019531 | 0.027344 | True | False |
| panasonic | HWFET_25C | windowed | ecm_ude | 0.4196 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_25C | windowed | ecm_ude | 0.0026 | 0.019531 | 0.019531 | True | False |
| calce | DST_25C | windowed | ecm_ude | -0.0352 | 0.019531 | 0.019531 | True | False |
| calce | US06_25C | windowed | ecm_ude | 0.1062 | 0.019531 | 0.019531 | True | True |
| calce | BJDST_25C | windowed | ecm_ude | 0.0956 | 0.019531 | 0.019531 | True | True |
| calce | FUDS_0C | windowed | ecm_ude | 0.1500 | 0.019531 | 0.019531 | True | False |
| calce | DST_0C | windowed | ecm_ude | 0.1448 | 0.019531 | 0.019531 | True | False |
| calce | US06_0C | windowed | ecm_ude | 0.1502 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_0C | windowed | ecm_ude | 0.1262 | 0.019531 | 0.019531 | True | False |
| calce | FUDS_45C | windowed | ecm_ude | 0.0580 | 0.019531 | 0.019531 | True | False |
| calce | DST_45C | windowed | ecm_ude | 0.0054 | 0.019531 | 0.019531 | True | False |
| calce | US06_45C | windowed | ecm_ude | -0.0285 | 0.019531 | 0.019531 | True | False |
| calce | BJDST_45C | windowed | ecm_ude | -0.0514 | 0.019531 | 0.019531 | True | False |

## LSTM menos2RC (janelas apenas)

| Dataset | Condicao | Mediana(mV) | Bootstrap95%(mV) |
|---|---|---:|---|
| panasonic | UDDS_25C | 39.266 | 35.706--45.078 |
| panasonic | UDDS_10C | 29.482 | 17.595--39.128 |
| panasonic | UDDS_0C | 46.130 | 24.592--62.788 |
| panasonic | UDDS_-10C | 58.785 | 22.374--80.108 |
| panasonic | UDDS_-20C | 66.285 | 16.460--87.992 |
| panasonic | US06_25C | 24.036 | 21.712--25.666 |
| panasonic | LA92_25C | 7.760 | 7.147--8.983 |
| panasonic | HWFET_25C | 15.862 | 14.261--17.952 |
| calce | FUDS_25C | 60.145 | 57.714--61.452 |
| calce | DST_25C | 16.072 | 15.737--16.842 |
| calce | US06_25C | 2.399 | -0.244--3.677 |
| calce | BJDST_25C | 2.630 | 0.053--4.186 |
| calce | FUDS_0C | 28.200 | 9.036--46.982 |
| calce | DST_0C | 29.247 | 8.762--47.586 |
| calce | US06_0C | 30.447 | 10.569--47.733 |
| calce | BJDST_0C | 29.225 | 11.360--49.304 |
| calce | FUDS_45C | 32.295 | 24.611--45.088 |
| calce | DST_45C | 32.680 | 25.026--45.897 |
| calce | US06_45C | 32.558 | 24.896--45.199 |
| calce | BJDST_45C | 32.187 | 24.310--45.666 |

## Ressalvas

- FUDS25/UDDS25 fonte: held-out e validacao usada emearly stopping, nao teste independente.
- Seeds nao replicam celulas;Wilcoxon usa simetria das diferencas deMAE porseed.
- Gradiente e fisica aprendivel sem rede. Treinaveis fisicos partem de1RC, referencia e2RC.
- CALCE US06/BJDST25 fallbackz0=1 ficam na familia20 mas excluidos dos achados principais.
- NLS deterministico nao e multiplicado como dez ajustes independentes.
- Bootstrap porseed, nao poramostra temporal; sem inferenciaLSTM fora de janelas.
- [Manifesto](summary.json),[estatisticas](statistics.json),[tabela](table.csv).
