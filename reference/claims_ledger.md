# Registro de afirmacoes: paper-v2

Este registro separa observacoes descritivas de inferencia e causalidade. Nao
reescreve vereditos anteriores. Seeds nao sao replicas de celulas; tempos de
amostragem nao sao replicas estatisticas. Etapa 1: nenhuma nova experiencia.

## 1. Equivalencia de MAE com ECM-2RC na maioria das condicoes

**Afirmacao:** com dez seeds, TOST de Wilcoxon exato com margem +/-1 mV
demonstra equivalencia do MAE dos modelos fisicos treinaveis ao ECM-2RC
na maioria, mas nao em todas as condicoes avaliadas.

- Seeds 42--51; ambas as caudas devem ter pHolm<0.05, familias fixas de
  20 condicoes por modelo/modo/direcao. Continuo e principal; janelas secundario.
- As duas condicoes CALCE US06/BJDST25 com fallback continuam nas familias
  pre-registradas, mas nao sustentam a afirmacao principal: nao se substituem
  suas metricas por resultados corrigidos dentro de um TOST ja executado.

| Modelo | Continuo equivalente/20 | Principal sem fallback/18 | Janelas equivalente/20 | Janelas sem fallback/18 |
|---|---:|---:|---:|---:|
| Fisica por gradiente |17|15|19|17|
| Residual na saida |17|15|19|17|
| ECM-UDE |16|14|19|17|

Todas as condicoes sem equivalencia demonstrada sao Panasonic:

| Modelo | Modo | Condicao | Delta mediana (mV) | pHolm superior | Motivo |
|---|---|---|---:|---:|---|
| Gradiente |continuo|US06 25 C|2.969|1.000000|Diferenca mediana maior que +1 mV|
| Gradiente |continuo|LA92 25 C|1.225|1.000000|Diferenca mediana maior que +1 mV|
| Gradiente |continuo|HWFET 25 C|0.911|0.055664|Teste inconclusivo apos Holm; mediana dentro da margem|
| Gradiente |janelas|US06 25 C|1.817|1.000000|Diferenca mediana maior que +1 mV|
| Residual |continuo|US06 25 C|2.967|1.000000|Diferenca mediana maior que +1 mV|
| Residual |continuo|LA92 25 C|1.224|1.000000|Diferenca mediana maior que +1 mV|
| Residual |continuo|HWFET 25 C|0.909|0.055664|Teste inconclusivo apos Holm; mediana dentro da margem|
| Residual |janelas|US06 25 C|1.824|1.000000|Diferenca mediana maior que +1 mV|
| ECM-UDE |continuo|UDDS -20 C|0.621|0.054688|Teste inconclusivo apos Holm; mediana dentro da margem|
| ECM-UDE |continuo|US06 25 C|3.192|1.000000|Diferenca mediana maior que +1 mV|
| ECM-UDE |continuo|LA92 25 C|1.378|1.000000|Diferenca mediana maior que +1 mV|
| ECM-UDE |continuo|HWFET 25 C|1.194|1.000000|Diferenca mediana maior que +1 mV|
| ECM-UDE |janelas|US06 25 C|1.938|1.000000|Diferenca mediana maior que +1 mV|

pHolm inferior=0.019531 em todas essas linhas. O motivo "maior que a margem"
descreve a mediana observada, nao um novo teste de diferenca ou inferioridade.
Falha do TOST nao prova beneficio, equivalencia universal ou ausencia de
pequenas melhorias nominais. Seeds representam variacao do otimizador, nao celulas.
- B/gradiente nao tem rede neural: e fisica com parametros/SOC/eta aprendiveis.
  Residual e UDE usam estrutura1RC inicial, nao um2RC neuralmente aumentado;
  budgets iguais30epocas nao garantem otimizacao perfeita.

Origem: [tabela dez seeds](results/unified_10seeds/table.csv),
[TOST](results/unified_10seeds/statistics.json),
[manifesto](results/unified_10seeds/summary.json),
[protocolo](paper_v2_seeds_protocol.json), [relatorio](results/unified_10seeds/report.md).
Commits: registro `d5cdc37`, resultados `81ac420`; primeiras cinco seeds
do treino `e10d552`, geracao historica `21a2a8c`.
Ressalvas: early stopping usa held-out da fonte (validacao, nao teste independente);
alvos zero-shot; NLS deterministico; RBF/GP posteriores nao participam desta evidencia.

## 2. Estrutura fisica tem erro menor que LSTM, com exclusoes explicitadas

**Afirmacao:** nas18 condicoes sem o fallback CALCE25 comprometido, ECM-2RC
em janelas tem MAE menor que LSTM em todas, sem extrapolar para superioridade universal.

- Razao MAE_LSTM/MAE_2RC em18 condicoes:1.292--5.414, mediana1.846.
- Panasonic UDDS25:42.849 versus82.381mV; LA9225:16.594 versus24.263mV.
- CALCE FUDS25:13.177 versus71.336mV; DST25:5.789 versus29.006mV.
- LSTM aparece apenas em janelas, nunca comparar diretamente seu MAE com2RC continuo.
- Duas vitorias nominais LSTM em CALCE US06/BJDST25 sao excluidas abaixo.
  Origens: [tabela original](results/unified/table_main.csv),
  [manifesto original](results/unified/summary.json),
  [protocolo executavel](protocol.py), [relatorio original](results/unified/report.md).
  Commits `e10d552` e `21a2a8c`. LSTM tem ate150epocas,
  demais treinaveis30; cinco seeds e mesma celula por dataset nao generalizam
  para outras arquiteturas, orcamentos ou fabricantes.
- Atualizacao dez seeds: diferenca mediana LSTM menos2RC de 7.760 a
  66.285 mV nas18 condicoes principais, com todos os18 intervalos bootstrap95%
  inteiramente positivos; apenas tamanho de efeito, sem TOST.
  CALCE corrigido: 2RC/LSTM em janelas US06=19.001/38.668 mV e
  BJDST=18.899/36.252 mV (LSTM media dez seeds).
  Fontes: [efeitos](results/unified_10seeds/statistics.json),
  [avaliacao corrigida](results/calce_z0_fix/summary.json), commit `f48deb8`.

## 3. ESC nao melhora robustamente as duas replicas LFP

**Afirmacao:** o ESC de Plett falhou no teste primario A123, mas passou na
replica Mendeley; portanto a melhoria nao e robusta entre as duas celulas.

| Celula | Perfil | Melhor V1/V2 | V3 ESC | Reducao | Saturacao |
|---|---|---:|---:|---:|---:|
| A123 | FUDS |22.978|23.350|-1.62%|36.40%|
| A123 | US06 |20.506|21.765|-6.14%|55.15%|
| Mendeley | FUDS |18.842|16.357|13.19%|21.09%|
| Mendeley | US06 |17.325|13.173|23.97%|24.48%|

UnidadesmV. A NO-GO, B GO, global NO-GO pelo A123; nao omitir a discordancia.
Mendeley usa Q2.498902Ah, nao rotulo1.1Ah; temperatura25C nominal versus30D,
z0/h0 assumidos com autorizacao, h permanece positivo: ganho nao prova
identificacao de lacos menores. A123 atingeR2=0.5ohm; B/V3 tau=100000s.
Fonte: [protocolo](lfp_gonogo_protocol.json),
[A](results/lfp_gonogo/cell_A_a123/summary.json),
[B](results/lfp_gonogo/cell_B_mendeley/summary.json),
[relatorio](report_gonogo.md). Commits `da67c21`,`6f6ee1d`,`f1dbf9b`;
normalizacaoCSV/proveniencia `d8d048e`, sem mudanca numerica.

## 4. Reidentificacao do mapa OCV/SOC remove parcela grande do erro

**Afirmacao:** permitir recalibracao estatica OCV(z(t)) junto comRC remove
grande parcela do erro observado, sem identificar de forma unica sua causa fisica.

V1 versusV4 alvo (MAEmV):
A123 DST24.442->5.953,FUDS22.978->6.216,US0620.960->4.776;
Mendeley DST10.568->4.867,FUDS18.842->10.891,US0617.917->5.192.
V4 ajustaOCV grau9 E RC no proprio alvo: e oraculo com acesso a tensao alvo,
nao modelo transferivel nem prova de que o erro seja exclusivamenteOCV.
ConfundimentoQ/z0/OCV e polarizacao lenta permanece; aproximacaoOCV baixa
corrente e repouso finito; boundsRC ate100000s diferentes de testes estaticos.
Fontes/protocolo/commits: mesmos do achado3.

## 5. Recalibracao da escala de SOC (V6) a mesma temperatura; temperatura separada

**Afirmacao:** a recalibracao da escala de SOC (V6, OCV fixa e Q/z0/RC
ajustados na fonte) transfere fortemente entre os dois ciclos A123 a mesma
temperatura, mas nao demonstra ganho >=25% em todos os alvos das outras
celulas; a recuperacao termica por V6 local e interpolacao P2 e evidencia separada.

### 5a. V1 versus V6 a temperatura nominal fixa

| Celula | Alvo | V1 (mV) | V6 (mV) | Reducao V6/V1 |
|---|---|---:|---:|---:|
| A123 |FUDS|22.371|4.109|81.63%|
| A123 |US06|21.177|4.952|76.62%|
| Mendeley |FUDS|18.842|15.268|18.97%|
| Mendeley |US06|17.917|12.084|32.56%|
| Panasonic |US06|33.187|32.223|2.90%|
| Panasonic |LA92|16.999|14.009|17.59%|
| Panasonic |HWFET|26.915|26.778|0.51%|
| CALCE corrigido |DST|14.006|31.380|-124.05%|
| CALCE corrigido |US06|19.988|33.208|-66.14%|
| CALCE corrigido |BJDST|20.620|32.741|-58.78%|

Aplicando descritivamente o limiar25% a V6 sozinho em todos os alvos:
A123 passa; Mendeley, Panasonic e CALCE nao passam. Esta e uma leitura dos
ajustes salvos, NAO um novo teste pre-registrado nem substituicao retroativa do
criterio original best-of V5/V6/V7. V6 reajusta tambem RC, portanto nao isola
causalmente capacidade/estado inicial. Q limitado +/-10%; z0 em[.9,1].

Fontes: [A123](results/lfp_static/cell_A_a123/summary.json),
[Mendeley](results/lfp_static/cell_B_mendeley/summary.json),
[Panasonic](results/static_replication/panasonic/summary.json),
[CALCE fonte](results/static_replication/calce/summary.json),
[CALCE corrigido](results/calce_z0_fix/summary.json).
Protocolos/commits: [LFP](lfp_static_protocol.json) `50e85e1`/`a63c83d`;
[replicacao](static_replication_protocol.json) `78fefb3`/`4fa83f6`;
[z0](paper_v2_calce_z0_protocol.json) `308dc2a`/`f48deb8`.

CALCE US06/BJDST: ultima carga completa no proprio arquivo permite
z0=0.799969/0.799944 por contagem deAh, sem retreino. V1 corrigido;
V6 conserva Q/z0fonte, nao recebe z0alvo retrospectivamente.
O criterio original best-of, com baseline corrigido, e **NEGATIVO**:
US06 melhorV7=15.300 mV versusV1=19.988 mV, reducao23.45% (<25%);
BJDST=12.747 versus20.620 mV,38.18%. Veredito historico nao alterado;
V7 nao e resultado principal da reformulacao V6.

### 5b. Robustez monotona: falha e identificabilidade

Mendeley **falha** com V7m: FUDS18.082 mV (4.04% de reducao),
US0618.224 mV (-1.72%); nao atinge25%.
A123, Panasonic e CALCE sao **INCONCLUSIVOS** no teste monotono:
nenhuma das tres inicializacoes convergiu em500avaliacoes; nenhum parametro
falho foi pontuado nos alvos.
Ha nao identificabilidade estrutural de Q e z0 em TODOS os datasets V7m:
recalcular suporteSOCfonte cancela esses parametros na coordenada interpolada.
Isso compromete interpretacao fisica, mas nao demonstra que tenha sido a
causa unica da nao convergencia (inclusive Mendeley convergiu).
Fontes: [manifesto](results/static_monotone/summary.json),
[relatorio](results/static_monotone/report.md),
[protocolo](paper_v2_monotone_protocol.json),
[politica autorizada](paper_v2_monotone_failure_policy.json).
Commits registro `8a0e2e2`, politica `3c149a8`, resultados `0249395`.
V7m fica como diagnostico de robustez, nao resultado principal V6.

### 5c. Temperatura: V6 transferido, V6 local e interpolacao P2

Esta parte usa somente V6 com OCV25 fixa; NAO depende de V5, V7 ou V7m.

| T (C) | Perfil | V6 transferido25 (mV) | V6 local (mV) | Recuperacao local |
|---|---|---:|---:|---:|
|0|FUDS|95.232|19.589|79.43%|
|0|US06|88.308|12.743|85.57%|
|10|FUDS|35.158|6.135|82.55%|
|10|US06|47.559|7.300|84.65%|
|40|FUDS|36.891|4.571|87.61%|
|40|US06|29.063|9.578|67.04%|
|50|FUDS|40.510|5.583|86.22%|
|50|US06|42.583|5.929|86.08%|

V6 transferido nao reduz MAE >=25% versusV1 em nenhum dos oito alvos;
localV6, ajustado somente noDST da mesmaT, recupera67.04--87.61% da sua
perda e e diagnostico fora do criterio termico registrado.
Fonte: [temperatura](results/lfp_static_temperature/cell_A_a123/summary.json),
[diagnostico](a123_temperature_diagnostic.json),
[relatorio](report_static_temperature.md); commits `3d0d047`/`c0bab49`.

P2 interpola os parametros V6 calibrados apenas emDST0/25/50C.
Nos alvos retidos10FUDS,10US06,40FUDS,40US06, recupera respectivamente
61.83%,62.03%,71.09%,116.48% do intervalo V6transferido25--V6local:
**POSITIVO** pelo criterio registrado >=50% nas quatro condicoes.
Fonte: [interpolacao](results/a123_temperature_interp/summary.json),
[protocolo](a123_temperature_interp_protocol.json),
[relatorio](report_temperature_interp.md); commits `b2d945c`/`bb98d2e`.
P2>100% e valido: localNLS nao e limite matematico deMAE.
P3/P4 sao comparadores e P4 cobertura e diagnostico, nao evidencia principal:
58.87%,67.07%,90.61%,89.94%; incerteza NAO calibrada.

Ressalvas gerais: PanasonicV1 usaOCV grau5 do checkpoint dinamico, nao curva
independente; dt1s versus0.1s unificado e RCexato versusRK4.
Mendeley nominal25C versus30D e preparacao/z0assumidos. CALCE fonte~.823,
mas V6z0restrito[.9,1], portanto a compensacao e limitada/confundida.
Temperatura usaOCV25/Q25referencia, sem informacao alvo na calibracao P2;
posteriorGP200amostras condicionadas a bounds sem ruidoV/NLS/mediaP3.

## Numeros que NAO devem ser resultados principais

1. **CALCE US06/BJDST25 com fallback z0=1**, ambas preparadas nominal80SOC:
   - Unificado janelas:2RC123.575/123.366mV,LSTM120.748/120.469mV;
     continuo2RC121.807/122.683mV,UDE121.965/122.827mV.
   - Replicacao:V1=130.408/130.514mV,melhorV7=15.300/12.747mV,
     reducoes88.27/90.23%. VereditoCALCE original POSITIVA permanece
     historico, mas esses ganhos nao sustentam uma afirmacao principal.
     A auditoria ja corrigiu z0~.8; usar os novos baselines19.988/20.620mV.
     Nao usar os antigos ganhosV6 de74.54/74.91%: corrigidos sao
     -66.14/-58.78%. Nao atribuir compensacao de fallback aOCV.
2. **Qualquer V5 com extrapolacao nao fisica**:
   CALCE US0625=826.648mV,BJDST25=817.954mV; maximaV20.117/20.109V.
   Excluir como evidencia principal de desempenho/ganho; conservar como
   falha metodologica documentada. Nao excluir silenciosamente de
   tabelas historicas nem recortar erros para melhorar metricas.
3. **Best-of V5/V6/V7 como se fosse V6**: A123 US0676.91% nao e V6
   (V6=76.62%); Mendeley37.14/73.58% e Panasonic27.15/29.59/28.60%
   nao sustentam transferencia V6. Conservar como evidencia historica apenas.
4. **V7m/Q/z0**: parametros estruturalmente nao identificaveis, ou ajustes
   nao convergidos de A123/Panasonic/CALCE, nao sao estimativas fisicas nem
   vereditos negativos medidos. Mendeley V7m e diagnostico de falha, nao
   substituir seus numeros V6. V5/V7/V7m nao sao resultados principais do achado5.
5. **Equivalencia universal ou baseada em cinco seeds**: nao usar falta
   de significancia como TOST; listar todas as excecoes e distinguir
   medianas fora da margem de pHolm inconclusivo. TOST nos dois fallbacks
   nao prova equivalencia nas condicoes com z0corrigido (nao foi reexecutado).
6. Comparacoes ingenuas de oraculos ajustados no alvo com modeloszero-shot;
   ausencia dep<.05 com5seeds como prova deequivalencia; coberturaGP nominal
   como cobertura empirica; resultadoSOC fallback como propriedade quimica.
