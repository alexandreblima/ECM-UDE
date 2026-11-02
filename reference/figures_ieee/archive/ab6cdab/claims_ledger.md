# Registro de afirmacoes: paper-v2

Este registro separa observacoes descritivas de inferencia e causalidade. Nao
reescreve vereditos anteriores. Seeds nao sao replicas de celulas; tempos de
amostragem nao sao replicas estatisticas. Etapa 1: nenhuma nova experiencia.

## 1. Correcao da dinamica nao apresenta vantagem geral contra ECM-2RC

**Afirmacao:** nas condicoes avaliadas, as correcoes de dinamica/saida nao
demonstraram reducao geral robusta de MAE contra um ECM-2RC calibrado.

- ECM-2RC tem menor MAE medio em 15/20 condicoes continuas e nas oito
  condicoes Panasonic em janelas.
- Excluindo CALCE US06/BJDST25 com fallback, diferenca MAE modelo menos2RC
  no modo continuo: gradiente [-0.180,2.890]mV; residual [-0.193,2.894]mV;
  ECM-UDE [0.004,3.153]mV. Gradiente/residual melhoram nominalmente em5/18
  condicoes cada; UDE em0/18. Logo "nunca reduzem" seria uma afirmacao falsa.
- Panasonic US0625 continuo: 2RC31.120, gradiente34.010, residual34.014,
  UDE34.273mV. CALCE DST25 janelas: UDE5.751 versus2RC5.789mV
  (pequena melhoria nominal, nao significativa).
- Cinco seeds42--46: menor p bilateral Wilcoxon exato possivel0.0625; nao
  significancia NAO prova equivalencia. TOST com10 seeds e uma etapa nova.
- B/gradiente nao tem rede neural: e fisica com parametros/SOC/eta aprendiveis.
  Residual e UDE usam estrutura1RC inicial, nao um2RC neuralmente aumentado;
  budgets iguais30epocas nao garantem otimizacao perfeita.

Origem: [tabela](results/unified/table_main.csv),
[estatisticas](results/unified/statistics.json),
[manifesto/protocolo](results/unified/summary.json),
[protocolo executavel](protocol.py), [relatorio](results/unified/report.md).
Commits: treino `e10d552`, geracao `21a2a8c` (registrados no relatorio).
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
  Origens, protocolos e commits: mesmos do achado1. LSTM tem ate150epocas,
  demais treinaveis30; cinco seeds e mesma celula por dataset nao generalizam
  para outras arquiteturas, orcamentos ou fabricantes.

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

## 5. Transferencia entre ciclos, falha termica e recuperacao por interpolacao

**Afirmacao:** recalibracao estatica mostrou transferencia a temperatura
nominal fixa, falhou em transferencia termica A123 sem recalibracao, e
interpolacao linear de parametros em temperaturas calibradas recuperou
pelo menos61.83% do intervalo piso-teto nas quatro condicoes retidas.

| Ensaio | Numeros primarios | Origem e protocolo | Commits |
|---|---|---|---|
| LFP estatico |A FUDS81.63%,US0676.91%; B FUDS37.14%,US0673.58%; positivos|[resultados](results/lfp_static/),[protocolo](lfp_static_protocol.json),[relatorio](report_static.md)|`50e85e1`,`a63c83d`|
| Replicacao |Panasonic US0627.15%,LA9229.59%,HWFET28.60%; positivo; CALCE DST43.41% elegivel isoladamente|[resultados](results/static_replication/),[protocolo](static_replication_protocol.json),[relatorio](report_static_replication.md)|`78fefb3`,`4fa83f6`|
| Termico A123 |0/4 temperaturas passamFUDS eUS06; NEGATIVA; localV6 recupera67.04--87.61% versusV6 transferido|[resultados](results/lfp_static_temperature/cell_A_a123/),[diagnostico](a123_temperature_diagnostic.json),[relatorio](report_static_temperature.md)|`3d0d047`,`c0bab49`|
| Interpolacao |P2 fracoes10FUDS61.83%,10US0662.03%,40FUDS71.09%,40US06116.48%; POSITIVO|[resultados](results/a123_temperature_interp/),[protocolo](a123_temperature_interp_protocol.json),[relatorio](report_temperature_interp.md)|`b2d945c`,`bb98d2e`|

Ressalvas: melhor pode diferir por alvo; positividade nao valida todas asOCVs.
PanasonicV1 usaOCV grau5 do checkpoint2RC dinamico, naoOCV independente;
dt1s versus0.1s unificado; RC exatoZOH versusRK4; fontes completas versus
janelas guardadas. Mendeley conserva ambiguidadeT/identidade e z0 assumido.
CALCE V6/V7 z0[.9,1] apesar fonte~.823. V7 fortemente confundido. Termico
usaOCV25 fixa eQ25, sem normalizacao alvo; localV6 e diagnostico fora do criterio.
P2>100% e valido porque teto local nao e limite matematico deMAE.
P4 cobertura10FUDS58.87%,10US0667.07%,40FUDS90.61%,40US0689.94%:
incerteza NAO calibrada.200 amostras condicionadas a bounds; sem ruidoV,
incertezaNLS ou da mediaP3; calibracao soDST0/25/50, nunca10/40.

## Numeros que NAO devem ser resultados principais

1. **CALCE US06/BJDST25 com fallback z0=1**, ambas preparadas nominal80SOC:
   - Unificado janelas:2RC123.575/123.366mV,LSTM120.748/120.469mV;
     continuo2RC121.807/122.683mV,UDE121.965/122.827mV.
   - Replicacao:V1=130.408/130.514mV,melhorV7=15.300/12.747mV,
     reducoes88.27/90.23%. VereditoCALCE original POSITIVA permanece
     historico, mas esses ganhos nao sustentam uma afirmacao principal
     sem auditoria/correcaoz0 da etapa3. ParametrosV6/V7 podem compensar
     fallback; nao atribuir esses ganhos aOCV isoladamente.
2. **Qualquer V5 com extrapolacao nao fisica**:
   CALCE US0625=826.648mV,BJDST25=817.954mV; maximaV20.117/20.109V.
   Excluir como evidencia principal de desempenho/ganho; conservar como
   falha metodologica documentada. Nao excluir silenciosamente de
   tabelas historicas nem recortar erros para melhorar metricas.
3. Comparacoes ingenuas de oraculos ajustados no alvo com modeloszero-shot;
   ausencia dep<.05 com5seeds como prova deequivalencia; coberturaGP nominal
   como cobertura empirica; resultadoSOC fallback como propriedade quimica.
