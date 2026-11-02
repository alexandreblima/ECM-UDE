# Reviewer round: ablacao, equivalencia direta, treino e tabelas

Branch paper-v2. Protocolo [reviewer_round_protocol.json](reviewer_round_protocol.json)
registrado no commit **d14935e**, antes de qualquer ajuste. Somente NLS na Parte A;
B/C/D usam resultados salvos, sem treinar redes, trocar hiperparametros ou alterar criterios.

## A. Ablacao de parametros

V1 e V6 conjuntos existentes foram reutilizados. V6-Q ajusta Q/z0 com RC de V1 fixos;
V6-RC ajusta somente RC com Q medido e z0 causal por perfil. OCV fixa e simulacao
continua a 1 s, RC inicial zero, estados nao reinicializados por janela. Todos os ajustes usam
somente a fonte completa; tres inicializacoes registradas, TRF, 500 avaliacoes, tolerancias 1e-8.
V6-Q local mantem RC de DST25; V6 local foi reutilizado, nao reajustado.

### A1. Auditoria CALCE e V6-wide

| Perfil25C | z0 auditado | Metodo |
|---|---:|---|
| FUDS_25C | 0.799972169 | same_file_charge_count |
| DST_25C | 0.799973117 | same_file_charge_count |
| US06_25C | 0.799969414 | same_file_charge_count |
| BJDST_25C | 0.799944042 | same_file_charge_count |

A auditoria da fonte FUDS encontrou 0.799972, nao o antigo 0.822910 por inversao OCV.
V1 nao foi reajustado: os parametros antigos sao reavaliados com z0auditado em todos os perfis.
V6 antigo permanece como referencia solicitada, inclusive limitesz0[.9,1] e ajuste com
inicializacao historica da fonte; isso e ressalva de comparabilidade, nao refit silencioso.
V6-wide usa z0[.7,1], mesma fonte auditada e transfere Q/z0/RC sem substituir z0 por alvo.
Os alvos auditados definem baselineV1/V6-RC e mascarasSOC, nao uma recalibracao wide por alvo.

| Alvo | V1 auditado(mV) | V6-wide(mV) | Reducao | Passa>=25% |
|---|---:|---:|---:|---|
| DST_25C | 16.281 | 9.942 | 38.94% | True |
| US06_25C | 19.988 | 10.788 | 46.03% | True |
| BJDST_25C | 20.620 | 11.516 | 44.15% | True |

Qwide=1.948450Ah; z0wide=0.810323.
Veredito registrado V6-wide: **POSITIVO**.
Esse e um novo teste com limite ampliado; nao reescreve o anterior CALCE corrigido NEGATIVO.

### A2. MAE e fracoes explicadas por condicao

f=(MAE_V1-MAE_variante)/(MAE_V1-MAE_V6). Fracoes assinadas, sem truncar;
denominador zero=>indisponivel com aviso; denominador negativo nao significa ganho explicado
e nao participa da atribuicao. Maioria estrita (>metade) por bloco, apenas ganho V6>0.
Ambos mecanismos atingindo70% na maioria, ou nenhum, implica conjunto.

| Bloco | Alvo | V1 | V6-Q | V6-RC | V6 | fQ | fRC | V6-OCVT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cell_A_a123 | FUDS | 22.371 | 18.393 | 22.371 | 4.109 | 0.218 | 0.000 | -- |
| cell_A_a123 | US06 | 21.177 | 18.372 | 21.177 | 4.952 | 0.173 | 0.000 | -- |
| cell_B_mendeley | FUDS | 18.842 | 20.215 | 18.842 | 15.268 | -0.384 | 0.000 | -- |
| cell_B_mendeley | US06 | 17.917 | 18.890 | 17.917 | 12.084 | -0.167 | 0.000 | -- |
| panasonic | US06_25C | 33.187 | 32.062 | 33.187 | 32.223 | 1.167 | 0.000 | -- |
| panasonic | LA92_25C | 16.999 | 17.491 | 16.999 | 14.009 | -0.164 | 0.000 | -- |
| panasonic | HWFET_25C | 26.915 | 25.403 | 26.915 | 26.778 | 11.108 | 0.000 | -- |
| calce | DST_25C | 16.281 | 33.927 | 10.339 | 31.380 | 1.169 | -0.394 | -- |
| calce | US06_25C | 19.988 | 33.387 | 11.905 | 33.208 | 1.014 | -0.611 | -- |
| calce | BJDST_25C | 20.620 | 33.821 | 12.576 | 32.741 | 1.089 | -0.664 | -- |
| a123_transfer_0C | FUDS | 83.465 | 79.910 | 83.465 | 95.232 | -0.302 | -0.000 | -- |
| a123_transfer_0C | US06 | 76.176 | 72.754 | 76.176 | 88.308 | -0.282 | -0.000 | -- |
| a123_local_0C | FUDS | 83.465 | 54.149 | 41.153 | 19.589 | 0.459 | 0.662 | 16.034 |
| a123_local_0C | US06 | 76.176 | 48.711 | 31.581 | 12.743 | 0.433 | 0.703 | 12.274 |
| a123_transfer_10C | FUDS | 24.351 | 21.234 | 24.351 | 35.158 | -0.288 | -0.000 | -- |
| a123_transfer_10C | US06 | 38.224 | 32.280 | 38.224 | 47.559 | -0.637 | -0.000 | -- |
| a123_local_10C | FUDS | 24.351 | 16.589 | 21.858 | 6.135 | 0.426 | 0.137 | -- |
| a123_local_10C | US06 | 38.224 | 18.005 | 32.403 | 7.300 | 0.654 | 0.188 | -- |
| a123_transfer_40C | FUDS | 27.776 | 54.138 | 27.776 | 36.891 | 2.892 | -0.000 | -- |
| a123_transfer_40C | US06 | 29.613 | 46.551 | 29.613 | 29.063 | -30.788 | 0.000 | -- |
| a123_local_40C | FUDS | 27.776 | 28.472 | 7.291 | 4.571 | -0.030 | 0.883 | 4.014 |
| a123_local_40C | US06 | 29.613 | 30.467 | 11.208 | 9.578 | -0.043 | 0.919 | 8.687 |
| a123_transfer_50C | FUDS | 31.047 | 57.754 | 31.047 | 40.510 | 2.822 | -0.000 | -- |
| a123_transfer_50C | US06 | 32.356 | 60.068 | 32.356 | 42.583 | 2.710 | -0.000 | -- |
| a123_local_50C | FUDS | 31.047 | 31.335 | 7.493 | 5.583 | -0.011 | 0.925 | 4.020 |
| a123_local_50C | US06 | 32.356 | 32.218 | 6.608 | 5.929 | 0.005 | 0.974 | 4.378 |

MAE em mV. transfer: parametrosDST25; local: ajusteDST da propriaT.
V1termico sempre transferidoDST25. Valoresnull nao sao zero nem resultados favoraveis.

| Bloco | Condicoes ganhoV6>0 / total | Q>=70% | RC>=70% | Veredito descritivo |
|---|---:|---:|---:|---|
| matched_temperature | 7/10 | 2 | 0 | **conjunto** |
| thermal_transfer | 1/8 | 0 | 0 | **conjunto** |
| thermal_local | 8/8 | 0 | 5 | **parametros RC** |

Mesma temperatura por celula (descritivo adicional):
- cell_A_a123: conjunto; condicoes elegiveis 2.
- cell_B_mendeley: conjunto; condicoes elegiveis 2.
- panasonic: escala de SOC; condicoes elegiveis 3.
- calce: INCONCLUSIVO; condicoes elegiveis 0.

### A3. OCV da propria temperatura e ressalvas

- **INCONCLUSIVO_NLS: a123_local_10C/V6-OCVT**. Nenhuma inicializacao convergiu em500avaliacoes; parametros falhos nao foram pontuados nos alvos.
- V6-OCVT conserva Q25como referencia dinamica +/-10%; ramosOCVlocais seguem a escala
  de capacidade de baixa corrente local do loader existente. Substitui so a curva, nao Qreferencia.
- OCVT local a0/40/50C e comparacao diagnostica fonteDST; a10C inconclusiva.
- V6-RC equivale matematicamente a V1 em Q/z0 medidos; reotimizacao pode achar
  outro minimo e CALCE muda a inicializacao auditada. Nao e evidencia causal independente.
- Mendeley: preparacao ausente, temperatura nominal ambigua 25/30 C e capacidade ~2.50 Ah;
  Panasonic usa curva OCV do checkpoint. Seu criterio SOC e descritivo e sensivel a ganhos
  pequenos: HWFET melhora apenas ~0.137 mV com V6 (fracao Q=11.11 nao significa ganho grande).
- Q/z0/RC continuam confundidos; a atribuicao>=70% e criterio descritivo, nao prova fisica.
- O bloco local completo e INCONCLUSIVO quanto ao conjunto de variantes pedidas, pois falta OCVT10.
  A contagem descritiva RC 5/8 usa apenas as variantes Q/RC/V6 convergentes nas 8 condicoes;
  nao constitui declaracao de bloco completo. Execucao complete=true nao significa todos os fits convergentes;
  all_requested_fits_converged=false registra a ressalva em JSON.
- [Manifestos e tentativas](results/param_ablation/summary.json),
  [tabelaCSV](results/param_ablation/table.csv), [metricas porSOC/inversoes](results/param_ablation/).

## B. Comparacao direta das variantes

| Diferenca | Equivalentes /18 |
|---|---:|
| ecm_ude - physics_gradient | 18/18 |
| output_residual - physics_gradient | 18/18 |

Mesmas seeds 42--51 em cada par; MAE continuo; margem +/-1 mV.
TOST inferior usa d+1, alternativa greater; superior d-1, alternativa less.
Holm 18 por par/direcao; ambos pHolm<.05. Exclui os dois fallbacks CALCE25,
sem substituir resultadoscorrigidos numa familia originalmente diferente.

### Estatistica, estimando e p exato

- Estatistica W+=soma dos ranks absolutos dos valores deslocados com sinal positivo.
- Wilcoxon estima localizacao/pseudomediana sob simetria; nao testa diretamente uma
  mediana amostral arbitraria. Reportamos mediana das diferencas E pseudomediana
  Hodges-Lehmann=mediana das55medias Walsh (diagonal incluida), nao as confundimos.
- Antes do teste: arredondar diferencas deslocadas a10decimais; remover zeros.
  Empates absolutos recebem rankmedio; todoszeros=>p=1.
- pexato por enumeracao de2^n sinais dos ranks, cauda inclusiva; sem aproximacao normal
  nem atalho scipy.wilcoxon com tratamento inadequado de empates.
- Implementacao Python/NumPy2.2.6; SciPy1.15.3 apenas rankdata; Holm implementado no repositorio.
- [Codigo efetivo](exp_paper_v2_seeds.py), [manifesto B](results/variant_tost/summary.json),
  [tabela B](results/variant_tost/comparison.csv).
- LSTM x2RC: unidade bootstrap e a diferenca de MAE porseed, nao amostra temporal.
  Reamostra10diferencas com reposicao,20000vezes; mediana porreamostragem;
  intervalo percentil2.5/97.5 com np.quantile linear, RNGdefault_rng(20261101).
  2RCdeterministico nao se torna10ajustes independentes. Apenas janelas.

## C. Auditoria dos treinos salvos e do codigo

[CSV com 80 linhas](results/training_audit/epochs.csv): dataset/modelo/seed, epoca selecionada,
epocas executadas, ultimas perdas treino/validacao e perdas na epoca selecionada.
Ultima perda nao e a perda dos pesos restaurados; treino da epoca e media ao longo dos
minibatches com pesos evoluindo. O checkpoint selecionado usa menor validacao estrita.

| Dataset | Modelo | Epoca selecionada mediana[min,max] | Executadas mediana[min,max] |
|---|---|---:|---:|
| panasonic | physics_gradient | 5.5[1,12] | 13.5[9,20] |
| panasonic | output_residual | 5.5[1,12] | 13.5[9,20] |
| panasonic | ecm_ude | 1[1,1] | 9[9,9] |
| panasonic | lstm | 107[79,133] | 137[109,150] |
| calce | physics_gradient | 2.5[1,15] | 10.5[9,23] |
| calce | output_residual | 2.5[1,15] | 10.5[9,23] |
| calce | ecm_ude | 1[1,6] | 9[9,14] |
| calce | lstm | 101[17,128] | 131[47,150] |

Oito PDFs vetoriais de 3.5 in, Times>=8 pt, sem titulos internos; mediana e faixa min/max.
Somente seeds observadas em cada epoca; NAO preenchemos caudas apos early stopping.
Contagem por epoca no manifesto evita comparar dez seeds iniciais com sobreviventes sem aviso.

- panasonic/physics_gradient: [panasonic_physics_gradient_learning.pdf](results/training_audit/panasonic_physics_gradient_learning.pdf).
- panasonic/output_residual: [panasonic_output_residual_learning.pdf](results/training_audit/panasonic_output_residual_learning.pdf).
- panasonic/ecm_ude: [panasonic_ecm_ude_learning.pdf](results/training_audit/panasonic_ecm_ude_learning.pdf).
- panasonic/lstm: [panasonic_lstm_learning.pdf](results/training_audit/panasonic_lstm_learning.pdf).
- calce/physics_gradient: [calce_physics_gradient_learning.pdf](results/training_audit/calce_physics_gradient_learning.pdf).
- calce/output_residual: [calce_output_residual_learning.pdf](results/training_audit/calce_output_residual_learning.pdf).
- calce/ecm_ude: [calce_ecm_ude_learning.pdf](results/training_audit/calce_ecm_ude_learning.pdf).
- calce/lstm: [calce_lstm_learning.pdf](results/training_audit/calce_lstm_learning.pdf).

### C1. Intervalo NLS do unificado (nao confundir com Parte A)

O ECM-1RC E o ECM-2RC usam **somente janelas de treino guardadas**, nao perfil inteiro:
[fit_nls](exp_unified.py#L180) seleciona ctx.tr; estadosRC reiniciados em cada janela.
Janelas sobrepostas repetem amostras na funcao objetivo. A normalizacao tambem usa so
a uniao das amostras de treino, [dataset_context](exp_unified.py#L148).

- panasonic: 350 janelas treino; samples[0,179712) (t=0--17971.1s); validacao comeca sample179712 (t=17971.2s), 87janelas; perfil224479amostras.
- calce: 69 janelas treino; samples[0,8960) (t=0--8959.0s); validacao comeca sample8960 (t=8960.0s), 17janelas; perfil11201amostras.

Parte A e estudos estaticos usam DST/UDDS/FUDS completos, nao essa particao.
A fonte retida seleciona early stopping: validacao, nao teste final independente.

### C2. Normalizacao, perda e otimizacao

- Entrada X=(I,T,SOC): I centrada/escalada por media/desvio de treino; Tmean=0,Tstd=25;
  SOCmean=.5,SOCstd=.3. Saida V por Vmean/Vstd so de treino. Estatisticas fixas nos alvos.
  [compute_norm_stats](data.py#L308), [renormalize](data.py#L443), [avaliacao](exp_unified.py#L233).
- Loss dos quatro treinaveis: MSE de V normalizada, media minibatch/amostras; nenhum compute_loss
  especial no UnifiedECMUDE. Residual_loss nas historias=0. Sem penalizacao fisica adicional oculta.
  [loss](training.py#L223), [MSE](training.py#L355), [media](training.py#L255).
- Adam com weightdecay1e-5; gradiente/residual/UDE lr2e-4,30epocas,clip.5,patience8;
  LSTM lr1e-3,150epocas,clip1,patience30. Batch32,CPU; configsalva e hashes auditados.
- LambdaLR: warmup3fisicos/5LSTM, depois coseno ate fator.01; step apos epoca;
  loglr e o usado antes do scheduler. [config](protocol.py#L20), [Adam](training.py#L340),
  [scheduler](training.py#L184), [earlystopping](training.py#L418).

### C3. Integracao, estados e limites NLS

- Unificado: RK4classico ZOH, dt=.1sPanasonic/1sCALCE; estados polarizacao0 e SOCcausal
  no inicio da janela; continuo inicia uma vez e carrega ate fim. Fontemetricas mascaradas
  apenasvalidacao. [integrate](unified_dynamics.py#L111), [nls](exp_unified.py#L168).
- Gradiente/residual/UDE inicializados do1RC, NAO2RC; C>=1F,R>=1e-4 por softplus;
  eta=1+.1tanh; sem tau3600 no treino. OCV grau5 aprendivel, avaliacaoSOCclip[0,1].
  [physical](unified_dynamics.py#L79), [construcao](unified_dynamics.py#L185).
- NLSunificado: graus5 Chebyshev coeff[-10,10], R0/Ri[.001,.5]ohm,
  tau[dt,10000]s. Uma inicializacao: OCVleast-squares, R0=.03,Ri=.02,tau1=5,tau2=40.
  Penalizacao coeff:1e-3*[0,2,4,8,16,32]*c anexada aos residuos; TRFlinear,xscalejac,
  max500avaliacoes,ftol/xtol/gtol1e-9; result.success exigido. [codigo](exp_unified.py#L180).
- Parte A: RCexatoZOH a1s, tau<=3600,R[1e-5,.5],Q+/-10%,
  z0[.9,1] ouwide[.7,1]; tres starts (tau2/40,10/300,1/1000; z0=1,.95,.9).
  TRFlinear, xscalejac, max500,tolerancias1e-8; menorSSE entreconvergentes; nenhum
  alvo escolheminimo. [bounds](exp_lfp_static.py#L101), [novo runner](exp_param_ablation.py).
- LSTM: sem integracao ODE; estados oculto/celula zerados no inicio de cada janela;
  entradas/saida usam a mesma normalizacao do benchmark. Nao ha inferencia continua LSTM reportada.
- [Manifesto codigo/historias](results/training_audit/summary.json) registra hashes, config-base
  (cinco seeds originais) e lista efetiva de dez seeds 42--51 separadamente.

## D. Figura e tabelas IEEE

- [Figura 2](figures_ieee/fig02_physics_vs_lstm.pdf): LSTM dez seeds, mediana[min,max], nao media/SD;
  2RC deterministico, janelas, 18 principais. So este PDF regenerado; outros sete intactos.
- [Indice](figures_ieee/figures_index.md) e manifesto atualizados; versoes anteriores arquivadas.
- TabelaE1 [CSV](results/paper_tables/E1.csv) / [booktabs table*](results/paper_tables/E1.tex):
  18condicoes, MAEcontinuo2RC e mediana[min,max] dosfisicos, TOSTcontra2RC egradient.
  Contra2RC conservaHolm20; variantesHolm18. Faixas nao saoICs.
- Condicoes [CSV](results/paper_tables/conditions.csv) / [TeX](results/paper_tables/conditions.tex):
  20doE1 e fontes/alvosE2--E4, origemz0,papeis,inclusao; sem misturar fallback e auditado.
- Criterios [CSV](results/paper_tables/criteria.csv) / [TeX](results/paper_tables/criteria.tex):
  E1unificado,E2histerese,E3estatico,E4termico/interpolacao; registros originais ewide separados.
  Exige pacote booktabs; table* ocupa largura de duas colunas.

## Preservacao e validacao

- 1131 versoesanteriores verificadas porSHA256: 1125 no lugar, 6 copiasexatas arquivadas.
- [Auditoria](results/reviewer_round_audit/preservation.json); [testes](results/reviewer_round_audit/tests.log).
- Nenhum resultadoanterior reescrito. Manuscrito e checkpoints intactos.
- Falha original de exportacao CSV (camposwide heterogeneos) foi corrigida sem refit:
  reexportacao e reavaliacao de fitssalvos. Log preservado; RC-only usa z0auditado de cada alvo.
- Nao se modifica nenhum criterio apos ver resultados; OCVT10 nao recebe orcamentoextra.
