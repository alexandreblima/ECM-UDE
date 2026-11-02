# Paper-v2: registro, equivalencia, SOC e robustez monotona

Branch paper-v2 criada de lfp-histerese. Protocolos commitados antes de seus experimentos.
Resultados anteriores e MANUSCRITO preservados, verificados por SHA-256.

## Etapa 1: registro de afirmacoes

Commit 7b555d0.

[claims_ledger.md](claims_ledger.md) contem cinco achados, numeros, protocolos, commits e ressalvas.
As afirmacoes foram qualificadas: nao significancia nao prova equivalencia; ESC passa na replica
mas falha no A123; oraculo estatico ajusta OCV E RC no alvo, nao identifica causa unica.
CALCE US06/BJDST 25 C com fallback z0=1 e V5 nao fisico ficam fora dos resultados principais.

## Etapa 2: dez seeds e TOST

Registro d5cdc37; resultados 81ac420.

Seeds 42--46 reutilizadas, 47--51 novas; mesmos loaders/hiperparametros e pesos NLS.
MAE continuo principal; janelas secundario. TOST Wilcoxon exato deslocado +/-1 mV;
Holm 20 condicoes por modelo/modo/direcao; equivalencia requer ambos pHolm<.05.

| Modo | Modelo | Equivalentes/20 | Principais equivalentes/18 |
|---|---|---:|---:|
| continuous | physics_gradient | 17/20 | 15/18 |
| continuous | output_residual | 17/20 | 15/18 |
| continuous | ecm_ude | 16/20 | 14/18 |
| windowed | physics_gradient | 19/20 | 17/18 |
| windowed | output_residual | 19/20 | 17/18 |
| windowed | ecm_ude | 19/20 | 17/18 |

Diferencas medianas descritivas no modo continuo, excluindo os dois fallbacks:
- physics_gradient: -0.185--2.969 mV; 4/18 melhorias nominais. Nao inferir ausencia universal de melhoria a partir de equivalencia parcial.
- output_residual: -0.189--2.967 mV; 4/18 melhorias nominais. Nao inferir ausencia universal de melhoria a partir de equivalencia parcial.
- ecm_ude: -0.047--3.192 mV; 3/18 melhorias nominais. Nao inferir ausencia universal de melhoria a partir de equivalencia parcial.

LSTM versus 2RC apenas janelas: diferenca mediana e intervalo bootstrap 95% por seed, sem TOST/p.
Entre 18 condicoes principais, delta mediana varia 7.760--66.285 mV;
18/18 intervalos inteiramente positivos.
[Relatorio completo](results/unified_10seeds/report.md),
[protocolo](paper_v2_seeds_protocol.json).
Equivalencia nao e universal: falha em certas condicoes nao implica beneficio;
early stopping usa validacao fonte, seeds nao sao celulas independentes.
As 20 condicoes da familia Holm conservam os dois fallbacks, que nao sustentam achados principais.

## Etapa 3: z0 CALCE auditado

Registro 308dc2a; resultados f48deb8.

| Condicao | Regra | z0 antigo | z0 corrigido | Elegivel |
|---|---|---:|---:|---|
| US06_25C | same_file_charge_count | 1.0 | 0.7999694136592465 | True |
| BJDST_25C | same_file_charge_count | 1.0 | 0.79994404241519 | True |

| Condicao | 2RC continuo antes/depois (mV) | V1 estatico corrigido (mV) | V7 (mV) | Reducao V7/V1 |
|---|---|---:|---:|---:|
| US06_25C | 121.807/20.794 | 19.988 | 15.300 | 23.45% |
| BJDST_25C | 122.683/19.515 | 20.620 | 12.747 | 38.18% |

Nenhum retreino; checkpoints das 10 seeds e NLS reaproveitados.
V1/V5 usam z0 corrigido; V6/V7 continuam transferindo Q/z0 fonte.
Veredito estatico CALCE historico=POSITIVA; hipotetico apos correcao=NEGATIVA. Historico NAO alterado.
[Antes/depois](results/calce_z0_fix/report.md),[regra](paper_v2_calce_z0_protocol.json).
Sem ancora, excluir: nunca usar sucesso aparente de compensacao Q/z0 como propriedade OCV.
Resultados V5 extrapolativos continuam ressalva separada.

## Etapa 4: V7m monotono

Registro 8a0e2e2; politica de falha 3c149a8; resultados 0249395.

PCHIP 11 nos com incrementos positivos, suporte SOC fonte e extrapolacao constante;
RC/Q/z0 ajustados somente na fonte com limites anteriores.
Robustez exige V7m reduzir >=25% versus V1 em TODOS os alvos na mesma T.

A123 nao convergiu nas tres inicializacoes com 500 avaliacoes. Foram verificadas coordenadas
afim-equivalentes e derivadas analiticas, sem mudar limites/criterio/inicializacoes nem consultar alvos.
O usuario autorizou preservar o protocolo, registrar A123 INCONCLUSIVO e continuar os outros datasets.
[Politica de falha](paper_v2_monotone_failure_policy.json) commitada antes dos ajustes restantes.

| Dataset | Variante principal | Robusto |
|---|---|---|
| cell_A_a123 | baseline V1 original | INCONCLUSIVA (NLS) |
| cell_B_mendeley | baseline V1 original | NEGATIVA |
| panasonic | baseline V1 original | INCONCLUSIVA (NLS) |
| calce | CALCE baseline V1 auditado | INCONCLUSIVA (NLS) |

Robustez conjunta dos quatro datasets: **NAO ESTABELECIDA: IDENTIFICACAO INCONCLUSIVA**.
Datasets negativos pelo criterio: cell_B_mendeley.
Inconclusivos por nao convergencia, nao por erro de transferencia medido: cell_A_a123, panasonic, calce.
Nao altera antigos vereditos positivos; testa esta variante restrita, sem best-of V5/V6/V7.
[Relatorio](results/static_monotone/report.md),[protocolo](paper_v2_monotone_protocol.json).

| Dataset | Perfil | V1 (mV) | V7m (mV) | Reducao |
|---|---|---:|---:|---:|
| cell_B_mendeley | FUDS | 18.842 | 18.082 | 4.04% |
| cell_B_mendeley | US06 | 17.917 | 18.224 | -1.72% |

## Ressalvas gerais

- Comparacao neural inicia de 1RC, referencia 2RC; nao e ablacao neural em 2RC identico.
- Janelas versus continuo nao intercambiaveis; LSTM so janelas; dt Panasonic 0.1 s no unificado, 1 s nos estaticos.
- NLS deterministico nao replicado por seed; TOST pressupoe simetria das diferencas e estima variacao do otimizador.
- OCV baixa corrente e aproximacao; Mendeley rotulo 1.1 Ah versus Q~2.50 Ah, nominal 25 C versus 30D, preparacao assumida.
- A123 A1-007 apenas, repouso 120 s e RC inicial 0 assumido; aging entre ensaios nao controlado.
- OCV/Q/z0/RC confundidos, limites atingidos e suporte SOC parcial nao garantem identificabilidade.
- CALCE auditado por contagem de Ah no arquivo, SP1 outra celula; V6/V7 z0 restrito [.9,1] apesar fonte ~80% SOC.
- V7m monotonia e restricao de forma, nao prova unicidade fisica; nenhum hiperparametro escolhido usando alvos.
- No desenho V7m, suporte SOC fonte recomputado cancela Q/z0 exatamente nas previsoes.
  Esses parametros nao sao identificaveis: seus valores/rotulos SOC nao admitem interpretacao causal.
- Nao convergencia nao e evidencia de transferencia negativa; nenhum parametro falho foi pontuado.
- Resultados historicos de calibracao termica/GP mantidos; cobertura P4 insuficiente em 10 C nao foi reinterpretada.

## Preservacao e artefatos

- **833 arquivos anteriores** verificados por SHA-256, incluindo MANUSCRITO: todos intactos.
- [Auditoria](results/paper_v2_audit/preservation.json)
- [Hashes anteriores](results/paper_v2_audit/previous_sha256.json)
- [Testes](results/paper_v2_audit/tests.log)
- Cada experimento possui tabelas CSV/TEX, manifesto/protocolo, logs e relatorio; nenhuma fase extra autorizada.
