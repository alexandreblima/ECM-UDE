# A123 CALCE: inventário e interrupção por ausência de dados

Data: 2026-11-01 (America/Sao_Paulo).
Branch: `lfp-histerese`, criada de `protocolo-unificado` no commit
`21a2a8cc075b64a56c1ca936bc7b91895b3efc3c`.

## Estado

**ATUALIZAÇÃO 2026-11-01: teste NLS das duas células concluído.**

- Célula A, A123 A1-007: **NO-GO**. Em FUDS, redução de MAE de
  **−1,62%** e saturação **36,40%**; em US06, redução **−6,14%**
  e saturação **55,15%**. A falha é no ganho mínimo, não na saturação.
- Célula B, Mendeley: **GO**, condicionado às hipóteses registradas.
  FUDS: redução **13,19%**, saturação **21,09%**; US06:
  redução **23,97%**, saturação **24,48%**.
- **Decisão global: NO-GO**, exclusivamente pela Célula A.
  B não concorda; sua réplica não altera a regra global.
- max|M| no platô SOC20–80: **36,174 mV (A)** e **18,600 mV (B)**.
- O critério não foi alterado. V4 é oráculo estático ajustado no alvo,
  não baseline elegível para a decisão.
- Relatório completo: [report_gonogo.md](./report_gonogo.md).
  Resultados: [A123](./results/lfp_gonogo/cell_A_a123/summary.json) e
  [Mendeley](./results/lfp_gonogo/cell_B_mendeley/summary.json), com
  tabelas, parâmetros/limites, estratos, trajetórias, gráficos e SHA256.
- **Fase 2 (ESC-UDE) não iniciada.** Nenhuma rede neural foi importada
  ou treinada. `MANUSCRITO/` não foi alterado.

O registro de ausência abaixo descreve a busca anterior, não o estado atual.
Os dois ZIPs estão agora em
`/Users/apple/Dropbox/Documentos/Academia/PESQUISA/baterias/SciML/pipelines/datasets/calce_a123_raw/`.
O protocolo NLS pré-registrado está em [lfp_gonogo_protocol.json](./lfp_gonogo_protocol.json);
o loader novo está em [data_lfp.py](./data_lfp.py). A1-007 foi selecionada
lexicograficamente antes dos ajustes; A1-008 é preservada, sem pooling.
Mendeley usa apenas a cópia standalone autorizada, Q de descarga 0,5C
2,498901583 Ah, e registra a divergência do rótulo 1,1 Ah.
Os perfis processados B não trazem preparação: o usuário autorizou
explicitamente z0=1, h0=+1 e polarizações iniciais zero como hipótese.
25 °C é nominal; a ambiguidade 25/30 °C continua documentada.

Logs e manifestos de leitura/OCV/NLS ficam em `results/lfp_gonogo/`,
com subpastas por célula. Instalar
[requirements-lfp.txt](./requirements-lfp.txt) quando necessário;
nenhuma dependência nova foi instalada nesta execução. Os 14 testes
focados passaram, incluindo recorrência escalar independente dos resultados
salvos e verificação dos limiares exatos >=10% e <90%.

### Registro histórico da primeira tentativa, antes da chegada dos ZIPs

A execução foi interrompida na preparação, conforme solicitado. Não houve
download de dados, identificação NLS, treinamento de rede, criação de loader
ou geração de métricas/figuras. Os resultados e o manuscrito existentes não
foram alterados.

## Busca local

- Busca recursiva de arquivos em `../datasets/` por A123, LFP, OCV e nomes dos
  perfis solicitados.
- Leitura dos índices de todos os 36 arquivos ZIP locais; nenhum estava
  ilegível. Nenhum membro correspondente aos dois ensaios A123 foi encontrado.
- Leitura dos índices dos dois RARs `Battery Test Data.rar`, com 47 entradas
  cada. São cópias do conjunto Mendeley `29kw38kzwj_v1`, com dados LFP 18650
  3,3 V/1100 mAh e outras células; não contêm os ensaios A123 CALCE solicitados.
- `../datasets/calce_20r_raw/` contém ensaios Samsung 20R, que não substituem
  os ensaios A123/LFP.

## Arquivos exatos a baixar manualmente

Fonte: [CALCE Battery Research Data](https://calce.umd.edu/data), seção
**A123 Battery**, links **Data for 25°C**:

| Ensaio | Arquivo ZIP oficial | Link |
|---|---|---|
| Low-Current OCV, 25 °C | `A123_OCV25-20120905.zip` | https://web.calce.umd.edu/batteries/data/A123_OCV25-20120905.zip |
| Dynamic Profile, 25 °C | `A123_DST-US06-FUDS-25.zip` | https://web.calce.umd.edu/batteries/data/A123_DST-US06-FUDS-25.zip |

Os nomes e URLs foram conferidos no HTML da página oficial, sem requisitar
os ZIPs. Os nomes dos arquivos internos não foram verificados, pois isso
exigiria obter os dados. Guardar ambos os ZIPs em `../datasets/`, por exemplo
em `../datasets/calce_a123_raw/`, preservando-os para rastreabilidade.

A página oficial lista capacidade nominal de **1110 mAh** para A123. Isso
não determina a capacidade real dos arquivos: Qnom deverá ser medido pela
descarga a baixa corrente, e somente então comparado a 1,1 Ah e 2,23 Ah.

## Critério registrado antes de qualquer ajuste

GO somente se, **em FUDS e em US06**, V3 reduzir o MAE em pelo menos 10%
relativamente ao melhor de V1/V2 e passar menos de 90% do tempo com
`|h| > 0,9`. Caso contrário, após um teste válido, NO-GO.

O critério foi aplicado separadamente a A e B. A decisão global é GO
se e somente se A for GO; B serve como réplica, não como substituta.

A ausência de dados não é um resultado NO-GO. Ganhos de V4 sobre V1–V3
deverão ser registrados como diagnóstico do componente estático de OCV,
sem alterar o critério de decisão.
