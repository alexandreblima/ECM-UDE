# Inventário local dos datasets de baterias

Data da inspeção: **2026-11-01**. Raiz: `../datasets/`, isto é,
`/Users/apple/Dropbox/Documentos/Academia/PESQUISA/baterias/SciML/pipelines/datasets/`.

## Escopo e interpretação

- Inspeção somente leitura dos datasets: enumeração recursiva, índices de
  arquivos compactados, documentação/código distribuídos e leitura de
  cabeçalhos e amostras brutas representativas. Não houve download,
  instalação, extração para disco, identificação, modelagem ou treinamento.
  A única entrega escrita no repositório é este relatório.
- A enumeração cobre as subpastas; a caracterização numérica é amostral,
  exceto quando explicitamente indicado. Contagens físicas incluem
  `.DS_Store` quando presente; membros de arquivos compactados são
  contabilizados separadamente, não como arquivos físicos adicionais.
- **ND** = não determinado nos materiais locais inspecionados. Ausência de
  documentação ou de um ensaio identificável não prova que o estudo original
  nunca o realizou. Nomes de arquivos evidenciam organização/intenção, não
  garantem qualidade, completude ou execução integral do ensaio.
- Química indicada pela documentação local é distinguida de inferência.
  A faixa de tensão, sozinha, não distingue NMC de NCA. Documentação do
  repositório não equivale a uma ficha química do fabricante.
- Capacidade nominal, capacidade de descarga registrada e variação de Ah em
  um perfil dinâmico são grandezas distintas. Não se usa capacidade do ciclo
  alvo para definir SOC no protocolo unificado.
- OCV a baixa corrente é uma **aproximação** de OCV. Pulsos com repouso podem
  fornecer OCV incremental, mas não são automaticamente um protocolo GITT
  formal ou um ensaio de laços menores de histerese.
- Cruzamentos de sinal de corrente, após remover `|I| <= 0,02 A`, indicam
  alternância de carga/descarga; não certificam laços menores quase estáticos
  nem a amplitude de SOC necessária para identificar histerese.
- Links externos abaixo são referências identificadas em materiais locais;
  não foram acessados para completar lacunas documentais.

## Tabela de inventário

| Pasta | Origem identificável | Célula, química e capacidade | Ensaios, temperaturas e envelhecimento | Aquisição, canais, formato e sinal | Loader neste repositório |
|---|---|---|---|---|---|
| `calb_l148n58a_raw` | Dois relatórios Arbin de maio/2024; `Global_Info` identifica `Cell_characterization\CalbL148N58A_DrivingCycles+CalbL148N58A.sdx` e procedimento EIS. README, artigo e DOI ND. | **CALB L148N58A** identificado pelo procedimento; formato ND. **Química não confirmada**: carga até 4,2 V torna LFP convencional improvável, mas não distingue NMC/NCA. `Capacity(Ah) = 58` é configuração, não medição. Nominal independente ND; descarga completa medida ND. Três descargas parciais EIS entregam aproximadamente **11,600 Ah cada**; contadores globais de 76,78291/34,80059 Ah não são capacidade da célula. | Dinâmicos sem nomes padronizados: duas famílias de etapas, uma com 12 repetições de aproximadamente 1.800 s; **2.267 inversões** com limiar 0,05 A. EIS em três pontos de SOC, repousos de 5.400 s. Sem par de baixa corrente carga/descarga, HPPC, GITT nomeado ou laços menores controlados identificado. **Somente T25** local; envelhecimento ND. | **2 XLSX**; tempo, I, V, Ah de carga/descarga, energias, ciclo/etapa, **duas T auxiliares medidas**. DrivingCycle: 375.349 amostras; mediana Δt **0,1001 s**, P90 aproximadamente 0,9995 s; T aproximadamente 24,77–26,41 °C. EIS: 34.049 amostras, mediana 1 s, lacunas até 9.100 s; folha ACIM com 120 pontos. I aproximadamente −60,08 a +36,17 A; **positivo carga, negativo descarga**. | **Não**. |
| `calce_20r_raw` | CALCE/University of Maryland, [Battery Research Data](https://calce.umd.edu/data); relatórios Arbin de 2015–2016 e identificação em [data_calce.py](./data_calce.py). DOI específico ND. | Samsung INR18650-20R, cilíndrica 18650; **NMC conforme documentação do loader**, não inferida apenas dos 2,5–4,2 V. Nominal **2,0 Ah**. Descargas iniciais a aproximadamente 1 A: **SP1 2,00944 Ah; SP2 2,02834 Ah; SP3 1,97751 Ah**, contador do ciclador no final do step 6. | SP1/SP3: baixa corrente e OCV incremental em **0, 25 e 45 °C**; SP2: **DST, FUDS, US06, BJDST**, cada um nas três temperaturas e com início 50/80% SOC. Carga e descarga de baixa corrente verificadas em SP1/25 °C; OCV incremental com pulsos de aproximadamente ±1 A e repousos verificado em SP1/25 °C. HPPC nominal e laços menores dedicados não identificados. Regeneração nos dinâmicos confirmada. Capacidade inicial de três células; campanha de envelhecimento longitudinal não identificada. | `.zip`, `.xls` e `.xlsx`; os `.xls` lidos são **OOXML/ZIP apesar da extensão**. Dinâmicos: tempo, I, V, Ah de carga e descarga, energias, índices; **sem T medida**, temperatura de câmara no nome/metadado. OCV SP1 tem coluna Temperature esparsa e I/V em mA/mV. Dinâmicos: mediana bruta do step 7 aproximadamente **1,003–1,016 s**, com registros mais curtos nas transições; loader reamostra a **1 s**. Baixa corrente SP1/25 °C: mediana **1 s**. **I > 0 carga, I < 0 descarga**, confirmado pela evolução dos contadores. | **Sim**: [data_calce.py](./data_calce.py) e leitura canônica em [exp_unified.py](./exp_unified.py). A suíte usa 12 arquivos SP2/80SOC e calibração independente SP1/25 °C; não usa automaticamente os arquivos 50SOC nem toda a coleção de OCV. |
| `ha_pozzato_onori_official_code` | Código de S. Y. Sara Ha, Stanford Energy Control Group, setembro/2023; referencia Pozzato, Allam e Onori, *Lithium-ion battery aging dataset based on electric vehicle real-driving profiles*, **[10.1016/j.dib.2022.107995](https://doi.org/10.1016/j.dib.2022.107995)**. DOI das ferramentas ND. | Fabricante/modelo/formato/química não explicitados localmente; **não confirmado LFP**. Trajetórias 2,5–4,2 V sugerem outra família, sem distinguir NMC/NCA. Nominal documental ND; aproximadamente 4,85 Ah inferível da relação 0,2425 A/C20, não ficha nominal. Descargas C20 de 101 diagnósticos: **4,45055–4,90307 Ah**. | Baixa corrente **carga e descarga C20/C40 a 23 °C**; descargas C20 a **10, 23, 40 °C**. **HPPC e EIS**; grid de dez células e até 15 diagnósticos, até 369 ciclos documentados. Sem GITT nomeado, perfis veiculares brutos ou laços menores controlados locais; HPPC tem pulsos de recuperação de carga. O título do artigo não significa que seus perfis de condução estejam nesta pasta. | **MAT e código MATLAB**; `time`, `curr`, `vcell`; **sem T/Ah adquiridos** nos registros temporais, T é condição do ensaio e Ah integrável. Predominantemente **1 s**, jitter milissegundo nas curvas complementares. **I negativo carga, positivo descarga**, documentado no código, oposto a Panasonic/CALCE. EIS: matrizes frequência/impedância. | **Não integrado**; existem ferramentas MATLAB distribuídas, não loader da suíte. |
| `lg_hg2_v2` | README e PDF técnico em `extracted/`, McMaster University/Phillip Kollmeyer; DOI não localizado na documentação local. | **LG HG2 18650**, cilíndrica; **H-NMC/graphite + SiO confirmado no PDF**, nominal **3,0 Ah**. Descargas `Cap_1C` a corte próximo de 2,8 V: **2,72639 e 2,69280 Ah a 25 °C**, 2,47337/2,54654/2,26338/1,67134 Ah em 0/10/−10/−20 °C, nas séries especificadas abaixo. **Quatro CSV em 10degC são rotulados LG MG1/2,9 Ah**, apesar de MAT com nome HG2: não misturar identidades. | **C20 carga e descarga**, HPPC bidirecional, descargas 0,5/1/2C; **UDDS, HWFET, LA92, US06, Mixed1–8** em **−20, −10, 0, 10, 25, 40 °C**. Mixed3 ausente em 0/10 °C. Sem GITT ou laços menores dedicados documentados; regeneração reduzida no frio. Célula inicialmente nova; aging controlado/SOH longitudinal ND. Há erros de temperatura de pasta em alguns ensaios de capacidade. | **ZIP, 208 CSV, 206 MAT**, TXT, PDF e MATLAB. I, V, **T superficial medida**, Ah, tempo; MAT `meas` compatível em campos com leitor Kollmeyer. Dinâmicos nominalmente **10 Hz**; outras etapas variáveis: C20/25 °C mediana 60 s, capacidade 1C mediana 10 s. Auditoria dos CSV: 4.954.917 registros, 5.756 Δt zero e quatro negativos. **I positivo carga, negativo descarga; Ah carga-positivo**. CSV/MAT não são equivalentes em todos os canais: Wh de um MAT está incorreto. | **Sem loader HG2 dedicado/configuração unificada**. [data.py](./data.py) pode ler os campos dos MATs, mas Qnom, nomes, seleção, anomalias e configuração precisam de tratamento; não executado aqui. |
| `mendeley_29kw38kzwj_v1` | Nome da pasta sugere **10.17632/29kw38kzwj.1**, **não confirmado por README/artigo local**. Pacote ZIP → RAR → `Battery Test Data`; 32 planilhas. | **Multiquímica**: LFP 18650/3,3 V/**1,1 Ah pelo rótulo**, fabricante/modelo ND; NCA Panasonic NCR18650B/18650/**3,4 Ah pelo nome do pacote**; NMC Murata VTC6/18650/**3,0 Ah pelo rótulo**. Químicas documentadas nos nomes, não datasheet. **Conflito LFP**: descargas medidas **2,33872–2,58501 Ah**, correntes “1C” aproximadamente 2,4–2,55 A. NCA/25 °C **2,76480–2,80504 Ah**; NMC/25 °C **2,29781–2,30938 Ah**. Não usar 1,1 Ah automaticamente. | **PCPD/OCV incremental com repousos e ramos de carga/descarga LFP**, não C20 contínuo; **DST, FUDS, UDDS, HPPC, WLTS, US06**, descargas 0,5/1/1,5C. LFP em pastas **5, 25, 45 °C**, mas PCPD “25” traz origem interna **30 °C**. NCA/NMC em **−5, 5, 25, 45 °C**, com arquivos trocados/rotulados incorretamente. Sem aging longitudinal ou laços menores controlados comprovados. Excursões parciais/regeneração frequentes confirmadas. | **ZIP, RAR, XLS/XLSX**; tempo, I, V, capacidade/energia; detalhes mA/mAh e V; **sem T medida nas folhas examinadas**. Processados predominantemente **1 s**; detalhes PCPD com timestamps duplicados/até 2 s. **I positivo carga, negativo descarga**. Ah pode resetar entre etapas; SOC/temperatura/rótulos precisam auditoria. | **Não**. |
| `molicel_p42a_raw` | Relatórios Arbin `A9` de 2024/2025; sem README, artigo ou DOI local. | **Molicel P42A/21700 inferido do nome da pasta**, não confirmado por metadado de célula. Agendamento `NMC_TRI\TRI_high_Crate_Pulse_w_GITT.sdu` sugere **NMC**, não prova química. Nominal 4,2 Ah não confirmado localmente; campo configurado `Capacity = 4`. Descarga step 5: **3,791256 Ah** entregues a aproximadamente −4,099 A, até 2,49997 V. | **Pulsos/GITT no agendamento**, descargas incrementais aproximadamente −0,8 A/1.799 s e repousos 3.599 s; pulsos ±32 A. **EIS em SOC50**, 43 frequências. Sem C20 bidirecional completo, HPPC padronizado, ciclo veicular nomeado ou laços menores controlados identificado. **Somente 25 °C**; diferença de datas não certifica aging. | **2 XLSX**; tempo, I, V, Ah carga/descarga, energia, **três T auxiliares medidas**, posição dos sensores ND. HCGT: 168.536 amostras; Δt min/mediana/max **0,0011/1/60,04134 s**, trechos próximos de 0,1 s; T auxiliares aproximadamente 24,72–29,67 °C. EIS tem 67 registros lentos I=0 mais 43 pontos ACIM, não waveform AC temporal. **I positivo carga, negativo descarga**. | **Não**. |
| `oxford_hallemans_raw` | README local referencia Hallemans, Seel-Mayer, Keil, Duncan e Howey, *Multisine electrochemical impedance spectroscopy for Li-ion battery characterisation*. DOI ND no README. | **Samsung SDI INR18650-15M**, cilíndrica 18650, **1,5 Ah nominal**; química documental **Ni-based/graphite**, não NMC confirmado. Capacidade medida ND. | **Somente EIS multisine/singlesines em SOC50/25 °C** local. README descreve C/5 incremental e relaxações de 4 h, mas esses dados **não estão presentes**. Sem curvas OCV completas, HPPC, laços menores controlados, perfis veiculares, aging ou variação térmica. Senoides não substituem driving cycles. | **MAT e README**, `i_multisine`, `v_multisine`, `fs`, séries `i/v/t_singlesine`; **sem T medida/Ah**. Multisine: 3.125.000 amostras, `fs=15625 Hz`, Δt **64 µs derivado de fs**, duração 200 s; 48 singlesines, frequências 0,02–1.000 Hz, Δt variável. Corrente AC aproximadamente ±0,303 A centrada em zero; **convenção física de sinal ND**. T25 é setpoint documental. | **Não**. |
| `Pan18650` | Kollmeyer, *Panasonic 18650PF Li-ion Battery Data*, Mendeley V1, DOI **[10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1)**, registrado em [paper1_refs.bib](./MANUSCRITO/paper1_refs.bib). Contém também uma cópia **externa multiquímica** do Mendeley `29kw38kzwj_v1`; a pasta inteira não é exclusivamente Panasonic. | Núcleo: Panasonic NCR18650PF, cilíndrica 18650; **NCA declarada no manuscrito local**, sem ficha de fabricante nesta pasta. Nominal **2,9 Ah** no protocolo. Capacidade medida por ensaio padronizado dedicado ND. Descarga líquida observada: **2,70149 Ah em UDDS/25 °C**, não substitui Qnom. A cópia externa contém células LFP, NMC e NCA descritas separadamente abaixo. | Dez MATs: UDDS em **−20, −10, 0, 10, 25 °C**; US06 em **−10 e 25 °C**; LA92 e HWFTa/b em **25 °C**. HWFT é o nome nos arquivos; HWFET é o rótulo canônico. Núcleo sem OCV independente, HPPC, GITT ou laços menores dedicados identificados. Inversões em 25/10 °C; nos quatro arquivos a 0/−10/−20 °C examinados, corrente não positiva. Campanha de envelhecimento não identificada. A cópia externa acrescenta outros ensaios e temperaturas, mas não à célula PF. | MAT MATLAB, struct `meas`: Time/TimeStamp, **I, V, T da célula, T da câmara, Ah**, Wh, Power. Todos os dez MATs foram lidos. Mediana bruta **0,099–0,101 s**, aproximadamente 10 Hz; transições irregulares e repouso térmico inicial próximo de 60 s em alguns arquivos frios. **I > 0 carga, I < 0 descarga; Ah líquido carga-positivo**, coerente com Ah decrescente sob descarga. Temperatura medida não é constante nem igual ao setpoint. | **Sim para os MATs PF**: [data.py](./data.py) e [exp_unified.py](./exp_unified.py). O loader canônico aponta para `./datasets/`, não diretamente para esta raiz externa; as dez cópias locais correspondentes têm **SHA256 idêntico**. Não há loader da coleção multiquímica aninhada. |
| `rwth_35e_raw` | `Metadata.pdf`, DOI confirmado **[10.18154/RWTH-2021-02814](https://doi.org/10.18154/RWTH-2021-02814)**. Coleção Everlast; documento descreve 28 células, mas só seis CSV completos locais. | **Samsung INR18650-35E**, cilíndrica 18650; **NCA/C+Si confirmada no PDF**, nominal **3,4 Ah**, não 3,5 Ah por inferência do modelo. Seis descargas completas aproximadamente 0,3C: **3,38253–3,40738 Ah** integrados, até aproximadamente 2,65 V. | **qOCV carga e descarga a 0,1C**, capacidade 0,3C, pulsos em SOC 20/50/80%, perfil dinâmico **baseado em potência, nome padronizado ND**. Sem GITT ou HPPC nominal identificado; sem laços menores controlados comprovados. Aging cíclico 80% DOD/SOC10–90, checkups a cada 30 ciclos; **câmara somente 25 °C**. Períodos locais 05/09–18/10/2018, variam por célula. | **6 ZIP/CSV**, PDF, ZIP de figuras e resíduos incompletos. CSV com `Time,Voltage,Current,Temperature`, **T medida**, **sem Ah**. Total **14.383.688 registros**; Δt mínimo 0,01 s, medianas **0,15–0,34 s**, trechos de 1/10 s e lacunas próximas de nove dias. **I positivo carga, negativo descarga**. Correntes espúrias aproximadamente 4,77 A permanecem em 002/003, e dois V ausentes; requer saneamento, não feito aqui. | **Não**. |
| `stanford_m50t_raw` | Relatórios Arbin em dez pastas de células; DOI/README não encontrado nos XLSX. Código Ha da outra pasta cita **10.1016/j.dib.2022.107995** e usa os mesmos IDs, indício de relação, **não prova de equivalência integral** dos arquivos brutos. | **INR21700 M50T**, modelo/formato documentados pelos nomes e schedules; **LG inferido do modelo**, não confirmado localmente. **NMC não confirmado** por documentação dos XLSX; faixa 2,5–4,2 V não distingue NMC/NCA. Configuração **4,85 Ah**, nominal independente ND. Onze descargas C20 medidas: **4,82070–4,90303 Ah**, incluindo W5/Diag2 4,832414 Ah. | **UDDS**, envelhecimento SOC20–80 com regimes de carga CC 0,25/0,5/1/3C e etiquetas N25–N125; 11 diagnósticos `0_05C` e dez HPPC. **C20 apenas descarga**: carga preparatória aproximadamente 1,455 A, não ramo C20 de carga. HPPC oferece OCV incremental de descarga, repousos de 3.600 s e pulsos bipolares. Sem GITT dedicado, EIS, outros driving cycles ou laços menores de histerese comprovados. **Somente T23**; etiquetas N e schedules apresentam conflitos. | **161 XLSX**; tempo, I, V, Ah carga/descarga, energias e T auxiliares em parte dos arquivos. Dinâmicos aproximadamente **0,1 s**, diagnósticos aproximadamente **1 s**, transições menores de 1 ms. **32 XLSX sem T**, e T dos dois diagnósticos V5 é toda zero. Unidade térmica corrompida no cabeçalho; valores compatíveis com °C e sensor ND. **I positivo carga, negativo descarga**. Exportações fragmentadas: nem todos os grupos começam no início. | **Não**. |

## Evidências locais: CALCE e Panasonic

### CALCE

- Enumeração completa: **28 subdiretórios, 68 arquivos físicos**; 27 ZIPs
  inspecionados e 39 planilhas já extraídas. `extracted/` contém 27 grupos:
  12 combinações SP2/temperatura/perfil com dois SOC iniciais, 12 combinações
  SP1 ou SP3/temperatura/LC ou IC e três capacidades iniciais.
- [Descarga inicial SP2](../datasets/calce_20r_raw/extracted/SP2_Initial_capacity_10_16_2015/10_16_2015_Initial%20capacity_SP20-2.xls):
  step 6, corrente aproximadamente −1 A, final a 2,49997 V,
  `Discharge_Capacity(Ah)` final 2,02834. SP1 e SP3 também foram lidos;
  valores são capacidades registradas nessas descargas, não uma média de
  capacidade nominal nem uma estimativa de envelhecimento.
- [Baixa corrente SP1/25 °C](../datasets/calce_20r_raw/extracted/SP1_25C_LC_OCV_11_5_2015/11_5_2015_low%20current%20OCV%20test_SP20-1.xlsx):
  cabeçalho `Duration (sec), Pgm step, mV, mA, Temperature`. Step 3:
  77.976 registros, 21,6625 h, 4,175 → 2,496 V, integral de corrente
  **−2,15685 Ah**. Step 5: 69.688 registros, 19,3603 h,
  2,998 → 4,151 V, integral **+1,92624 Ah**. Há duas direções, mas
  amplitudes/coberturas distintas; não presumir curvas coincidentes em
  todo o SOC nem tomar a carga registrada como capacidade nominal.
- [OCV incremental SP1/25 °C](../datasets/calce_20r_raw/extracted/SP1_25C_IC_OCV_12_2_2015/12_2_2015_Incremental%20OCV%20test_SP20-1.xlsx):
  **três folhas de dados**, que precisam ser concatenadas. Pulsos de
  descarga de aproximadamente 0,2 Ah, repousos e depois pulsos de carga
  com repousos; isso confirma ambas as direções incrementais, não uma
  sequência dedicada de laços menores.
- As 12 séries dinâmicas 80SOC foram examinadas. Em 25 °C, cruzamentos
  de sinal após remover `|I| <= 0,02 A`: **FUDS 1.020; US06 1.314;
  DST 236; BJDST 120**. São números de cruzamentos da série inteira,
  não contagens de ciclos, profundidades de SOC ou laços quase estáticos.
- O campo auxiliar de temperatura nos relatórios dinâmicos está desativado
  (`Temp = 0`, `Aux = False` em Info); os 17 canais de dados não incluem T.
  O protocolo existente utiliza o setpoint como T constante.
- A rotina histórica tem cache e pressupostos antigos de SOC; ela **não foi
  executada**. A leitura unificada usa Ah de descarga menos Ah de carga e
  não escreve nem reutiliza os caches históricos.

### Panasonic

- Enumeração recursiva: **19 subdiretórios, 48 arquivos físicos**, incluindo
  **dez MATs na raiz** e a coleção multiquímica aninhada em
  `external/mendeley_29kw38kzwj_v1/`.
- [UDDS/25 °C](../datasets/Pan18650/25degC_UDDS_Pan18650PF.mat):
  224.187 amostras, aproximadamente 6,236 h; I de −7,529 a +4,125 A;
  T da célula 25,62–27,11 °C; 1.616 cruzamentos de sinal.
- [US06/25 °C](../datasets/Pan18650/25degC_US06_Pan18650PF.mat):
  48.061 amostras; I de −20,822 a +7,575 A; T da célula
  25,61–32,97 °C; 524 cruzamentos. LA92/25 °C: 1.738 cruzamentos;
  HWFTa e HWFTb: 212 cada; UDDS/10 °C: 1.516.
- UDDS/0 °C, UDDS/−10 °C, US06/−10 °C e UDDS/−20 °C não apresentaram
  corrente de carga acima de 0,02 A. Perfis frios não devem ser descritos
  como regenerativos só porque o perfil nominal admite regeneração.
- A faixa térmica dos arquivos frios inclui a preparação/soak; por exemplo,
  UDDS/−20 °C registra T da célula entre −20,13 e +16,33 °C antes de
  qualquer recorte. O loader recorta apenas o bloco inicial previsto
  pelo protocolo, e a análise de prontidão não executou esse loader.

## Loader e requisitos do protocolo atual

Verificados os arquivos-fonte versionados, as definições de datasets em
[protocol.py](./protocol.py), os leitores em [data.py](./data.py),
[data_calce.py](./data_calce.py) e [exp_unified.py](./exp_unified.py).
Não confundir código MATLAB distribuído com um dataset com loader integrado
neste repositório.

Os seis nomes canônicos são `ecm_1rc_nls`, `ecm_2rc_nls`,
`physics_gradient` (**sem correção neural**), `output_residual`,
`ecm_ude` e `lstm`. **Ambiguidade de nomenclatura:** a “ablação B” histórica
em [exp_c3_ablation.py](./exp_c3_ablation.py) é `no_warm_start`, não
`physics_gradient`; não está listada como modelo na suíte unificada atual.
Os requisitos de dados abaixo atendem também essa ablação histórica, mas
isso não significa que ela já esteja integrada ao protocolo atual.
A replicação requer tempo, I, V, temperatura medida ou
setpoint explicitamente documentado, capacidade nominal e convenções de sinal
consistentes; SOC causal; um ciclo-fonte suficientemente longo para split
temporal 80/20 com remoção de sobreposição; ciclos/temperaturas alvo
independentes para zero-shot. Uma fonte sem T medida pode replicar a variante
CALCE, mas não a utilização de T da célula da variante Panasonic.

Disponibilidade de dados não implica integração automática: a configuração
atual aceita apenas `panasonic` e `calce`. Outras coleções exigiriam loaders,
seletores de ensaio e protocolo explícito, **não implementados nesta tarefa**.
Nenhuma classificação de prontidão abaixo é resultado de treinamento ou
certificação de erro dos modelos.

## Cobertura das demais subpastas e evidências críticas

| Pasta | Cobertura recursiva física | Grupos internos cobertos e limitações |
|---|---|---|
| `calb_l148n58a_raw` | 0 subpastas; 2 arquivos | Ambas as planilhas `L148N58A_59294_CH3_DrivingCycle_T25_Channel_3_Wb_1.xlsx` e `L148N58A_59294_CH3_EIS_T25_Channel_3_Wb_1.xlsx`; índices, metadados, séries e ACIM. |
| `ha_pozzato_onori_official_code` | 6 subpastas; 40 arquivos | Raiz `Code for Electrochemical Characterization Tools for Lithium-ion Batteries/` e módulos: 26 fontes MATLAB, 12 MAT, dois metadados macOS. Os 12 MAT são **cinco conjuntos únicos por hash**: `capacity_test_10cells` (5 cópias), `hppc_test_10cells` (3), `capacity_test_crate_data` (2), `capacity_test_temp_data` (1), `EIS_test_10cells` (1). Todos os conjuntos únicos inspecionados. |
| `lg_hg2_v2` | 7 subpastas; 419 arquivos | `extracted/` e seis grupos térmicos: 0degC **32 CSV/31 MAT**; 10degC **32/32**; 25degC **36/36**; 40degC **35/34**; n10degC **38/38**; n20degC **35/35**. Todos os CSV e MAT inspecionados; ZIP com 417 arquivos e seis diretórios, correspondentes à extração por CRC. `590_Charge10.csv` e `562_Charge10.csv` não têm MAT equivalente. |
| `mendeley_29kw38kzwj_v1` | 17 subpastas; 36 arquivos | `raw/`, `extracted/Battery Test Data/`, três químicas e seus grupos térmicos: LFP **5/25/45**; NCA/NMC **−5/5/25/45**. 32 planilhas, ZIP, RAR e dois metadados macOS. RAR: 32 planilhas + 15 diretórios; ZIP contém um RAR. Todos os índices de planilhas e grupos examinados, dados representativos de dinâmica e PCPD lidos. |
| `molicel_p42a_raw` | 0 subpastas; 2 arquivos | `20240422_A9_HCGT_25degC.xlsx` e `20250628_A9_EIS_SOC50_25degC_Channel_5.xlsx`, índices e canais inspecionados. |
| `oxford_hallemans_raw` | 1 subpasta; 2 arquivos | Somente `extracted_minimal/steady_state_README.md` e `multisine_and_singlesines_50SOC_70p5mArms.mat`; multisine e todas as 48 singlesines inspecionadas. |
| `rwth_35e_raw` | 0 subpastas; 250 arquivos | Seis ZIP Everlast válidos, ZIP de 28 figuras, PDF; **quatro `.zip.part` incompletos, 110 `.tmp-*` e 128 `.part-*` vazios**, sem índice ZIP válido. Os seis CSV foram lidos integralmente; resíduos não contados como ensaios. Figuras de outras células não substituem dados ausentes. |
| `stanford_m50t_raw` | 10 subpastas; 162 arquivos | **G1 22, V4 22, V5 12, W10 22, W3 22, W4 5, W5 6, W7 23, W8 22, W9 5 XLSX**; mais `.DS_Store`. 140 fragmentos UDDS e 21 diagnósticos. Todos os índices/metadados e registros iniciais inspecionados; onze C20 integrais, dois HPPC e dois fragmentos UDDS integrais. 36.449.497 linhas Channel declaradas, **não todas numericamente auditadas**; RawData não é ensaio independente. |

### LFP/Mendeley e cópia aninhada

- [PCPD LFP na pasta 25 Degree](<../datasets/mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/25 Degree/PCPD.xls>)
  tem folhas `OCV SOC Chr`, `OCV SOC Dis`, `Statis_9_1_3` e detalhes.
  São **11 pontos numéricos por ramo**, não 55 pontos de carga.
  Incrementos próximos de 250 mAh/±1,25 A, repouso de 30 min.
  Em aproximadamente 50% SOC: carga 3,3051 V e descarga 3,2868 V,
  com grids de SOC distintos. Isso sustenta duas curvas relaxadas,
  mas não valida equilíbrio completo ou uma interpolação específica.
- A mesma planilha traz caminho interno `LFP\30 Degree\PCPD.xls`
  e canais `SOC_30D_LFP_Chr`: **25 versus 30 °C não resolvido**.
  Não há T medida para arbitrar o conflito. O endpoint de carga em
  `5 Degree/PCPD.xls` também diverge dos repousos finais:
  3,3711 V tabelado versus aproximadamente 3,3373/3,3336 V.
- [Descargas LFP](<../datasets/mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/25 Degree/1.5C 1C 0.5C.xlsx>):
  spans de Capacity a 0,5/1/1,5C **2,498902/2,493894/2,489310 Ah**,
  até aproximadamente 2 V; integração concorda aproximadamente.
  Em 5 °C: **2,457065/2,338715/2,342899 Ah**; em 45 °C:
  **2,585011/2,521450/2,502555 Ah**.
  A discrepância com o rótulo 1,1 Ah impede certificar modelo/Qnom.
- [Dinâmicos LFP](<../datasets/mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh/25 Degree/DST FUDS UDDS HPPC WLTS US06.xlsx>):
  inversões após remover `|I| <= 1 mA`: **DST 656, FUDS 1.762,
  UDDS 996, HPPC 20, WLTS 969, US06 2.124**. Esse limiar
  difere do usado em CALCE/Panasonic: não comparar as contagens diretamente.
  HPPC inclui dez pulsos positivos de 11 s/aproximadamente 11,43 mAh;
  DST tem recuperação parcial de aproximadamente 0,84–3,34 mAh.
  São inversões parciais reais, sem prova de laços menores controlados.
- PCPD de 5/45 °C tem uma reversão global (descarga seguida de carga);
  PCPD “25” tem sete reversões incluindo preparação. Os ramos de OCV
  são essencialmente de **laço maior**, não uma identificação de minor loops.
- **Duplicação confirmada**: o RAR standalone, o RAR dentro do ZIP e o
  RAR em `Pan18650/external/mendeley_29kw38kzwj_v1/raw/` têm SHA256
  `60dac826be1f1f7b3d10e216ba03d56707f196d218a2a706b635e91d8657df07`
  e tamanho 102.805.962 bytes. As 32 planilhas extraídas têm hashes
  idênticos entre as duas cópias. A cópia aninhada tem 17 subpastas/
  36 arquivos, com três metadados macOS e sem o ZIP standalone.
  **Não são réplicas experimentais independentes.**
- Ressalvas multiquímicas: NMC `45 degree/DST FUDS UDDS HPPC WLTS US06.xls`
  contém **PCPD**, não os perfis do nome; NMC “5 degree” tem origem interna
  **Negative 5**. NMC `Negative 5 drgree/PCPD.xlsx` só tem ramo OCV de
  descarga identificado; NCA `N5 Degree/PCPD.xls` não tem ramos OCV
  bidirecionais identificados. Não assumir cobertura simétrica só pelo nome.

### LG HG2

- [README](<../datasets/lg_hg2_v2/extracted/Readme file - Description of Experimental Tests.txt>)
  e [PDF técnico](<../datasets/lg_hg2_v2/extracted/Technical Information and Experimental Test Results for LG 18650HG2.pdf>)
  confirmam ensaios, sensor superficial AD592 ±1 °C e química
  `Li[NiMnCo]O2 (H-NMC) / Graphite + SiO`. PDF: tensão nominal 3,6 V;
  CSV: 3,7 V, uma divergência documental.
- Em `10degC/`, **575_C20DisCh, 575_Dis_0p5C, 575_Dis_2C, 575_HPPC**
  são CSV com `Battery Name,LG ICR18650MG1_SN24AD`,
  capacidade nominal 2,9 Ah e comentário de termopar interno.
  Os MAT correspondentes rotulados LGHG2 não corrigem essa identidade.
- Capacidades da tabela principal referem-se a `551_Cap_1C` e
  `552_Cap_1C` em 25 °C, `589_Cap_1C` em 0 °C, `576_Cap_1C`
  em 10 °C, `596_Cap_1C` em −10 °C e `610_Cap_1C` em −20 °C.
  **Não usar** `10degC/571_Cap_1C` como capacidade a 10 °C:
  T mediana na descarga é 1,15674 °C. `40degC/557_Cap_1C`
  registra 2,50036 Ah, mas T mediana 1,840275 °C, não 40 °C.
- [UDDS/25 °C CSV](../datasets/lg_hg2_v2/extracted/25degC/551_UDDS.csv):
  159.646 registros, Δt mínimo/mediana/máximo **0,004/0,1/1,419 s**;
  I −9,18451 a +5,27935 A. `WhAccu` varia −9,41467 a −0,00001 Wh,
  enquanto `meas.Wh` do MAT é constante 3, aparentemente o canal Cnt.
  Esse erro não certifica defeito em I/V/T, mas proíbe presumir
  equivalência de todos os campos da conversão.
- `10degC/582_LA92.csv` tem quatro retrocessos de tempo e três registros
  sem I válido. `n10degC/602_Mixed4.csv` tem 4.160 tempos duplicados
  e lacuna de 415,993 s; `25degC/549_Charge.csv` tem lacunas de dezenas
  de milhares de segundos. A ordenação/reamostragem não deve atravessar
  intervalos sem aquisição como se fossem continuidade física.

### Ferramentas Ha e caracterização isolada

- [Run_Module_1.m](<../datasets/ha_pozzato_onori_official_code/Code for Electrochemical Characterization Tools for Lithium-ion Batteries/Module 1/Run_Module_1.m>)
  cita o DOI Pozzato e documenta aging. `capacity_test_10cells.mat`:
  grid 15 × 10 com **101 registros válidos**; W8 cai de 4,876859
  para 4,456854 Ah entre diagnósticos 1/15 e 0/347 ciclos.
  `hppc_test_10cells.mat` tem 100 entradas válidas; EIS tem 61,
  geralmente 19 frequências × SOC20/50/80.
- [capacity_test_crate_data.mat](<../datasets/ha_pozzato_onori_official_code/Code for Electrochemical Characterization Tools for Lithium-ion Batteries/Module 2/capacity_test_crate_data.mat>)
  contém carga C20/C40 **4,805746/4,843461 Ah** e descarga C20/C40
  **4,831238/4,866949 Ah**, 23 °C documental; sem reversões internas.
  Dados complementares de descarga C20 a 10/23/40 °C entregam
  **4,793479/4,836903/4,891364 Ah**.
  A função oficial integra `abs(I)`; isso **não** fornece Ah líquido
  quando há ambas as polaridades.
- CALB tem EIS de 120 pontos, aproximadamente 0,01–3.722 Hz,
  e dinâmicos bidirecionais; a tensão até 4,2 V e ausência de documentação
  LFP impedem promovê-lo a candidato LFP pelo nome “CALB”.
- Molicel tem GITT/pulsos no agendamento, mas não OOD térmico ou driving
  cycle identificado. OXFORD tem apenas excitação EIS local: curvas de
  relaxação descritas no README **não podem ser inventariadas como presentes**.

### RWTH e Stanford

- [Metadata RWTH](../datasets/rwth_35e_raw/Metadata.pdf), páginas 3/5:
  célula/química/3,4 Ah, câmara25, qOCV0,1C e DOI.
  Descargas completas integradas nos CSV 001–006:
  **3,398481; 3,390134; 3,406746; 3,382528; 3,390263; 3,407378 Ah**.
  A primeira descarga de 001 é parcial (0,8704 Ah); não foi usada como Q.
  qOCV de 001: descarga **3,42362 Ah**, carga **3,37344 Ah**.
- RWTH mantém 6.655 valores espúrios aproximadamente 4,765 A em 002
  e 26.385 aproximadamente 4,777 A em 003, embora o PDF descreva a
  correção desse artefato para zero. T medida dos seis CSV varia
  aproximadamente 25,25–32,375 °C, **não seis setpoints térmicos**.
  Metadados de falha térmica após dezembro/2020 não correspondem
  ao intervalo de 2018 da cópia local.
- [C20 G1 Stanford](../datasets/stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_0_05C_G1_Channel_3.xlsx):
  Global_Info configura 4,85 Ah; step 5 tem 72.780 registros, descarga
  −0,2425 A por aproximadamente 72.779 s até 2,49999 V e 4,903030 Ah.
  Onze descargas G1/V4/V5/W10/W3/W4/W5/W5Diag2/W7/W8/W9:
  **4,903030/4,868689/4,860165/4,865831/4,820700/4,822398/
  4,856494/4,832414/4,882482/4,876809/4,874209 Ah**.
- [HPPC G1 Stanford](../datasets/stanford_m50t_raw/G1/INR21700_M50T_T23_Aging_HPPC_G1_Channel_3.xlsx):
  descarga −4,85 A/360 s (aproximadamente 0,485 Ah), repouso1h,
  pulso −2,66 A/10 s, repouso40 s, pulso +2 A/10 s, repouso40 s.
  Isso permite OCV incremental de descarga, não garante ramo de carga.
- As famílias Stanford com dez XLSX são **fragmentos**, não dez ciclos
  independentes. Algumas só têm `.10.xlsx`, sem início local.
  Schedules N50 aparecem em nomes N75/N125; `Cycle_Index=1` não
  certifica o número de ciclos do nome.
- Sem T: dez G1/N100Part3, vinte V4/UDDS, W5/N125 e W7/N125.
  Diagnósticos V5: **90.472 T zero no C20 e 55.257 no HPPC**.
  Há T útil em outros arquivos, por exemplo UDDS W3 aproximadamente
  22,89–29,95 °C; isso é aquecimento no ensaio T23, não OOD de ambiente.

## 1. Quais pastas são LFP e servem ao go/no-go de histerese?

**LFP documentada pelos rótulos/folhas locais está somente no subconjunto
LFP de `mendeley_29kw38kzwj_v1` e na sua cópia aninhada em `Pan18650`.**
Fabricante/modelo exatos e a capacidade nominal continuam não certificados.
Nenhuma outra pasta foi confirmada como LFP. Em particular, **CALB não
significa automaticamente LFP**, e os arquivos Ha não documentam essa química.

Para tornar a classificação reproduzível:

- **ADEQUADO**: identidade/capacidade/condições consistentes e ramos de OCV
  bidirecionais aproveitáveis, com excitação dinâmica reversível e condições
  identificáveis para um teste válido.
- **PARCIAL**: há componentes essenciais, mas inconsistências ou lacunas
  impedem declarar prontidão definitiva.
- **INADEQUADO**: faltam os dados essenciais ou sequer há confirmação LFP.
  Não é conclusão de **NO-GO físico/modelístico**.

| Subconjunto LFP | OCV_chg e OCV_dis | M(z) | Inversões frequentes / laços menores | Classificação e justificativa |
|---|---|---|---|---|
| `mendeley_29kw38kzwj_v1/extracted/Battery Test Data/LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh` | **Possíveis por OCV incremental relaxada**, 11 pontos em cada direção nos três grupos; não curvas contínuas C20. É necessário reconstruir SOC, repousos e endpoints e resolver temperatura 25/30 °C. | **Possível em princípio** como `M(z) = [OCV_chg(z) − OCV_dis(z)]/2`, com `OCV_m(z) = [OCV_chg(z) + OCV_dis(z)]/2`, mesma escala SOC, temperatura e intervalo comum. Não calculado nem ajustado aqui. Não usar ramos desencontrados como M confiável. | **Sim**, FUDS/US06/DST/UDDS/WLTS e pulsos HPPC recuperam carga parcialmente. **Não comprovados** laços menores quase estáticos controlados; ramos PCPD são predominantemente laço maior. | **PARCIAL — melhor candidato LFP local.** Há ramos bidirecionais e FUDS/US06 ricos, mas nominal1,1 versus descarga aproximadamente2,5 Ah, temperatura ambígua e endpoints conflitantes impedem teste validamente parametrizado sem saneamento. |
| `Pan18650/external/mendeley_29kw38kzwj_v1/.../LiFePO4 18650 Rechargeable Cell 3.3V 1100 mAh` | **Exatamente as mesmas possibilidades/limitações**, hashes idênticos. | Idem; não constitui nova evidência nem réplica. | Idem; não confundir com os MATs Panasonic PF. | **PARCIAL**, duplicata do candidato anterior, não uma segunda célula/dataset independente. |

**Nenhum LFP local é classificado ADEQUADO sem ressalvas.**
Os dois candidatos aparentes `calb_l148n58a_raw` e
`ha_pozzato_onori_official_code` são **INADEQUADOS para um teste de LFP
confirmado**, embora contenham caracterizações úteis de química não
certificada. Oxford tampouco é LFP documental.

O critério previamente registrado em [lfp_data_inventory.md](./lfp_data_inventory.md)
é GO somente se **em FUDS e US06** houver redução de MAE ≥10% de V3
contra o melhor V1/V2 e tempo com `|h|>0,9` inferior a 90%.
Este inventário **não executa nem decide esse critério**. Os dados LFP
Mendeley não são os ensaios A123/CALCE anteriormente solicitados;
substituí-los muda a célula e requer decisão explícita antes do teste.

## 2. Não LFP: dinâmica e temperatura para replicar o protocolo unificado

| Pasta/subconjunto | Dinâmica e cobertura térmica | Prontidão para os modelos e OOD | Limitações/integração |
|---|---|---|---|
| **Pan18650 — MATs PF** | UDDS em cinco temperaturas, vários driving cycles a 25 °C, I/V/T/Ah. | **SIM, principal réplica direta existente.** Dados suficientes para os seis modelos canônicos e OOD de ciclo/temperatura previstos. | Loader/configuração existentes; calibração OCV independente ausente tem fallback explícito no protocolo. Excluir coleção externa da seleção PF. |
| **calce_20r_raw — Samsung20R** | Quatro perfis em 0/25/45 °C, OCV independente, I/V/Ah. | **SIM, réplica direta da variante CALCE.** | T é setpoint, não temperatura da célula medida; é suficiente para a variante já adotada, não para estudar T superficial. Loader/configuração existentes. |
| **lg_hg2_v2 — HG2 confirmado** | Quatro perfis e Mixed em seis grupos térmicos, I/V/T/Ah, C20/HPPC. | **SIM quanto à cobertura; CONDICIONAL quanto à integridade.** É o candidato adicional mais forte para réplica de ciclo/temperatura. | Excluir MG1, auditar setpoints/identidades, segmentar gaps/retrocessos/duplicatas, não confiar em todos os campos MAT. Leitor MAT genérico compatível, **sem configuração HG2 integrada**. |
| **mendeley — NCA NCR18650B** | Vários perfis e grupos −5/5/25/45, I/V/capacidade; T apenas rótulo. | **PARCIAL/CONDICIONAL.** Cobertura estrutural suficiente para variante com T constante documentada, mas não pronta sem validar segmentos e SOC. | Nominal e capacidade real diferentes; Ah resetável e rótulos/protocolos precisam validação. Não usar T ambiente não verificada como sensor. Sem loader. |
| **mendeley — NMC VTC6** | Vários perfis e grupos térmicos nominais, mas arquivo “dinâmico45” é PCPD e “5” aponta para −5. | **PARCIAL.** Não certificar OOD térmico pelos nomes; subconjunto útil pode existir após resolver conflitos. | Sem T medida/loader; fontes e alvos precisam seleção comprovada, não todos os perfis nas quatro temperaturas presumidos. Cópias em Pan18650 são os mesmos dados. |
| **rwth_35e_raw** | Perfil dinâmico de potência, T medida e aging, **câmara apenas25**. | **PARCIAL**, permite comparação dos seis modelos em fonte/aging; **não réplica completa de OOD térmico/ciclos distintos**. | Mesma política dinâmica/setpoint nas células; aquecimento não substitui ambiente OOD. Arquivos incompletos, artefatos I/gaps/V ausente e sem loader. |
| **stanford_m50t_raw** | UDDS, regimes de carga/aging, T útil em parte dos arquivos, **apenasT23**. | **PARCIAL**, útil para fonte/aging; **não réplica completa de OOD de temperatura e driving cycle**. | Só UDDS confirmado, fragmentos incompletos, T ausente/inválida em parte; fabricante/química não certificados. Sem loader. |
| **calb_l148n58a_raw** | Dinâmicos bidirecionais longos e T medida, **apenasT25**; perfis sem identificação padronizada. | **PARCIAL**, pode sustentar comparação interna dos seis modelos com protocolo novo; **não OOD térmico completo**. | Qnom/identidade química/formato não certificados; EIS não adiciona driving cycle OOD. Sem loader. |
| **molicel_p42a_raw** | GITT/pulsos/EIS e T medida, **apenasT25**, sem driving cycle nomeado. | **NÃO para a réplica completa**; apenas caracterização/experimento distinto. | Pulsos não substituem conjunto de ciclos veiculares e condições térmicas independentes. Identidade nominal inferida; sem loader. |
| **ha_pozzato_onori_official_code** | C20/HPPC/EIS/aging e capacidade em três temperaturas, **sem perfis veiculares brutos locais**. | **NÃO para a réplica dinâmica completa**. | Variação térmica só em ensaio de capacidade não equivale a dinâmica OOD; sem T medida e loader integrado. |
| **oxford_hallemans_raw** | EIS estreito em SOC50/T25. | **NÃO**. | Milhões de amostras AC não fornecem cobertura de SOC/temperaturas/ciclos exigida; sem Ah/T/sinal físico documentado. |

**Resposta operacional:** Panasonic PF e CALCE20R já atendem ao protocolo
implementado. LG HG2 é a expansão mais promissora, condicionada à auditoria
dos problemas encontrados. NCA/NMC Mendeley oferecem cobertura candidata,
mas não podem ser declarados prontos apenas pelos nomes das pastas.
Os demais não têm, localmente, simultaneamente driving cycles distintos
e temperaturas de ambiente independentes suficientes para replicar
integralmente as duas modalidades de OOD.

Nenhum loader foi executado/criado, nenhum dado corrigido, nenhum arquivo
incompleto removido e nenhum modelo ajustado. As limitações estão registradas
para decisão posterior, não tratadas silenciosamente como dados válidos.
