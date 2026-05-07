# Pipeline do Paper 1: ECM, LSTM e ECM-UDE

Este repositório implementa o pipeline experimental do Paper 1 para estimativa de tensão de bateria com três abordagens:

- `ecm_1rc`: baseline físico com circuito equivalente de 1 RC.
- `lstm`: baseline puramente orientado a dados.
- `ecm_ude`: modelo híbrido com ECM + Universal Differential Equation.

O fluxo cobre:

- comparação in-distribution em `UDDS @ 25°C`;
- sensibilidade a ruído de SoC na inferência;
- generalização zero-shot para temperaturas não vistas;
- generalização zero-shot para ciclos de condução não vistos;
- geração das figuras do artigo.

## Visão geral do pipeline

1. `exp1_ude.py`
Treina e compara `ECM-1RC`, `LSTM` e `ECM-UDE` no ciclo `25degC_UDDS_Pan18650PF.mat`.

2. `exp1_soc_noise_sensitivity.py`
Usa os melhores checkpoints do experimento 1 para medir robustez a ruido no canal de `SoC`. No estado atual do codigo, ele tambem depende de `results/p1_exp1/ecm_params_cache.json`.

3. `exp2_ude_temperature.py`
Avalia zero-shot em `UDDS` sob temperaturas `10°C`, `0°C`, `-10°C` e `-20°C`.

4. `exp3_ude_cycle.py`
Avalia zero-shot em ciclos não vistos a `25°C`: `US06`, `LA92` e `HWFT`.

5. `plots_paper1.py`
Gera as figuras finais a partir dos resultados já salvos em `results/`.

## Estrutura do repositório

- `data.py`: leitura dos arquivos `.mat`, derivação de `SoC`, normalização e janelamento.
- `training.py`: loop de treino, early stopping, scheduler e inferência em lote.
- `ecm.py`: identificação e inferência do baseline físico `ECM-1RC`.
- `ude.py`: implementação do `ECM-UDE` com `torchdiffeq`.
- `models.py`: arquiteturas neurais; neste pipeline principal, o script usa `LSTM`.
- `utils.py`: seleção de dispositivo e fixação de sementes.
- `exp1_ude.py`: experimento principal do Paper 1.
- `exp1_soc_noise_sensitivity.py`: robustez a erro de `SoC`.
- `exp2_ude_temperature.py`: OOD de temperatura.
- `exp3_ude_cycle.py`: OOD de ciclo de condução.
- `plots_paper1.py`: geração das figuras para publicação.
- `datasets/`: arquivos `.mat` de entrada.
- `results/`: saídas, checkpoints e figuras geradas.
- `anteriores/`: scripts antigos, fora do pipeline principal atual.

## Dados esperados

O código assume arquivos MATLAB no formato Kollmeyer dentro de `datasets/`. Cada `.mat` deve conter uma struct `meas` com os campos:

- `Time`
- `Current`
- `Voltage`
- `Battery_Temp_degC`
- `Ah`

O `SoC` é derivado internamente a partir de `Ah`.

Arquivos usados pelo pipeline principal:

- `datasets/25degC_UDDS_Pan18650PF.mat`
- `datasets/10degC_UDDS_Pan18650PF.mat`
- `datasets/0degC_UDDS_Pan18650PF.mat`
- `datasets/n10degC_UDDS_Pan18650PF.mat`
- `datasets/n20degC_UDDS_Pan18650PF.mat`
- `datasets/25degC_US06_Pan18650PF.mat`
- `datasets/25degC_LA92_Pan18650PF.mat`
- `datasets/25degC_HWFTa_Pan18650PF.mat`

## Ambiente

Execute tudo a partir da raiz deste repositório.

Os scripts nao usam argumentos de linha de comando. Caminhos, hiperparametros e configuracoes ficam definidos como constantes no topo de cada arquivo.

O projeto nao traz `requirements.txt` no estado atual, entao a instalacao precisa ser feita manualmente. Um setup minimo e:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib torch torchdiffeq
```

Dependencia opcional:

```bash
pip install pytorch-wavelets
```

`pytorch-wavelets` so e necessario se voce for usar as classes `WNO`/wavelet em `models.py` ou scripts antigos. O pipeline principal deste diretório usa `LSTM`, `ECM-1RC` e `ECM-UDE`.

## Protocolo usado no pipeline

- Features de entrada por passo de tempo: `I`, `T_cell` e `SoC`.
- Alvo: trajetoria de tensao terminal `V(t)`.
- Janelas: `length=1024`, `stride=512`.
- Split: temporal `80/20` com guard band para evitar vazamento entre treino e validacao.
- `I` e `V` usam estatisticas empiricas do split de treino.
- `T` e `SoC` usam constantes fisicas fixas.
- O experimento 1 roda `30` seeds e salva resultados incrementais, de forma retomavel.

## Como executar

### 1. Experimento principal

```bash
python exp1_ude.py
```

Saidas esperadas em `results/p1_exp1/`:

- `summary.json`
- `history_{model}_seed{seed}.json`
- `{model}_seed{seed}_best.pt`
- `{model}_best.pt`

Observacoes:

- o script e retomavel; se interromper, basta rodar novamente;
- `ECM-1RC` e ajustado uma vez no split de treino e reutilizado entre seeds;
- `LSTM` e `ECM-UDE` sao treinados seed a seed.

### 2. Sensibilidade a ruido de SoC

```bash
python exp1_soc_noise_sensitivity.py
```

Dependencias:

- `results/p1_exp1/lstm_best.pt`
- `results/p1_exp1/ecm_ude_best.pt`
- `results/p1_exp1/ecm_params_cache.json`

Nota:

- esse cache nao e gerado por `exp1_ude.py`;
- rode `exp2_ude_temperature.py` ou `exp3_ude_cycle.py` antes desta etapa.

Saida:

- `results/p1_exp1_soc_noise/summary.json`

### 3. OOD de temperatura

```bash
python exp2_ude_temperature.py
```

Dependencias:

- artefatos do experimento 1 em `results/p1_exp1/`

Saidas:

- `results/p1_exp2/summary.json`
- `results/p1_exp1/ecm_params_cache.json`

### 4. OOD de ciclo de conducao

```bash
python exp3_ude_cycle.py
```

Dependencias:

- artefatos do experimento 1 em `results/p1_exp1/`

Saidas:

- `results/p1_exp3/summary.json`
- pode reutilizar ou criar `results/p1_exp1/ecm_params_cache.json`

### 5. Figuras do artigo

```bash
python plots_paper1.py
```

Para gerar todas as figuras, rode antes:

1. `python exp1_ude.py`
2. `python exp2_ude_temperature.py`
3. `python exp3_ude_cycle.py`
4. `python exp1_soc_noise_sensitivity.py`

Saidas em `results/figures/`:

- `fig2_voltage_trace.*`
- `fig3_boxplot_seeds.*`
- `fig4_temp_ood.*`
- `fig5_cycle_ood.*`
- `fig6_us06_trace.*`
- `fig_ocv.*`
- `fig_learning_curves.*`
- `soc_noise_sensitivity.*`

Cada figura e salva em `PDF`, `PNG` e `SVG`.

## Ordem recomendada

Se a ideia for reproduzir o pipeline completo do paper:

```bash
python exp1_ude.py
python exp2_ude_temperature.py
python exp3_ude_cycle.py
python exp1_soc_noise_sensitivity.py
python plots_paper1.py
```

Essa ordem garante que o cache `results/p1_exp1/ecm_params_cache.json` exista antes da etapa de figuras.

## Saidas e versionamento

- `results/` esta ignorado no `.gitignore`.
- Em um clone limpo, e normal que os resultados ainda nao existam.
- Checkpoints e arquivos `summary.json` sao gerados sob demanda pelos scripts.

## Personalizacao rapida

- Para trocar datasets, altere `MAT_PATH`, `TRAIN_MAT` ou `TEST_CYCLES` nos scripts.
- Para mudar a granularidade temporal, altere `WINDOW_CFG`.
- Para mudar hiperparametros de treino, ajuste `TRAIN_CFG_*` em `exp1_ude.py`.
- Para mudar o diretorio de saida, altere `OUTPUT_DIR` no script correspondente.

## Observacoes importantes

- Os imports tentam primeiro `wno_battery.*` e depois caem para os modulos locais. Na pratica, rodar da raiz do repositorio ja e suficiente.
- `plots_paper1.py` exige `results/p1_exp1/ecm_params_cache.json`; esse arquivo e produzido pelos scripts OOD.
- `models.py` contem classes adicionais como `FNO`, `WNO` e hibridos, mas elas nao fazem parte do fluxo principal documentado aqui.
