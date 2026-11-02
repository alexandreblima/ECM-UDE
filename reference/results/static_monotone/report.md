# V7m: robustez com OCV monotona

Protocolo8a0e2e2. Fonte apenas;11nos PCHIP de incrementospositivos; retencao constante fora doSOC fonte.
Criterio: V7m>=25% reducao versusV1 em TODOS os alvos a mesmaT, sem escolher outros modelos.

| Dataset | Perfil | V1 | V7m | Reducao | Passa |
|---|---|---:|---:|---:|---|
| cell_A_a123 | identification failed | -- | -- | -- | INCONCLUSIVA |
| cell_B_mendeley | FUDS | 18.842 | 18.082 | 4.04% | False |
| cell_B_mendeley | US06 | 17.917 | 18.224 | -1.72% | False |
| panasonic | identification failed | -- | -- | -- | INCONCLUSIVA |
| calce | identification failed | -- | -- | -- | INCONCLUSIVA |

- cell_A_a123: robusto=None; status=INCONCLUSIVE_NLS.
- cell_B_mendeley: robusto=False; status=complete.
- panasonic: robusto=None; status=INCONCLUSIVE_NLS.
- calce: robusto=None; status=INCONCLUSIVE_NLS.
- cell_B_mendeley: parametros proximos aos limites=0 [].

## Ressalvas

- Vereditos originais nao foram reescritos; este e teste novo de uma unica varianteV7m.
- A123/Mendeley conservam saneamento e suposicoes de estadosiniciais anteriores.
- PanasonicV1OCV checkpoint grau5; dt1s difere do unificado; comparacao intraexperimento.
- CALCEV7m z0 fonte[.9,1] conservado, apesar fonte~80SOC; baselineprincipalV1corrigido porcontagemdeAh.
- SuporteOCV recalculado durante ajuste pelaSOC fonteQ/z0, nunca definido peloalvo.
- Incrementosmonotonicos sao restricao deforma, nao prova deidentificabilidadefisica.
- Q/z0 nao sao identificaveis neste protocolo: o suporteSOCfonte recalculado cancela ambos exatamente.
  CoordenadaAh normalizada elimina Jacobiano de puro arredondamento, sem alterar modelo ou hiperparametros.
  Jacobiano PCHIP/RC analitico validado por diferencas centrais; tentativas numericas falhas preservadas.
- Alvos e temperaturas secundarias avaliados somente se o ajuste fonte convergiu; nenhum criterio novo.
  Falhas NLS em A123/Panasonic/CALCE sao inconclusivas, nao melhorias ou perdas medidas.
