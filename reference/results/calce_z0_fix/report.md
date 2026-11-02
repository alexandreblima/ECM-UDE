# CALCE: auditoria z0 e avaliacao sem retreino

Regra congelada308dc2a. Contagem de carga tem prioridade sobre inversaoOCV. Nenhum alvo escolhe parametros.

- US06_25C: elegivel=True, metodo=same_file_charge_count, z0 antigo=1.0, novo=0.7999694136592465.
- BJDST_25C: elegivel=True, metodo=same_file_charge_count, z0 antigo=1.0, novo=0.79994404241519.

| Condicao | Familia | Modelo | Modo | MAE antes | MAE depois |
|---|---|---|---|---:|---:|
| US06_25C | unified | ecm_1rc_nls | windowed | 123.701 | 19.045 |
| US06_25C | unified | ecm_1rc_nls | continuous | 122.049 | 20.742 |
| US06_25C | unified | ecm_2rc_nls | windowed | 123.575 | 19.001 |
| US06_25C | unified | ecm_2rc_nls | continuous | 121.807 | 20.794 |
| US06_25C | unified | physics_gradient | windowed | 123.707 | 19.072 |
| US06_25C | unified | physics_gradient | continuous | 122.037 | 20.789 |
| US06_25C | unified | output_residual | windowed | 123.712 | 19.068 |
| US06_25C | unified | output_residual | continuous | 122.042 | 20.785 |
| US06_25C | unified | ecm_ude | windowed | 123.650 | 19.114 |
| US06_25C | unified | ecm_ude | continuous | 121.987 | 20.824 |
| US06_25C | unified | lstm | windowed | 123.371 | 38.668 |
| BJDST_25C | unified | ecm_1rc_nls | windowed | 123.481 | 18.970 |
| BJDST_25C | unified | ecm_1rc_nls | continuous | 122.911 | 19.482 |
| BJDST_25C | unified | ecm_2rc_nls | windowed | 123.366 | 18.899 |
| BJDST_25C | unified | ecm_2rc_nls | continuous | 122.683 | 19.515 |
| BJDST_25C | unified | physics_gradient | windowed | 123.487 | 18.996 |
| BJDST_25C | unified | physics_gradient | continuous | 122.899 | 19.527 |
| BJDST_25C | unified | output_residual | windowed | 123.492 | 18.992 |
| BJDST_25C | unified | output_residual | continuous | 122.904 | 19.523 |
| BJDST_25C | unified | ecm_ude | windowed | 123.430 | 19.040 |
| BJDST_25C | unified | ecm_ude | continuous | 122.849 | 19.564 |
| BJDST_25C | unified | lstm | windowed | 123.292 | 36.252 |
| US06_25C | static | V1 | continuous | 130.408 | 19.988 |
| US06_25C | static | V5 | continuous | 826.648 | 37.139 |
| US06_25C | static | V6 | continuous | 33.208 | 33.208 |
| US06_25C | static | V7 | continuous | 15.300 | 15.300 |
| BJDST_25C | static | V1 | continuous | 130.514 | 20.620 |
| BJDST_25C | static | V5 | continuous | 817.954 | 34.616 |
| BJDST_25C | static | V6 | continuous | 32.741 | 32.741 |
| BJDST_25C | static | V7 | continuous | 12.747 | 12.747 |

Veredito estatico historico: POSITIVA; hipotetico com z0 corrigido: **NEGATIVA**. Historico nao alterado.

- Unificado: pesos/split/normalizacao fonte preservados; dez seeds, nenhuma atualizacao.
- V1/V5 recebem z0 porperfil; V6/V7 conservam Q/z0 ajustados na fonte, portanto nao sao corrigidos retroativamente.
- V5 nao fisico continua excluido dos resultados principais, mesmo se a novaSOC reduzir extrapolacao.
- Mudanca do veredito e contrafactual sob novas condicoes auditadas, nao reescrita do criterio antigo.
- Tabelas anteriores e dados brutos intactos; hashes dos checkpoints/ancoras no manifesto.
