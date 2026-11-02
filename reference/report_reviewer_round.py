"""Assemble the reviewer report and verify complete previous-artifact preservation."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import numpy as np

from data_lfp import ROOT, save_json, sha256

AUDIT = ROOT/"results/reviewer_round_audit"


def read(relative: str):
    return json.loads((ROOT/relative).read_text())


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def preservation() -> dict:
    previous = read("results/reviewer_round_audit/previous_sha256.json")
    files = []
    for relative, expected in previous.items():
        path = ROOT/relative
        if not path.is_file():
            raise RuntimeError(f"Previous file removed:{relative}")
        same = sha256(path) == expected
        archive = ROOT/"figures_ieee/archive/ca3ca9b"/relative
        if not same and (not archive.is_file() or sha256(archive) != expected):
            raise RuntimeError(f"Previous bytes not preserved:{relative}")
        files.append({"path": relative, "previous_sha256": expected, "current_sha256": sha256(path),
                      "unchanged_in_place": same,
                      "archive": None if same else str(archive.relative_to(ROOT))})
    result = {"all_previous_bytes_preserved": True, "previous_count": len(files),
              "unchanged_in_place": sum(f["unchanged_in_place"] for f in files),
              "archived_exact": sum(not f["unchanged_in_place"] for f in files),
              "user_authorization": "Archive previous figure2/index/manifest/generator and related documentation",
              "registration_commit": "d14935e", "files": files}
    save_json(AUDIT/"preservation.json", result)
    return result


def main() -> None:
    audit = preservation()
    a = read("results/param_ablation/summary.json")
    b = read("results/variant_tost/summary.json")
    c = read("results/training_audit/summary.json")
    lines = ["# Reviewer round: ablacao, equivalencia direta, treino e tabelas", "",
             "Branch paper-v2. Protocolo [reviewer_round_protocol.json](reviewer_round_protocol.json)",
             "registrado no commit **d14935e**, antes de qualquer ajuste. Somente NLS na Parte A;",
             "B/C/D usam resultados salvos, sem treinar redes, trocar hiperparametros ou alterar criterios.", "",
             "## A. Ablacao de parametros", "",
             "V1 e V6 conjuntos existentes foram reutilizados. V6-Q ajusta Q/z0 com RC de V1 fixos;",
             "V6-RC ajusta somente RC com Q medido e z0 causal por perfil. OCV fixa e simulacao",
             "continua a 1 s, RC inicial zero, estados nao reinicializados por janela. Todos os ajustes usam",
             "somente a fonte completa; tres inicializacoes registradas, TRF, 500 avaliacoes, tolerancias 1e-8.",
             "V6-Q local mantem RC de DST25; V6 local foi reutilizado, nao reajustado.", "",
             "### A1. Auditoria CALCE e V6-wide", "",
             "| Perfil25C | z0 auditado | Metodo |", "|---|---:|---|"]
    for profile, row in a["calce_audit"].items():
        lines.append(f'| {profile} | {row["z0"]:.9f} | {row["method"]} |')
    lines += ["", "A auditoria da fonte FUDS encontrou 0.799972, nao o antigo 0.822910 por inversao OCV.",
              "V1 nao foi reajustado: os parametros antigos sao reavaliados com z0auditado em todos os perfis.",
              "V6 antigo permanece como referencia solicitada, inclusive limitesz0[.9,1] e ajuste com",
              "inicializacao historica da fonte; isso e ressalva de comparabilidade, nao refit silencioso.",
              "V6-wide usa z0[.7,1], mesma fonte auditada e transfere Q/z0/RC sem substituir z0 por alvo.",
              "Os alvos auditados definem baselineV1/V6-RC e mascarasSOC, nao uma recalibracao wide por alvo.", "",
              "| Alvo | V1 auditado(mV) | V6-wide(mV) | Reducao | Passa>=25% |",
              "|---|---:|---:|---:|---|"]
    wide = a["blocks"]["calce"]
    for row in wide["rows"]:
        d = wide["wide_decision"]["profiles"][row["target"]]
        lines.append(f'| {row["target"]} | {row["V1_mae_mV"]:.3f} | {number(row["V6-wide_mae_mV"])} | '
                     f'{100*d["reduction"]:.2f}% | {d["passes"]} |')
    p = wide["fits"]["V6-wide"].get("parameters")
    if p:
        lines += ["", f'Qwide={p["Qnom_Ah"]:.6f}Ah; z0wide={p["z0"]:.6f}.']
    lines += [f'Veredito registrado V6-wide: **{wide["wide_decision"]["verdict"]}**.',
              "Esse e um novo teste com limite ampliado; nao reescreve o anterior CALCE corrigido NEGATIVO.", "",
              "### A2. MAE e fracoes explicadas por condicao", "",
              "f=(MAE_V1-MAE_variante)/(MAE_V1-MAE_V6). Fracoes assinadas, sem truncar;",
              "denominador zero=>indisponivel com aviso; denominador negativo nao significa ganho explicado",
              "e nao participa da atribuicao. Maioria estrita (>metade) por bloco, apenas ganho V6>0.",
              "Ambos mecanismos atingindo70% na maioria, ou nenhum, implica conjunto.", "",
              "| Bloco | Alvo | V1 | V6-Q | V6-RC | V6 | fQ | fRC | V6-OCVT |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, block in a["blocks"].items():
        for r in block["rows"]:
            lines.append(f'| {name} | {r["target"]} | {r["V1_mae_mV"]:.3f} | {number(r.get("V6-Q_mae_mV"))} | '
                         f'{number(r.get("V6-RC_mae_mV"))} | {r["V6_mae_mV"]:.3f} | '
                         f'{number(r["V6-Q_fraction"])} | {number(r["V6-RC_fraction"])} | '
                         f'{number(r.get("V6-OCVT_mae_mV"))} |')
    lines += ["", "MAE em mV. transfer: parametrosDST25; local: ajusteDST da propriaT.",
              "V1termico sempre transferidoDST25. Valoresnull nao sao zero nem resultados favoraveis.", "",
              "| Bloco | Condicoes ganhoV6>0 / total | Q>=70% | RC>=70% | Veredito descritivo |",
              "|---|---:|---:|---:|---|"]
    for group, result in a["interpretation"].items():
        lines.append(f'| {group} | {result["eligible_positive_joint_gain"]}/{result["total_conditions"]} | '
                     f'{result["counts_at_least_70pct"]["V6-Q"]} | {result["counts_at_least_70pct"]["V6-RC"]} | '
                     f'**{result["verdict"]}** |')
    lines += ["", "Mesma temperatura por celula (descritivo adicional):"]
    for cell in ("cell_A_a123", "cell_B_mendeley", "panasonic", "calce"):
        r = a["blocks"][cell]["interpretation"]
        lines.append(f'- {cell}: {r["verdict"]}; condicoes elegiveis {r["eligible_positive_joint_gain"]}.')
    failures = [(name, model) for name, block in a["blocks"].items()
                for model, fit in block["fits"].items() if not fit["success"]]
    lines += ["", "### A3. OCV da propria temperatura e ressalvas", ""]
    for name, model in failures:
        lines.append(f'- **INCONCLUSIVO_NLS: {name}/{model}**. Nenhuma inicializacao convergiu em500avaliacoes; '
                     "parametros falhos nao foram pontuados nos alvos.")
    lines += ["- V6-OCVT conserva Q25como referencia dinamica +/-10%; ramosOCVlocais seguem a escala",
              "  de capacidade de baixa corrente local do loader existente. Substitui so a curva, nao Qreferencia.",
              "- OCVT local a0/40/50C e comparacao diagnostica fonteDST; a10C inconclusiva.",
              "- V6-RC equivale matematicamente a V1 em Q/z0 medidos; reotimizacao pode achar",
              "  outro minimo e CALCE muda a inicializacao auditada. Nao e evidencia causal independente.",
              "- Mendeley: preparacao ausente, temperatura nominal ambigua 25/30 C e capacidade ~2.50 Ah;",
              "  Panasonic usa curva OCV do checkpoint. Seu criterio SOC e descritivo e sensivel a ganhos",
              "  pequenos: HWFET melhora apenas ~0.137 mV com V6 (fracao Q=11.11 nao significa ganho grande).",
              "- Q/z0/RC continuam confundidos; a atribuicao>=70% e criterio descritivo, nao prova fisica.",
              "- O bloco local completo e INCONCLUSIVO quanto ao conjunto de variantes pedidas, pois falta OCVT10.",
              "  A contagem descritiva RC 5/8 usa apenas as variantes Q/RC/V6 convergentes nas 8 condicoes;",
              "  nao constitui declaracao de bloco completo. Execucao complete=true nao significa todos os fits convergentes;",
              "  all_requested_fits_converged=false registra a ressalva em JSON.",
              "- [Manifestos e tentativas](results/param_ablation/summary.json),",
              "  [tabelaCSV](results/param_ablation/table.csv), [metricas porSOC/inversoes](results/param_ablation/).", "",
              "## B. Comparacao direta das variantes", "",
              "| Diferenca | Equivalentes /18 |", "|---|---:|"]
    for model in ("ecm_ude", "output_residual"):
        records = [r for r in b["records"] if r["model"] == model]
        lines.append(f'| {model} - physics_gradient | {sum(r["equivalent"] for r in records)}/18 |')
    lines += ["",     "Mesmas seeds 42--51 em cada par; MAE continuo; margem +/-1 mV.",
    "TOST inferior usa d+1, alternativa greater; superior d-1, alternativa less.",
    "Holm 18 por par/direcao; ambos pHolm<.05. Exclui os dois fallbacks CALCE25,",
              "sem substituir resultadoscorrigidos numa familia originalmente diferente.", "",
              "### Estatistica, estimando e p exato", "",
              "- Estatistica W+=soma dos ranks absolutos dos valores deslocados com sinal positivo.",
              "- Wilcoxon estima localizacao/pseudomediana sob simetria; nao testa diretamente uma",
              "  mediana amostral arbitraria. Reportamos mediana das diferencas E pseudomediana",
              "  Hodges-Lehmann=mediana das55medias Walsh (diagonal incluida), nao as confundimos.",
              "- Antes do teste: arredondar diferencas deslocadas a10decimais; remover zeros.",
              "  Empates absolutos recebem rankmedio; todoszeros=>p=1.",
              "- pexato por enumeracao de2^n sinais dos ranks, cauda inclusiva; sem aproximacao normal",
              "  nem atalho scipy.wilcoxon com tratamento inadequado de empates.",
              f'- Implementacao Python/NumPy{b["library_versions"]["numpy"]}; '
              f'SciPy{b["library_versions"]["scipy"]} apenas rankdata; Holm implementado no repositorio.',
              "- [Codigo efetivo](exp_paper_v2_seeds.py), [manifesto B](results/variant_tost/summary.json),",
              "  [tabela B](results/variant_tost/comparison.csv).",
              "- LSTM x2RC: unidade bootstrap e a diferenca de MAE porseed, nao amostra temporal.",
              "  Reamostra10diferencas com reposicao,20000vezes; mediana porreamostragem;",
              "  intervalo percentil2.5/97.5 com np.quantile linear, RNGdefault_rng(20261101).",
              "  2RCdeterministico nao se torna10ajustes independentes. Apenas janelas.", "",
              "## C. Auditoria dos treinos salvos e do codigo", "",
              "[CSV com 80 linhas](results/training_audit/epochs.csv): dataset/modelo/seed, epoca selecionada,",
              "epocas executadas, ultimas perdas treino/validacao e perdas na epoca selecionada.",
              "Ultima perda nao e a perda dos pesos restaurados; treino da epoca e media ao longo dos",
              "minibatches com pesos evoluindo. O checkpoint selecionado usa menor validacao estrita.", "",
              "| Dataset | Modelo | Epoca selecionada mediana[min,max] | Executadas mediana[min,max] |",
              "|---|---|---:|---:|"]
    for dataset in ("panasonic", "calce"):
        for model in (*("physics_gradient", "output_residual", "ecm_ude"), "lstm"):
            runs = [r for r in c["runs"] if r["dataset"] == dataset and r["model"] == model]
            epochs, executed = ([r[key] for r in runs] for key in ("selected_epoch", "executed_epochs"))
            lines.append(f'| {dataset} | {model} | {np.median(epochs):g}[{min(epochs)},{max(epochs)}] | '
                         f'{np.median(executed):g}[{min(executed)},{max(executed)}] |')
    lines += ["", "Oito PDFs vetoriais de 3.5 in, Times>=8 pt, sem titulos internos; mediana e faixa min/max.",
              "Somente seeds observadas em cada epoca; NAO preenchemos caudas apos early stopping.",
              "Contagem por epoca no manifesto evita comparar dez seeds iniciais com sobreviventes sem aviso.", ""]
    for key, entry in c["curves"].items():
        lines.append(f'- {key}: [{Path(entry["file"]).name}](results/training_audit/{Path(entry["file"]).name}).')
    lines += ["", "### C1. Intervalo NLS do unificado (nao confundir com Parte A)", "",
              "O ECM-1RC E o ECM-2RC usam **somente janelas de treino guardadas**, nao perfil inteiro:",
              "[fit_nls](exp_unified.py#L180) seleciona ctx.tr; estadosRC reiniciados em cada janela.",
              "Janelas sobrepostas repetem amostras na funcao objetivo. A normalizacao tambem usa so",
              "a uniao das amostras de treino, [dataset_context](exp_unified.py#L148).", ""]
    for dataset, split in c["NLS_source_intervals"].items():
        dt = .1 if dataset == "panasonic" else 1
        last, first = split["train_last_sample_exclusive"], split["heldout_first_sample"]
        lines.append(f'- {dataset}: {split["train_windows"]} janelas treino; samples[0,{last}) '
                     f'(t=0--{(last-1)*dt:.1f}s); validacao comeca sample{first} (t={first*dt:.1f}s), '
                     f'{split["heldout_windows"]}janelas; perfil{split["n_samples"]}amostras.')
    lines += ["", "Parte A e estudos estaticos usam DST/UDDS/FUDS completos, nao essa particao.",
              "A fonte retida seleciona early stopping: validacao, nao teste final independente.", "",
              "### C2. Normalizacao, perda e otimizacao", "",
              "- Entrada X=(I,T,SOC): I centrada/escalada por media/desvio de treino; Tmean=0,Tstd=25;",
              "  SOCmean=.5,SOCstd=.3. Saida V por Vmean/Vstd so de treino. Estatisticas fixas nos alvos.",
              "  [compute_norm_stats](data.py#L308), [renormalize](data.py#L443), [avaliacao](exp_unified.py#L233).",
              "- Loss dos quatro treinaveis: MSE de V normalizada, media minibatch/amostras; nenhum compute_loss",
              "  especial no UnifiedECMUDE. Residual_loss nas historias=0. Sem penalizacao fisica adicional oculta.",
              "  [loss](training.py#L223), [MSE](training.py#L355), [media](training.py#L255).",
              "- Adam com weightdecay1e-5; gradiente/residual/UDE lr2e-4,30epocas,clip.5,patience8;",
              "  LSTM lr1e-3,150epocas,clip1,patience30. Batch32,CPU; configsalva e hashes auditados.",
              "- LambdaLR: warmup3fisicos/5LSTM, depois coseno ate fator.01; step apos epoca;",
              "  loglr e o usado antes do scheduler. [config](protocol.py#L20), [Adam](training.py#L340),",
              "  [scheduler](training.py#L184), [earlystopping](training.py#L418).", "",
              "### C3. Integracao, estados e limites NLS", "",
              "- Unificado: RK4classico ZOH, dt=.1sPanasonic/1sCALCE; estados polarizacao0 e SOCcausal",
              "  no inicio da janela; continuo inicia uma vez e carrega ate fim. Fontemetricas mascaradas",
              "  apenasvalidacao. [integrate](unified_dynamics.py#L111), [nls](exp_unified.py#L168).",
              "- Gradiente/residual/UDE inicializados do1RC, NAO2RC; C>=1F,R>=1e-4 por softplus;",
              "  eta=1+.1tanh; sem tau3600 no treino. OCV grau5 aprendivel, avaliacaoSOCclip[0,1].",
              "  [physical](unified_dynamics.py#L79), [construcao](unified_dynamics.py#L185).",
              "- NLSunificado: graus5 Chebyshev coeff[-10,10], R0/Ri[.001,.5]ohm,",
              "  tau[dt,10000]s. Uma inicializacao: OCVleast-squares, R0=.03,Ri=.02,tau1=5,tau2=40.",
              "  Penalizacao coeff:1e-3*[0,2,4,8,16,32]*c anexada aos residuos; TRFlinear,xscalejac,",
              "  max500avaliacoes,ftol/xtol/gtol1e-9; result.success exigido. [codigo](exp_unified.py#L180).",
              "- Parte A: RCexatoZOH a1s, tau<=3600,R[1e-5,.5],Q+/-10%,",
              "  z0[.9,1] ouwide[.7,1]; tres starts (tau2/40,10/300,1/1000; z0=1,.95,.9).",
              "  TRFlinear, xscalejac, max500,tolerancias1e-8; menorSSE entreconvergentes; nenhum",
              "  alvo escolheminimo. [bounds](exp_lfp_static.py#L101), [novo runner](exp_param_ablation.py).",
              "- LSTM: sem integracao ODE; estados oculto/celula zerados no inicio de cada janela;",
              "  entradas/saida usam a mesma normalizacao do benchmark. Nao ha inferencia continua LSTM reportada.",
              "- [Manifesto codigo/historias](results/training_audit/summary.json) registra hashes, config-base",
              "  (cinco seeds originais) e lista efetiva de dez seeds 42--51 separadamente.", "",
              "## D. Figura e tabelas IEEE", "",
              "- [Figura 2](figures_ieee/fig02_physics_vs_lstm.pdf): LSTM dez seeds, mediana[min,max], nao media/SD;",
              "  2RC deterministico, janelas, 18 principais. So este PDF regenerado; outros sete intactos.",
              "- [Indice](figures_ieee/figures_index.md) e manifesto atualizados; versoes anteriores arquivadas.",
              "- TabelaE1 [CSV](results/paper_tables/E1.csv) / [booktabs table*](results/paper_tables/E1.tex):",
              "  18condicoes, MAEcontinuo2RC e mediana[min,max] dosfisicos, TOSTcontra2RC egradient.",
              "  Contra2RC conservaHolm20; variantesHolm18. Faixas nao saoICs.",
              "- Condicoes [CSV](results/paper_tables/conditions.csv) / [TeX](results/paper_tables/conditions.tex):",
              "  20doE1 e fontes/alvosE2--E4, origemz0,papeis,inclusao; sem misturar fallback e auditado.",
              "- Criterios [CSV](results/paper_tables/criteria.csv) / [TeX](results/paper_tables/criteria.tex):",
              "  E1unificado,E2histerese,E3estatico,E4termico/interpolacao; registros originais ewide separados.",
              "  Exige pacote booktabs; table* ocupa largura de duas colunas.", "",
              "## Preservacao e validacao", "",
              f'- {audit["previous_count"]} versoesanteriores verificadas porSHA256: '
              f'{audit["unchanged_in_place"]} no lugar, {audit["archived_exact"]} copiasexatas arquivadas.',
              "- [Auditoria](results/reviewer_round_audit/preservation.json); [testes](results/reviewer_round_audit/tests.log).",
              "- Nenhum resultadoanterior reescrito. Manuscrito e checkpoints intactos.",
              "- Falha original de exportacao CSV (camposwide heterogeneos) foi corrigida sem refit:",
              "  reexportacao e reavaliacao de fitssalvos. Log preservado; RC-only usa z0auditado de cada alvo.",
              "- Nao se modifica nenhum criterio apos ver resultados; OCVT10 nao recebe orcamentoextra.", ""]
    (ROOT/"report_reviewer_round.md").write_text("\n".join(lines))
    audit["new_artifact_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for name in
                                    ("param_ablation", "variant_tost", "training_audit", "paper_tables")
                                    for p in (ROOT/"results"/name).rglob("*") if p.is_file()}
    audit["report_sha256"] = sha256(ROOT/"report_reviewer_round.md")
    audit["producer_sha256"] = {name: sha256(ROOT/name) for name in
                                ("exp_param_ablation.py", "reviewer_saved_analysis.py", "report_reviewer_round.py",
                                 "figures_ieee.py", "test_reviewer_round.py", "test_figures_ieee.py")}
    validation = {}
    test_path = AUDIT/"tests.log"
    if test_path.is_file():
        text = test_path.read_text()
        match = re.search(r"Ran (\d+) tests", text)
        validation["focused_tests"] = {"count": None if match is None else int(match.group(1)),
                                       "passed": text.rstrip().endswith("OK"), "sha256": sha256(test_path)}
    latex_path = AUDIT/"latex_validation.log"
    if latex_path.is_file():
        text = latex_path.read_text()
        validation["IEEEtran_tables"] = {"passed": "Output written" in text and "\n!" not in text,
                                         "overfull_boxes": text.count("Overfull"),
                                         "sha256": sha256(latex_path)}
    audit["validation"] = validation
    audit["execution_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    save_json(AUDIT/"preservation.json", audit)
    print("Reviewer report generated; previous artifacts preserved:", audit["previous_count"])


if __name__ == "__main__":
    main()
