#!/usr/bin/env python3
"""Gera o README a partir da evidência medida, não da memória.

Cada tabela é montada lendo `out/*.json` — arquivos que os scripts de avaliação
gravaram. Se um arquivo exigido faltar, o script falha: melhor não publicar do
que publicar com número sem lastro.

Uso:
    python scripts/gerar_readme.py            # escreve README.md
    python scripts/gerar_readme.py --conferir # só valida que tudo existe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path("out")


class Faltando(RuntimeError):
    pass


def carregar(nome: str) -> dict:
    caminho = OUT / nome
    if not caminho.exists():
        raise Faltando("evidência ausente: %s" % caminho)
    return json.loads(caminho.read_text(encoding="utf-8"))


def pct(valor: float) -> str:
    return ("%.2f%%" % valor).replace(".", ",")


def ic(intervalo) -> str:
    if not intervalo:
        return "—"
    return "[%s; %s]" % (("%.2f" % intervalo[0]).replace(".", ","),
                         ("%.2f" % intervalo[1]).replace(".", ","))


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------

def tabela_camadas() -> str:
    linhas = [
        "| Modelo | Tabuleiro exato | Cor | Lances legais (F1) | Melhor lance = solução |",
        "|---|---|---|---|---|",
    ]
    base = carregar("aval-teste-sft.json")
    linhas.append("| Base + SFT direto (só o lance) | não pede | — | — | %s %s |"
                  % (pct(base["melhor_correto_pct"]), ic(base["melhor_correto_ic95"])))
    grpo = carregar("aval-teste-grpo-sf.json")
    linhas.append("| Base + SFT + GRPO com Stockfish | não pede | — | — | %s %s |"
                  % (pct(grpo["melhor_correto_pct"]), ic(grpo["melhor_correto_ic95"])))

    for rotulo, arquivo in (("Modelo de mundo, 30k posições", "aval-mav-30k.json"),
                            ("Modelo de mundo, 70k posições", "aval-mav-final.json"),
                            ("**Modelo de mundo, 70k + lance verificado**",
                             "aval-mav-verificado.json")):
        d = carregar(arquivo)
        linhas.append("| %s | %s | %s | %s (%s) | %s %s |" % (
            rotulo, pct(d["grade_exata_pct"]), pct(d["cor_certa_pct"]),
            pct(d["legais_exatos_pct"]), ("%.3f" % d["legais_f1_medio"]).replace(".", ","),
            pct(d["melhor_correto_pct"]), ic(d["melhor_correto_ic95"])))
    return "\n".join(linhas)


def tabela_elo() -> str:
    dados = carregar("elo-puzzle.json")
    linhas = ["| Modelo | n | Acertos | Elo de puzzle | IC95 |", "|---|---|---|---|---|"]
    rotulos = {
        "aval-teste-sft": "SFT direto",
        "aval-teste-grpo-sf": "GRPO + Stockfish",
        "aval-mav-30k": "Modelo de mundo, 30k",
        "aval-mav-final": "Modelo de mundo, 70k",
        "aval-mav-verificado": "Modelo de mundo, 70k + verificado",
    }
    for r in dados:
        elo = "indefinido" if r["elo_puzzle"] is None else "%.0f" % r["elo_puzzle"]
        chave = Path(r["arquivo"]).stem
        linhas.append("| %s | %d | %s | **%s** | %s |" % (
            rotulos.get(chave, chave), r["n"],
            pct(100 * r["taxa"]) if r["taxa"] is not None else "—", elo, ic(r["ic95"])))
    return "\n".join(linhas)


def tabela_partidas() -> str:
    linhas = ["| Oponente | Partidas | V/E/D | Pontuação | Elo estimado | Lances ilegais |",
              "|---|---|---|---|---|---|"]
    for rotulo, arquivo in (
            ("Jogador aleatório", "partidas-aleatorio-verificado.json"),
            ("Stockfish `UCI_Elo=1320`", "partidas-stockfish-verificado.json")):
        d = carregar(arquivo)
        elo = ("—" if d.get("elo_estimado") is None
               else "**%.0f** %s" % (d["elo_estimado"], ic(d.get("elo_ic95"))))
        linhas.append("| %s | %d | %d/%d/%d | %s | %s | %s |" % (
            rotulo, len(d["partidas"]),
            d["vitorias"], d["empates"], d["derrotas"],
            ("%.3f" % d["pontuacao"]).replace(".", ","), elo,
            pct(100 * d["taxa_lances_ilegais"])))
    return "\n".join(linhas)


def tabela_generalizacao() -> str:
    base = carregar("generalizacao-base.json")
    treinado = carregar("generalizacao-treinado.json")
    linhas = ["| Tarefa | Modelo base | Modelo treinado | Δ |", "|---|---|---|---|"]
    for rotulo, chave in (("Lig 4: vencer agora", ("lig4", "vencer")),
                          ("Lig 4: bloquear a ameaça", ("lig4", "bloquear")),
                          ("Lig 4: total (acaso: %s)", ("lig4", "total")),
                          ("GSM8K (matemática)", ("gsm8k",))):
        a = base[chave[0]]
        b = treinado[chave[0]]
        for k in chave[1:]:
            a, b = a[k], b[k]
        delta = b["acerto_pct"] - a["acerto_pct"]
        nome = rotulo % (("%.1f%%" % b["acaso_medio_pct"]).replace(".", ",")
                         if "%s" in rotulo else ())
        linhas.append("| %s | %s | %s | %s pp |" % (
            nome, pct(a["acerto_pct"]), pct(b["acerto_pct"]),
            ("%+.1f" % delta).replace(".", ",")))
    return "\n".join(linhas)


def tabela_espelho() -> str:
    linhas = ["| Modelo | Original | Espelhado | Queda | Descartados |",
              "|---|---|---|---|---|"]
    for rotulo, arq, esp in (
            ("SFT direto", "aval-teste-sft.json", "espelho-sft.json"),
            ("Modelo de mundo", "aval-mav-verificado.json", "espelho-mav-verificado.json")):
        a, b = carregar(arq), carregar(esp)
        queda = a["melhor_correto_pct"] - b["melhor_correto_pct"]
        linhas.append("| %s | %s | %s | %s pp | %d |" % (
            rotulo, pct(a["melhor_correto_pct"]), pct(b["melhor_correto_pct"]),
            ("%+.1f" % queda).replace(".", ","), b.get("descartados_espelho", 0)))
    return "\n".join(linhas)


def tabela_escala() -> str:
    linhas = ["| Etapa | Exemplos | Passos | Tempo | VRAM pico |", "|---|---|---|---|---|"]
    for rotulo, arquivo in (("Modelo de mundo, 30k", "mav-30k-metricas.json"),
                            ("Modelo de mundo, +40k (continuação)", "mav-final-metricas.json")):
        d = carregar(arquivo)
        log = d.get("log") or [{}]
        passos = log[-1].get("step") or log[-1].get("epoch") or "—"
        runtime = log[-1].get("train_runtime")
        tempo = ("%.0f min" % (runtime / 60)) if runtime else "—"
        linhas.append("| %s | %d | %s | %s | %s GB |" % (
            rotulo, d.get("tracos", 0), passos, tempo,
            ("%.2f" % d.get("vram_pico_gb", 0)).replace(".", ",")))
    return "\n".join(linhas)


def tabela_outros_metodos() -> str:
    """Os métodos que não são o foco, com os números medidos."""
    linhas = ["| Método | Acerto em puzzle | Observação |", "|---|---|---|"]
    for rotulo, arquivo, nota in (
            ("Base, sem treino", "baseline-qwen1.5b.json", "responde `e2e4` sempre"),
            ("SFT direto", "aval-teste-sft.json", "aprende a forma do lance"),
            ("SFT + GRPO com Stockfish", "aval-teste-grpo-sf.json", "recompensa densa"),
            ("SFT + raciocínio com candidatos", "cot2-600.json", "amostra diferente"),
            ("Raciocínio + GRPO", "grpo-cot-600.json", "amostra diferente"),
            ("Varredura de lances forçados", "grpo-forcados-600.json", "amostra diferente")):
        d = carregar(arquivo)
        if "melhor_correto_pct" in d:
            valor, n = d["melhor_correto_pct"], d["n"]
        elif "raciocinio" in d:
            valor, n = 100 * d["raciocinio"]["resumo"]["correto"]["taxa"], d["raciocinio"]["resumo"]["correto"]["total"]
        else:
            valor = 100 * d["original"]["correto"]["taxa"]
            n = d["original"]["correto"]["total"]
        linhas.append("| %s | %s (n=%d) | %s |" % (rotulo, pct(valor), n, nota))
    return "\n".join(linhas)


TABELAS = {
    "{{TABELA_CAMADAS}}": tabela_camadas,
    "{{TABELA_ELO}}": tabela_elo,
    "{{TABELA_PARTIDAS}}": tabela_partidas,
    "{{TABELA_GENERALIZACAO}}": tabela_generalizacao,
    "{{TABELA_ESPELHO}}": tabela_espelho,
    "{{TABELA_ESCALA}}": tabela_escala,
    "{{TABELA_OUTROS}}": tabela_outros_metodos,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="docs/readme_template.md")
    ap.add_argument("--saida", default="README.md")
    ap.add_argument("--conferir", action="store_true")
    args = ap.parse_args()

    texto = Path(args.template).read_text(encoding="utf-8")
    faltando = [m for m in TABELAS if m not in texto]
    if faltando:
        print("template sem marcadores:", faltando)
        return 1

    if args.conferir:
        problemas = []
        for marcador, fn in TABELAS.items():
            try:
                fn()
                print("ok   ", marcador)
            except Faltando as exc:
                problemas.append(str(exc))
                print("FALTA", marcador, exc)
        return 1 if problemas else 0

    for marcador, fn in TABELAS.items():
        texto = texto.replace(marcador, fn())
    Path(args.saida).write_text(texto, encoding="utf-8")
    print("escrito", args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
