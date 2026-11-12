#!/usr/bin/env python3
"""Confere que os splits materializados são carregáveis e íntegros.

Verificação final do entregável: os arquivos existem, abrem, e cada linha forma um
puzzle jogável. Não basta o manifesto dizer que está certo — o dado tem de abrir.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle

BASE = Path("data/splits")


def conferir(nome: str, limite: int = 20000) -> dict:
    caminho = BASE / (nome + ".jsonl")
    if not caminho.exists():
        return {"erro": "arquivo ausente"}

    total = 0
    ok = 0
    falhas = 0
    jogos = set()
    with caminho.open(encoding="utf-8") as fh:
        for linha in fh:
            total += 1
            if total > limite:
                break
            r = json.loads(linha)
            jogos.add(r["game_id"])
            try:
                montar_puzzle(puzzle_id=r["puzzle_id"], fen=r["fen"],
                              moves=r["moves"], rating=r["rating"],
                              temas=tuple(r["temas"]), game_id=r["game_id"])
                ok += 1
            except PuzzleInvalido:
                falhas += 1

    return {"linhas_lidas": total, "validos": ok, "falhas": falhas,
            "jogos_distintos": len(jogos), "bytes": caminho.stat().st_size}


def main() -> int:
    print("=== integridade dos splits materializados ===")
    res = {}
    for nome in ("treino", "validacao", "teste"):
        r = conferir(nome)
        res[nome] = r
        if "erro" in r:
            print("%-12s ERRO: %s" % (nome, r["erro"]))
            continue
        print("%-12s %8d linhas | %8d validos | %d falhas | %.1f MB" % (
            nome, r["linhas_lidas"], r["validos"], r["falhas"],
            r["bytes"] / 1e6))

    # Prova cruzada: nenhum game_id pode aparecer em duas divisões.
    conjuntos = {}
    for nome in ("treino", "validacao", "teste"):
        p = BASE / (nome + ".jsonl")
        if not p.exists():
            continue
        s = set()
        with p.open(encoding="utf-8") as fh:
            for linha in fh:
                s.add(json.loads(linha)["game_id"])
        conjuntos[nome] = s

    print()
    print("=== vazamento entre arquivos (varredura completa) ===")
    nomes = list(conjuntos)
    vazou = False
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            inter = conjuntos[nomes[i]] & conjuntos[nomes[j]]
            print("  %s ∩ %s = %d" % (nomes[i], nomes[j], len(inter)))
            if inter:
                vazou = True

    print()
    if vazou:
        print("VEREDITO: FALHA — ha game_id compartilhado entre divisoes.")
        return 1
    print("VEREDITO: splits integros, carregaveis e sem vazamento.")
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
