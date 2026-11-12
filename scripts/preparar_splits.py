#!/usr/bin/env python3
"""Materializa os splits do dataset Lichess com prova de não-vazamento.

Varre o dataset completo em streaming (6,1M de puzzles), filtra, divide por GameId
estratificando por (motivo × faixa de rating), e grava:
  - os três arquivos JSONL de split
  - um manifesto com contagens, auditoria de vazamento e cobertura de estratos

O manifesto é a prova: sem ele, qualquer alegação sobre "split sem vazamento" é fé.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.dados import (
    auditar_estratos,
    auditar_vazamento,
    dividir_estratificado,
    faixa_de_rating,
    filtrar,
    motivo_de,
    normalizar,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0,
                    help="0 = dataset inteiro; >0 limita a varredura (amostragem)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="data/splits")
    ap.add_argument("--popularidade-min", type=int, default=80)
    ap.add_argument("--partidas-min", type=int, default=100)
    args = ap.parse_args()

    from datasets import load_dataset

    print("varrendo dataset...", flush=True)
    inicio = time.time()
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    brutos = []
    vistos = 0
    for row in ds:
        vistos += 1
        brutos.append(normalizar(row))
        if vistos % 200_000 == 0:
            print("  varridos %d (%.1fs)" % (vistos, time.time() - inicio), flush=True)
        if args.limite and vistos >= args.limite:
            break
    print("total varrido: %d em %.1fs" % (vistos, time.time() - inicio), flush=True)

    filtros = {"popularidade_min": args.popularidade_min,
               "partidas_min": args.partidas_min,
               "rating_min": 400, "rating_max": 3000}
    limpos = filtrar(brutos, **filtros)
    print("apos filtro: %d (%.1f%% do varrido)" % (len(limpos), 100 * len(limpos) / max(1, vistos)),
          flush=True)

    div = dividir_estratificado(limpos, seed=args.seed)
    print("divisoes:", div.resumo(), flush=True)

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)

    def gravar(nome, linhas):
        caminho = destino / (nome + ".jsonl")
        with caminho.open("w", encoding="utf-8") as fh:
            for l in linhas:
                fh.write(json.dumps({
                    "puzzle_id": l["puzzle_id"], "game_id": l["game_id"],
                    "fen": l["fen"], "moves": l["moves"], "rating": l["rating"],
                    "temas": list(l["temas"]),
                    "motivo": motivo_de(l["temas"]),
                    "faixa": faixa_de_rating(l["rating"]),
                }, ensure_ascii=False) + "\n")
        return caminho.stat().st_size

    tamanhos = {
        "treino": gravar("treino", div.treino),
        "validacao": gravar("validacao", div.validacao),
        "teste": gravar("teste", div.teste),
    }

    vaz = auditar_vazamento(div)
    est = auditar_estratos(div)

    manifesto = {
        "fonte": "Lichess/chess-puzzles",
        "seed": args.seed,
        "varridos": vistos,
        "apos_filtro": len(limpos),
        "filtros": filtros,
        "divisoes": div.resumo(),
        "bytes": tamanhos,
        "vazamento": vaz,
        "estratos": est,
        "sem_vazamento": (len(vaz["jogos_compartilhados"]) == 0
                          and len(vaz["puzzles_compartilhados"]) == 0),
        "tempo_s": round(time.time() - inicio, 1),
    }
    (destino / "manifesto.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== AUDITORIA DE VAZAMENTO ===")
    print("jogos compartilhados  :", len(vaz["jogos_compartilhados"]))
    print("puzzles compartilhados:", len(vaz["puzzles_compartilhados"]))
    print("SEM VAZAMENTO         :", "SIM" if manifesto["sem_vazamento"] else "NAO")
    print()
    print("=== COBERTURA DE ESTRATOS ===")
    print("motivos no treino:", est["motivos_treino"], "| no teste:", est["motivos_teste"])
    print("motivos so no treino:", est["motivos_so_treino"])
    print("faixas no teste:", est["faixas_teste"])
    print()
    print("manifesto:", destino / "manifesto.json")

    if not manifesto["sem_vazamento"]:
        print("\nVEREDITO: FALHA — ha vazamento entre divisoes.")
        return 1
    print("\nVEREDITO: splits sem vazamento, estratificados por motivo e rating.")
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
