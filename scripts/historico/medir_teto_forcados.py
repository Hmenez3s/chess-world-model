#!/usr/bin/env python3
"""Quanto recall se perde ao limitar a lista de forçados?

A varredura completa cobre 80,2% das soluções, mas custa ~3,9 candidatos em
média — e algumas posições têm muitas capturas. Este script mede o recall para
diferentes tetos, para escolher o menor orçamento que ainda capture quase tudo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chess

from chess_rlvr.dados import normalizar
from chess_rlvr.raciocinio import lances_forcados
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle

TETOS = [3, 4, 5, 6, 8, 12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rating-max", type=int, default=None)
    ap.add_argument("--saida", default="out/teto-forcados.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    cobertura = {t: 0 for t in TETOS}
    soma_itens = {t: 0 for t in TETOS}
    total = 0
    for bruto in ds:
        if total >= args.puzzles:
            break
        l = normalizar(bruto)
        try:
            p = montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                              rating=l["rating"], temas=l["temas"], game_id=l["game_id"])
        except PuzzleInvalido:
            continue
        if args.rating_max and p.rating > args.rating_max:
            continue

        alvo = chess.Move.from_uci(p.primeiro_lance)
        total += 1
        for t in TETOS:
            lista = lances_forcados(p.tabuleiro, maximo=t)
            soma_itens[t] += len(lista)
            if alvo in lista:
                cobertura[t] += 1
        if total % 500 == 0:
            print("  %d" % total, flush=True)

    print()
    print("=== RECALL POR TETO DA LISTA DE FORCADOS (n=%d) ===" % total)
    print("%-6s %-10s %-12s" % ("teto", "recall", "itens/traço"))
    for t in TETOS:
        print("%-6d %6.1f%%    %8.2f" % (t, 100 * cobertura[t] / max(1, total),
                                         soma_itens[t] / max(1, total)))

    resumo = {"n": total, "por_teto": {
        str(t): {"recall_pct": 100 * cobertura[t] / max(1, total),
                 "itens_por_traco": soma_itens[t] / max(1, total)} for t in TETOS}}
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resumo, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print("\nsalvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
