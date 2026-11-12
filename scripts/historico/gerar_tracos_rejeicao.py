#!/usr/bin/env python3
"""Gera dataset de raciocínio por rejection sampling (estilo STaR).

A ideia científica: um traço de raciocínio só é bom se ele LEVA à resposta certa.
Então não se inventa o traço — verifica-se.

Processo:
  1. Para cada puzzle, o Stockfish lista os melhores lances (MultiPV).
  2. O traço é construído com candidatos REAIS e avaliações MEDIDAS.
  3. O traço só é mantido se a conclusão for o lance da solução do puzzle.

Isso elimina o defeito do template anterior: lá o traço descrevia um lance já
escolhido; aqui o traço compara alternativas e a escolha é verificada.

Saída: JSONL com pares (prompt, traço) apenas dos casos corretos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chess

from chess_rlvr.dados import normalizar
from chess_rlvr.raciocinio import (
    gerar_traco_candidatos, gerar_traco_forcados, prompt_candidatos, prompt_forcados,
)
from chess_rlvr.stockfish import Motor, localizar_motor
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantidade", type=int, default=3000,
                    help="quantos puzzles tentar (o dataset final e menor)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rating-max", type=int, default=None)
    ap.add_argument("--profundidade", type=int, default=6)
    ap.add_argument("--candidatos", type=int, default=3)
    ap.add_argument("--formato", choices=["candidatos", "forcados"], default="candidatos",
                    help="forcados = varredura de xeques e capturas (recall 79,9%%)")
    ap.add_argument("--saida", default="data/tracos.jsonl")
    args = ap.parse_args()

    caminho_motor = localizar_motor()
    if caminho_motor is None:
        print("FATAL: Stockfish ausente")
        return 1
    print("motor:", caminho_motor, "depth", args.profundidade, flush=True)

    from datasets import load_dataset
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    tentados = 0
    mantidos = 0
    descartados = 0
    invalidos = 0
    inicio = time.time()

    with Motor(caminho_motor, profundidade=args.profundidade) as motor, \
            destino.open("w", encoding="utf-8") as fh:
        for bruto in ds:
            if tentados >= args.quantidade:
                break
            l = normalizar(bruto)
            try:
                p = montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"],
                                  moves=l["moves"], rating=l["rating"],
                                  temas=l["temas"], game_id=l["game_id"])
            except PuzzleInvalido:
                invalidos += 1
                continue
            if args.rating_max and p.rating > args.rating_max:
                continue

            tentados += 1
            try:
                alvo = chess.Move.from_uci(p.primeiro_lance)
                if args.formato == "forcados":
                    traco = gerar_traco_forcados(p.tabuleiro, alvo, motor)
                else:
                    traco = gerar_traco_candidatos(p.tabuleiro, alvo, motor,
                                                   k=args.candidatos)
            except Exception as exc:
                descartados += 1
                if descartados <= 3:
                    print("  falha ao gerar:", str(exc)[:80], flush=True)
                continue

            # A verificação que define o rejection sampling: a conclusão bate
            # com a solução do puzzle?
            linha_lance = [ln for ln in traco.splitlines() if ln.startswith("LANCE:")]
            if not linha_lance or linha_lance[-1].split(":", 1)[1].strip() != p.primeiro_lance:
                descartados += 1
                continue

            fh.write(json.dumps({
                "prompt": (prompt_forcados(p.tabuleiro) if args.formato == "forcados"
                           else prompt_candidatos(p.tabuleiro)),
                "resposta": traco,
                "lance": p.primeiro_lance,
                "rating": p.rating,
                "puzzle_id": p.id,
                "tokens_traco": len(traco.split()),
                "formato": args.formato,
            }, ensure_ascii=False) + "\n")
            mantidos += 1

            if mantidos % 200 == 0:
                print("  mantidos %d (tentados %d, %.0fs)" % (
                    mantidos, tentados, time.time() - inicio), flush=True)

    dur = time.time() - inicio
    print()
    print("=== REJECTION SAMPLING ===")
    print("tentados   :", tentados)
    print("mantidos   :", mantidos)
    print("descartados:", descartados)
    print("invalidos  :", invalidos)
    print("taxa de aproveitamento: %.1f%%" % (100 * mantidos / max(1, tentados)))
    print("tempo: %.0fs (%.2f s/puzzle)" % (dur, dur / max(1, tentados)))
    print("arquivo:", destino)

    if mantidos == 0:
        print("\nVEREDITO: nenhum traco mantido — investigar antes de treinar.")
        return 1
    print("\nVEREDITO: dataset de raciocinio verificado gerado.")
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
