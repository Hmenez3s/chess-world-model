#!/usr/bin/env python3
"""Valida o verificador e o espelho em puzzles reais.

Prova que:
1. `montar_puzzle` aceita linhas reais e produz um tabuleiro com o solver a jogar.
2. O espelho é um isomorfismo fiel: a solução transformada é legal na posição transformada.
3. Conta quantos puzzles precisam ser excluídos do teste de espelho e por quê.
"""

from __future__ import annotations

import collections
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.dados import normalizar
from chess_rlvr.espelho import EspelhoInvalido, espelhar_puzzle
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle


def main(amostra: int = 500) -> int:
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    ok = 0
    invalidos = collections.Counter()
    espelhados = 0
    espelho_falhou = collections.Counter()
    exemplos_falha: list[str] = []

    for i, bruto in enumerate(ds):
        if i >= amostra:
            break
        linha = normalizar(bruto)
        try:
            p = montar_puzzle(
                puzzle_id=linha["puzzle_id"],
                fen=linha["fen"],
                moves=linha["moves"],
                rating=linha["rating"],
                temas=linha["temas"],
                game_id=linha["game_id"],
            )
        except PuzzleInvalido as exc:
            invalidos[str(exc).split(":")[1].strip()[:40]] += 1
            continue
        ok += 1

        # O solver tem de ser quem joga depois do lance do oponente.
        assert p.tabuleiro.turn == p.lado
        assert p.primeiro_lance in {m.uci() for m in p.tabuleiro.legal_moves}, \
            f"{p.id}: solução ilegal no tabuleiro do solver"

        try:
            e = espelhar_puzzle(p)
        except EspelhoInvalido as exc:
            motivo = str(exc).split(":")[1].strip()[:40] if ":" in str(exc) else str(exc)[:40]
            espelho_falhou[motivo] += 1
            if len(exemplos_falha) < 3:
                exemplos_falha.append(str(exc)[:120])
            continue
        espelhados += 1

        # O espelho não pode ser idêntico ao original (senão o teste não testa nada).
        assert e.tabuleiro.fen() != p.tabuleiro.fen(), f"{p.id}: espelho idêntico"
        # A solução espelhada tem de ser legal na posição espelhada.
        assert e.primeiro_lance in {m.uci() for m in e.tabuleiro.legal_moves}

    print("=== amostra:", amostra)
    print("puzzles válidos        :", ok)
    print("espelhados com sucesso :", espelhados,
          "(%.1f%%)" % (100 * espelhados / ok if ok else 0))
    if invalidos:
        print("inválidos por motivo   :", dict(invalidos))
    if espelho_falhou:
        print("espelho rejeitado      :", dict(espelho_falhou))
        for e in exemplos_falha:
            print("   ex:", e)

    # Critério: se menos de 90% espelha, o teste de memorização fica fraco.
    taxa = espelhados / ok if ok else 0
    if taxa >= 0.90:
        print("\nVEREDITO: espelho viável como teste (>=90% dos puzzles espelham).")
        return 0
    print("\nVEREDITO: taxa de espelho baixa (%.1f%%). Revisar antes de usar como métrica." % (100 * taxa))
    return 1


if __name__ == "__main__":
    arg = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    codigo = main(arg)
    sys.stdout.flush()
    os._exit(codigo)
