#!/usr/bin/env python3
"""Diagnóstico: a solução é um lance forçado?

O recall de candidatos medido foi 12,83% com ~3 candidatos. Se o modelo
escolhesse 3 lances ao acaso entre os legais, daria ~10% — ou seja, a geração
de candidatos está quase no nível do aleatório.

Este script mede o TETO de uma varredura sistemática:

  - que fração das soluções é xeque? captura? xeque-mate?
  - quantos xeques existem por posição? quantas capturas?
  - quantos candidatos seriam necessários para "listar todos os xeques e capturas"

Se a solução for forçada em quase todos os casos, ensinar a varrer xeques e
capturas é um currículo com teto alto e verificável. Se não for, o problema é
outro e a conclusão muda.
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
from chess_rlvr.verificador import PuzzleInvalido, montar_puzzle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="out/diagnostico-forcados.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    total = 0
    eh_xeque = 0
    eh_captura = 0
    eh_mate = 0
    eh_promocao = 0
    forcado = 0          # xeque ou captura ou promoção
    soma_legais = 0
    soma_xeques = 0
    soma_capturas = 0
    soma_forcados = 0
    recall_xeque = []    # solução coberta por "listar todos os xeques"?
    recall_captura = []
    recall_forcado = []
    por_rating: dict[str, dict] = {}

    for bruto in ds:
        if total >= args.puzzles:
            break
        l = normalizar(bruto)
        try:
            p = montar_puzzle(puzzle_id=l["puzzle_id"], fen=l["fen"], moves=l["moves"],
                              rating=l["rating"], temas=l["temas"], game_id=l["game_id"])
        except PuzzleInvalido:
            continue

        tab = p.tabuleiro
        alvo = chess.Move.from_uci(p.primeiro_lance)
        legais = list(tab.legal_moves)

        xeques = [m for m in legais if tab.gives_check(m)]
        capturas = [m for m in legais if tab.is_capture(m)]
        promos = [m for m in legais if m.promotion]
        forcs = set(xeques) | set(capturas) | set(promos)

        # a solução dá mate? (aplica e vê se o oponente tem resposta)
        tab2 = tab.copy()
        tab2.push(alvo)
        mate = tab2.is_checkmate()

        is_x = alvo in xeques
        is_c = alvo in capturas
        is_p = bool(alvo.promotion)
        is_f = is_x or is_c or is_p

        total += 1
        soma_legais += len(legais)
        soma_xeques += len(xeques)
        soma_capturas += len(capturas)
        soma_forcados += len(forcs)
        eh_xeque += is_x
        eh_captura += is_c
        eh_promocao += is_p
        eh_mate += mate
        forcado += is_f
        recall_xeque.append(is_x)
        recall_captura.append(is_c)
        recall_forcado.append(is_f)

        faixa = "%d-%d" % (p.rating // 400 * 400, p.rating // 400 * 400 + 400)
        d = por_rating.setdefault(faixa, {"n": 0, "forcado": 0, "legais": 0})
        d["n"] += 1
        d["forcado"] += is_f
        d["legais"] += len(legais)

        if total % 500 == 0:
            print("  %d puzzles" % total, flush=True)

    def pct(a, b):
        return 100.0 * a / max(1, b)

    resumo = {
        "n": total,
        "solucao_e_xeque_pct": pct(eh_xeque, total),
        "solucao_e_captura_pct": pct(eh_captura, total),
        "solucao_e_promocao_pct": pct(eh_promocao, total),
        "solucao_da_mate_pct": pct(eh_mate, total),
        "solucao_forcada_pct": pct(forcado, total),
        "media_lances_legais": soma_legais / max(1, total),
        "media_xeques": soma_xeques / max(1, total),
        "media_capturas": soma_capturas / max(1, total),
        "media_forcados": soma_forcados / max(1, total),
        "recall_listando_todos_xeques_pct": pct(sum(recall_xeque), total),
        "recall_listando_todas_capturas_pct": pct(sum(recall_captura), total),
        "recall_listando_forcados_pct": pct(sum(recall_forcado), total),
        "recall_aleatorio_3_de_legais_pct": 100.0 * 3 / (soma_legais / max(1, total)),
        "por_rating": por_rating,
    }

    print()
    print("=== A SOLUCAO E UM LANCE FORCADO? (n=%d) ===" % total)
    print("xeque                        : %5.1f%%" % resumo["solucao_e_xeque_pct"])
    print("captura                      : %5.1f%%" % resumo["solucao_e_captura_pct"])
    print("promocao                     : %5.1f%%" % resumo["solucao_e_promocao_pct"])
    print("mate imediato                : %5.1f%%" % resumo["solucao_da_mate_pct"])
    print("FORCADA (xeque|captura|promo) : %5.1f%%" % resumo["solucao_forcada_pct"])
    print()
    print("lances legais por posicao     : %.1f" % resumo["media_lances_legais"])
    print("xeques por posicao            : %.1f" % resumo["media_xeques"])
    print("capturas por posicao          : %.1f" % resumo["media_capturas"])
    print("forcados por posicao          : %.1f" % resumo["media_forcados"])
    print()
    print("=== TETO DE UMA VARREDURA SISTEMATICA ===")
    print("recall listando TODOS xeques      : %5.1f%%" % resumo["recall_listando_todos_xeques_pct"])
    print("recall listando TODAS capturas    : %5.1f%%" % resumo["recall_listando_todas_capturas_pct"])
    print("recall listando TODOS forcados    : %5.1f%%" % resumo["recall_listando_forcados_pct"])
    print("recall aleatorio (3 de %d legais) : %5.1f%%" % (
        round(resumo["media_lances_legais"]), resumo["recall_aleatorio_3_de_legais_pct"]))
    print()
    print("=== POR RATING ===")
    for f in sorted(por_rating, key=lambda x: int(x.split('-')[0])):
        d = por_rating[f]
        print("  %-10s n=%4d forcada %5.1f%% | legais %.1f" % (
            f, d["n"], pct(d["forcado"], d["n"]), d["legais"] / d["n"]))

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(resumo, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print("\nsalvo em", args.saida)
    return 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
