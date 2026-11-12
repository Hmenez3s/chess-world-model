#!/usr/bin/env python3
"""Gera dados de "modelo de mundo" (formato MAV) a partir do split de TREINO.

Cada linha de puzzle rende várias posições: a do solver e as intermediárias da
linha. Para cada posição, o Stockfish dá os 5 melhores lances com chance de
vitória. Metade dos exemplos pede a transição (FEN anterior + lance), metade
parte do FEN atual — o MAV também varia o comando.

Só lê o split de treino: validação e teste continuam sem vazamento.

Uso:
    python scripts/gerar_mav.py --posicoes 100000 --processos 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import chess.engine

from chess_rlvr import mav
from chess_rlvr.dados import normalizar

_MOTOR = None
_PROF = 10


def _iniciar(caminho: str, profundidade: int):
    global _MOTOR, _PROF
    _MOTOR = chess.engine.SimpleEngine.popen_uci(caminho)
    _MOTOR.configure({"Threads": 1, "Hash": 32})
    _PROF = profundidade


def _rotular(item):
    fen_anterior, lance, fen, transicao = item
    tab = chess.Board(fen)
    if tab.is_game_over():
        return None
    try:
        top = mav.avaliar_top(tab, _MOTOR, _PROF)
    except Exception:
        return None
    if not top:
        return None
    prompt = (mav.prompt_transicao(fen_anterior, lance) if transicao
              else mav.prompt_posicao(fen))
    return {"prompt": prompt, "resposta": mav.resposta(tab, top),
            "fen": fen, "melhor": top[0][0]}


def posicoes(caminho: str, limite: int, seed: int, pular: int = 0):
    """Le posicoes do split de treino em ordem, pulando as primeiras `pular`.

    A leitura e sequencial e deterministica, entao rodar de novo com
    `pular=N` continua exatamente onde a rodada anterior parou, sem sobrepor
    posicoes ja rotuladas. `seed` so decide, por posicao, se o exemplo usa o
    formato de transicao ou o de posicao direta.
    """
    rnd = random.Random(seed)
    vistos = 0
    with open(caminho, encoding="utf-8") as fh:
        for linha in fh:
            try:
                l = normalizar(json.loads(linha))
                tab = chess.Board(l["fen"])
                lances = l["moves"].split() if isinstance(l["moves"], str) else list(l["moves"])
            except Exception:
                continue
            for uci in lances:
                anterior = tab.fen()
                try:
                    tab.push_uci(uci)
                except ValueError:
                    break
                if vistos < pular:
                    vistos += 1
                    continue
                vistos += 1
                yield (anterior, uci, tab.fen(), rnd.random() < 0.5)
                limite -= 1
                if limite <= 0:
                    return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/treino.jsonl")
    ap.add_argument("--posicoes", type=int, default=100_000)
    ap.add_argument("--processos", type=int, default=12)
    ap.add_argument("--profundidade", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pular", type=int, default=0,
                    help="posicoes ja rotuladas numa rodada anterior com a mesma seed")
    ap.add_argument("--saida", default="data/mav-treino.jsonl")
    args = ap.parse_args()

    motor = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/bin/stockfish"))
    inicio = time.time()
    escritos = 0
    with Pool(args.processos, initializer=_iniciar,
              initargs=(motor, args.profundidade)) as pool, \
            open(args.saida, "w", encoding="utf-8") as fh:
        for ex in pool.imap_unordered(_rotular, posicoes(args.split, args.posicoes,
                                                        args.seed, args.pular),
                                      chunksize=64):
            if ex is None:
                continue
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            escritos += 1
            if escritos % 10_000 == 0:
                print("  %d exemplos (%.0f/s)" % (escritos, escritos / (time.time() - inicio)),
                      flush=True)
    dur = time.time() - inicio
    print("exemplos: %d em %.0fs (%.0f/s) -> %s" % (escritos, dur, escritos / dur, args.saida))
    return 0


if __name__ == "__main__":
    sys.exit(main())
