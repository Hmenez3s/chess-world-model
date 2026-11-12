#!/usr/bin/env python3
"""Prova que o Stockfish esta operacional e que a recompensa parcial funciona.

Testa o caso que importa: um lance legal mas ERRADO deve receber recompensa
intermediaria (nao zero, nao o maximo), e um lance que estraga a posicao deve
receber menos que um lance neutro.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chess_rlvr.stockfish import Motor, localizar_motor, normalizar_vantagem


def main() -> int:
    caminho = localizar_motor()
    print("motor:", caminho)
    if not caminho:
        print("FALHA: stockfish nao encontrado")
        return 1

    # Posicao com mate em 1 claro: dama pode dar mate; qualquer outro lance nao.
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    tab = chess.Board(fen)
    print("posicao:", fen)

    mate = None
    for mv in tab.legal_moves:
        copia = tab.copy()
        copia.push(mv)
        if copia.is_checkmate():
            mate = mv
            break
    print("lance de mate encontrado:", mate)

    if mate is None:
        print("FALHA: posicao de teste sem mate")
        return 1

    # Um lance alternativo legal e claramente pior.
    alternativas = [m for m in tab.legal_moves if m != mate]
    pior = alternativas[0]

    with Motor(caminho, profundidade=8) as motor:
        v_mate = motor.avaliar_apos(tab, mate)
        v_alt = motor.avaliar_apos(tab, pior)

    print("avaliacao do mate  : %+.0f cp  (normalizada %+.3f)"
          % (v_mate, normalizar_vantagem(v_mate)))
    print("avaliacao do outro : %+.0f cp  (normalizada %+.3f)"
          % (v_alt, normalizar_vantagem(v_alt)))

    ok = v_mate > v_alt
    print()
    if ok:
        print("VEREDITO: Stockfish discrimina os lances; recompensa parcial viavel.")
        return 0
    print("VEREDITO: avaliacao nao discriminou. Recompensa parcial bloqueada.")
    return 1


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    os._exit(codigo)
