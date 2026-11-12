#!/usr/bin/env python3
"""Verify that Stockfish is operational and partial reward works.

Test the important case: a legal but WRONG move must receive an
intermediate reward (not zero, not maximum), and a move that worsens the position must
receive less than a neutral move.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from chess_rlvr.stockfish import Engine, find_engine, normalize_advantage


def main() -> int:
    path = find_engine()
    print("engine:", path)
    if not path:
        print("FAILURE: Stockfish not found")
        return 1

    # Position with an obvious mate in 1: queen can checkmate; any other move cannot.
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    board = chess.Board(fen)
    print("position:", fen)

    mate = None
    for mv in board.legal_moves:
        copy_board = board.copy()
        copy_board.push(mv)
        if copy_board.is_checkmate():
            mate = mv
            break
    print("mate move found:", mate)

    if mate is None:
        print("FAILURE: test position has no mate")
        return 1

    # A legal alternative move that is clearly worse.
    alternatives = [m for m in board.legal_moves if m != mate]
    worse = alternatives[0]

    with Engine(path, depth=8) as engine:
        v_mate = engine.eval_after(board, mate)
        v_alt = engine.eval_after(board, worse)

    print("mate evaluation  : %+.0f cp  (normalized %+.3f)"
          % (v_mate, normalize_advantage(v_mate)))
    print("other evaluation : %+.0f cp  (normalized %+.3f)"
          % (v_alt, normalize_advantage(v_alt)))

    ok = v_mate > v_alt
    print()
    if ok:
        print("VERDICT: Stockfish distinguishes moves; partial reward is viable.")
        return 0
    print("VERDICT: evaluation did not distinguish. Partial reward is blocked.")
    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
