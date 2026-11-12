#!/usr/bin/env python3
"""Validate the verifier and mirror on real puzzles.

Prove that:
1. `build_puzzle` accepts real rows and produces a board with the solver to move.
2. The mirror is a faithful isomorphism: the transformed solution is legal in the transformed position.
3. Count how many puzzles must be excluded from the mirror test and why.
"""

from __future__ import annotations

import collections
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.data import normalize
from chess_rlvr.mirror import InvalidMirror, mirror_puzzle
from chess_rlvr.verifier import InvalidPuzzle, build_puzzle


def main(sample: int = 500) -> int:
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    ok = 0
    invalid = collections.Counter()
    mirrored = 0
    mirror_failed = collections.Counter()
    failure_examples: list[str] = []

    for i, raw in enumerate(ds):
        if i >= sample:
            break
        row = normalize(raw)
        try:
            p = build_puzzle(
                puzzle_id=row["puzzle_id"],
                fen=row["fen"],
                moves=row["moves"],
                rating=row["rating"],
                themes=row["themes"],
                game_id=row["game_id"],
            )
        except InvalidPuzzle as exc:
            invalid[str(exc).split(":")[1].strip()[:40]] += 1
            continue
        ok += 1

        # The solver must be to move after the opponent move.
        assert p.board.turn == p.side
        assert p.first_move in {m.uci() for m in p.board.legal_moves}, \
            f"{p.id}: illegal solution on the solver board"

        try:
            e = mirror_puzzle(p)
        except InvalidMirror as exc:
            reason = str(exc).split(":")[1].strip()[:40] if ":" in str(exc) else str(exc)[:40]
            mirror_failed[reason] += 1
            if len(failure_examples) < 3:
                failure_examples.append(str(exc)[:120])
            continue
        mirrored += 1

        # The mirror must not be identical to the original (otherwise the test proves nothing).
        assert e.board.fen() != p.board.fen(), f"{p.id}: mirror identical"
        # The mirrored solution must be legal in the mirrored position.
        assert e.first_move in {m.uci() for m in e.board.legal_moves}

    print("=== sample:", sample)
    print("puzzles valid        :", ok)
    print("mirrored successfully :", mirrored,
          "(%.1f%%)" % (100 * mirrored / ok if ok else 0))
    if invalid:
        print("invalid by reason   :", dict(invalid))
    if mirror_failed:
        print("mirror rejected      :", dict(mirror_failed))
        for e in failure_examples:
            print("   ex:", e)

    # Criterion: if fewer than 90% mirror successfully, the memorization test is weak.
    rate = mirrored / ok if ok else 0
    if rate >= 0.90:
        print("\nVERDICT: mirror viable as a test (>=90% of puzzles mirror).")
        return 0
    print("\nVERDICT: mirror rate is low (%.1f%%). Review before using as a metric." % (100 * rate))
    return 1


if __name__ == "__main__":
    arg = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    exit_code = main(arg)
    sys.stdout.flush()
    os._exit(exit_code)
