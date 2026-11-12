#!/usr/bin/env python3
"""Generate "world model" data (MAV format) from the TRAIN split.

Each puzzle row yields several positions: the solver position and intermediate positions in the
line. For each position, Stockfish returns the 5 best moves with win
probabilities. Half the examples request a transition (previous FEN + move), half
start from the current FEN — MAV also varies the instruction.

Only reads the training split: validation and test remain leakage-free.

Usage:
    python scripts/generate_mav.py --positions 100000 --workers 12
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
from chess_rlvr.data import normalize

_MOTOR = None
_PROF = 10


def _initialize(path: str, depth: int):
    global _MOTOR, _PROF
    _MOTOR = chess.engine.SimpleEngine.popen_uci(path)
    _MOTOR.configure({"Threads": 1, "Hash": 32})
    _PROF = depth


def _label(item):
    previous_fen, move, fen, transition = item
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    try:
        top = mav.evaluate_top(board, _MOTOR, _PROF)
    except Exception:
        return None
    if not top:
        return None
    prompt = (mav.transition_prompt(previous_fen, move) if transition
              else mav.position_prompt(fen))
    return {"prompt": prompt, "response": mav.response(board, top),
            "fen": fen, "best_move": top[0][0]}


def positions(path: str, limit: int, seed: int, skip: int = 0):
    """Read positions from the training split in order, skipping the first `skip`.

    Reading is sequential and deterministic, so rerunning with
    `skip=N` resumes exactly where the previous run stopped, without overlapping
    positions already labeled. `seed` only decides, per position, whether the example uses the
    transition format or the direct-position format.
    """
    rnd = random.Random(seed)
    seen = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                l = normalize(json.loads(line))
                board = chess.Board(l["fen"])
                moves = l["moves"].split() if isinstance(l["moves"], str) else list(l["moves"])
            except Exception:
                continue
            for uci in moves:
                previous = board.fen()
                try:
                    board.push_uci(uci)
                except ValueError:
                    break
                if seen < skip:
                    seen += 1
                    continue
                seen += 1
                yield (previous, uci, board.fen(), rnd.random() < 0.5)
                limit -= 1
                if limit <= 0:
                    return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/train.jsonl")
    ap.add_argument("--positions", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", type=int, default=0,
                    help="positions already labeled in a previous run with the same seed")
    ap.add_argument("--out", default="data/mav-train.jsonl")
    args = ap.parse_args()

    engine = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/bin/stockfish"))
    start_time = time.time()
    written = 0
    with Pool(args.workers, initializer=_initialize,
              initargs=(engine, args.depth)) as pool, \
            open(args.out, "w", encoding="utf-8") as fh:
        for ex in pool.imap_unordered(_label, positions((args.split if os.path.exists(args.split) else "data/splits/treino.jsonl"), args.positions,
                                                        args.seed, args.skip),
                                      chunksize=64):
            if ex is None:
                continue
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
            if written % 10_000 == 0:
                print("  %d examples (%.0f/s)" % (written, written / (time.time() - start_time)),
                      flush=True)
    duration = time.time() - start_time
    print("examples: %d in %.0fs (%.0f/s) -> %s" % (written, duration, written / duration, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
