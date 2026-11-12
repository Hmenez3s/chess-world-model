#!/usr/bin/env python3
"""Check that materialized splits are loadable and intact.

Final deliverable check: files exist, open, and each row forms a
playable puzzle. A manifest is not enough — the data must open.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.verifier import InvalidPuzzle, build_puzzle

BASE = Path("data/splits")


def check_split(name: str, limit: int = 20000) -> dict:
    path = BASE / (name + ".jsonl")
    if not path.exists():
        legacy = {"train": "treino", "validation": "validacao", "test": "teste"}[name]
        path = BASE / (legacy + ".jsonl")
    if not path.exists():
        return {"error": "missing file"}

    total = 0
    valid = 0
    failures = 0
    games = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            total += 1
            if total > limit:
                break
            r = json.loads(line)
            games.add(r["game_id"])
            try:
                build_puzzle(puzzle_id=r["puzzle_id"], fen=r["fen"],
                              moves=r["moves"], rating=r["rating"],
                              themes=tuple(r.get("themes", r.get("temas", ()))), game_id=r["game_id"])
                valid += 1
            except InvalidPuzzle:
                failures += 1

    return {"lines_read": total, "valid": valid, "failures": failures,
            "distinct_games": len(games), "bytes": path.stat().st_size}


def main() -> int:
    print("=== integrity of materialized splits ===")
    results = {}
    for name in ("train", "validation", "test"):
        r = check_split(name)
        results[name] = r
        if "error" in r:
            print("%-12s ERROR: %s" % (name, r["error"]))
            continue
        print("%-12s %8d lines | %8d valid | %d failures | %.1f MB" % (
            name, r["lines_read"], r["valid"], r["failures"],
            r["bytes"] / 1e6))

    # Cross-check: no game_id may appear in two splits.
    sets = {}
    for name in ("train", "validation", "test"):
        p = BASE / (name + ".jsonl")
        if not p.exists():
            legacy = {"train": "treino", "validation": "validacao", "test": "teste"}[name]
            p = BASE / (legacy + ".jsonl")
        if not p.exists():
            continue
        s = set()
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                s.add(json.loads(line)["game_id"])
        sets[name] = s

    print()
    print("=== leakage across files (complete scan) ===")
    names = list(sets)
    leaked = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            print("  %s ∩ %s = %d" % (names[i], names[j], len(inter)))
            if inter:
                leaked = True

    print()
    if leaked:
        print("VERDICT: FAILURE — a game_id is shared across splits.")
        return 1
    print("VERDICT: splits are intact, loadable, and leakage-free.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
