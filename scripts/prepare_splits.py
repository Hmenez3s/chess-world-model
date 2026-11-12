#!/usr/bin/env python3
"""Materialize Lichess dataset splits with leakage-proof evidence.

Stream the full dataset (6.1M puzzles), filter, split by GameId
stratifying by (motif × rating band), and write:
  - the three split JSONL files
  - a manifest with counts, leakage audit, and stratum coverage

The manifest is the evidence: without it, any claim of a "leakage-free split" is faith.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.data import (
    audit_strata,
    audit_leakage,
    stratified_split,
    rating_band,
    filter_rows,
    motif_of,
    write_manifest,
    normalize,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full dataset; >0 limits the scan (sampling)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--min-popularity", type=int, default=80)
    ap.add_argument("--min-plays", type=int, default=100)
    args = ap.parse_args()

    from datasets import load_dataset

    print("Scanning dataset...", flush=True)
    start_time = time.time()
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    raw_rows = []
    scanned = 0
    for row in ds:
        scanned += 1
        raw_rows.append(normalize(row))
        if scanned % 200_000 == 0:
            print("  scanned %d (%.1fs)" % (scanned, time.time() - start_time), flush=True)
        if args.limit and scanned >= args.limit:
            break
    print("total scanned: %d in %.1fs" % (scanned, time.time() - start_time), flush=True)

    filters = {"min_popularity": args.min_popularity,
               "min_plays": args.min_plays,
               "rating_min": 400, "rating_max": 3000}
    filtered_rows = filter_rows(raw_rows, **filters)
    print("after filtering: %d (%.1f%% of scanned)" % (len(filtered_rows), 100 * len(filtered_rows) / max(1, scanned)),
          flush=True)

    div = stratified_split(filtered_rows, seed=args.seed)
    print("splits:", div.summary(), flush=True)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name, rows):
        path = output_dir / (name + ".jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for l in rows:
                fh.write(json.dumps({
                    "puzzle_id": l["puzzle_id"], "game_id": l["game_id"],
                    "fen": l["fen"], "moves": l["moves"], "rating": l["rating"],
                    "themes": list(l["themes"]),
                    "motif": motif_of(l["themes"]),
                    "band": rating_band(l["rating"]),
                }, ensure_ascii=False) + "\n")
        return path.stat().st_size

    sizes = {
        "train": write_split("train", div.train),
        "validation": write_split("validation", div.validation),
        "test": write_split("test", div.test),
    }

    leakage = audit_leakage(div)
    strata = audit_strata(div)

    manifest = write_manifest(div, output_dir, source="Lichess/chess-puzzles",
                              filters=filters, seed=args.seed)
    manifest["scanned"] = scanned
    manifest["after_filter"] = len(filtered_rows)
    manifest["bytes"] = sizes
    manifest["strata"] = strata
    manifest["leakage_free"] = (len(leakage["shared_games"]) == 0
                                and len(leakage["shared_puzzles"]) == 0)
    manifest["elapsed_seconds"] = round(time.time() - start_time, 1)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== LEAKAGE AUDIT ===")
    print("shared games  :", len(leakage["shared_games"]))
    print("shared puzzles:", len(leakage["shared_puzzles"]))
    print("LEAKAGE FREE         :", "YES" if manifest["leakage_free"] else "NO")
    print()
    print("=== STRATUM COVERAGE ===")
    print("motifs in train:", strata["train_motifs"], "| in test:", strata["test_motifs"])
    print("train-only motifs:", strata["train_only_motifs"])
    print("bands in test:", strata["test_bands"])
    print()
    print("manifest:", output_dir / "manifest.json")

    if not manifest["leakage_free"]:
        print("\nVERDICT: FAILURE — leakage exists between splits.")
        return 1
    print("\nVERDICT: splits are leakage-free, stratified by motif and rating.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
