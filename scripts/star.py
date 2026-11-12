#!/usr/bin/env python3
"""Self-taught reasoning (STaR / rejection-sampling fine-tuning), one iteration.

The model's world-model trace (board -> legal moves -> move values -> best move)
is its reasoning. pass@6 on the test set is 42.5% against 18% greedy: the right
move is often *sampled* but not *chosen*. This script turns those lucky samples
into training data:

  1. sample k answers per training puzzle at temperature T;
  2. keep an answer only if its own trace is consistent and correct:
       - the board it wrote is exactly the real board after the move, and
       - its raw `best:` is the puzzle solution (no verification fallback);
  3. write the kept traces as SFT examples (the model's own words).

Puzzles come from the train split, after the lines already used to build the
world-model data, so the test split is never touched.

Usage:
    python scripts/star.py --adapter out/mav-final --puzzles 1000 --k 4 \
        --out data/star-iter1.jsonl --report out/star-iter1-sampling.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_rlvr import mav
from chess_rlvr.data import normalize


def load_train_puzzles(path: str, skip_lines: int, n: int, seed: int) -> list[dict]:
    if not os.path.exists(path):
        legacy = path.replace("train.jsonl", "treino.jsonl")
        if os.path.exists(legacy):
            path = legacy
    lines = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < skip_lines:
                continue
            lines.append(line)
            if len(lines) >= 20 * n:
                break
    random.Random(seed).shuffle(lines)
    puzzles = []
    for line in lines:
        if len(puzzles) >= n:
            break
        try:
            row = normalize(json.loads(line))
            moves = row["moves"].split() if isinstance(row["moves"], str) else list(row["moves"])
            board = chess.Board(row["fen"])
            board.push_uci(moves[0])
        except Exception:
            continue
        puzzles.append({"fen0": row["fen"], "m0": moves[0], "board": board,
                        "solution": moves[1], "rating": row["rating"]})
    return puzzles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="data/splits/train.jsonl")
    ap.add_argument("--skip-lines", type=int, default=40000,
                    help="train-split lines already used for the world-model data")
    ap.add_argument("--puzzles", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--keep-per-puzzle", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=440)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/star-iter1.jsonl")
    ap.add_argument("--report", default="out/star-iter1-sampling.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    puzzles = load_train_puzzles(args.split, args.skip_lines, args.puzzles, args.seed)
    print("train puzzles:", len(puzzles), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = [mav.transition_prompt(p["fen0"], p["m0"]) for p in puzzles]
    expanded = [(i, p) for i, p in enumerate(prompts) for _ in range(args.k)]
    texts: list[str] = []
    for start in range(0, len(expanded), args.batch):
        chunk = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True)
                 for _i, p in expanded[start:start + args.batch]]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=True,
                                 temperature=args.temperature, pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            texts.append(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True))
        print("  sampled %d/%d" % (len(texts), len(expanded)), flush=True)

    kept, correct_samples, solved_any, consistent_board = [], 0, 0, 0
    for i, puzzle in enumerate(puzzles):
        truth = mav.board_grid(puzzle["board"])
        hits = []
        for text in texts[i * args.k:(i + 1) * args.k]:
            fields = mav.extract_fields(text)
            if fields["board"] == truth:
                consistent_board += 1
            if fields["best"] == puzzle["solution"]:
                correct_samples += 1
                if fields["board"] == truth:
                    hits.append(text)
        if hits:
            solved_any += 1
            for text in list(dict.fromkeys(hits))[:args.keep_per_puzzle]:
                kept.append({"prompt": prompts[i], "response": text.strip(),
                             "rating": puzzle["rating"]})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(puzzles) * args.k
    report = {"puzzles": len(puzzles), "k": args.k, "temperature": args.temperature,
              "samples": total, "correct_samples": correct_samples,
              "sample_accuracy_pct": 100 * correct_samples / max(1, total),
              "board_exact_pct": 100 * consistent_board / max(1, total),
              "puzzles_with_kept_trace": solved_any,
              "pass_at_k_pct": 100 * solved_any / max(1, len(puzzles)),
              "kept_examples": len(kept)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
