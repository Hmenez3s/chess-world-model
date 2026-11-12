#!/usr/bin/env python3
"""Evaluate the world model (MAV format) on the TEST split.

Measure each layer separately in the same response:
  - state: is the board_grid after the previous move exact, square by square
           (including color, via uppercase/lowercase)?
  - rules: is the legal-move set exact? What is its F1?
  - decision: is the best move legal? Is it the puzzle solution?

The puzzle is already in transition format: previous FEN + opponent move.
"""

from __future__ import annotations

import argparse
import json
import math
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
from chess_rlvr.mirror import mirror_board, mirror_uci
from chess_rlvr.verifier import build_prompt, normalize_uci


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, c - m), 100 * min(1.0, c + m))


def load(path: str, n: int, seed: int):
    if not os.path.exists(path) and path == "data/splits/test.jsonl":
        path = "data/splits/teste.jsonl"
    lines = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 50_000:
                break
            lines.append(line)
    random.Random(seed).shuffle(lines)
    items = []
    for line in lines:
        if len(items) >= n:
            break
        try:
            row = normalize(json.loads(line))
            moves = row["moves"].split() if isinstance(row["moves"], str) else list(row["moves"])
            board = chess.Board(row["fen"])
            board.push_uci(moves[0])
        except Exception:
            continue
        items.append({"fen0": row["fen"], "m0": moves[0], "board": board,
                      "solution": moves[1], "rating": row["rating"]})
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="No adapter means the unmodified base model")
    ap.add_argument("--split", default="data/splits/test.jsonl")
    ap.add_argument("--puzzles", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=440)
    ap.add_argument("--format", choices=["mav", "direct"], default="mav",
                    help="direct = older move-only adapters, using the same test puzzles")
    ap.add_argument("--save-text", action="store_true",
                    help="save each item's complete response for consistency analysis")
    ap.add_argument("--verify", action="store_true",
                    help="validate the final move and fall back to the model's top-5/legal moves")
    ap.add_argument("--mirror", action="store_true",
                    help="evaluate the same puzzles after vertical mirroring")
    ap.add_argument("--out", default="out/eval-mav.json")
    args = ap.parse_args()

    items = load(args.split, args.puzzles, args.seed)
    mirror_discarded = 0
    if args.mirror:
        mirrored = []
        for item in items:
            initial_board = mirror_board(chess.Board(item["fen0"]))
            m0 = mirror_uci(item["m0"])
            solution = mirror_uci(item["solution"])
            # Compare Move objects; testing a string against legal_moves raises AttributeError.
            if chess.Move.from_uci(m0) not in initial_board.legal_moves:
                mirror_discarded += 1
                continue
            board = initial_board.copy()
            board.push_uci(m0)
            if chess.Move.from_uci(solution) not in board.legal_moves:
                mirror_discarded += 1
                continue
            mirrored.append({**item, "fen0": initial_board.fen(), "m0": m0,
                             "solution": solution, "board": board})
        items = mirrored
    if args.format == "direct":
        args.max_tokens = 16
    print("test puzzles:", len(items), flush=True)
    if args.mirror:
        print("mirror_discarded:", mirror_discarded, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    texts = []
    for i in range(0, len(items), args.batch):
        batch = items[i:i + args.batch]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": (mav.transition_prompt(x["fen0"], x["m0"])
                                          if args.format == "mav" else build_prompt(x["board"]))}],
            tokenize=False, add_generation_prompt=True) for x in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            texts.append(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True))
        print("  %d/%d" % (len(texts), len(items)), flush=True)

    counts = {"board_exact": 0, "squares_correct": 0, "squares_total": 0,
              "color_correct": 0, "color_total": 0, "legal_exact": 0, "f1_sum": 0.0,
              "best_legal": 0, "best_correct": 0}
    samples = []
    records = []
    for x, text in zip(items, texts):
        board = x["board"]
        fields = (mav.extract_fields(text) if args.format == "mav"
                  else {"board": None, "legal": None, "best": normalize_uci(text)})
        expected_grid = mav.board_grid(board)
        if fields["board"] == expected_grid:
            counts["board_exact"] += 1
        if fields["board"]:
            for predicted, expected in zip(fields["board"].splitlines()[:8], expected_grid.splitlines()[:8]):
                for a, b in zip(predicted.split()[1:], expected.split()[1:]):
                    counts["squares_total"] += 1
                    counts["squares_correct"] += a == b
                    if b != ".":
                        counts["color_total"] += 1
                        counts["color_correct"] += a != "." and a.isupper() == b.isupper()
        legal_moves = {m.uci() for m in board.legal_moves}
        predicted_legal = fields["legal"] or set()
        if predicted_legal == legal_moves:
            counts["legal_exact"] += 1
        tp = len(predicted_legal & legal_moves)
        if predicted_legal and tp:
            precision, recall = tp / len(predicted_legal), tp / len(legal_moves)
            counts["f1_sum"] += 2 * precision * recall / (precision + recall)
        best = fields["best"]
        if args.verify:
            best, _source = mav.verified_move(text, board)
        if best and best in legal_moves:
            counts["best_legal"] += 1
        correct = best == x["solution"]
        legal = bool(best and best in legal_moves)
        if correct:
            counts["best_correct"] += 1
        record = {"rating": int(x["rating"]), "solution": x["solution"],
                  "best": best, "correct": correct, "legal": legal,
                  "board_exact": (fields["board"] == expected_grid
                                  if args.format == "mav" else None)}
        if args.save_text:
            record["text"] = text
        records.append(record)
        if len(samples) < 3:
            samples.append({"expected": x["solution"], "text": text[:700]})

    n = len(items)
    result = {
        "n": n, "adapter": args.adapter, "verified": args.verify,
        "items": records,
        "board_exact_pct": 100 * counts["board_exact"] / n,
        "squares_correct_pct": 100 * counts["squares_correct"] / max(1, counts["squares_total"]),
        "color_correct_pct": 100 * counts["color_correct"] / max(1, counts["color_total"]),
        "legal_exact_pct": 100 * counts["legal_exact"] / n,
        "legal_f1_mean": counts["f1_sum"] / n,
        "best_legal_pct": 100 * counts["best_legal"] / n,
        "best_correct": counts["best_correct"],
        "best_correct_pct": 100 * counts["best_correct"] / n,
        "best_correct_ci95": wilson(counts["best_correct"], n),
        "samples": samples,
        "records": records,
    }
    if args.mirror:
        result["mirror_discarded"] = mirror_discarded
    print()
    print("=== WORLD MODEL (test, n=%d) ===" % n)
    print("state   exact board_grid      : %6.2f%%" % result["board_exact_pct"])
    print("state   correct squares       : %6.2f%%" % result["squares_correct_pct"])
    print("state   correct color         : %6.2f%%" % result["color_correct_pct"])
    print("rules   exact legal moves     : %6.2f%%" % result["legal_exact_pct"])
    print("rules   mean legal-move F1    : %6.3f" % result["legal_f1_mean"])
    print("decision best move is legal  : %6.2f%%" % result["best_legal_pct"])
    print("decision best = solution     : %d/%d = %.2f%% CI95 [%.2f; %.2f]" % (
        counts["best_correct"], n, result["best_correct_pct"], *result["best_correct_ci95"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved to", args.out)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
