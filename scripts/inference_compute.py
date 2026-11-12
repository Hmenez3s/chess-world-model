#!/usr/bin/env python3
"""Can the model think ahead using its OWN world model? (without training)

Measure three things on the same test puzzles:

  1. greedy — normal output (top-5 → best), with a verified move.
  2. pass@k — does the solution appear in any of k samples? This is the ceiling
     RL/STaR can unlock (RLVR reorganizes probability within pass@k; it cannot
     create outcomes that are absent).
  3. depth-2 search — minimax using only the model: apply each top-5 move, ask
     the model for the opponent position's top-5, then use its best value.
     move_value = 100 − opponent's best value. No Stockfish inference calls.

If (3) beats (1), the world model can already think ahead; the next step is to
amortize that search during training (search traces + STaR).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import chess
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import load, wilson
from chess_rlvr import mav


def parse_top5(text: str) -> list[tuple[str, int]]:
    for line in (text or "").splitlines():
        if line.startswith("top"):
            moves = []
            for item in line.split(":", 1)[1].split():
                uci, _, value = item.partition(":")
                try:
                    moves.append((uci.strip().lower(), int(value)))
                except ValueError:
                    pass
            return moves
    return []


def generate(model, tok, prompts, batch, max_tokens, sample=False, temperature=0.8):
    texts = []
    for i in range(0, len(prompts), batch):
        chunk = [tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                        add_generation_prompt=True)
                 for prompt in prompts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        options = {"max_new_tokens": max_tokens, "do_sample": sample,
                   "pad_token_id": tok.eos_token_id}
        if sample:
            options["temperature"] = temperature
        with torch.no_grad():
            out = model.generate(**enc, **options)
        for j in range(len(chunk)):
            texts.append(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True))
        print("  generated %d/%d" % (len(texts), len(prompts)), flush=True)
    return texts


def value_for_mover(board_after: chess.Board, opponent_text: str) -> float:
    """Value (0..100) of a move to its player, from the opponent's perspective."""
    if board_after.is_checkmate():
        return 100.0
    if board_after.is_game_over(claim_draw=True):
        return 50.0
    legal_moves = {m.uci() for m in board_after.legal_moves}
    values = [value for u, value in parse_top5(opponent_text) if u in legal_moves]
    if not values:
        return 50.0          # No opponent reading: neutral, do not invent one.
    return 100.0 - max(values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="data/splits/test.jsonl")
    ap.add_argument("--puzzles", type=int, default=200)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=440)
    ap.add_argument("--out", default="out/inference-compute.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    items = load(args.split, args.puzzles, args.seed)
    print("puzzles:", len(items), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = [mav.transition_prompt(x["fen0"], x["m0"]) for x in items]

    print("== 1. greedy ==", flush=True)
    greedy_texts = generate(model, tok, prompts, args.batch, args.max_tokens)

    print("== 2. pass@%d ==" % args.k, flush=True)
    samples_text = generate(model, tok, [p for p in prompts for _ in range(args.k)],
                            args.batch, args.max_tokens, sample=True)

    print("== 3. depth-2 search with the model itself ==", flush=True)
    candidates = []          # (puzzle index, move, resulting board)
    for i, (x, text) in enumerate(zip(items, greedy_texts)):
        legal_moves = {m.uci() for m in x["board"].legal_moves}
        seen = []
        for u, _value in parse_top5(text):
            if u in legal_moves and u not in seen:
                seen.append(u)
        if not seen:
            move, _source = mav.verified_move(text, x["board"])
            seen = [move] if move else []
        for move in seen:
            after = x["board"].copy()
            after.push(chess.Move.from_uci(move))
            candidates.append((i, move, after))
    opponent_texts = generate(
        model, tok,
        [mav.position_prompt(board.fen()) for _i, _move, board in candidates
         if not board.is_game_over(claim_draw=True)],
        args.batch, args.max_tokens)
    opponent_iter = iter(opponent_texts)

    by_puzzle: dict[int, list[tuple[str, float]]] = {}
    for i, move, after in candidates:
        opponent_text = "" if after.is_game_over(claim_draw=True) else next(opponent_iter)
        by_puzzle.setdefault(i, []).append((move, value_for_mover(after, opponent_text)))

    correct_counts = {"greedy": 0, "sampled": 0, "pass_at_k": 0, "majority": 0, "search_depth2": 0}
    records = []
    for i, item in enumerate(items):
        solution = item["solution"]
        greedy, _source = mav.verified_move(greedy_texts[i], item["board"])
        sample_moves = [mav.verified_move(text, item["board"])[0]
                        for text in samples_text[i * args.k:(i + 1) * args.k]]
        majority = Counter(move for move in sample_moves if move).most_common(1)
        majority = majority[0][0] if majority else None
        options = by_puzzle.get(i, [])
        search = max(options, key=lambda pair: pair[1])[0] if options else greedy
        correct_counts["greedy"] += greedy == solution
        correct_counts["sampled"] += sample_moves[0] == solution
        correct_counts["pass_at_k"] += solution in sample_moves
        correct_counts["majority"] += majority == solution
        correct_counts["search_depth2"] += search == solution
        records.append({"rating": int(item["rating"]), "solution": solution, "greedy": greedy,
                        "search_depth2": search, "candidates": options, "samples": sample_moves})

    n = len(items)
    result = {"n": n, "k": args.k, "adapter": args.adapter, "items": records}
    print()
    print("=== CAN THE MODEL THINK AHEAD USING ITS OWN WORLD MODEL? (n=%d) ===" % n)
    for key, label in (("greedy", "greedy (normal output)"),
                       ("sampled", "1 sample (T=0.8)"),
                       ("majority", "%d-sample majority" % args.k),
                       ("search_depth2", "depth-2 search"),
                       ("pass_at_k", "pass@%d (RL/STaR ceiling)" % args.k)):
        count = correct_counts[key]
        lo, hi = wilson(count, n)
        result[key] = {"correct": count, "pct": 100 * count / n, "ci95": [lo, hi]}
        print("%-30s %3d/%d = %5.2f%%  CI95 [%.2f; %.2f]" % (label, count, n, 100 * count / n, lo, hi))

    greedy_correct, search_correct = correct_counts["greedy"], correct_counts["search_depth2"]
    p = (greedy_correct + search_correct) / (2 * n)
    se = math.sqrt(p * (1 - p) * 2 / n) if 0 < p < 1 else 0
    result["z_search_vs_greedy"] = (search_correct - greedy_correct) / n / se if se else 0.0
    print("search vs greedy: z = %+.2f" % result["z_search_vs_greedy"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved to", args.out)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    os._exit(exit_code)
