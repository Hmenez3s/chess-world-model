#!/usr/bin/env python3
"""Play games in batches against a reference opponent."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import chess
import chess.engine


def load_model(model_id: str, adapter: str):
    # Late imports keep fake-player mode runnable with python-chess alone.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model, torch


def generate_moves(games, tok, model, torch, output_format: str,
                   sample: bool) -> list[tuple[str, str]]:
    from chess_rlvr import mav
    from chess_rlvr.verifier import build_prompt, normalize_uci

    prompts = []
    for game in games:
        content = (mav.position_prompt(game["board"].fen()) if output_format == "mav"
                   else build_prompt(game["board"]))
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True))
    encoded = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    max_tokens = 440 if output_format == "mav" else 16
    options = {"max_new_tokens": max_tokens, "do_sample": sample,
               "pad_token_id": tok.eos_token_id}
    if sample:
        options["temperature"] = 0.7
    with torch.no_grad():
        outputs = model.generate(**encoded, **options)
    results = []
    for i, _game in enumerate(games):
        text = tok.decode(outputs[i][encoded.input_ids.shape[1]:], skip_special_tokens=True)
        uci = mav.extract_fields(text)["best"] if output_format == "mav" else normalize_uci(text)
        results.append((uci or "", text))
    return results


def generate_in_batches(games, tok, model, torch, output_format: str, sample: bool,
                        batch_size: int) -> list[tuple[str, str]]:
    outputs = []
    for start in range(0, len(games), batch_size):
        outputs.extend(generate_moves(games[start:start + batch_size], tok, model,
                                      torch, output_format, sample))
    return outputs


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    margin = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--format", choices=["mav", "direct"], default="mav")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--opponent", choices=["random", "stockfish"], default="random")
    ap.add_argument("--elo", type=int, default=1320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--verify", action="store_true",
                    help="use the model's own top-5/legal moves to correct illegal moves")
    ap.add_argument("--fake-player", choices=["random"])
    ap.add_argument("--out", default="out/games.json")
    args = ap.parse_args()
    if args.games <= 0 or args.batch <= 0 or args.max_plies <= 0:
        ap.error("games, batch, and max-plies must be positive")

    rng = random.Random(0)
    tok = model = torch = None
    if not args.fake_player:
        root = os.path.join(os.path.dirname(__file__), "..", "src")
        import sys
        sys.path.insert(0, root)
        tok, model, torch = load_model(args.model, args.adapter)

    motor = None
    if args.opponent == "stockfish":
        caminho = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/bin/stockfish"))
        motor = chess.engine.SimpleEngine.popen_uci(caminho)
        motor.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})

    games = [{"board": chess.Board(), "model_white": i % 2 == 0,
              "uci": [], "finished": False, "result": None, "reason": None}
             for i in range(args.games)]
    model_moves = illegal_moves = 0
    verification_sources: dict[str, int] = {}
    try:
        while any(not game["finished"] for game in games):
            # Group positions where the model moves this round.
            # Finished boards have no legal moves; without this guard, random
            # fallback fails with "Cannot choose from an empty sequence".
            pending = [game for game in games if not game["finished"]
                       and not game["board"].is_game_over(claim_draw=True)
                       and game["board"].turn == (chess.WHITE if game["model_white"] else chess.BLACK)]
            choices = ([(rng.choice(list(game["board"].legal_moves)).uci(), "")
                        for game in pending]
                       if args.fake_player else
                       generate_in_batches(pending, tok, model, torch, args.format,
                                           False, args.batch))
            def is_legal(game, uci):
                try:
                    return chess.Move.from_uci(uci) in game["board"].legal_moves
                except (ValueError, TypeError):
                    return False

            retry = [k for k, (game, (uci, _text)) in enumerate(zip(pending, choices))
                     if not is_legal(game, uci)]
            if retry and not args.fake_player:
                alternatives = generate_in_batches([pending[k] for k in retry], tok, model,
                                                   torch, args.format, True, args.batch)
                for k, pair in zip(retry, alternatives):
                    choices[k] = pair
            for game, (uci, text) in zip(pending, choices):
                if game["board"].is_game_over(claim_draw=True):
                    continue
                model_moves += 1
                try:
                    move = chess.Move.from_uci(uci)
                except (ValueError, TypeError):
                    move = None
                if move not in game["board"].legal_moves:
                    if args.verify and args.format == "mav" and text:
                        from chess_rlvr.mav import verified_move
                        alternative, source = verified_move(text, game["board"])
                        if alternative:
                            move = chess.Move.from_uci(alternative)
                            verification_sources[source] = verification_sources.get(source, 0) + 1
                    if move not in game["board"].legal_moves:
                        illegal_moves += 1
                        move = rng.choice(list(game["board"].legal_moves))
                game["board"].push(move)
                game["uci"].append(move.uci())

            # Apply opponent moves after model moves.
            for game in games:
                if game["finished"]:
                    continue
                if game["board"].is_game_over(claim_draw=True):
                    outcome = game["board"].outcome(claim_draw=True)
                    if outcome.winner is None:
                        game["result"], game["reason"] = "D", "draw_rules"
                    else:
                        won_by_model = outcome.winner == (chess.WHITE if game["model_white"] else chess.BLACK)
                        game["result"] = "W" if won_by_model else "L"
                        game["reason"] = "checkmate" if game["board"].is_checkmate() else "rules"
                    game["finished"] = True
                elif game["board"].ply() >= args.max_plies:
                    game["result"], game["reason"], game["finished"] = "D", "adjudicated_max_plies", True

            opponents = [game for game in games if not game["finished"] and
                         game["board"].turn != (chess.WHITE if game["model_white"] else chess.BLACK)]
            for game in opponents:
                if args.opponent == "random":
                    move = rng.choice(list(game["board"].legal_moves))
                else:
                    move = motor.play(game["board"], chess.engine.Limit(time=0.05)).move
                game["board"].push(move)
                game["uci"].append(move.uci())
    finally:
        if motor is not None:
            motor.quit()

    wins = sum(game["result"] == "W" for game in games)
    draws = sum(game["result"] == "D" for game in games)
    losses = sum(game["result"] == "L" for game in games)
    score = (wins + 0.5 * draws) / args.games
    result = {"games": [{"model_color": "white" if game["model_white"] else "black",
                          "result": game["result"], "reason": game["reason"], "uci": game["uci"]}
                         for game in games],
              "wins": wins, "draws": draws, "losses": losses, "score": score,
              "model_moves": model_moves, "illegal_moves": illegal_moves,
              "illegal_move_rate": illegal_moves / model_moves if model_moves else 0.0,
              "verified": args.verify,
              "verification_sources": verification_sources}
    if args.opponent == "stockfish":
        if 0 < score < 1:
            lo, hi = wilson_interval(score, args.games)
            transform = lambda value: args.elo + 400 * math.log10(value / (1 - value))
            result["elo_estimate"] = args.elo + 400 * math.log10(score / (1 - score))
            result["elo_ci95"] = [transform(lo), transform(hi)]
        else:
            # Scores of 0 or 1 have no Elo (the log diverges). Wilson's confidence
            # interval still provides a bound: "Elo at most X".
            result["elo_estimate"] = None
            result["elo_ci95"] = None
            result["elo_status"] = "undefined: score is 0 or 1"
            lo, hi = wilson_interval(score, args.games)
            bound = lo if score == 1.0 else hi
            if 0 < bound < 1:
                result["elo_bound"] = args.elo + 400 * math.log10(bound / (1 - bound))
                result["elo_bound_side"] = "upper" if score == 0 else "lower"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"W/D/L: {wins}/{draws}/{losses}; score: {score:.3f}; "
          f"illegal moves: {illegal_moves}/{model_moves} ({result['illegal_move_rate']:.2%})")
    if args.opponent == "stockfish":
        print("Estimated Elo:", result["elo_estimate"] if result["elo_estimate"] is not None else "undefined")
    print("saved to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
