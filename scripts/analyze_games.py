#!/usr/bin/env python3
"""Where does the model lose its games? Per-move analysis with Stockfish.

For every move the model played, compares the position value before the move
(best play) with the value after it, from the mover's point of view:

    loss = value_before - value_after          (centipawns, capped at 3000)
    blunder  = loss >= 300  (three pawns or more)
    mistake  = 100 <= loss < 300

Reports blunders per move and the move number of the first blunder in each game.

Usage:
    python scripts/analyze_games.py out/games-stockfish-verified.json --out out/game-analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import chess
import chess.engine

CAP = 3000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("games")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--out", default="out/game-analysis.json")
    args = ap.parse_args()

    data = json.loads(Path(args.games).read_text(encoding="utf-8"))
    games = data["games"]
    engine = chess.engine.SimpleEngine.popen_uci(
        os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/bin/stockfish")))
    limit = chess.engine.Limit(depth=args.depth)

    def value(board: chess.Board) -> int:
        return engine.analyse(board, limit)["score"].pov(board.turn).score(mate_score=10000)

    per_game, losses = [], []
    for g in games:
        model_white = g["model_color"] == "white"
        board = chess.Board()
        blunders = mistakes = moves = 0
        first_blunder = None
        for ply, uci in enumerate(g.get("uci", [])):
            move = chess.Move.from_uci(uci)
            if (board.turn == chess.WHITE) == model_white:
                before = min(value(board), CAP)
                board.push(move)
                after = min(-value(board), CAP)
                loss = max(0, before - after)
                losses.append(loss)
                moves += 1
                if loss >= 300:
                    blunders += 1
                    if first_blunder is None:
                        first_blunder = ply // 2 + 1
                elif loss >= 100:
                    mistakes += 1
            else:
                board.push(move)
        per_game.append({"model_color": "white" if model_white else "black",
                         "result": g["result"],
                         "model_moves": moves, "blunders": blunders, "mistakes": mistakes,
                         "first_blunder_move": first_blunder})
    engine.quit()

    total_moves = sum(x["model_moves"] for x in per_game)
    total_blunders = sum(x["blunders"] for x in per_game)
    firsts = [x["first_blunder_move"] for x in per_game if x["first_blunder_move"]]
    losses.sort()
    summary = {
        "games": len(per_game), "model_moves": total_moves,
        "blunders": total_blunders, "mistakes": sum(x["mistakes"] for x in per_game),
        "moves_per_blunder": total_moves / total_blunders if total_blunders else None,
        "median_loss_cp": losses[len(losses) // 2] if losses else None,
        "mean_first_blunder_move": sum(firsts) / len(firsts) if firsts else None,
        "games_with_blunder_by_move_5": sum(1 for f in firsts if f <= 5),
        "depth": args.depth, "per_game": per_game,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("model moves: %d | blunders: %d (one every %.1f moves) | first blunder: move %.1f on average"
          % (total_moves, total_blunders, summary["moves_per_blunder"] or 0,
             summary["mean_first_blunder_move"] or 0))
    print("saved to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
