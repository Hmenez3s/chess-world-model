"""Move evaluation with Stockfish, for partial reward.

Without this, the reward is binary (correct/incorrect) and ~90% of GRPO groups
end up with zero variance, so the model learns nothing. Stockfish provides a
continuous signal: "not the solution move, but it does not ruin the position either".

The engine runs at a low, deterministic depth to be fast and reproducible.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import chess
import chess.engine

DEFAULT_DEPTH = 8


def find_engine() -> str | None:
    """Locate the Stockfish binary without relying on sudo or a global PATH."""
    candidates = [
        os.environ.get("STOCKFISH_PATH"),
        os.path.expanduser("~/bin/stockfish"),
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        shutil.which("stockfish"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


@dataclass
class Engine:
    """Thin wrapper over python-chess.engine that guarantees the process is closed."""
    path: str
    depth: int = DEFAULT_DEPTH
    _engine: object = None

    def open(self):
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def eval_after(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluation in centipawns from the point of view of the side TO MOVE in `board`.

        Positive = good for the side making the move. Mate is scaled to ±10000.
        """
        after = board.copy()
        after.push(move)
        info = self.open().analyse(after, chess.engine.Limit(depth=self.depth))
        score = info["score"].pov(board.turn)
        # `score(mate_score=...)` already converts mate to the sentinel value with
        # the correct sign. Do not special-case mate by hand: mate in 0 (checkmate
        # already delivered) has `mate() == 0` and would fall into the negative branch.
        return float(score.score(mate_score=10000))


def normalize_advantage(centipawns: float, scale: float = 300.0) -> float:
    """Compress the evaluation to [-1, 1], saturating at ~3 pawns.

    Keeps an already-won position from dominating the reward: the difference
    between +8 and +12 pawns does not matter for learning, but the difference
    between 0 and +1 matters a lot.
    """
    if scale <= 0:
        return 0.0
    x = centipawns / scale
    return max(-1.0, min(1.0, x))
