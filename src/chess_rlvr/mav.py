"""World-model training format, adapted from MAV.

Reference: Schultz et al., "Mastering Board Games by External and Internal
Planning with Language Models" (DeepMind, arXiv 2412.12119, v3 2025). A 1B
transformer trained this way reaches an estimated ~2,820 Elo without search.

What the format teaches, and plain move-only training does not:
  1. state transition: previous position + move -> resulting board
  2. the legal moves of the new position
  3. the value of each candidate move (top-k with win probability)
  4. the best move

Training only item 4 ("behavioral cloning") is the weakest target: in
ChessBench (arXiv 2402.04494) predicting a value per move solves 83.3% of
puzzles versus 65.7% for predicting only the best move, in the same model.

The board is written as an 8x8 grid, one character per square (uppercase =
white, lowercase = black), with rank and file labels. Each square lands on a
fixed position, as in MAV, instead of depending on how BPE splits a FEN.
"""

from __future__ import annotations

import math

import chess
import chess.engine

TOP_K = 5

# The released adapter was trained on the Portuguese strings below. Keep
# LEGACY_FORMAT = True to evaluate or play with it; set it to False to train a
# new model on the English format. extract_fields() reads both formats.
LEGACY_FORMAT = True

_LEGACY = {
    "transition": ("FEN anterior: %s\nLance jogado: %s\n"
                   "Mostre o tabuleiro resultante, os lances legais, os %d melhores "
                   "com chance de vitória e o melhor lance."),
    "position": ("FEN: %s\nMostre o tabuleiro, os lances legais, os %d melhores "
                 "com chance de vitória e o melhor lance."),
    "board": "estado:", "to_move": "vez: %s", "white": "brancas", "black": "pretas",
    "legal": "legais: %s", "best": "melhor: %s",
}
_ENGLISH = {
    "transition": ("Previous FEN: %s\nMove played: %s\n"
                   "Show the resulting board, the legal moves, the %d best moves "
                   "with their win probability, and the best move."),
    "position": ("FEN: %s\nShow the board, the legal moves, the %d best moves "
                 "with their win probability, and the best move."),
    "board": "board:", "to_move": "to move: %s", "white": "white", "black": "black",
    "legal": "legal: %s", "best": "best: %s",
}

_BOARD_LABELS = ("estado:", "board:")
_LEGAL_LABELS = ("legais:", "legal:")
_BEST_LABELS = ("melhor:", "best:")


def _strings() -> dict:
    return _LEGACY if LEGACY_FORMAT else _ENGLISH


def board_grid(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        squares = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            squares.append(piece.symbol() if piece else ".")
        rows.append("%d %s" % (rank + 1, " ".join(squares)))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def win_probability(cp: int) -> int:
    """Centipawns -> win probability 0..100 (Lichess formula)."""
    p = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)
    return max(0, min(100, int(round(p))))


def _side_name(board: chess.Board) -> str:
    s = _strings()
    return s["white"] if board.turn == chess.WHITE else s["black"]


def transition_prompt(previous_fen: str, move: str) -> str:
    return _strings()["transition"] % (previous_fen, move, TOP_K)


def position_prompt(fen: str) -> str:
    return _strings()["position"] % (fen, TOP_K)


def response(board: chess.Board, top: list[tuple[str, int]]) -> str:
    """top: [(uci, win probability 0..100)] sorted from best to worst."""
    s = _strings()
    legal = " ".join(sorted(m.uci() for m in board.legal_moves))
    lines = [
        s["board"],
        board_grid(board),
        s["to_move"] % _side_name(board),
        s["legal"] % legal,
        "top%d: %s" % (TOP_K, " ".join("%s:%d" % (m, v) for m, v in top)),
        s["best"] % top[0][0],
    ]
    return "\n".join(lines)


def evaluate_top(board: chess.Board, engine, depth: int = 10) -> list[tuple[str, int]]:
    """Top-k moves with win probability, from the side to move's point of view."""
    infos = engine.analyse(board, chess.engine.Limit(depth=depth),
                           multipv=min(TOP_K, board.legal_moves.count()))
    if isinstance(infos, dict):
        infos = [infos]
    top = []
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        cp = info["score"].pov(board.turn).score(mate_score=10000)
        top.append((info["pv"][0].uci(), win_probability(cp)))
    return top


# --------------------------------------------------------------------------
# Reading a model answer (evaluation)
# --------------------------------------------------------------------------

def extract_fields(text: str) -> dict:
    """Parse an answer in either format into {"board", "legal", "best"}."""
    fields: dict = {"board": None, "legal": None, "best": None}
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip() in _BOARD_LABELS and i + 9 <= len(lines):
            fields["board"] = "\n".join(lines[i + 1:i + 10])
        elif line.startswith(_LEGAL_LABELS):
            fields["legal"] = set(line.split(":", 1)[1].split())
        elif line.startswith(_BEST_LABELS):
            value = line.split(":", 1)[1].strip().split()
            fields["best"] = value[0] if value else None
    return fields


def verified_move(text: str, board: chess.Board) -> tuple[str | None, str]:
    """Final move checked against the real position.

    The model writes its own legal-move list and a top-5 with values. When
    `best:` is illegal, its own answer still contains valid candidates: use the
    first legal move of its top-5; if none is legal, the first legal move of its
    list.

    Returns (move, source) with source in {"model", "top5", "legal", "none"}.
    This gives the model no new information; it only stops it from playing
    outside its own list.
    """
    fields = extract_fields(text)
    real_legal = {m.uci() for m in board.legal_moves}

    best = fields["best"]
    if best in real_legal:
        return best, "model"

    for line in (text or "").splitlines():
        if not line.startswith("top"):
            continue
        for item in line.split(":", 1)[1].split():
            uci = item.split(":")[0].strip().lower()
            if uci in real_legal:
                return uci, "top5"

    for uci in sorted(fields["legal"] or ()):
        if uci in real_legal:
            return uci, "legal"

    return None, "none"
