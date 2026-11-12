"""Chess puzzle verification with a legality mask and partial reward.

Lichess dataset convention (checked empirically against the data):
the FEN is the position BEFORE the opponent's move, and `Moves[0]` is that
opponent move. After applying it, the side to move is the SOLVER, whose moves
are `Moves[1], Moves[3], ...`.

This module is deliberately pure (no I/O, no network) so it can be tested on CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import chess

class InvalidPuzzle(ValueError):
    """The record is not a legal line; quarantine it, do not repair it."""

@dataclass(frozen=True)
class Puzzle:
    id: str
    rating: int
    themes: tuple[str, ...]
    game_id: str
    # Position with the opponent's move already applied: this is where the solver moves.
    board: chess.Board = field(compare=False)
    solution: tuple[str, ...] = ()   # solver moves, UCI
    replies: tuple[str, ...] = ()    # opponent replies, UCI

    @property
    def side(self) -> chess.Color:
        return self.board.turn

    @property
    def first_move(self) -> str:
        if not self.solution:
            raise InvalidPuzzle(f"{self.id}: empty solution")
        return self.solution[0]

    @property
    def fen(self) -> str:
        return self.board.fen()

def build_puzzle(
    *,
    puzzle_id: str,
    fen: str,
    moves: str,
    rating: int,
    themes: tuple[str, ...] = (),
    game_id: str = "",
) -> Puzzle:
    """Build a Puzzle from a Lichess dataset row.

    Raises `InvalidPuzzle` when the line is not playable. It never tries to
    repair it: a bad record is quarantined, never turned into an invented target.
    """
    ucis = moves.split()
    if len(ucis) < 2:
        raise InvalidPuzzle(f"{puzzle_id}: line too short ({len(ucis)} moves)")

    board = chess.Board(fen)

    # Moves[0] belongs to the side to move in the FEN (the opponent).
    try:
        opponent_move = chess.Move.from_uci(ucis[0])
    except ValueError as exc:
        raise InvalidPuzzle(f"{puzzle_id}: invalid UCI in Moves[0]: {ucis[0]}") from exc
    if opponent_move not in board.legal_moves:
        raise InvalidPuzzle(f"{puzzle_id}: illegal opponent move in {fen}: {ucis[0]}")
    board.push(opponent_move)

    solution = tuple(ucis[1::2])
    replies = tuple(ucis[2::2])
    if not solution:
        raise InvalidPuzzle(f"{puzzle_id}: no solution move")

    # Validate the whole line; any illegal move invalidates the record.
    trial = board.copy()
    for i, uci in enumerate(ucis[1:], start=1):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise InvalidPuzzle(f"{puzzle_id}: invalid UCI: {uci}") from exc
        if mv not in trial.legal_moves:
            raise InvalidPuzzle(f"{puzzle_id}: move {i} illegal ({uci}) in {trial.fen()}")
        trial.push(mv)

    return Puzzle(
        id=puzzle_id,
        rating=rating,
        themes=themes,
        game_id=game_id,
        board=board,
        solution=solution,
        replies=replies,
    )

# --------------------------------------------------------------------------
# Extracting a move from model output
# --------------------------------------------------------------------------

# Canonical UCI: origin square, destination square, optional promotion.
_UCI_PATTERN = re.compile(r"\b([a-h][1-8])[- ]?([a-h][1-8])([qrbn])?\b", re.IGNORECASE)

def normalize_uci(text: str) -> str | None:
    """Extract the first plausible UCI move from free text.

    Accepts `e2e4`, `e2-e4` and `e7e8q`, and ignores code fences and punctuation.
    Returns None when there is no candidate, which is different from an illegal move.
    """
    if not text:
        return None
    for m in _UCI_PATTERN.finditer(text):
        origin, destination, promo = m.group(1).lower(), m.group(2).lower(), (m.group(3) or "").lower()
        return f"{origin}{destination}{promo}"
    return None

# --------------------------------------------------------------------------
# Legality mask
# --------------------------------------------------------------------------

def legal_moves_uci(board: chess.Board) -> set[str]:
    return {m.uci() for m in board.legal_moves}

def mask_logits(logits_by_move: dict[str, float], board: chess.Board) -> dict[str, float]:
    """Set the logit of every illegal move to -inf.

    Used to measure the effect of the mask: without it the model spends its
    budget learning that `Ke9` does not exist.
    """
    legal = legal_moves_uci(board)
    return {u: (v if u in legal else float("-inf")) for u, v in logits_by_move.items()}

# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveResult:
    status: str            # "correct" | "legal_good" | "legal_bad" | "illegal" | "unparseable"
    move: str | None
    reward: float

WEIGHTS = {
    "correct": 1.0,
    "legal_good": 0.3,
    "legal_bad": -0.2,
    "illegal": -1.0,
    "unparseable": -1.0,
}

def score_move(*, board: chess.Board, predicted_move: str | None, solution_move: str, alternative_eval: float | None = None) -> MoveResult:
    """Classify a move and return its reward.

    `alternative_eval` is the evaluation (in centipawns, from the point of view
    of the side to move) of the predicted move, computed by an external oracle
    (Stockfish). When absent, a legal move that is not the solution gets the
    conservative `legal_bad` weight, never `legal_good` by assumption.
    """
    if predicted_move is None:
        return MoveResult("unparseable", None, WEIGHTS["unparseable"])

    try:
        mv = chess.Move.from_uci(predicted_move)
    except ValueError:
        return MoveResult("illegal", predicted_move, WEIGHTS["illegal"])

    if mv not in board.legal_moves:
        return MoveResult("illegal", predicted_move, WEIGHTS["illegal"])

    if predicted_move == solution_move:
        return MoveResult("correct", predicted_move, WEIGHTS["correct"])

    if alternative_eval is not None and alternative_eval >= 0:
        return MoveResult("legal_good", predicted_move, WEIGHTS["legal_good"])

    return MoveResult("legal_bad", predicted_move, WEIGHTS["legal_bad"])


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

# Portuguese piece names: part of the model-facing prompt used in the published
# measurements. Do not edit while LEGACY_FORMAT is True.
PIECE_NAMES = {
    "P": "peão branco", "N": "cavalo branco", "B": "bispo branco",
    "R": "torre branca", "Q": "dama branca", "K": "rei branco",
    "p": "peão preto", "n": "cavalo preto", "b": "bispo preto",
    "r": "torre preta", "q": "dama preta", "k": "rei preto",
}


def describe_position(board: chess.Board) -> str:
    """List the pieces square by square.

    This format forces state tracking: the model cannot "feel" the board and
    has to reconstruct it from the text.
    """
    names = PIECE_NAMES if LEGACY_FORMAT else PIECE_NAMES_EN
    lines = []
    for square, piece in sorted(board.piece_map().items(), key=lambda kv: kv[0]):
        lines.append(f"{chess.square_name(square)}: {names[piece.symbol()]}")
    return "\n".join(lines)


# The published direct-evaluation measurements used the Portuguese prompt below.
# Keep True to reproduce them; False switches to an English equivalent.
LEGACY_FORMAT = True

PIECE_NAMES_EN = {
    "P": "white pawn", "N": "white knight", "B": "white bishop",
    "R": "white rook", "Q": "white queen", "K": "white king",
    "p": "black pawn", "n": "black knight", "b": "black bishop",
    "r": "black rook", "q": "black queen", "k": "black king",
}

def build_prompt(board: chess.Board) -> str:
    if LEGACY_FORMAT:
        side = "brancas" if board.turn == chess.WHITE else "pretas"
        return (
            "Você joga com as " + side + " e é a sua vez.\n\n"
            "Posição (peças por casa):\n" + describe_position(board) + "\n\n"
            "FEN: " + board.fen() + "\n\n"
            "Responda com o melhor lance em notação UCI (ex.: e2e4). Escreva apenas o lance."
        )
    side = "white" if board.turn == chess.WHITE else "black"
    return (
        "You are playing as " + side + " and it is your turn.\n\n"
        "Position (pieces by square):\n" + describe_position(board) + "\n\n"
        "FEN: " + board.fen() + "\n\n"
        "Respond with the best move in UCI notation (e.g. e2e4). Write only the move."
    )
