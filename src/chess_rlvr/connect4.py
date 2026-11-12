"""Minimal Connect Four engine for generating evaluation positions."""

from __future__ import annotations

import random

ROWS = 6
COLS = 7
EMPTY = "."


def new_board() -> list[list[str]]:
    """Create a 6x7 grid with rows in visual order (top row first)."""
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def legal_columns(board: list[list[str]]) -> list[int]:
    """Return the playable columns, numbered 1 to 7."""
    return [c + 1 for c in range(COLS) if board[0][c] == EMPTY]


def play(board: list[list[str]], column: int, side: str) -> None:
    """Drop ``side`` into the column (1-7), modifying the board in place."""
    if side not in ("X", "O"):
        raise ValueError("side must be 'X' or 'O'")
    if not 1 <= column <= COLS:
        raise ValueError("column must be between 1 and 7")
    c = column - 1
    for r in range(ROWS - 1, -1, -1):
        if board[r][c] == EMPTY:
            board[r][c] = side
            return
    raise ValueError("column is full")


def winner(board: list[list[str]]) -> str | None:
    """Return ``X``/``O`` if there are four in a row, otherwise None."""
    for r in range(ROWS):
        for c in range(COLS):
            side = board[r][c]
            if side == EMPTY:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + 3 * dr, c + 3 * dc
                if 0 <= er < ROWS and 0 <= ec < COLS and all(
                    board[r + i * dr][c + i * dc] == side for i in range(1, 4)
                ):
                    return side
    return None


def _winning_columns(board: list[list[str]], side: str) -> set[int]:
    winning = set()
    for column in legal_columns(board):
        trial = [row[:] for row in board]
        play(trial, column, side)
        if winner(trial) == side:
            winning.add(column)
    return winning


def generate_positions(n: int, kind: str, seed: int) -> list[dict]:
    """Generate ``n`` distinct positions suited to ``"win"`` or ``"block"``.

    ``kind`` also accepts the legacy Portuguese names ``"vencer"`` and
    ``"bloquear"``. Each item contains ``board``, ``side`` and
    ``correct_columns`` (columns 1-7). Boards are mutable grids; each position
    gets its own copy.
    """
    if n < 0:
        raise ValueError("n must not be negative")
    kind = {"vencer": "win", "bloquear": "block"}.get(kind, kind)
    if kind not in ("win", "block"):
        raise ValueError("kind must be 'win' or 'block'")
    if n == 0:
        return []

    rng = random.Random(seed)
    found: dict[tuple, dict] = {}
    attempts = 0
    while len(found) < n and attempts < max(100_000, n * 2_000):
        attempts += 1
        board = new_board()
        side = "X"
        while legal_columns(board) and winner(board) is None:
            legal = legal_columns(board)
            if kind == "win":
                correct_columns = _winning_columns(board, side)
                if correct_columns:
                    key = (tuple("".join(l) for l in board), side)
                    found.setdefault(key, {
                        "board": [line[:] for line in board],
                        "side": side,
                        "correct_columns": set(correct_columns),
                    })
                    break
            else:
                if not _winning_columns(board, side):
                    threats = _winning_columns(board, "O" if side == "X" else "X")
                    if len(threats) == 1:
                        key = (tuple("".join(l) for l in board), side)
                        found.setdefault(key, {
                            "board": [line[:] for line in board],
                            "side": side,
                            "correct_columns": set(threats),
                        })
                        break
            play(board, rng.choice(legal), side)
            if winner(board) is not None:
                break
            side = "O" if side == "X" else "X"

    if len(found) < n:
        raise RuntimeError(
            "could not generate %d distinct positions of kind %r (generated %d)"
            % (n, kind, len(found))
        )
    return list(found.values())[:n]


def render(board: list[list[str]]) -> str:
    """Render the board top to bottom, with numbered columns."""
    return "\n".join([" ".join(line) for line in board] + ["1 2 3 4 5 6 7"])
