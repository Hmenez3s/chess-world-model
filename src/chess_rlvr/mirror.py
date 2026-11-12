"""Position mirroring as a memorization test.

Core idea: a puzzle's tactic is invariant under a vertical flip of the board
with the colors swapped. A model that *understands* the position solves the
mirrored version at the same rate. A model that *memorized* the puzzle drops
close to zero, because the coordinates changed.

This turns "did it memorize?" from an opinion into a measurement.
"""

from __future__ import annotations

import chess


def mirror_square(square: int) -> int:
    """Flip a square vertically: a1 <-> a8, e2 <-> e7."""
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    return chess.square(file_, 7 - rank)


def mirror_uci(uci: str) -> str:
    mv = chess.Move.from_uci(uci)
    promo = mv.promotion
    return chess.Move(
        mirror_square(mv.from_square),
        mirror_square(mv.to_square),
        promotion=promo,
    ).uci()


def mirror_board(board: chess.Board) -> chess.Board:
    """Vertical flip + color swap.

    This is a chess isomorphism: the resulting position is legal and the same
    tactic holds. Castling rights are swapped between sides along with the
    colors, and the en passant square is flipped.
    """
    mirrored = chess.Board(None)  # empty board

    for square, piece in board.piece_map().items():
        mirrored_square = mirror_square(square)
        mirrored_piece = chess.Piece(piece.piece_type, not piece.color)
        mirrored.set_piece_at(mirrored_square, mirrored_piece)

    # Turn flipped: the side that was to move is still to move, but with the other color.
    mirrored.turn = not board.turn

    # Castling: one side's right becomes the other side's right in the opposite color.
    if board.has_kingside_castling_rights(chess.WHITE):
        mirrored.castling_rights |= chess.BB_H8
    if board.has_queenside_castling_rights(chess.WHITE):
        mirrored.castling_rights |= chess.BB_A8
    if board.has_kingside_castling_rights(chess.BLACK):
        mirrored.castling_rights |= chess.BB_H1
    if board.has_queenside_castling_rights(chess.BLACK):
        mirrored.castling_rights |= chess.BB_A1

    if board.ep_square is not None:
        mirrored.ep_square = mirror_square(board.ep_square)

    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number

    return mirrored


class InvalidMirror(ValueError):
    """The mirrored position is not playable, so the puzzle is unusable for the test."""


def mirror_puzzle(puzzle):
    """Return a mirrored copy of the puzzle, with the solution transformed.

    Raises `InvalidMirror` when the flipped position is not legal (for example,
    pawns on terminal ranks). Those cases are excluded from the metric, not
    counted as model failures.
    """
    mirrored_board = mirror_board(puzzle.board)

    if mirrored_board.status() != chess.STATUS_VALID:
        raise InvalidMirror(f"{puzzle.id}: invalid mirrored position ({mirrored_board.status()})")

    solution = tuple(mirror_uci(u) for u in puzzle.solution)
    replies = tuple(mirror_uci(u) for u in puzzle.replies)

    # Proof that the mirror is faithful: the transformed line must be legal.
    trial = mirrored_board.copy()
    for i, uci in enumerate(solution):
        mv = chess.Move.from_uci(uci)
        if mv not in trial.legal_moves:
            raise InvalidMirror(
                f"{puzzle.id}: illegal mirrored solution at move {i} ({uci})"
            )
        trial.push(mv)
        if i < len(replies):
            mv_r = chess.Move.from_uci(replies[i])
            if mv_r not in trial.legal_moves:
                raise InvalidMirror(
                    f"{puzzle.id}: illegal mirrored reply at move {i} ({replies[i]})"
                )
            trial.push(mv_r)

    from .verifier import Puzzle

    return Puzzle(
        id=puzzle.id + "::mirror",
        rating=puzzle.rating,
        themes=puzzle.themes,
        game_id=puzzle.game_id,
        board=mirrored_board,
        solution=solution,
        replies=replies,
    )
