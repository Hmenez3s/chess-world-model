"""Tests for the invariants the README claims.

They cover what can break silently and invalidate results:
  - the legality mask really blocks illegal moves
  - the mirror is an isomorphism (same tactic, legal line)
  - the GameId split does not leak
  - the Wilson CI has the properties of a confidence interval

They do not test the model or the GPUs: that is measured by the scripts in scripts/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import pytest

from chess_rlvr.evaluation import wilson
from chess_rlvr.data import split_by_game, audit_leakage, rating_band
from chess_rlvr.mirror import mirror_board, mirror_uci, mirror_puzzle
from chess_rlvr.verifier import (
    score_move, legal_moves_uci, mask_logits, build_puzzle,
    normalize_uci,
)

# The starting position and a well-known mate-in-1 position.
STARTING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_IN_1 = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K w - - 1 25"  # Qh6-h7#


# --------------------------------------------------------------------------
# verifier
# --------------------------------------------------------------------------

def test_normalize_uci_accepts_common_formats():
    assert normalize_uci("e2e4") == "e2e4"
    assert normalize_uci("  e2-e4 ") == "e2e4"
    assert normalize_uci("E2E4") == "e2e4"
    assert normalize_uci("e7e8q") == "e7e8q"


def test_normalize_uci_rejects_garbage():
    assert normalize_uci("") is None
    assert normalize_uci("nothing") is None
    assert normalize_uci("z9z9") is None


def test_mask_blocks_illegal_move():
    board = chess.Board(STARTING)
    logits = {"e2e4": 5.0, "d2d4": 4.0, "e2e5": 99.0, "a1a8": 88.0}
    masked = mask_logits(logits, board)
    # illegal moves must disappear or become -inf
    for illegal in ("e2e5", "a1a8"):
        assert illegal not in masked or masked[illegal] == float("-inf")
    # legal moves remain
    assert masked.get("e2e4") == 5.0


def test_mask_never_lets_illegal_through():
    board = chess.Board(MATE_IN_1)
    legal = legal_moves_uci(board)
    assert "h6h7" in legal
    # a move that is geometrically possible but illegal here
    assert "h6h8" not in legal


def test_score_move_rewards_the_mate():
    board = chess.Board(MATE_IN_1)
    r = score_move(board=board, predicted_move="h6h7", solution_move="h6h7")
    assert r.status == "correct"
    assert r.reward == 1.0


def test_score_move_illegal_gets_negative_weight():
    board = chess.Board(MATE_IN_1)
    r = score_move(board=board, predicted_move="h6h8", solution_move="h6h7")
    assert r.status == "illegal"
    assert r.reward < 0


def test_score_move_legal_wrong_move_is_not_good_by_assumption():
    """Without an oracle, a legal move that is not the solution must NOT be rewarded."""
    board = chess.Board(STARTING)
    r = score_move(board=board, predicted_move="e2e4", solution_move="d2d4")
    assert r.status == "legal_bad"
    assert r.reward < 0


def test_build_puzzle_starts_at_the_solver_move():
    # Moves[0] is the opponent's; the solver plays Moves[1].
    p = build_puzzle(puzzle_id="test", fen=STARTING, moves="e2e4 e7e5",
                      rating=1500, themes=("opening",), game_id="g1")
    assert p.first_move == "e7e5"
    assert p.side == chess.BLACK
    assert p.board.turn == chess.BLACK


# --------------------------------------------------------------------------
# mirror
# --------------------------------------------------------------------------

def test_mirror_square_is_an_involution():
    from chess_rlvr.mirror import mirror_square
    for square in chess.SQUARES:
        assert mirror_square(mirror_square(square)) == square


def test_mirror_uci_preserves_promotion():
    assert mirror_uci("e2e4") == "e7e5"
    assert mirror_uci("e7e8q") == "e2e1q"
    assert mirror_uci("a1h8") == "a8h1"


def test_mirror_preserves_legality():
    """If the mirror does not preserve legality, the memorization test is invalid."""
    for fen in (STARTING, MATE_IN_1):
        board = chess.Board(fen)
        mirrored = mirror_board(board)
        # same piece count, sides swapped
        assert len(board.piece_map()) == len(mirrored.piece_map())
        assert mirrored.turn != board.turn
        # every legal move maps to a legal move
        for mv in board.legal_moves:
            uci = mv.uci()
            assert chess.Move.from_uci(mirror_uci(uci)) in mirrored.legal_moves


def test_mirror_puzzle_keeps_rating_and_flips_the_solution():
    p = build_puzzle(puzzle_id="x", fen=STARTING, moves="e2e4 e7e5 d2d4",
                      rating=1500, themes=("opening",), game_id="g1")
    mirrored = mirror_puzzle(p)
    assert mirrored.rating == p.rating
    assert mirrored.solution == tuple(mirror_uci(u) for u in p.solution)
    # the mirrored line is legal from start to finish
    board = chess.Board(mirrored.board.fen())
    for uci in mirrored.solution:
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves
        board.push(mv)


# --------------------------------------------------------------------------
# data: leakage
# --------------------------------------------------------------------------

def _row(i: int, game: str) -> dict:
    return {"puzzle_id": "p%d" % i, "game_id": game, "fen": STARTING,
            "moves": "e2e4 e7e5", "rating": 1200 + i,
            "themes": ["opening"], "popularity": 95}


def test_split_by_game_does_not_leak():
    """Puzzles from the same game must land in the same split."""
    lines = [_row(i, "g%d" % (i // 5)) for i in range(200)]
    split = split_by_game(lines, seed=7)
    report = audit_leakage(split)
    assert len(report["shared_games"]) == 0
    assert len(report["shared_puzzles"]) == 0
    # every puzzle went to exactly one split
    total = sum(len(getattr(split, n)) for n in ("train", "validation", "test"))
    assert total == len(lines)


def test_split_is_deterministic_per_seed():
    lines = [_row(i, "g%d" % (i // 5)) for i in range(200)]
    a = split_by_game(lines, seed=7)
    b = split_by_game(lines, seed=7)
    assert [x["puzzle_id"] for x in a.train] == [x["puzzle_id"] for x in b.train]
    c = split_by_game(lines, seed=8)
    assert [x["puzzle_id"] for x in a.train] != [x["puzzle_id"] for x in c.train]


def test_rating_bands_are_contiguous():
    assert rating_band(0) == rating_band(599)
    assert rating_band(600) != rating_band(599)
    # the last band is open-ended, with no ceiling
    assert rating_band(2600) == "2600-3000"
    assert rating_band(3000) == rating_band(5000) == "3000+"


# --------------------------------------------------------------------------
# evaluation: Wilson CI
# --------------------------------------------------------------------------

def test_wilson_has_ci_properties():
    lo, hi = wilson(5, 100)
    assert 0.0 <= lo < 0.05 < hi <= 1.0
    # more data -> narrower interval
    lo2, hi2 = wilson(50, 1000)
    assert (hi2 - lo2) < (hi - lo)
    # 0 successes still has a positive upper bound
    lo0, hi0 = wilson(0, 600)
    assert lo0 == 0.0 and hi0 > 0.0
    # 100% success still has a lower bound below 1
    lo100, hi100 = wilson(600, 600)
    assert lo100 > 0.0 and hi100 == pytest.approx(1.0, abs=1e-9)
