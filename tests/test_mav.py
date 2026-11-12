"""Tests for the MAV format: the board grid must be reconstructible from the text."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import pytest

from chess_rlvr import mav

FEN = "r4rk1/2q3p1/p1bbp3/1p3p2/4NP2/5Q1P/PPPB2P1/4RRK1 w - - 0 20"


def test_grid_uses_uppercase_for_white():
    board = chess.Board(FEN)
    lines = mav.board_grid(board).splitlines()
    assert len(lines) == 9                       # 8 ranks + the file-label row
    assert lines[-1].split() == list("abcdefgh")
    # rank 4 = "4NP2": white knight on e4, white pawn on f4
    rank4 = lines[4].split()
    assert rank4[0] == "4"
    assert rank4[1:] == list("....NP..")


def test_grid_has_64_squares():
    board_grid = mav.board_grid(chess.Board(FEN))
    squares = [c for ln in board_grid.splitlines()[:8] for c in ln.split()[1:]]
    assert len(squares) == 64
    assert sum(1 for c in squares if c != ".") == len(chess.Board(FEN).piece_map())


def test_win_probability_maps_extremes():
    assert mav.win_probability(0) == 50
    assert mav.win_probability(10000) == 100
    assert mav.win_probability(-10000) == 0
    # monotonic
    assert mav.win_probability(300) > mav.win_probability(100) > mav.win_probability(0)


def test_response_is_reconstructible():
    """The generated text must carry the exact board grid, which is what the evaluation checks."""
    board = chess.Board(FEN)
    board.push_uci("e4d6")
    text = mav.response(board, [("c6f3", 85), ("c7d6", 40)])
    fields = mav.extract_fields(text)
    assert fields["board"] == mav.board_grid(board)
    assert fields["best"] == "c6f3"
    assert fields["legal"] == {m.uci() for m in board.legal_moves}


def test_extract_fields_survives_empty_text():
    fields = mav.extract_fields("")
    assert fields["board"] is None and fields["best"] is None

def test_extract_fields_parses_legacy_and_english_formats():
    grid = mav.board_grid(chess.Board(FEN))
    for board_label, legal_label, best_label in (
        ("estado:", "legais:", "melhor:"),
        ("board:", "legal:", "best:"),
    ):
        text = f"{board_label}\n{grid}\nto move: white\n{legal_label} e2e4\nbest: e2e4"
        text = text.replace("best: e2e4", f"{best_label} e2e4")
        fields = mav.extract_fields(text)
        assert fields["board"] == grid
        assert fields["legal"] == {"e2e4"}
        assert fields["best"] == "e2e4"


@pytest.mark.parametrize("uci", ["e4d6", "c6f3"])
def test_transition_prompt_carries_fen_and_move(uci):
    text = mav.transition_prompt(FEN, uci)
    assert FEN in text and uci in text

def test_verified_move_prefers_the_model_move():
    board = chess.Board(FEN)
    board.push_uci("e4d6")
    text = mav.response(board, [("c6f3", 85), ("c7d6", 40)])
    move, source = mav.verified_move(text, board)
    assert move == "c6f3" and source == "model"


def test_verified_move_falls_back_to_top5_when_best_is_illegal():
    board = chess.Board(FEN)
    board.push_uci("e4d6")
    # "best" is deliberately illegal; top5 contains a valid move
    text = "top5: a1a8:90 c6f3:80\nbest: a1a8\nlegal: c6f3 c7d6"
    move, source = mav.verified_move(text, board)
    assert move == "c6f3" and source == "top5"


def test_verified_move_falls_back_to_the_legal_list():
    board = chess.Board(FEN)
    board.push_uci("e4d6")
    text = "top5: a1a8:90\nbest: a1a8\nlegal: c6f3 c7d6"
    move, source = mav.verified_move(text, board)
    assert move in {"c6f3", "c7d6"} and source == "legal"


def test_verified_move_without_a_valid_candidate():
    board = chess.Board(FEN)
    board.push_uci("e4d6")
    text = "top5: a1a8:90\nbest: a1a8\nlegal: a1a8"
    move, source = mav.verified_move(text, board)
    assert move is None and source == "none"
