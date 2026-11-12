from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chess_rlvr.connect4 import (
    legal_columns, generate_positions, play, new_board, winner,
)


def test_winner_detects_four_directions():
    directions = (
        ((5, 0), (5, 1), (5, 2), (5, 3)),  # horizontal
        ((2, 6), (3, 6), (4, 6), (5, 6)),  # vertical
        ((2, 0), (3, 1), (4, 2), (5, 3)),  # rising diagonal
        ((2, 3), (3, 2), (4, 1), (5, 0)),  # falling diagonal
    )
    for squares in directions:
        board = new_board()
        for r, c in squares:
            board[r][c] = "X"
        assert winner(board) == "X"


def test_win_positions_have_a_winning_move():
    for position in generate_positions(10, "win", seed=42):
        assert winner(position["board"]) is None
        assert position["correct_columns"]
        for column in position["correct_columns"]:
            board = [line[:] for line in position["board"]]
            play(board, column, position["side"])
            assert winner(board) == position["side"]


def test_block_positions_have_a_single_threat_and_a_correct_reply():
    for position in generate_positions(10, "block", seed=42):
        board = position["board"]
        side = position["side"]
        opponent = "O" if side == "X" else "X"
        assert winner(board) is None
        threats = set()
        for column in legal_columns(board):
            attempt = [line[:] for line in board]
            play(attempt, column, opponent)
            if winner(attempt) == opponent:
                threats.add(column)
        assert threats == position["correct_columns"]
        for column in position["correct_columns"]:
            blocked = [line[:] for line in board]
            play(blocked, column, side)
            for next_column in legal_columns(blocked):
                response = [line[:] for line in blocked]
                play(response, next_column, opponent)
                assert winner(response) != opponent
