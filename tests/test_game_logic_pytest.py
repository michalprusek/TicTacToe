"""
Pytest tests for game_logic module.
"""
import pytest
import io
import sys
from unittest.mock import patch

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    create_board, get_available_moves, get_valid_moves,
    check_winner, get_winning_line, is_board_full, is_game_over,
    print_board, get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff
)


class TestGameLogicPytest:
    """Pytest test class for game logic."""
    
    def test_create_board(self):
        """Test board creation function."""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)
    
    def test_get_available_moves_comprehensive(self):
        """Test get_available_moves with various board states."""
        board = create_board()
        moves = get_available_moves(board)
        assert len(moves) == 9
        
        board[1][1] = PLAYER_X
        moves = get_available_moves(board)
        assert len(moves) == 8
        assert (1, 1) not in moves
    
    def test_check_winner_rows(self):
        """Test check_winner for row wins."""
        for row in range(3):
            board = create_board()
            for col in range(3):
                board[row][col] = PLAYER_X
            assert check_winner(board) == PLAYER_X
    
    def test_check_winner_columns(self):
        """Test check_winner for column wins."""
        for col in range(3):
            board = create_board()
            for row in range(3):
                board[row][col] = PLAYER_O
            assert check_winner(board) == PLAYER_O
    
    def test_get_other_player(self):
        """Test get_other_player function."""
        assert get_other_player(PLAYER_X) == PLAYER_O
        assert get_other_player(PLAYER_O) == PLAYER_X