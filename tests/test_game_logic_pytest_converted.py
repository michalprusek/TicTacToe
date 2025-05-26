# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive pytest tests for game_logic module (converted from unittest).
"""
import io
import sys
import math
import pytest
from unittest.mock import patch, Mock

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    create_board, get_available_moves, get_valid_moves,
    check_winner, get_winning_line, is_board_full, is_game_over,
    print_board, get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff
)


class TestGameLogicPytestConverted:
    """Test class for game_logic module using pytest."""
    
    def test_create_board(self):
        """Test board creation function."""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)
    
    def test_get_available_moves_comprehensive(self):
        """Test get_available_moves with various board states."""
        # Empty board
        board = create_board()
        moves = get_available_moves(board)
        assert len(moves) == 9
        
        # Partially filled board
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        moves = get_available_moves(board)
        assert len(moves) == 7
        assert (0, 0) not in moves
        assert (1, 1) not in moves

    def test_check_winner_all_scenarios(self):
        """Test check_winner with all winning scenarios."""
        # Horizontal wins
        for row in range(3):
            board = create_board()
            for col in range(3):
                board[row][col] = PLAYER_X
            assert check_winner(board) == PLAYER_X
        
        # Vertical wins
        for col in range(3):
            board = create_board()
            for row in range(3):
                board[row][col] = PLAYER_O
            assert check_winner(board) == PLAYER_O

        # Diagonal wins
        board = create_board()
        for i in range(3):
            board[i][i] = PLAYER_X
        assert check_winner(board) == PLAYER_X

        board = create_board()
        for i in range(3):
            board[i][2-i] = PLAYER_O
        assert check_winner(board) == PLAYER_O

    def test_is_board_full(self):
        """Test is_board_full function."""
        board = create_board()
        assert not is_board_full(board)
        
        # Fill board
        for row in range(3):
            for col in range(3):
                board[row][col] = PLAYER_X
        assert is_board_full(board)

    def test_get_other_player(self):
        """Test get_other_player function."""
        assert get_other_player(PLAYER_X) == PLAYER_O
        assert get_other_player(PLAYER_O) == PLAYER_X

    def test_board_to_string(self):
        """Test board_to_string function."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        result = board_to_string(board)
        assert isinstance(result, str)
        assert PLAYER_X in result
        assert PLAYER_O in result
