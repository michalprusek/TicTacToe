"""
Comprehensive tests for game_logic module to improve coverage.
"""
import pytest
import io
import sys
import math
from unittest.mock import patch, Mock

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    create_board, get_available_moves, get_valid_moves,
    check_winner, get_winning_line, is_board_full, is_game_over,
    print_board, get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff
)


class TestGameLogicComprehensive:
    """Comprehensive test class for game_logic module."""
    
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
        
        # One move made
        board[1][1] = PLAYER_X
        moves = get_available_moves(board)
        assert len(moves) == 8
        assert (1, 1) not in moves
        
        # Full board
        for r in range(3):
            for c in range(3):
                board[r][c] = PLAYER_X if (r + c) % 2 == 0 else PLAYER_O
        moves = get_available_moves(board)
        assert len(moves) == 0