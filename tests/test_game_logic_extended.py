"""
Extended tests for game_logic module to improve coverage.
"""
import pytest
import io
import sys
from unittest.mock import patch

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    print_board, minimax, get_best_move
)


class TestGameLogicExtended:
    """Extended test class for game_logic functionality."""
    
    def test_print_board(self):
        """Test board printing functionality."""
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_O]
        ]
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_board(board)
            output = captured_output.getvalue()
            
            # Check that output contains board elements
            assert 'X' in output
            assert 'O' in output
            assert '|' in output  # Board borders
            assert '-' in output  # Board separators
            
        finally:
            sys.stdout = sys.__stdout__    
    def test_minimax_old_format_compatibility(self):
        """Test minimax with old parameter format."""
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # Old format: minimax(board, depth, is_maximizing, alpha, beta, ai_player)
        score, move = minimax(board, 0, True, float('-inf'), float('inf'), PLAYER_X)
        
        # X should win by playing at (0, 2)
        assert score > 0
        assert move == (0, 2)
    
    def test_minimax_new_format(self):
        """Test minimax with new parameter format."""
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # New format: minimax(board, player, depth, alpha, beta, ai_player)
        score, move = minimax(board, PLAYER_X, 0, float('-inf'), float('inf'), PLAYER_X)
        
        # X should win by playing at (0, 2)
        assert score > 0
        assert move == (0, 2)
    
    def test_minimax_alpha_beta_pruning(self):
        """Test that alpha-beta pruning works."""
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # This should trigger pruning in a complex game tree
        score, move = minimax(board, PLAYER_X, 0, float('-inf'), float('inf'), PLAYER_X)
        
        # Should return a valid move
        assert move is not None
        assert 0 <= move[0] <= 2
        assert 0 <= move[1] <= 2    
    def test_get_best_move_second_move_center(self):
        """Test best move chooses center on second move if available."""
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        move = get_best_move(board, PLAYER_O)
        assert move == (1, 1)  # Should choose center
    
    def test_get_best_move_center_occupied(self):
        """Test best move when center is occupied."""
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        move = get_best_move(board, PLAYER_X)
        # Should return some valid move (not center since it's occupied)
        assert move is not None
        assert move != (1, 1)
        assert board[move[0]][move[1]] == EMPTY
    
    def test_minimax_depth_scoring(self):
        """Test minimax depth-based scoring."""
        # AI wins immediately
        board = [
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        score, move = minimax(board, PLAYER_O, 0, float('-inf'), float('inf'), PLAYER_O)
        assert score == 9  # Win at depth 1 (10 - 1)
        assert move == (0, 2)  # Winning move
        
        # Human wins immediately (AI's perspective)
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        score, move = minimax(board, PLAYER_O, 0, float('-inf'), float('inf'), PLAYER_O)
        assert score < 0  # Negative score from AI perspective
