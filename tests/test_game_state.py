# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Tests for game_state module.
"""
import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

from app.core.game_state import (
    GameState, EMPTY, PLAYER_X, PLAYER_O, TIE,
    GRID_POINTS_COUNT, IDEAL_GRID_POINTS_CANONICAL,
    robust_sort_grid_points
)


class TestGameState:
    """Test class for GameState functionality."""
    
    def test_initialization(self):
        """Test GameState initialization."""
        gs = GameState()
        
        # Check initial board state
        assert gs.board == [[EMPTY] * 3 for _ in range(3)]
        assert gs.grid_points is None
        assert gs.detection_results == []
        assert gs.changed_cells_this_turn == []
        assert gs.is_physical_grid_valid() is False
        assert gs.winner is None
        assert gs.error_message is None    
    def test_reset_game(self):
        """Test game reset functionality."""
        gs = GameState()
        
        # Set some state
        gs._board_state[0][0] = PLAYER_X
        gs._detection_results = [{'test': 'data'}]
        gs._changed_cells_this_turn = [(0, 0)]
        gs.error_message = "test error"
        gs.winner = PLAYER_X
        
        # Reset
        gs.reset_game()
        
        # Check everything is reset
        assert gs.board == [[EMPTY] * 3 for _ in range(3)]
        assert gs.detection_results == []
        assert gs.changed_cells_this_turn == []
        assert gs.error_message is None
        assert gs.winner is None
    
    def test_count_symbols(self):
        """Test symbol counting."""
        gs = GameState()
        
        # Empty board
        x_count, o_count = gs.count_symbols()
        assert x_count == 0
        assert o_count == 0
        
        # Add some symbols
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][1] = PLAYER_O
        gs._board_state[2][2] = PLAYER_X
        
        x_count, o_count = gs.count_symbols()
        assert x_count == 2
        assert o_count == 1    
    def test_is_valid_turn_sequence(self):
        """Test turn sequence validation."""
        gs = GameState()
        
        # Empty board is valid
        assert gs.is_valid_turn_sequence() is True
        
        # X starts (valid)
        gs._board_state[0][0] = PLAYER_X
        assert gs.is_valid_turn_sequence() is True
        
        # X then O (valid)
        gs._board_state[1][1] = PLAYER_O
        assert gs.is_valid_turn_sequence() is True
        
        # Two X, one O (valid)
        gs._board_state[2][2] = PLAYER_X
        assert gs.is_valid_turn_sequence() is True
        
        # Two O, two X (valid - equal count)
        gs._board_state[0][1] = PLAYER_O
        assert gs.is_valid_turn_sequence() is True    
    def test_get_valid_moves(self):
        """Test getting valid moves."""
        gs = GameState()
        
        # All moves valid on empty board
        valid_moves = gs.get_valid_moves()
        assert len(valid_moves) == 9
        assert (0, 0) in valid_moves
        assert (2, 2) in valid_moves
        
        # Place some symbols
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][1] = PLAYER_O
        
        valid_moves = gs.get_valid_moves()
        assert len(valid_moves) == 7
        assert (0, 0) not in valid_moves
        assert (1, 1) not in valid_moves
        assert (0, 1) in valid_moves    
    def test_check_winner_rows(self):
        """Test winner detection for rows."""
        gs = GameState()
        
        # No winner initially
        assert gs.check_winner() is None
        
        # X wins first row
        gs._board_state[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        assert gs.check_winner() == PLAYER_X
        
        # Reset and test O wins second row
        gs.reset_game()
        gs._board_state[1] = [PLAYER_O, PLAYER_O, PLAYER_O]
        assert gs.check_winner() == PLAYER_O
    
    def test_check_winner_columns(self):
        """Test winner detection for columns."""
        gs = GameState()
        
        # X wins first column
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][0] = PLAYER_X
        gs._board_state[2][0] = PLAYER_X
        assert gs.check_winner() == PLAYER_X
        
        # Reset and test O wins third column
        gs.reset_game()
        gs._board_state[0][2] = PLAYER_O
        gs._board_state[1][2] = PLAYER_O
        gs._board_state[2][2] = PLAYER_O
        assert gs.check_winner() == PLAYER_O    
    def test_check_winner_diagonals(self):
        """Test winner detection for diagonals."""
        gs = GameState()
        
        # X wins main diagonal
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][1] = PLAYER_X
        gs._board_state[2][2] = PLAYER_X
        assert gs.check_winner() == PLAYER_X
        
        # Reset and test O wins anti-diagonal
        gs.reset_game()
        gs._board_state[0][2] = PLAYER_O
        gs._board_state[1][1] = PLAYER_O
        gs._board_state[2][0] = PLAYER_O
        assert gs.check_winner() == PLAYER_O
    
    def test_is_board_full(self):
        """Test board full detection."""
        gs = GameState()
        
        assert gs.is_board_full() is False
        
        # Fill board without winner
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        gs._board_state = board
        assert gs.is_board_full() is True    
    def test_board_to_string(self):
        """Test board string representation."""
        gs = GameState()
        
        # Empty board - strip() removes trailing whitespace
        board_str = gs.board_to_string()
        expected = ""  # Empty string after strip
        assert board_str == expected
        
        # Board with symbols - strip() removes trailing whitespace
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][1] = PLAYER_O
        board_str = gs.board_to_string()
        expected = "X  \n O"  # Strip removes trailing newline and spaces
        assert board_str == expected
    
    def test_error_handling(self):
        """Test error message handling."""
        gs = GameState()
        
        # No error initially
        assert gs.get_error() is None
        assert gs.is_error_active() is False
        
        # Set error
        gs.set_error("Test error")
        assert gs.get_error() == "Test error"
        assert gs.is_error_active() is True
        
        # Clear error
        gs.clear_error_message()
        assert gs.get_error() is None
        assert gs.is_error_active() is False    
    def test_fatal_error_protection(self):
        """Test that fatal errors cannot be overwritten."""
        gs = GameState()
        
        # Set fatal error
        gs.set_error("FATAL: Critical error")
        assert gs.get_error() == "FATAL: Critical error"
        
        # Try to overwrite with non-fatal
        gs.set_error("Regular error")
        assert gs.get_error() == "FATAL: Critical error"  # Should not change
        
        # Can overwrite with another fatal
        gs.set_error("FATAL: Another critical error")
        assert gs.get_error() == "FATAL: Another critical error"


class TestRobustSortGridPoints:
    """Test class for robust_sort_grid_points function."""
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        # None input
        result, homography = robust_sort_grid_points(None)
        assert result is None
        assert homography is None
        
        # Wrong number of points
        points = np.array([[100, 100], [200, 200]])
        result, homography = robust_sort_grid_points(points)
        assert result is None
        assert homography is None    
    @patch('cv2.findHomography')
    def test_valid_grid_sorting(self, mock_find_homography):
        """Test valid grid point sorting."""
        # Mock successful homography computation
        mock_homography = np.eye(3, dtype=np.float32)
        mock_find_homography.return_value = (mock_homography, None)
        
        # Create sample grid points
        sample_grid_points = np.array([
            [100, 100], [200, 100], [300, 100], [400, 100],
            [100, 200], [200, 200], [300, 200], [400, 200],
            [100, 300], [200, 300], [300, 300], [400, 300],
            [100, 400], [200, 400], [300, 400], [400, 400]
        ], dtype=np.float32)
        
        result, homography = robust_sort_grid_points(sample_grid_points)
        
        # Should return valid results
        assert result is not None
        assert homography is not None
        assert len(result) == 16
        assert homography.shape == (3, 3)
