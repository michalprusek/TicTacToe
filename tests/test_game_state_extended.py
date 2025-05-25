"""
Extended tests for game_state module to improve coverage.
"""
import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock

from app.core.game_state import (
    GameState, EMPTY, PLAYER_X, PLAYER_O, TIE,
    GRID_POINTS_COUNT, robust_sort_grid_points
)


class TestGameStateExtended:
    """Extended test class for GameState functionality."""
    
    def test_grid_visibility_properties(self):
        """Test grid visibility related properties."""
        gs = GameState()
        
        # Initial state
        assert gs.grid_fully_visible is False
        assert gs.missing_grid_points_count == 0
        assert gs.game_paused_due_to_incomplete_grid is False
        
        # After reset
        gs.reset_game()
        assert gs.grid_fully_visible is False
        assert gs.missing_grid_points_count == 0
    
    def test_is_game_paused_due_to_incomplete_grid(self):
        """Test grid pause state checking."""
        gs = GameState()
        
        assert gs.is_game_paused_due_to_incomplete_grid() is False
        
        gs.game_paused_due_to_incomplete_grid = True
        assert gs.is_game_paused_due_to_incomplete_grid() is True    
    def test_is_game_over_methods(self):
        """Test various game over detection methods."""
        gs = GameState()
        
        # Not over initially
        assert gs.is_game_over() is False
        assert gs.is_game_over_due_to_error() is False
        
        # With winner
        gs.winner = PLAYER_X
        assert gs.is_game_over() is True
        
        # With fatal error
        gs.winner = None
        gs.set_error("FATAL: Critical error")
        assert gs.is_game_over_due_to_error() is True
    
    def test_getter_methods(self):
        """Test various getter methods."""
        gs = GameState()
        
        # Test getters
        assert gs.get_winner() is None
        assert gs.get_winning_line_indices() is None
        assert gs.get_error_message() is None
        assert gs.get_timestamp() == 0
        assert gs.get_homography() is None
        assert gs.get_transformed_grid_points_for_drawing() is None
        assert gs.get_cell_centers_uv_transformed() is None
        assert gs.get_current_frame() is None
        
        # After setting values
        gs.winner = PLAYER_X
        gs.winning_line_indices = [(0, 0), (0, 1), (0, 2)]
        gs.set_error("Test error")
        gs._timestamp = 123.45
        
        assert gs.get_winner() == PLAYER_X
        assert gs.get_winning_line_indices() == [(0, 0), (0, 1), (0, 2)]
        assert gs.get_error_message() == "Test error"
        assert gs.get_timestamp() == 123.45    
    def test_is_valid_board_validation(self):
        """Test board validation."""
        gs = GameState()
        
        # Valid board
        assert gs.is_valid() is True
        
        # Invalid symbol
        gs._board_state[0][0] = "invalid"
        assert gs.is_valid() is False
    
    def test_convert_symbols_to_expected_format(self):
        """Test symbol conversion to expected format."""
        gs = GameState()
        
        # Test detector format
        symbols = [{
            'box': [100, 100, 150, 150],
            'label': 'X',
            'confidence': 0.9,
            'class_id': 0
        }]
        
        converted = gs._convert_symbols_to_expected_format(symbols, {0: 'X'})
        assert len(converted) == 1
        assert converted[0]['player'] == 'X'
        assert converted[0]['confidence'] == 0.9
        assert np.array_equal(converted[0]['center_uv'], np.array([125, 125]))
    
    def test_convert_symbols_already_expected_format(self):
        """Test symbol conversion when already in expected format."""
        gs = GameState()
        
        symbols = [{
            'center_uv': np.array([100, 100]),
            'player': 'O',
            'confidence': 0.8
        }]
        
        converted = gs._convert_symbols_to_expected_format(symbols, {})
        assert len(converted) == 1
        assert converted[0]['player'] == 'O'
        assert converted[0]['confidence'] == 0.8    
    def test_update_from_detection_incomplete_grid(self):
        """Test update from detection with incomplete grid."""
        gs = GameState()
        
        # Update with None grid points
        gs.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=None,
            homography=None,
            detected_symbols=[],
            class_id_to_player={0: 'X', 1: 'O'},
            timestamp=1.0
        )
        
        assert gs._is_valid_grid is False
        assert gs.game_paused_due_to_incomplete_grid is True
        assert gs.error_message == gs.ERROR_GRID_INCOMPLETE_PAUSE
    
    def test_update_from_detection_insufficient_grid_points(self):
        """Test update with insufficient grid points."""
        gs = GameState()
        
        # Only 8 points instead of 16
        incomplete_points = np.array([[i*10, i*10] for i in range(8)], dtype=np.float32)
        
        gs.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=incomplete_points,
            homography=np.eye(3),
            detected_symbols=[],
            class_id_to_player={0: 'X', 1: 'O'},
            timestamp=1.0
        )
        
        assert gs._is_valid_grid is False
        assert gs.game_paused_due_to_incomplete_grid is True    
    def test_get_cell_center_uv(self):
        """Test getting cell center UV coordinates."""
        gs = GameState()
        
        # No cell centers computed
        center = gs.get_cell_center_uv(1, 1)
        assert center is None
        
        # Set up mock cell centers
        gs._cell_centers_uv_transformed = np.array([
            [100, 100], [200, 100], [300, 100],  # Row 0
            [100, 200], [200, 200], [300, 200],  # Row 1
            [100, 300], [200, 300], [300, 300]   # Row 2
        ], dtype=np.float32)
        
        # Test valid cell
        center = gs.get_cell_center_uv(1, 1)  # Middle cell
        np.testing.assert_array_equal(center, np.array([200, 200]))
        
        # Test invalid cell
        center = gs.get_cell_center_uv(5, 5)
        assert center is None
    
    def test_get_latest_derived_cell_polygons(self):
        """Test getting cell polygons."""
        gs = GameState()
        
        # No polygons initially
        polygons = gs.get_latest_derived_cell_polygons()
        assert polygons is None
        
        # Set mock polygons
        mock_polygons = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) for _ in range(9)]
        gs._cell_polygons_uv_transformed = mock_polygons
        
        polygons = gs.get_latest_derived_cell_polygons()
        assert polygons == mock_polygons