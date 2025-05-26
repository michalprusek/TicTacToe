# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.core.game_state module using pytest.
Tests GameState class with all methods and properties.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.core.game_state import (
    GameState, EMPTY, PLAYER_X, PLAYER_O, TIE,
    GUI_GRID_COLOR, GUI_X_COLOR, GUI_O_COLOR, GUI_LINE_THICKNESS, GUI_SYMBOL_THICKNESS,
    GRID_POINTS_COUNT, IDEAL_GRID_POINTS_CANONICAL
)


class TestGameStateConstants:
    """Test GameState constants."""
    
    def test_player_constants(self):
        """Test player and state constants."""
        assert EMPTY == ' '
        assert PLAYER_X == 'X'
        assert PLAYER_O == 'O'
        assert TIE == "TIE"
    
    def test_gui_colors(self):
        """Test GUI color constants."""
        assert GUI_GRID_COLOR == (255, 255, 255)  # White
        assert GUI_X_COLOR == (0, 0, 255)         # Red in BGR
        assert GUI_O_COLOR == (0, 255, 0)         # Green in BGR
        
        colors = [GUI_GRID_COLOR, GUI_X_COLOR, GUI_O_COLOR]
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for component in color:
                assert 0 <= component <= 255
    
    def test_gui_thickness_constants(self):
        """Test GUI thickness constants."""
        assert GUI_LINE_THICKNESS == 2
        assert GUI_SYMBOL_THICKNESS == 3
        assert isinstance(GUI_LINE_THICKNESS, int)
        assert isinstance(GUI_SYMBOL_THICKNESS, int)
    
    def test_grid_constants(self):
        """Test grid-related constants."""
        assert GRID_POINTS_COUNT == 16
        assert isinstance(GRID_POINTS_COUNT, int)
        
        assert isinstance(IDEAL_GRID_POINTS_CANONICAL, np.ndarray)
        assert IDEAL_GRID_POINTS_CANONICAL.shape == (16, 2)
        assert IDEAL_GRID_POINTS_CANONICAL.dtype == np.float32


class TestGameStateInitialization:
    """Test GameState initialization."""
    
    def test_init_default_state(self):
        """Test GameState initialization with default values."""
        game_state = GameState()
        
        # Board should be empty 3x3
        assert len(game_state.board) == 3
        assert all(len(row) == 3 for row in game_state.board)
        assert all(cell == EMPTY for row in game_state.board for cell in row)
        
        # Grid properties should be None/empty
        assert game_state.grid_points is None
        assert game_state._homography is None
        assert game_state._detection_results == []
        assert game_state._timestamp == 0
        assert game_state._is_valid_grid is False
        assert game_state._changed_cells_this_turn == []
        
        # Error and pause states
        assert game_state.error_message is None
        assert game_state.game_paused_due_to_incomplete_grid is False
        assert game_state.grid_fully_visible is False
        assert game_state.missing_grid_points_count == 0
        
        # Move timing
        assert game_state._last_move_timestamp is None
        assert game_state._move_cooldown_seconds == 1.0
        
        # Game result
        assert game_state.winner is None
        assert game_state.winning_line_indices is None
        
        # Other attributes
        assert game_state._previous_rotation_angle is None
        assert game_state._transformed_grid_points_for_drawing is None
        assert game_state._cell_centers_uv_transformed is None
        assert game_state._cell_polygons_uv_transformed is None
        assert game_state._frame is None
    
    def test_logger_initialization(self):
        """Test logger is properly initialized."""
        game_state = GameState()
        assert game_state.logger is not None
        assert hasattr(game_state.logger, 'info')
        assert hasattr(game_state.logger, 'debug')


class TestGameStateBoard:
    """Test board-related methods."""
    
    def test_board_property_returns_copy(self):
        """Test board property returns a copy, not reference."""
        game_state = GameState()
        board1 = game_state.board
        board2 = game_state.board
        
        # Should be equal but different objects
        assert board1 == board2
        assert board1 is not board2
        
        # Modifying returned board shouldn't affect internal state
        board1[0][0] = PLAYER_X
        assert game_state.board[0][0] == EMPTY
    
    def test_board_structure(self):
        """Test board has correct structure."""
        game_state = GameState()
        board = game_state.board
        
        assert isinstance(board, list)
        assert len(board) == 3
        for row in board:
            assert isinstance(row, list)
            assert len(row) == 3
            for cell in row:
                assert cell == EMPTY


class TestGameStateReset:
    """Test game state reset functionality."""
    
    def test_reset_game_clears_board(self):
        """Test reset_game clears the board."""
        game_state = GameState()
        
        # Simulate some moves
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_O
        
        game_state.reset_game()
        
        board = game_state.board
        assert all(cell == EMPTY for row in board for cell in row)
    
    def test_reset_game_clears_detection_data(self):
        """Test reset_game clears detection-related data."""
        game_state = GameState()
        
        # Set some detection data
        game_state._detection_results = [1, 2, 3]
        game_state._changed_cells_this_turn = [(0, 0), (1, 1)]
        game_state.error_message = "test error"
        game_state._last_move_timestamp = 123.456
        
        game_state.reset_game()
        
        assert game_state._detection_results == []
        assert game_state._changed_cells_this_turn == []
        assert game_state.error_message is None
        assert game_state._last_move_timestamp is None
    
    def test_reset_game_clears_pause_state(self):
        """Test reset_game clears pause and grid state."""
        game_state = GameState()
        
        # Set pause state
        game_state.game_paused_due_to_incomplete_grid = True
        game_state.grid_fully_visible = True
        game_state.missing_grid_points_count = 5
        
        game_state.reset_game()
        
        assert game_state.game_paused_due_to_incomplete_grid is False
        assert game_state.grid_fully_visible is False
        assert game_state.missing_grid_points_count == 0
    
    def test_reset_game_clears_winner_state(self):
        """Test reset_game clears winner and winning line."""
        game_state = GameState()
        
        # Set winner state
        game_state.winner = PLAYER_X
        game_state.winning_line_indices = [(0, 0), (0, 1), (0, 2)]
        
        game_state.reset_game()
        
        assert game_state.winner is None
        assert game_state.winning_line_indices is None
    
    def test_reset_game_keeps_grid_data(self):
        """Test reset_game keeps grid points and homography."""
        game_state = GameState()
        
        # Set grid data
        test_grid_points = np.array([[0, 0], [1, 1]], dtype=np.float32)
        test_homography = np.eye(3)
        game_state._grid_points = test_grid_points
        game_state._homography = test_homography
        game_state._is_valid_grid = True
        
        game_state.reset_game()
        
        # Grid data should be preserved
        np.testing.assert_array_equal(game_state._grid_points, test_grid_points)
        np.testing.assert_array_equal(game_state._homography, test_homography)
        assert game_state._is_valid_grid is True


class TestGameStateProperties:
    """Test GameState properties."""
    
    def test_grid_points_property(self):
        """Test grid_points property."""
        game_state = GameState()
        
        # Initially None
        assert game_state.grid_points is None
        
        # Set grid points
        test_points = np.array([[0, 0], [1, 1]], dtype=np.float32)
        game_state._grid_points = test_points
        
        # Should return the same array
        np.testing.assert_array_equal(game_state.grid_points, test_points)


class TestGameStateErrorStates:
    """Test error state handling."""
    
    def test_error_grid_incomplete_pause_constant(self):
        """Test error constant is properly defined."""
        assert hasattr(GameState, 'ERROR_GRID_INCOMPLETE_PAUSE')
        assert GameState.ERROR_GRID_INCOMPLETE_PAUSE == "GRID_INCOMPLETE_PAUSE_STATE"
    
    def test_error_message_handling(self):
        """Test error message can be set and retrieved."""
        game_state = GameState()
        
        # Initially None
        assert game_state.error_message is None
        
        # Set error message
        test_error = "Test error message"
        game_state.error_message = test_error
        assert game_state.error_message == test_error
    
    def test_pause_state_handling(self):
        """Test pause state can be managed."""
        game_state = GameState()
        
        # Initially not paused
        assert game_state.game_paused_due_to_incomplete_grid is False
        
        # Set paused
        game_state.game_paused_due_to_incomplete_grid = True
        assert game_state.game_paused_due_to_incomplete_grid is True
    
    def test_grid_visibility_tracking(self):
        """Test grid visibility tracking."""
        game_state = GameState()
        
        # Initially not fully visible
        assert game_state.grid_fully_visible is False
        assert game_state.missing_grid_points_count == 0
        
        # Set visibility state
        game_state.grid_fully_visible = True
        game_state.missing_grid_points_count = 3
        
        assert game_state.grid_fully_visible is True
        assert game_state.missing_grid_points_count == 3


class TestGameStateMoveCooldown:
    """Test move cooldown functionality."""
    
    def test_move_cooldown_initialization(self):
        """Test move cooldown is properly initialized."""
        game_state = GameState()
        
        assert game_state._last_move_timestamp is None
        assert game_state._move_cooldown_seconds == 1.0
    
    def test_move_cooldown_setting(self):
        """Test move cooldown can be set."""
        game_state = GameState()
        
        # Set cooldown
        game_state._move_cooldown_seconds = 2.5
        assert game_state._move_cooldown_seconds == 2.5
        
        # Set timestamp
        game_state._last_move_timestamp = 123.456
        assert game_state._last_move_timestamp == 123.456


class TestGameStateWinnerTracking:
    """Test winner and winning line tracking."""
    
    def test_winner_initialization(self):
        """Test winner tracking is properly initialized."""
        game_state = GameState()
        
        assert game_state.winner is None
        assert game_state.winning_line_indices is None
    
    def test_winner_setting(self):
        """Test winner can be set."""
        game_state = GameState()
        
        # Set winner
        game_state.winner = PLAYER_X
        assert game_state.winner == PLAYER_X
        
        # Set winning line
        winning_line = [(0, 0), (0, 1), (0, 2)]
        game_state.winning_line_indices = winning_line
        assert game_state.winning_line_indices == winning_line
    
    def test_winner_all_players(self):
        """Test winner can be set to all possible values."""
        game_state = GameState()
        
        # Test all possible winner values
        for winner in [PLAYER_X, PLAYER_O, TIE, None]:
            game_state.winner = winner
            assert game_state.winner == winner


class TestGameStateAdvancedProperties:
    """Test advanced GameState properties and attributes."""
    
    def test_rotation_angle_tracking(self):
        """Test rotation angle tracking."""
        game_state = GameState()
        
        # Initially None
        assert game_state._previous_rotation_angle is None
        
        # Set rotation angle
        game_state._previous_rotation_angle = 45.0
        assert game_state._previous_rotation_angle == 45.0
    
    def test_transformed_grid_points(self):
        """Test transformed grid points for drawing."""
        game_state = GameState()
        
        # Initially None
        assert game_state._transformed_grid_points_for_drawing is None
        
        # Set transformed points
        test_points = np.array([[100, 100], [200, 200]], dtype=np.float32)
        game_state._transformed_grid_points_for_drawing = test_points
        
        np.testing.assert_array_equal(
            game_state._transformed_grid_points_for_drawing, test_points
        )
    
    def test_cell_centers_and_polygons(self):
        """Test cell centers and polygons in UV space."""
        game_state = GameState()
        
        # Initially None
        assert game_state._cell_centers_uv_transformed is None
        assert game_state._cell_polygons_uv_transformed is None
        
        # Set cell centers
        test_centers = np.array([[50, 50], [150, 150]], dtype=np.float32)
        game_state._cell_centers_uv_transformed = test_centers
        
        # Set cell polygons
        test_polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        game_state._cell_polygons_uv_transformed = [test_polygon]
        
        np.testing.assert_array_equal(
            game_state._cell_centers_uv_transformed, test_centers
        )
        assert len(game_state._cell_polygons_uv_transformed) == 1
        np.testing.assert_array_equal(
            game_state._cell_polygons_uv_transformed[0], test_polygon
        )
    
    def test_frame_storage(self):
        """Test current frame storage."""
        game_state = GameState()
        
        # Initially None
        assert game_state._frame is None
        
        # Set frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        game_state._frame = test_frame
        
        np.testing.assert_array_equal(game_state._frame, test_frame)


class TestGameStateHomography:
    """Test homography-related functionality."""
    
    def test_homography_initialization(self):
        """Test homography is initially None."""
        game_state = GameState()
        assert game_state._homography is None
    
    def test_homography_setting(self):
        """Test homography can be set."""
        game_state = GameState()
        
        # Set homography matrix
        test_homography = np.eye(3, dtype=np.float32)
        game_state._homography = test_homography
        
        np.testing.assert_array_equal(game_state._homography, test_homography)
    
    def test_grid_validity_tracking(self):
        """Test grid validity tracking."""
        game_state = GameState()
        
        # Initially invalid
        assert game_state._is_valid_grid is False
        
        # Set valid
        game_state._is_valid_grid = True
        assert game_state._is_valid_grid is True


class TestGameStateChangedCells:
    """Test changed cells tracking."""
    
    def test_changed_cells_initialization(self):
        """Test changed cells list is initially empty."""
        game_state = GameState()
        assert game_state._changed_cells_this_turn == []
    
    def test_changed_cells_tracking(self):
        """Test changed cells can be tracked."""
        game_state = GameState()
        
        # Add changed cells
        game_state._changed_cells_this_turn.append((0, 0))
        game_state._changed_cells_this_turn.append((1, 2))
        
        assert len(game_state._changed_cells_this_turn) == 2
        assert (0, 0) in game_state._changed_cells_this_turn
        assert (1, 2) in game_state._changed_cells_this_turn
    
    def test_changed_cells_reset_on_game_reset(self):
        """Test changed cells are cleared on game reset."""
        game_state = GameState()
        
        # Add some changed cells
        game_state._changed_cells_this_turn = [(0, 0), (1, 1), (2, 2)]
        
        game_state.reset_game()
        
        assert game_state._changed_cells_this_turn == []