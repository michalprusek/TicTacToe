"""
Comprehensive tests for game_state module to improve coverage to 80%.
"""
import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
import cv2

from app.core.game_state import (
    GameState, EMPTY, PLAYER_X, PLAYER_O, TIE,
    GRID_POINTS_COUNT, IDEAL_GRID_POINTS_CANONICAL,
    robust_sort_grid_points, GUI_GRID_COLOR, GUI_X_COLOR, GUI_O_COLOR
)


class TestGameStateComprehensiveCoverage:
    """Comprehensive test coverage for GameState."""

    def test_constants(self):
        """Test module constants."""
        assert EMPTY == ' '
        assert PLAYER_X == 'X'
        assert PLAYER_O == 'O'
        assert TIE == "TIE"
        assert GRID_POINTS_COUNT == 16
        assert len(IDEAL_GRID_POINTS_CANONICAL) == 16
        assert GUI_GRID_COLOR == (255, 255, 255)
        assert GUI_X_COLOR == (0, 0, 255)
        assert GUI_O_COLOR == (0, 255, 0)

    def test_initialization_comprehensive(self):
        """Test comprehensive GameState initialization."""
        gs = GameState()
        
        # Board state
        assert gs._board_state == [[EMPTY] * 3 for _ in range(3)]
        assert gs.board == [[EMPTY] * 3 for _ in range(3)]
        
        # Grid and detection state
        assert gs._grid_points is None
        assert gs._homography is None
        assert gs._detection_results == []
        assert gs._timestamp == 0
        assert gs._is_valid_grid is False
        assert gs._changed_cells_this_turn == []
        
        # Error and game state
        assert gs.error_message is None
        assert gs.game_paused_due_to_incomplete_grid is False
        assert gs.grid_fully_visible is False
        assert gs.missing_grid_points_count == 0
        
        # Move cooldown
        assert gs._last_move_timestamp is None
        assert gs._move_cooldown_seconds == 1.0
        
        # Game result
        assert gs.winner is None
        assert gs.winning_line_indices is None
        
        # Transformation data
        assert gs._previous_rotation_angle is None
        assert gs._transformed_grid_points_for_drawing is None
        assert gs._cell_centers_uv_transformed is None
        assert gs._cell_polygons_uv_transformed is None

    def test_board_property(self):
        """Test board property returns copy."""
        gs = GameState()
        gs._board_state[0][0] = PLAYER_X
        
        board = gs.board
        # Modifying returned board shouldn't affect internal state
        board[0][1] = PLAYER_O
        
        assert gs._board_state[0][0] == PLAYER_X
        assert gs._board_state[0][1] == EMPTY

    def test_reset_game_comprehensive(self):
        """Test comprehensive game reset."""
        gs = GameState()
        
        # Set up some state
        gs._board_state[0][0] = PLAYER_X
        gs._detection_results = [{'test': 'data'}]
        gs._changed_cells_this_turn = [(0, 0)]
        gs.error_message = "test error"
        gs._last_move_timestamp = 123.0
        gs.game_paused_due_to_incomplete_grid = True
        gs.grid_fully_visible = True
        gs.missing_grid_points_count = 5
        gs.winner = PLAYER_X
        gs.winning_line_indices = [(0, 0), (0, 1), (0, 2)]
        
        # Reset
        gs.reset_game()
        
        # Check reset state
        assert gs._board_state == [[EMPTY] * 3 for _ in range(3)]
        assert gs._detection_results == []
        assert gs._changed_cells_this_turn == []
        assert gs.error_message is None
        assert gs._last_move_timestamp is None
        assert gs.game_paused_due_to_incomplete_grid is False
        assert gs.grid_fully_visible is False
        assert gs.missing_grid_points_count == 0
        assert gs.winner is None
        assert gs.winning_line_indices is None

    def test_is_valid_comprehensive(self):
        """Test board validation with various invalid symbols."""
        gs = GameState()
        
        # Valid board
        assert gs.is_valid() is True
        
        # Invalid symbol
        gs._board_state[0][0] = "invalid"
        assert gs.is_valid() is False
        
        # Another invalid symbol
        gs._board_state[0][0] = "Y"
        assert gs.is_valid() is False
        
        # Number as symbol
        gs._board_state[1][1] = 1
        assert gs.is_valid() is False

    def test_board_to_string_comprehensive(self):
        """Test board string representation."""
        gs = GameState()
        
        # Empty board - strip() removes all whitespace, so result is empty
        result = gs.board_to_string()
        expected = ""  # Empty after strip()
        assert result == expected
        
        # Board with mixed symbols
        gs._board_state = [
            [PLAYER_X, EMPTY, PLAYER_O],
            [EMPTY, PLAYER_X, EMPTY],
            [PLAYER_O, EMPTY, PLAYER_X]
        ]
        result = gs.board_to_string()
        expected = "X O\n X \nO X"
        assert result == expected
        
        # Full board
        gs._board_state = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        result = gs.board_to_string()
        expected = "XOX\nOXO\nXOX"
        assert result == expected

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling."""
        gs = GameState()
        
        # Set non-fatal error
        gs.set_error("Regular error")
        assert gs.get_error() == "Regular error"
        assert gs.is_error_active() is True
        assert not gs.is_game_over_due_to_error()
        
        # Set fatal error
        gs.set_error("FATAL: Critical error")
        assert gs.get_error() == "FATAL: Critical error"
        assert gs.is_error_active() is True
        assert gs.is_game_over_due_to_error() is True
        
        # Try to overwrite fatal with non-fatal (should fail)
        gs.set_error("Another regular error")
        assert gs.get_error() == "FATAL: Critical error"  # Unchanged
        
        # Overwrite with another fatal error (should work)
        gs.set_error("FATAL: Another critical error")
        assert gs.get_error() == "FATAL: Another critical error"
        
        # Clear error
        gs.clear_error_message()
        assert gs.get_error() is None
        assert gs.is_error_active() is False
        assert not gs.is_game_over_due_to_error()

    def test_get_methods_comprehensive(self):
        """Test all getter methods."""
        gs = GameState()
        
        # Set up test data
        gs.winner = PLAYER_X
        gs.winning_line_indices = [(0, 0), (0, 1), (0, 2)]
        gs._timestamp = 123.45
        gs._homography = np.eye(3)
        gs._transformed_grid_points_for_drawing = np.ones((16, 2))
        gs._cell_centers_uv_transformed = np.ones((9, 2))
        
        # Test getters
        assert gs.get_winner() == PLAYER_X
        assert gs.get_winning_line_indices() == [(0, 0), (0, 1), (0, 2)]
        assert gs.get_timestamp() == 123.45
        assert np.array_equal(gs.get_homography(), np.eye(3))
        assert gs.get_transformed_grid_points_for_drawing().shape == (16, 2)
        assert gs.get_cell_centers_uv_transformed().shape == (9, 2)
        
        # Test get_error_message alias
        gs.set_error("test error")
        assert gs.get_error_message() == "test error"

    def test_get_cell_center_uv(self):
        """Test getting specific cell center coordinates."""
        gs = GameState()
        
        # No cell centers computed
        assert gs.get_cell_center_uv(1, 1) is None
        
        # Set up mock cell centers
        gs._cell_centers_uv_transformed = np.array([
            [100, 100], [200, 100], [300, 100],  # Row 0
            [100, 200], [200, 200], [300, 200],  # Row 1
            [100, 300], [200, 300], [300, 300]   # Row 2
        ], dtype=np.float32)
        
        # Test valid cells
        center = gs.get_cell_center_uv(0, 0)
        np.testing.assert_array_equal(center, np.array([100, 100]))
        
        center = gs.get_cell_center_uv(1, 1)
        np.testing.assert_array_equal(center, np.array([200, 200]))
        
        center = gs.get_cell_center_uv(2, 2)
        np.testing.assert_array_equal(center, np.array([300, 300]))
        
        # Test invalid cells
        assert gs.get_cell_center_uv(3, 3) is None
        assert gs.get_cell_center_uv(-1, 0) is None

    def test_get_latest_derived_cell_polygons(self):
        """Test getting cell polygons."""
        gs = GameState()
        
        # No polygons initially
        assert gs.get_latest_derived_cell_polygons() is None
        
        # Set mock polygons
        mock_polygons = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) for _ in range(9)]
        gs._cell_polygons_uv_transformed = mock_polygons
        
        polygons = gs.get_latest_derived_cell_polygons()
        assert polygons == mock_polygons
        assert len(polygons) == 9

    def test_compute_grid_transformation_success(self):
        """Test successful grid transformation computation."""
        gs = GameState()
        
        # Set up valid 16 grid points
        gs._grid_points = np.array([
            # Row 0
            [100, 100], [200, 100], [300, 100], [400, 100],
            # Row 1  
            [100, 200], [200, 200], [300, 200], [400, 200],
            # Row 2
            [100, 300], [200, 300], [300, 300], [400, 300],
            # Row 3
            [100, 400], [200, 400], [300, 400], [400, 400]
        ], dtype=np.float32)
        
        # Test transformation
        success = gs._compute_grid_transformation()
        assert success is True
        assert gs._cell_centers_uv_transformed is not None
        assert len(gs._cell_centers_uv_transformed) == 9
        
        # Check specific cell centers
        # Cell (0,0) should be average of points [0,1,4,5]
        expected_center_00 = np.array([150.0, 150.0])  # Average of corners
        np.testing.assert_array_almost_equal(
            gs._cell_centers_uv_transformed[0], expected_center_00
        )

    def test_compute_grid_transformation_failure(self):
        """Test grid transformation computation failures."""
        gs = GameState()
        
        # No grid points
        assert gs._compute_grid_transformation() is False
        
        # Insufficient grid points
        gs._grid_points = np.array([[100, 100], [200, 200]])  # Only 2 points
        assert gs._compute_grid_transformation() is False
        
        # Invalid grid points (None)
        gs._grid_points = None
        assert gs._compute_grid_transformation() is False

    @patch('cv2.perspectiveTransform')
    def test_update_board_with_symbols_robust_success(self, mock_transform):
        """Test robust symbol update with successful homography."""
        gs = GameState()
        
        # Mock grid points
        gs._grid_points = np.ones((16, 2), dtype=np.float32)
        
        # Mock symbols
        symbols = [{
            'center_uv': np.array([250, 250]),
            'player': PLAYER_X,
            'confidence': 0.9
        }]
        
        # Mock perspective transform to return normalized coordinates
        mock_transform.return_value = np.array([[[150.0, 150.0]]])  # Cell (1,1)
        
        with patch('app.core.game_state.robust_sort_grid_points') as mock_robust:
            mock_robust.return_value = (gs._grid_points, np.eye(3))
            
            changed = gs._update_board_with_symbols_robust(symbols, gs._grid_points, {0: 'X'})
            
            # Should place symbol in cell (1,1)
            assert (1, 1) in changed
            assert gs._board_state[1][1] == PLAYER_X

    def test_update_board_with_symbols_robust_failures(self):
        """Test robust symbol update failure cases."""
        gs = GameState()
        
        # No grid points
        changed = gs._update_board_with_symbols_robust([], None, {})
        assert changed == []
        
        # Wrong number of grid points
        wrong_points = np.ones((8, 2))
        changed = gs._update_board_with_symbols_robust([], wrong_points, {})
        assert changed == []
        
        # No symbols
        grid_points = np.ones((16, 2))
        changed = gs._update_board_with_symbols_robust([], grid_points, {})
        assert changed == []

    def test_convert_symbols_to_expected_format(self):
        """Test symbol format conversion."""
        gs = GameState()
        
        # Test detector format conversion
        detector_symbols = [{
            'box': [100, 100, 150, 150],
            'label': 'X',
            'confidence': 0.9,
            'class_id': 0
        }]
        
        converted = gs._convert_symbols_to_expected_format(detector_symbols, {0: 'X'})
        assert len(converted) == 1
        assert converted[0]['player'] == 'X'
        assert converted[0]['confidence'] == 0.9
        np.testing.assert_array_equal(converted[0]['center_uv'], np.array([125, 125]))
        
        # Test already expected format
        expected_symbols = [{
            'center_uv': np.array([100, 100]),
            'player': 'O',
            'confidence': 0.8
        }]
        
        converted = gs._convert_symbols_to_expected_format(expected_symbols, {})
        assert len(converted) == 1
        assert converted[0]['player'] == 'O'
        
        # Test invalid format
        invalid_symbols = [{'invalid': 'format'}]
        converted = gs._convert_symbols_to_expected_format(invalid_symbols, {})
        assert len(converted) == 0

    def test_update_from_detection_incomplete_grid(self):
        """Test update with incomplete grid detection."""
        gs = GameState()
        
        # Test with None grid points
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
        
        # Test with insufficient grid points
        insufficient_points = np.array([[i*10, i*10] for i in range(8)])
        gs.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=insufficient_points,
            homography=None,
            detected_symbols=[],
            class_id_to_player={0: 'X', 1: 'O'},
            timestamp=2.0
        )
        
        assert gs._is_valid_grid is False
        assert gs.game_paused_due_to_incomplete_grid is True

    def test_update_from_detection_complete_grid(self):
        """Test update with complete grid detection."""
        gs = GameState()
        
        # First set incomplete grid
        gs.game_paused_due_to_incomplete_grid = True
        gs.error_message = gs.ERROR_GRID_INCOMPLETE_PAUSE
        
        # Valid 16 grid points
        valid_points = np.array([
            [100 + c*100, 100 + r*100] for r in range(4) for c in range(4)
        ], dtype=np.float32)
        
        gs.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=valid_points,
            homography=np.eye(3),
            detected_symbols=[],
            class_id_to_player={0: 'X', 1: 'O'},
            timestamp=3.0
        )
        
        assert gs._is_valid_grid is True
        assert gs.game_paused_due_to_incomplete_grid is False
        assert gs.error_message is None
        assert np.array_equal(gs._grid_points, valid_points)

    def test_move_cooldown(self):
        """Test move cooldown mechanism."""
        gs = GameState()
        
        # Set up valid grid and cell centers
        gs._is_valid_grid = True
        gs._cell_centers_uv_transformed = np.array([
            [150, 150], [250, 150], [350, 150],
            [150, 250], [250, 250], [350, 250],
            [150, 350], [250, 350], [350, 350]
        ], dtype=np.float32)
        
        # Mock successful robust symbol update
        with patch.object(gs, '_update_board_with_symbols_robust') as mock_update:
            mock_update.return_value = [(0, 0)]  # Simulate move made
            
            # First move should work
            gs.update_from_detection(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                ordered_kpts_uv=np.ones((16, 2)),
                homography=np.eye(3),
                detected_symbols=[],
                class_id_to_player={},
                timestamp=1.0
            )
            
            assert gs._last_move_timestamp == 1.0
            assert gs._changed_cells_this_turn == [(0, 0)]
            
            # Second move too soon (should be blocked by cooldown)
            mock_update.return_value = [(0, 1)]
            gs.update_from_detection(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                ordered_kpts_uv=np.ones((16, 2)),
                homography=np.eye(3),
                detected_symbols=[],
                class_id_to_player={},
                timestamp=1.5  # Only 0.5s later, less than 1.0s cooldown
            )
            
            assert gs._last_move_timestamp == 1.0  # Unchanged
            
            # Third move after cooldown should work
            mock_update.return_value = [(0, 1)]
            gs.update_from_detection(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                ordered_kpts_uv=np.ones((16, 2)),
                homography=np.eye(3),
                detected_symbols=[],
                class_id_to_player={},
                timestamp=2.5  # 1.5s later, more than 1.0s cooldown
            )
            
            assert gs._last_move_timestamp == 2.5

    def test_check_win_conditions_comprehensive(self):
        """Test comprehensive win condition checking."""
        gs = GameState()
        gs._is_valid_grid = True
        
        # Test no winner initially
        gs._check_win_conditions()
        assert gs.winner is None
        
        # Test row win
        gs._board_state[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        gs._check_win_conditions()
        assert gs.winner == PLAYER_X
        assert gs.winning_line_indices == [(0, 0), (0, 1), (0, 2)]
        
        # Reset for column test
        gs.winner = None
        gs.winning_line_indices = None
        gs._board_state = [[EMPTY] * 3 for _ in range(3)]
        
        # Test column win
        for row in range(3):
            gs._board_state[row][1] = PLAYER_O
        gs._check_win_conditions()
        assert gs.winner == PLAYER_O
        assert gs.winning_line_indices == [(0, 1), (1, 1), (2, 1)]
        
        # Reset for diagonal test
        gs.winner = None
        gs.winning_line_indices = None
        gs._board_state = [[EMPTY] * 3 for _ in range(3)]
        
        # Test diagonal win
        for i in range(3):
            gs._board_state[i][i] = PLAYER_X
        gs._check_win_conditions()
        assert gs.winner == PLAYER_X
        assert gs.winning_line_indices == [(0, 0), (1, 1), (2, 2)]
        
        # Reset for anti-diagonal test
        gs.winner = None
        gs.winning_line_indices = None
        gs._board_state = [[EMPTY] * 3 for _ in range(3)]
        
        # Test anti-diagonal win
        for i in range(3):
            gs._board_state[i][2-i] = PLAYER_O
        gs._check_win_conditions()
        assert gs.winner == PLAYER_O
        assert gs.winning_line_indices == [(0, 2), (1, 1), (2, 0)]
        
        # Reset for draw test
        gs.winner = None
        gs.winning_line_indices = None
        
        # Test draw
        gs._board_state = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        gs._check_win_conditions()
        assert gs.winner == "Draw"

    def test_check_win_conditions_skip_invalid_grid(self):
        """Test win condition checking skips when grid invalid."""
        gs = GameState()
        gs._is_valid_grid = False
        
        # Set up winning board
        gs._board_state[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        
        # Should not detect win due to invalid grid
        gs._check_win_conditions()
        assert gs.winner is None

    def test_check_win_conditions_existing_winner(self):
        """Test win condition checking with existing winner."""
        gs = GameState()
        gs._is_valid_grid = True
        gs.winner = PLAYER_X  # Already has winner
        
        # Set up different winning board
        gs._board_state[1] = [PLAYER_O, PLAYER_O, PLAYER_O]
        
        # Should not change existing winner
        gs._check_win_conditions()
        assert gs.winner == PLAYER_X  # Unchanged


class TestRobustSortGridPointsComprehensive:
    """Comprehensive tests for robust_sort_grid_points function."""

    def test_invalid_inputs(self):
        """Test invalid input handling."""
        # None input
        result, homography = robust_sort_grid_points(None)
        assert result is None
        assert homography is None
        
        # Wrong number of points
        points = np.array([[100, 100], [200, 200]])
        result, homography = robust_sort_grid_points(points)
        assert result is None
        assert homography is None
        
        # Empty array
        points = np.array([])
        result, homography = robust_sort_grid_points(points)
        assert result is None
        assert homography is None

    @patch('cv2.findHomography')
    def test_successful_corner_detection(self, mock_homography):
        """Test successful corner detection and processing."""
        # Mock homography computation
        mock_homography.return_value = (np.eye(3, dtype=np.float32), None)
        
        # Create 16 points in roughly 4x4 grid
        points = np.array([
            [100 + c*100, 100 + r*100] for r in range(4) for c in range(4)
        ], dtype=np.float32)
        
        result, homography = robust_sort_grid_points(points)
        
        assert result is not None
        assert homography is not None
        assert len(result) == 16
        assert homography.shape == (3, 3)

    @patch('cv2.findHomography')
    @patch('cv2.minAreaRect')
    @patch('cv2.boxPoints')
    def test_fallback_corner_detection(self, mock_box_points, mock_min_area_rect, mock_homography):
        """Test fallback corner detection when heuristics fail."""
        # Mock homography computation to succeed
        mock_homography.return_value = (np.eye(3, dtype=np.float32), None)
        
        # Mock minAreaRect and boxPoints for fallback
        mock_min_area_rect.return_value = ((200, 200), (200, 200), 0)
        mock_box_points.return_value = np.array([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ], dtype=np.float32)
        
        # Create points where corner heuristics might fail (all at same position)
        points = np.array([[150, 150]] * 16, dtype=np.float32)
        
        result, homography = robust_sort_grid_points(points)
        
        # Should use fallback method
        mock_min_area_rect.assert_called_once()
        mock_box_points.assert_called_once()
        
        assert result is not None
        assert homography is not None

    @patch('cv2.findHomography')
    def test_homography_failure(self, mock_homography):
        """Test handling of homography computation failure."""
        # Mock homography to fail
        mock_homography.return_value = (None, None)
        
        points = np.array([
            [100 + c*100, 100 + r*100] for r in range(4) for c in range(4)
        ], dtype=np.float32)
        
        result, homography = robust_sort_grid_points(points)
        
        assert result is None
        assert homography is None

    def test_exception_handling(self):
        """Test exception handling in robust_sort_grid_points."""
        # Create points that will cause cv2.findHomography to fail
        # All points at the same location
        points = np.array([[100, 100]] * 16, dtype=np.float32)
        
        with patch('cv2.findHomography') as mock_homography:
            # Force homography to fail
            mock_homography.return_value = (None, None)
            
            result, homography = robust_sort_grid_points(points)
            assert result is None
            assert homography is None

    def test_unique_corner_detection(self):
        """Test corner detection with unique indices."""
        # Create points in proper grid formation
        points = []
        for r in range(4):
            for c in range(4):
                points.append([100 + c*50, 100 + r*50])
        points = np.array(points, dtype=np.float32)
        
        with patch('cv2.findHomography') as mock_homography:
            mock_homography.return_value = (np.eye(3, dtype=np.float32), None)
            
            result, homography = robust_sort_grid_points(points)
            
            assert result is not None
            assert len(result) == 16

    def test_with_logger(self):
        """Test function with custom logger."""
        logger = logging.getLogger('test')
        
        # Invalid input with logger
        result, homography = robust_sort_grid_points(None, logger)
        assert result is None
        assert homography is None
        
        # Valid input with logger
        points = np.array([
            [100 + c*100, 100 + r*100] for r in range(4) for c in range(4)
        ], dtype=np.float32)
        
        with patch('cv2.findHomography') as mock_homography:
            mock_homography.return_value = (np.eye(3, dtype=np.float32), None)
            
            result, homography = robust_sort_grid_points(points, logger)
            assert result is not None
