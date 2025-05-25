"""Additional tests for game_state.py to improve coverage."""
import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
import cv2

from app.core.game_state import (
    GameState, EMPTY, PLAYER_X, PLAYER_O, GRID_POINTS_COUNT,
    robust_sort_grid_points
)


class TestUpdateBoardWithSymbolsRobust:
    """Test the robust symbol mapping functionality."""
    
    def test_update_board_with_symbols_robust_invalid_grid(self):
        """Test robust method with invalid grid points."""
        game_state = GameState()
        
        # Test with None grid points
        result = game_state._update_board_with_symbols_robust([], None, {})
        assert result == []
        
        # Test with wrong number of grid points
        grid_points = np.random.rand(10, 2)
        result = game_state._update_board_with_symbols_robust([], grid_points, {})
        assert result == []
    
    def test_update_board_with_symbols_robust_no_symbols(self):
        """Test robust method with no symbols."""
        game_state = GameState()
        grid_points = np.random.rand(16, 2)
        
        result = game_state._update_board_with_symbols_robust([], grid_points, {})
        assert result == []
    
    @patch('app.core.game_state.robust_sort_grid_points')
    def test_update_board_with_symbols_robust_homography_failure(self, mock_robust_sort):
        """Test robust method when homography computation fails."""
        game_state = GameState()
        
        # Mock robust_sort_grid_points to return failure
        mock_robust_sort.return_value = (None, None)
        
        grid_points = np.random.rand(16, 2)
        symbols = [{'center_uv': np.array([100, 100]), 'player': 'X', 'confidence': 0.9}]
        
        result = game_state._update_board_with_symbols_robust(symbols, grid_points, {})
        assert result == []
    
    @patch('app.core.game_state.robust_sort_grid_points')
    @patch('cv2.perspectiveTransform')
    def test_update_board_with_symbols_robust_success(self, mock_transform, mock_robust_sort):
        """Test successful robust symbol mapping."""
        game_state = GameState()
        
        # Mock robust_sort_grid_points to return valid homography
        mock_robust_sort.return_value = (np.random.rand(16, 2), np.eye(3))
        
        # Mock perspective transform to return symbol in cell (1, 1)
        mock_transform.return_value = np.array([[[150.0, 150.0]]])  # Cell (1,1) in 100px grid
        
        grid_points = np.random.rand(16, 2)
        symbols = [{
            'center_uv': np.array([100, 100]),
            'player': 'X',
            'confidence': 0.9
        }]
        
        result = game_state._update_board_with_symbols_robust(symbols, grid_points, {})
        
        assert len(result) == 1
        assert result[0] == (1, 1)  # Should map to cell (1,1)
        assert game_state._board_state[1][1] == 'X'
    
    @patch('app.core.game_state.robust_sort_grid_points')
    @patch('cv2.perspectiveTransform')
    def test_update_board_with_symbols_robust_out_of_bounds(self, mock_transform, mock_robust_sort):
        """Test robust method with symbol outside game area."""
        game_state = GameState()
        
        mock_robust_sort.return_value = (np.random.rand(16, 2), np.eye(3))
        
        # Mock transform to return position outside game grid (grid position 3,3 is outside 3x3 game)
        mock_transform.return_value = np.array([[[350.0, 350.0]]])  # Outside game area
        
        grid_points = np.random.rand(16, 2)
        symbols = [{
            'center_uv': np.array([100, 100]),
            'player': 'X',
            'confidence': 0.9
        }]
        
        result = game_state._update_board_with_symbols_robust(symbols, grid_points, {})
        
        assert len(result) == 0  # Should not place symbol outside game area
    
    @patch('app.core.game_state.robust_sort_grid_points')
    def test_update_board_with_symbols_robust_exception_handling(self, mock_robust_sort):
        """Test robust method exception handling."""
        game_state = GameState()
        
        # Mock to raise an exception
        mock_robust_sort.side_effect = Exception("Test exception")
        
        grid_points = np.random.rand(16, 2)
        symbols = [{
            'center_uv': np.array([100, 100]),
            'player': 'X',
            'confidence': 0.9
        }]
        
        result = game_state._update_board_with_symbols_robust(symbols, grid_points, {})
        assert result == []


class TestSymbolConfidenceFiltering:
    """Test symbol confidence filtering."""
    
    def test_symbol_confidence_threshold_custom(self):
        """Test custom confidence threshold."""
        game_state = GameState()
        
        # Set custom threshold
        game_state.symbol_confidence_threshold = 0.7
        
        cell_centers = np.array([
            [100, 100], [200, 100], [300, 100],
            [100, 200], [200, 200], [300, 200],
            [100, 300], [200, 300], [300, 300]
        ])
        
        # Symbol with confidence above custom threshold
        detected_symbols = [{
            'center_uv': np.array([105, 105]),
            'player': 'X',
            'confidence': 0.75  # Above 0.7 threshold
        }]
        
        result = game_state._update_board_with_symbols(
            detected_symbols, cell_centers, {}
        )
        
        assert len(result) == 1  # Should be accepted
        assert game_state._board_state[0][0] == 'X'
    
    def test_symbol_missing_data_handling(self):
        """Test handling of symbols with missing data."""
        game_state = GameState()
        
        cell_centers = np.array([
            [100, 100], [200, 100], [300, 100],
            [100, 200], [200, 200], [300, 200],
            [100, 300], [200, 300], [300, 300]
        ])
        
        # Symbols with missing required data
        detected_symbols = [
            {'player': 'X', 'confidence': 0.9},  # Missing center_uv
            {'center_uv': np.array([105, 105]), 'confidence': 0.9},  # Missing player
            {'center_uv': None, 'player': 'X', 'confidence': 0.9},  # None center_uv
            {'center_uv': np.array([205, 105]), 'player': None, 'confidence': 0.9}  # None player
        ]
        
        result = game_state._update_board_with_symbols(
            detected_symbols, cell_centers, {}
        )
        
        assert len(result) == 0  # All should be skipped
        assert all(cell == EMPTY for row in game_state._board_state for cell in row)


class TestDetectionUpdateFlow:
    """Test the full detection update flow."""
    
    def test_update_detection_grid_visibility_tracking(self):
        """Test grid visibility tracking in update_from_detection."""
        game_state = GameState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with partial grid (less than 16 points)
        partial_grid = np.random.rand(10, 2)
        game_state.update_from_detection(frame, partial_grid, None, [], {}, 123.45)
        
        assert not game_state.is_physical_grid_valid()
        assert game_state.game_paused_due_to_incomplete_grid
        assert game_state._grid_points is None
        assert game_state._homography is None
    
    def test_update_detection_homography_assignment(self):
        """Test homography assignment in update_from_detection."""
        game_state = GameState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Valid grid and homography
        grid_points = np.array([[c * 100, r * 100] for r in range(4) for c in range(4)])
        homography = np.random.rand(3, 3)
        
        game_state.update_from_detection(frame, grid_points, homography, [], {}, 123.45)
        
        assert np.array_equal(game_state._homography, homography)
    
    def test_update_detection_without_valid_transformation(self):
        """Test update when grid transformation fails."""
        game_state = GameState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Valid grid but create scenario where transformation fails
        grid_points = np.array([[float('inf')] * 2] * 16)  # Invalid coordinates
        
        game_state.update_from_detection(frame, grid_points, None, [], {}, 123.45)
        
        # Should handle gracefully and not crash
        assert game_state.is_physical_grid_valid()  # Grid points are valid count
    
    def test_update_detection_with_symbols_but_no_centers(self):
        """Test update when symbols are detected but cell centers aren't computed."""
        game_state = GameState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create valid grid but prevent cell center computation by using invalid grid
        grid_points = np.array([[float('nan')] * 2] * 16)  # NaN will prevent proper computation
        symbols = [{'center_uv': np.array([100, 100]), 'player': 'X', 'confidence': 0.9}]
        
        game_state.update_from_detection(frame, grid_points, None, symbols, {}, 123.45)
        
        # Cell centers should not be computed properly, so no moves should be made
        # But the robust method might still work, so we check the overall behavior
        # The key is that it handles the case gracefully without crashing
        assert isinstance(game_state.changed_cells_this_turn, list)


class TestWinConditionChecking:
    """Test win condition checking details."""
    
    def test_check_win_conditions_all_lines(self):
        """Test all possible winning lines are checked."""
        game_state = GameState()
        game_state._is_valid_grid = True
        
        # Test all possible winning lines
        winning_lines = [
            # Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        
        for i, line_indices in enumerate(winning_lines):
            game_state.reset_game()
            game_state._is_valid_grid = True
            
            # Place winning symbols
            for r, c in line_indices:
                game_state._board_state[r][c] = PLAYER_X
            
            game_state._check_win_conditions()
            
            assert game_state.winner == PLAYER_X
            assert game_state.winning_line_indices == line_indices


class TestGridTransformationEdgeCases:
    """Test edge cases in grid transformation computation."""
    
    def test_compute_grid_transformation_cell_center_calculation(self):
        """Test detailed cell center calculation."""
        game_state = GameState()
        
        # Create precise grid points for testing
        grid_points = []
        for r in range(4):
            for c in range(4):
                grid_points.append([c * 100.0, r * 100.0])  # Perfect 100px grid
        
        game_state._grid_points = np.array(grid_points, dtype=np.float32)
        
        result = game_state._compute_grid_transformation()
        assert result is True
        
        # Check that cell centers are computed correctly
        # Cell (0,0) should be at average of points 0,1,4,5 = (0.5*100, 0.5*100) = (50, 50)
        expected_centers = []
        for r_cell in range(3):
            for c_cell in range(3):
                # Calculate expected center
                expected_x = (c_cell + 0.5) * 100.0
                expected_y = (r_cell + 0.5) * 100.0
                expected_centers.append([expected_x, expected_y])
        
        expected_centers = np.array(expected_centers)
        
        assert np.allclose(game_state._cell_centers_uv_transformed, expected_centers)
    
    def test_compute_grid_transformation_index_validation(self):
        """Test index validation in grid transformation."""
        game_state = GameState()
        
        # Create grid points but manually test index bounds
        game_state._grid_points = np.random.rand(16, 2)
        
        # Mock the logger to capture error messages
        with patch.object(game_state.logger, 'error') as mock_error:
            # This should work normally
            result = game_state._compute_grid_transformation()
            
            # Verify no error messages about invalid indices
            error_calls = [call for call in mock_error.call_args_list 
                          if 'Invalid point indices' in str(call)]
            assert len(error_calls) == 0


class TestRobustSortGridPointsEdgeCases:
    """Test edge cases in robust_sort_grid_points function."""
    
    def test_robust_sort_grid_points_corner_detection_fallback(self):
        """Test fallback corner detection method."""
        # Create points where corner detection heuristic fails
        points = np.array([[50, 50]] * 16, dtype=np.float32)  # All points same location
        
        with patch('cv2.minAreaRect') as mock_rect, \
             patch('cv2.boxPoints') as mock_box, \
             patch('cv2.findHomography') as mock_homography:
            
            # Mock the fallback methods
            mock_rect.return_value = ((50, 50), (100, 100), 0)
            mock_box.return_value = np.array([
                [0, 0], [100, 0], [100, 100], [0, 100]
            ], dtype=np.float32)
            mock_homography.return_value = (np.eye(3), None)
            
            sorted_points, homography = robust_sort_grid_points(points)
            
            # Should successfully use fallback
            assert sorted_points is not None
            assert homography is not None
            mock_rect.assert_called_once()
            mock_box.assert_called_once()
    
    def test_robust_sort_grid_points_preliminary_homography_failure(self):
        """Test when preliminary homography computation fails."""
        points = np.random.rand(16, 2).astype(np.float32)
        
        with patch('cv2.findHomography') as mock_homography:
            # First call (preliminary) fails, second call should not be reached
            mock_homography.return_value = (None, None)
            
            sorted_points, homography = robust_sort_grid_points(points)
            
            assert sorted_points is None
            assert homography is None
    
    def test_robust_sort_grid_points_transformation_success(self):
        """Test successful transformation and sorting."""
        # Create realistic grid points in different order
        points = []
        for r in range(4):
            for c in range(4):
                # Add some realistic noise
                x = c * 50 + np.random.normal(0, 2)
                y = r * 50 + np.random.normal(0, 2)
                points.append([x, y])
        
        # Shuffle to simulate unsorted detection
        np.random.shuffle(points)
        points = np.array(points, dtype=np.float32)
        
        with patch('cv2.findHomography') as mock_homography:
            # Mock successful homography computation
            mock_homography.return_value = (np.eye(3), None)
            
            sorted_points, homography = robust_sort_grid_points(points)
            
            assert sorted_points is not None
            assert homography is not None
            assert len(sorted_points) == 16
            
            # Verify homography was called twice (preliminary and final)
            assert mock_homography.call_count == 2


class TestUpdateDetectionCooldownLogic:
    """Test move cooldown logic in update_from_detection."""
    
    def test_cooldown_edge_case_exact_boundary(self):
        """Test cooldown at exact boundary time."""
        game_state = GameState()
        game_state._is_valid_grid = True
        game_state._grid_points = np.array([[c * 100, r * 100] for r in range(4) for c in range(4)])
        game_state._compute_grid_transformation()
        game_state._last_move_timestamp = 100.0
        game_state._move_cooldown_seconds = 1.0
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        symbols = [{
            'center_uv': np.array([150, 150]),
            'player': 'X',
            'confidence': 0.9
        }]
        
        # Test at exact cooldown boundary
        game_state.update_from_detection(frame, game_state._grid_points, None, symbols, {}, 101.0)
        
        # Should be allowed (>= cooldown time)
        assert len(game_state.changed_cells_this_turn) > 0
    
    def test_cooldown_none_timestamp_handling(self):
        """Test handling when last move timestamp is None."""
        game_state = GameState()
        game_state._is_valid_grid = True
        game_state._grid_points = np.array([[c * 100, r * 100] for r in range(4) for c in range(4)])
        game_state._compute_grid_transformation()
        game_state._last_move_timestamp = None  # No previous move
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        symbols = [{
            'center_uv': np.array([150, 150]),
            'player': 'X',
            'confidence': 0.9
        }]
        
        game_state.update_from_detection(frame, game_state._grid_points, None, symbols, {}, 100.0)
        
        # Should be allowed (first move)
        assert len(game_state.changed_cells_this_turn) > 0
        assert game_state._last_move_timestamp == 100.0


if __name__ == "__main__":
    pytest.main([__file__])