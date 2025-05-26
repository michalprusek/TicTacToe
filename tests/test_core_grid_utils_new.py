# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.core.grid_utils module using pytest.
Tests robust_sort_grid_points function with various scenarios.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch
from app.core.grid_utils import robust_sort_grid_points


class TestRobustSortGridPoints:
    """Test robust_sort_grid_points function."""
    
    def test_none_input(self):
        """Test with None input."""
        result = robust_sort_grid_points(None)
        assert result == (None, None)
    
    def test_invalid_point_count(self):
        """Test with incorrect number of points."""
        # Test with too few points
        points = np.array([[0, 0], [1, 1], [2, 2]])
        result = robust_sort_grid_points(points)
        assert result == (None, None)
        
        # Test with too many points
        points = np.array([[i, i] for i in range(20)])
        result = robust_sort_grid_points(points)
        assert result == (None, None)
    
    def test_valid_16_points_perfect_grid(self):
        """Test with perfect 4x4 grid points."""
        # Create perfect 4x4 grid
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 100, r * 100])
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
        assert sorted_points.shape == (16, 2)
        assert homography.shape == (3, 3)
    
    def test_valid_16_points_random_order(self):
        """Test with 16 grid points in random order."""
        # Create 4x4 grid and shuffle
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 50 + 10, r * 50 + 10])
        
        points = np.array(points, dtype=np.float32)
        np.random.shuffle(points)  # Random order
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
        assert sorted_points.shape == (16, 2)
    
    def test_with_custom_logger(self):
        """Test with custom logger."""
        logger = Mock(spec=logging.Logger)
        
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 30, r * 30])
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points, logger)
        
        assert sorted_points is not None
        assert homography is not None
        # Verify logger was called
        assert logger.debug.called
    
    def test_corner_detection_fallback(self):
        """Test fallback to minAreaRect when corner detection fails."""
        # Create points where corner detection might fail
        # (e.g., duplicate corner candidates)
        points = np.array([
            [0, 0], [0, 0], [100, 0], [100, 0],  # Duplicates
            [0, 50], [50, 50], [100, 50], [150, 50],
            [0, 100], [50, 100], [100, 100], [150, 100],
            [0, 150], [50, 150], [100, 150], [150, 150]
        ], dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
    
    @patch('cv2.findHomography')
    def test_preliminary_homography_failure(self, mock_find_homography):
        """Test when preliminary homography computation fails."""
        mock_find_homography.return_value = (None, None)
        
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 30, r * 30])
        points = np.array(points, dtype=np.float32)
        
        result = robust_sort_grid_points(points)
        assert result == (None, None)
    
    @patch('cv2.findHomography')
    def test_final_homography_failure(self, mock_find_homography):
        """Test when final homography computation fails."""
        # First call succeeds (preliminary), second fails (final)
        mock_find_homography.side_effect = [
            (np.eye(3), None),  # Preliminary succeeds
            (None, None)        # Final fails
        ]
        
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 30, r * 30])
        points = np.array(points, dtype=np.float32)
        
        result = robust_sort_grid_points(points)
        assert result == (None, None)
    
    def test_exception_handling(self):
        """Test exception handling in the function."""
        # Create invalid input that will cause exception
        with patch('app.core.grid_utils.np.array', side_effect=ValueError("Test error")):
            result = robust_sort_grid_points([[0, 0]] * 16)
            assert result == (None, None)
    
    def test_edge_case_points_close_together(self):
        """Test with points very close together."""
        points = []
        for r in range(4):
            for c in range(4):
                # Very small spacing
                points.append([c * 0.1, r * 0.1])
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        # Should still work even with small coordinates
        assert sorted_points is not None
        assert homography is not None
    
    def test_edge_case_negative_coordinates(self):
        """Test with negative coordinates."""
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 30 - 100, r * 30 - 100])
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
    
    def test_grid_indices_function_boundary_cases(self):
        """Test internal grid indices calculation with boundary cases."""
        points = []
        for r in range(4):
            for c in range(4):
                points.append([c * 100, r * 100])
        points = np.array(points, dtype=np.float32)
        
        # This should exercise the clamping logic in get_grid_indices
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
    
    def test_different_aspect_ratio_grid(self):
        """Test with non-square grid (different aspect ratio)."""
        points = []
        for r in range(4):
            for c in range(4):
                # Wide grid
                points.append([c * 150, r * 50])
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
    
    def test_rotated_grid(self):
        """Test with rotated grid points."""
        import math
        
        # Create grid and rotate it
        points = []
        angle = math.pi / 6  # 30 degrees
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        for r in range(4):
            for c in range(4):
                x, y = c * 50, r * 50
                # Apply rotation
                x_rot = x * cos_a - y * sin_a + 200
                y_rot = x * sin_a + y * cos_a + 200
                points.append([x_rot, y_rot])
        
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
    
    def test_perspective_distorted_grid(self):
        """Test with perspective-distorted grid points."""
        # Simulate perspective distortion
        points = [
            [10, 10], [110, 15], [205, 25], [295, 40],      # Top row - perspective distortion
            [15, 110], [115, 115], [210, 120], [300, 130],  # Second row
            [25, 205], [125, 210], [220, 215], [310, 225],  # Third row  
            [40, 295], [140, 300], [235, 305], [325, 315]   # Bottom row
        ]
        points = np.array(points, dtype=np.float32)
        
        sorted_points, homography = robust_sort_grid_points(points)
        
        assert sorted_points is not None
        assert homography is not None
    
    def test_logger_warning_messages(self):
        """Test that appropriate warning messages are logged."""
        logger = Mock(spec=logging.Logger)
        
        # Test invalid input logging
        robust_sort_grid_points(None, logger)
        logger.warning.assert_called()
        
        # Test with wrong number of points
        logger.reset_mock()
        robust_sort_grid_points([[0, 0]] * 5, logger)
        logger.warning.assert_called()