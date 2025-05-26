# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.core.detector_constants module using pytest.
Tests all detector constants and their values.
"""

import pytest
import numpy as np
from app.core.detector_constants import (
    BBOX_CONF_THRESHOLD, POSE_CONF_THRESHOLD, KEYPOINT_VISIBLE_THRESHOLD,
    MIN_POINTS_FOR_HOMOGRAPHY, RANSAC_REPROJ_THRESHOLD,
    GRID_DIST_STD_DEV_THRESHOLD, GRID_ANGLE_TOLERANCE_DEG,
    MAX_GRID_DETECTION_RETRIES, DEFAULT_DETECT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH,
    DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_BBOX_THICKNESS, DEBUG_FPS_COLOR,
    WARNING_BG_COLOR, ERROR_BG_COLOR, IDEAL_GRID_NORM
)


class TestDetectorConstants:
    """Test all detector constants."""
    
    def test_confidence_thresholds(self):
        """Test confidence threshold constants."""
        assert BBOX_CONF_THRESHOLD == 0.45
        assert POSE_CONF_THRESHOLD == 0.45
        assert KEYPOINT_VISIBLE_THRESHOLD == 0.3
        
        # Should be valid probabilities
        assert 0.0 <= BBOX_CONF_THRESHOLD <= 1.0
        assert 0.0 <= POSE_CONF_THRESHOLD <= 1.0
        assert 0.0 <= KEYPOINT_VISIBLE_THRESHOLD <= 1.0
        
        assert isinstance(BBOX_CONF_THRESHOLD, float)
        assert isinstance(POSE_CONF_THRESHOLD, float)
        assert isinstance(KEYPOINT_VISIBLE_THRESHOLD, float)
    
    def test_homography_constants(self):
        """Test homography and RANSAC constants."""
        assert MIN_POINTS_FOR_HOMOGRAPHY == 4
        assert RANSAC_REPROJ_THRESHOLD == 10.0
        
        assert isinstance(MIN_POINTS_FOR_HOMOGRAPHY, int)
        assert isinstance(RANSAC_REPROJ_THRESHOLD, float)
        assert MIN_POINTS_FOR_HOMOGRAPHY >= 4  # Minimum for homography
        assert RANSAC_REPROJ_THRESHOLD > 0
    
    def test_grid_validation_constants(self):
        """Test grid validation constants."""
        assert GRID_DIST_STD_DEV_THRESHOLD == 300.0
        assert GRID_ANGLE_TOLERANCE_DEG == 30.0
        
        assert isinstance(GRID_DIST_STD_DEV_THRESHOLD, float)
        assert isinstance(GRID_ANGLE_TOLERANCE_DEG, float)
        assert GRID_DIST_STD_DEV_THRESHOLD > 0
        assert 0 <= GRID_ANGLE_TOLERANCE_DEG <= 180
    
    def test_retry_constants(self):
        """Test retry constants."""
        assert MAX_GRID_DETECTION_RETRIES == 3
        assert isinstance(MAX_GRID_DETECTION_RETRIES, int)
        assert MAX_GRID_DETECTION_RETRIES > 0
    
    def test_model_paths(self):
        """Test model path constants."""
        assert DEFAULT_DETECT_MODEL_PATH == "weights/best_detection.pt"
        assert DEFAULT_POSE_MODEL_PATH == "weights/best_pose.pt"
        
        assert isinstance(DEFAULT_DETECT_MODEL_PATH, str)
        assert isinstance(DEFAULT_POSE_MODEL_PATH, str)
        assert DEFAULT_DETECT_MODEL_PATH.endswith('.pt')
        assert DEFAULT_POSE_MODEL_PATH.endswith('.pt')
        assert 'detection' in DEFAULT_DETECT_MODEL_PATH.lower()
        assert 'pose' in DEFAULT_POSE_MODEL_PATH.lower()
    
    def test_debug_colors(self):
        """Test debug drawing color constants."""
        assert DEBUG_UV_KPT_COLOR == (0, 255, 255)  # Yellow
        assert DEBUG_BBOX_COLOR == (255, 255, 255)  # White
        assert DEBUG_FPS_COLOR == (0, 255, 0)       # Green
        
        colors = [DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_FPS_COLOR]
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for component in color:
                assert 0 <= component <= 255
                assert isinstance(component, int)
    
    def test_debug_thickness(self):
        """Test debug thickness constant."""
        assert DEBUG_BBOX_THICKNESS == 1
        assert isinstance(DEBUG_BBOX_THICKNESS, int)
        assert DEBUG_BBOX_THICKNESS > 0
    
    def test_message_colors(self):
        """Test message color constants."""
        assert WARNING_BG_COLOR == (25, 25, 150)  # Dark red
        assert ERROR_BG_COLOR == (0, 0, 100)      # Dark blue
        
        colors = [WARNING_BG_COLOR, ERROR_BG_COLOR]
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for component in color:
                assert 0 <= component <= 255
                assert isinstance(component, int)
    
    def test_ideal_grid_norm(self):
        """Test ideal grid coordinates."""
        assert isinstance(IDEAL_GRID_NORM, np.ndarray)
        assert IDEAL_GRID_NORM.dtype == np.float32
        assert IDEAL_GRID_NORM.shape == (16, 2)  # 4x4 grid = 16 points, 2D coords
        
        # Test grid point coordinates are in expected range
        assert np.all(IDEAL_GRID_NORM >= 0)
        assert np.all(IDEAL_GRID_NORM <= 3)
        
        # Test specific grid points
        expected_points = [(i % 4, i // 4) for i in range(16)]
        for i, expected_point in enumerate(expected_points):
            np.testing.assert_array_equal(IDEAL_GRID_NORM[i], expected_point)
    
    def test_ideal_grid_structure(self):
        """Test ideal grid has correct 4x4 structure."""
        # Extract x and y coordinates
        x_coords = IDEAL_GRID_NORM[:, 0]
        y_coords = IDEAL_GRID_NORM[:, 1]
        
        # Should have 4 unique x values (0, 1, 2, 3)
        unique_x = np.unique(x_coords)
        assert len(unique_x) == 4
        np.testing.assert_array_equal(unique_x, [0, 1, 2, 3])
        
        # Should have 4 unique y values (0, 1, 2, 3)
        unique_y = np.unique(y_coords)
        assert len(unique_y) == 4
        np.testing.assert_array_equal(unique_y, [0, 1, 2, 3])
        
        # Each combination should appear exactly once
        for x in range(4):
            for y in range(4):
                matches = np.where((x_coords == x) & (y_coords == y))[0]
                assert len(matches) == 1, f"Point ({x}, {y}) should appear exactly once"
    
    def test_constants_immutability(self):
        """Test that constants are appropriate types."""
        # Numeric constants
        numeric_constants = [
            BBOX_CONF_THRESHOLD, POSE_CONF_THRESHOLD, KEYPOINT_VISIBLE_THRESHOLD,
            RANSAC_REPROJ_THRESHOLD, GRID_DIST_STD_DEV_THRESHOLD, GRID_ANGLE_TOLERANCE_DEG
        ]
        for const in numeric_constants:
            assert isinstance(const, (int, float))
        
        # Integer constants
        int_constants = [MIN_POINTS_FOR_HOMOGRAPHY, MAX_GRID_DETECTION_RETRIES, DEBUG_BBOX_THICKNESS]
        for const in int_constants:
            assert isinstance(const, int)
        
        # String constants
        string_constants = [DEFAULT_DETECT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH]
        for const in string_constants:
            assert isinstance(const, str)
        
        # Tuple constants (immutable)
        tuple_constants = [DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_FPS_COLOR, WARNING_BG_COLOR, ERROR_BG_COLOR]
        for const in tuple_constants:
            assert isinstance(const, tuple)
        
        # NumPy array (check it's read-only would be ideal, but not enforced)
        assert isinstance(IDEAL_GRID_NORM, np.ndarray)
    
    def test_threshold_relationships(self):
        """Test logical relationships between thresholds."""
        # All confidence thresholds should be reasonable
        assert BBOX_CONF_THRESHOLD < 1.0  # Not requiring perfect confidence
        assert POSE_CONF_THRESHOLD < 1.0
        assert KEYPOINT_VISIBLE_THRESHOLD < 1.0
        
        # Keypoint threshold might be lower than detection thresholds
        assert KEYPOINT_VISIBLE_THRESHOLD <= max(BBOX_CONF_THRESHOLD, POSE_CONF_THRESHOLD)
    
    def test_color_semantic_meaning(self):
        """Test colors have appropriate semantic meaning."""
        # Yellow for keypoints (high visibility)
        assert DEBUG_UV_KPT_COLOR[1] == 255  # Green component high
        assert DEBUG_UV_KPT_COLOR[2] == 255  # Red component high
        
        # White for bounding box (neutral)
        assert all(c == 255 for c in DEBUG_BBOX_COLOR)
        
        # Green for FPS (positive/good)
        assert DEBUG_FPS_COLOR[1] == 255  # Green component high
        assert DEBUG_FPS_COLOR[0] == 0    # Blue component low
        assert DEBUG_FPS_COLOR[2] == 0    # Red component low