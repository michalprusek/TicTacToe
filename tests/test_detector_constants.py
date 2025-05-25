"""
Pytest tests for detector_constants module.
"""
import pytest
import numpy as np

from app.core.detector_constants import (
    BBOX_CONF_THRESHOLD, POSE_CONF_THRESHOLD, KEYPOINT_VISIBLE_THRESHOLD,
    MIN_POINTS_FOR_HOMOGRAPHY, RANSAC_REPROJ_THRESHOLD,
    GRID_DIST_STD_DEV_THRESHOLD, GRID_ANGLE_TOLERANCE_DEG,
    MAX_GRID_DETECTION_RETRIES,
    DEFAULT_DETECT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH,
    DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_BBOX_THICKNESS, DEBUG_FPS_COLOR,
    WARNING_BG_COLOR, ERROR_BG_COLOR,
    IDEAL_GRID_NORM
)


class TestDetectorConstants:
    """Test class for detector constants."""
    
    def test_detection_thresholds(self):
        """Test detection threshold constants."""
        assert 0.0 <= BBOX_CONF_THRESHOLD <= 1.0
        assert 0.0 <= POSE_CONF_THRESHOLD <= 1.0
        assert 0.0 <= KEYPOINT_VISIBLE_THRESHOLD <= 1.0
        
        assert BBOX_CONF_THRESHOLD == 0.45
        assert POSE_CONF_THRESHOLD == 0.45
        assert KEYPOINT_VISIBLE_THRESHOLD == 0.3
    
    def test_homography_constants(self):
        """Test homography and RANSAC constants."""
        assert MIN_POINTS_FOR_HOMOGRAPHY == 4
        assert RANSAC_REPROJ_THRESHOLD == 10.0
        assert isinstance(MIN_POINTS_FOR_HOMOGRAPHY, int)
        assert isinstance(RANSAC_REPROJ_THRESHOLD, float)
        assert MIN_POINTS_FOR_HOMOGRAPHY >= 4
        assert RANSAC_REPROJ_THRESHOLD > 0
    
    def test_grid_validation_constants(self):
        """Test grid validation constants."""
        assert GRID_DIST_STD_DEV_THRESHOLD == 300.0
        assert GRID_ANGLE_TOLERANCE_DEG == 30.0
        assert MAX_GRID_DETECTION_RETRIES == 3
        
        assert GRID_DIST_STD_DEV_THRESHOLD > 0
        assert 0 < GRID_ANGLE_TOLERANCE_DEG <= 180
        assert MAX_GRID_DETECTION_RETRIES > 0    
    def test_model_paths(self):
        """Test default model path constants."""
        assert DEFAULT_DETECT_MODEL_PATH == "weights/best_detection.pt"
        assert DEFAULT_POSE_MODEL_PATH == "weights/best_pose.pt"
        
        assert isinstance(DEFAULT_DETECT_MODEL_PATH, str)
        assert isinstance(DEFAULT_POSE_MODEL_PATH, str)
        assert DEFAULT_DETECT_MODEL_PATH.endswith('.pt')
        assert DEFAULT_POSE_MODEL_PATH.endswith('.pt')
    
    def test_debug_colors(self):
        """Test debug color constants."""
        colors = [DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_FPS_COLOR,
                 WARNING_BG_COLOR, ERROR_BG_COLOR]
        
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for value in color:
                assert 0 <= value <= 255
                assert isinstance(value, int)
        
        assert DEBUG_BBOX_THICKNESS == 1
        assert isinstance(DEBUG_BBOX_THICKNESS, int)
        assert DEBUG_BBOX_THICKNESS > 0
    
    def test_ideal_grid_norm(self):
        """Test IDEAL_GRID_NORM array."""
        assert isinstance(IDEAL_GRID_NORM, np.ndarray)
        assert IDEAL_GRID_NORM.shape == (16, 2)
        assert IDEAL_GRID_NORM.dtype == np.float32
        
        # Test that coordinates are in expected range (0-3)
        assert np.all(IDEAL_GRID_NORM >= 0)
        assert np.all(IDEAL_GRID_NORM <= 3)
        
        # Test first and last points
        np.testing.assert_array_equal(IDEAL_GRID_NORM[0], [0, 0])
        np.testing.assert_array_equal(IDEAL_GRID_NORM[15], [3, 3])
    
    @pytest.mark.parametrize("threshold,expected", [
        (BBOX_CONF_THRESHOLD, 0.45),
        (POSE_CONF_THRESHOLD, 0.45),
        (KEYPOINT_VISIBLE_THRESHOLD, 0.3),
    ])
    def test_threshold_values_parametrized(self, threshold, expected):
        """Test threshold values using pytest parametrize."""
        assert threshold == expected
        assert 0.0 <= threshold <= 1.0