# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Pytest tests for detector constants module.
"""
import pytest
from app.core.detector_constants import (
    BBOX_CONF_THRESHOLD, POSE_CONF_THRESHOLD, KEYPOINT_VISIBLE_THRESHOLD,
    MIN_POINTS_FOR_HOMOGRAPHY, RANSAC_REPROJ_THRESHOLD,
    DEFAULT_DETECT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH,
    DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_FPS_COLOR,
    WARNING_BG_COLOR, ERROR_BG_COLOR, IDEAL_GRID_NORM,
    GRID_DIST_STD_DEV_THRESHOLD, GRID_ANGLE_TOLERANCE_DEG,
    MAX_GRID_DETECTION_RETRIES
)


class TestDetectorConstants:
    """Test class for detector constants."""
    
    def test_detection_thresholds(self):
        """Test detection threshold constants."""
        assert 0.0 <= BBOX_CONF_THRESHOLD <= 1.0
        assert 0.0 <= POSE_CONF_THRESHOLD <= 1.0
        assert 0.0 <= KEYPOINT_VISIBLE_THRESHOLD <= 1.0
        
        assert BBOX_CONF_THRESHOLD == 0.90
        assert POSE_CONF_THRESHOLD == 0.80
        assert KEYPOINT_VISIBLE_THRESHOLD == 0.3
    
    def test_homography_parameters(self):
        """Test homography calculation parameters."""
        assert isinstance(MIN_POINTS_FOR_HOMOGRAPHY, int)
        assert MIN_POINTS_FOR_HOMOGRAPHY > 0
        assert MIN_POINTS_FOR_HOMOGRAPHY == 4
        
        assert isinstance(RANSAC_REPROJ_THRESHOLD, (int, float))
        assert RANSAC_REPROJ_THRESHOLD > 0
        assert RANSAC_REPROJ_THRESHOLD == 10.0
    
    def test_model_paths(self):
        """Test model file path constants."""
        assert isinstance(DEFAULT_DETECT_MODEL_PATH, str)
        assert isinstance(DEFAULT_POSE_MODEL_PATH, str)
        assert DEFAULT_DETECT_MODEL_PATH.endswith('.pt')
        assert DEFAULT_POSE_MODEL_PATH.endswith('.pt')
        assert 'weights' in DEFAULT_DETECT_MODEL_PATH
        assert 'weights' in DEFAULT_POSE_MODEL_PATH    
    def test_grid_parameters(self):
        """Test grid detection parameters."""
        assert isinstance(GRID_DIST_STD_DEV_THRESHOLD, (int, float))
        assert GRID_DIST_STD_DEV_THRESHOLD > 0
        assert GRID_DIST_STD_DEV_THRESHOLD == 300.0
        
        assert isinstance(GRID_ANGLE_TOLERANCE_DEG, (int, float))
        assert GRID_ANGLE_TOLERANCE_DEG > 0
        assert GRID_ANGLE_TOLERANCE_DEG == 30.0
        
        assert isinstance(MAX_GRID_DETECTION_RETRIES, int)
        assert MAX_GRID_DETECTION_RETRIES > 0
        assert MAX_GRID_DETECTION_RETRIES == 3
    
    def test_debug_colors(self):
        """Test debug visualization colors."""
        colors = [DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_FPS_COLOR, 
                 WARNING_BG_COLOR, ERROR_BG_COLOR]
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, int) for c in color)
            assert all(0 <= c <= 255 for c in color)
    
    def test_ideal_grid_norm(self):
        """Test ideal grid normalization array."""
        import numpy as np
        assert isinstance(IDEAL_GRID_NORM, np.ndarray)
        assert IDEAL_GRID_NORM.dtype == np.float32
        assert IDEAL_GRID_NORM.shape[1] == 2
    
    def test_bbox_thickness(self):
        """Test bbox thickness parameter."""
        from app.core.detector_constants import DEBUG_BBOX_THICKNESS
        assert isinstance(DEBUG_BBOX_THICKNESS, int)
        assert DEBUG_BBOX_THICKNESS > 0
        assert DEBUG_BBOX_THICKNESS == 1
