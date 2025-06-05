# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Extended tests for GameDetector module without model dependencies.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.core.detector_constants import (
    BBOX_CONF_THRESHOLD,
    POSE_CONF_THRESHOLD,
    DEFAULT_DETECT_MODEL_PATH,
    DEFAULT_POSE_MODEL_PATH
)


class TestGameDetectorConstants:
    """Test GameDetector constants and basic functionality."""

    def test_detector_constants_values(self):
        """Test detector constant values are reasonable."""
        assert 0.0 <= BBOX_CONF_THRESHOLD <= 1.0
        assert 0.0 <= POSE_CONF_THRESHOLD <= 1.0
        assert isinstance(DEFAULT_DETECT_MODEL_PATH, str)
        assert isinstance(DEFAULT_POSE_MODEL_PATH, str)
        
        # Check file extensions
        assert DEFAULT_DETECT_MODEL_PATH.endswith('.pt')
        assert DEFAULT_POSE_MODEL_PATH.endswith('.pt')

    def test_model_paths_different(self):
        """Test that detection and pose model paths are different."""
        assert DEFAULT_DETECT_MODEL_PATH != DEFAULT_POSE_MODEL_PATH

    def test_confidence_thresholds_reasonable(self):
        """Test confidence thresholds are in reasonable range."""
        # High confidence thresholds for precise detection
        assert 0.8 <= BBOX_CONF_THRESHOLD <= 1.0
        assert 0.8 <= POSE_CONF_THRESHOLD <= 1.0

    def test_detector_constants_import(self):
        """Test that all required constants can be imported."""
        from app.core.detector_constants import (
            MIN_POINTS_FOR_HOMOGRAPHY,
            RANSAC_REPROJ_THRESHOLD,
            GRID_DIST_STD_DEV_THRESHOLD,
            GRID_ANGLE_TOLERANCE_DEG
        )
        
        assert isinstance(MIN_POINTS_FOR_HOMOGRAPHY, int)
        assert isinstance(RANSAC_REPROJ_THRESHOLD, float)
        assert isinstance(GRID_DIST_STD_DEV_THRESHOLD, float)
        assert isinstance(GRID_ANGLE_TOLERANCE_DEG, float)
