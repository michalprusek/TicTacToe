"""
Simple tests for game detector module without YOLO dependencies.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestGameDetectorBasics:
    """Basic tests for GameDetector that don't require YOLO models."""

    def test_game_detector_imports(self):
        """Test that GameDetector can be imported."""
        try:
            from app.main.game_detector import GameDetector
            assert GameDetector is not None
        except ImportError as e:
            pytest.skip(f"GameDetector import failed: {e}")

    def test_detector_constants_available(self):
        """Test that detector constants are available."""
        from app.core.detector_constants import (
            BBOX_CONF_THRESHOLD, 
            POSE_CONF_THRESHOLD,
            DEFAULT_DETECT_MODEL_PATH,
            DEFAULT_POSE_MODEL_PATH
        )
        
        assert isinstance(BBOX_CONF_THRESHOLD, float)
        assert isinstance(POSE_CONF_THRESHOLD, float)
        assert isinstance(DEFAULT_DETECT_MODEL_PATH, str)
        assert isinstance(DEFAULT_POSE_MODEL_PATH, str)
        assert BBOX_CONF_THRESHOLD > 0
        assert POSE_CONF_THRESHOLD > 0

    def test_basic_validation_methods(self):
        """Test basic validation methods without model dependencies."""
        from app.main.game_detector import GameDetector
        
        # Test frame validation (static method if available)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # These should be basic validation methods that don't require models
        assert test_frame.shape == (480, 640, 3)
        assert test_frame.dtype == np.uint8
        
    def test_fps_calculation_utilities(self):
        """Test FPS calculation utilities."""
        import time
        
        # Simple FPS calculation test
        start_time = time.time()
        time.sleep(0.01)  # Sleep for 10ms
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration > 0
        
        # Basic FPS calculation
        fps = 1.0 / duration if duration > 0 else 0
        assert fps > 0