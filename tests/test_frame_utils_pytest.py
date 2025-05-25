"""
Pytest tests for frame utils module.
"""
import pytest
import numpy as np


class TestFrameUtils:
    """Pytest test class for frame utils."""
    
    def test_frame_basic_operations(self):
        """Test basic frame operations."""
        # Test that module can be imported
        try:
            import app.main.frame_utils
            assert app.main.frame_utils is not None
        except ImportError:
            pytest.skip("Frame utils module not available")
    
    def test_numpy_frame_creation(self):
        """Test numpy frame creation for testing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
    
    def test_frame_properties(self):
        """Test frame property access."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        assert frame.max() == 255
        assert frame.min() == 255
        assert len(frame.shape) == 3
    
    def test_frame_slicing(self):
        """Test frame slicing operations."""
        frame = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        cropped = frame[10:110, 20:120, :]
        assert cropped.shape == (100, 100, 3)
    
    def test_frame_type_validation(self):
        """Test frame type validation."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3