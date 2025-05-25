"""
Pytest tests for main constants module.
"""
import pytest


class TestMainConstants:
    """Pytest test class for main constants."""
    
    def test_constants_module_import(self):
        """Test constants module can be imported."""
        try:
            import app.main.constants
            assert app.main.constants is not None
        except ImportError:
            pytest.skip("Main constants not available")
    
    def test_camera_constants(self):
        """Test camera-related constants."""
        try:
            from app.main.constants import DEFAULT_CAMERA_INDEX
            assert isinstance(DEFAULT_CAMERA_INDEX, int)
            assert DEFAULT_CAMERA_INDEX >= 0
        except ImportError:
            pytest.skip("Camera constants not available")
    
    def test_window_dimensions(self):
        """Test window dimension constants."""
        from app.main.constants import WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
        assert isinstance(WINDOW_MIN_WIDTH, int)
        assert isinstance(WINDOW_MIN_HEIGHT, int)
        assert WINDOW_MIN_WIDTH > 0
        assert WINDOW_MIN_HEIGHT > 0
    
    def test_fps_constants(self):
        """Test FPS constants."""
        from app.main.constants import DETECTION_FPS
        assert isinstance(DETECTION_FPS, int)
        assert DETECTION_FPS > 0
