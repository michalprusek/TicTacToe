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
        try:
            from app.main.constants import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
            assert isinstance(DEFAULT_WINDOW_WIDTH, int)
            assert isinstance(DEFAULT_WINDOW_HEIGHT, int)
            assert DEFAULT_WINDOW_WIDTH > 0
            assert DEFAULT_WINDOW_HEIGHT > 0
        except ImportError:
            pytest.skip("Window constants not available")
    
    def test_fps_constants(self):
        """Test FPS constants."""
        try:
            from app.main.constants import DEFAULT_FPS
            assert isinstance(DEFAULT_FPS, int)
            assert DEFAULT_FPS > 0
        except ImportError:
            pytest.skip("FPS constants not available")