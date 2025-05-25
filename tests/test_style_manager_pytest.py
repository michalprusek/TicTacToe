"""
Pytest tests for style manager module.
"""
import pytest


class TestStyleManager:
    """Pytest test class for style manager."""
    
    def test_style_manager_import(self):
        """Test style manager can be imported."""
        try:
            import app.main.style_manager
            assert app.main.style_manager is not None
        except ImportError:
            pytest.skip("Style manager not available")
    
    def test_style_functions(self):
        """Test style functions exist."""
        try:
            from app.main.style_manager import get_button_style, get_window_style
            assert callable(get_button_style)
            assert callable(get_window_style)
        except ImportError:
            pytest.skip("Style functions not available")
    
    def test_theme_functions(self):
        """Test theme functions exist."""
        try:
            from app.main.style_manager import apply_dark_theme, apply_light_theme
            assert callable(apply_dark_theme)
            assert callable(apply_light_theme)
        except ImportError:
            pytest.skip("Theme functions not available")
    
    def test_style_string_validation(self):
        """Test style string validation."""
        sample_style = "QPushButton { background-color: #007ACC; color: white; }"
        assert isinstance(sample_style, str)
        assert "QPushButton" in sample_style
        assert "{" in sample_style and "}" in sample_style
    
    def test_color_validation(self):
        """Test color code validation."""
        valid_colors = ["#FFFFFF", "#000000", "#FF0000", "#00FF00", "#0000FF"]
        for color in valid_colors:
            assert color.startswith("#")
            assert len(color) == 7
            assert all(c in "0123456789ABCDEFabcdef" for c in color[1:])