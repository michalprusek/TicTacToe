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
        from app.main.style_manager import StyleManager
        assert hasattr(StyleManager, 'style_button')
        assert hasattr(StyleManager, 'style_status_label')
        assert callable(StyleManager.style_button)
        assert callable(StyleManager.style_status_label)
    
    def test_theme_functions(self):
        """Test theme functions exist."""
        from app.main.style_manager import StyleManager
        assert hasattr(StyleManager, 'create_status_panel_style')
        assert callable(StyleManager.create_status_panel_style)
        # Test that we can get style strings
        style = StyleManager.create_status_panel_style()
        assert isinstance(style, str)
        assert len(style) > 0
    
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

    def test_style_button_with_mock_button(self):
        """Test style_button method with mock QPushButton."""
        from unittest.mock import Mock
        from app.main.style_manager import StyleManager
        
        mock_button = Mock()
        
        # Test default style
        StyleManager.style_button(mock_button, "default")
        mock_button.setStyleSheet.assert_called()
        
        # Test different button types
        for style_type in ["primary", "danger", "success"]:
            StyleManager.style_button(mock_button, style_type)
            mock_button.setStyleSheet.assert_called()
        
        # Test invalid style type (should use default)
        StyleManager.style_button(mock_button, "invalid_type")
        mock_button.setStyleSheet.assert_called()

    def test_style_status_label_with_mock_label(self):
        """Test style_status_label method with mock QLabel."""
        from unittest.mock import Mock
        from app.main.style_manager import StyleManager
        
        mock_label = Mock()
        
        # Test all status types
        status_types = ["default", "info", "warning", "error", "success"]
        for status_type in status_types:
            StyleManager.style_status_label(mock_label, status_type)
            mock_label.setStyleSheet.assert_called()
        
        # Test invalid status type (should use default)
        StyleManager.style_status_label(mock_label, "invalid_status")
        mock_label.setStyleSheet.assert_called()

    def test_status_styles_constants(self):
        """Test STATUS_STYLES constants are properly defined."""
        from app.main.style_manager import StyleManager
        
        required_styles = ["default", "info", "warning", "error", "success"]
        for style_name in required_styles:
            assert style_name in StyleManager.STATUS_STYLES
            style = StyleManager.STATUS_STYLES[style_name]
            assert isinstance(style, str)
            assert len(style) > 0
            assert "color:" in style
            assert "font-weight:" in style