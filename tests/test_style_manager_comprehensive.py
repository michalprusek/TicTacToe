"""
Comprehensive pytest test suite for style_manager module.
Tests all styling methods, color variations, and widget integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys

from app.main.style_manager import StyleManager


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for testing PyQt widgets."""
    if not QApplication.instance():
        app = QApplication(sys.argv)
        yield app
        app.quit()
    else:
        yield QApplication.instance()


class TestStyleManager:
    """Test StyleManager class functionality."""
    
    def test_class_constants_exist(self):
        """Test that all required class constants are defined."""
        assert hasattr(StyleManager, 'BASE_BUTTON_STYLE')
        assert hasattr(StyleManager, 'STATUS_STYLES')
        assert isinstance(StyleManager.BASE_BUTTON_STYLE, str)
        assert isinstance(StyleManager.STATUS_STYLES, dict)
    
    def test_base_button_style_content(self):
        """Test BASE_BUTTON_STYLE contains expected CSS properties."""
        style = StyleManager.BASE_BUTTON_STYLE
        assert "background-color: #3498db" in style
        assert "color: white" in style
        assert "border: none" in style
        assert "border-radius: 5px" in style
        assert "padding: 8px 16px" in style
        assert "font-weight: bold" in style
        assert "QPushButton:hover" in style
        assert "QPushButton:pressed" in style
    
    def test_status_styles_all_types(self):
        """Test STATUS_STYLES contains all expected status types."""
        expected_types = ["default", "info", "warning", "error", "success"]
        for status_type in expected_types:
            assert status_type in StyleManager.STATUS_STYLES
            assert isinstance(StyleManager.STATUS_STYLES[status_type], str)
    
    def test_status_styles_color_codes(self):
        """Test STATUS_STYLES contains correct color codes."""
        styles = StyleManager.STATUS_STYLES
        assert "#333333" in styles["default"]  # Gray
        assert "#3498db" in styles["info"]     # Blue
        assert "#f39c12" in styles["warning"]  # Orange
        assert "#e74c3c" in styles["error"]    # Red
        assert "#27ae60" in styles["success"]  # Green


class TestStyleButton:
    """Test style_button method."""
    
    def test_style_button_default(self, qapp):
        """Test applying default button style."""
        button = QPushButton("Test")
        StyleManager.style_button(button)
        
        style = button.styleSheet()
        assert "#3498db" in style
        assert "color: white" in style
    
    def test_style_button_primary(self, qapp):
        """Test applying primary button style (same as default)."""
        button = QPushButton("Test")
        StyleManager.style_button(button, "primary")
        
        style = button.styleSheet()
        assert "#3498db" in style
    
    def test_style_button_danger(self, qapp):
        """Test applying danger button style."""
        button = QPushButton("Test")
        StyleManager.style_button(button, "danger")
        
        style = button.styleSheet()
        assert "#e74c3c" in style  # Red color
        assert "#c0392b" in style  # Darker red for hover
        assert "#a93226" in style  # Darkest red for pressed
    
    def test_style_button_success(self, qapp):
        """Test applying success button style."""
        button = QPushButton("Test")
        StyleManager.style_button(button, "success")
        
        style = button.styleSheet()
        assert "#27ae60" in style  # Green color
        assert "#229954" in style  # Darker green for hover
        assert "#1e8449" in style  # Darkest green for pressed
    
    def test_style_button_invalid_type(self, qapp):
        """Test applying invalid style type falls back to default."""
        button = QPushButton("Test")
        StyleManager.style_button(button, "invalid_type")
        
        style = button.styleSheet()
        assert "#3498db" in style  # Should fall back to default blue
    
    def test_style_button_none_type(self, qapp):
        """Test applying None style type uses default."""
        button = QPushButton("Test")
        StyleManager.style_button(button, None)
        
        style = button.styleSheet()
        assert "#3498db" in style
    
    def test_style_button_empty_string(self, qapp):
        """Test applying empty string style type uses default."""
        button = QPushButton("Test")
        StyleManager.style_button(button, "")
        
        style = button.styleSheet()
        assert "#3498db" in style
    
    @patch('app.main.style_manager.QPushButton')
    def test_style_button_mock_setStyleSheet(self, mock_button_class):
        """Test that setStyleSheet is called with correct style."""
        mock_button = Mock()
        
        StyleManager.style_button(mock_button, "danger")
        
        mock_button.setStyleSheet.assert_called_once()
        call_args = mock_button.setStyleSheet.call_args[0][0]
        assert "#e74c3c" in call_args


class TestStyleStatusLabel:
    """Test style_status_label method."""
    
    def test_style_status_label_default(self, qapp):
        """Test applying default status label style."""
        label = QLabel("Test Status")
        StyleManager.style_status_label(label)
        
        style = label.styleSheet()
        assert "#333333" in style
        assert "font-weight: bold" in style
        assert "font-size: 14px" in style
    
    def test_style_status_label_info(self, qapp):
        """Test applying info status label style."""
        label = QLabel("Info")
        StyleManager.style_status_label(label, "info")
        
        style = label.styleSheet()
        assert "#3498db" in style
    
    def test_style_status_label_warning(self, qapp):
        """Test applying warning status label style."""
        label = QLabel("Warning")
        StyleManager.style_status_label(label, "warning")
        
        style = label.styleSheet()
        assert "#f39c12" in style
    
    def test_style_status_label_error(self, qapp):
        """Test applying error status label style."""
        label = QLabel("Error")
        StyleManager.style_status_label(label, "error")
        
        style = label.styleSheet()
        assert "#e74c3c" in style
    
    def test_style_status_label_success(self, qapp):
        """Test applying success status label style."""
        label = QLabel("Success")
        StyleManager.style_status_label(label, "success")
        
        style = label.styleSheet()
        assert "#27ae60" in style
    
    def test_style_status_label_invalid_type(self, qapp):
        """Test applying invalid status type falls back to default."""
        label = QLabel("Test")
        StyleManager.style_status_label(label, "invalid_status")
        
        style = label.styleSheet()
        assert "#333333" in style  # Should fall back to default gray
    
    def test_style_status_label_none_type(self, qapp):
        """Test applying None status type uses default."""
        label = QLabel("Test")
        StyleManager.style_status_label(label, None)
        
        style = label.styleSheet()
        assert "#333333" in style
    
    @patch('app.main.style_manager.QLabel')
    def test_style_status_label_mock_setStyleSheet(self, mock_label_class):
        """Test that setStyleSheet is called with correct style."""
        mock_label = Mock()
        
        StyleManager.style_status_label(mock_label, "error")
        
        mock_label.setStyleSheet.assert_called_once()
        call_args = mock_label.setStyleSheet.call_args[0][0]
        assert "#e74c3c" in call_args


class TestCreateStatusPanelStyle:
    """Test create_status_panel_style method."""
    
    def test_create_status_panel_style_returns_string(self):
        """Test that method returns a string."""
        result = StyleManager.create_status_panel_style()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_status_panel_style_content(self):
        """Test that status panel style contains expected CSS properties."""
        style = StyleManager.create_status_panel_style()
        assert "QWidget" in style
        assert "background-color: #f8f9fa" in style
        assert "border: 1px solid #dee2e6" in style
        assert "border-radius: 8px" in style
        assert "padding: 10px" in style
        assert "margin: 5px" in style
    
    def test_create_status_panel_style_consistent(self):
        """Test that method returns consistent results."""
        style1 = StyleManager.create_status_panel_style()
        style2 = StyleManager.create_status_panel_style()
        assert style1 == style2
    
    def test_create_status_panel_style_multiline(self):
        """Test that status panel style is properly formatted."""
        style = StyleManager.create_status_panel_style()
        lines = style.strip().split('\n')
        assert len(lines) > 1  # Should be multiline
        assert any('QWidget' in line for line in lines)


class TestStyleManagerIntegration:
    """Integration tests for StyleManager methods."""
    
    def test_multiple_widgets_styling(self, qapp):
        """Test styling multiple widgets with different types."""
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")
        label1 = QLabel("Label 1")
        label2 = QLabel("Label 2")
        
        StyleManager.style_button(button1, "primary")
        StyleManager.style_button(button2, "danger")
        StyleManager.style_status_label(label1, "info")
        StyleManager.style_status_label(label2, "error")
        
        # Verify different styles applied
        assert "#3498db" in button1.styleSheet()
        assert "#e74c3c" in button2.styleSheet()
        assert "#3498db" in label1.styleSheet()
        assert "#e74c3c" in label2.styleSheet()
    
    def test_widget_restying(self, qapp):
        """Test that widgets can be restyled with different types."""
        button = QPushButton("Test")
        
        StyleManager.style_button(button, "primary")
        assert "#3498db" in button.styleSheet()
        
        StyleManager.style_button(button, "danger")
        assert "#e74c3c" in button.styleSheet()
        assert "#3498db" not in button.styleSheet()
    
    def test_all_button_styles_applied(self, qapp):
        """Test that all button style types can be applied successfully."""
        button = QPushButton("Test")
        style_types = ["default", "primary", "danger", "success"]
        
        for style_type in style_types:
            StyleManager.style_button(button, style_type)
            style = button.styleSheet()
            assert len(style) > 0
            assert "QPushButton" in style
    
    def test_all_status_styles_applied(self, qapp):
        """Test that all status label style types can be applied successfully."""
        label = QLabel("Test")
        status_types = ["default", "info", "warning", "error", "success"]
        
        for status_type in status_types:
            StyleManager.style_status_label(label, status_type)
            style = label.styleSheet()
            assert len(style) > 0
            assert "font-weight: bold" in style
            assert "font-size: 14px" in style
    
    def test_status_panel_with_widgets(self, qapp):
        """Test combining status panel style with widget styling."""
        widget = QWidget()
        panel_style = StyleManager.create_status_panel_style()
        
        widget.setStyleSheet(panel_style)
        style = widget.styleSheet()
        
        assert "#f8f9fa" in style
        assert "border-radius: 8px" in style


@pytest.mark.parametrize("button_style", ["default", "primary", "danger", "success"])
def test_button_style_parametrized(qapp, button_style):
    """Parametrized test for all button style types."""
    button = QPushButton("Test")
    StyleManager.style_button(button, button_style)
    
    style = button.styleSheet()
    assert len(style) > 0
    assert "QPushButton" in style
    assert "background-color:" in style


@pytest.mark.parametrize("status_type", ["default", "info", "warning", "error", "success"])
def test_status_label_style_parametrized(qapp, status_type):
    """Parametrized test for all status label style types."""
    label = QLabel("Test")
    StyleManager.style_status_label(label, status_type)
    
    style = label.styleSheet()
    assert len(style) > 0
    assert "font-weight: bold" in style
    assert "font-size: 14px" in style
    assert "color:" in style