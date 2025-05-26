"""
Comprehensive tests for GUI Factory module.
Tests widget creation factories and standardized styling.
"""

import pytest
from unittest.mock import Mock, patch
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys

from app.main.gui_factory import ButtonFactory, LayoutFactory, LabelFactory


@pytest.fixture
def qt_app():
    """Create QApplication if it doesn't exist."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestButtonFactory:
    """Test ButtonFactory functionality."""

    @patch('app.main.gui_factory.StyleManager.style_button')
    def test_create_button_basic(self, mock_style_button, qt_app):
        """Test basic button creation."""
        button = ButtonFactory.create_button("Test Button")
        
        assert isinstance(button, QPushButton)
        assert button.text() == "Test Button"
        assert button.minimumWidth() == 100
        mock_style_button.assert_called_once_with(button, "default")

    @patch('app.main.gui_factory.StyleManager.style_button')
    def test_create_button_with_style(self, mock_style_button, qt_app):
        """Test button creation with custom style."""
        button = ButtonFactory.create_button("Test", style_type="primary")
        
        mock_style_button.assert_called_once_with(button, "primary")

    @patch('app.main.gui_factory.StyleManager.style_button')
    def test_create_button_with_click_handler(self, mock_style_button, qt_app):
        """Test button creation with click handler."""
        mock_handler = Mock()
        button = ButtonFactory.create_button("Test", click_handler=mock_handler)
        
        # Simulate button click
        button.clicked.emit()
        mock_handler.assert_called_once()

    @patch('app.main.gui_factory.StyleManager.style_button')
    def test_create_button_custom_dimensions(self, mock_style_button, qt_app):
        """Test button creation with custom font size and width."""
        button = ButtonFactory.create_button(
            "Test", font_size=14, min_width=200
        )
        
        assert button.minimumWidth() == 200
        font = button.font()
        assert font.pointSize() == 14
        assert font.weight() == QFont.Bold
        assert font.family() == "Arial"

    @patch('app.main.gui_factory.ButtonFactory.create_button')
    def test_create_control_button(self, mock_create_button, qt_app):
        """Test control button creation."""
        mock_handler = Mock()
        mock_button = Mock(spec=QPushButton)
        mock_create_button.return_value = mock_button
        
        result = ButtonFactory.create_control_button("Control", mock_handler)
        
        mock_create_button.assert_called_once_with(
            "Control", "primary", mock_handler,
            font_size=10, min_width=120
        )
        mock_button.setEnabled.assert_called_once_with(True)
        assert result == mock_button

    @patch('app.main.gui_factory.ButtonFactory.create_button')
    def test_create_control_button_disabled(self, mock_create_button, qt_app):
        """Test control button creation in disabled state."""
        mock_handler = Mock()
        mock_button = Mock(spec=QPushButton)
        mock_create_button.return_value = mock_button
        
        ButtonFactory.create_control_button("Control", mock_handler, enabled=False)
        
        mock_button.setEnabled.assert_called_once_with(False)

    @patch('app.main.gui_factory.ButtonFactory.create_button')
    def test_create_danger_button(self, mock_create_button, qt_app):
        """Test danger button creation."""
        mock_handler = Mock()
        mock_button = Mock(spec=QPushButton)
        mock_create_button.return_value = mock_button
        
        result = ButtonFactory.create_danger_button("Danger", mock_handler)
        
        mock_create_button.assert_called_once_with(
            "Danger", "danger", mock_handler,
            font_size=10, min_width=120
        )
        assert result == mock_button


class TestLayoutFactory:
    """Test LayoutFactory functionality."""

    def test_create_vbox_layout_default(self, qt_app):
        """Test vertical layout creation with defaults."""
        layout = LayoutFactory.create_vbox_layout()
        
        assert isinstance(layout, QVBoxLayout)
        assert layout.spacing() == 5
        margins = layout.getContentsMargins()
        assert margins == (5, 5, 5, 5)

    def test_create_vbox_layout_custom(self, qt_app):
        """Test vertical layout creation with custom parameters."""
        layout = LayoutFactory.create_vbox_layout(spacing=10, margins=(2, 4, 6, 8))
        
        assert layout.spacing() == 10
        margins = layout.getContentsMargins()
        assert margins == (2, 4, 6, 8)

    def test_create_hbox_layout_default(self, qt_app):
        """Test horizontal layout creation with defaults."""
        layout = LayoutFactory.create_hbox_layout()
        
        assert isinstance(layout, QHBoxLayout)
        assert layout.spacing() == 5
        margins = layout.getContentsMargins()
        assert margins == (5, 5, 5, 5)

    def test_create_hbox_layout_custom(self, qt_app):
        """Test horizontal layout creation with custom parameters."""
        layout = LayoutFactory.create_hbox_layout(spacing=15, margins=(1, 2, 3, 4))
        
        assert layout.spacing() == 15
        margins = layout.getContentsMargins()
        assert margins == (1, 2, 3, 4)

    def test_create_button_row(self, qt_app):
        """Test button row creation."""
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")
        buttons = [button1, button2]
        
        layout = LayoutFactory.create_button_row(buttons, spacing=20)
        
        assert isinstance(layout, QHBoxLayout)
        assert layout.spacing() == 20
        assert layout.count() == 3  # 2 buttons + 1 stretch
        assert layout.itemAt(0).widget() == button1
        assert layout.itemAt(1).widget() == button2

    def test_create_label_value_row(self, qt_app):
        """Test label-value row creation."""
        value_widget = QLabel("Value")
        
        layout = LayoutFactory.create_label_value_row("Label Text", value_widget)
        
        assert isinstance(layout, QHBoxLayout)
        assert layout.count() == 3  # label + value + stretch
        
        label_widget = layout.itemAt(0).widget()
        assert isinstance(label_widget, QLabel)
        assert label_widget.text() == "Label Text:"
        assert label_widget.minimumWidth() == 100
        
        assert layout.itemAt(1).widget() == value_widget


class TestLabelFactory:
    """Test LabelFactory functionality."""

    @patch('app.main.gui_factory.StyleManager.style_status_label')
    def test_create_status_label_default(self, mock_style_label, qt_app):
        """Test status label creation with defaults."""
        label = LabelFactory.create_status_label()
        
        assert isinstance(label, QLabel)
        assert label.text() == ""
        assert label.alignment() == Qt.AlignCenter
        mock_style_label.assert_called_once_with(label, "default")

    @patch('app.main.gui_factory.StyleManager.style_status_label')
    def test_create_status_label_custom(self, mock_style_label, qt_app):
        """Test status label creation with custom parameters."""
        label = LabelFactory.create_status_label("Status Text", "success")
        
        assert label.text() == "Status Text"
        mock_style_label.assert_called_once_with(label, "success")

    def test_create_info_label_default(self, qt_app):
        """Test info label creation with defaults."""
        label = LabelFactory.create_info_label("Info text")
        
        assert isinstance(label, QLabel)
        assert label.text() == "Info text"
        assert label.wordWrap() is True
        
        font = label.font()
        assert font.pointSize() == 10
        assert font.family() == "Arial"

    def test_create_info_label_custom_font(self, qt_app):
        """Test info label creation with custom font size."""
        label = LabelFactory.create_info_label("Info text", font_size=14)
        
        font = label.font()
        assert font.pointSize() == 14
