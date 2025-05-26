# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Simple tests for GUI Factory module.
Tests factory patterns without actual PyQt5 widget creation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock PyQt5 modules globally
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()
sys.modules['app.main.style_manager'] = MagicMock()

from app.main.gui_factory import ButtonFactory, LayoutFactory, LabelFactory


class TestButtonFactoryMethods:
    """Test ButtonFactory method calls and parameters."""

    @patch('app.main.gui_factory.StyleManager')
    @patch('app.main.gui_factory.QPushButton')
    @patch('app.main.gui_factory.QFont')
    def test_create_button_method_calls(self, mock_qfont, mock_qpushbutton, mock_style_manager):
        """Test that create_button makes correct method calls."""
        mock_button = Mock()
        mock_qpushbutton.return_value = mock_button
        mock_font = Mock()
        mock_qfont.return_value = mock_font
        
        result = ButtonFactory.create_button("Test", "primary", None, 12, 150)
        
        # Verify QPushButton creation
        mock_qpushbutton.assert_called_once_with("Test")
        
        # Verify button configuration
        mock_button.setMinimumWidth.assert_called_once_with(150)
        mock_button.setFont.assert_called_once_with(mock_font)
        
        # Verify font configuration
        mock_qfont.assert_called_once_with("Arial", 12, mock_qfont.Bold)
        
        # Verify styling
        mock_style_manager.style_button.assert_called_once_with(mock_button, "primary")
        
        assert result == mock_button

    @patch('app.main.gui_factory.ButtonFactory.create_button')
    def test_create_control_button_calls_create_button(self, mock_create_button):
        """Test that create_control_button calls create_button with correct params."""
        mock_handler = Mock()
        mock_button = Mock()
        mock_create_button.return_value = mock_button
        
        result = ButtonFactory.create_control_button("Control", mock_handler, True)
        
        mock_create_button.assert_called_once_with(
            "Control", "primary", mock_handler,
            font_size=10, min_width=120
        )
        mock_button.setEnabled.assert_called_once_with(True)
        assert result == mock_button

    @patch('app.main.gui_factory.ButtonFactory.create_button')
    def test_create_danger_button_calls_create_button(self, mock_create_button):
        """Test that create_danger_button calls create_button with correct params."""
        mock_handler = Mock()
        mock_button = Mock()
        mock_create_button.return_value = mock_button
        
        result = ButtonFactory.create_danger_button("Danger", mock_handler)
        
        mock_create_button.assert_called_once_with(
            "Danger", "danger", mock_handler,
            font_size=10, min_width=120
        )
        assert result == mock_button


class TestLayoutFactoryMethods:
    """Test LayoutFactory method calls and parameters."""

    @patch('app.main.gui_factory.QVBoxLayout')
    def test_create_vbox_layout_method_calls(self, mock_qvboxlayout):
        """Test that create_vbox_layout makes correct method calls."""
        mock_layout = Mock()
        mock_qvboxlayout.return_value = mock_layout
        
        result = LayoutFactory.create_vbox_layout(10, (1, 2, 3, 4))
        
        mock_qvboxlayout.assert_called_once()
        mock_layout.setSpacing.assert_called_once_with(10)
        mock_layout.setContentsMargins.assert_called_once_with(1, 2, 3, 4)
        assert result == mock_layout

    @patch('app.main.gui_factory.QHBoxLayout')
    def test_create_hbox_layout_method_calls(self, mock_qhboxlayout):
        """Test that create_hbox_layout makes correct method calls."""
        mock_layout = Mock()
        mock_qhboxlayout.return_value = mock_layout
        
        result = LayoutFactory.create_hbox_layout(15, (5, 6, 7, 8))
        
        mock_qhboxlayout.assert_called_once()
        mock_layout.setSpacing.assert_called_once_with(15)
        mock_layout.setContentsMargins.assert_called_once_with(5, 6, 7, 8)
        assert result == mock_layout

    @patch('app.main.gui_factory.LayoutFactory.create_hbox_layout')
    def test_create_button_row_calls_hbox_layout(self, mock_create_hbox):
        """Test that create_button_row uses create_hbox_layout."""
        mock_layout = Mock()
        mock_create_hbox.return_value = mock_layout
        mock_buttons = [Mock(), Mock()]
        
        result = LayoutFactory.create_button_row(mock_buttons, 20)
        
        mock_create_hbox.assert_called_once_with(20)
        assert mock_layout.addWidget.call_count == 2
        mock_layout.addStretch.assert_called_once()
        assert result == mock_layout

    @patch('app.main.gui_factory.LayoutFactory.create_hbox_layout')
    @patch('app.main.gui_factory.QLabel')
    def test_create_label_value_row_method_calls(self, mock_qlabel, mock_create_hbox):
        """Test that create_label_value_row makes correct method calls."""
        mock_layout = Mock()
        mock_create_hbox.return_value = mock_layout
        mock_label = Mock()
        mock_qlabel.return_value = mock_label
        mock_value_widget = Mock()
        
        result = LayoutFactory.create_label_value_row("Test Label", mock_value_widget)
        
        mock_create_hbox.assert_called_once()
        mock_qlabel.assert_called_once_with("Test Label:")
        mock_label.setMinimumWidth.assert_called_once_with(100)
        assert mock_layout.addWidget.call_count == 2
        mock_layout.addStretch.assert_called_once()
        assert result == mock_layout


class TestLabelFactoryMethods:
    """Test LabelFactory method calls and parameters."""

    @patch('app.main.gui_factory.StyleManager')
    @patch('app.main.gui_factory.QLabel')
    @patch('app.main.gui_factory.Qt')
    def test_create_status_label_method_calls(self, mock_qt, mock_qlabel, mock_style_manager):
        """Test that create_status_label makes correct method calls."""
        mock_label = Mock()
        mock_qlabel.return_value = mock_label
        mock_qt.AlignCenter = "AlignCenter"
        
        result = LabelFactory.create_status_label("Status", "success")
        
        mock_qlabel.assert_called_once_with("Status")
        mock_style_manager.style_status_label.assert_called_once_with(mock_label, "success")
        mock_label.setAlignment.assert_called_once_with("AlignCenter")
        assert result == mock_label

    @patch('app.main.gui_factory.QLabel')
    @patch('app.main.gui_factory.QFont')
    def test_create_info_label_method_calls(self, mock_qfont, mock_qlabel):
        """Test that create_info_label makes correct method calls."""
        mock_label = Mock()
        mock_qlabel.return_value = mock_label
        mock_font = Mock()
        mock_qfont.return_value = mock_font
        
        result = LabelFactory.create_info_label("Info text", 14)
        
        mock_qlabel.assert_called_once_with("Info text")
        mock_qfont.assert_called_once_with("Arial", 14)
        mock_label.setFont.assert_called_once_with(mock_font)
        mock_label.setWordWrap.assert_called_once_with(True)
        assert result == mock_label


class TestFactoryConstants:
    """Test factory method parameter validation."""

    def test_button_factory_has_required_methods(self):
        """Test that ButtonFactory has all required methods."""
        assert hasattr(ButtonFactory, 'create_button')
        assert hasattr(ButtonFactory, 'create_control_button')
        assert hasattr(ButtonFactory, 'create_danger_button')
        assert callable(ButtonFactory.create_button)
        assert callable(ButtonFactory.create_control_button)
        assert callable(ButtonFactory.create_danger_button)

    def test_layout_factory_has_required_methods(self):
        """Test that LayoutFactory has all required methods."""
        assert hasattr(LayoutFactory, 'create_vbox_layout')
        assert hasattr(LayoutFactory, 'create_hbox_layout')
        assert hasattr(LayoutFactory, 'create_button_row')
        assert hasattr(LayoutFactory, 'create_label_value_row')
        assert callable(LayoutFactory.create_vbox_layout)
        assert callable(LayoutFactory.create_hbox_layout)

    def test_label_factory_has_required_methods(self):
        """Test that LabelFactory has all required methods."""
        assert hasattr(LabelFactory, 'create_status_label')
        assert hasattr(LabelFactory, 'create_info_label')
        assert callable(LabelFactory.create_status_label)
        assert callable(LabelFactory.create_info_label)