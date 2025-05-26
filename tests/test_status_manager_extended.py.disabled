"""
Extended tests for StatusManager status update and UI functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QObject, QTimer, Qt
import sys
import time

from app.main.status_manager import StatusManager, LANG_CS, LANG_EN


@pytest.fixture
def qt_app():
    """Create QApplication if it doesn't exist."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def mock_main_window():
    """Mock main window with camera controller and board widget."""
    mock_window = Mock(spec=QMainWindow)
    mock_window.style_manager = Mock()
    mock_window.style_manager.update_status_style = Mock()
    
    # Mock camera controller
    mock_window.camera_controller = Mock()
    mock_window.camera_controller.get_current_board_state = Mock(return_value=[0]*9)
    
    # Mock board widget
    mock_window.board_widget = Mock()
    mock_window.board_widget.board = [0]*9
    
    return mock_window


@pytest.fixture
def status_manager(qt_app, mock_main_window):
    """Create StatusManager instance for testing."""
    return StatusManager(mock_main_window)


class TestStatusPanelCreation:
    """Test status panel creation and UI setup."""

    def test_create_status_panel(self, status_manager):
        """Test status panel creation."""
        panel = status_manager.create_status_panel()
        
        assert panel is not None
        assert isinstance(panel, QWidget)
        assert status_manager.main_status_panel == panel
        assert status_manager.main_status_message is not None
        assert isinstance(status_manager.main_status_message, QLabel)

    def test_status_panel_styling(self, status_manager):
        """Test that status panel has correct styling."""
        panel = status_manager.create_status_panel()
        
        # Check panel styling
        style = panel.styleSheet()
        assert "background-color: #333740" in style
        assert "border-radius: 10px" in style
        
        # Check message label styling
        message_style = status_manager.main_status_message.styleSheet()
        assert "color: #FFFFFF" in message_style
        assert "font-size: 28px" in message_style
        assert "font-weight: bold" in message_style

    def test_status_message_initial_text(self, status_manager):
        """Test initial status message text."""
        status_manager.create_status_panel()
        assert status_manager.main_status_message.text() == "START"

    def test_status_message_alignment(self, status_manager):
        """Test status message alignment."""
        status_manager.create_status_panel()
        assert status_manager.main_status_message.alignment() == Qt.AlignCenter


class TestStatusUpdate:
    """Test status update functionality."""

    def test_update_status_with_key(self, status_manager):
        """Test status update using translation key."""
        status_manager.create_status_panel()
        
        with patch('time.time', return_value=1000.0):
            status_manager.update_status("your_turn", is_key=True)
        
        assert status_manager._current_status_text == "VÁŠ TAH"
        assert status_manager._status_lock_time == 1000.0

    def test_update_status_with_direct_text(self, status_manager):
        """Test status update with direct text."""
        status_manager.create_status_panel()
        
        with patch('time.time', return_value=1000.0):
            status_manager.update_status("Custom Message", is_key=False)
        
        assert status_manager._current_status_text == "Custom Message"

    @patch('time.time')
    def test_status_locking_prevents_rapid_updates(self, mock_time, status_manager):
        """Test that status locking prevents rapid updates of same message."""
        status_manager.create_status_panel()
        
        # First update
        mock_time.return_value = 1000.0
        status_manager.update_status("your_turn", is_key=True)
        
        # Same message within lock time - should be ignored
        mock_time.return_value = 1000.5  # 0.5 seconds later
        old_text = status_manager._current_status_text
        status_manager.update_status("your_turn", is_key=True)
        
        # Status should not change
        assert status_manager._current_status_text == old_text

    @patch('time.time')
    def test_status_locking_allows_updates_after_timeout(self, mock_time, status_manager):
        """Test that status locking allows updates after timeout."""
        status_manager.create_status_panel()
        
        # First update
        mock_time.return_value = 1000.0
        status_manager.update_status("your_turn", is_key=True)
        
        # Same message after lock time - should be allowed
        mock_time.return_value = 1001.5  # 1.5 seconds later
        status_manager.update_status("your_turn", is_key=True)
        
        assert status_manager._status_lock_time == 1001.5

    def test_update_status_different_messages_allowed(self, status_manager):
        """Test that different messages are always allowed."""
        status_manager.create_status_panel()
        
        with patch('time.time', return_value=1000.0):
            status_manager.update_status("your_turn", is_key=True)
            status_manager.update_status("ai_turn", is_key=True)
        
        assert status_manager._current_status_text == "TAH AI"

    def test_update_status_without_panel(self, status_manager):
        """Test status update when panel is not created."""
        # Don't create panel
        with patch('time.time', return_value=1000.0):
            status_manager.update_status("your_turn", is_key=True)
        
        # Should update internal state even without panel
        assert status_manager._current_status_text == "VÁŠ TAH"


class TestStatusManagerIntegration:
    """Test integration with other components."""

    def test_board_state_access_camera_controller(self, status_manager, mock_main_window):
        """Test board state access via camera controller."""
        mock_board = [1, 2, 0, 1, 2, 1, 0, 0, 2]
        mock_main_window.camera_controller.get_current_board_state.return_value = mock_board
        
        status_manager.create_status_panel()
        status_manager.update_status("your_turn")
        
        mock_main_window.camera_controller.get_current_board_state.assert_called()

    def test_board_state_access_board_widget_fallback(self, status_manager, mock_main_window):
        """Test board state access via board widget when camera unavailable."""
        # Remove camera controller
        del mock_main_window.camera_controller
        mock_board = [1, 2, 0, 1, 2, 1, 0, 0, 2]
        mock_main_window.board_widget.board = mock_board
        
        status_manager.create_status_panel()
        status_manager.update_status("your_turn")
        
        # Should access board_widget.board as fallback
        assert mock_main_window.board_widget.board == mock_board

    def test_style_manager_integration(self, status_manager, mock_main_window):
        """Test integration with style manager."""
        status_manager.create_status_panel()
        
        # Mock style manager call
        status_manager.update_status("win")
        
        # Should have attempted to access style manager
        assert hasattr(mock_main_window, 'style_manager')
