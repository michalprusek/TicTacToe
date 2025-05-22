"""
Unit tests for DebugWindow class.
"""
import pytest
from unittest.mock import MagicMock
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

# Ensure QApplication exists
if not QApplication.instance():
    app = QApplication(sys.argv)

from app.main.debug_window import DebugWindow
from app.main.camera_view import CameraView
from app.core.config import AppConfig


@pytest.fixture
def config():
    """config fixture for tests."""
    return AppConfig()


@pytest.fixture
def debug_window(config):
    """debug_window fixture for tests."""
    debug_window = DebugWindow(config=config)
    debug_window.parent = MagicMock()
    debug_window.parent.handle_camera_changed = MagicMock()
    debug_window.camera_combo = MagicMock()
    debug_window.autofocus_checkbox = MagicMock()
    debug_window.refresh_button = MagicMock()
    debug_window.conf_slider = MagicMock()
    debug_window.conf_value_label = MagicMock()
    debug_window.show_detections_checkbox = MagicMock()
    debug_window.show_grid_checkbox = MagicMock()
    debug_window.status_label = MagicMock()
    debug_window.fps_label = MagicMock()
    debug_window.board_state_label = MagicMock()
    debug_window.camera_view = MagicMock()
    debug_window.logger = MagicMock()
    return debug_window


@pytest.fixture
def camera_view():
    """camera_view fixture for tests."""
    return CameraView()


class TestDebugWindow():
    """Test DebugWindow class."""

    def test_init(self, debug_window, config):
        """Test initialization."""
        assert debug_window.config == config
        assert debug_window.windowTitle() == config.game.debug_window_title

    @pytest.mark.skip(reason="Difficult to mock parent properly")
    def test_handle_camera_changed(self, debug_window):
        """Test handle_camera_changed method."""
        # This test is skipped because it's difficult to mock the parent
        pass

    def test_handle_conf_changed(self, debug_window):
        """Test handle_conf_changed method."""
        # Call the method
        debug_window.handle_conf_changed(50)

        # Check that conf_value_label was updated
        debug_window.conf_value_label.setText.assert_called_once_with("0.50")

    def test_handle_refresh_clicked(self, debug_window):
        """Test handle_refresh_clicked method."""
        # Call the method
        debug_window.handle_refresh_clicked()

        # No assertions needed, just checking it doesn't crash

    def test_handle_autofocus_changed(self, debug_window):
        """Test handle_autofocus_changed method."""
        # Call the method
        debug_window.handle_autofocus_changed(2)  # Qt.Checked

        # No assertions needed, just checking it doesn't crash

    def test_handle_show_detections_changed(self, debug_window):
        """Test handle_show_detections_changed method."""
        # Call the method
        debug_window.handle_show_detections_changed(2)  # Qt.Checked

        # No assertions needed, just checking it doesn't crash

    def test_handle_show_grid_changed(self, debug_window):
        """Test handle_show_grid_changed method."""
        # Call the method
        debug_window.handle_show_grid_changed(2)  # Qt.Checked

        # No assertions needed, just checking it doesn't crash

    def test_update_camera_view(self, debug_window):
        """Test update_camera_view method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Call the method
        debug_window.update_camera_view(frame)

        # Check that camera_view.update_frame was called
        debug_window.camera_view.update_frame.assert_called_once()

    def test_update_status(self, debug_window):
        """Test update_status method."""
        # Call the method
        debug_window.update_status("Test status")

        # Check that status_label was updated
        status_label = debug_window.status_label
        status_label.setText.assert_called_once_with("Test status")

    def test_update_fps(self, debug_window):
        """Test update_fps method."""
        # Call the method
        debug_window.update_fps(30.5)

        # Check that fps_label was updated
        debug_window.fps_label.setText.assert_called_once_with("FPS: 30.5")

    def test_update_board_state(self, debug_window):
        """Test update_board_state method."""
        # Create a dummy board state
        board_state = [
            ['X', '', ''],
            ['', 'O', ''],
            ['', '', '']
        ]

        # Call the method
        debug_window.update_board_state(board_state)

        # Check that board_state_label was updated
        debug_window.board_state_label.setText.assert_called_once()


class TestCameraView():
    """Test CameraView class."""

    def test_init(self, camera_view):
        """Test initialization."""
        assert camera_view.text() == "Kamera nedostupná"
        assert camera_view.minimumSize().width() == 320
        assert camera_view.minimumSize().height() == 240

    def test_update_frame(self, camera_view):
        """Test update_frame method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Call the method
        camera_view.update_frame(frame)

        # No assertions needed, just checking it doesn't crash

    def test_update_frame_none(self, camera_view):
        """Test update_frame method with None frame."""
        # Call the method
        camera_view.update_frame(None)

        # No assertions needed, just checking it doesn't crash