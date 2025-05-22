"""
Tests for advanced handler methods in TicTacToeApp.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys
from app.main import game_logic


@pytest.fixture
def tic_tac_toe_app():
    """tic_tac_toe_app fixture for tests."""
    tic_tac_toe_app = MockTicTacToeApp()
    tic_tac_toe_app.camera_thread = MagicMock()
    tic_tac_toe_app.arm_controller = MagicMock()
    tic_tac_toe_app.board_widget = MagicMock()
    tic_tac_toe_app.status_label = MagicMock()
    tic_tac_toe_app.strategy_selector = MagicMock()
    tic_tac_toe_app.debug_window = MagicMock()
    tic_tac_toe_app.game_over = False
    tic_tac_toe_app.current_turn = game_logic.PLAYER_X
    tic_tac_toe_app.human_player = game_logic.PLAYER_X
    tic_tac_toe_app.ai_player = game_logic.PLAYER_O
    tic_tac_toe_app.check_game_end = MagicMock()
    tic_tac_toe_app.update_camera_view = MagicMock()
    tic_tac_toe_app.handle_detected_game_state = MagicMock()
    tic_tac_toe_app.update_fps_display = MagicMock()
    return tic_tac_toe_app


# Ensure QApplication exists
if not QApplication.instance():
    app = QApplication(sys.argv)

from app.main.pyqt_gui import TicTacToeApp, CameraThread
from tests.unit.test_pyqt_gui_app_helper import MockTicTacToeApp, PyQtGuiAppTestCase


class TestPyQtGuiAdvancedHandlers():
    """Test advanced handler methods in TicTacToeApp."""

    def setUp(self):
        """Set up test environment."""
        # Create QApplication instance if it doesn't exist
        self.app = PyQtGuiAppTestCase.create_app_instance()

        # Set up patches
        self.patches = PyQtGuiAppTestCase.setup_patches()

        # Create a mock instance
        self.tic_tac_toe_app = MockTicTacToeApp()

        # Set up mock attributes
        self.tic_tac_toe_app.camera_thread = MagicMock()
        self.tic_tac_toe_app.arm_controller = MagicMock()
        self.tic_tac_toe_app.board_widget = MagicMock()
        self.tic_tac_toe_app.status_label = MagicMock()
        self.tic_tac_toe_app.strategy_selector = MagicMock()
        self.tic_tac_toe_app.debug_window = MagicMock()

        # Set up game state
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_X
        self.tic_tac_toe_app.human_player = game_logic.PLAYER_X
        self.tic_tac_toe_app.ai_player = game_logic.PLAYER_O
        self.tic_tac_toe_app.check_game_end = MagicMock()
        self.tic_tac_toe_app.update_camera_view = MagicMock()
        self.tic_tac_toe_app.handle_detected_game_state = MagicMock()
        self.tic_tac_toe_app.update_fps_display = MagicMock()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop all patches
        for p in self.patches:
            p.stop()

    def test_handle_camera_changed(self):
        """Test handle_camera_changed method."""
        # Mock CameraThread
        with patch('pyqt_gui.CameraThread') as mock_camera_thread:
            # Set up mock camera thread
            mock_camera_thread.return_value = MagicMock()

            # Call the method
            self.tic_tac_toe_app.handle_camera_changed(1)

            # Check that camera_thread.stop was called
            self.tic_tac_toe_app.camera_thread.stop.assert_called_once()

            # Check that camera_thread.wait was called
            self.tic_tac_toe_app.camera_thread.wait.assert_called_once()

            # Check that CameraThread was created with correct camera_index
            mock_camera_thread.assert_called_once_with(camera_index=1)

            # Check that signals were connected
            mock_camera_thread.return_value.frame_ready.connect.assert_called_with(
                self.tic_tac_toe_app.update_camera_view)
            mock_camera_thread.return_value.game_state_updated.connect.assert_called_with(
                self.tic_tac_toe_app.handle_detected_game_state)
            mock_camera_thread.return_value.fps_updated.connect.assert_called_with(
                self.tic_tac_toe_app.update_fps_display)

            # Check that camera_thread.start was called
            mock_camera_thread.return_value.start.assert_called_once()

    def test_handle_camera_changed_no_camera_thread(self):
        """Test handle_camera_changed method when camera_thread is not available."""
        # Remove camera_thread
        self.tic_tac_toe_app.camera_thread = None

        # Call the method - should not raise an exception
        self.tic_tac_toe_app.handle_camera_changed(1)

    def test_handle_arm_connection_toggled_connect(self):
        """Test handle_arm_connection_toggled method when connecting."""
        # Call the method with True
        self.tic_tac_toe_app.handle_arm_connection_toggled(True)

        # Check that arm_controller.connect was called
        self.tic_tac_toe_app.arm_controller.connect.assert_called_once()

    def test_handle_arm_connection_toggled_disconnect(self):
        """Test handle_arm_connection_toggled method when disconnecting."""
        # Set arm_controller.connected to True
        self.tic_tac_toe_app.arm_controller.connected = True

        # Call the method with False
        self.tic_tac_toe_app.handle_arm_connection_toggled(False)

        # Check that arm_controller.disconnect was called
        self.tic_tac_toe_app.arm_controller.disconnect.assert_called_once()

    def test_handle_arm_connection_toggled_no_change(self):
        """Test handle_arm_connection_toggled method when no change is needed."""
        # Set arm_controller.connected to True
        self.tic_tac_toe_app.arm_controller.connected = True

        # Call the method with True
        self.tic_tac_toe_app.handle_arm_connection_toggled(True)

        # Check that arm_controller.connect was not called
        self.tic_tac_toe_app.arm_controller.connect.assert_not_called()

        # Check that arm_controller.disconnect was not called
        self.tic_tac_toe_app.arm_controller.disconnect.assert_not_called()

    def test_handle_arm_connection_toggled_no_arm_controller(self):
        """Test handle_arm_connection_toggled method when arm_controller is not available."""
        # Remove arm_controller
        self.tic_tac_toe_app.arm_controller = None

        # Call the method - should not raise an exception
        self.tic_tac_toe_app.handle_arm_connection_toggled(True)

    def test_handle_detected_game_state(self):
        """Test handle_detected_game_state method."""
        # Create a mock board
        board = [
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Call the method
        self.tic_tac_toe_app.handle_detected_game_state(board)

        # Check that board_widget.board was updated
        assert self.tic_tac_toe_app.board_widget.board == board

        # Check that board_widget.update was called
        self.tic_tac_toe_app.board_widget.update.assert_called_once()

        # Check that check_game_end was called
        self.tic_tac_toe_app.check_game_end.assert_called_once()

    def test_handle_detected_game_state_ai_turn(self):
        """Test handle_detected_game_state method when it's AI's turn."""
        # Create a mock board
        board = [
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Set current_turn to AI player
        self.tic_tac_toe_app.current_turn = self.tic_tac_toe_app.ai_player

        # Mock make_ai_move method
        self.tic_tac_toe_app.make_ai_move = MagicMock()

        # Call the method
        self.tic_tac_toe_app.handle_detected_game_state(board)

        # Check that make_ai_move was called
        self.tic_tac_toe_app.make_ai_move.assert_called_once()

    def test_handle_detected_game_state_game_over(self):
        """Test handle_detected_game_state method when game is over."""
        # Create a mock board
        board = [
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Set game_over to True
        self.tic_tac_toe_app.game_over = True

        # Mock make_ai_move method
        self.tic_tac_toe_app.make_ai_move = MagicMock()

        # Call the method
        self.tic_tac_toe_app.handle_detected_game_state(board)

        # Check that make_ai_move was not called
        self.tic_tac_toe_app.make_ai_move.assert_not_called()

    def test_handle_detected_game_state_no_board_widget(self):
        """Test handle_detected_game_state method when board_widget is not available."""
        # Create a mock board
        board = [
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Remove board_widget
        self.tic_tac_toe_app.board_widget = None

        # Call the method - should not raise an exception
        self.tic_tac_toe_app.handle_detected_game_state(board)

    def test_make_ai_move(self):
        """Test make_ai_move method."""
        # Set up board
        self.tic_tac_toe_app.board_widget.board = [
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Set up strategy_selector to return a move
        self.tic_tac_toe_app.strategy_selector.get_move.return_value = (1, 1)

        # Call the method
        self.tic_tac_toe_app.make_ai_move()

        # Check that strategy_selector.get_move was called with correct arguments
        self.tic_tac_toe_app.strategy_selector.get_move.assert_called_once_with(
            self.tic_tac_toe_app.board_widget.board, self.tic_tac_toe_app.ai_player)

        # Check that board was updated
        assert self.tic_tac_toe_app.board_widget.board[1][1] == self.tic_tac_toe_app.ai_player

        # Check that board_widget.update was called
        self.tic_tac_toe_app.board_widget.update.assert_called_once()

        # Check that check_game_end was called
        self.tic_tac_toe_app.check_game_end.assert_called_once()

    def test_make_ai_move_no_move(self):
        """Test make_ai_move method when no move is available."""
        # Set up strategy_selector to return None
        self.tic_tac_toe_app.strategy_selector.get_move.return_value = None

        # Call the method
        self.tic_tac_toe_app.make_ai_move()

        # Check that status_label.setText was called with correct message
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with("AI nemůže najít vhodný tah!")

        # Check that board_widget.update was not called
        self.tic_tac_toe_app.board_widget.update.assert_not_called()

        # Check that check_game_end was not called
        self.tic_tac_toe_app.check_game_end.assert_not_called()

    def test_make_ai_move_game_not_over(self):
        """Test make_ai_move method when game is not over after move."""
        # Set up board
        self.tic_tac_toe_app.board_widget.board = [
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Set up strategy_selector to return a move
        self.tic_tac_toe_app.strategy_selector.get_move.return_value = (1, 1)

        # Set check_game_end to set game_over to False
        def check_game_end_mock():
            self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.check_game_end.side_effect = check_game_end_mock

        # Call the method
        self.tic_tac_toe_app.make_ai_move()

        # Check that current_turn was set to human_player
        assert self.tic_tac_toe_app.current_turn == self.tic_tac_toe_app.human_player

        # Check that status_label.setText was called with correct message
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with(
            f"Váš tah ({self.tic_tac_toe_app.human_player})")

    def test_make_ai_move_no_strategy_selector(self):
        """Test make_ai_move method when strategy_selector is not available."""
        # Remove strategy_selector
        self.tic_tac_toe_app.strategy_selector = None

        # Call the method - should not raise an exception
        self.tic_tac_toe_app.make_ai_move()



