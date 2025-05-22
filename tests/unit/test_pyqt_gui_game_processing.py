"""
Unit tests for game processing methods in pyqt_gui.py
"""
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import time

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from app.main.pyqt_gui import TicTacToeApp
from app.main import game_logic


@pytest.fixture
def tic_tac_toe_app():
    """tic_tac_toe_app fixture for tests."""
    tic_tac_toe_app = TicTacToeApp()
    tic_tac_toe_app.camera_thread = MagicMock()
    tic_tac_toe_app.arm_thread = MagicMock()
    tic_tac_toe_app.arm_controller = MagicMock()
    tic_tac_toe_app.game_state = MagicMock()
    tic_tac_toe_app.strategy_selector = MagicMock()
    tic_tac_toe_app.current_turn = None
    tic_tac_toe_app.game_over = False
    tic_tac_toe_app.debug_mode = False
    tic_tac_toe_app.debug_window = None
    tic_tac_toe_app.detection_timeout_counter = 0
    tic_tac_toe_app.max_detection_retries = 5
    tic_tac_toe_app.last_detection_time = 0
    tic_tac_toe_app.detection_timeout = 2.0
    tic_tac_toe_app.human_player = game_logic.PLAYER_X
    tic_tac_toe_app.ai_player = game_logic.PLAYER_O
    tic_tac_toe_app.board = MagicMock()
    tic_tac_toe_app.status_label = MagicMock()
    return tic_tac_toe_app



class TestPyQtGuiGameProcessing():
    """Test cases for game processing methods in TicTacToeApp"""

    def setUp(self):
        """Set up test fixtures"""
        # Create QApplication instance if it doesn't exist
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])

        # Mock TicTacToeApp.__init__ to avoid calling super().__init__()
        with patch.object(TicTacToeApp, '__init__', return_value=None):
            # Create TicTacToeApp instance
            self.tic_tac_toe_app = TicTacToeApp()

            # Set up necessary attributes
            self.tic_tac_toe_app.camera_thread = MagicMock()
            self.tic_tac_toe_app.arm_thread = MagicMock()
            self.tic_tac_toe_app.arm_controller = MagicMock()
            self.tic_tac_toe_app.game_state = MagicMock()
            self.tic_tac_toe_app.strategy_selector = MagicMock()
            self.tic_tac_toe_app.current_turn = None
            self.tic_tac_toe_app.game_over = False
            self.tic_tac_toe_app.debug_mode = False
            self.tic_tac_toe_app.debug_window = None
            self.tic_tac_toe_app.detection_timeout_counter = 0
            self.tic_tac_toe_app.max_detection_retries = 5
            self.tic_tac_toe_app.last_detection_time = 0
            self.tic_tac_toe_app.detection_timeout = 2.0
            self.tic_tac_toe_app.human_player = game_logic.PLAYER_X
            self.tic_tac_toe_app.ai_player = game_logic.PLAYER_O

            # Mock UI components
            self.tic_tac_toe_app.board = MagicMock()
            self.tic_tac_toe_app.status_label = MagicMock()

    def tearDown(self):
        """Tear down test fixtures"""
        pass

    def test_process_game_state_first_move(self):
        """Test process_game_state method with first move"""
        # Set up mock objects
        self.tic_tac_toe_app.current_turn = None
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.game_state.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.tic_tac_toe_app.game_state.check_winner.return_value = None  # Ongoing game

        # Call the method
        self.tic_tac_toe_app.process_game_state()

        # Check that current_turn was set to game_logic.PLAYER_X
        assert self.tic_tac_toe_app.current_turn == game_logic.PLAYER_X

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with("Váš tah (X)")

        # Check that board.update_board was called
        self.tic_tac_toe_app.board.update_board.assert_called_once_with(
            self.tic_tac_toe_app.game_state.board, None
        )

    def test_process_game_state_human_turn(self):
        """Test process_game_state method with human turn"""
        # Set up mock objects
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_X
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.game_state.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.tic_tac_toe_app.game_state.check_winner.return_value = None  # Ongoing game

        # Call the method
        self.tic_tac_toe_app.process_game_state()

        # Check that current_turn is still game_logic.PLAYER_X
        assert self.tic_tac_toe_app.current_turn == game_logic.PLAYER_X

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with("Váš tah (X)")

        # Check that board.update_board was called
        self.tic_tac_toe_app.board.update_board.assert_called_once_with(
            self.tic_tac_toe_app.game_state.board, None
        )

    def test_process_game_state_ai_turn(self):
        """Test process_game_state method with AI turn"""
        # Set up mock objects
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_O
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.game_state.board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.tic_tac_toe_app.game_state.check_winner.return_value = None  # Ongoing game
        self.tic_tac_toe_app.strategy_selector.get_move.return_value = (1, 1)

        # Mock get_cell_coordinates
        self.tic_tac_toe_app.get_cell_coordinates = MagicMock()
        self.tic_tac_toe_app.get_cell_coordinates.return_value = (200, 200)

        # Mock draw_ai_symbol
        self.tic_tac_toe_app.draw_ai_symbol = MagicMock()

        # Call the method
        self.tic_tac_toe_app.process_game_state()

        # Check that strategy_selector.get_move was called
        self.tic_tac_toe_app.strategy_selector.get_move.assert_called_once_with(
            self.tic_tac_toe_app.game_state.board, game_logic.PLAYER_O
        )

        # Check that get_cell_coordinates was called
        self.tic_tac_toe_app.get_cell_coordinates.assert_called_once_with(1, 1)

        # Check that draw_ai_symbol was called
        self.tic_tac_toe_app.draw_ai_symbol.assert_called_once_with(1, 1, game_logic.PLAYER_O)

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_with("Čekám na detekci tahu...")

        # Check that current_turn was set to None
        assert self.tic_tac_toe_app.current_turn == None

    def test_process_game_state_game_over_x_wins(self):
        """Test process_game_state method with game over (X wins)"""
        # Set up mock objects
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_X
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.game_state.board = [[1, 1, 1], [0, 2, 0], [2, 0, 2]]
        self.tic_tac_toe_app.game_state.check_winner.return_value = game_logic.PLAYER_X
        self.tic_tac_toe_app.game_state.get_winning_line.return_value = [(0, 0), (0, 1), (0, 2)]

        # Call the method
        self.tic_tac_toe_app.process_game_state()

        # Check that game_over was set to True
        assert self.tic_tac_toe_app.game_over

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with("Vyhráli jste!")

        # Check that board.update_board was called with winning line
        self.tic_tac_toe_app.board.update_board.assert_called_once_with(
            self.tic_tac_toe_app.game_state.board, [(0, 0), (0, 1), (0, 2)]
        )

    def test_process_game_state_game_over_o_wins(self):
        """Test process_game_state method with game over (O wins)"""
        # Set up mock objects
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_O
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.game_state.board = [[1, 0, 1], [0, 2, 0], [2, 2, 2]]
        self.tic_tac_toe_app.game_state.check_winner.return_value = game_logic.PLAYER_O
        self.tic_tac_toe_app.game_state.get_winning_line.return_value = [(2, 0), (2, 1), (2, 2)]

        # Call the method
        self.tic_tac_toe_app.process_game_state()

        # Check that game_over was set to True
        assert self.tic_tac_toe_app.game_over

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with("Robotická ruka vyhrála!")

        # Check that board.update_board was called with winning line
        self.tic_tac_toe_app.board.update_board.assert_called_once_with(
            self.tic_tac_toe_app.game_state.board, [(2, 0), (2, 1), (2, 2)]
        )

    def test_process_game_state_game_over_tie(self):
        """Test process_game_state method with game over (tie)"""
        # Set up mock objects
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_X
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.game_state.board = [[1, 2, 1], [2, 1, 2], [2, 1, 2]]
        self.tic_tac_toe_app.game_state.check_winner.return_value = game_logic.TIE
        self.tic_tac_toe_app.game_state.get_winning_line.return_value = None

        # Call the method
        self.tic_tac_toe_app.process_game_state()

        # Check that game_over was set to True
        assert self.tic_tac_toe_app.game_over

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with("Remíza!")

        # Check that board.update_board was called with no winning line
        self.tic_tac_toe_app.board.update_board.assert_called_once_with(
            self.tic_tac_toe_app.game_state.board, None
        )

    def test_handle_detection_timeout_first_retry(self):
        """Test handle_detection_timeout method with first retry"""
        # Set up mock objects
        self.tic_tac_toe_app.detection_timeout_counter = 0
        self.tic_tac_toe_app.max_detection_retries = 5
        self.tic_tac_toe_app.last_detection_time = time.time() - 3.0  # 3 seconds ago
        self.tic_tac_toe_app.detection_timeout = 2.0
        self.tic_tac_toe_app.current_turn = None

        # Call the method
        self.tic_tac_toe_app.handle_detection_timeout()

        # Check that detection_timeout_counter was incremented
        assert self.tic_tac_toe_app.detection_timeout_counter == 1

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with(
            "Čekám na detekci tahu... (pokus 1/5)"
        )

    def test_handle_detection_timeout_max_retries(self):
        """Test handle_detection_timeout method with max retries reached"""
        # Set up mock objects
        self.tic_tac_toe_app.detection_timeout_counter = 4
        self.tic_tac_toe_app.max_detection_retries = 5
        self.tic_tac_toe_app.last_detection_time = time.time() - 3.0  # 3 seconds ago
        self.tic_tac_toe_app.detection_timeout = 2.0
        self.tic_tac_toe_app.current_turn = None

        # Mock update_game_state
        self.tic_tac_toe_app.update_game_state = MagicMock()

        # Call the method
        self.tic_tac_toe_app.handle_detection_timeout()

        # Check that detection_timeout_counter was incremented
        assert self.tic_tac_toe_app.detection_timeout_counter == 5

        # Check that status_label.setText was called
        self.tic_tac_toe_app.status_label.setText.assert_called_once_with(
            "Čekám na detekci tahu... (pokus 5/5)"
        )

        # Check that update_game_state was called
        self.tic_tac_toe_app.update_game_state.assert_called_once_with(detection_timeout=True)

    def test_handle_detection_timeout_no_timeout(self):
        """Test handle_detection_timeout method with no timeout"""
        # Set up mock objects
        self.tic_tac_toe_app.detection_timeout_counter = 0
        self.tic_tac_toe_app.max_detection_retries = 5
        self.tic_tac_toe_app.last_detection_time = time.time() - 1.0  # 1 second ago
        self.tic_tac_toe_app.detection_timeout = 2.0

        # Call the method
        self.tic_tac_toe_app.handle_detection_timeout()

        # Check that detection_timeout_counter was not incremented
        assert self.tic_tac_toe_app.detection_timeout_counter == 0

        # Check that status_label.setText was not called
        self.tic_tac_toe_app.status_label.setText.assert_not_called()

    def test_handle_detection_timeout_not_waiting(self):
        """Test handle_detection_timeout method when not waiting for detection"""
        # Set up mock objects
        self.tic_tac_toe_app.detection_timeout_counter = 0
        self.tic_tac_toe_app.max_detection_retries = 5
        self.tic_tac_toe_app.last_detection_time = time.time() - 3.0  # 3 seconds ago
        self.tic_tac_toe_app.detection_timeout = 2.0
        self.tic_tac_toe_app.current_turn = game_logic.PLAYER_X  # Not waiting for detection

        # Call the method
        self.tic_tac_toe_app.handle_detection_timeout()

        # Check that detection_timeout_counter was not incremented
        assert self.tic_tac_toe_app.detection_timeout_counter == 0

        # Check that status_label.setText was not called
        self.tic_tac_toe_app.status_label.setText.assert_not_called()



