"""
Unit tests for update_game_state method in TicTacToeApp
"""
import pytest
from unittest.mock import MagicMock

from app.main import game_logic
from tests.conftest_common import MockTicTacToeApp, qt_app
from tests.conftest_common import PyQtGuiTestCaseBase


@pytest.fixture
def tic_tac_toe_app():
    """tic_tac_toe_app fixture for tests."""
    tic_tac_toe_app = MockTicTacToeApp()
    tic_tac_toe_app.human_player = game_logic.PLAYER_X
    tic_tac_toe_app.ai_player = game_logic.PLAYER_O
    tic_tac_toe_app.current_turn = None
    tic_tac_toe_app.game_over = False
    tic_tac_toe_app.winner = None
    tic_tac_toe_app.waiting_for_detection = False
    tic_tac_toe_app.ai_move_row = None
    tic_tac_toe_app.ai_move_col = None
    tic_tac_toe_app.ai_move_retry_count = 0
    tic_tac_toe_app.max_retry_count = 3
    tic_tac_toe_app.detection_wait_time = 0
    tic_tac_toe_app.max_detection_wait_time = 5.0
    tic_tac_toe_app.board_widget = MagicMock()
    tic_tac_toe_app.board_widget.board = game_logic.create_board()
    tic_tac_toe_app.status_label = MagicMock()
    tic_tac_toe_app.strategy_selector = MagicMock()
    tic_tac_toe_app.draw_ai_symbol = MagicMock()
    tic_tac_toe_app.check_game_end = MagicMock()
    return tic_tac_toe_app



class TestTicTacToeAppUpdateState():
    """Test cases for update_game_state method in TicTacToeApp"""

    def setUp(self):
        """Set up test fixtures"""
        # Create QApplication instance if it doesn't exist
        self.app = PyQtGuiTestCaseBase.create_app_instance()

        # Set up patches
        self.patches = PyQtGuiTestCaseBase.setup_patches()

        # Create a mock instance
        self.tic_tac_toe_app = MockTicTacToeApp()

        # Set up necessary attributes manually
        self.tic_tac_toe_app.human_player = game_logic.PLAYER_X
        self.tic_tac_toe_app.ai_player = game_logic.PLAYER_O
        self.tic_tac_toe_app.current_turn = None
        self.tic_tac_toe_app.game_over = False
        self.tic_tac_toe_app.winner = None
        self.tic_tac_toe_app.waiting_for_detection = False
        self.tic_tac_toe_app.ai_move_row = None
        self.tic_tac_toe_app.ai_move_col = None
        self.tic_tac_toe_app.ai_move_retry_count = 0
        self.tic_tac_toe_app.max_retry_count = 3
        self.tic_tac_toe_app.detection_wait_time = 0
        self.tic_tac_toe_app.max_detection_wait_time = 5.0

        # Mock UI components
        self.tic_tac_toe_app.board_widget = MagicMock()
        self.tic_tac_toe_app.board_widget.board = game_logic.create_board()
        self.tic_tac_toe_app.status_label = MagicMock()

        # Mock strategy selector
        self.tic_tac_toe_app.strategy_selector = MagicMock()

        # Mock draw_ai_symbol method
        self.tic_tac_toe_app.draw_ai_symbol = MagicMock()

        # Mock check_game_end method
        self.tic_tac_toe_app.check_game_end = MagicMock()

    def tearDown(self):
        """Tear down test fixtures"""
        # Stop all patches
        for p in self.patches:
            p.stop()

    def test_update_game_state_ai_turn(self):
        """Test update_game_state method when it's AI's turn"""
        # Set up initial state
        self.tic_tac_toe_app.current_turn = self.tic_tac_toe_app.ai_player

        # Set up strategy selector to return a move
        self.tic_tac_toe_app.strategy_selector.get_move.return_value = (1, 1)
        self.tic_tac_toe_app.strategy_selector.select_strategy.return_value = "minimax"

        # Set up draw_ai_symbol to succeed
        self.tic_tac_toe_app.draw_ai_symbol.return_value = True

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that strategy_selector.get_move was called
        self.tic_tac_toe_app.strategy_selector.get_move.assert_called_once_with(
            self.tic_tac_toe_app.board_widget.board, self.tic_tac_toe_app.ai_player)

        # Check that ai_move_row and ai_move_col were set
        assert self.tic_tac_toe_app.ai_move_row == 1
        assert self.tic_tac_toe_app.ai_move_col == 1

        # Check that draw_ai_symbol was called
        self.tic_tac_toe_app.draw_ai_symbol.assert_called_once_with(1, 1)

        # Check that waiting_for_detection was set to True
        assert self.tic_tac_toe_app.waiting_for_detection

        # Check that detection_wait_time was reset
        assert self.tic_tac_toe_app.detection_wait_time == 0

        # Check that ai_move_retry_count was reset
        assert self.tic_tac_toe_app.ai_move_retry_count == 0

        # Check that status was updated
        self.tic_tac_toe_app.status_label.setText.assert_called()

    def test_update_game_state_ai_turn_no_arm(self):
        """Test update_game_state method when it's AI's turn but no arm is connected"""
        # Set up initial state
        self.tic_tac_toe_app.current_turn = self.tic_tac_toe_app.ai_player

        # Set up strategy selector to return a move
        self.tic_tac_toe_app.strategy_selector.get_move.return_value = (1, 1)
        self.tic_tac_toe_app.strategy_selector.select_strategy.return_value = "minimax"

        # Set up draw_ai_symbol to fail (no arm connected)
        self.tic_tac_toe_app.draw_ai_symbol.return_value = False

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that strategy_selector.get_move was called
        self.tic_tac_toe_app.strategy_selector.get_move.assert_called_once_with(
            self.tic_tac_toe_app.board_widget.board, self.tic_tac_toe_app.ai_player)

        # Check that ai_move_row and ai_move_col were set
        assert self.tic_tac_toe_app.ai_move_row == 1
        assert self.tic_tac_toe_app.ai_move_col == 1

        # Check that draw_ai_symbol was called
        self.tic_tac_toe_app.draw_ai_symbol.assert_called_once_with(1, 1)

        # Check that waiting_for_detection was not set to True
        assert not self.tic_tac_toe_app.waiting_for_detection

        # Check that status was updated
        self.tic_tac_toe_app.status_label.setText.assert_called()

    def test_update_game_state_waiting_for_detection(self):
        """Test update_game_state method when waiting for detection"""
        # Set up initial state
        self.tic_tac_toe_app.waiting_for_detection = True
        self.tic_tac_toe_app.ai_move_row = 1
        self.tic_tac_toe_app.ai_move_col = 1
        self.tic_tac_toe_app.detection_wait_time = 0

        # Set up camera_thread with detected board
        self.tic_tac_toe_app.camera_thread = MagicMock()
        self.tic_tac_toe_app.camera_thread.last_board_state = [
            [0, 0, 0],
            [0, game_logic.PLAYER_O, 0],
            [0, 0, 0]
        ]

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that waiting_for_detection was set to False
        assert not self.tic_tac_toe_app.waiting_for_detection

        # Check that detection_wait_time was reset
        assert self.tic_tac_toe_app.detection_wait_time == 0

        # Check that ai_move_retry_count was reset
        assert self.tic_tac_toe_app.ai_move_retry_count == 0

        # Check that board was updated
        self.tic_tac_toe_app.board_widget.update.assert_called_once()

        # Check that current_turn was updated to human's turn
        assert self.tic_tac_toe_app.current_turn == self.tic_tac_toe_app.human_player

        # Check that status was updated
        self.tic_tac_toe_app.status_label.setText.assert_called()

        # Check that check_game_end was called
        self.tic_tac_toe_app.check_game_end.assert_called_once()

    def test_update_game_state_detection_timeout(self):
        """Test update_game_state method when detection times out"""
        # Set up initial state
        self.tic_tac_toe_app.waiting_for_detection = True
        self.tic_tac_toe_app.ai_move_row = 1
        self.tic_tac_toe_app.ai_move_col = 1
        self.tic_tac_toe_app.detection_wait_time = 5.0  # Max wait time
        self.tic_tac_toe_app.ai_move_retry_count = 0

        # Set up draw_ai_symbol to succeed
        self.tic_tac_toe_app.draw_ai_symbol.return_value = True

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that detection_wait_time was reset
        assert self.tic_tac_toe_app.detection_wait_time == 0

        # Check that ai_move_retry_count was incremented
        assert self.tic_tac_toe_app.ai_move_retry_count == 1

        # Check that draw_ai_symbol was called again
        self.tic_tac_toe_app.draw_ai_symbol.assert_called_once_with(1, 1)

        # Check that waiting_for_detection was set to True again
        assert self.tic_tac_toe_app.waiting_for_detection

        # Check that status was updated
        self.tic_tac_toe_app.status_label.setText.assert_called()

    def test_update_game_state_detection_timeout_max_retries(self):
        """Test update_game_state method when detection times out and max retries reached"""
        # Set up initial state
        self.tic_tac_toe_app.waiting_for_detection = True
        self.tic_tac_toe_app.ai_move_row = 1
        self.tic_tac_toe_app.ai_move_col = 1
        self.tic_tac_toe_app.detection_wait_time = 5.0  # Max wait time
        self.tic_tac_toe_app.ai_move_retry_count = 3  # Max retries

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that detection_wait_time was reset
        assert self.tic_tac_toe_app.detection_wait_time == 0

        # Check that waiting_for_detection was set to False
        assert not self.tic_tac_toe_app.waiting_for_detection

        # Check that current_turn was updated to human's turn
        assert self.tic_tac_toe_app.current_turn == self.tic_tac_toe_app.human_player

        # Check that status was updated
        self.tic_tac_toe_app.status_label.setText.assert_called()

        # Check that draw_ai_symbol was not called again
        self.tic_tac_toe_app.draw_ai_symbol.assert_not_called()

    def test_update_game_state_detection_timeout_draw_failure(self):
        """Test update_game_state method when detection times out and drawing fails"""
        # Set up initial state
        self.tic_tac_toe_app.waiting_for_detection = True
        self.tic_tac_toe_app.ai_move_row = 1
        self.tic_tac_toe_app.ai_move_col = 1
        self.tic_tac_toe_app.detection_wait_time = 5.0  # Max wait time
        self.tic_tac_toe_app.ai_move_retry_count = 0

        # Set up draw_ai_symbol to fail
        self.tic_tac_toe_app.draw_ai_symbol.return_value = False

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that detection_wait_time was reset
        assert self.tic_tac_toe_app.detection_wait_time == 0

        # Check that waiting_for_detection was set to False
        assert not self.tic_tac_toe_app.waiting_for_detection

        # Check that current_turn was updated to human's turn
        assert self.tic_tac_toe_app.current_turn == self.tic_tac_toe_app.human_player

        # Check that status was updated
        self.tic_tac_toe_app.status_label.setText.assert_called()

        # Check that draw_ai_symbol was called
        self.tic_tac_toe_app.draw_ai_symbol.assert_called_once_with(1, 1)

    def test_update_game_state_game_over(self):
        """Test update_game_state method when game is over"""
        # Set up initial state
        self.tic_tac_toe_app.game_over = True

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that strategy_selector.get_move was not called
        self.tic_tac_toe_app.strategy_selector.get_move.assert_not_called()

        # Check that draw_ai_symbol was not called
        self.tic_tac_toe_app.draw_ai_symbol.assert_not_called()

        # Check that check_game_end was not called
        self.tic_tac_toe_app.check_game_end.assert_not_called()

    def test_update_game_state_human_turn(self):
        """Test update_game_state method when it's human's turn"""
        # Set up initial state
        self.tic_tac_toe_app.current_turn = self.tic_tac_toe_app.human_player

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that strategy_selector.get_move was not called
        self.tic_tac_toe_app.strategy_selector.get_move.assert_not_called()

        # Check that draw_ai_symbol was not called
        self.tic_tac_toe_app.draw_ai_symbol.assert_not_called()

        # Check that check_game_end was not called
        self.tic_tac_toe_app.check_game_end.assert_not_called()

    def test_update_game_state_no_current_turn(self):
        """Test update_game_state method when current_turn is None"""
        # Set up initial state
        self.tic_tac_toe_app.current_turn = None

        # Call update_game_state
        self.tic_tac_toe_app.update_game_state()

        # Check that strategy_selector.get_move was not called
        self.tic_tac_toe_app.strategy_selector.get_move.assert_not_called()

        # Check that draw_ai_symbol was not called
        self.tic_tac_toe_app.draw_ai_symbol.assert_not_called()

        # Check that check_game_end was not called
        self.tic_tac_toe_app.check_game_end.assert_not_called()



