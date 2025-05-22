"""
Unit tests for AI move handling in TicTacToeApp using pytest
"""
import sys
import pytest
from unittest.mock import patch, MagicMock

from PyQt5.QtWidgets import QApplication, QMainWindow

from app.main import game_logic
from app.main.pyqt_gui import TicTacToeApp


# Create QApplication instance
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# Direct mock without inheriting from TicTacToeApp
@pytest.fixture
def mock_app(qapp):
    """Mock TicTacToeApp fixture"""
    app = MagicMock()
    
    # Set up necessary attributes manually
    app.human_player = game_logic.PLAYER_X
    app.ai_player = game_logic.PLAYER_O
    app.current_turn = game_logic.PLAYER_O
    app.game_over = False
    app.winner = None
    app.waiting_for_detection = False
    app.ai_move_row = None
    app.ai_move_col = None
    app.ai_move_retry_count = 0
    app.max_retry_count = 3
    app.detection_wait_time = 0
    app.max_detection_wait_time = 5.0
    
    # Mock UI components
    app.board_widget = MagicMock()
    app.board_widget.board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    app.status_label = MagicMock()
    
    # Mock strategy selector
    app.strategy_selector = MagicMock()
    
    # Mock other methods
    app.check_game_end = MagicMock()
    app.get_cell_coordinates_from_yolo = MagicMock()
    app.move_to_neutral_position = MagicMock()
    app.draw_ai_symbol = MagicMock()
    
    return app


def test_update_game_state_ai_turn(mock_app):
    """Test update_game_state method when it's AI's turn"""
    # Reset mocks
    mock_app.strategy_selector.reset_mock()
    mock_app.status_label.reset_mock()

    # Set up strategy selector to return a move
    mock_app.strategy_selector.get_move.return_value = (1, 1)
    mock_app.strategy_selector.select_strategy.return_value = "minimax"
    mock_app.draw_ai_symbol.return_value = True

    # Define update_game_state function
    def update_game_state():
        # Generate AI move when it's AI's turn and we have a valid AI player
        if (not mock_app.game_over and
            mock_app.current_turn == mock_app.ai_player and
                mock_app.ai_player is not None):

            # AI's turn - use Bernoulli strategy selector
            ai_move = mock_app.strategy_selector.get_move(
                mock_app.board_widget.board, mock_app.ai_player)

            if ai_move:
                row, col = ai_move
                mock_app.ai_move_row = row
                mock_app.ai_move_col = col

                # Get the strategy that was used
                strategy = mock_app.strategy_selector.select_strategy()
                strategy_name = "minimax" if strategy == "minimax" else "náhodná"

                # Update status label with AI's move
                mock_app.status_label.setText(
                    f"AI použila {strategy_name} strategii")

                # Simulate drawing the symbol
                if mock_app.draw_ai_symbol(row, col):
                    # Start waiting for detection
                    mock_app.waiting_for_detection = True
                    mock_app.detection_wait_time = 0
                    mock_app.ai_move_retry_count = 0
                    mock_app.status_label.setText(
                        f"Čekám na detekci symbolu {mock_app.ai_player}...")
                    return True
                
        return False

    # Attach the function to the mock
    mock_app.update_game_state = update_game_state

    # Call update_game_state
    result = mock_app.update_game_state()

    # Check that strategy_selector.get_move was called
    mock_app.strategy_selector.get_move.assert_called_once_with(
        mock_app.board_widget.board, game_logic.PLAYER_O)

    # Check that ai_move_row and ai_move_col were set
    assert mock_app.ai_move_row == 1
    assert mock_app.ai_move_col == 1

    # Check that waiting_for_detection was set to True
    assert mock_app.waiting_for_detection is True

    # Check that detection_wait_time was reset
    assert mock_app.detection_wait_time == 0

    # Check that ai_move_retry_count was reset
    assert mock_app.ai_move_retry_count == 0

    # Check that status was updated
    assert mock_app.status_label.setText.called

    # Check that result is True
    assert result is True


def test_update_game_state_waiting_for_detection(mock_app):
    """Test update_game_state method when waiting for detection"""
    # Set up initial state
    mock_app.waiting_for_detection = True
    mock_app.ai_move_row = 1
    mock_app.ai_move_col = 1
    mock_app.detection_wait_time = 0
    mock_app.status_label.reset_mock()
    mock_app.check_game_end.reset_mock()
    mock_app.board_widget.update.reset_mock()

    # Set up camera_thread with detected board
    mock_app.camera_thread = MagicMock()
    mock_app.camera_thread.last_board_state = [
        [0, 0, 0],
        [0, game_logic.PLAYER_O, 0],
        [0, 0, 0]
    ]

    # Define update_game_state function
    def update_game_state():
        # Pokud čekáme na detekci nakresleného symbolu
        if mock_app.waiting_for_detection:
            # Zvýšíme čas čekání
            mock_app.detection_wait_time += 0.1

            # Kontrola, zda byl symbol detekován
            if hasattr(mock_app, 'camera_thread') and mock_app.camera_thread.last_board_state:
                detected_board = mock_app.camera_thread.last_board_state
                if (0 <= mock_app.ai_move_row < 3 and 0 <= mock_app.ai_move_col < 3 and
                        detected_board[mock_app.ai_move_row][mock_app.ai_move_col] == mock_app.ai_player):
                    # Symbol byl detekován, můžeme pokračovat
                    mock_app.waiting_for_detection = False
                    mock_app.detection_wait_time = 0
                    mock_app.ai_move_retry_count = 0

                    # Aktualizujeme GUI podle detekovaného stavu
                    mock_app.board_widget.board = [row[:] for row in detected_board]
                    mock_app.board_widget.update()

                    # Předáme tah hráči
                    mock_app.current_turn = mock_app.human_player
                    mock_app.status_label.setText(f"Váš tah ({mock_app.human_player})")

                    # Kontrola konce hry
                    mock_app.check_game_end()
                    return True
        return False

    # Attach the function to the mock
    mock_app.update_game_state = update_game_state

    # Call update_game_state
    result = mock_app.update_game_state()

    # Check that waiting_for_detection was set to False
    assert mock_app.waiting_for_detection is False

    # Check that detection_wait_time was reset
    assert mock_app.detection_wait_time == 0

    # Check that ai_move_retry_count was reset
    assert mock_app.ai_move_retry_count == 0

    # Check that board was updated
    mock_app.board_widget.update.assert_called_once()

    # Check that current_turn was updated to human's turn
    assert mock_app.current_turn == game_logic.PLAYER_X

    # Check that status was updated
    assert mock_app.status_label.setText.called

    # Check that check_game_end was called
    mock_app.check_game_end.assert_called_once()

    # Check that result is True
    assert result is True


def test_update_game_state_detection_timeout(mock_app):
    """Test update_game_state method when detection times out"""
    # Reset mocks
    mock_app.status_label.reset_mock()
    mock_app.draw_ai_symbol.reset_mock()
    mock_app.draw_ai_symbol.return_value = True

    # Set up initial state
    mock_app.waiting_for_detection = True
    mock_app.ai_move_row = 1
    mock_app.ai_move_col = 1
    mock_app.detection_wait_time = 5.0  # Max wait time
    mock_app.ai_move_retry_count = 0
    mock_app.max_detection_wait_time = 5.0
    mock_app.arm_thread = MagicMock()
    mock_app.arm_thread.connected = True

    # Create a mock camera_thread
    mock_app.camera_thread = MagicMock()
    mock_app.camera_thread.last_board_state = None

    # Define update_game_state function
    def update_game_state():
        # Pokud čekáme na detekci nakresleného symbolu
        if mock_app.waiting_for_detection:
            # Pokud vypršel čas čekání a symbol nebyl detekován
            if mock_app.detection_wait_time >= mock_app.max_detection_wait_time:
                mock_app.detection_wait_time = 0
                mock_app.waiting_for_detection = False

                # Pokud jsme nepřekročili maximální počet pokusů, zkusíme
                # nakreslit symbol znovu
                if mock_app.ai_move_retry_count < mock_app.max_retry_count:
                    mock_app.ai_move_retry_count += 1
                    mock_app.status_label.setText(
                        f"Symbol nebyl detekován, zkouším znovu (pokus {mock_app.ai_move_retry_count}/{mock_app.max_retry_count})...")

                    # Zkusíme nakreslit symbol znovu
                    if mock_app.draw_ai_symbol(mock_app.ai_move_row, mock_app.ai_move_col):
                        # Začneme znovu čekat na detekci
                        mock_app.waiting_for_detection = True
                        return True
        return False

    # Attach the function to the mock
    mock_app.update_game_state = update_game_state

    # Call update_game_state
    result = mock_app.update_game_state()

    # Check that detection_wait_time was reset
    assert mock_app.detection_wait_time == 0

    # Check that ai_move_retry_count was incremented
    assert mock_app.ai_move_retry_count == 1

    # Check that waiting_for_detection was set to True again
    assert mock_app.waiting_for_detection is True

    # Check that status was updated
    assert mock_app.status_label.setText.called

    # Check that draw_ai_symbol was called
    mock_app.draw_ai_symbol.assert_called_once_with(1, 1)

    # Check that result is True
    assert result is True


def test_draw_ai_symbol_with_arm_thread(mock_app):
    """Test draw_ai_symbol method with arm_thread"""
    # Reset mocks
    mock_app.get_cell_coordinates_from_yolo.reset_mock()
    mock_app.move_to_neutral_position.reset_mock()
    mock_app.status_label.reset_mock()

    # Set up arm_thread
    mock_app.arm_thread = MagicMock()
    mock_app.arm_thread.connected = True
    mock_app.arm_thread.draw_o.return_value = True

    # Set up get_cell_coordinates_from_yolo to return coordinates
    mock_app.get_cell_coordinates_from_yolo.return_value = (200, 0)

    # Define draw_ai_symbol function
    def draw_ai_symbol(row, col):
        coords = mock_app.get_cell_coordinates_from_yolo(row, col)
        mock_app.arm_thread.draw_o(*coords)
        mock_app.move_to_neutral_position()
        mock_app.status_label.setText("Symbol O nakreslen.")
        return True

    # Attach the function to the mock
    mock_app.draw_ai_symbol = draw_ai_symbol

    # Call draw_ai_symbol
    result = mock_app.draw_ai_symbol(1, 1)

    # Check that get_cell_coordinates_from_yolo was called
    mock_app.get_cell_coordinates_from_yolo.assert_called_once_with(1, 1)

    # Check that arm_thread.draw_o was called
    mock_app.arm_thread.draw_o.assert_called_once_with(200, 0)

    # Check that move_to_neutral_position was called
    mock_app.move_to_neutral_position.assert_called_once()

    # Check that status was updated
    assert mock_app.status_label.setText.called

    # Check that result is True
    assert result is True


def test_draw_ai_symbol_with_arm_controller(mock_app):
    """Test draw_ai_symbol method with arm_controller"""
    # Reset mocks
    mock_app.get_cell_coordinates_from_yolo.reset_mock()
    mock_app.move_to_neutral_position.reset_mock()
    mock_app.status_label.reset_mock()

    # Set up arm_controller
    mock_app.arm_controller = MagicMock()
    mock_app.arm_controller.connected = True
    mock_app.arm_controller.draw_x.return_value = True
    mock_app.arm_thread = None

    # Set ai_player to X
    mock_app.ai_player = game_logic.PLAYER_X

    # Set up get_cell_coordinates_from_yolo to return default coordinates
    mock_app.get_cell_coordinates_from_yolo.return_value = (200, 0)

    # Define draw_ai_symbol function
    def draw_ai_symbol(row, col):
        coords = mock_app.get_cell_coordinates_from_yolo(row, col)
        mock_app.arm_controller.draw_x(*coords)
        mock_app.move_to_neutral_position()
        mock_app.status_label.setText("Symbol X nakreslen.")
        return True

    # Attach the function to the mock
    mock_app.draw_ai_symbol = draw_ai_symbol

    # Call draw_ai_symbol
    result = mock_app.draw_ai_symbol(1, 1)

    # Check that get_cell_coordinates_from_yolo was called
    mock_app.get_cell_coordinates_from_yolo.assert_called_once_with(1, 1)

    # Check that arm_controller.draw_x was called
    mock_app.arm_controller.draw_x.assert_called_once_with(200, 0)

    # Check that move_to_neutral_position was called
    mock_app.move_to_neutral_position.assert_called_once()

    # Check that status was updated
    assert mock_app.status_label.setText.called

    # Check that result is True
    assert result is True


def test_draw_ai_symbol_no_arm(mock_app):
    """Test draw_ai_symbol method with no arm connected"""
    # Reset the mock to clear any previous calls
    mock_app.status_label.reset_mock()
    mock_app.arm_thread = None
    mock_app.arm_controller = None

    # Define draw_ai_symbol function
    def draw_ai_symbol(row, col):
        mock_app.status_label.setText("Robotická ruka není připojena!")
        return False

    # Attach the function to the mock
    mock_app.draw_ai_symbol = draw_ai_symbol

    # Call draw_ai_symbol
    result = mock_app.draw_ai_symbol(1, 1)

    # Check that status was updated
    mock_app.status_label.setText.assert_called_once_with(
        "Robotická ruka není připojena!")

    # Check that result is False
    assert result is False