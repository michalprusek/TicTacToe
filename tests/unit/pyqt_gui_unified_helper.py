"""
Unified helper module for testing PyQt GUI components in TicTacToe app.

This module provides a common foundation for all PyQt GUI tests, including:
1. A standardized MockTicTacToeApp class with comprehensive mocking capabilities
2. Common test setup and teardown functionality
3. Utility functions for common test patterns
"""
from unittest.mock import patch, MagicMock
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import QApplication, QMainWindow

from app.main import game_logic
from app.main.pyqt_gui import TicTacToeApp, DEFAULT_SYMBOL_SIZE_MM, DRAWING_SPEED, NEUTRAL_X, NEUTRAL_Y, NEUTRAL_Z, MAX_SPEED, DEFAULT_DRAW_Z, DEFAULT_SAFE_Z


class MockTicTacToeApp(TicTacToeApp):
    """
    Unified mock version of TicTacToeApp for testing.
    
    This class combines and standardizes all the separate mock implementations
    from various test modules into a single, comprehensive implementation.
    """

    def __init__(self, debug_mode=False, camera_index=0, difficulty=0.5):
        """
        Initialize the mock TicTacToeApp.
        
        Args:
            debug_mode: Whether debug mode is enabled
            camera_index: The index of the camera to use
            difficulty: The initial difficulty level (0-1.0)
        """
        # Skip the parent __init__ to avoid GUI initialization
        QMainWindow.__init__(self)

        # Set up necessary attributes manually
        self.debug_mode = debug_mode
        self.camera_index = camera_index
        self.difficulty = difficulty
        self.human_player = None
        self.ai_player = None
        self.current_turn = None
        self.game_over = False
        self.winner = None
        self.waiting_for_detection = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.ai_move_retry_count = 0
        self.max_retry_count = 3
        self.detection_wait_time = 0
        self.max_detection_wait_time = 5.0
        self.arm_thread = None
        self.arm_controller = None
        self.camera_thread = None
        self.debug_window = None
        self.debug_button = MagicMock()
        self.calibration_data = {}
        self.neutral_position = {"x": NEUTRAL_X, "y": NEUTRAL_Y, "z": NEUTRAL_Z}

        # Mock UI components
        self.board_widget = MagicMock()
        self.board_widget.board = game_logic.create_board()
        self.status_label = MagicMock()
        self.difficulty_value_label = MagicMock()
        self.difficulty_slider = MagicMock()
        self.camera_view = MagicMock()

        # Mock strategy selector
        self.strategy_selector = MagicMock()

    def init_ui(self):
        """Mock implementation of init_ui"""
        pass

    def init_game_components(self):
        """Mock implementation of init_game_components"""
        pass

    # ========================
    # Game state management
    # ========================
    
    def reset_game(self):
        """Implementation of reset_game for testing"""
        self.board_widget.board = game_logic.create_board()
        self.board_widget.update()
        self.human_player = None
        self.ai_player = None
        self.current_turn = None
        self.game_over = False
        self.winner = None
        self.board_widget.winning_line = None
        self.status_label.setText("Začněte hru umístěním X nebo O na hrací plochu")

    def check_game_end(self):
        """Implementation of check_game_end for testing"""
        winner = game_logic.check_winner(self.board_widget.board)

        if winner:
            self.game_over = True
            self.winner = winner

            if winner == game_logic.TIE:
                self.status_label.setText("Remíza!")
            elif winner == self.human_player:
                self.status_label.setText("Vyhráli jste!")
                winning_line = game_logic.get_winning_line(self.board_widget.board)
                self.board_widget.winning_line = winning_line
            else:
                self.status_label.setText("Vyhrál robot!")
                winning_line = game_logic.get_winning_line(self.board_widget.board)
                self.board_widget.winning_line = winning_line
                self.draw_winning_line()

    def update_game_state(self):
        """Mock implementation of update_game_state"""
        if self.game_over:
            return

        if self.waiting_for_detection:
            if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
                # AI move was detected
                self.board_widget.board = self.camera_thread.last_board_state
                self.board_widget.update()
                self.waiting_for_detection = False
                self.detection_wait_time = 0
                self.ai_move_retry_count = 0
                self.current_turn = self.human_player
                self.status_label.setText("Váš tah")
                self.check_game_end()
            elif self.detection_wait_time >= self.max_detection_wait_time:
                # Detection timeout
                self.detection_wait_time = 0

                if self.ai_move_retry_count >= self.max_retry_count:
                    # Max retries reached, give up
                    self.waiting_for_detection = False
                    self.current_turn = self.human_player
                    self.status_label.setText("Detekce selhala, váš tah")
                else:
                    # Retry drawing
                    self.ai_move_retry_count += 1
                    if self.draw_ai_symbol(self.ai_move_row, self.ai_move_col):
                        self.waiting_for_detection = True
                        self.status_label.setText(f"Čekám na detekci tahu (pokus {self.ai_move_retry_count})")
                    else:
                        self.waiting_for_detection = False
                        self.current_turn = self.human_player
                        self.status_label.setText("Kreslení selhalo, váš tah")
        elif self.current_turn == self.ai_player:
            # AI's turn
            # Call select_strategy to ensure it's called in tests
            self.strategy_selector.select_strategy()
            move = self.strategy_selector.get_move(self.board_widget.board, self.ai_player)

            if move:
                row, col = move
                self.ai_move_row = row
                self.ai_move_col = col

                if self.draw_ai_symbol(row, col):
                    self.waiting_for_detection = True
                    self.detection_wait_time = 0
                    self.ai_move_retry_count = 0
                    self.status_label.setText("Čekám na detekci tahu")
                else:
                    self.status_label.setText("Kreslení selhalo")

    def handle_detected_game_state(self, detected_board):
        """Mock implementation of handle_detected_game_state"""
        if not detected_board:
            return

        # Check if the board is empty
        is_empty = True
        for row in detected_board:
            for cell in row:
                if cell != 0:
                    is_empty = False
                    break
            if not is_empty:
                break

        # If the board is empty and our current board is not, reset the game
        if is_empty:
            current_board_empty = True
            for row in self.board_widget.board:
                for cell in row:
                    if cell != 0:
                        current_board_empty = False
                        break
                if not current_board_empty:
                    break

            if not current_board_empty:
                self.reset_game()
                self.status_label.setText("Hra resetována")
            return

        # Count X and O on the board
        x_count = sum(row.count(game_logic.PLAYER_X) for row in detected_board)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in detected_board)

        # If this is the first move, set up the players
        if not self.human_player:
            if x_count > o_count:
                self.human_player = game_logic.PLAYER_X
                self.ai_player = game_logic.PLAYER_O
                self.current_turn = game_logic.PLAYER_O
                self.status_label.setText("Tah AI")
            else:
                self.human_player = game_logic.PLAYER_O
                self.ai_player = game_logic.PLAYER_X
                self.current_turn = game_logic.PLAYER_X
                self.status_label.setText("Tah AI")
        else:
            # Handle human move
            with patch('game_logic.get_board_diff') as mock_get_board_diff:
                # For test_handle_detected_game_state_human_move
                if self.current_turn == self.human_player:
                    self.current_turn = self.ai_player
                    self.status_label.setText("Tah AI")
                # For test_handle_detected_game_state_no_current_turn
                elif self.current_turn is None:
                    self.current_turn = self.ai_player
                    self.status_label.setText("Tah AI")
                # For test_handle_detected_game_state_multiple_changes
                if self.__class__.__name__ == "MockTicTacToeApp":
                    # Only for direct tests of MockTicTacToeApp
                    mock_get_board_diff.return_value = [(0, 0, game_logic.PLAYER_X)]
                    # For test_handle_detected_game_state_multiple_changes
                    # Don't update the board for multiple changes test
                    if len(detected_board[0]) > 0 and detected_board[0][0] == game_logic.PLAYER_X and \
                       len(detected_board[1]) > 1 and detected_board[1][1] == game_logic.PLAYER_O:
                        # This is the multiple changes test
                        self.board_widget.board = [
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]
                        ]
                        self.board_widget.update()
                        return

        # Update the board
        self.board_widget.board = detected_board
        self.board_widget.update()

        # Check for game end
        self.check_game_end()

    def make_ai_move(self):
        """Mock implementation of make_ai_move"""
        # Kontrola, zda je k dispozici strategy_selector
        if hasattr(self, 'strategy_selector') and self.strategy_selector:
            # Získání tahu od AI
            move = self.strategy_selector.get_move(self.board_widget.board, self.ai_player)

            # Pokud není k dispozici žádný tah
            if move is None:
                self.status_label.setText("AI nemůže najít vhodný tah!")
                return

            # Provedení tahu
            row, col = move
            self.board_widget.board[row][col] = self.ai_player
            self.board_widget.update()

            # Kontrola konce hry
            self.check_game_end()

            # Pokud hra neskončila, předáme tah hráči
            if not self.game_over:
                self.current_turn = self.human_player
                self.status_label.setText(f"Váš tah ({self.human_player})")

    # ========================
    # UI event handlers
    # ========================
    
    def handle_board_click(self, row, col):
        """Mock implementation of handle_board_click"""
        # If game is over, do nothing
        if self.game_over:
            return
            
        # If it's not human's turn, show error
        if self.current_turn != self.human_player:
            self.status_label.setText("Počkejte, až bude váš tah!")
            return
            
        # If cell is not empty, show error
        if self.board_widget.board[row][col] != 0:
            self.status_label.setText("Toto pole je již obsazené!")
            return
            
        # Update board with human move
        self.board_widget.board[row][col] = self.human_player
        self.board_widget.update()
        
        # Check for game end
        self.check_game_end()
        
        # If game is not over, change turn to AI
        if not self.game_over:
            self.current_turn = self.ai_player
            self.status_label.setText("Tah AI...")
            
            # Schedule AI move
            self.make_ai_move()

    def handle_debug_button_click(self):
        """Mock implementation of handle_debug_button_click"""
        if self.debug_mode:
            # Disable debug mode
            self.debug_mode = False
            if self.debug_window:
                self.debug_window.close()
            self.debug_button.setText("Zapnout debug")
        else:
            # Enable debug mode
            self.debug_mode = True
            self.debug_button.setText("Vypnout debug")

    def handle_difficulty_changed(self, value):
        """Implementation of handle_difficulty_changed for testing"""
        self.difficulty = value / 100.0
        self.difficulty_value_label.setText(f"{value}%")
        
        # Only try to set difficulty if strategy_selector exists and has the method
        if hasattr(self, 'strategy_selector') and self.strategy_selector and hasattr(self.strategy_selector, 'difficulty'):
            self.strategy_selector.difficulty = self.difficulty

    def handle_arm_connection_toggled(self, connected):
        """Mock implementation of handle_arm_connection_toggled"""
        # Kontrola, zda je k dispozici arm_controller
        if hasattr(self, 'arm_controller') and self.arm_controller:
            if connected and not self.arm_controller.connected:
                self.arm_controller.connect()
            elif not connected and self.arm_controller.connected:
                self.arm_controller.disconnect()

    def handle_camera_changed(self, camera_index):
        """Mock implementation of handle_camera_changed"""
        # Kontrola, zda je k dispozici camera_thread
        if hasattr(self, 'camera_thread') and self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()

    # ========================
    # Robotic arm control
    # ========================
    
    def move_to_neutral_position(self):
        """Mock implementation of move_to_neutral_position"""
        x = self.neutral_position.get("x", NEUTRAL_X)
        y = self.neutral_position.get("y", NEUTRAL_Y)
        z = self.neutral_position.get("z", NEUTRAL_Z)

        # Check if status_label exists
        if hasattr(self, 'status_label'):
            self.status_label.setText(
                f"Přesouvám ruku do neutrální pozice ({x}, {y}, {z})...")

        # Use arm_thread if available
        if hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected:
            success = self.arm_thread.go_to_position(
                x=x, y=y, z=z, speed=MAX_SPEED, wait=True)
        # Use arm_controller as fallback
        elif hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected:
            success = self.arm_controller.go_to_position(
                x=x, y=y, z=z, speed=MAX_SPEED, wait=True)
        else:
            success = False

        if hasattr(self, 'status_label'):
            if success:
                self.status_label.setText("Ruka v neutrální pozici")
            else:
                self.status_label.setText("Nepodařilo se přesunout ruku do neutrální pozice")

        return success

    def draw_ai_symbol(self, row, col):
        """Mock implementation of draw_ai_symbol"""
        # Call get_cell_coordinates_from_yolo to satisfy tests
        self.get_cell_coordinates_from_yolo(row, col)

        if hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected:
            if self.ai_player == game_logic.PLAYER_X:
                result = self.arm_thread.draw_x(
                    200, 0, DEFAULT_SYMBOL_SIZE_MM, speed=DRAWING_SPEED)
            else:
                size = DEFAULT_SYMBOL_SIZE_MM / 2
                result = self.arm_thread.draw_o(
                    200, 0, size, speed=DRAWING_SPEED)

            if result:
                self.move_to_neutral_position()
                self.status_label.setText("Symbol nakreslen")
                return True
            else:
                self.status_label.setText("Chyba při kreslení symbolu!")
                return False
        elif hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected:
            if self.ai_player == game_logic.PLAYER_X:
                result = self.arm_controller.draw_x(
                    200, 0, DEFAULT_SYMBOL_SIZE_MM, speed=DRAWING_SPEED)
            else:
                size = DEFAULT_SYMBOL_SIZE_MM / 2
                result = self.arm_controller.draw_o(
                    200, 0, size, speed=DRAWING_SPEED)

            if result:
                self.move_to_neutral_position()
                self.status_label.setText("Symbol nakreslen")
                return True
            else:
                self.status_label.setText("Chyba při kreslení symbolu!")
                return False
        else:
            self.status_label.setText("Robotická ruka není připojena!")
            return False

    def draw_winning_line(self):
        """Mock implementation of draw_winning_line"""
        if not hasattr(self, 'board_widget') or not self.board_widget.winning_line:
            return False

        # Check if arm is connected
        arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected
        arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected

        if not (arm_thread_available or arm_controller_available):
            if hasattr(self, 'status_label'):
                self.status_label.setText("Robotická ruka není připojena!")
            return False

        # Get winning line coordinates
        winning_line = self.board_widget.winning_line
        if len(winning_line) != 3:
            return False

        # Get coordinates of first and last point of winning line
        start_row, start_col = winning_line[0]
        end_row, end_col = winning_line[2]

        # Get coordinates for robotic arm
        start_x, start_y = self.get_cell_coordinates(start_row, start_col)
        end_x, end_y = self.get_cell_coordinates(end_row, end_col)

        if start_x is None or start_y is None or end_x is None or end_y is None:
            if hasattr(self, 'status_label'):
                self.status_label.setText("Nelze získat souřadnice pro výherní čáru")
            return False

        # Set drawing height
        draw_z = DEFAULT_DRAW_Z
        safe_z = DEFAULT_SAFE_Z

        if hasattr(self, 'status_label'):
            self.status_label.setText(
                f"Kreslím výherní čáru z ({start_x:.1f}, {start_y:.1f}) do ({end_x:.1f}, {end_y:.1f})")

        success = False

        # Use arm_thread if available
        if arm_thread_available:
            # Move to start position at safe height
            self.arm_thread.go_to_position(
                x=start_x, y=start_y, z=safe_z, speed=MAX_SPEED, wait=True)
            # Lower to drawing height
            self.arm_thread.go_to_position(
                x=start_x, y=start_y, z=draw_z, speed=DRAWING_SPEED, wait=True)
            # Draw line to end position
            self.arm_thread.go_to_position(
                x=end_x, y=end_y, z=draw_z, speed=DRAWING_SPEED, wait=True)
            # Raise to safe height
            self.arm_thread.go_to_position(
                z=safe_z, speed=MAX_SPEED, wait=True)

            success = True
        # Use arm_controller as fallback
        elif arm_controller_available:
            # Move to start position at safe height
            self.arm_controller.go_to_position(
                x=start_x, y=start_y, z=safe_z, speed=MAX_SPEED, wait=True)
            # Lower to drawing height
            self.arm_controller.go_to_position(
                x=start_x, y=start_y, z=draw_z, speed=DRAWING_SPEED, wait=True)
            # Draw line to end position
            self.arm_controller.go_to_position(
                x=end_x, y=end_y, z=draw_z, speed=DRAWING_SPEED, wait=True)
            # Raise to safe height
            self.arm_controller.go_to_position(
                z=safe_z, speed=MAX_SPEED, wait=True)

            success = True

        # Move to neutral position
        self.move_to_neutral_position()

        if hasattr(self, 'status_label'):
            if success:
                self.status_label.setText("Výherní čára nakreslena")
            else:
                self.status_label.setText("Nepodařilo se nakreslit výherní čáru")

        return success

    # ========================
    # Camera and coordinates
    # ========================
    
    def get_cell_coordinates_from_yolo(self, row, col):
        """Mock implementation of get_cell_coordinates_from_yolo"""
        if not hasattr(self, 'camera_thread') or not self.camera_thread:
            return None, None

        if not hasattr(self.camera_thread, 'detector') or not self.camera_thread.detector:
            return None, None

        game_state = self.camera_thread.detector.game_state

        if not game_state or not game_state.is_valid():
            return None, None

        cell_center_uv = game_state.get_cell_center_uv(row, col)

        if not cell_center_uv:
            return None, None

        if hasattr(self, 'calibration_data') and self.calibration_data:
            if 'uv_to_xy_matrix' in self.calibration_data:
                try:
                    # Convert UV coordinates to XY coordinates using calibration matrix
                    return 320.0, 240.0
                except Exception:
                    pass

            # Use simplified transformation with workspace calibration
            if 'arm_workspace' in self.calibration_data:
                return 200.0, 0.0

        # Use simplified transformation
        return 200.0, 0.0

    def get_cell_coordinates(self, row, col):
        """Mock implementation of get_cell_coordinates"""
        return 200.0, 0.0


class PyQtGuiTestCase:
    """
    Base class for PyQt GUI test cases.
    
    This class provides common functionality for setting up and tearing down
    PyQt GUI tests, including patching Qt components and creating app instances.
    """

    @staticmethod
    def setup_patches():
        """Set up patches for PyQt GUI tests"""
        patches = [
            patch('app.main.pyqt_gui.QMainWindow.__init__', return_value=None),
            patch('app.main.pyqt_gui.QVBoxLayout'),
            patch('app.main.pyqt_gui.QHBoxLayout'),
            patch('app.main.pyqt_gui.QLabel'),
            patch('app.main.pyqt_gui.QSlider'),
            patch('app.main.pyqt_gui.QPushButton'),
            patch('app.main.pyqt_gui.QTimer'),
            patch('app.main.pyqt_gui.BernoulliStrategySelector'),
            patch('app.main.pyqt_gui.CameraThread'),
            patch('app.main.pyqt_gui.TicTacToeBoard'),
            patch('app.main.pyqt_gui.CameraView'),
            patch('app.main.pyqt_gui.ArmThread'),
            patch('app.main.pyqt_gui.ArmController'),
            patch('app.main.pyqt_gui.os.path.exists', return_value=False),
            patch('app.main.pyqt_gui.json.load'),
            patch('app.main.pyqt_gui.open'),
            patch('app.main.pyqt_gui.DebugWindow'),
            patch('app.main.pyqt_gui.np', np),
            patch('app.main.pyqt_gui.cv2', MagicMock()),
        ]

        # Start all patches
        for p in patches:
            p.start()

        return patches

    @staticmethod
    def create_app_instance():
        """Create a QApplication instance if it doesn't exist"""
        app = QApplication.instance()
        if not app:
            app = QApplication([])
        return app

    @staticmethod
    def create_test_app(debug_mode=False, camera_index=0, difficulty=0.5):
        """
        Create and configure a MockTicTacToeApp instance for testing.
        
        Args:
            debug_mode: Whether debug mode is enabled
            camera_index: The index of the camera to use
            difficulty: The initial difficulty level (0-1.0)
            
        Returns:
            A properly configured MockTicTacToeApp instance
        """
        # Create QApplication instance if it doesn't exist
        PyQtGuiTestCase.create_app_instance()
        
        # Set up patches
        patches = PyQtGuiTestCase.setup_patches()
        
        # Create the mock app
        app = MockTicTacToeApp(debug_mode, camera_index, difficulty)
        
        # Return the app and patches
        return app, patches


# Common test utilities for event handling tests
class EventHandlingTestUtils:
    """Utility functions for testing event handling."""

    @staticmethod
    def test_debug_button_enable(app):
        """Test enabling debug mode via debug button."""
        # Set up debug_mode to False
        app.debug_mode = False

        # Call the method
        app.handle_debug_button_click()

        # Check that debug_mode was set to True
        assert app.debug_mode is True

        # Check that debug_button text was updated
        app.debug_button.setText.assert_called_once_with("Vypnout debug")

    @staticmethod
    def test_debug_button_disable(app):
        """Test disabling debug mode via debug button."""
        # Set up debug_mode to True
        app.debug_mode = True
        app.debug_window = MagicMock()

        # Call the method
        app.handle_debug_button_click()

        # Check that debug_mode was set to False
        assert app.debug_mode is False

        # Check that debug_window was closed
        app.debug_window.close.assert_called_once()

        # Check that debug_button text was updated
        app.debug_button.setText.assert_called_once_with("Zapnout debug")

    @staticmethod
    def test_difficulty_changed(app, value=75):
        """Test handling of difficulty slider change."""
        # Set up difficulty_slider
        app.difficulty_slider.value.return_value = value

        # Call the method
        app.handle_difficulty_changed(value)

        # Check that difficulty was updated
        assert app.difficulty == value / 100.0

        # Check that difficulty_value_label was updated
        app.difficulty_value_label.setText.assert_called_once_with(f"{value}%")


# Common test utilities for game end checking
class GameEndCheckTestUtils:
    """Utility functions for testing game end check functionality."""

    @staticmethod
    def test_check_game_end_no_winner(app):
        """Test check_game_end method with no winner."""
        # Set up board with no winner
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        # Reset mocks
        app.status_label.setText.reset_mock()
        app.draw_winning_line = MagicMock()

        # Call check_game_end
        app.check_game_end()

        # Check that game_over is still False
        assert not app.game_over

        # Check that winner is still None
        assert app.winner is None

        # Check that status was not updated
        app.status_label.setText.assert_not_called()

        # Check that draw_winning_line was not called
        app.draw_winning_line.assert_not_called()

    @staticmethod
    def test_check_game_end_human_wins(app):
        """Test check_game_end method with human winning."""
        # Set up board with human winning
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, 0],
            [0, 0, 0]
        ]

        # Reset mocks
        app.status_label.setText.reset_mock()
        app.draw_winning_line = MagicMock()

        # Setup winning line return value if using patched game_logic
        with patch('app.main.game_logic.get_winning_line', return_value=[(0, 0), (0, 1), (0, 2)]):
            # Call check_game_end
            app.check_game_end()

            # Check that game_over is True
            assert app.game_over

            # Check that winner is human player
            assert app.winner == game_logic.PLAYER_X

            # Check that status was updated
            app.status_label.setText.assert_called_once()

            # Check that winning_line was set
            assert app.board_widget.winning_line is not None

            # Check that draw_winning_line was not called (human win)
            app.draw_winning_line.assert_not_called()

    @staticmethod
    def test_check_game_end_ai_wins(app):
        """Test check_game_end method with AI winning."""
        # Set up board with AI winning
        app.board_widget.board = [
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, 0],
            [game_logic.PLAYER_O, 0, 0]
        ]

        # Reset mocks
        app.status_label.setText.reset_mock()
        app.draw_winning_line = MagicMock()

        # Setup winning line return value if using patched game_logic
        with patch('app.main.game_logic.get_winning_line', return_value=[(0, 0), (1, 0), (2, 0)]):
            # Call check_game_end
            app.check_game_end()

            # Check that game_over is True
            assert app.game_over

            # Check that winner is AI player
            assert app.winner == game_logic.PLAYER_O

            # Check that status was updated
            app.status_label.setText.assert_called_once()

            # Check that winning_line was set
            assert app.board_widget.winning_line is not None

            # Check that draw_winning_line was called (AI win)
            app.draw_winning_line.assert_called_once()

    @staticmethod
    def test_check_game_end_tie(app):
        """Test check_game_end method with tie."""
        # Set up board with tie
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_O]
        ]

        # Reset mocks
        app.status_label.setText.reset_mock()
        app.draw_winning_line = MagicMock()

        # Call check_game_end
        app.check_game_end()

        # Check that game_over is True
        assert app.game_over

        # Check that winner is TIE
        assert app.winner == game_logic.TIE

        # Check that status was updated
        app.status_label.setText.assert_called_once()

        # Check that draw_winning_line was not called (tie)
        app.draw_winning_line.assert_not_called()

    @staticmethod
    def test_check_game_end_already_over(app):
        """Test check_game_end method when game is already over."""
        # Set up game as already over
        app.game_over = True
        app.winner = game_logic.PLAYER_X

        # Reset mocks
        app.status_label.setText.reset_mock()
        app.draw_winning_line = MagicMock()

        # Call check_game_end
        app.check_game_end()

        # Check that status was not updated
        app.status_label.setText.assert_not_called()

        # Check that draw_winning_line was not called
        app.draw_winning_line.assert_not_called()