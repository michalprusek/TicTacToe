"""
Safe PyQt testing fixtures and utilities.
Provides tools for testing PyQt applications without segmentation faults.
"""
import pytest
import sys
from unittest.mock import MagicMock

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from app.main import game_logic


@pytest.fixture(scope="session")
def qt_app():
    """
    Create a QApplication instance for the test session.
    
    This fixture ensures only one QApplication instance is created
    for the entire test session, which helps prevent segfaults.
    """
    app = QApplication.instance()
    if app is None:
        # Create a new application with offscreen rendering
        app = QApplication(["-platform", "offscreen"])
    
    # Make sure the platform is set to offscreen
    app.setAttribute(Qt.AA_UseDesktopOpenGL, False)
    app.setAttribute(Qt.AA_UseSoftwareOpenGL, True)
    
    yield app
    
    # Do not call app.quit() here as it can cause segfaults during cleanup


class MockTicTacToeAppSafe:
    """
    Safe mock version of TicTacToeApp for testing.
    
    This class doesn't inherit from any PyQt classes and only implements
    the minimum functionality needed for testing.
    """
    
    def __init__(self, debug_mode=False, camera_index=0, difficulty=0.5):
        """
        Initialize the safe mock TicTacToeApp.
        
        Args:
            debug_mode: Whether debug mode is enabled
            camera_index: Camera index to use
            difficulty: AI difficulty level
        """
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
        
        # Mock UI components
        self.board_widget = MagicMock()
        self.board_widget.board = game_logic.create_board()
        self.status_label = MagicMock()
        self.difficulty_value_label = MagicMock()
        self.difficulty_slider = MagicMock()
        self.camera_view = MagicMock()
        self.debug_button = MagicMock()
        
        # Mock controllers and threads
        self.arm_thread = MagicMock()
        self.arm_controller = MagicMock()
        self.camera_thread = MagicMock()
        self.strategy_selector = MagicMock()
        self.debug_window = MagicMock()
        
        # Mock position data
        self.neutral_position = {"x": 200, "y": 0, "z": 100}
        self.calibration_data = {}
    
    # Game state methods
    def reset_game(self):
        """Reset the game state."""
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
        """Check if the game has ended."""
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
    
    # Event handlers
    def handle_board_click(self, row, col):
        """Handle board click event."""
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
            self.make_ai_move()
    
    def handle_debug_button_click(self):
        """Handle debug button click event."""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.debug_button.setText("Vypnout debug")
        else:
            if self.debug_window:
                self.debug_window.close()
            self.debug_button.setText("Zapnout debug")
    
    def handle_difficulty_changed(self, value):
        """Handle difficulty slider change event."""
        self.difficulty = value / 100.0
        self.difficulty_value_label.setText(f"{value}%")
        
        if hasattr(self.strategy_selector, 'set_difficulty'):
            self.strategy_selector.set_difficulty(self.difficulty)
    
    # Drawing methods
    def draw_winning_line(self):
        """Draw the winning line."""
        return True
    
    def draw_ai_symbol(self, row, col):
        """Draw the AI symbol."""
        return True
    
    def move_to_neutral_position(self):
        """Move the arm to neutral position."""
        return True
    
    def make_ai_move(self):
        """Make AI move."""
        move = self.strategy_selector.get_move(self.board_widget.board, self.ai_player)
        
        if move is None:
            self.status_label.setText("AI nemůže najít vhodný tah!")
            return
        
        row, col = move
        self.board_widget.board[row][col] = self.ai_player
        self.board_widget.update()
        
        self.check_game_end()
        
        if not self.game_over:
            self.current_turn = self.human_player
            self.status_label.setText(f"Váš tah ({self.human_player})")


@pytest.fixture
def safe_tic_tac_toe_app(qt_app):
    """
    Create a safe mock TicTacToeApp instance for testing.
    
    This fixture depends on qt_app to ensure that a QApplication
    exists, but the MockTicTacToeAppSafe doesn't actually use it.
    
    Args:
        qt_app: QApplication instance from the qt_app fixture
        
    Returns:
        MockTicTacToeAppSafe: Safe mock app for testing
    """
    return MockTicTacToeAppSafe()


class GameEndCheckTestUtilsSafe:
    """
    Utility methods for testing game end checking functionality.
    
    These are safe versions that work with the MockTicTacToeAppSafe class.
    """
    
    @staticmethod
    def prepare_app_for_testing(app):
        """Set up the app for testing game end conditions."""
        app.human_player = game_logic.PLAYER_X
        app.ai_player = game_logic.PLAYER_O
        app.current_turn = game_logic.PLAYER_X
        app.game_over = False
        app.winner = None
        app.draw_winning_line = MagicMock()
        app.status_label.setText.reset_mock()
        app.board_widget.update.reset_mock()
    
    @staticmethod
    def test_check_game_end_no_winner(app):
        """Test check_game_end method with no winner."""
        # Set up board with no winner
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        
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
        
        # Call check_game_end
        app.check_game_end()
        
        # Check that game_over is True
        assert app.game_over
        
        # Check that winner is human player
        assert app.winner == game_logic.PLAYER_X
        
        # Check that status was updated
        app.status_label.setText.assert_called_once_with("Vyhráli jste!")
        
        # Check that winning_line was set
        assert app.board_widget.winning_line is not None
        
        # Check that draw_winning_line was not called (human win)
        app.draw_winning_line.assert_not_called()


class EventHandlingTestUtilsSafe:
    """
    Utility methods for testing event handling functionality.
    
    These are safe versions that work with the MockTicTacToeAppSafe class.
    """
    
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
        # Set up strategy_selector
        app.strategy_selector = MagicMock()
        app.difficulty_value_label = MagicMock()
        
        # Call the method
        app.handle_difficulty_changed(value)
        
        # Check that difficulty was updated
        assert app.difficulty == value / 100.0
        
        # Check that difficulty_value_label was updated
        app.difficulty_value_label.setText.assert_called_once_with(f"{value}%")
        
        # Check that strategy_selector.set_difficulty was called
        app.strategy_selector.set_difficulty.assert_called_once_with(value / 100.0)


class PyQtGuiTestCaseSafe:
    """
    Base class for PyQt GUI test cases that avoids segmentation faults.
    
    This class provides methods for safely creating and testing PyQt components.
    """
    
    @staticmethod
    def create_app_instance(args=None):
        """
        Create a QApplication instance for testing.
        
        Args:
            args: Command line arguments for QApplication
            
        Returns:
            QApplication: The application instance
        """
        if args is None:
            args = ["-platform", "offscreen"]
            
        app = QApplication.instance()
        if app is None:
            app = QApplication(args)
            
        # Ensure we're using software rendering
        app.setAttribute(Qt.AA_UseDesktopOpenGL, False)
        app.setAttribute(Qt.AA_UseSoftwareOpenGL, True)
        
        return app
    
    @staticmethod
    def create_test_app(debug_mode=False, camera_index=0, difficulty=0.5):
        """
        Create a safe mock TicTacToeApp instance for testing.
        
        Args:
            debug_mode: Whether debug mode is enabled
            camera_index: Camera index to use
            difficulty: AI difficulty level
            
        Returns:
            tuple: (MockTicTacToeAppSafe, qt_app) - The app instance and QApplication
        """
        # Create QApplication instance
        qt_app = PyQtGuiTestCaseSafe.create_app_instance()
        
        # Create mock app
        app = MockTicTacToeAppSafe(debug_mode, camera_index, difficulty)
        
        return app, qt_app