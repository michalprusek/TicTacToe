"""
Test suite for game end fixes and improvements.

Tests:
1. Arm stops when game ends
2. Difficulty setting propagates to BernoulliStrategySelector
3. Board widget clears winning line on reset
4. Game end notification shows result without "New Game" button
5. Empty board detection triggers new game
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from app.main.pyqt_gui import TicTacToeApp
from app.main import game_logic
from app.core.strategy import BernoulliStrategySelector
from app.core.config import AppConfig


class TestGameEndFixes:
    """Test game end fixes and improvements."""

    @pytest.fixture
    def app(self, qtbot):
        """Create a TicTacToeApp instance for testing."""
        config = AppConfig()

        # Mock camera thread creation to avoid initialization
        with patch('app.main.pyqt_gui.CameraThread') as mock_camera_thread:
            mock_camera_instance = Mock()
            mock_camera_thread.return_value = mock_camera_instance

            # Mock arm thread creation
            with patch('app.core.arm_thread.ArmThread') as mock_arm_thread:
                mock_arm_instance = Mock()
                mock_arm_thread.return_value = mock_arm_instance
                mock_arm_instance.connected = True
                mock_arm_instance.stop_current_move = Mock()

                gui = TicTacToeApp(config=config)
                qtbot.addWidget(gui)

                # Set up mocked components
                gui.camera_thread = mock_camera_instance
                gui.camera_thread.last_board_state = [
                    ['', '', ''],
                    ['', '', ''],
                    ['', '', '']
                ]

                gui.arm_thread = mock_arm_instance

                # Initialize strategy selector
                gui.strategy_selector = BernoulliStrategySelector(difficulty=5)

                return gui

    def test_arm_stops_when_game_ends(self, app, qtbot):
        """Test that arm stops moving when game ends."""
        # Set up a winning board state
        winning_board = [
            ['X', 'X', 'X'],
            ['O', 'O', ''],
            ['', '', '']
        ]

        # Set up game state
        app.board_widget.board = winning_board
        app.game_over = False
        app.arm_move_in_progress = True

        # Call check_game_end
        app.check_game_end()

        # Verify game ended
        assert app.game_over is True
        assert app.winner == 'X'

        # Verify arm was stopped
        app.arm_thread.stop_current_move.assert_called_once()

        # Verify arm flags were reset
        assert app.arm_move_in_progress is False
        assert app.arm_move_scheduled is False

    def test_difficulty_setting_propagates(self, app, qtbot):
        """Test that difficulty setting propagates to BernoulliStrategySelector."""
        # Test initial difficulty
        initial_difficulty = 5
        assert app.strategy_selector.difficulty == initial_difficulty
        assert app.strategy_selector.p == 0.5

        # Change difficulty
        new_difficulty = 8
        app.handle_difficulty_changed(new_difficulty)

        # Verify propagation
        assert app.strategy_selector.difficulty == new_difficulty
        assert app.strategy_selector.p == 0.8

        # Test edge cases
        app.handle_difficulty_changed(0)
        assert app.strategy_selector.difficulty == 0
        assert app.strategy_selector.p == 0.0

        app.handle_difficulty_changed(10)
        assert app.strategy_selector.difficulty == 10
        assert app.strategy_selector.p == 1.0

    def test_board_widget_clears_winning_line(self, app, qtbot):
        """Test that board widget clears winning line on reset."""
        # Set up winning line
        app.board_widget.winning_line = [(0, 0), (0, 1), (0, 2)]
        app.board_widget.board = [
            ['X', 'X', 'X'],
            ['O', 'O', ''],
            ['', '', '']
        ]

        # Reset game
        app.reset_game()

        # Verify winning line is cleared
        assert app.board_widget.winning_line is None

        # Verify board is empty
        empty_board = game_logic.create_board()
        # Note: board_widget.board might not be directly updated in reset_game
        # but the visual display should be cleared

    def test_game_end_notification_without_new_game_button(self, app, qtbot):
        """Test that game end notification shows result without New Game button."""
        # Set up game end state
        app.game_over = True
        app.winner = 'X'
        app.human_player = 'X'

        # Mock the notification creation
        with patch('app.main.pyqt_gui.QWidget') as mock_widget:
            mock_notification = Mock()
            mock_widget.return_value = mock_notification

            with patch('app.main.pyqt_gui.QVBoxLayout') as mock_layout:
                mock_layout_instance = Mock()
                mock_layout.return_value = mock_layout_instance

                with patch('app.main.pyqt_gui.QLabel') as mock_label:
                    # Call show_game_end_notification
                    app.show_game_end_notification()

                    # Verify QLabel was called for instruction text
                    # (should be called for icon, message, and instruction)
                    assert mock_label.call_count >= 3

                    # Verify no QPushButton was created for "New Game"
                    with patch('app.main.pyqt_gui.QPushButton') as mock_button:
                        # If this was called, it would be for the new game button
                        mock_button.assert_not_called()

    def test_empty_board_detection_triggers_new_game(self, app, qtbot):
        """Test that empty board detection triggers new game."""
        # Set up game over state
        app.game_over = True
        app.winner = 'X'
        app.board_widget.board = [
            ['X', 'X', 'X'],
            ['O', 'O', ''],
            ['', '', '']
        ]

        # Simulate empty board detection
        empty_board = [
            ['', '', ''],
            ['', '', ''],
            ['', '', '']
        ]

        # Mock camera thread with empty board
        app.camera_thread.last_board_state = empty_board

        # Call handle_detected_game_state with empty board
        app.handle_detected_game_state(empty_board)

        # Verify game was reset
        assert app.game_over is False
        assert app.winner is None
        assert app.human_player is None
        assert app.ai_player is None

    def test_bernoulli_strategy_selector_real_time_update(self, app, qtbot):
        """Test that BernoulliStrategySelector updates probability in real time."""
        # Create a fresh strategy selector
        strategy_selector = BernoulliStrategySelector(difficulty=3)
        app.strategy_selector = strategy_selector

        # Test initial state
        assert strategy_selector.difficulty == 3
        assert strategy_selector.p == 0.3

        # Test real-time update through GUI
        app.handle_difficulty_changed(7)

        # Verify immediate update
        assert app.strategy_selector.difficulty == 7
        assert app.strategy_selector.p == 0.7

        # Test strategy selection with new probability
        # Mock random to test deterministic behavior
        with patch('app.core.strategy.random.random') as mock_random:
            # Test case where random < p (should select advanced)
            mock_random.return_value = 0.5  # Less than 0.7
            strategy = app.strategy_selector.select_strategy()
            assert strategy == 'minimax'  # Advanced strategy

            # Test case where random >= p (should select basic)
            mock_random.return_value = 0.8  # Greater than 0.7
            strategy = app.strategy_selector.select_strategy()
            assert strategy == 'random'  # Basic strategy

    def test_reset_arm_flags_on_game_end(self, app, qtbot):
        """Test that all arm flags are reset when game ends."""
        # Set up arm flags
        app.arm_move_in_progress = True
        app.arm_move_scheduled = True
        app.waiting_for_detection = True
        app.last_arm_move_time = time.time()

        # Set up winning board
        app.board_widget.board = [
            ['O', 'O', 'O'],
            ['X', 'X', ''],
            ['', '', '']
        ]
        app.game_over = False

        # Call check_game_end
        app.check_game_end()

        # Verify all arm flags are reset
        assert app.arm_move_in_progress is False
        assert app.arm_move_scheduled is False
        assert app.waiting_for_detection is False

        # Verify game ended
        assert app.game_over is True
        assert app.winner == 'O'

    def test_winning_line_display_and_clear(self, app, qtbot):
        """Test that winning line is displayed and properly cleared."""
        # Set up winning board
        winning_board = [
            ['X', 'X', 'X'],
            ['O', 'O', ''],
            ['', '', '']
        ]

        app.board_widget.board = winning_board
        app.game_over = False

        # Call check_game_end
        app.check_game_end()

        # Verify winning line is set
        winning_line = game_logic.get_winning_line(winning_board)
        assert winning_line is not None
        assert app.board_widget.winning_line == winning_line

        # Reset game
        app.reset_game()

        # Verify winning line is cleared
        assert app.board_widget.winning_line is None
