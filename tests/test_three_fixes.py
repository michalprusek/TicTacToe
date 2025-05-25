"""
Test suite for the three specific fixes:
1. Škrtanec výherní čáry se nesmaže po výměně papíru
2. Ruka nereaguje po výměně papíru
3. Obtížnost 10 nehraje optimální minimax tahy
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from app.core.strategy import BernoulliStrategySelector
from app.core.config import AppConfig
from app.main import game_logic


class TestThreeFixes:
    """Test the three specific fixes."""

    @pytest.fixture
    def strategy_selector(self):
        """Create a BernoulliStrategySelector for testing."""
        return BernoulliStrategySelector(difficulty=10)

    def test_fix_1_winning_line_cleared_on_empty_board_detection(self, qtbot):
        """Test that winning line is cleared when empty board is detected."""
        # Mock the TicTacToeApp to avoid full initialization
        with patch('app.main.pyqt_gui.CameraThread'), \
             patch('app.core.arm_thread.ArmThread'):

            from app.main.pyqt_gui import TicTacToeApp
            config = AppConfig()
            app = TicTacToeApp(config=config)
            qtbot.addWidget(app)

            # Set up a winning line
            app.board_widget.winning_line = [(0, 0), (0, 1), (0, 2)]
            app.board_widget.board = [
                [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X],
                [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.EMPTY],
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
            ]

            # Mock logger
            app.logger = Mock()

            # Simulate empty board detection (using game_logic.EMPTY)
            empty_board = [
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
            ]

            # Call handle_detected_game_state with empty board
            app.handle_detected_game_state(empty_board)

            # Verify winning line was cleared
            assert app.board_widget.winning_line is None

            # Verify logging messages
            app.logger.info.assert_any_call("🆕 Prázdná hrací plocha detekována - resetuji hru a čistím výherní čáru")
            app.logger.info.assert_any_call("✅ Výherní čára vymazána")

    def test_fix_2_arm_flags_reset_on_empty_board_detection(self, qtbot):
        """Test that arm flags are reset when empty board is detected."""
        # Mock the TicTacToeApp to avoid full initialization
        with patch('app.main.pyqt_gui.CameraThread'), \
             patch('app.core.arm_thread.ArmThread'):

            from app.main.pyqt_gui import TicTacToeApp
            config = AppConfig()
            app = TicTacToeApp(config=config)
            qtbot.addWidget(app)

            # Set up arm flags as if arm was busy
            app.arm_move_in_progress = True
            app.arm_move_scheduled = True
            app.waiting_for_detection = True
            app.last_arm_move_time = time.time()

            # Set up non-empty board initially
            app.board_widget.board = [
                [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
                [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_O],
                [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.EMPTY]
            ]

            # Mock logger and reset_arm_flags method
            app.logger = Mock()
            app.reset_arm_flags = Mock()

            # Simulate empty board detection (using game_logic.EMPTY)
            empty_board = [
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
                [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
            ]

            # Call handle_detected_game_state with empty board
            app.handle_detected_game_state(empty_board)

            # Verify reset_arm_flags was called
            app.reset_arm_flags.assert_called_once()

            # Verify logging messages
            app.logger.info.assert_any_call("✅ Všechny arm flags resetovány pro novou hru")

    def test_fix_3_difficulty_10_uses_minimax_strategy(self, strategy_selector):
        """Test that difficulty 10 always uses minimax strategy."""
        # Test that difficulty 10 sets p=1.0
        assert strategy_selector.difficulty == 10
        assert strategy_selector.p == 1.0

        # Test strategy selection multiple times to ensure it's always minimax
        for _ in range(10):
            strategy = strategy_selector.select_strategy()
            assert strategy == 'minimax'

    def test_fix_3_strategy_logging(self, strategy_selector):
        """Test that strategy selection is properly logged."""
        # Mock the logger
        strategy_selector.logger = Mock()

        # Test strategy selection
        strategy = strategy_selector.select_strategy()

        # Verify logging was called
        strategy_selector.logger.info.assert_called()

        # Check the log message contains expected information
        log_call = strategy_selector.logger.info.call_args[0][0]
        assert "🎯 STRATEGY SELECTION:" in log_call
        assert "difficulty=10" in log_call
        assert "p=1.00" in log_call
        assert "selected='minimax'" in log_call

    def test_fix_3_get_move_logging(self, strategy_selector):
        """Test that get_move method logs which strategy is used."""
        # Mock the logger
        strategy_selector.logger = Mock()

        # Create a simple board
        board = [
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # Test get_move
        move = strategy_selector.get_move(board, game_logic.PLAYER_O)

        # Verify logging was called for strategy selection and move execution
        strategy_selector.logger.info.assert_any_call("🧠 USING MINIMAX STRATEGY for player O")
        strategy_selector.logger.info.assert_any_call(f"🎯 MINIMAX SELECTED MOVE: {move}")

    def test_fix_3_minimax_vs_random_at_different_difficulties(self):
        """Test that different difficulties produce different strategy selections."""
        # Test difficulty 0 (should always use random)
        strategy_0 = BernoulliStrategySelector(difficulty=0)
        strategy_0.logger = Mock()

        # Test multiple selections for difficulty 0
        random_count = 0
        for _ in range(10):
            if strategy_0.select_strategy() == 'random':
                random_count += 1

        # Should be mostly random (allowing for some variance due to randomness)
        assert random_count >= 8

        # Test difficulty 10 (should always use minimax)
        strategy_10 = BernoulliStrategySelector(difficulty=10)
        strategy_10.logger = Mock()

        # Test multiple selections for difficulty 10
        minimax_count = 0
        for _ in range(10):
            if strategy_10.select_strategy() == 'minimax':
                minimax_count += 1

        # Should always be minimax
        assert minimax_count == 10

    def test_fix_3_minimax_algorithm_quality(self):
        """Test that minimax algorithm produces optimal moves."""
        # Test case 1: Winning move detection
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]

        # X should win by playing (0, 2)
        move = game_logic.get_best_move(board, game_logic.PLAYER_X)
        assert move == (0, 2)

        # O should block X by playing (0, 2) or win by playing (1, 2)
        move = game_logic.get_best_move(board, game_logic.PLAYER_O)
        assert move in [(0, 2), (1, 2)]  # Both are valid optimal moves

        # Test case 2: Center preference on empty board
        empty_board = game_logic.create_board()
        move = game_logic.get_best_move(empty_board, game_logic.PLAYER_X)
        assert move == (1, 1)  # Center

        # Test case 3: Fork creation/blocking
        board = [
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.PLAYER_O]
        ]

        # O should block the diagonal win - multiple valid moves possible
        move = game_logic.get_best_move(board, game_logic.PLAYER_O)
        # Valid blocking moves: (2, 0) blocks diagonal, (0, 2) blocks diagonal
        assert move in [(2, 0), (0, 2)]  # Both block the diagonal

    def test_bernoulli_strategy_selector_probability_calculation(self):
        """Test that BernoulliStrategySelector correctly calculates probabilities."""
        test_cases = [
            (0, 0.0),
            (1, 0.1),
            (5, 0.5),
            (8, 0.8),
            (10, 1.0),
            (15, 1.0),  # Should be clamped to 1.0
            (-5, 0.0),  # Should be clamped to 0.0
        ]

        for difficulty, expected_p in test_cases:
            selector = BernoulliStrategySelector(difficulty=difficulty)
            assert selector.p == expected_p
            assert selector.difficulty == max(0, min(10, difficulty))

    def test_empty_board_detection_logic(self):
        """Test the logic for detecting empty boards."""
        # Test empty board
        empty_board = [
            ['', '', ''],
            ['', '', ''],
            ['', '', '']
        ]

        is_empty = all(cell == '' for row in empty_board for cell in row)
        assert is_empty is True

        # Test non-empty board
        non_empty_board = [
            ['X', '', ''],
            ['', '', ''],
            ['', '', '']
        ]

        is_empty = all(cell == '' for row in non_empty_board for cell in row)
        assert is_empty is False

        # Test board with game_logic.EMPTY
        empty_board_with_constants = game_logic.create_board()
        is_empty = all(cell == game_logic.EMPTY for row in empty_board_with_constants for cell in row)
        assert is_empty is True
