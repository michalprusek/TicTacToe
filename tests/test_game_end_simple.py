"""
Simple test for game end fixes without GUI initialization.
"""

import pytest
from unittest.mock import Mock, patch

from app.core.strategy import BernoulliStrategySelector
from app.main import game_logic


class TestGameEndSimple:
    """Simple tests for game end functionality."""

    def test_bernoulli_strategy_difficulty_setting(self):
        """Test BernoulliStrategySelector difficulty setting."""
        # Test initial difficulty
        strategy = BernoulliStrategySelector(difficulty=5)
        assert strategy.difficulty == 5
        assert strategy.p == 0.5

        # Test setting difficulty
        strategy.difficulty = 8
        assert strategy.difficulty == 8
        assert strategy.p == 0.8

        # Test edge cases
        strategy.difficulty = 0
        assert strategy.difficulty == 0
        assert strategy.p == 0.0

        strategy.difficulty = 10
        assert strategy.difficulty == 10
        assert strategy.p == 1.0

        # Test clamping
        strategy.difficulty = 15  # Should be clamped to 10
        assert strategy.difficulty == 10
        assert strategy.p == 1.0

        strategy.difficulty = -5  # Should be clamped to 0
        assert strategy.difficulty == 0
        assert strategy.p == 0.0

    def test_bernoulli_strategy_selection(self):
        """Test BernoulliStrategySelector strategy selection."""
        strategy = BernoulliStrategySelector(difficulty=7)  # p = 0.7

        # Mock random to test deterministic behavior
        with patch('app.core.strategy.random.random') as mock_random:
            # Test case where random < p (should select advanced)
            mock_random.return_value = 0.5  # Less than 0.7
            selected = strategy.select_strategy()
            assert selected == 'minimax'  # Advanced strategy

            # Test case where random >= p (should select basic)
            mock_random.return_value = 0.8  # Greater than 0.7
            selected = strategy.select_strategy()
            assert selected == 'random'  # Basic strategy

    def test_game_logic_winner_detection(self):
        """Test game logic winner detection."""
        # Test horizontal win
        board = [
            ['X', 'X', 'X'],
            ['O', 'O', ''],
            ['', '', '']
        ]
        winner = game_logic.check_winner(board)
        assert winner == 'X'

        # Test vertical win
        board = [
            ['O', 'X', ''],
            ['O', 'X', ''],
            ['O', '', '']
        ]
        winner = game_logic.check_winner(board)
        assert winner == 'O'

        # Test diagonal win
        board = [
            ['X', 'O', ''],
            ['O', 'X', ''],
            ['', '', 'X']
        ]
        winner = game_logic.check_winner(board)
        assert winner == 'X'

        # Test tie
        board = [
            ['X', 'O', 'X'],
            ['O', 'O', 'X'],
            ['O', 'X', 'O']
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.TIE

        # Test no winner
        board = [
            ['X', 'O', ''],
            ['O', 'X', ''],
            ['', '', '']
        ]
        winner = game_logic.check_winner(board)
        assert winner == '' or winner is None

    def test_winning_line_detection(self):
        """Test winning line detection."""
        # Test horizontal winning line
        board = [
            ['X', 'X', 'X'],
            ['O', 'O', ''],
            ['', '', '']
        ]
        winning_line = game_logic.get_winning_line(board)
        expected = [(0, 0), (0, 1), (0, 2)]
        assert winning_line == expected

        # Test vertical winning line
        board = [
            ['O', 'X', ''],
            ['O', 'X', ''],
            ['O', '', '']
        ]
        winning_line = game_logic.get_winning_line(board)
        expected = [(0, 0), (1, 0), (2, 0)]
        assert winning_line == expected

        # Test diagonal winning line
        board = [
            ['X', 'O', ''],
            ['O', 'X', ''],
            ['', '', 'X']
        ]
        winning_line = game_logic.get_winning_line(board)
        expected = [(0, 0), (1, 1), (2, 2)]
        assert winning_line == expected

    def test_empty_board_detection(self):
        """Test empty board detection."""
        # Test empty board
        board = [
            ['', '', ''],
            ['', '', ''],
            ['', '', '']
        ]
        is_empty = all(cell == '' for row in board for cell in row)
        assert is_empty is True

        # Test non-empty board
        board = [
            ['X', '', ''],
            ['', '', ''],
            ['', '', '']
        ]
        is_empty = all(cell == '' for row in board for cell in row)
        assert is_empty is False

    def test_board_creation(self):
        """Test board creation."""
        board = game_logic.create_board()
        assert len(board) == 3
        assert len(board[0]) == 3
        assert all(cell == game_logic.EMPTY for row in board for cell in row)

    def test_strategy_selector_probability_property(self):
        """Test BernoulliStrategySelector probability property."""
        strategy = BernoulliStrategySelector(p=0.3)
        assert strategy.p == 0.3
        assert strategy.difficulty == 3

        # Test setting probability
        strategy.p = 0.7
        assert strategy.p == 0.7
        assert strategy.difficulty == 7

        # Test clamping
        strategy.p = 1.5  # Should be clamped to 1.0
        assert strategy.p == 1.0

        strategy.p = -0.5  # Should be clamped to 0.0
        assert strategy.p == 0.0
