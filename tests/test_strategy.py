"""
Unit tests for Strategy classes.
"""
import unittest
from unittest.mock import patch, MagicMock
import random

from app.core.strategy import (
    RandomStrategy,
    MinimaxStrategy,
    BernoulliStrategySelector,
    FixedStrategySelector,
    create_strategy,
    STRATEGY_MAP
)
from app.core.game_state import GameState, PLAYER_X, PLAYER_O, EMPTY


class TestRandomStrategy(unittest.TestCase):
    """Test cases for RandomStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = RandomStrategy(PLAYER_X)

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.player, PLAYER_X)
        self.assertEqual(self.strategy.opponent, PLAYER_O)

    @patch('random.choice')
    def test_suggest_move(self, mock_choice):
        """Test move suggestion."""
        game_state = GameState()
        
        # Mock random choice
        expected_move = (1, 1)
        mock_choice.return_value = expected_move
        
        move = self.strategy.suggest_move(game_state)
        
        self.assertEqual(move, expected_move)
        mock_choice.assert_called_once()

    def test_suggest_move_no_valid_moves(self):
        """Test suggestion when no moves available."""
        game_state = GameState()
        
        # Fill the board
        for r in range(3):
            for c in range(3):
                game_state._board_state[r][c] = PLAYER_X
        
        move = self.strategy.suggest_move(game_state)
        self.assertIsNone(move)


class TestMinimaxStrategy(unittest.TestCase):
    """Test cases for MinimaxStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MinimaxStrategy(PLAYER_O)

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.player, PLAYER_O)
        self.assertEqual(self.strategy.opponent, PLAYER_X)

    def test_suggest_move_empty_board(self):
        """Test suggestion on empty board."""
        game_state = GameState()
        
        move = self.strategy.suggest_move(game_state)
        
        # Should suggest center on empty board
        self.assertEqual(move, (1, 1))

    def test_suggest_move_winning_opportunity(self):
        """Test winning move detection."""
        game_state = GameState()
        
        # Set up board where O can win
        game_state._board_state[0][0] = PLAYER_O
        game_state._board_state[0][1] = PLAYER_O
        # game_state._board_state[0][2] = EMPTY  # O can win here
        
        move = self.strategy.suggest_move(game_state)
        
        # Should choose winning move
        self.assertEqual(move, (0, 2))

    def test_suggest_move_blocking(self):
        """Test blocking opponent's win."""
        game_state = GameState()
        
        # Set up board where X is about to win
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[0][1] = PLAYER_X
        # game_state._board_state[0][2] = EMPTY  # X can win here, O should block
        
        move = self.strategy.suggest_move(game_state)
        
        # Should block X's winning move
        self.assertEqual(move, (0, 2))

    def test_suggest_move_no_moves(self):
        """Test suggestion when board is full."""
        game_state = GameState()
        
        # Fill board
        for r in range(3):
            for c in range(3):
                game_state._board_state[r][c] = PLAYER_X
        
        move = self.strategy.suggest_move(game_state)
        self.assertIsNone(move)


class TestBernoulliStrategySelector(unittest.TestCase):
    """Test cases for BernoulliStrategySelector."""

    def test_initialization_with_probability(self):
        """Test initialization with probability."""
        selector = BernoulliStrategySelector(p=0.7)
        self.assertEqual(selector.p, 0.7)
        self.assertEqual(selector.difficulty, 7)

    def test_initialization_with_difficulty(self):
        """Test initialization with difficulty."""
        selector = BernoulliStrategySelector(difficulty=8)
        self.assertEqual(selector.difficulty, 8)
        self.assertEqual(selector.p, 0.8)

    def test_probability_clamping(self):
        """Test probability value clamping."""
        selector = BernoulliStrategySelector(p=1.5)
        self.assertEqual(selector.p, 1.0)
        
        selector.p = -0.5
        self.assertEqual(selector.p, 0.0)

    def test_difficulty_clamping(self):
        """Test difficulty value clamping."""
        selector = BernoulliStrategySelector(difficulty=15)
        self.assertEqual(selector.difficulty, 10)
        
        selector.difficulty = -5
        self.assertEqual(selector.difficulty, 0)

    @patch('random.random')
    def test_select_strategy(self, mock_random):
        """Test strategy selection."""
        selector = BernoulliStrategySelector(p=0.7)
        
        # Mock random to return 0.5 (< 0.7, should select minimax)
        mock_random.return_value = 0.5
        strategy = selector.select_strategy()
        self.assertEqual(strategy, 'minimax')
        
        # Mock random to return 0.8 (>= 0.7, should select random)
        mock_random.return_value = 0.8
        strategy = selector.select_strategy()
        self.assertEqual(strategy, 'random')


class TestFixedStrategySelector(unittest.TestCase):
    """Test cases for FixedStrategySelector."""

    def test_initialization_valid(self):
        """Test initialization with valid strategy."""
        selector = FixedStrategySelector('minimax')
        self.assertEqual(selector.strategy_type, 'minimax')

    def test_initialization_invalid(self):
        """Test initialization with invalid strategy."""
        with self.assertRaises(ValueError):
            FixedStrategySelector('invalid_strategy')

    def test_select_strategy(self):
        """Test strategy selection."""
        selector = FixedStrategySelector('random')
        self.assertEqual(selector.select_strategy(), 'random')
        
        selector = FixedStrategySelector('minimax')
        self.assertEqual(selector.select_strategy(), 'minimax')


class TestStrategyFactory(unittest.TestCase):
    """Test cases for strategy factory functions."""

    def test_create_strategy_valid(self):
        """Test creating valid strategies."""
        strategy = create_strategy('random', PLAYER_X)
        self.assertIsInstance(strategy, RandomStrategy)
        self.assertEqual(strategy.player, PLAYER_X)
        
        strategy = create_strategy('minimax', PLAYER_O)
        self.assertIsInstance(strategy, MinimaxStrategy)
        self.assertEqual(strategy.player, PLAYER_O)

    def test_create_strategy_invalid(self):
        """Test creating invalid strategy."""
        with self.assertRaises(ValueError):
            create_strategy('invalid_strategy', PLAYER_X)

    def test_strategy_map(self):
        """Test strategy mapping."""
        self.assertIn('random', STRATEGY_MAP)
        self.assertIn('minimax', STRATEGY_MAP)
        self.assertEqual(STRATEGY_MAP['random'], RandomStrategy)
        self.assertEqual(STRATEGY_MAP['minimax'], MinimaxStrategy)


if __name__ == '__main__':
    unittest.main()