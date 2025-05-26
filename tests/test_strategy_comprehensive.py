"""
Comprehensive pytest tests for strategy module.
Tests all strategy classes including RandomStrategy, MinimaxStrategy, and BernoulliStrategySelector.
"""

import pytest
from unittest.mock import Mock, patch
import random

from app.core.strategy import (
    Strategy, RandomStrategy, MinimaxStrategy, BernoulliStrategySelector,
    PLAYER_X, PLAYER_O, EMPTY
)
from app.core.game_state import GameState


@pytest.fixture
def mock_game_state():
    """Create mock GameState for testing."""
    mock_state = Mock(spec=GameState)
    mock_state.board = [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]
    mock_state.get_valid_moves.return_value = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    mock_state.board_to_string.return_value = "Board string representation"
    mock_state.check_winner.return_value = None
    mock_state.is_board_full.return_value = False
    return mock_state


@pytest.fixture
def partial_game_state():
    """Create game state with some moves already made."""
    mock_state = Mock(spec=GameState)
    mock_state.board = [
        [PLAYER_X, EMPTY, EMPTY],
        [EMPTY, PLAYER_O, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]
    mock_state.get_valid_moves.return_value = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    mock_state.board_to_string.return_value = "Partial board"
    mock_state.check_winner.return_value = None
    mock_state.is_board_full.return_value = False
    return mock_state


@pytest.fixture
def winning_game_state():
    """Create game state where someone has won."""
    mock_state = Mock(spec=GameState)
    mock_state.board = [
        [PLAYER_X, PLAYER_X, PLAYER_X],
        [PLAYER_O, PLAYER_O, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]
    mock_state.get_valid_moves.return_value = []
    mock_state.check_winner.return_value = PLAYER_X
    mock_state.is_board_full.return_value = False
    return mock_state


class TestStrategyBase:
    """Test abstract Strategy base class."""

    def test_strategy_initialization_x(self):
        """Test Strategy initialization with X player."""
        # We can't instantiate abstract class, so test via subclass
        strategy = RandomStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O

    def test_strategy_initialization_o(self):
        """Test Strategy initialization with O player."""
        strategy = RandomStrategy(PLAYER_O)
        assert strategy.player == PLAYER_O
        assert strategy.opponent == PLAYER_X

    def test_strategy_logger_setup(self):
        """Test that Strategy sets up logger correctly."""
        strategy = RandomStrategy(PLAYER_X)
        assert strategy.logger is not None
        assert "RandomStrategy" in strategy.logger.name

    def test_strategy_abstract_method(self):
        """Test that Strategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Strategy(PLAYER_X)


class TestRandomStrategy:
    """Test RandomStrategy implementation."""

    def test_random_strategy_valid_moves(self, mock_game_state):
        """Test RandomStrategy returns valid move."""
        strategy = RandomStrategy(PLAYER_X)
        
        # Set random seed for reproducible test
        random.seed(42)
        move = strategy.suggest_move(mock_game_state)
        
        assert move is not None
        assert move in mock_game_state.get_valid_moves()
        mock_game_state.get_valid_moves.assert_called_once()

    def test_random_strategy_no_moves(self):
        """Test RandomStrategy when no moves available."""
        strategy = RandomStrategy(PLAYER_X)
        
        mock_state = Mock(spec=GameState)
        mock_state.get_valid_moves.return_value = []
        
        move = strategy.suggest_move(mock_state)
        assert move is None

    def test_random_strategy_single_move(self):
        """Test RandomStrategy with only one available move."""
        strategy = RandomStrategy(PLAYER_O)
        
        mock_state = Mock(spec=GameState)
        mock_state.get_valid_moves.return_value = [(1, 1)]
        
        move = strategy.suggest_move(mock_state)
        assert move == (1, 1)

    def test_random_strategy_multiple_calls(self, mock_game_state):
        """Test RandomStrategy returns different moves on multiple calls."""
        strategy = RandomStrategy(PLAYER_X)
        
        moves = []
        for _ in range(10):
            move = strategy.suggest_move(mock_game_state)
            moves.append(move)
        
        # All moves should be valid
        for move in moves:
            assert move in mock_game_state.get_valid_moves()
        
        # Should get some variation (not all the same move)
        unique_moves = set(moves)
        assert len(unique_moves) >= 2  # Should have at least some variation


class TestMinimaxStrategy:
    """Test MinimaxStrategy implementation."""

    def test_minimax_initialization(self):
        """Test MinimaxStrategy initialization."""
        strategy = MinimaxStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O

    def test_minimax_finished_game(self, winning_game_state):
        """Test MinimaxStrategy with finished game."""
        strategy = MinimaxStrategy(PLAYER_X)
        move = strategy.suggest_move(winning_game_state)
        assert move is None

    def test_minimax_empty_board(self, mock_game_state):
        """Test MinimaxStrategy with empty board."""
        strategy = MinimaxStrategy(PLAYER_X)
        move = strategy.suggest_move(mock_game_state)
        
        assert move is not None
        assert move in mock_game_state.get_valid_moves()

    def test_minimax_partial_board(self, partial_game_state):
        """Test MinimaxStrategy with partially filled board."""
        strategy = MinimaxStrategy(PLAYER_O)
        move = strategy.suggest_move(partial_game_state)
        
        assert move is not None
        assert move in partial_game_state.get_valid_moves()

    def test_minimax_board_methods(self):
        """Test MinimaxStrategy internal board methods."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Test _is_board_full
        full_board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        assert strategy._is_board_full(full_board) is True
        
        empty_board = [
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert strategy._is_board_full(empty_board) is False

    def test_minimax_get_available_moves(self):
        """Test MinimaxStrategy _get_available_moves method."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        moves = strategy._get_available_moves(board)
        expected_moves = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        assert set(moves) == set(expected_moves)

    def test_minimax_check_winner_methods(self):
        """Test MinimaxStrategy winner checking methods."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Test X wins row
        x_wins_board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert strategy._check_winner(x_wins_board) == PLAYER_X
        
        # Test O wins column
        o_wins_board = [
            [PLAYER_O, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        assert strategy._check_winner(o_wins_board) == PLAYER_O
        
        # Test no winner
        no_winner_board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert strategy._check_winner(no_winner_board) is None


class TestBernoulliStrategySelector:
    """Test BernoulliStrategySelector implementation."""

    def test_bernoulli_initialization(self):
        """Test BernoulliStrategySelector initialization."""
        selector = BernoulliStrategySelector(PLAYER_X, difficulty=5)
        assert selector.player == PLAYER_X
        assert selector.difficulty == 5
        assert isinstance(selector.random_strategy, RandomStrategy)
        assert isinstance(selector.minimax_strategy, MinimaxStrategy)

    def test_bernoulli_difficulty_bounds(self):
        """Test BernoulliStrategySelector with boundary difficulty values."""
        # Minimum difficulty
        selector_min = BernoulliStrategySelector(PLAYER_X, difficulty=1)
        assert selector_min.difficulty == 1
        
        # Maximum difficulty
        selector_max = BernoulliStrategySelector(PLAYER_X, difficulty=10)
        assert selector_max.difficulty == 10

    def test_bernoulli_probability_calculation(self):
        """Test probability calculation based on difficulty."""
        # Low difficulty should favor random moves
        selector_low = BernoulliStrategySelector(PLAYER_X, difficulty=2)
        prob_low = selector_low._calculate_minimax_probability()
        assert 0.0 <= prob_low <= 1.0
        assert prob_low < 0.5  # Should favor random at low difficulty
        
        # High difficulty should favor minimax moves
        selector_high = BernoulliStrategySelector(PLAYER_X, difficulty=9)
        prob_high = selector_high._calculate_minimax_probability()
        assert 0.0 <= prob_high <= 1.0
        assert prob_high > 0.5  # Should favor minimax at high difficulty

    def test_bernoulli_suggest_move_calls_strategies(self, mock_game_state):
        """Test that BernoulliStrategySelector calls appropriate strategies."""
        selector = BernoulliStrategySelector(PLAYER_X, difficulty=5)
        
        # Mock the strategies
        selector.random_strategy = Mock()
        selector.minimax_strategy = Mock()
        selector.random_strategy.suggest_move.return_value = (0, 0)
        selector.minimax_strategy.suggest_move.return_value = (1, 1)
        
        # Test multiple calls to see both strategies get used
        moves = []
        for _ in range(20):
            move = selector.suggest_move(mock_game_state)
            moves.append(move)
        
        # Should have called both strategies
        assert selector.random_strategy.suggest_move.call_count > 0
        assert selector.minimax_strategy.suggest_move.call_count > 0

    @patch('random.random')
    def test_bernoulli_strategy_selection_random(self, mock_random, mock_game_state):
        """Test strategy selection when random number favors random strategy."""
        mock_random.return_value = 0.9  # High value should select random strategy
        
        selector = BernoulliStrategySelector(PLAYER_X, difficulty=8)  # High difficulty
        selector.random_strategy = Mock()
        selector.minimax_strategy = Mock()
        selector.random_strategy.suggest_move.return_value = (0, 0)
        
        move = selector.suggest_move(mock_game_state)
        
        selector.random_strategy.suggest_move.assert_called_once()
        selector.minimax_strategy.suggest_move.assert_not_called()
        assert move == (0, 0)

    @patch('random.random')
    def test_bernoulli_strategy_selection_minimax(self, mock_random, mock_game_state):
        """Test strategy selection when random number favors minimax strategy."""
        mock_random.return_value = 0.1  # Low value should select minimax strategy
        
        selector = BernoulliStrategySelector(PLAYER_X, difficulty=8)  # High difficulty
        selector.random_strategy = Mock()
        selector.minimax_strategy = Mock()
        selector.minimax_strategy.suggest_move.return_value = (1, 1)
        
        move = selector.suggest_move(mock_game_state)
        
        selector.minimax_strategy.suggest_move.assert_called_once()
        selector.random_strategy.suggest_move.assert_not_called()
        assert move == (1, 1)

    def test_bernoulli_difficulty_extremes(self, mock_game_state):
        """Test behavior at difficulty extremes."""
        # Difficulty 1 should mostly use random strategy
        selector_min = BernoulliStrategySelector(PLAYER_X, difficulty=1)
        prob_min = selector_min._calculate_minimax_probability()
        assert prob_min < 0.2  # Should be very low
        
        # Difficulty 10 should mostly use minimax strategy
        selector_max = BernoulliStrategySelector(PLAYER_X, difficulty=10)
        prob_max = selector_max._calculate_minimax_probability()
        assert prob_max > 0.8  # Should be very high


class TestStrategyIntegration:
    """Test integration scenarios between strategies."""

    def test_strategy_consistency(self, mock_game_state):
        """Test that strategies return consistent valid moves."""
        strategies = [
            RandomStrategy(PLAYER_X),
            MinimaxStrategy(PLAYER_X),
            BernoulliStrategySelector(PLAYER_X, difficulty=5)
        ]
        
        for strategy in strategies:
            move = strategy.suggest_move(mock_game_state)
            assert move is None or move in mock_game_state.get_valid_moves()

    def test_strategy_performance_characteristics(self, mock_game_state):
        """Test performance characteristics of different strategies."""
        # RandomStrategy should be fast
        random_strategy = RandomStrategy(PLAYER_X)
        move = random_strategy.suggest_move(mock_game_state)
        assert move is not None
        
        # MinimaxStrategy should also work
        minimax_strategy = MinimaxStrategy(PLAYER_X)
        move = minimax_strategy.suggest_move(mock_game_state)
        assert move is not None
        
        # BernoulliStrategySelector should combine both
        bernoulli_strategy = BernoulliStrategySelector(PLAYER_X, difficulty=5)
        move = bernoulli_strategy.suggest_move(mock_game_state)
        assert move is not None

    def test_all_strategies_handle_edge_cases(self):
        """Test that all strategies handle edge cases properly."""
        # No moves available
        no_moves_state = Mock(spec=GameState)
        no_moves_state.get_valid_moves.return_value = []
        
        strategies = [
            RandomStrategy(PLAYER_X),
            MinimaxStrategy(PLAYER_X),
            BernoulliStrategySelector(PLAYER_X, difficulty=5)
        ]
        
        for strategy in strategies:
            move = strategy.suggest_move(no_moves_state)
            assert move is None