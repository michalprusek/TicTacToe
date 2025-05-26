# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Extended tests for strategy module covering more functionality.
"""
import pytest
from unittest.mock import Mock, patch
from app.core.strategy import BernoulliStrategySelector, RandomStrategy
from app.main.game_logic import PLAYER_X, PLAYER_O
from app.core.game_state import GameState


class TestRandomStrategy:
    """Test RandomStrategy class."""

    def test_init(self):
        """Test RandomStrategy initialization."""
        strategy = RandomStrategy(player=PLAYER_X)
        assert strategy.player == PLAYER_X
        assert hasattr(strategy, 'logger')

    def test_suggest_move_with_available_moves(self):
        """Test suggest_move when moves are available."""
        strategy = RandomStrategy(player=PLAYER_X)
        
        # Mock game state with available moves
        game_state = Mock()
        game_state.get_valid_moves.return_value = [(0, 0), (0, 1), (1, 1)]
        
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = (0, 0)
            
            move = strategy.suggest_move(game_state)
            assert move == (0, 0)
            mock_choice.assert_called_once_with([(0, 0), (0, 1), (1, 1)])

    def test_suggest_move_no_available_moves(self):
        """Test suggest_move when no moves are available."""
        strategy = RandomStrategy(player=PLAYER_X)
        
        # Mock game state with no available moves
        game_state = Mock()
        game_state.get_valid_moves.return_value = []
        
        move = strategy.suggest_move(game_state)
        assert move is None


class TestBernoulliStrategySelectorExtended:
    """Extended tests for BernoulliStrategySelector."""

    def test_select_strategy_low_probability(self):
        """Test strategy selection with low probability."""
        selector = BernoulliStrategySelector(p=0.1)
        
        # With p=0.1, should mostly select 'random'
        with patch('random.random', return_value=0.5):  # > 0.1
            strategy = selector.select_strategy()
            assert strategy == 'random'

    def test_select_strategy_high_probability(self):
        """Test strategy selection with high probability."""
        selector = BernoulliStrategySelector(p=0.9)
        
        # With p=0.9, should mostly select 'minimax'
        with patch('random.random', return_value=0.5):  # < 0.9
            strategy = selector.select_strategy()
            assert strategy == 'minimax'

    def test_get_move_method_exists(self):
        """Test get_move method exists and can be called."""
        selector = BernoulliStrategySelector(p=0.5)
        
        # Test that get_move method exists
        assert hasattr(selector, 'get_move')
        assert callable(selector.get_move)

    def test_logger_setup(self):
        """Test logger is properly set up."""
        selector = BernoulliStrategySelector(p=0.5)
        
        assert hasattr(selector, 'logger')
        assert selector.logger is not None
