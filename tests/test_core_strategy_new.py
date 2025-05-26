"""
Comprehensive tests for app.core.strategy module using pytest.
Tests all strategy classes and related functionality.
"""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock
from app.core.strategy import (
    Strategy, RandomStrategy, MinimaxStrategy, StrategySelector, 
    FixedStrategySelector, BernoulliStrategySelector, STRATEGY_MAP, create_strategy
)
from app.core.game_state import GameState, PLAYER_X, PLAYER_O, EMPTY


class TestStrategy:
    """Test base Strategy class."""
    
    def test_strategy_is_abstract(self):
        """Test that Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy(PLAYER_X)
    
    def test_strategy_subclass_initialization(self):
        """Test Strategy subclass initialization."""
        
        class TestStrategy(Strategy):
            def suggest_move(self, game_state):
                return None
        
        strategy = TestStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O
        assert strategy.logger is not None
        
        strategy_o = TestStrategy(PLAYER_O)
        assert strategy_o.player == PLAYER_O
        assert strategy_o.opponent == PLAYER_X


class TestRandomStrategy:
    """Test RandomStrategy class."""
    
    def test_random_strategy_initialization(self):
        """Test RandomStrategy initialization."""
        strategy = RandomStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O
    
    def test_suggest_move_with_valid_moves(self):
        """Test suggest_move with available moves."""
        strategy = RandomStrategy(PLAYER_X)
        game_state = Mock()
        game_state.get_valid_moves.return_value = [(0, 0), (0, 1), (1, 1)]
        
        with patch('random.choice', return_value=(0, 1)):
            move = strategy.suggest_move(game_state)
            
        assert move == (0, 1)
        game_state.get_valid_moves.assert_called_once()
    
    def test_suggest_move_no_valid_moves(self):
        """Test suggest_move with no available moves."""
        strategy = RandomStrategy(PLAYER_X)
        game_state = Mock()
        game_state.get_valid_moves.return_value = []
        
        move = strategy.suggest_move(game_state)
        
        assert move is None


class TestMinimaxStrategy:
    """Test MinimaxStrategy class."""
    
    def test_minimax_strategy_initialization(self):
        """Test MinimaxStrategy initialization."""
        strategy = MinimaxStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O
    
    def test_suggest_move_game_finished(self):
        """Test suggest_move when game is already finished."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = Mock()
        game_state.check_winner.return_value = PLAYER_X
        game_state.board = [[EMPTY] * 3 for _ in range(3)]
        
        move = strategy.suggest_move(game_state)
        
        assert move is None
    
    def test_suggest_move_board_full(self):
        """Test suggest_move when board is full."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = Mock()
        game_state.check_winner.return_value = None
        game_state.is_board_full.return_value = True
        game_state.board = [[EMPTY] * 3 for _ in range(3)]
        
        move = strategy.suggest_move(game_state)
        
        assert move is None
    
    def test_suggest_move_valid_game(self):
        """Test suggest_move with valid game state."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = Mock()
        game_state.check_winner.return_value = None
        game_state.is_board_full.return_value = False
        game_state.board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY], 
            [EMPTY, EMPTY, EMPTY]
        ]
        game_state.board_to_string.return_value = "Board string"
        
        move = strategy.suggest_move(game_state)
        
        assert move is not None
        assert isinstance(move, tuple)
        assert len(move) == 2
    
    def test_get_best_move_empty_board(self):
        """Test _get_best_move with empty board returns center."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [[EMPTY] * 3 for _ in range(3)]
        
        move = strategy._get_best_move(board, PLAYER_X)
        
        assert move == (1, 1)  # Center
    
    def test_get_best_move_almost_empty_board(self):
        """Test _get_best_move with almost empty board."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_O, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        move = strategy._get_best_move(board, PLAYER_X)
        
        assert move == (1, 1)  # Should take center
    
    def test_get_best_move_single_move_left(self):
        """Test _get_best_move with only one move available."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, EMPTY]
        ]
        
        move = strategy._get_best_move(board, PLAYER_X)
        
        assert move == (2, 2)
    
    def test_get_best_move_full_board(self):
        """Test _get_best_move with full board."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        
        move = strategy._get_best_move(board, PLAYER_X)
        
        assert move is None
    
    def test_get_available_moves(self):
        """Test _get_available_moves method."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # MinimaxStrategy doesn't have _get_available_moves method
        # Test suggest_move instead
        from app.core.game_state import GameState
        game_state = GameState()
        game_state._board = board
        move = strategy.suggest_move(game_state)
        assert move is not None
    
    def test_is_board_full_empty_board(self):
        """Test _is_board_full with empty board."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [[EMPTY] * 3 for _ in range(3)]
        
        assert not strategy._is_board_full(board)
    
    def test_is_board_full_full_board(self):
        """Test _is_board_full with full board."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        
        assert strategy._is_board_full(board)
    
    def test_check_winner_row_win(self):
        """Test _check_winner with row win."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        winner = strategy._check_winner(board)
        
        assert winner == PLAYER_X
    
    def test_check_winner_column_win(self):
        """Test _check_winner with column win."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_O, EMPTY, EMPTY],
            [PLAYER_O, EMPTY, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        
        winner = strategy._check_winner(board)
        
        assert winner == PLAYER_O
    
    def test_check_winner_diagonal_win(self):
        """Test _check_winner with diagonal win."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_X]
        ]
        
        winner = strategy._check_winner(board)
        
        assert winner == PLAYER_X
    
    def test_check_winner_anti_diagonal_win(self):
        """Test _check_winner with anti-diagonal win."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [EMPTY, EMPTY, PLAYER_O],
            [EMPTY, PLAYER_O, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        
        winner = strategy._check_winner(board)
        
        assert winner == PLAYER_O
    
    def test_check_winner_tie(self):
        """Test _check_winner with tie game."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        
        winner = strategy._check_winner(board)
        
        assert winner == "TIE"
    
    def test_check_winner_no_winner(self):
        """Test _check_winner with no winner yet."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        winner = strategy._check_winner(board)
        
        assert winner is None
    
    def test_get_other_player(self):
        """Test _get_other_player method."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        assert strategy._get_other_player(PLAYER_X) == PLAYER_O
        assert strategy._get_other_player(PLAYER_O) == PLAYER_X


class TestFixedStrategySelector:
    """Test FixedStrategySelector class."""
    
    def test_fixed_strategy_selector_minimax(self):
        """Test FixedStrategySelector with minimax."""
        selector = FixedStrategySelector('minimax')
        assert selector.strategy_type == 'minimax'
        assert selector.select_strategy() == 'minimax'
    
    def test_fixed_strategy_selector_random(self):
        """Test FixedStrategySelector with random."""
        selector = FixedStrategySelector('random')
        assert selector.strategy_type == 'random'
        assert selector.select_strategy() == 'random'
    
    def test_fixed_strategy_selector_case_insensitive(self):
        """Test FixedStrategySelector is case insensitive."""
        selector = FixedStrategySelector('MINIMAX')
        assert selector.select_strategy() == 'minimax'
    
    def test_fixed_strategy_selector_invalid_type(self):
        """Test FixedStrategySelector with invalid type."""
        with pytest.raises(ValueError, match="Invalid fixed strategy type"):
            FixedStrategySelector('invalid')
    
    def test_fixed_strategy_selector_default(self):
        """Test FixedStrategySelector with default value."""
        selector = FixedStrategySelector()
        assert selector.strategy_type == 'minimax'
        assert selector.select_strategy() == 'minimax'


class TestBernoulliStrategySelector:
    """Test BernoulliStrategySelector class."""
    
    def test_bernoulli_selector_with_probability(self):
        """Test BernoulliStrategySelector with probability."""
        selector = BernoulliStrategySelector(p=0.7)
        assert selector.p == 0.7
    
    def test_bernoulli_selector_with_difficulty(self):
        """Test BernoulliStrategySelector with difficulty."""
        selector = BernoulliStrategySelector(difficulty=8)
        assert selector.p == 0.8
        assert selector.difficulty == 8
    
    def test_bernoulli_selector_probability_clamping(self):
        """Test probability value clamping."""
        selector = BernoulliStrategySelector(p=1.5)
        assert selector.p == 1.0
        
        selector.p = -0.5
        assert selector.p == 0.0
    
    def test_bernoulli_selector_difficulty_clamping(self):
        """Test difficulty value clamping."""
        selector = BernoulliStrategySelector(difficulty=15)
        assert selector.difficulty == 10
        assert selector.p == 1.0
        
        selector.difficulty = -5
        assert selector.difficulty == 0
        assert selector.p == 0.0
    
    def test_bernoulli_selector_property_setters(self):
        """Test property setters."""
        selector = BernoulliStrategySelector()
        
        selector.p = 0.3
        assert selector.p == 0.3
        assert selector.difficulty == 3
        
        selector.difficulty = 7
        assert selector.difficulty == 7
        assert selector.p == 0.7
    
    def test_select_strategy_minimax(self):
        """Test select_strategy returns minimax for low random values."""
        selector = BernoulliStrategySelector(p=0.8)
        
        with patch('random.random', return_value=0.5):
            strategy = selector.select_strategy()
            assert strategy == 'minimax'
    
    def test_select_strategy_random(self):
        """Test select_strategy returns random for high random values."""
        selector = BernoulliStrategySelector(p=0.3)
        
        with patch('random.random', return_value=0.5):
            strategy = selector.select_strategy()
            assert strategy == 'random'
    
    def test_get_move_minimax_strategy(self):
        """Test get_move with minimax strategy."""
        selector = BernoulliStrategySelector(p=1.0)  # Always minimax
        board = [[EMPTY] * 3 for _ in range(3)]
        
        with patch('app.main.game_logic.get_best_move', return_value=(1, 1)) as mock_best:
            move = selector.get_move(board, PLAYER_X)
            
        assert move == (1, 1)
        mock_best.assert_called_once_with(board, PLAYER_X)
    
    def test_get_move_random_strategy(self):
        """Test get_move with random strategy."""
        selector = BernoulliStrategySelector(p=0.0)  # Always random
        board = [[EMPTY] * 3 for _ in range(3)]
        
        with patch('app.main.game_logic.get_random_move', return_value=(0, 0)) as mock_random:
            move = selector.get_move(board, PLAYER_X)
            
        assert move == (0, 0)
        mock_random.assert_called_once_with(board, PLAYER_X)


class TestStrategyFactory:
    """Test strategy factory functions."""
    
    def test_strategy_map_contains_expected_strategies(self):
        """Test STRATEGY_MAP contains expected strategies."""
        assert 'random' in STRATEGY_MAP
        assert 'minimax' in STRATEGY_MAP
        assert STRATEGY_MAP['random'] == RandomStrategy
        assert STRATEGY_MAP['minimax'] == MinimaxStrategy
    
    def test_create_strategy_random(self):
        """Test create_strategy with random strategy."""
        strategy = create_strategy('random', PLAYER_X)
        assert isinstance(strategy, RandomStrategy)
        assert strategy.player == PLAYER_X
    
    def test_create_strategy_minimax(self):
        """Test create_strategy with minimax strategy."""
        strategy = create_strategy('minimax', PLAYER_O)
        assert isinstance(strategy, MinimaxStrategy)
        assert strategy.player == PLAYER_O
    
    def test_create_strategy_case_insensitive(self):
        """Test create_strategy is case insensitive."""
        strategy = create_strategy('RANDOM', PLAYER_X)
        assert isinstance(strategy, RandomStrategy)
        
        strategy = create_strategy('MiniMax', PLAYER_O)
        assert isinstance(strategy, MinimaxStrategy)
    
    def test_create_strategy_invalid_type(self):
        """Test create_strategy with invalid strategy type."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('invalid', PLAYER_X)


class TestMinimaxAlgorithm:
    """Test minimax algorithm implementation."""
    
    def test_minimax_ai_winning_move(self):
        """Test minimax detects winning move for AI."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        score, move = strategy._minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        
        assert move == (0, 2)  # Winning move
        assert score > 0  # Positive score for AI win
    
    def test_minimax_block_opponent_win(self):
        """Test minimax blocks opponent winning move."""
        strategy = MinimaxStrategy(PLAYER_X)
        board = [
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        move = strategy._get_best_move(board, PLAYER_X)
        
        assert move == (0, 2)  # Block opponent win
    
    def test_minimax_terminal_states(self):
        """Test minimax handles terminal states correctly."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # AI wins
        board_ai_wins = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        score, move = strategy._minimax(board_ai_wins, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score > 0
        assert move is None
        
        # Human wins
        board_human_wins = [
            [PLAYER_O, PLAYER_O, PLAYER_O],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        score, move = strategy._minimax(board_human_wins, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score < 0
        assert move is None
        
        # Tie
        board_tie = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        score, move = strategy._minimax(board_tie, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score == 0
        assert move is None