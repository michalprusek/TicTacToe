# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Fixed tests for strategy module.
Simple tests that don't require complex game state setup.
"""
import pytest
from app.core.strategy import BernoulliStrategySelector, MinimaxStrategy
from app.main.game_logic import PLAYER_X, PLAYER_O
from app.core.game_state import EMPTY


class TestBernoulliStrategySelector:
    """Test BernoulliStrategySelector."""

    def test_init(self):
        """Test initialization."""
        selector = BernoulliStrategySelector(difficulty=5)
        assert selector._p == 0.5  # difficulty 5 -> p = 5/10 = 0.5

    def test_difficulty_bounds(self):
        """Test difficulty bounds."""
        # Test minimum difficulty
        selector = BernoulliStrategySelector(difficulty=1)
        assert selector._p == 0.1  # difficulty 1 -> p = 1/10 = 0.1

        # Test maximum difficulty  
        selector = BernoulliStrategySelector(difficulty=10)
        assert selector._p == 1.0  # difficulty 10 -> p = 10/10 = 1.0

    def test_probability_assignment(self):
        """Test probability assignment."""
        selector = BernoulliStrategySelector(p=0.7)
        assert selector._p == 0.7


class TestMinimaxStrategy:
    """Test MinimaxStrategy with basic functionality."""

    def test_init(self):
        """Test initialization."""
        strategy = MinimaxStrategy(player=PLAYER_X)
        assert strategy.player == PLAYER_X

        strategy = MinimaxStrategy(player=PLAYER_O)
        assert strategy.player == PLAYER_O

    def test_get_other_player(self):
        """Test _get_other_player method."""
        strategy = MinimaxStrategy(player=PLAYER_X)
        
        # Test getting opposite player
        assert strategy._get_other_player(PLAYER_X) == PLAYER_O
        assert strategy._get_other_player(PLAYER_O) == PLAYER_X

    def test_get_available_moves_empty_board(self):
        """Test get_available_moves on empty board."""
        from app.core.minimax_algorithm import get_available_moves
        
        # Create empty 3x3 board
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        
        moves = get_available_moves(board)
        assert len(moves) == 9  # All positions should be available
        assert (0, 0) in moves
        assert (1, 1) in moves  
        assert (2, 2) in moves

    def test_get_available_moves_partial_board(self):
        """Test get_available_moves on partially filled board."""
        from app.core.minimax_algorithm import get_available_moves
        
        # Create board with some moves
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        moves = get_available_moves(board)
        assert len(moves) == 7  # 7 empty positions
        assert (0, 0) not in moves  # Position (0,0) is taken by X
        assert (1, 1) not in moves  # Position (1,1) is taken by O
        assert (0, 1) in moves     # Position (0,1) is empty    def test_check_winner_simple(self):
        """Test _check_winner with simple scenarios."""
        strategy = MinimaxStrategy(player=PLAYER_X)

        # Test horizontal win
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert strategy._check_winner(board) == PLAYER_X

        # Test vertical win
        board = [
            [PLAYER_O, EMPTY, EMPTY],
            [PLAYER_O, EMPTY, EMPTY], 
            [PLAYER_O, EMPTY, EMPTY]
        ]
        assert strategy._check_winner(board) == PLAYER_O

        # Test diagonal win
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_X]
        ]
        assert strategy._check_winner(board) == PLAYER_X

        # Test no winner
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert strategy._check_winner(board) is None  # No winner returns None
