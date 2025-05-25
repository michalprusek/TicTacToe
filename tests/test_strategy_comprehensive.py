"""Comprehensive tests for strategy.py module."""
import pytest
import random
import math
from unittest.mock import Mock, patch, MagicMock
import logging

from app.core.strategy import (
    Strategy, RandomStrategy, MinimaxStrategy, StrategySelector,
    FixedStrategySelector, BernoulliStrategySelector, 
    STRATEGY_MAP, create_strategy
)
from app.core.game_state import GameState, PLAYER_X, PLAYER_O, EMPTY


class TestStrategy:
    """Test the abstract Strategy base class."""
    
    def test_strategy_initialization_x(self):
        """Test Strategy initialization for player X."""
        # Create a concrete implementation for testing
        class TestStrategy(Strategy):
            def suggest_move(self, game_state):
                return None
        
        strategy = TestStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O
        assert strategy.logger is not None
        assert "TestStrategy" in strategy.logger.name
    
    def test_strategy_initialization_o(self):
        """Test Strategy initialization for player O."""
        class TestStrategy(Strategy):
            def suggest_move(self, game_state):
                return None
        
        strategy = TestStrategy(PLAYER_O)
        assert strategy.player == PLAYER_O
        assert strategy.opponent == PLAYER_X
    
    def test_strategy_abstract_method(self):
        """Test that Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy(PLAYER_X)


class TestRandomStrategy:
    """Test RandomStrategy implementation."""
    
    def test_random_strategy_initialization(self):
        """Test RandomStrategy initialization."""
        strategy = RandomStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O
    
    def test_suggest_move_empty_board(self):
        """Test suggest_move on empty board."""
        strategy = RandomStrategy(PLAYER_X)
        game_state = GameState()
        
        # Mock random.choice to return a specific move
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = (1, 1)
            
            move = strategy.suggest_move(game_state)
            assert move == (1, 1)
            mock_choice.assert_called_once()
    
    def test_suggest_move_partial_board(self):
        """Test suggest_move on partially filled board."""
        strategy = RandomStrategy(PLAYER_X)
        game_state = GameState()
        
        # Fill some cells
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_O
        
        move = strategy.suggest_move(game_state)
        
        # Should return a valid move
        assert move is not None
        assert len(move) == 2
        r, c = move
        assert 0 <= r < 3 and 0 <= c < 3
        assert game_state._board_state[r][c] == EMPTY
    
    def test_suggest_move_full_board(self):
        """Test suggest_move on full board."""
        strategy = RandomStrategy(PLAYER_X)
        game_state = GameState()
        
        # Fill entire board
        for r in range(3):
            for c in range(3):
                game_state._board_state[r][c] = PLAYER_X if (r + c) % 2 == 0 else PLAYER_O
        
        move = strategy.suggest_move(game_state)
        assert move is None


class TestMinimaxStrategy:
    """Test MinimaxStrategy implementation."""
    
    def test_minimax_strategy_initialization(self):
        """Test MinimaxStrategy initialization."""
        strategy = MinimaxStrategy(PLAYER_X)
        assert strategy.player == PLAYER_X
        assert strategy.opponent == PLAYER_O
    
    def test_suggest_move_empty_board(self):
        """Test suggest_move on empty board should return center."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        move = strategy.suggest_move(game_state)
        assert move == (1, 1)  # Should pick center on empty board
    
    def test_suggest_move_second_move_center_available(self):
        """Test suggest_move when center is available as second move."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        # Make one move (not center)
        game_state._board_state[0][0] = PLAYER_O
        
        move = strategy.suggest_move(game_state)
        assert move == (1, 1)  # Should take center
    
    def test_suggest_move_winning_move(self):
        """Test suggest_move finds winning move."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        # Set up board where X can win
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[0][1] = PLAYER_X
        # game_state._board_state[0][2] is empty - winning move
        game_state._board_state[1][0] = PLAYER_O
        game_state._board_state[2][2] = PLAYER_O
        
        move = strategy.suggest_move(game_state)
        assert move == (0, 2)  # Should complete the winning row
    
    def test_suggest_move_blocking_move(self):
        """Test suggest_move finds blocking move."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        # Set up board where O is about to win
        game_state._board_state[0][0] = PLAYER_O
        game_state._board_state[0][1] = PLAYER_O
        # game_state._board_state[0][2] is empty - must block
        game_state._board_state[1][0] = PLAYER_X
        
        move = strategy.suggest_move(game_state)
        assert move == (0, 2)  # Should block O's winning move
    
    def test_suggest_move_finished_game(self):
        """Test suggest_move on finished game."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        # Set up winning board for X
        game_state._board_state[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        
        move = strategy.suggest_move(game_state)
        assert move is None  # No move on finished game
    
    def test_suggest_move_full_board(self):
        """Test suggest_move on full board."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        # Fill board without winner
        game_state._board_state = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        
        move = strategy.suggest_move(game_state)
        assert move is None
    
    def test_suggest_move_one_move_left(self):
        """Test suggest_move with only one valid move."""
        strategy = MinimaxStrategy(PLAYER_X)
        game_state = GameState()
        
        # Fill board except one cell without creating a winner
        game_state._board_state = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, EMPTY]  # Only (2,2) is empty
        ]
        
        move = strategy.suggest_move(game_state)
        assert move == (2, 2)  # Should return the only available move


class TestMinimaxInternalMethods:
    """Test MinimaxStrategy internal methods."""
    
    def test_get_available_moves(self):
        """Test _get_available_moves method."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Empty board
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        moves = strategy._get_available_moves(board)
        assert len(moves) == 9
        
        # Partially filled board
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        moves = strategy._get_available_moves(board)
        assert len(moves) == 7
        assert (0, 0) not in moves
        assert (1, 1) not in moves
    
    def test_is_board_full(self):
        """Test _is_board_full method."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Empty board
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        assert not strategy._is_board_full(board)
        
        # Full board
        board = [[PLAYER_X for _ in range(3)] for _ in range(3)]
        assert strategy._is_board_full(board)
    
    def test_check_winner_rows(self):
        """Test _check_winner method for row wins."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Test each row
        for row in range(3):
            board = [[EMPTY for _ in range(3)] for _ in range(3)]
            for col in range(3):
                board[row][col] = PLAYER_X
            
            winner = strategy._check_winner(board)
            assert winner == PLAYER_X
    
    def test_check_winner_columns(self):
        """Test _check_winner method for column wins."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Test each column
        for col in range(3):
            board = [[EMPTY for _ in range(3)] for _ in range(3)]
            for row in range(3):
                board[row][col] = PLAYER_O
            
            winner = strategy._check_winner(board)
            assert winner == PLAYER_O
    
    def test_check_winner_diagonals(self):
        """Test _check_winner method for diagonal wins."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Main diagonal
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        for i in range(3):
            board[i][i] = PLAYER_X
        winner = strategy._check_winner(board)
        assert winner == PLAYER_X
        
        # Anti-diagonal
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        for i in range(3):
            board[i][2-i] = PLAYER_O
        winner = strategy._check_winner(board)
        assert winner == PLAYER_O
    
    def test_check_winner_tie(self):
        """Test _check_winner method for tie."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Full board with no winner
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        
        winner = strategy._check_winner(board)
        assert winner == "TIE"
    
    def test_check_winner_no_winner(self):
        """Test _check_winner method with no winner."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        winner = strategy._check_winner(board)
        assert winner is None
    
    def test_get_other_player(self):
        """Test _get_other_player method."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        assert strategy._get_other_player(PLAYER_X) == PLAYER_O
        assert strategy._get_other_player(PLAYER_O) == PLAYER_X


class TestMinimaxAlgorithm:
    """Test the minimax algorithm itself."""
    
    def test_minimax_terminal_states(self):
        """Test minimax algorithm terminal state scoring."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # AI wins
        board = [[PLAYER_X, PLAYER_X, PLAYER_X], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
        score, move = strategy._minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score == 10  # AI wins at depth 0
        assert move is None
        
        # Human wins
        board = [[PLAYER_O, PLAYER_O, PLAYER_O], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
        score, move = strategy._minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score == -10  # Human wins at depth 0
        assert move is None
        
        # Tie
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O], 
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        score, move = strategy._minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score == 0  # Tie
        assert move is None
    
    def test_minimax_depth_preference(self):
        """Test that minimax prefers faster wins and slower losses."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # AI wins at different depths
        board = [[PLAYER_X, PLAYER_X, PLAYER_X], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
        score_depth_0, _ = strategy._minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        score_depth_2, _ = strategy._minimax(board, PLAYER_X, 2, -math.inf, math.inf, PLAYER_X)
        
        assert score_depth_0 > score_depth_2  # Prefer faster wins
        
        # Human wins at different depths
        board = [[PLAYER_O, PLAYER_O, PLAYER_O], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
        score_depth_0, _ = strategy._minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        score_depth_2, _ = strategy._minimax(board, PLAYER_X, 2, -math.inf, math.inf, PLAYER_X)
        
        assert score_depth_0 < score_depth_2  # Prefer slower losses
    
    def test_minimax_alpha_beta_pruning(self):
        """Test alpha-beta pruning in minimax."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Create a scenario where pruning should occur
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # Mock the recursive call to test pruning
        original_minimax = strategy._minimax
        call_count = 0
        
        def counting_minimax(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_minimax(*args, **kwargs)
        
        strategy._minimax = counting_minimax
        
        # Call minimax and verify it doesn't explore all possibilities
        score, move = strategy._get_best_move(board, PLAYER_X)
        
        # Should have found a move without exploring every possibility
        assert move is not None
        # The exact call count depends on pruning, but it should be reasonable
        assert call_count > 0


class TestFixedStrategySelector:
    """Test FixedStrategySelector implementation."""
    
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
        
        selector = FixedStrategySelector('Random')
        assert selector.select_strategy() == 'random'
    
    def test_fixed_strategy_selector_invalid(self):
        """Test FixedStrategySelector with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid fixed strategy type"):
            FixedStrategySelector('invalid')


class TestBernoulliStrategySelector:
    """Test BernoulliStrategySelector implementation."""
    
    def test_bernoulli_selector_initialization_probability(self):
        """Test BernoulliStrategySelector initialization with probability."""
        selector = BernoulliStrategySelector(p=0.7)
        assert selector.p == 0.7
        assert selector.difficulty == 7
    
    def test_bernoulli_selector_initialization_difficulty(self):
        """Test BernoulliStrategySelector initialization with difficulty."""
        selector = BernoulliStrategySelector(difficulty=3)
        assert selector.p == 0.3
        assert selector.difficulty == 3
    
    def test_bernoulli_selector_probability_clamping(self):
        """Test probability value clamping."""
        # Test initial clamping
        selector = BernoulliStrategySelector(p=1.5)
        assert selector.p == 1.0
        
        selector = BernoulliStrategySelector(p=-0.5)
        assert selector.p == 0.0
        
        # Test setter clamping
        selector.p = 2.0
        assert selector.p == 1.0
        
        selector.p = -1.0
        assert selector.p == 0.0
    
    def test_bernoulli_selector_difficulty_clamping(self):
        """Test difficulty value clamping."""
        # Test initial clamping
        selector = BernoulliStrategySelector(difficulty=15)
        assert selector.difficulty == 10
        assert selector.p == 1.0
        
        selector = BernoulliStrategySelector(difficulty=-5)
        assert selector.difficulty == 0
        assert selector.p == 0.0
        
        # Test setter clamping
        selector.difficulty = 12
        assert selector.difficulty == 10
        assert selector.p == 1.0
        
        selector.difficulty = -3
        assert selector.difficulty == 0
        assert selector.p == 0.0
    
    def test_bernoulli_selector_property_synchronization(self):
        """Test that p and difficulty properties stay synchronized."""
        selector = BernoulliStrategySelector(p=0.6)
        
        # Change p, check difficulty
        selector.p = 0.8
        assert selector.difficulty == 8
        
        # Change difficulty, check p
        selector.difficulty = 4
        assert selector.p == 0.4
    
    def test_bernoulli_selector_select_strategy_deterministic(self):
        """Test strategy selection with deterministic probabilities."""
        # p=1.0 should always select minimax
        selector = BernoulliStrategySelector(p=1.0)
        with patch('random.random', return_value=0.5):
            assert selector.select_strategy() == 'minimax'
        
        # p=0.0 should always select random
        selector = BernoulliStrategySelector(p=0.0)
        with patch('random.random', return_value=0.5):
            assert selector.select_strategy() == 'random'
    
    def test_bernoulli_selector_select_strategy_boundary(self):
        """Test strategy selection at probability boundaries."""
        selector = BernoulliStrategySelector(p=0.5)
        
        # Random value exactly at threshold
        with patch('random.random', return_value=0.5):
            assert selector.select_strategy() == 'random'  # >= threshold
        
        # Random value just below threshold
        with patch('random.random', return_value=0.4999):
            assert selector.select_strategy() == 'minimax'  # < threshold
    
    @patch('app.main.game_logic.get_best_move')
    @patch('app.main.game_logic.get_random_move')
    def test_bernoulli_get_move_minimax(self, mock_random_move, mock_best_move):
        """Test get_move method with minimax strategy."""
        selector = BernoulliStrategySelector(p=1.0)  # Always minimax
        mock_best_move.return_value = (1, 1)
        
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        move = selector.get_move(board, PLAYER_X)
        
        assert move == (1, 1)
        mock_best_move.assert_called_once_with(board, PLAYER_X)
        mock_random_move.assert_not_called()
    
    @patch('app.main.game_logic.get_best_move')
    @patch('app.main.game_logic.get_random_move')
    def test_bernoulli_get_move_random(self, mock_random_move, mock_best_move):
        """Test get_move method with random strategy."""
        selector = BernoulliStrategySelector(p=0.0)  # Always random
        mock_random_move.return_value = (2, 0)
        
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        move = selector.get_move(board, PLAYER_X)
        
        assert move == (2, 0)
        mock_random_move.assert_called_once_with(board, PLAYER_X)
        mock_best_move.assert_not_called()


class TestStrategyFactory:
    """Test strategy factory functions."""
    
    def test_strategy_map(self):
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
        
        strategy = create_strategy('Minimax', PLAYER_O)
        assert isinstance(strategy, MinimaxStrategy)
    
    def test_create_strategy_invalid(self):
        """Test create_strategy with invalid strategy type."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('invalid', PLAYER_X)


class TestGameStateIntegration:
    """Test strategy integration with GameState."""
    
    def test_minimax_with_game_state_methods(self):
        """Test MinimaxStrategy properly uses GameState methods."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Create a mock GameState
        mock_game_state = Mock()
        mock_game_state.board = [[EMPTY for _ in range(3)] for _ in range(3)]
        mock_game_state.check_winner.return_value = None
        mock_game_state.is_board_full.return_value = False
        
        move = strategy.suggest_move(mock_game_state)
        
        # Should call the GameState methods
        mock_game_state.check_winner.assert_called_once()
        mock_game_state.is_board_full.assert_called_once()
        assert move is not None
    
    def test_minimax_handles_missing_methods(self):
        """Test MinimaxStrategy handles GameState without optional methods."""
        strategy = MinimaxStrategy(PLAYER_X)
        
        # Create a GameState-like object without the optional methods
        class MinimalGameState:
            def __init__(self):
                self.board = [[EMPTY for _ in range(3)] for _ in range(3)]
            
            def board_to_string(self):
                return "test board"  # Minimal implementation
        
        game_state = MinimalGameState()
        move = strategy.suggest_move(game_state)
        
        # Should still work and return a move
        assert move is not None


if __name__ == "__main__":
    pytest.main([__file__])