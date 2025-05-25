"""
Pure pytest tests for strategy module.
"""
import pytest
from app.core.strategy import Strategy, RandomStrategy
from app.core.game_state import PLAYER_X, PLAYER_O, EMPTY


class TestStrategy:
    """Pure pytest test class for Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance for testing."""
        return RandomStrategy(PLAYER_O)
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy is not None
        assert strategy.player == PLAYER_O
        assert strategy.opponent == PLAYER_X
    
    def test_suggest_move_empty_board(self, strategy):
        """Test suggest move on empty board."""
        from app.core.game_state import GameState
        game_state = GameState()
        game_state.reset_board()
        move = strategy.suggest_move(game_state)
        assert move is not None
        assert len(move) == 2
        assert 0 <= move[0] <= 2
        assert 0 <= move[1] <= 2
    
    def test_random_strategy_move(self, strategy):
        """Test random strategy suggests valid moves."""
        from app.core.game_state import GameState
        game_state = GameState()
        game_state.reset_board()
        
        # Make some moves
        game_state.make_move(0, 0)
        game_state.make_move(1, 1)
        
        move = strategy.suggest_move(game_state)
        assert move is not None
        assert len(move) == 2
        assert 0 <= move[0] <= 2
        assert 0 <= move[1] <= 2
        
        # Ensure move is on empty cell
        board = game_state.get_board()
        assert board[move[0]][move[1]] == EMPTY    
    @pytest.mark.parametrize("player", [PLAYER_X, PLAYER_O])
    def test_different_players(self, player):
        """Test strategy with different players."""
        strategy = RandomStrategy(player)
        assert strategy.player == player
        
        from app.core.game_state import GameState
        game_state = GameState()
        game_state.reset_board()
        move = strategy.suggest_move(game_state)
        assert move is not None
        assert len(move) == 2
    
    def test_strategy_full_board(self, strategy):
        """Test strategy behavior with full board."""
        from app.core.game_state import GameState
        game_state = GameState()
        game_state.reset_board()
        
        # Fill the board
        board = game_state.get_board()
        for i in range(3):
            for j in range(3):
                board[i][j] = PLAYER_X if (i + j) % 2 == 0 else PLAYER_O
        
        move = strategy.suggest_move(game_state)
        assert move is None