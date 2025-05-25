"""
Pure pytest tests for game state module.
"""
import pytest
import numpy as np
from app.core.game_state import GameState


class TestGameState:
    """Pure pytest test class for GameState."""
    
    @pytest.fixture
    def game_state(self):
        """Create game state instance for testing."""
        return GameState()
    
    def test_game_state_initialization(self, game_state):
        """Test game state initialization."""
        assert game_state is not None
        assert hasattr(game_state, 'board')
        assert hasattr(game_state, 'current_player')
    
    def test_reset_board(self, game_state):
        """Test board reset functionality."""
        game_state.reset_board()
        board = game_state.get_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
    
    def test_make_move_valid(self, game_state):
        """Test making valid moves."""
        game_state.reset_board()
        result = game_state.make_move(0, 0)
        assert result is True
        board = game_state.get_board()
        assert board[0][0] != ' '
    
    def test_make_move_invalid(self, game_state):
        """Test making invalid moves."""
        game_state.reset_board()
        game_state.make_move(0, 0)
        result = game_state.make_move(0, 0)  # Same position
        assert result is False
    
    def test_get_current_player(self, game_state):
        """Test current player tracking."""
        game_state.reset_board()
        player = game_state.get_current_player()
        assert player in ['X', 'O']    
    def test_switch_player(self, game_state):
        """Test player switching."""
        game_state.reset_board()
        initial_player = game_state.get_current_player()
        game_state.make_move(0, 0)
        new_player = game_state.get_current_player()
        assert initial_player != new_player
    
    def test_check_winner(self, game_state):
        """Test winner detection."""
        game_state.reset_board()
        # Simulate winning condition
        board = game_state.get_board()
        for i in range(3):
            board[0][i] = 'X'
        winner = game_state.check_winner()
        assert winner is not None
    
    def test_is_board_full(self, game_state):
        """Test board full detection."""
        game_state.reset_board()
        is_full = game_state.is_board_full()
        assert is_full is False
        
        # Fill board
        board = game_state.get_board()
        for i in range(3):
            for j in range(3):
                board[i][j] = 'X' if (i + j) % 2 == 0 else 'O'
        is_full = game_state.is_board_full()
        assert is_full is True