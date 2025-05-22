"""Integration tests for game components."""
import pytest
from unittest.mock import MagicMock, patch

from app.main import game_logic
from app.core.strategy import BernoulliStrategySelector


@pytest.fixture
def strategy_selector():
    """strategy_selector fixture for tests."""
    strategy_selector = BernoulliStrategySelector(p=1.0)
    return strategy_selector


@pytest.fixture
def board():
    """board fixture for tests."""
    board = game_logic.create_board()
    return board



class TestGameIntegration():
    """Integration tests for game components."""

    # Convert setUp method to pytest fixture
    @pytest.fixture(autouse=True)
    def setup(self, board, strategy_selector):
        """Set up test fixtures."""
        self.board = board
        self.strategy_selector = strategy_selector
        return None

    def test_strategy_with_game_logic(self):
        """Test integration between strategy selector and game logic."""
        # Make some moves on the board
        self.board[0][0] = game_logic.PLAYER_X
        self.board[1][1] = game_logic.PLAYER_O

        # Get AI move using strategy selector
        move = self.strategy_selector.get_move(self.board, game_logic.PLAYER_X)

        # Verify move is valid
        assert move is not None
        row, col = move
        assert 0 <= row < 3 and 0 <= col < 3
        assert self.board[row][col] == game_logic.EMPTY

        # Apply move to board
        self.board[row][col] = game_logic.PLAYER_X

        # Check if game state is valid
        winner = game_logic.check_winner(self.board)
        assert winner in [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.TIE, None]

    def test_winning_sequence(self):
        """Test a complete winning sequence."""
        # Create a winning sequence for X
        # X in diagonal
        self.board[0][0] = game_logic.PLAYER_X
        self.board[1][1] = game_logic.PLAYER_X
        self.board[2][2] = game_logic.PLAYER_X

        # O in some other positions
        self.board[0][1] = game_logic.PLAYER_O
        self.board[1][0] = game_logic.PLAYER_O

        # Check winner
        winner = game_logic.check_winner(self.board)
        assert winner == game_logic.PLAYER_X

        # Check winning line
        winning_line = game_logic.get_winning_line(self.board)
        assert winning_line == [(0, 0), (1, 1), (2, 2)]

        # Verify game is over
        assert game_logic.is_game_over(self.board)

    @patch('app.main.game_logic.get_best_move')
    def test_strategy_selection(self, mock_best_move):
        """Test strategy selection based on difficulty."""
        # Set up mock
        mock_best_move.return_value = (0, 0)

        # Test with p=1.0 (always minimax)
        selector = BernoulliStrategySelector(p=1.0)
        move = selector.get_move(self.board, game_logic.PLAYER_X)
        assert move == (0, 0)
        mock_best_move.assert_called_once()

        # Reset mock
        mock_best_move.reset_mock()

        # Test with p=0.0 (always random)
        selector = BernoulliStrategySelector(p=0.0)
        with patch('app.main.game_logic.get_random_move', return_value=(1, 1)):
            move = selector.get_move(self.board, game_logic.PLAYER_X)
            assert move == (1, 1)
            mock_best_move.assert_not_called()



