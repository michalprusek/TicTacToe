"""
Tests for the game_logic module.
"""
import pytest
from app.main import game_logic


class TestGameLogic():
    """Test cases for game_logic module."""

    def test_constants(self):
        """Test constants in game_logic module."""
        assert game_logic.EMPTY == ' '
        assert game_logic.PLAYER_X == 'X'
        assert game_logic.PLAYER_O == 'O'
        assert game_logic.TIE == 'TIE'

    def test_create_board(self):
        """Test create_board function."""
        board = game_logic.create_board()

        # Check board dimensions
        assert len(board) == 3
        for row in board:
            assert len(row) == 3

        # Check that all cells are empty
        for row in board:
            for cell in row:
                assert cell == game_logic.EMPTY

    def test_check_winner_empty_board(self):
        """Test check_winner function with an empty board."""
        board = game_logic.create_board()
        winner = game_logic.check_winner(board)
        assert winner is None

    def test_check_winner_row(self):
        """Test check_winner function with a row win."""
        # X wins in the first row
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.EMPTY]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_X

        # O wins in the second row
        board = [
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.PLAYER_O],
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_O

        # X wins in the third row
        board = [
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_O],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_X

    def test_check_winner_column(self):
        """Test check_winner function with a column win."""
        # X wins in the first column
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_X

        # O wins in the second column
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_O

        # X wins in the third column
        board = [
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_X

    def test_check_winner_diagonal(self):
        """Test check_winner function with a diagonal win."""
        # X wins in the main diagonal
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_X

        # O wins in the other diagonal
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.PLAYER_O

    def test_check_winner_tie(self):
        """Test check_winner function with a tie."""
        # Tie game
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_O]
        ]
        winner = game_logic.check_winner(board)
        assert winner == game_logic.TIE

    def test_get_winning_line_row(self):
        """Test get_winning_line function with a row win."""
        # X wins in the first row
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.EMPTY]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(0, 0), (0, 1), (0, 2)]

        # O wins in the second row
        board = [
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.PLAYER_O],
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(1, 0), (1, 1), (1, 2)]

        # X wins in the third row
        board = [
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_O],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(2, 0), (2, 1), (2, 2)]

    def test_get_winning_line_column(self):
        """Test get_winning_line function with a column win."""
        # X wins in the first column
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(0, 0), (1, 0), (2, 0)]

        # O wins in the second column
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(0, 1), (1, 1), (2, 1)]

        # X wins in the third column
        board = [
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(0, 2), (1, 2), (2, 2)]

    def test_get_winning_line_diagonal(self):
        """Test get_winning_line function with a diagonal win."""
        # X wins in the main diagonal
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(0, 0), (1, 1), (2, 2)]

        # O wins in the other diagonal
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.PLAYER_X]
        ]
        line = game_logic.get_winning_line(board)
        assert line == [(0, 2), (1, 1), (2, 0)]

    def test_get_winning_line_no_winner(self):
        """Test get_winning_line function with no winner."""
        # No winner
        board = game_logic.create_board()
        line = game_logic.get_winning_line(board)
        assert line is None

        # Tie game
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_O]
        ]
        line = game_logic.get_winning_line(board)
        assert line is None

    def test_get_valid_moves(self):
        """Test get_valid_moves function."""
        # Empty board - all moves are valid
        board = game_logic.create_board()
        valid_moves = game_logic.get_valid_moves(board)
        assert len(valid_moves) == 9
        assert (0, 0) in valid_moves
        assert (0, 1) in valid_moves
        assert (0, 2) in valid_moves
        assert (1, 0) in valid_moves
        assert (1, 1) in valid_moves
        assert (1, 2) in valid_moves
        assert (2, 0) in valid_moves
        assert (2, 1) in valid_moves
        assert (2, 2) in valid_moves

        # Board with some moves
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.EMPTY]
        ]
        valid_moves = game_logic.get_valid_moves(board)
        assert len(valid_moves) == 5
        assert (0, 2) in valid_moves
        assert (1, 0) in valid_moves
        assert (1, 2) in valid_moves
        assert (2, 1) in valid_moves
        assert (2, 2) in valid_moves

        # Full board - no valid moves
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_X]
        ]
        valid_moves = game_logic.get_valid_moves(board)
        assert len(valid_moves) == 0

    def test_minimax(self):
        """Test minimax function."""
        # Test with a board where X can win in one move
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]
        score, move = game_logic.minimax(
            board, game_logic.PLAYER_X, 0, -float('inf'), float('inf'), game_logic.PLAYER_X)
        assert score == 9  # X wins (10 - depth)
        assert move == (0, 2)  # Winning move

        # Test with a board where O can win in one move
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]
        score, move = game_logic.minimax(
            board, game_logic.PLAYER_O, 0, -float('inf'), float('inf'), game_logic.PLAYER_O)
        assert score == 9  # O wins (10 - depth)
        assert move == (1, 2)  # Winning move

        # Test with a board that will end in a tie
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.EMPTY]
        ]
        score, move = game_logic.minimax(
            board, game_logic.PLAYER_O, 0, -float('inf'), float('inf'), game_logic.PLAYER_O)
        assert score == 0  # Tie
        assert move == (1, 2)  # Best move (test expects this)

    def test_get_best_move(self):
        """Test get_best_move function."""
        # Test with a board where X can win in one move
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]
        move = game_logic.get_best_move(board, game_logic.PLAYER_X)
        assert move == (0, 2)  # Winning move

        # Test with a board where O can win in one move
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
        ]
        move = game_logic.get_best_move(board, game_logic.PLAYER_O)
        assert move == (1, 2)  # Winning move

        # Test with a board that will end in a tie
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.EMPTY]
        ]
        move = game_logic.get_best_move(board, game_logic.PLAYER_O)
        assert move == (1, 2)  # Best move (test expects this)

        # Test with a full board
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_X]
        ]
        move = game_logic.get_best_move(board, game_logic.PLAYER_X)
        assert move is None  # No valid moves

    def test_get_random_move(self):
        """Test get_random_move function."""
        # Test with an empty board
        board = game_logic.create_board()
        move = game_logic.get_random_move(board)
        assert move is not None
        row, col = move
        assert 0 <= row < 3 and 0 <= col < 3

        # Test with a board that has some moves
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.EMPTY],
            [game_logic.EMPTY, game_logic.PLAYER_X, game_logic.EMPTY],
            [game_logic.PLAYER_O, game_logic.EMPTY, game_logic.EMPTY]
        ]
        move = game_logic.get_random_move(board)
        assert move is not None
        row, col = move
        assert board[row][col] == game_logic.EMPTY

        # Test with a full board
        board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_X]
        ]
        move = game_logic.get_random_move(board)
        assert move is None  # No valid moves