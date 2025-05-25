"""
Corrected comprehensive tests for game_logic module using pytest.
Tests all game logic functions based on actual implementation.
"""

import pytest
from app.main.game_logic import (
    create_board, get_available_moves, get_valid_moves, check_winner,
    get_winning_line, is_board_full, is_game_over, print_board,
    get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff, EMPTY, PLAYER_X, PLAYER_O, TIE
)


class TestBoardCreation:
    """Test board creation and basic utilities."""

    def test_create_board(self):
        """Test creating an empty board."""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)

    def test_board_constants(self):
        """Test game constants."""
        assert EMPTY == ' '
        assert PLAYER_X == 'X'
        assert PLAYER_O == 'O'
        assert TIE == "TIE"

    def test_print_board(self):
        """Test board printing function."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        
        # Should not raise exception
        print_board(board)

    def test_board_to_string(self):
        """Test converting board to string."""
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_O]
        ]
        result = board_to_string(board)
        assert isinstance(result, str)
        assert PLAYER_X in result
        assert PLAYER_O in result


class TestMoveValidation:
    """Test move validation and available moves."""

    def test_get_available_moves_empty_board(self):
        """Test getting available moves on empty board."""
        board = create_board()
        moves = get_available_moves(board)
        assert len(moves) == 9
        expected_moves = [(r, c) for r in range(3) for c in range(3)]
        assert set(moves) == set(expected_moves)

    def test_get_available_moves_partial_board(self):
        """Test getting available moves on partially filled board."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        
        moves = get_available_moves(board)
        assert len(moves) == 7
        assert (0, 0) not in moves
        assert (1, 1) not in moves

    def test_get_available_moves_full_board(self):
        """Test getting available moves on full board."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        moves = get_available_moves(board)
        assert len(moves) == 0

    def test_get_valid_moves_alias(self):
        """Test that get_valid_moves is alias for get_available_moves."""
        board = create_board()
        board[0][0] = PLAYER_X
        
        available = get_available_moves(board)
        valid = get_valid_moves(board)
        assert available == valid

    def test_get_random_move(self):
        """Test getting random move."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        
        move = get_random_move(board)
        assert move in get_available_moves(board)

    def test_get_random_move_full_board(self):
        """Test getting random move on full board."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        move = get_random_move(board)
        assert move is None


class TestPlayerUtils:
    """Test player utility functions."""

    def test_get_other_player_x(self):
        """Test getting other player when current is X."""
        other = get_other_player(PLAYER_X)
        assert other == PLAYER_O

    def test_get_other_player_o(self):
        """Test getting other player when current is O."""
        other = get_other_player(PLAYER_O)
        assert other == PLAYER_X


class TestWinnerDetection:
    """Test winner detection logic."""

    def test_check_winner_no_winner(self):
        """Test no winner scenario."""
        board = create_board()
        winner = check_winner(board)
        assert winner is None

    def test_check_winner_x_wins_row(self):
        """Test X wins with a row."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        winner = check_winner(board)
        assert winner == PLAYER_X

    def test_check_winner_o_wins_column(self):
        """Test O wins with a column."""
        board = [
            [PLAYER_O, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        winner = check_winner(board)
        assert winner == PLAYER_O

    def test_check_winner_x_wins_diagonal(self):
        """Test X wins with main diagonal."""
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_X]
        ]
        winner = check_winner(board)
        assert winner == PLAYER_X

    def test_check_winner_o_wins_anti_diagonal(self):
        """Test O wins with anti-diagonal."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        winner = check_winner(board)
        assert winner == PLAYER_O

    def test_check_winner_tie(self):
        """Test tie scenario."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        winner = check_winner(board)
        assert winner == TIE

    def test_get_winning_line_row(self):
        """Test getting winning line for row."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        line = get_winning_line(board)
        assert line == [(0, 0), (0, 1), (0, 2)]

    def test_get_winning_line_no_winner(self):
        """Test getting winning line when no winner."""
        board = create_board()
        line = get_winning_line(board)
        assert line is None


class TestBoardState:
    """Test board state checking functions."""

    def test_is_board_full_empty(self):
        """Test is_board_full on empty board."""
        board = create_board()
        assert is_board_full(board) is False

    def test_is_board_full_partial(self):
        """Test is_board_full on partially filled board."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        assert is_board_full(board) is False

    def test_is_board_full_complete(self):
        """Test is_board_full on completely filled board."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        assert is_board_full(board) is True

    def test_is_game_over_winner(self):
        """Test is_game_over when there's a winner."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert is_game_over(board) is True

    def test_is_game_over_full_board(self):
        """Test is_game_over when board is full."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        assert is_game_over(board) is True

    def test_is_game_over_ongoing(self):
        """Test is_game_over when game is ongoing."""
        board = create_board()
        board[0][0] = PLAYER_X
        assert is_game_over(board) is False


class TestBoardDifference:
    """Test board difference detection."""

    def test_get_board_diff_no_change(self):
        """Test board diff when no change."""
        board1 = create_board()
        board1[0][0] = PLAYER_X
        
        board2 = create_board()
        board2[0][0] = PLAYER_X
        
        diff = get_board_diff(board1, board2)
        assert diff == []

    def test_get_board_diff_single_change(self):
        """Test board diff with single change."""
        board1 = create_board()
        
        board2 = create_board()
        board2[1][1] = PLAYER_O
        
        diff = get_board_diff(board1, board2)
        assert len(diff) == 1
        assert diff[0] == (1, 1, PLAYER_O)

    def test_get_board_diff_multiple_changes(self):
        """Test board diff with multiple changes."""
        board1 = create_board()
        board1[0][0] = PLAYER_X
        
        board2 = create_board()
        board2[0][0] = PLAYER_X
        board2[1][1] = PLAYER_O
        board2[2][2] = PLAYER_X
        
        diff = get_board_diff(board1, board2)
        assert len(diff) == 2
        assert (1, 1, PLAYER_O) in diff
        assert (2, 2, PLAYER_X) in diff


class TestMinimaxAI:
    """Test minimax AI algorithm."""

    def test_minimax_terminal_x_wins(self):
        """Test minimax on terminal position where X wins."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        # Test with player parameter format
        score = minimax(board, PLAYER_X)
        assert isinstance(score, (int, float))

    def test_get_best_move_center(self):
        """Test getting best move."""
        board = create_board()
        move = get_best_move(board, PLAYER_X)
        
        # Should return valid coordinates
        assert isinstance(move, tuple)
        assert len(move) == 2
        row, col = move
        assert 0 <= row <= 2
        assert 0 <= col <= 2
        assert board[row][col] == EMPTY

    def test_get_best_move_blocking(self):
        """Test that AI blocks winning move."""
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],  # AI should block at (0,2)
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        move = get_best_move(board, PLAYER_O)
        # Should block the winning move
        assert move == (0, 2)

    def test_get_best_move_winning(self):
        """Test that AI takes winning move."""
        board = [
            [PLAYER_O, PLAYER_O, EMPTY],  # AI should win at (0,2)
            [PLAYER_X, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        move = get_best_move(board, PLAYER_O)
        # Should take the winning move
        assert move == (0, 2)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""

    def test_complete_game_scenario(self):
        """Test a complete game scenario."""
        board = create_board()
        
        # Make some moves
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        board[0][1] = PLAYER_X
        board[1][0] = PLAYER_O
        board[0][2] = PLAYER_X  # X wins
        
        assert check_winner(board) == PLAYER_X
        assert is_game_over(board) is True
        assert get_winning_line(board) == [(0, 0), (0, 1), (0, 2)]

    def test_tie_game_scenario(self):
        """Test a tie game scenario."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        
        assert check_winner(board) == TIE
        assert is_game_over(board) is True
        assert is_board_full(board) is True
        assert get_available_moves(board) == []

    def test_game_flow_with_ai(self):
        """Test game flow with AI moves."""
        board = create_board()
        
        # Human move
        board[1][1] = PLAYER_X
        
        # AI should make a good move
        ai_move = get_best_move(board, PLAYER_O)
        assert ai_move in get_available_moves(board)
        
        # Apply AI move
        board[ai_move[0]][ai_move[1]] = PLAYER_O
        
        # Game should still be ongoing
        assert check_winner(board) is None
        assert not is_game_over(board)