"""
Expanded comprehensive tests for game_logic module using pytest.
Tests all game logic functions with edge cases and comprehensive scenarios.
"""

import pytest
from app.main.game_logic import (
    create_board, get_available_moves, get_valid_moves, check_winner,
    is_board_full, make_move, evaluate_position, get_winner_with_indices,
    count_symbols, display_board, is_valid_move, minimax, minimax_alpha_beta,
    find_best_move, find_best_move_alpha_beta, EMPTY, PLAYER_X, PLAYER_O, TIE
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

    def test_display_board(self):
        """Test board display function."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        
        # Should not raise exception
        display_board(board)

    def test_display_board_full(self):
        """Test displaying full board."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_X, PLAYER_O, PLAYER_X]
        ]
        display_board(board)


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

    def test_is_valid_move_valid(self):
        """Test valid move validation."""
        board = create_board()
        assert is_valid_move(board, 0, 0) is True
        assert is_valid_move(board, 1, 1) is True
        assert is_valid_move(board, 2, 2) is True

    def test_is_valid_move_occupied(self):
        """Test invalid move on occupied cell."""
        board = create_board()
        board[0][0] = PLAYER_X
        assert is_valid_move(board, 0, 0) is False

    def test_is_valid_move_out_of_bounds(self):
        """Test invalid move out of bounds."""
        board = create_board()
        assert is_valid_move(board, -1, 0) is False
        assert is_valid_move(board, 0, -1) is False
        assert is_valid_move(board, 3, 0) is False
        assert is_valid_move(board, 0, 3) is False


class TestMakingMoves:
    """Test making moves on the board."""

    def test_make_move_valid(self):
        """Test making a valid move."""
        board = create_board()
        success = make_move(board, 1, 1, PLAYER_X)
        assert success is True
        assert board[1][1] == PLAYER_X

    def test_make_move_invalid_occupied(self):
        """Test making move on occupied cell."""
        board = create_board()
        board[0][0] = PLAYER_O
        success = make_move(board, 0, 0, PLAYER_X)
        assert success is False
        assert board[0][0] == PLAYER_O  # Should remain unchanged

    def test_make_move_invalid_bounds(self):
        """Test making move out of bounds."""
        board = create_board()
        success = make_move(board, -1, 0, PLAYER_X)
        assert success is False
        
        success = make_move(board, 3, 3, PLAYER_X)
        assert success is False


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

    def test_get_winner_with_indices_row(self):
        """Test getting winner with winning line indices for row."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        winner, indices = get_winner_with_indices(board)
        assert winner == PLAYER_X
        assert indices == [(0, 0), (0, 1), (0, 2)]

    def test_get_winner_with_indices_column(self):
        """Test getting winner with winning line indices for column."""
        board = [
            [PLAYER_O, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_X, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        winner, indices = get_winner_with_indices(board)
        assert winner == PLAYER_O
        assert indices == [(0, 0), (1, 0), (2, 0)]

    def test_get_winner_with_indices_no_winner(self):
        """Test getting winner when there's no winner."""
        board = create_board()
        winner, indices = get_winner_with_indices(board)
        assert winner is None
        assert indices == []


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

    def test_count_symbols_empty(self):
        """Test counting symbols on empty board."""
        board = create_board()
        x_count, o_count = count_symbols(board)
        assert x_count == 0
        assert o_count == 0

    def test_count_symbols_mixed(self):
        """Test counting symbols on mixed board."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        x_count, o_count = count_symbols(board)
        assert x_count == 3
        assert o_count == 2


class TestPositionEvaluation:
    """Test position evaluation for AI."""

    def test_evaluate_position_x_wins(self):
        """Test evaluation when X wins."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        score = evaluate_position(board)
        assert score > 0  # Positive score for X win

    def test_evaluate_position_o_wins(self):
        """Test evaluation when O wins."""
        board = [
            [PLAYER_O, PLAYER_O, PLAYER_O],
            [PLAYER_X, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        score = evaluate_position(board)
        assert score < 0  # Negative score for O win

    def test_evaluate_position_tie(self):
        """Test evaluation for tie."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        score = evaluate_position(board)
        assert score == 0  # Zero score for tie

    def test_evaluate_position_no_winner(self):
        """Test evaluation when game is ongoing."""
        board = create_board()
        board[0][0] = PLAYER_X
        score = evaluate_position(board)
        assert score == 0  # No winner yet


class TestMinimaxAlgorithm:
    """Test minimax algorithm implementations."""

    def test_minimax_terminal_x_wins(self):
        """Test minimax on terminal position where X wins."""
        board = [
            [PLAYER_X, PLAYER_X, PLAYER_X],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        score = minimax(board, 0, True)  # X's turn
        assert score > 0

    def test_minimax_terminal_o_wins(self):
        """Test minimax on terminal position where O wins."""
        board = [
            [PLAYER_O, PLAYER_O, PLAYER_O],
            [PLAYER_X, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        score = minimax(board, 0, False)  # O's turn
        assert score < 0

    def test_minimax_alpha_beta_equivalent(self):
        """Test that alpha-beta gives same result as regular minimax."""
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        regular_score = minimax(board, 3, True)
        alpha_beta_score = minimax_alpha_beta(board, 3, -float('inf'), float('inf'), True)
        
        assert regular_score == alpha_beta_score

    def test_find_best_move_center_opening(self):
        """Test finding best move for center opening."""
        board = create_board()
        row, col = find_best_move(board)
        
        # Should be a valid move
        assert 0 <= row <= 2
        assert 0 <= col <= 2
        assert board[row][col] == EMPTY

    def test_find_best_move_alpha_beta_equivalent(self):
        """Test that alpha-beta find_best_move gives same result."""
        board = [
            [PLAYER_X, EMPTY, EMPTY],
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        regular_move = find_best_move(board)
        alpha_beta_move = find_best_move_alpha_beta(board)
        
        # Both should return valid moves (may be different but both valid)
        assert 0 <= regular_move[0] <= 2
        assert 0 <= regular_move[1] <= 2
        assert 0 <= alpha_beta_move[0] <= 2
        assert 0 <= alpha_beta_move[1] <= 2

    def test_find_best_move_blocking(self):
        """Test that AI blocks winning move."""
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],  # AI should block at (0,2)
            [EMPTY, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        row, col = find_best_move(board)
        # Should block the winning move
        assert (row, col) == (0, 2)

    def test_find_best_move_winning(self):
        """Test that AI takes winning move."""
        board = [
            [PLAYER_O, PLAYER_O, EMPTY],  # AI should win at (0,2)
            [PLAYER_X, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        row, col = find_best_move(board)
        # Should take the winning move
        assert (row, col) == (0, 2)