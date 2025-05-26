# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Tests for game_logic module.
"""
import pytest
import math
from unittest.mock import patch

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    create_board, get_available_moves, get_valid_moves,
    check_winner, get_winning_line, is_board_full, is_game_over,
    get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff
)


class TestBasicBoardFunctions:
    """Test basic board manipulation functions."""
    
    def test_create_board(self):
        """Test board creation."""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)
    
    def test_get_available_moves_empty_board(self):
        """Test getting moves from empty board."""
        board = create_board()
        moves = get_available_moves(board)
        assert len(moves) == 9
        assert (0, 0) in moves
        assert (2, 2) in moves
    
    def test_get_available_moves_partial_board(self):
        """Test getting moves from partially filled board."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        
        moves = get_available_moves(board)
        assert len(moves) == 7
        assert (0, 0) not in moves
        assert (1, 1) not in moves
        assert (0, 1) in moves    
    def test_get_valid_moves_alias(self):
        """Test that get_valid_moves is an alias for get_available_moves."""
        board = create_board()
        board[0][0] = PLAYER_X
        
        available = get_available_moves(board)
        valid = get_valid_moves(board)
        assert available == valid
    
    def test_is_board_full_empty(self):
        """Test board full check on empty board."""
        board = create_board()
        assert is_board_full(board) is False
    
    def test_is_board_full_partial(self):
        """Test board full check on partial board."""
        board = create_board()
        board[0][0] = PLAYER_X
        assert is_board_full(board) is False
    
    def test_is_board_full_complete(self):
        """Test board full check on complete board."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        assert is_board_full(board) is True
    
    def test_board_to_string(self):
        """Test board to string conversion."""
        board = create_board()
        assert board_to_string(board) == ""
        
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        assert board_to_string(board) == "XO"
class TestWinnerDetection:
    """Test winner detection functions."""
    
    def test_check_winner_row(self):
        """Test winner detection for rows."""
        board = create_board()
        
        # No winner initially
        assert check_winner(board) is None
        
        # X wins first row
        board[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        assert check_winner(board) == PLAYER_X
        
        # Reset and test other rows
        board = create_board()
        board[1] = [PLAYER_O, PLAYER_O, PLAYER_O]
        assert check_winner(board) == PLAYER_O
    
    def test_check_winner_column(self):
        """Test winner detection for columns."""
        board = create_board()
        
        # X wins first column
        board[0][0] = PLAYER_X
        board[1][0] = PLAYER_X
        board[2][0] = PLAYER_X
        assert check_winner(board) == PLAYER_X
        
        # Reset and test third column
        board = create_board()
        board[0][2] = PLAYER_O
        board[1][2] = PLAYER_O
        board[2][2] = PLAYER_O
        assert check_winner(board) == PLAYER_O
    
    def test_check_winner_diagonal(self):
        """Test winner detection for diagonals."""
        board = create_board()
        
        # Main diagonal
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_X
        board[2][2] = PLAYER_X
        assert check_winner(board) == PLAYER_X
        
        # Anti-diagonal
        board = create_board()
        board[0][2] = PLAYER_O
        board[1][1] = PLAYER_O
        board[2][0] = PLAYER_O
        assert check_winner(board) == PLAYER_O    
    def test_check_winner_tie(self):
        """Test tie detection."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        assert check_winner(board) == TIE
    
    def test_get_winning_line_row(self):
        """Test getting winning line for row."""
        board = create_board()
        board[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        
        line = get_winning_line(board)
        assert line == [(0, 0), (0, 1), (0, 2)]
    
    def test_get_winning_line_column(self):
        """Test getting winning line for column."""
        board = create_board()
        board[0][1] = PLAYER_O
        board[1][1] = PLAYER_O
        board[2][1] = PLAYER_O
        
        line = get_winning_line(board)
        assert line == [(0, 1), (1, 1), (2, 1)]
    
    def test_get_winning_line_diagonal(self):
        """Test getting winning line for diagonals."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_X
        board[2][2] = PLAYER_X
        
        line = get_winning_line(board)
        assert line == [(0, 0), (1, 1), (2, 2)]
        
        # Anti-diagonal
        board = create_board()
        board[0][2] = PLAYER_O
        board[1][1] = PLAYER_O
        board[2][0] = PLAYER_O
        
        line = get_winning_line(board)
        assert line == [(0, 2), (1, 1), (2, 0)]    
    def test_get_winning_line_no_winner(self):
        """Test getting winning line when no winner."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        
        line = get_winning_line(board)
        assert line is None
    
    def test_is_game_over(self):
        """Test game over detection."""
        board = create_board()
        assert is_game_over(board) is False
        
        # With winner
        board[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        assert is_game_over(board) is True
        
        # With tie
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        assert is_game_over(board) is True


class TestAIFunctions:
    """Test AI-related functions."""
    
    def test_get_other_player(self):
        """Test getting the other player."""
        assert get_other_player(PLAYER_X) == PLAYER_O
        assert get_other_player(PLAYER_O) == PLAYER_X
    
    @patch('random.choice')
    def test_get_random_move(self, mock_choice):
        """Test random move selection."""
        board = create_board()
        board[0][0] = PLAYER_X
        
        mock_choice.return_value = (1, 1)
        move = get_random_move(board)
        assert move == (1, 1)    
    def test_get_random_move_no_moves(self):
        """Test random move when no moves available."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        move = get_random_move(board)
        assert move is None
    
    def test_get_best_move_empty_board(self):
        """Test best move on empty board (should choose center)."""
        board = create_board()
        move = get_best_move(board, PLAYER_X)
        assert move == (1, 1)  # Center
    
    def test_get_best_move_winning_opportunity(self):
        """Test best move when can win."""
        board = create_board()
        board[0][0] = PLAYER_X
        board[0][1] = PLAYER_X
        # X can win at (0, 2)
        
        move = get_best_move(board, PLAYER_X)
        assert move == (0, 2)
    
    def test_get_best_move_block_opponent(self):
        """Test best move when must block opponent."""
        board = create_board()
        board[0][0] = PLAYER_O
        board[0][1] = PLAYER_O
        # Must block O at (0, 2)
        
        move = get_best_move(board, PLAYER_X)
        assert move == (0, 2)
    
    def test_get_best_move_no_moves(self):
        """Test best move when board is full."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        move = get_best_move(board, PLAYER_X)
        assert move is None    
    def test_get_best_move_one_move_left(self):
        """Test best move when only one move available."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O],
            [PLAYER_O, PLAYER_X, EMPTY]
        ]
        move = get_best_move(board, PLAYER_X)
        assert move == (2, 2)  # Only available move
    
    def test_minimax_terminal_states(self):
        """Test minimax terminal state evaluations."""
        # AI wins
        board = create_board()
        board[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        
        score, _ = minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score == 10  # Win at depth 0
        
        # AI loses
        board = create_board()
        board[0] = [PLAYER_O, PLAYER_O, PLAYER_O]
        
        score, _ = minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score == -10  # Loss at depth 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_board_diff_no_changes(self):
        """Test board diff with no changes."""
        board1 = create_board()
        board2 = create_board()
        
        diff = get_board_diff(board1, board2)
        assert diff == []
    
    def test_get_board_diff_with_changes(self):
        """Test board diff with valid changes."""
        board1 = create_board()
        board2 = create_board()
        board2[0][0] = PLAYER_X
        board2[1][1] = PLAYER_O
        
        diff = get_board_diff(board1, board2)
        expected = [(0, 0, PLAYER_X), (1, 1, PLAYER_O)]
        assert sorted(diff) == sorted(expected)    
    def test_get_board_diff_invalid_changes(self):
        """Test board diff ignores invalid changes."""
        board1 = create_board()
        board1[0][0] = PLAYER_X
        
        board2 = create_board()
        board2[0][0] = PLAYER_O  # Overwriting existing symbol
        
        diff = get_board_diff(board1, board2)
        assert diff == []  # Should ignore this change
    
    def test_get_board_diff_removal(self):
        """Test board diff ignores symbol removals."""
        board1 = create_board()
        board1[0][0] = PLAYER_X
        
        board2 = create_board()
        # board2[0][0] remains EMPTY (removal)
        
        diff = get_board_diff(board1, board2)
        assert diff == []  # Should ignore removals
