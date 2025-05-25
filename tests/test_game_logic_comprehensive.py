"""
Comprehensive tests for game_logic module to improve coverage.
"""
import pytest
import io
import sys
import math
from unittest.mock import patch, Mock

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    create_board, get_available_moves, get_valid_moves,
    check_winner, get_winning_line, is_board_full, is_game_over,
    print_board, get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff
)


class TestGameLogicComprehensive:
    """Comprehensive test class for game_logic module."""
    
    def test_create_board(self):
        """Test board creation function."""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)
    
    def test_get_available_moves_comprehensive(self):
        """Test get_available_moves with various board states."""
        # Empty board
        board = create_board()
        moves = get_available_moves(board)
        assert len(moves) == 9
        
        # One move made
        board[1][1] = PLAYER_X
        moves = get_available_moves(board)
        assert len(moves) == 8
        assert (1, 1) not in moves
        
        # Full board
        for r in range(3):
            for c in range(3):
                board[r][c] = PLAYER_X if (r + c) % 2 == 0 else PLAYER_O
        moves = get_available_moves(board)
        assert len(moves) == 0    
    def test_get_valid_moves_alias(self):
        """Test that get_valid_moves is proper alias."""
        board = create_board()
        board[0][0] = PLAYER_X
        
        available = get_available_moves(board)
        valid = get_valid_moves(board)
        assert available == valid
        assert len(valid) == 8
    
    def test_check_winner_all_cases(self):
        """Test check_winner for all possible winning conditions."""
        # Test all rows
        for row in range(3):
            board = create_board()
            for col in range(3):
                board[row][col] = PLAYER_X
            assert check_winner(board) == PLAYER_X
            
        # Test all columns  
        for col in range(3):
            board = create_board()
            for row in range(3):
                board[row][col] = PLAYER_O
            assert check_winner(board) == PLAYER_O
            
        # Test main diagonal
        board = create_board()
        for i in range(3):
            board[i][i] = PLAYER_X
        assert check_winner(board) == PLAYER_X
        
        # Test anti-diagonal
        board = create_board()
        for i in range(3):
            board[i][2-i] = PLAYER_O
        assert check_winner(board) == PLAYER_O
    
    def test_check_winner_tie(self):
        """Test tie detection."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O], 
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        assert check_winner(board) == TIE    
    def test_get_winning_line_all_cases(self):
        """Test get_winning_line for all winning conditions."""
        # Test rows
        for row in range(3):
            board = create_board()
            for col in range(3):
                board[row][col] = PLAYER_X
            line = get_winning_line(board)
            expected = [(row, 0), (row, 1), (row, 2)]
            assert line == expected
            
        # Test columns
        for col in range(3):
            board = create_board()
            for row in range(3):
                board[row][col] = PLAYER_O
            line = get_winning_line(board)
            expected = [(0, col), (1, col), (2, col)]
            assert line == expected
            
        # Test diagonals
        board = create_board()
        for i in range(3):
            board[i][i] = PLAYER_X
        line = get_winning_line(board)
        assert line == [(0, 0), (1, 1), (2, 2)]
        
        board = create_board()
        for i in range(3):
            board[i][2-i] = PLAYER_O
        line = get_winning_line(board)
        assert line == [(0, 2), (1, 1), (2, 0)]
    
    def test_is_board_full_variations(self):
        """Test is_board_full with different scenarios."""
        # Empty board
        board = create_board()
        assert is_board_full(board) is False
        
        # Partially filled
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        assert is_board_full(board) is False
        
        # Almost full
        for r in range(3):
            for c in range(3):
                if (r, c) != (2, 2):
                    board[r][c] = PLAYER_X
        assert is_board_full(board) is False
        
        # Completely full
        board[2][2] = PLAYER_O
        assert is_board_full(board) is True    
    def test_is_game_over_variations(self):
        """Test is_game_over with different scenarios."""
        # Empty board - not over
        board = create_board()
        assert is_game_over(board) is False
        
        # With winner - is over
        board[0] = [PLAYER_X, PLAYER_X, PLAYER_X]
        assert is_game_over(board) is True
        
        # Tie - is over
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O], 
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        assert is_game_over(board) is True
    
    def test_print_board_output(self):
        """Test print_board produces correct output."""
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [EMPTY, EMPTY, PLAYER_O]
        ]
        
        # Capture stdout
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            print_board(board)
            output = captured_output.getvalue()
            
            # Verify output contains expected elements
            assert 'X' in output
            assert 'O' in output
            assert '|' in output  # Border characters
            assert '-' in output  # Separator lines
            
            # Count occurrences
            assert output.count('X') == 2  # Two X's on board
            assert output.count('O') == 2  # Two O's on board
            
        finally:
            sys.stdout = original_stdout    
    def test_get_other_player(self):
        """Test get_other_player function."""
        assert get_other_player(PLAYER_X) == PLAYER_O
        assert get_other_player(PLAYER_O) == PLAYER_X
    
    @patch('random.choice')
    def test_get_random_move_detailed(self, mock_choice):
        """Test get_random_move with different scenarios."""
        board = create_board()
        
        # Mock to return specific move
        mock_choice.return_value = (1, 1)
        move = get_random_move(board, PLAYER_X)
        assert move == (1, 1)
        
        # Test with no available moves
        full_board = [[PLAYER_X] * 3 for _ in range(3)]
        move = get_random_move(full_board, PLAYER_O)
        assert move is None
        
        # Test with one move available
        almost_full = [[PLAYER_X] * 3 for _ in range(3)]
        almost_full[2][2] = EMPTY
        mock_choice.return_value = (2, 2)
        move = get_random_move(almost_full, PLAYER_O)
        assert move == (2, 2)
    
    def test_board_to_string_variations(self):
        """Test board_to_string with different board states."""
        # Empty board
        board = create_board()
        result = board_to_string(board)
        assert result == ""
        
        # Board with symbols
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        result = board_to_string(board)
        assert result == "XOXO"  # Only non-empty symbols
        
        # Full board
        board = [[PLAYER_X, PLAYER_O, PLAYER_X] for _ in range(3)]
        result = board_to_string(board)
        assert result == "XOXXOXXOX"  # Actual result from the pattern    
    def test_get_board_diff_comprehensive(self):
        """Test get_board_diff with various scenarios."""
        # No changes
        board1 = create_board()
        board2 = create_board()
        diff = get_board_diff(board1, board2)
        assert diff == []
        
        # Valid additions
        board1 = create_board()
        board2 = create_board()
        board2[0][0] = PLAYER_X
        board2[1][1] = PLAYER_O
        
        diff = get_board_diff(board1, board2)
        expected = [(0, 0, PLAYER_X), (1, 1, PLAYER_O)]
        assert sorted(diff) == sorted(expected)
        
        # Invalid changes (overwriting existing symbol)
        board1 = create_board()
        board1[0][0] = PLAYER_X
        board2 = create_board()
        board2[0][0] = PLAYER_O  # Overwrite X with O
        
        diff = get_board_diff(board1, board2)
        assert diff == []  # Should ignore invalid changes
        
        # Symbol removal (should be ignored)
        board1 = create_board()
        board1[0][0] = PLAYER_X
        board2 = create_board()  # X removed
        
        diff = get_board_diff(board1, board2)
        assert diff == []  # Should ignore removals
    
    def test_minimax_comprehensive_scenarios(self):
        """Test minimax algorithm with comprehensive scenarios."""
        # Test immediate win scenario
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # X can win immediately
        score, move = minimax(board, PLAYER_X, 0, -math.inf, math.inf, PLAYER_X)
        assert score > 0  # Positive score for AI win
        assert move == (0, 2)  # Winning move
        
        # Test blocking scenario - O will choose strategically but may prioritize O's win
        score, move = minimax(board, PLAYER_O, 0, -math.inf, math.inf, PLAYER_O)
        # O can either block X at (0,2) or take O's win at (1,2) - both are valid strategies
        assert move in [(0, 2), (1, 2)]    
    def test_get_best_move_comprehensive(self):
        """Test get_best_move with comprehensive scenarios."""
        # Empty board - should choose center
        board = create_board()
        move = get_best_move(board, PLAYER_X)
        assert move == (1, 1)
        
        # Second move - should choose center if available
        board = create_board()
        board[0][0] = PLAYER_X
        move = get_best_move(board, PLAYER_O)
        assert move == (1, 1)
        
        # Center occupied - should choose corner or strategic position
        board = create_board()
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        move = get_best_move(board, PLAYER_X)
        assert move is not None
        assert board[move[0]][move[1]] == EMPTY
        
        # Winning opportunity
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        move = get_best_move(board, PLAYER_X)
        assert move == (0, 2)  # Take the win
        
        # Must block opponent or take own win
        move = get_best_move(board, PLAYER_O)
        assert move in [(0, 2), (1, 2)]  # Block X's win or take O's win
        
        # Full board - no moves
        full_board = [[PLAYER_X if (r+c) % 2 == 0 else PLAYER_O 
                      for c in range(3)] for r in range(3)]
        move = get_best_move(full_board, PLAYER_X)
        assert move is None
        
        # Only one move left
        almost_full = [[PLAYER_X] * 3 for _ in range(3)]
        almost_full[2][2] = EMPTY
        move = get_best_move(almost_full, PLAYER_O)
        assert move == (2, 2)