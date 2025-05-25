"""
Comprehensive unittest tests for game_logic module.
"""
import io
import sys
import math
import unittest
from unittest.mock import patch, Mock

from app.main.game_logic import (
    EMPTY, PLAYER_X, PLAYER_O, TIE,
    create_board, get_available_moves, get_valid_moves,
    check_winner, get_winning_line, is_board_full, is_game_over,
    print_board, get_random_move, get_other_player, minimax, get_best_move,
    board_to_string, get_board_diff
)


class TestGameLogic(unittest.TestCase):
    
    def test_create_board(self):
        """Test board creation function."""
        board = create_board()
        self.assertEqual(len(board), 3)
        self.assertTrue(all(len(row) == 3 for row in board))
        self.assertTrue(all(cell == EMPTY for row in board for cell in row))
    
    def test_get_available_moves_comprehensive(self):
        """Test get_available_moves with various board states."""
        # Empty board
        board = create_board()
        moves = get_available_moves(board)
        self.assertEqual(len(moves), 9)
        
        # One move made
        board[1][1] = PLAYER_X
        moves = get_available_moves(board)
        self.assertEqual(len(moves), 8)
        self.assertNotIn((1, 1), moves)
        
        # Full board
        for r in range(3):
            for c in range(3):
                board[r][c] = PLAYER_X if (r + c) % 2 == 0 else PLAYER_O
        moves = get_available_moves(board)
        self.assertEqual(len(moves), 0    )
    def test_get_valid_moves_alias(self):
        """Test that get_valid_moves is proper alias."""
        board = create_board()
        board[0][0] = PLAYER_X
        
        available = get_available_moves(board)
        valid = get_valid_moves(board)
        self.assertEqual(available, valid)
        self.assertEqual(len(valid), 8)
    
    def test_check_winner_all_cases(self):
        """Test check_winner for all possible winning conditions."""
        # Test all rows
        for row in range(3):
            board = create_board()
            for col in range(3):
                board[row][col] = PLAYER_X
            self.assertEqual(check_winner(board), PLAYER_X)
            
        # Test all columns  
        for col in range(3):
            board = create_board()
            for row in range(3):
                board[row][col] = PLAYER_O
            self.assertEqual(check_winner(board), PLAYER_O)
            
        # Test main diagonal
        board = create_board()
        for i in range(3):
            board[i][i] = PLAYER_X
        self.assertEqual(check_winner(board), PLAYER_X)
        
        # Test anti-diagonal
        board = create_board()
        for i in range(3):
            board[i][2-i] = PLAYER_O
        self.assertEqual(check_winner(board), PLAYER_O)
    
    def test_check_winner_tie(self):
        """Test tie detection."""
        board = [
            [PLAYER_X, PLAYER_O, PLAYER_X],
            [PLAYER_O, PLAYER_X, PLAYER_O], 
            [PLAYER_O, PLAYER_X, PLAYER_O]
        ]
        self.assertEqual(check_winner(board), TIE    )
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
            self.assertIn('X', output)
            self.assertIn('O', output)
            self.assertIn('|', output)  # Border characters
            self.assertIn('-', output)  # Separator lines
            
            # Count occurrences
            self.assertEqual(output.count('X'), 2)  # Two X's on board
            self.assertEqual(output.count('O'), 2)  # Two O's on board
            
        finally:
            sys.stdout = original_stdout
    
    def test_get_other_player(self):
        """Test get_other_player function."""
        self.assertEqual(get_other_player(PLAYER_X), PLAYER_O)
        self.assertEqual(get_other_player(PLAYER_O), PLAYER_X)
    
    @patch('random.choice')
    def test_get_random_move_detailed(self, mock_choice):
        """Test get_random_move with different scenarios."""
        board = create_board()
        
        # Mock to return specific move
        mock_choice.return_value = (1, 1)
        move = get_random_move(board, PLAYER_X)
        self.assertEqual(move, (1, 1))
        
        # Test with no available moves
        full_board = [[PLAYER_X] * 3 for _ in range(3)]
        move = get_random_move(full_board, PLAYER_O)
        self.assertIsNone(move)    
    def test_board_to_string_variations(self):
        """Test board_to_string with different board states."""
        # Empty board
        board = create_board()
        result = board_to_string(board)
        self.assertEqual(result, "")
        
        # Board with symbols
        board = [
            [PLAYER_X, PLAYER_O, EMPTY],
            [EMPTY, PLAYER_X, EMPTY],
            [PLAYER_O, EMPTY, EMPTY]
        ]
        result = board_to_string(board)
        self.assertEqual(result, "XOXO")  # Only non-empty symbols
        
        # Full board
        board = [[PLAYER_X, PLAYER_O, PLAYER_X] for _ in range(3)]
        result = board_to_string(board)
        self.assertEqual(result, "XOXXOXXOX")
    
    def test_get_best_move_scenarios(self):
        """Test get_best_move with various scenarios."""
        # Empty board - should choose center
        board = create_board()
        move = get_best_move(board, PLAYER_X)
        self.assertEqual(move, (1, 1))
        
        # Winning opportunity
        board = [
            [PLAYER_X, PLAYER_X, EMPTY],
            [PLAYER_O, PLAYER_O, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        move = get_best_move(board, PLAYER_X)
        self.assertEqual(move, (0, 2))  # Take the win
        
        # PLAYER_O can win immediately - takes the win over blocking
        move = get_best_move(board, PLAYER_O)
        self.assertEqual(move, (1, 2))  # Take the win
        
        # Full board - no moves
        full_board = [[PLAYER_X if (r+c) % 2 == 0 else PLAYER_O 
                      for c in range(3)] for r in range(3)]
        move = get_best_move(full_board, PLAYER_X)
        self.assertIsNone(move)