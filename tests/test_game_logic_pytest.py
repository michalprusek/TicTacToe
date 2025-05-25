"""
Pytest tests for game logic module.
"""
import pytest
from app.main.game_logic import (
    create_board, check_winner, is_board_full, 
    get_available_moves, EMPTY, PLAYER_X, PLAYER_O, TIE
)


class TestGameLogicPytest:
    """Pytest test class for game logic."""
    
    def test_create_board(self):
        """Test board creation function."""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)
    
    def test_board_manipulation(self):
        """Test basic board manipulation."""
        board = create_board()
        
        # Test setting values directly
        board[0][0] = PLAYER_X
        assert board[0][0] == PLAYER_X
        
        board[1][1] = PLAYER_O
        assert board[1][1] == PLAYER_O
        
        # Test that other cells remain empty
        assert board[0][1] == EMPTY
        assert board[2][2] == EMPTY    
    @pytest.mark.parametrize("board_setup,expected", [
        ([[PLAYER_X, PLAYER_X, PLAYER_X], [EMPTY, PLAYER_O, EMPTY], [PLAYER_O, EMPTY, EMPTY]], PLAYER_X),
        ([[PLAYER_X, PLAYER_O, EMPTY], [PLAYER_X, PLAYER_O, EMPTY], [PLAYER_X, EMPTY, EMPTY]], PLAYER_X),
        ([[PLAYER_X, PLAYER_O, PLAYER_X], [PLAYER_O, PLAYER_X, PLAYER_O], [PLAYER_X, PLAYER_O, PLAYER_X]], PLAYER_X),
        ([[PLAYER_O, PLAYER_O, PLAYER_O], [PLAYER_X, PLAYER_X, EMPTY], [EMPTY, EMPTY, EMPTY]], PLAYER_O),
        ([[PLAYER_X, PLAYER_O, PLAYER_X], [PLAYER_O, PLAYER_X, PLAYER_O], [PLAYER_O, PLAYER_X, PLAYER_O]], 'TIE'),
    ])
    def test_check_winner_scenarios(self, board_setup, expected):
        """Test various win scenarios."""
        assert check_winner(board_setup) == expected
    
    def test_is_board_full(self):
        """Test board full detection."""
        board = create_board()
        assert is_board_full(board) is False
        
        for i in range(3):
            for j in range(3):
                board[i][j] = PLAYER_X if (i + j) % 2 == 0 else PLAYER_O
        assert is_board_full(board) is True
    
    def test_get_available_moves(self):
        """Test getting available moves."""
        board = create_board()
        moves = get_available_moves(board)
        assert len(moves) == 9
        assert (0, 0) in moves
        assert (2, 2) in moves
        
        board[0][0] = PLAYER_X
        board[1][1] = PLAYER_O
        moves = get_available_moves(board)
        assert len(moves) == 7
        assert (0, 0) not in moves
        assert (1, 1) not in moves