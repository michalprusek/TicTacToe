"""
Pure pytest tests for constants module.
"""
import pytest
from app.core.constants import (
    EMPTY, PLAYER_X, PLAYER_O, TIE, X, O, BOARD_SIZE
)


class TestConstants:
    """Pure pytest test class for constants."""
    
    def test_constants_import(self):
        """Test that constants can be imported successfully."""
        assert EMPTY is not None
        assert PLAYER_X is not None
        assert PLAYER_O is not None
        assert TIE is not None
        assert X is not None
        assert O is not None
        assert BOARD_SIZE is not None
    
    def test_empty_constant(self):
        """Test EMPTY constant value."""
        assert isinstance(EMPTY, str)
        assert EMPTY == ' '
    
    def test_player_constants(self):
        """Test player constants."""
        assert isinstance(PLAYER_X, str)
        assert isinstance(PLAYER_O, str)
        assert PLAYER_X == 'X'
        assert PLAYER_O == 'O'
        assert PLAYER_X != PLAYER_O
    
    def test_x_o_constants(self):
        """Test X and O constants."""
        assert X == PLAYER_X
        assert O == PLAYER_O
    
    def test_tie_constant(self):
        """Test TIE constant."""
        assert isinstance(TIE, str)
        assert TIE == "TIE"
    
    def test_board_size_constant(self):
        """Test BOARD_SIZE constant."""
        assert isinstance(BOARD_SIZE, int)
        assert BOARD_SIZE == 3
        assert BOARD_SIZE > 0