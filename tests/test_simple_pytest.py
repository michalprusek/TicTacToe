"""
Simple working pytest tests.
"""
import pytest
import numpy as np


class TestSimplePytest:
    """Simple pytest test class."""
    
    def test_basic_assertions(self):
        """Test basic pytest assertions."""
        assert True
        assert 1 == 1
        assert "hello" == "hello"
        assert [1, 2, 3] == [1, 2, 3]
    
    @pytest.mark.parametrize("input,expected", [
        (2, 4), (3, 9), (4, 16), (5, 25)
    ])
    def test_parametrized_square(self, input, expected):
        """Test parametrized square function."""
        assert input ** 2 == expected
    
    def test_numpy_arrays(self):
        """Test numpy array operations."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
        assert len(arr) == 5
    
    def test_string_operations(self):
        """Test string operations."""
        text = "TicTacToe"
        assert text.lower() == "tictactoe"
        assert text.upper() == "TICTACTOE"
        assert len(text) == 9
        assert "Tac" in text
    
    def test_list_operations(self):
        """Test list operations."""
        lst = [1, 2, 3]
        lst.append(4)
        assert lst == [1, 2, 3, 4]
        assert len(lst) == 4
    
    def test_exception_handling(self):
        """Test exception handling."""
        with pytest.raises(ZeroDivisionError):
            1 / 0
