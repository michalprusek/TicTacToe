"""
Extended tests for game_utils module.
"""
import pytest
import logging
from unittest.mock import Mock, patch
from app.main.game_utils import (
    convert_board_1d_to_2d, 
    get_board_symbol_counts,
    setup_logger
)


class TestGameUtilsExtended:
    """Extended tests for game_utils functions."""

    def test_convert_1d_to_2d_board_valid(self):
        """Test converting valid 1D board to 2D."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        result = convert_board_1d_to_2d(board_1d)
        expected = [
            ['X', 'O', ' '],
            ['X', 'O', ' '],
            ['X', 'O', ' ']
        ]
        assert result == expected

    def test_convert_1d_to_2d_board_invalid_length(self):
        """Test convert with invalid length board."""
        board_1d = ['X', 'O', ' ', 'X']  # Only 4 elements
        result = convert_board_1d_to_2d(board_1d)
        assert result == board_1d  # Should return original

    def test_convert_1d_to_2d_board_not_list(self):
        """Test convert with non-list input."""
        result = convert_board_1d_to_2d("not a list")
        assert result == "not a list"  # Should return original    def test_get_board_symbol_counts_2d_board(self):
        """Test symbol counting with 2D board."""
        board_2d = [
            ['X', 'O', ' '],
            ['X', 'X', 'O'],
            [' ', ' ', ' ']
        ]
        result = get_board_symbol_counts(board_2d)
        expected = {'X': 3, 'O': 2, ' ': 4}
        assert result == expected

    def test_get_board_symbol_counts_1d_board(self):
        """Test symbol counting with 1D board."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        result = get_board_symbol_counts(board_1d)
        expected = {'X': 3, 'O': 3, ' ': 3}
        assert result == expected

    def test_get_board_symbol_counts_with_unknown_symbols(self):
        """Test symbol counting with unknown symbols."""
        board_1d = ['X', 'O', '?', 'X', 'invalid', ' ']
        result = get_board_symbol_counts(board_1d)
        # Unknown symbols are counted as empty spaces
        assert result['X'] == 2
        assert result['O'] == 1
        assert result[' '] == 3  # 1 actual space + 2 unknown symbols

    def test_get_board_symbol_counts_empty_board(self):
        """Test symbol counting with empty board."""
        result = get_board_symbol_counts([])
        expected = {'X': 0, 'O': 0, ' ': 0}
        assert result == expected

    def test_setup_logger_basic(self):
        """Test setup_logger creates logger."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
