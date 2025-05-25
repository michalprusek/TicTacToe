"""
Tests for game_utils module.
"""
import pytest
import logging

from app.main.game_utils import (
    convert_board_1d_to_2d, get_board_symbol_counts, setup_logger
)


class TestBoardConversion:
    """Test board conversion utilities."""
    
    def test_convert_board_1d_to_2d_valid(self):
        """Test converting valid 1D board to 2D."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', ' ', ' ', 'X']
        expected = [
            ['X', 'O', ' '],
            ['X', 'O', ' '],
            [' ', ' ', 'X']
        ]
        
        result = convert_board_1d_to_2d(board_1d)
        assert result == expected
    
    def test_convert_board_1d_to_2d_invalid_length(self):
        """Test converting invalid length board."""
        board_1d = ['X', 'O', ' ']  # Only 3 elements
        
        result = convert_board_1d_to_2d(board_1d)
        assert result == board_1d  # Should return original
    
    def test_convert_board_1d_to_2d_not_list(self):
        """Test converting non-list input."""
        board_1d = "not a list"
        
        result = convert_board_1d_to_2d(board_1d)
        assert result == board_1d  # Should return original
    
    def test_convert_board_1d_to_2d_already_2d(self):
        """Test converting already 2D board."""
        board_2d = [['X', 'O'], ['X', 'O']]  # Not 9 elements
        
        result = convert_board_1d_to_2d(board_2d)
        assert result == board_2d  # Should return original
class TestBoardSymbolCounts:
    """Test board symbol counting utilities."""
    
    def test_get_board_symbol_counts_2d(self):
        """Test symbol counting on 2D board."""
        board_2d = [
            ['X', 'O', ' '],
            ['X', 'O', ' '],
            [' ', ' ', 'X']
        ]
        
        counts = get_board_symbol_counts(board_2d)
        assert counts['X'] == 3
        assert counts['O'] == 2
        assert counts[' '] == 4
    
    def test_get_board_symbol_counts_1d(self):
        """Test symbol counting on 1D board."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', ' ', ' ', 'X']
        
        counts = get_board_symbol_counts(board_1d)
        assert counts['X'] == 3
        assert counts['O'] == 2
        assert counts[' '] == 4
    
    def test_get_board_symbol_counts_empty_board(self):
        """Test symbol counting on empty board."""
        board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        
        counts = get_board_symbol_counts(board)
        assert counts['X'] == 0
        assert counts['O'] == 0
        assert counts[' '] == 9
    
    def test_get_board_symbol_counts_unknown_symbols(self):
        """Test symbol counting with unknown symbols."""
        board_1d = ['X', 'Y', 'Z', 'X', 'O', ' ', ' ', ' ', 'X']
        
        counts = get_board_symbol_counts(board_1d)
        assert counts['X'] == 3
        assert counts['O'] == 1
        assert counts[' '] == 5  # Unknown symbols counted as empty    
    def test_get_board_symbol_counts_invalid_board(self):
        """Test symbol counting with invalid board."""
        counts = get_board_symbol_counts("not a board")
        assert counts['X'] == 0
        assert counts['O'] == 0
        assert counts[' '] == 0
    
    def test_get_board_symbol_counts_none_board(self):
        """Test symbol counting with None board."""
        counts = get_board_symbol_counts(None)
        assert counts['X'] == 0
        assert counts['O'] == 0
        assert counts[' '] == 0


class TestLogger:
    """Test logger setup utilities."""
    
    def test_setup_logger_default(self):
        """Test logger setup with defaults."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_setup_logger_custom_level(self):
        """Test logger setup with custom level."""
        logger = setup_logger("test_logger_debug", logging.DEBUG)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger_debug"
    
    def test_setup_logger_no_duplicate_handlers(self):
        """Test that setup_logger doesn't create duplicate handlers."""
        logger1 = setup_logger("test_duplicate")
        handler_count_1 = len(logger1.handlers)
        
        logger2 = setup_logger("test_duplicate")
        handler_count_2 = len(logger2.handlers)
        
        # Should be same logger instance
        assert logger1 is logger2
        # Handler count should not increase
        assert handler_count_1 == handler_count_2