# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.main.game_utils module using pytest.
Tests all utility functions for game board management.
"""

import pytest
import logging
from unittest.mock import patch
from app.main.game_utils import (
    convert_board_1d_to_2d, get_board_symbol_counts, setup_logger
)


class TestConvertBoard1dTo2d:
    """Test convert_board_1d_to_2d function."""
    
    def test_valid_1d_board_conversion(self):
        """Test conversion of valid 1D board to 2D."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        result = convert_board_1d_to_2d(board_1d)
        
        expected = [
            ['X', 'O', ' '],
            ['X', 'O', ' '],
            ['X', 'O', ' ']
        ]
        assert result == expected
    
    def test_empty_1d_board_conversion(self):
        """Test conversion of empty 1D board."""
        board_1d = [' '] * 9
        result = convert_board_1d_to_2d(board_1d)
        
        expected = [[' ', ' ', ' ']] * 3
        assert result == expected
    
    def test_mixed_symbols_1d_board(self):
        """Test conversion with mixed symbols."""
        board_1d = ['X', 'X', 'O', 'O', 'X', 'O', ' ', ' ', ' ']
        result = convert_board_1d_to_2d(board_1d)
        
        expected = [
            ['X', 'X', 'O'],
            ['O', 'X', 'O'],
            [' ', ' ', ' ']
        ]
        assert result == expected
    
    def test_invalid_length_board(self):
        """Test with invalid length board (not 9 elements)."""
        board_1d = ['X', 'O', ' ', 'X']  # Only 4 elements
        result = convert_board_1d_to_2d(board_1d)
        
        # Should return original
        assert result == board_1d
    
    def test_empty_list(self):
        """Test with empty list."""
        board_1d = []
        result = convert_board_1d_to_2d(board_1d)
        
        assert result == []
    
    def test_non_list_input(self):
        """Test with non-list input."""
        inputs = [None, "string", 123, {'a': 1}, (1, 2, 3)]
        
        for input_val in inputs:
            result = convert_board_1d_to_2d(input_val)
            assert result == input_val
    
    def test_list_with_wrong_length(self):
        """Test with list of wrong length."""
        board_1d = ['X'] * 10  # 10 elements instead of 9
        result = convert_board_1d_to_2d(board_1d)
        
        assert result == board_1d


class TestGetBoardSymbolCounts:
    """Test get_board_symbol_counts function."""
    
    def test_empty_2d_board(self):
        """Test counting symbols on empty 2D board."""
        board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        counts = get_board_symbol_counts(board)
        
        assert counts == {'X': 0, 'O': 0, ' ': 9}
    
    def test_mixed_2d_board(self):
        """Test counting symbols on mixed 2D board."""
        board = [
            ['X', 'O', ' '],
            ['X', 'X', 'O'],
            [' ', ' ', 'O']
        ]
        counts = get_board_symbol_counts(board)
        
        assert counts == {'X': 3, 'O': 3, ' ': 3}
    
    def test_full_2d_board_x_wins(self):
        """Test counting on board where X dominates."""
        board = [
            ['X', 'X', 'X'],
            ['O', 'O', ' '],
            [' ', ' ', ' ']
        ]
        counts = get_board_symbol_counts(board)
        
        assert counts == {'X': 3, 'O': 2, ' ': 4}
    
    def test_1d_board_counting(self):
        """Test counting symbols on 1D board."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        counts = get_board_symbol_counts(board_1d)
        
        assert counts == {'X': 3, 'O': 3, ' ': 3}
    
    def test_empty_1d_board(self):
        """Test counting on empty 1D board."""
        board_1d = [' '] * 9
        counts = get_board_symbol_counts(board_1d)
        
        assert counts == {'X': 0, 'O': 0, ' ': 9}
    
    def test_unknown_symbols_counted_as_empty(self):
        """Test that unknown symbols are counted as empty."""
        board_1d = ['X', 'O', 'Y', 'Z', '1', '2', ' ', ' ', ' ']
        counts = get_board_symbol_counts(board_1d)
        
        # Y, Z, 1, 2 should be counted as empty
        assert counts == {'X': 1, 'O': 1, ' ': 7}
    
    def test_non_list_board(self):
        """Test with non-list board."""
        board = "not a list"
        counts = get_board_symbol_counts(board)
        
        # Should handle gracefully
        assert counts == {'X': 0, 'O': 0, ' ': 0}
    
    def test_empty_list_board(self):
        """Test with empty list."""
        board = []
        counts = get_board_symbol_counts(board)
        
        assert counts == {'X': 0, 'O': 0, ' ': 0}
    
    def test_malformed_2d_board(self):
        """Test with malformed 2D board."""
        board = ['X', 'O', 'X']  # 1D board instead of malformed 2D
        counts = get_board_symbol_counts(board)
        
        # Should still count the symbols present
        assert counts == {'X': 2, 'O': 1, ' ': 0}


class TestSetupLogger:
    """Test setup_logger function."""
    
    def test_logger_creation(self):
        """Test basic logger creation."""
        logger_name = "test_logger"
        logger = setup_logger(logger_name)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name
    
    def test_logger_default_level(self):
        """Test logger created with default level."""
        logger = setup_logger("test_default")
        
        # Should have INFO level by default (though basicConfig might override)
        assert isinstance(logger, logging.Logger)
    
    def test_logger_custom_level(self):
        """Test logger created with custom level."""
        logger = setup_logger("test_custom", logging.DEBUG)
        
        assert isinstance(logger, logging.Logger)
    
    def test_logger_no_duplicate_handlers(self):
        """Test that duplicate handlers are not added."""
        logger_name = "test_no_duplicates"
        
        # Create logger twice
        logger1 = setup_logger(logger_name)
        logger2 = setup_logger(logger_name)
        
        # Should be the same logger instance
        assert logger1 is logger2
    
    @patch('logging.basicConfig')
    def test_basic_config_called_for_new_logger(self, mock_basic_config):
        """Test that basicConfig is called for new logger."""
        # Create a fresh logger that doesn't exist
        logger_name = "brand_new_logger_unique_name"
        
        # Mock getLogger to return a logger with no handlers
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = logging.Logger(logger_name)
            mock_logger.handlers = []  # No handlers
            mock_get_logger.return_value = mock_logger
            
            setup_logger(logger_name, logging.DEBUG)
            
            # basicConfig should have been called
            mock_basic_config.assert_called_once_with(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            )
    
    @patch('logging.basicConfig')
    def test_basic_config_not_called_for_existing_logger(self, mock_basic_config):
        """Test that basicConfig is not called if logger already has handlers."""
        logger_name = "existing_logger"
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = logging.Logger(logger_name)
            mock_logger.handlers = [logging.StreamHandler()]  # Has handlers
            mock_get_logger.return_value = mock_logger
            
            setup_logger(logger_name)
            
            # basicConfig should NOT have been called
            mock_basic_config.assert_not_called()
    
    def test_logger_format_string(self):
        """Test that the format string is correctly specified."""
        expected_format = '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        
        with patch('logging.basicConfig') as mock_basic_config:
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = logging.Logger("test")
                mock_logger.handlers = []
                mock_get_logger.return_value = mock_logger
                
                setup_logger("test")
                
                # Check that the format was passed correctly
                mock_basic_config.assert_called_once()
                call_args = mock_basic_config.call_args
                assert call_args[1]['format'] == expected_format
    
    def test_different_logger_names(self):
        """Test creating loggers with different names."""
        logger1 = setup_logger("logger_one")
        logger2 = setup_logger("logger_two")
        
        assert logger1.name == "logger_one"
        assert logger2.name == "logger_two"
        assert logger1 is not logger2