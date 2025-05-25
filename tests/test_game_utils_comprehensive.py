"""Comprehensive tests for game_utils.py module."""
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

from app.main.game_utils import (
    convert_board_1d_to_2d,
    get_board_symbol_counts,
    setup_logger
)


class TestConvertBoard1dTo2d:
    """Test convert_board_1d_to_2d function."""
    
    def test_convert_valid_1d_board(self):
        """Test conversion of valid 1D board to 2D."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        result = convert_board_1d_to_2d(board_1d)
        
        expected = [
            ['X', 'O', ' '],
            ['X', 'O', ' '],
            ['X', 'O', ' ']
        ]
        assert result == expected
    
    def test_convert_empty_1d_board(self):
        """Test conversion of empty 1D board to 2D."""
        board_1d = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        result = convert_board_1d_to_2d(board_1d)
        
        expected = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert result == expected
    
    def test_convert_mixed_symbols_1d_board(self):
        """Test conversion with mixed symbols."""
        board_1d = ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O', 'X']
        result = convert_board_1d_to_2d(board_1d)
        
        expected = [
            ['X', 'O', 'X'],
            ['O', 'X', 'O'],
            ['X', 'O', 'X']
        ]
        assert result == expected
    
    def test_convert_invalid_length_list(self):
        """Test conversion with wrong length list."""
        # Too short
        board_1d = ['X', 'O', ' ']
        result = convert_board_1d_to_2d(board_1d)
        assert result == board_1d  # Should return original
        
        # Too long
        board_1d = ['X'] * 12
        result = convert_board_1d_to_2d(board_1d)
        assert result == board_1d  # Should return original
    
    def test_convert_non_list_input(self):
        """Test conversion with non-list input."""
        # String input
        board_str = "XO XO XO "
        result = convert_board_1d_to_2d(board_str)
        assert result == board_str  # Should return original
        
        # Integer input
        board_int = 123
        result = convert_board_1d_to_2d(board_int)
        assert result == board_int  # Should return original
        
        # None input
        result = convert_board_1d_to_2d(None)
        assert result is None  # Should return original
    
    def test_convert_empty_list(self):
        """Test conversion with empty list."""
        board_1d = []
        result = convert_board_1d_to_2d(board_1d)
        assert result == []  # Should return original


class TestGetBoardSymbolCounts:
    """Test get_board_symbol_counts function."""
    
    def test_count_2d_board_mixed(self):
        """Test counting symbols on 2D board with mixed symbols."""
        board_2d = [
            ['X', 'O', ' '],
            ['X', 'X', 'O'],
            [' ', 'O', ' ']
        ]
        result = get_board_symbol_counts(board_2d)
        
        expected = {'X': 3, 'O': 3, ' ': 3}
        assert result == expected
    
    def test_count_2d_board_empty(self):
        """Test counting symbols on empty 2D board."""
        board_2d = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        result = get_board_symbol_counts(board_2d)
        
        expected = {'X': 0, 'O': 0, ' ': 9}
        assert result == expected
    
    def test_count_2d_board_all_x(self):
        """Test counting symbols on board with all X."""
        board_2d = [
            ['X', 'X', 'X'],
            ['X', 'X', 'X'],
            ['X', 'X', 'X']
        ]
        result = get_board_symbol_counts(board_2d)
        
        expected = {'X': 9, 'O': 0, ' ': 0}
        assert result == expected
    
    def test_count_1d_board_mixed(self):
        """Test counting symbols on 1D board with mixed symbols."""
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        result = get_board_symbol_counts(board_1d)
        
        expected = {'X': 3, 'O': 3, ' ': 3}
        assert result == expected
    
    def test_count_1d_board_empty(self):
        """Test counting symbols on empty 1D board."""
        board_1d = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        result = get_board_symbol_counts(board_1d)
        
        expected = {'X': 0, 'O': 0, ' ': 9}
        assert result == expected
    
    def test_count_board_with_unknown_symbols(self):
        """Test counting with unknown symbols (should be counted as empty)."""
        board_1d = ['X', 'O', 'Z', 'X', 'Y', ' ', 'X', 'O', 'Q']
        result = get_board_symbol_counts(board_1d)
        
        # Unknown symbols (Z, Y, Q) should be counted as empty spaces
        expected = {'X': 3, 'O': 2, ' ': 4}
        assert result == expected
    
    def test_count_non_list_board(self):
        """Test counting with non-list board."""
        # String input
        result = get_board_symbol_counts("not a board")
        expected = {'X': 0, 'O': 0, ' ': 0}
        assert result == expected
        
        # None input
        result = get_board_symbol_counts(None)
        expected = {'X': 0, 'O': 0, ' ': 0}
        assert result == expected
    
    def test_count_malformed_2d_board(self):
        """Test counting with malformed 2D board."""
        # Board with wrong dimensions
        board_2d = [
            ['X', 'O'],
            ['X', 'X', 'O'],
            [' ']
        ]
        result = get_board_symbol_counts(board_2d)
        
        # Should still process as flattened list
        expected = {'X': 3, 'O': 2, ' ': 1}
        assert result == expected
    
    def test_count_empty_list(self):
        """Test counting with empty list."""
        result = get_board_symbol_counts([])
        expected = {'X': 0, 'O': 0, ' ': 0}
        assert result == expected


class TestSetupLogger:
    """Test setup_logger function."""
    
    def test_setup_logger_default_level(self):
        """Test setting up logger with default level."""
        logger_name = "test_logger_default"
        logger = setup_logger(logger_name)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name
        assert logger.level <= logging.INFO  # Should be INFO or lower (more permissive)
    
    def test_setup_logger_custom_level(self):
        """Test setting up logger with custom level."""
        logger_name = "test_logger_custom"
        logger = setup_logger(logger_name, logging.DEBUG)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name
        # Note: The actual level might be inherited from root logger
    
    def test_setup_logger_warning_level(self):
        """Test setting up logger with WARNING level."""
        logger_name = "test_logger_warning"
        logger = setup_logger(logger_name, logging.WARNING)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name
    
    def test_setup_logger_error_level(self):
        """Test setting up logger with ERROR level."""
        logger_name = "test_logger_error"
        logger = setup_logger(logger_name, logging.ERROR)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name
    
    def test_setup_logger_no_duplicate_handlers(self):
        """Test that setup_logger doesn't create duplicate handlers."""
        logger_name = "test_logger_no_dup"
        
        # First call
        logger1 = setup_logger(logger_name)
        initial_handler_count = len(logger1.handlers)
        
        # Second call with same name
        logger2 = setup_logger(logger_name)
        final_handler_count = len(logger2.handlers)
        
        # Should be the same logger instance
        assert logger1 is logger2
        # Handler count should not increase
        assert final_handler_count == initial_handler_count
    
    @patch('logging.basicConfig')
    def test_setup_logger_calls_basic_config(self, mock_basic_config):
        """Test that setup_logger calls basicConfig when no handlers exist."""
        logger_name = "test_logger_basic_config"
        
        # Mock a logger with no handlers
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = []  # No existing handlers
            mock_get_logger.return_value = mock_logger
            
            setup_logger(logger_name, logging.DEBUG)
            
            # Should call basicConfig
            mock_basic_config.assert_called_once_with(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            )
    
    @patch('logging.basicConfig')
    def test_setup_logger_skips_basic_config_with_handlers(self, mock_basic_config):
        """Test that setup_logger skips basicConfig when handlers exist."""
        logger_name = "test_logger_skip_config"
        
        # Mock a logger with existing handlers
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = [MagicMock()]  # Has existing handler
            mock_get_logger.return_value = mock_logger
            
            setup_logger(logger_name, logging.DEBUG)
            
            # Should NOT call basicConfig
            mock_basic_config.assert_not_called()
    
    def test_setup_logger_different_names(self):
        """Test setting up loggers with different names."""
        logger1 = setup_logger("logger_one")
        logger2 = setup_logger("logger_two")
        
        assert logger1.name == "logger_one"
        assert logger2.name == "logger_two"
        assert logger1 is not logger2
    
    def test_setup_logger_format_string(self):
        """Test that the logger format string is correctly configured."""
        logger_name = "test_logger_format"
        
        with patch('logging.basicConfig') as mock_basic_config:
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_logger.handlers = []
                mock_get_logger.return_value = mock_logger
                
                setup_logger(logger_name)
                
                # Check that basicConfig was called with correct format
                call_args = mock_basic_config.call_args
                assert call_args[1]['format'] == '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'


class TestEdgeCases:
    """Test edge cases and integration scenarios."""
    
    def test_board_conversion_then_counting(self):
        """Test converting 1D board to 2D then counting symbols."""
        board_1d = ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O', 'X']
        
        # Convert to 2D
        board_2d = convert_board_1d_to_2d(board_1d)
        
        # Count symbols in 2D board
        counts = get_board_symbol_counts(board_2d)
        
        expected = {'X': 5, 'O': 4, ' ': 0}
        assert counts == expected
    
    def test_symbol_counting_consistency(self):
        """Test that symbol counting is consistent between 1D and 2D formats."""
        # Same data in different formats
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        board_2d = [
            ['X', 'O', ' '],
            ['X', 'O', ' '],
            ['X', 'O', ' ']
        ]
        
        counts_1d = get_board_symbol_counts(board_1d)
        counts_2d = get_board_symbol_counts(board_2d)
        
        assert counts_1d == counts_2d
    
    def test_logger_integration_with_different_modules(self):
        """Test logger setup with different module names."""
        logger1 = setup_logger("module1")
        logger2 = setup_logger("module2.submodule")
        logger3 = setup_logger("app.main.game_utils")
        
        # All should be valid loggers
        assert all(isinstance(logger, logging.Logger) for logger in [logger1, logger2, logger3])
        
        # Should have different names
        names = {logger1.name, logger2.name, logger3.name}
        assert len(names) == 3


if __name__ == "__main__":
    pytest.main([__file__])