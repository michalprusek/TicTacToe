"""
Tests for game_utils module.
"""
import unittest
import logging
from unittest.mock import patch

from app.main.game_utils import (
    convert_board_1d_to_2d, get_board_symbol_counts, setup_logger
)


class TestGameUtils(unittest.TestCase):
    
    def test_convert_board_1d_to_2d_valid(self):
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', 'X', 'O', ' ']
        expected = [['X', 'O', ' '], ['X', 'O', ' '], ['X', 'O', ' ']]
        result = convert_board_1d_to_2d(board_1d)
        self.assertEqual(result, expected)
    
    def test_convert_board_1d_to_2d_invalid_length(self):
        board_1d = ['X', 'O', 'X']
        result = convert_board_1d_to_2d(board_1d)
        self.assertEqual(result, board_1d)
    
    def test_get_board_symbol_counts_2d_board(self):
        board_2d = [['X', 'O', ' '], ['X', 'O', ' '], ['X', 'O', ' ']]
        result = get_board_symbol_counts(board_2d)
        expected = {'X': 3, 'O': 3, ' ': 3}
        self.assertEqual(result, expected)
    
    def test_get_board_symbol_counts_1d_board(self):
        board_1d = ['X', 'X', 'O', 'O', 'O', ' ', ' ', ' ', ' ']
        result = get_board_symbol_counts(board_1d)
        expected = {'X': 2, 'O': 3, ' ': 4}
        self.assertEqual(result, expected)
    
    def test_get_board_symbol_counts_empty_board(self):
        board_1d = [' '] * 9
        result = get_board_symbol_counts(board_1d)
        expected = {'X': 0, 'O': 0, ' ': 9}
        self.assertEqual(result, expected)
    
    @patch('logging.getLogger')
    def test_setup_logger(self, mock_getLogger):
        mock_logger = unittest.mock.Mock()
        mock_logger.handlers = []
        mock_getLogger.return_value = mock_logger        
        result = setup_logger("test_logger")
        self.assertEqual(result, mock_logger)


if __name__ == '__main__':
    unittest.main()