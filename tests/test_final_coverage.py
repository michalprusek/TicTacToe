"""
Final coverage tests for TicTacToe application.
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from app.core.game_state import GameState, EMPTY, PLAYER_X, PLAYER_O
from app.core.strategy import RandomStrategy, MinimaxStrategy
from app.core.config import AppConfig, GameDetectorConfig
from app.core.utils import FPSCalculator
from app.main.game_logic import create_board, get_available_moves, check_winner
from app.main.game_utils import convert_board_1d_to_2d, get_board_symbol_counts
from app.main.path_utils import get_project_root


class TestFinalCoverage(unittest.TestCase):
    """Final coverage tests."""
    
    def test_game_state_properties(self):
        """Test GameState basic properties."""
        gs = GameState()
        self.assertEqual(gs.board, [[EMPTY] * 3 for _ in range(3)])
        self.assertIsNone(gs.grid_points)
        self.assertEqual(gs.detection_results, [])
        self.assertFalse(gs.is_physical_grid_valid())
    
    def test_game_state_symbol_counting(self):
        """Test symbol counting."""
        gs = GameState()
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][1] = PLAYER_O
        
        x_count, o_count = gs.count_symbols()
        self.assertEqual(x_count, 1)
        self.assertEqual(o_count, 1)    
    def test_game_state_validation(self):
        """Test board validation."""
        gs = GameState()
        self.assertTrue(gs.is_valid())
        self.assertTrue(gs.is_valid_turn_sequence())
    
    def test_game_state_error_handling(self):
        """Test error handling."""
        gs = GameState()
        self.assertFalse(gs.is_error_active())
        
        gs.set_error("Test error")
        self.assertTrue(gs.is_error_active())
        self.assertEqual(gs.get_error(), "Test error")
        
        gs.clear_error_message()
        self.assertFalse(gs.is_error_active())
    
    def test_fps_calculator(self):
        """Test FPS calculator."""
        calc = FPSCalculator()
        self.assertEqual(calc.buffer_size, 10)
        self.assertEqual(calc.get_fps(), 0.0)
        
        calc.reset()
        self.assertEqual(calc.get_fps(), 0.0)
    
    def test_config_classes(self):
        """Test configuration classes."""
        config = AppConfig()
        self.assertIsInstance(config.game_detector, GameDetectorConfig)
        
        config_dict = config.to_dict()
        self.assertIn('game_detector', config_dict)
        
        new_config = AppConfig.from_dict(config_dict)
        self.assertIsInstance(new_config, AppConfig)
    
    def test_utility_functions(self):
        """Test utility functions."""
        # Test board conversion
        board_1d = ['X', 'O', ' '] * 3
        board_2d = convert_board_1d_to_2d(board_1d)
        self.assertEqual(len(board_2d), 3)
        
        # Test symbol counting
        counts = get_board_symbol_counts(board_2d)
        self.assertEqual(counts['X'], 3)
        
        # Test path utilities
        root = get_project_root()
        self.assertTrue(root.endswith('TicTacToe'))


if __name__ == '__main__':
    unittest.main()