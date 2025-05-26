# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Final coverage tests for TicTacToe application.
"""
import numpy as np
from unittest.mock import Mock, patch
import pytest

from app.core.game_state import GameState, EMPTY, PLAYER_X, PLAYER_O
from app.core.strategy import RandomStrategy, MinimaxStrategy
from app.core.config import AppConfig, GameDetectorConfig
from app.core.utils import FPSCalculator
from app.main.game_logic import create_board, get_available_moves, check_winner
from app.main.game_utils import convert_board_1d_to_2d, get_board_symbol_counts
from app.main.path_utils import get_project_root


class TestFinalCoverage:
    
    def test_game_state_properties(self):
        """Test GameState basic properties."""
        gs = GameState()
        assert gs.board == [[EMPTY] * 3 for _ in range(3)]
        assert gs.grid_points is None
        assert gs.detection_results == []
        assert not gs.is_physical_grid_valid()
    
    def test_game_state_symbol_counting(self):
        """Test symbol counting."""
        gs = GameState()
        gs._board_state[0][0] = PLAYER_X
        gs._board_state[1][1] = PLAYER_O
        
        x_count, o_count = gs.count_symbols()
        assert x_count == 1
        assert o_count == 1    
    def test_game_state_validation(self):
        """Test board validation."""
        gs = GameState()
        assert gs.is_valid()
        assert gs.is_valid_turn_sequence()
    
    def test_game_state_error_handling(self):
        """Test error handling."""
        gs = GameState()
        assert not gs.is_error_active()
        
        gs.set_error("Test error")
        assert gs.is_error_active()
        assert gs.get_error() == "Test error"
        
        gs.clear_error_message()
        assert not gs.is_error_active()
    
    def test_fps_calculator(self):
        """Test FPS calculator."""
        calc = FPSCalculator()
        assert calc.buffer_size == 10
        assert calc.get_fps() == 0.0
        
        calc.reset()
        assert calc.get_fps() == 0.0
    
    def test_config_classes(self):
        """Test configuration classes."""
        config = AppConfig()
        assert isinstance(config.game_detector, GameDetectorConfig)
        
        config_dict = config.to_dict()
        assert 'game_detector' in config_dict
        
        new_config = AppConfig.from_dict(config_dict)
        assert isinstance(new_config, AppConfig)
    
    def test_utility_functions(self):
        """Test utility functions."""
        # Test board conversion
        board_1d = ['X', 'O', ' '] * 3
        board_2d = convert_board_1d_to_2d(board_1d)
        assert len(board_2d) == 3
        
        # Test symbol counting
        counts = get_board_symbol_counts(board_2d)
        assert counts['X'] == 3
        
        # Test path utilities
        root = get_project_root()
        assert root.endswith('TicTacToe')
