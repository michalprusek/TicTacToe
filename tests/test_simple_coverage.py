# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Simple tests to improve coverage for key modules.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.core.game_state import GameState
from app.core.utils import FPSCalculator
from app.main.game_logic import print_board, EMPTY, PLAYER_X, PLAYER_O
from app.main.game_utils import convert_board_1d_to_2d, get_board_symbol_counts
from app.main.path_utils import get_project_root, get_weights_path


class TestSimpleCoverage:
    """Simple tests to improve coverage."""

    def test_game_state_properties(self):
        """Test GameState property methods."""
        gs = GameState()

        # Test all property getters
        assert gs.board == [[' '] * 3 for _ in range(3)]
        assert gs.grid_points is None
        assert gs.detection_results == []
        assert gs.changed_cells_this_turn == []
        assert gs.is_physical_grid_valid() is False
        assert gs.is_game_paused_due_to_incomplete_grid() is False
        assert gs.get_winner() is None
        assert gs.get_winning_line_indices() is None
        assert gs.get_error() is None
        assert gs.is_error_active() is False
        assert gs.get_timestamp() == 0
        assert gs.get_homography() is None

        # Test reset changed cells
        gs.reset_changed_cells()
        assert gs.changed_cells_this_turn == []
    def test_game_state_validation(self):
        """Test GameState validation methods."""
        gs = GameState()

        # Test valid board
        assert gs.is_valid() is True

        # Test turn sequence validation
        assert gs.is_valid_turn_sequence() is True

        # Add a symbol and test
        gs._board_state[0][0] = PLAYER_X
        assert gs.is_valid_turn_sequence() is True

        # Test symbol counting
        x_count, o_count = gs.count_symbols()
        assert x_count == 1
        assert o_count == 0

    def test_fps_calculator_basic(self):
        """Test FPSCalculator basic functionality."""
        calc = FPSCalculator()

        # Test initialization
        assert calc.buffer_size == 10
        assert calc.get_fps() == 0.0

        # Test reset
        calc.reset()
        assert calc.get_fps() == 0.0

    def test_path_utilities(self):
        """Test path utility functions."""
        root = get_project_root()
        assert str(root).endswith('prusemic')

        weights = get_weights_path()
        assert weights.name == 'weights'

    def test_game_utils_simple(self):
        """Test game utility functions."""
        # Test 1D to 2D conversion
        board_1d = ['X', 'O', ' ', 'X', 'O', ' ', ' ', ' ', 'X']
        board_2d = convert_board_1d_to_2d(board_1d)
        assert len(board_2d) == 3
        assert len(board_2d[0]) == 3

        # Test symbol counting
        counts = get_board_symbol_counts(board_2d)
        assert counts['X'] == 3
        assert counts['O'] == 2
        assert counts[' '] == 4
