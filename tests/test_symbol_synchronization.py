# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Test symbol synchronization logic in GameState.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from app.core.game_state import GameState, EMPTY


class TestSymbolSynchronization:
    """Test symbol synchronization with YOLO detections."""

    def setup_method(self):
        """Set up test fixtures."""
        self.game_state = GameState()
        self.game_state.logger = Mock()
        
        # Set up mock cell centers for testing
        self.game_state._cell_centers_uv_transformed = np.array([
            [100, 100], [200, 100], [300, 100],  # Row 0
            [100, 200], [200, 200], [300, 200],  # Row 1
            [100, 300], [200, 300], [300, 300]   # Row 2
        ], dtype=np.float32)
        
        self.class_id_to_player = {0: 'X', 1: 'O'}

    def test_empty_detections_clears_board(self):
        """Test that empty detections clear the board."""
        # Set up initial board state with some symbols
        self.game_state._board_state = [
            ['X', 'O', EMPTY],
            [EMPTY, 'X', EMPTY],
            [EMPTY, EMPTY, 'O']
        ]
        
        # Call synchronization with empty detections
        changed_cells = self.game_state._synchronize_board_with_detections(
            [], self.class_id_to_player
        )
        
        # Board should be cleared
        expected_board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        assert self.game_state._board_state == expected_board
        
        # Should return all previously occupied cells as changed
        assert len(changed_cells) == 4  # X, O, X, O were removed

    def test_low_confidence_symbols_clear_board(self):
        """Test that low confidence symbols clear the board."""
        # Set up initial board state
        self.game_state._board_state = [
            ['X', EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        
        # Create low confidence detection
        detected_symbols = [{
            'label': 'X',
            'confidence': 0.5,  # Below default threshold of 0.90
            'box': [90, 90, 110, 110],
            'class_id': 0
        }]
        
        # Call synchronization
        changed_cells = self.game_state._synchronize_board_with_detections(
            detected_symbols, self.class_id_to_player
        )
        
        # Board should be cleared due to low confidence
        expected_board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        assert self.game_state._board_state == expected_board

    def test_high_confidence_symbols_placed(self):
        """Test that high confidence symbols are placed correctly."""
        # Start with empty board
        self.game_state._board_state = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
        
        # Create high confidence detections
        detected_symbols = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [90, 90, 110, 110],  # Near cell (0,0)
                'class_id': 0
            },
            {
                'label': 'O', 
                'confidence': 0.92,
                'box': [190, 190, 210, 210],  # Near cell (1,1)
                'class_id': 1
            }
        ]
        
        # Call synchronization
        changed_cells = self.game_state._synchronize_board_with_detections(
            detected_symbols, self.class_id_to_player
        )
        
        # Check that symbols were placed correctly
        assert self.game_state._board_state[0][0] == 'X'
        assert self.game_state._board_state[1][1] == 'O'
        
        # Check changed cells
        assert (0, 0) in changed_cells
        assert (1, 1) in changed_cells

    def test_board_rebuilds_from_current_detections(self):
        """Test that board is rebuilt from current detections only."""
        # Set up initial board with symbols
        self.game_state._board_state = [
            ['X', 'O', EMPTY],
            [EMPTY, 'X', EMPTY],
            [EMPTY, EMPTY, 'O']
        ]
        
        # Provide detections for only some of the existing symbols
        detected_symbols = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [90, 90, 110, 110],  # Only X at (0,0)
                'class_id': 0
            }
        ]
        
        # Call synchronization
        changed_cells = self.game_state._synchronize_board_with_detections(
            detected_symbols, self.class_id_to_player
        )
        
        # Board should only contain the detected symbol
        expected_board = [
            ['X', EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        assert self.game_state._board_state == expected_board
        
        # Should report changes for removed symbols
        assert len(changed_cells) >= 3  # O, X, O were removed

    def test_get_changed_cells_helper(self):
        """Test the helper method for detecting changed cells."""
        old_board = [
            ['X', 'O', EMPTY],
            [EMPTY, 'X', EMPTY],
            [EMPTY, EMPTY, 'O']
        ]
        
        new_board = [
            ['X', EMPTY, EMPTY],  # O removed at (0,1)
            [EMPTY, 'X', 'O'],    # O moved from (2,2) to (1,2)
            [EMPTY, EMPTY, EMPTY] # O removed at (2,2)
        ]
        
        changed_cells = self.game_state._get_changed_cells(old_board, new_board)
        
        # Should detect all changed positions
        expected_changes = {(0, 1), (1, 2), (2, 2)}
        assert set(changed_cells) == expected_changes

    def test_confidence_threshold_respected(self):
        """Test that symbol confidence threshold is respected."""
        # Set custom confidence threshold
        self.game_state.symbol_confidence_threshold = 0.8
        
        # Create detections with different confidence levels
        detected_symbols = [
            {
                'label': 'X',
                'confidence': 0.85,  # Above threshold
                'box': [90, 90, 110, 110],
                'class_id': 0
            },
            {
                'label': 'O',
                'confidence': 0.75,  # Below threshold
                'box': [190, 190, 210, 210],
                'class_id': 1
            }
        ]
        
        # Call synchronization
        self.game_state._synchronize_board_with_detections(
            detected_symbols, self.class_id_to_player
        )
        
        # Only high confidence symbol should be placed
        assert self.game_state._board_state[0][0] == 'X'
        assert self.game_state._board_state[1][1] == EMPTY  # O rejected
