"""
Extended tests for GameState class.
"""
import unittest
import numpy as np
from unittest.mock import Mock
from app.core.game_state import GameState, EMPTY, PLAYER_X, PLAYER_O


class TestGameStateExtended(unittest.TestCase):
    
    def setUp(self):
        self.game_state = GameState()
    
    def test_board_state_modification(self):
        """Test direct board state modifications."""
        self.game_state._board_state[1][1] = PLAYER_X
        board = self.game_state.board
        self.assertEqual(board[1][1], PLAYER_X)
        self.assertEqual(board[0][0], EMPTY)
        self.assertEqual(board[2][2], EMPTY)
    
    def test_grid_points_setting(self):
        """Test setting and getting grid points."""
        test_points = np.random.random((16, 2)).astype(np.float32)
        self.game_state._grid_points = test_points
        
        retrieved_points = self.game_state.grid_points
        np.testing.assert_array_equal(retrieved_points, test_points)
        
        self.game_state._grid_points = None
        self.assertIsNone(self.game_state.grid_points)
    
    def test_detection_results_management(self):
        """Test detection results storage and retrieval."""
        self.assertEqual(len(self.game_state.detection_results), 0)
        
        mock_result1 = Mock()
        mock_result1.confidence = 0.95
        self.game_state._detection_results = [mock_result1]
        results = self.game_state.detection_results
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].confidence, 0.95)
    
    def test_changed_cells_tracking(self):
        """Test tracking of changed cells."""
        self.assertEqual(len(self.game_state.changed_cells_this_turn), 0)
        test_cells = [(0, 1), (2, 2)]
        self.game_state._changed_cells_this_turn = test_cells
        self.assertEqual(self.game_state.changed_cells_this_turn, test_cells)
    
    def test_error_message_handling(self):
        """Test error message functionality."""
        self.assertIsNone(self.game_state.error_message)
        
        self.game_state.error_message = "Test error"
        self.assertEqual(self.game_state.error_message, "Test error")
        
        self.game_state.reset_game()
        self.assertIsNone(self.game_state.error_message)
    
    def test_game_pause_state(self):
        """Test game pause state due to incomplete grid."""
        self.assertFalse(self.game_state.game_paused_due_to_incomplete_grid)
        
        self.game_state.game_paused_due_to_incomplete_grid = True
        self.assertTrue(self.game_state.game_paused_due_to_incomplete_grid)
        
        self.game_state.reset_game()
        self.assertFalse(self.game_state.game_paused_due_to_incomplete_grid)
    
    def test_grid_visibility_tracking(self):
        """Test grid visibility status."""
        self.assertFalse(self.game_state.grid_fully_visible)
        self.assertEqual(self.game_state.missing_grid_points_count, 0)
        
        self.game_state.grid_fully_visible = True
        self.game_state.missing_grid_points_count = 3
        
        self.assertTrue(self.game_state.grid_fully_visible)
        self.assertEqual(self.game_state.missing_grid_points_count, 3)
        
        self.game_state.reset_game()
        self.assertFalse(self.game_state.grid_fully_visible)
        self.assertEqual(self.game_state.missing_grid_points_count, 0)
    
    def test_winner_and_winning_line(self):
        """Test winner and winning line tracking."""
        self.assertIsNone(self.game_state.winner)
        self.assertIsNone(self.game_state.winning_line_indices)
        
        self.game_state.winner = PLAYER_X
        self.game_state.winning_line_indices = [(0, 0), (0, 1), (0, 2)]
        
        self.assertEqual(self.game_state.winner, PLAYER_X)
        self.assertEqual(self.game_state.winning_line_indices, [(0, 0), (0, 1), (0, 2)])
        
        self.game_state.reset_game()
        self.assertIsNone(self.game_state.winner)
        self.assertIsNone(self.game_state.winning_line_indices)


if __name__ == '__main__':
    unittest.main()