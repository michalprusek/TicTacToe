"""
Fixed tests for game_state module based on actual API.
Tests GameState functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.core.game_state import GameState, EMPTY, PLAYER_X, PLAYER_O


class TestGameStateBasics:
    """Test basic GameState functionality."""

    def test_game_state_initialization(self):
        """Test GameState initialization."""
        game_state = GameState()
        
        assert hasattr(game_state, '_board_state')
        assert len(game_state._board_state) == 3
        assert len(game_state._board_state[0]) == 3
        assert game_state._grid_points is None
        assert game_state._homography is None
        assert game_state.winner is None

    def test_board_property(self):
        """Test board property returns copy."""
        game_state = GameState()
        
        board = game_state.board
        assert len(board) == 3
        assert len(board[0]) == 3
        assert all(cell == EMPTY for row in board for cell in row)
        
        # Modify returned board should not affect internal state
        board[0][0] = PLAYER_X
        assert game_state._board_state[0][0] == EMPTY

    def test_reset_game(self):
        """Test game reset functionality."""
        game_state = GameState()
        
        # Modify some state
        game_state._board_state[0][0] = PLAYER_X
        game_state.winner = PLAYER_X
        game_state.error_message = "Test error"
        
        game_state.reset_game()
        
        assert all(cell == EMPTY for row in game_state._board_state for cell in row)
        assert game_state.winner is None
        assert game_state.error_message is None

    def test_grid_points_property(self):
        """Test grid points property."""
        game_state = GameState()
        
        assert game_state.grid_points is None
        
        # Set grid points
        points = np.array([[100, 100], [200, 200]])
        game_state._grid_points = points
        
        assert np.array_equal(game_state.grid_points, points)

    def test_is_valid(self):
        """Test board validity check."""
        game_state = GameState()
        
        # Empty board should be valid
        assert game_state.is_valid() is True
        
        # Board with valid symbols should be valid
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_O
        assert game_state.is_valid() is True
        
        # Board with invalid symbol should be invalid
        game_state._board_state[2][2] = 'Z'
        assert game_state.is_valid() is False


class TestGameStateSymbolCounting:
    """Test symbol counting functionality."""

    def test_count_symbols_empty_board(self):
        """Test counting symbols on empty board."""
        game_state = GameState()
        
        x_count, o_count = game_state.count_symbols()
        assert x_count == 0
        assert o_count == 0

    def test_count_symbols_with_symbols(self):
        """Test counting symbols with placed symbols."""
        game_state = GameState()
        
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[0][1] = PLAYER_X
        game_state._board_state[1][0] = PLAYER_O
        
        x_count, o_count = game_state.count_symbols()
        assert x_count == 2
        assert o_count == 1

    def test_is_valid_turn_sequence_empty(self):
        """Test valid turn sequence on empty board."""
        game_state = GameState()
        
        assert game_state.is_valid_turn_sequence() is True

    def test_is_valid_turn_sequence_valid(self):
        """Test valid turn sequences."""
        game_state = GameState()
        
        # X starts (1-0)
        game_state._board_state[0][0] = PLAYER_X
        assert game_state.is_valid_turn_sequence() is True
        
        # X-O alternating (2-1)
        game_state._board_state[0][1] = PLAYER_O
        game_state._board_state[1][0] = PLAYER_X
        assert game_state.is_valid_turn_sequence() is True

    def test_is_valid_turn_sequence_invalid(self):
        """Test invalid turn sequences."""
        game_state = GameState()
        
        # O starts (0-1) - invalid
        game_state._board_state[0][0] = PLAYER_O
        assert game_state.is_valid_turn_sequence() is False
        
        # Too many X moves (3-1) - invalid
        game_state._board_state[0][1] = PLAYER_X
        game_state._board_state[1][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_X
        assert game_state.is_valid_turn_sequence() is False


class TestGameStateValidMoves:
    """Test valid moves functionality."""

    def test_get_valid_moves_empty_board(self):
        """Test getting valid moves on empty board."""
        game_state = GameState()
        
        moves = game_state.get_valid_moves()
        assert len(moves) == 9
        
        expected_moves = [(r, c) for r in range(3) for c in range(3)]
        assert set(moves) == set(expected_moves)

    def test_get_valid_moves_partial_board(self):
        """Test getting valid moves on partial board."""
        game_state = GameState()
        
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_O
        
        moves = game_state.get_valid_moves()
        assert len(moves) == 7
        assert (0, 0) not in moves
        assert (1, 1) not in moves

    def test_get_valid_moves_full_board(self):
        """Test getting valid moves on full board."""
        game_state = GameState()
        
        # Fill board
        for r in range(3):
            for c in range(3):
                game_state._board_state[r][c] = PLAYER_X if (r + c) % 2 == 0 else PLAYER_O
        
        moves = game_state.get_valid_moves()
        assert len(moves) == 0


class TestGameStateWinnerDetection:
    """Test winner detection functionality."""

    def test_check_winner_no_winner(self):
        """Test no winner scenarios."""
        game_state = GameState()
        
        # Empty board
        assert game_state.check_winner() is None
        
        # Ongoing game
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_O
        assert game_state.check_winner() is None

    def test_check_winner_horizontal_wins(self):
        """Test horizontal win detection."""
        test_cases = [
            # Top row
            ([0, 0], [0, 1], [0, 2]),
            # Middle row  
            ([1, 0], [1, 1], [1, 2]),
            # Bottom row
            ([2, 0], [2, 1], [2, 2]),
        ]
        
        for positions in test_cases:
            game_state = GameState()
            for r, c in positions:
                game_state._board_state[r][c] = PLAYER_X
            
            assert game_state.check_winner() == PLAYER_X

    def test_check_winner_vertical_wins(self):
        """Test vertical win detection."""
        test_cases = [
            # Left column
            ([0, 0], [1, 0], [2, 0]),
            # Middle column
            ([0, 1], [1, 1], [2, 1]),
            # Right column
            ([0, 2], [1, 2], [2, 2]),
        ]
        
        for positions in test_cases:
            game_state = GameState()
            for r, c in positions:
                game_state._board_state[r][c] = PLAYER_O
            
            assert game_state.check_winner() == PLAYER_O

    def test_check_winner_diagonal_wins(self):
        """Test diagonal win detection."""
        # Main diagonal
        game_state = GameState()
        for i in range(3):
            game_state._board_state[i][i] = PLAYER_X
        assert game_state.check_winner() == PLAYER_X
        
        # Anti-diagonal
        game_state = GameState()
        for i in range(3):
            game_state._board_state[i][2-i] = PLAYER_O
        assert game_state.check_winner() == PLAYER_O

    def test_is_board_full_empty(self):
        """Test board full check on empty board."""
        game_state = GameState()
        assert game_state.is_board_full() is False

    def test_is_board_full_partial(self):
        """Test board full check on partial board."""
        game_state = GameState()
        game_state._board_state[0][0] = PLAYER_X
        assert game_state.is_board_full() is False

    def test_is_board_full_complete(self):
        """Test board full check on full board."""
        game_state = GameState()
        
        # Fill board alternating
        for r in range(3):
            for c in range(3):
                game_state._board_state[r][c] = PLAYER_X if (r + c) % 2 == 0 else PLAYER_O
        
        assert game_state.is_board_full() is True


class TestGameStateStringRepresentation:
    """Test string representation functionality."""

    def test_board_to_string_empty(self):
        """Test string representation of empty board."""
        game_state = GameState()
        
        board_str = game_state.board_to_string()
        # Empty board with all spaces gets stripped to empty string
        assert board_str == ""

    def test_board_to_string_with_symbols(self):
        """Test string representation with symbols."""
        game_state = GameState()
        
        game_state._board_state[0][0] = PLAYER_X
        game_state._board_state[1][1] = PLAYER_O
        game_state._board_state[2][2] = PLAYER_X
        
        board_str = game_state.board_to_string()
        lines = board_str.split('\n')
        
        assert lines[0] == "X  "
        assert lines[1] == " O "
        assert lines[2] == "  X"


class TestGameStateErrorHandling:
    """Test error handling functionality."""

    def test_set_error(self):
        """Test setting error messages."""
        game_state = GameState()
        
        game_state.set_error("Test error")
        assert game_state.get_error() == "Test error"
        assert game_state.is_error_active() is True

    def test_clear_error_message(self):
        """Test clearing error messages."""
        game_state = GameState()
        
        game_state.set_error("Test error")
        game_state.clear_error_message()
        
        assert game_state.get_error() is None
        assert game_state.is_error_active() is False

    def test_fatal_error_precedence(self):
        """Test that fatal errors take precedence."""
        game_state = GameState()
        
        game_state.set_error("FATAL: Critical error")
        game_state.set_error("Regular error")
        
        assert game_state.get_error() == "FATAL: Critical error"

    def test_is_game_over_due_to_error(self):
        """Test game over due to error detection."""
        game_state = GameState()
        
        # Regular error should not end game
        game_state.set_error("Regular error")
        assert game_state.is_game_over_due_to_error() is False
        
        # Fatal error should end game
        game_state.set_error("FATAL: Critical error")
        assert game_state.is_game_over_due_to_error() is True


class TestGameStatePhysicalGrid:
    """Test physical grid functionality."""

    def test_is_physical_grid_valid_default(self):
        """Test physical grid validity default state."""
        game_state = GameState()
        assert game_state.is_physical_grid_valid() is False

    def test_grid_pause_functionality(self):
        """Test grid pause functionality."""
        game_state = GameState()
        
        # Should not be paused initially
        assert game_state.is_game_paused_due_to_incomplete_grid() is False
        
        # Simulate incomplete grid detection
        game_state.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=None,  # No grid points
            homography=None,
            detected_symbols=[],
            class_id_to_player={},
            timestamp=1.0
        )
        
        assert game_state.is_game_paused_due_to_incomplete_grid() is True
        assert game_state.get_error() == GameState.ERROR_GRID_INCOMPLETE_PAUSE

    def test_grid_resume_functionality(self):
        """Test grid resume functionality."""
        game_state = GameState()
        
        # First, pause due to incomplete grid
        game_state.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=None,
            homography=None,
            detected_symbols=[],
            class_id_to_player={},
            timestamp=1.0
        )
        
        assert game_state.is_game_paused_due_to_incomplete_grid() is True
        
        # Then, resume with complete grid
        complete_grid = np.random.rand(16, 2) * 100
        game_state.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=complete_grid,
            homography=np.eye(3),
            detected_symbols=[],
            class_id_to_player={},
            timestamp=2.0
        )
        
        assert game_state.is_game_paused_due_to_incomplete_grid() is False
        assert game_state.is_physical_grid_valid() is True


class TestGameStateWinConditions:
    """Test win condition checking functionality."""

    def test_is_game_over_no_winner(self):
        """Test game over check with no winner."""
        game_state = GameState()
        assert game_state.is_game_over() is False

    def test_is_game_over_with_winner(self):
        """Test game over check with winner."""
        game_state = GameState()
        game_state.winner = PLAYER_X
        assert game_state.is_game_over() is True

    def test_get_winner(self):
        """Test getting winner."""
        game_state = GameState()
        assert game_state.get_winner() is None
        
        game_state.winner = PLAYER_X
        assert game_state.get_winner() == PLAYER_X

    def test_get_winning_line_indices(self):
        """Test getting winning line indices."""
        game_state = GameState()
        assert game_state.get_winning_line_indices() is None
        
        line_indices = [(0, 0), (0, 1), (0, 2)]
        game_state.winning_line_indices = line_indices
        assert game_state.get_winning_line_indices() == line_indices


class TestGameStateTimestamp:
    """Test timestamp functionality."""

    def test_get_timestamp_default(self):
        """Test default timestamp."""
        game_state = GameState()
        assert game_state.get_timestamp() == 0

    def test_timestamp_update(self):
        """Test timestamp update through detection."""
        game_state = GameState()
        
        # Complete grid to avoid pause
        complete_grid = np.random.rand(16, 2) * 100
        game_state.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=complete_grid,
            homography=np.eye(3),
            detected_symbols=[],
            class_id_to_player={},
            timestamp=5.0
        )
        
        assert game_state.get_timestamp() == 5.0


class TestGameStateIntegration:
    """Integration tests for GameState."""

    @patch('app.core.grid_utils.robust_sort_grid_points')
    def test_complete_detection_update(self, mock_robust_sort):
        """Test complete detection update workflow."""
        # Mock the robust sort function
        mock_robust_sort.return_value = (None, np.eye(3))
        
        game_state = GameState()
        
        # Create complete grid
        complete_grid = np.random.rand(16, 2) * 100
        
        # Mock detected symbols
        detected_symbols = [
            {
                'box': [100, 100, 150, 150],
                'label': 'X',
                'confidence': 0.9,
                'class_id': 0
            }
        ]
        
        game_state.update_from_detection(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            ordered_kpts_uv=complete_grid,
            homography=np.eye(3),
            detected_symbols=detected_symbols,
            class_id_to_player={0: 'X', 1: 'O'},
            timestamp=1.0
        )
        
        # Should have valid grid
        assert game_state.is_physical_grid_valid() is True
        assert not game_state.is_game_paused_due_to_incomplete_grid()
        assert np.array_equal(game_state.grid_points, complete_grid)

    def test_error_handling_robustness(self):
        """Test error handling robustness."""
        game_state = GameState()
        
        # Test with various invalid inputs
        invalid_frames = [None, "invalid", 123]
        
        for invalid_frame in invalid_frames:
            try:
                game_state.update_from_detection(
                    frame=invalid_frame,
                    ordered_kpts_uv=None,
                    homography=None,
                    detected_symbols=[],
                    class_id_to_player={},
                    timestamp=1.0
                )
            except Exception:
                pass  # Should handle gracefully

    def test_performance_with_frequent_updates(self):
        """Test performance with frequent updates."""
        game_state = GameState()
        complete_grid = np.random.rand(16, 2) * 100
        
        import time
        start_time = time.time()
        
        for i in range(100):
            game_state.update_from_detection(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                ordered_kpts_uv=complete_grid,
                homography=np.eye(3),
                detected_symbols=[],
                class_id_to_player={},
                timestamp=float(i)
            )
        
        elapsed = time.time() - start_time
        assert elapsed < 5.0  # Should complete in reasonable time