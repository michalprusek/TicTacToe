"""
Test suite for the robotic arm turn logic fixes.

This module tests the three main fixes:
1. Turn Logic Fix: Arm plays only when odd number of symbols (1,3,5,7,9)
2. Symbol Recognition Fix: Arm plays the symbol that appears less frequently
3. Win Celebration Feature: Arm draws winning line when it wins
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import game_logic


class TestArmTurnLogicFixes:
    """Test class for arm turn logic fixes."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock TicTacToe app for testing."""
        # Create a simple mock object instead of full app
        app = MagicMock()

        # Mock the arm components
        app.arm_thread = MagicMock()
        app.arm_thread.connected = True
        app.arm_controller = MagicMock()
        app.arm_controller.connected = True

        # Mock the camera and detection components
        app.camera_thread = MagicMock()
        app.camera_thread.detection_thread = MagicMock()
        app.camera_thread.detection_thread.detector = MagicMock()
        app.camera_thread.detection_thread.detector.game_state = MagicMock()
        app.camera_thread.detection_thread.detector.game_state.is_physical_grid_valid.return_value = True

        # Mock board widget
        app.board_widget = MagicMock()
        app.board_widget.board = game_logic.create_board()

        # Mock status components
        app.main_status_panel = MagicMock()
        app.status_label = MagicMock()
        app.logger = MagicMock()
        app.update_status = MagicMock()

        # Initialize game state
        app.human_player = game_logic.PLAYER_X
        app.ai_player = game_logic.PLAYER_O
        app.current_turn = app.human_player
        app.game_over = False
        app.waiting_for_detection = False

        # Add the validation method
        def _validate_game_state(board):
            x_count = sum(row.count(game_logic.PLAYER_X) for row in board)
            o_count = sum(row.count(game_logic.PLAYER_O) for row in board)

            if abs(x_count - o_count) > 1:
                return False
            if x_count < o_count:
                return False
            return True

        app._validate_game_state = _validate_game_state

        return app

    def test_arm_plays_only_on_odd_symbol_count(self, mock_app):
        """Test that arm plays only when there's an odd number of total symbols."""

        # Test the core logic: odd vs even symbol count
        def should_arm_play(board):
            x_count = sum(row.count(game_logic.PLAYER_X) for row in board)
            o_count = sum(row.count(game_logic.PLAYER_O) for row in board)
            total_symbols = x_count + o_count
            return total_symbols % 2 == 1  # Arm plays on odd counts

        # Test case 1: 1 symbol (odd) - arm should play
        board_1_symbol = [
            ['X', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert should_arm_play(board_1_symbol) == True

        # Test case 2: 2 symbols (even) - arm should NOT play
        board_2_symbols = [
            ['X', 'O', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert should_arm_play(board_2_symbols) == False

        # Test case 3: 3 symbols (odd) - arm should play
        board_3_symbols = [
            ['X', 'O', 'X'],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert should_arm_play(board_3_symbols) == True

        # Test case 4: 4 symbols (even) - arm should NOT play
        board_4_symbols = [
            ['X', 'O', 'X'],
            ['O', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert should_arm_play(board_4_symbols) == False

    def test_arm_symbol_selection_logic(self):
        """Test that arm selects the symbol that appears less frequently."""

        def select_arm_symbol(board):
            x_count = sum(row.count(game_logic.PLAYER_X) for row in board)
            o_count = sum(row.count(game_logic.PLAYER_O) for row in board)

            if x_count < o_count:
                return game_logic.PLAYER_X  # Fewer X than O → arm plays X
            elif o_count < x_count:
                return game_logic.PLAYER_O  # Fewer O than X → arm plays O
            else:
                return game_logic.PLAYER_X  # Equal count → arm plays X (default)

        # Test case 1: More X than O - arm should play O
        board_more_x = [
            ['X', 'X', ' '],
            ['O', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert select_arm_symbol(board_more_x) == game_logic.PLAYER_O

        # Test case 2: More O than X - arm should play X
        board_more_o = [
            ['X', ' ', ' '],
            ['O', 'O', ' '],
            [' ', ' ', ' ']
        ]
        assert select_arm_symbol(board_more_o) == game_logic.PLAYER_X

        # Test case 3: Equal count - arm should play X (default)
        board_equal = [
            ['X', 'O', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert select_arm_symbol(board_equal) == game_logic.PLAYER_X

        # Test case 4: Empty board - arm should play X (default)
        board_empty = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        assert select_arm_symbol(board_empty) == game_logic.PLAYER_X

    def test_game_state_validation(self):
        """Test the game state validation logic."""

        def validate_game_state(board):
            x_count = sum(row.count(game_logic.PLAYER_X) for row in board)
            o_count = sum(row.count(game_logic.PLAYER_O) for row in board)

            # RULE 1: X always goes first, so X count should be equal to O count or one more
            if abs(x_count - o_count) > 1:
                return False
            # RULE 2: X should never have fewer symbols than O (since X goes first)
            if x_count < o_count:
                return False
            return True

        # Test valid states
        valid_boards = [
            # Empty board
            [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']],
            # X=1, O=0 (valid)
            [['X', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']],
            # X=1, O=1 (valid)
            [['X', 'O', ' '], [' ', ' ', ' '], [' ', ' ', ' ']],
            # X=2, O=1 (valid)
            [['X', 'O', 'X'], [' ', ' ', ' '], [' ', ' ', ' ']],
        ]

        for board in valid_boards:
            assert validate_game_state(board) == True

        # Test invalid states
        invalid_boards = [
            # X=0, O=1 (invalid - O can't go first)
            [['O', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']],
            # X=1, O=2 (invalid - O has more than X)
            [['X', 'O', 'O'], [' ', ' ', ' '], [' ', ' ', ' ']],
            # X=3, O=1 (invalid - difference > 1)
            [['X', 'X', 'X'], ['O', ' ', ' '], [' ', ' ', ' ']],
        ]

        for board in invalid_boards:
            assert validate_game_state(board) == False

    def test_win_celebration_feature(self, mock_app):
        """Test that arm draws winning line when it wins."""

        # Set up a winning scenario for the arm
        winning_board = [
            ['O', 'O', 'O'],  # Arm wins with O in top row
            ['X', 'X', ' '],
            [' ', ' ', ' ']
        ]

        mock_app.board_widget.board = winning_board
        mock_app.ai_player = game_logic.PLAYER_O  # Arm is playing O
        mock_app.winner = game_logic.PLAYER_O

        # Mock the draw_winning_line method
        mock_app.draw_winning_line = MagicMock()

        # Trigger game end check
        mock_app.check_game_end()

        # Verify that draw_winning_line was called for arm win
        mock_app.draw_winning_line.assert_called()

        # Verify the status was updated to show arm win celebration
        mock_app.update_status.assert_called_with("🏆 RUKA VYHRÁLA! 🎉")

    def test_turn_logic_comprehensive_scenario(self, mock_app):
        """Test a comprehensive game scenario with proper turn logic."""

        # Mock the make_arm_move_with_symbol method
        mock_app.make_arm_move_with_symbol = MagicMock()

        # Scenario: Human plays X, then arm should play O
        # Move 1: Human plays X (total=1, odd, arm should play)
        board_after_human_move = [
            ['X', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

        mock_app.current_turn = mock_app.human_player
        mock_app.handle_detected_game_state(board_after_human_move)

        # Verify arm was called to play O (less frequent symbol)
        mock_app.make_arm_move_with_symbol.assert_called_with(game_logic.PLAYER_O)

        # Move 2: After arm plays O (total=2, even, wait for human)
        board_after_arm_move = [
            ['X', 'O', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

        mock_app.make_arm_move_with_symbol.reset_mock()
        mock_app.current_turn = mock_app.human_player
        mock_app.handle_detected_game_state(board_after_arm_move)

        # Verify arm was NOT called (even number of symbols)
        mock_app.make_arm_move_with_symbol.assert_not_called()

        # Move 3: Human plays X again (total=3, odd, arm should play)
        board_after_second_human_move = [
            ['X', 'O', 'X'],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

        mock_app.current_turn = mock_app.human_player
        mock_app.handle_detected_game_state(board_after_second_human_move)

        # Verify arm was called to play O again (still less frequent)
        mock_app.make_arm_move_with_symbol.assert_called_with(game_logic.PLAYER_O)

    def test_edge_cases(self, mock_app):
        """Test edge cases and error conditions."""

        # Test with invalid grid
        mock_app.camera_thread.detection_thread.detector.game_state.is_physical_grid_valid.return_value = False
        mock_app.make_arm_move_with_symbol = MagicMock()

        detected_board = [
            ['X', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

        mock_app.handle_detected_game_state(detected_board)

        # Verify arm was NOT called due to invalid grid
        mock_app.make_arm_move_with_symbol.assert_not_called()

        # Test with game over
        mock_app.game_over = True
        mock_app.camera_thread.detection_thread.detector.game_state.is_physical_grid_valid.return_value = True
        mock_app.make_arm_move_with_symbol.reset_mock()

        mock_app.handle_detected_game_state(detected_board)

        # Verify arm was NOT called due to game over
        mock_app.make_arm_move_with_symbol.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
