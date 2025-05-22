"""
Unified tests for game end check functionality in TicTacToeApp.

This module consolidates previously duplicated test code for game end checking
from the following files:
- test_pyqt_gui_check_game_end.py
- test_pyqt_gui_app_simple.py
- test_pyqt_gui_comprehensive.py
- test_pyqt_gui_drawing.py
"""
import pytest
from unittest.mock import patch, MagicMock

from app.main import game_logic
from tests.conftest_common import MockTicTacToeApp, qt_app, PyQtGuiTestCaseBase, GameEndCheckTestUtils


@pytest.fixture
def app():
    """Fixture that creates a properly configured mock app."""
    # Create the app
    app, patches = PyQtGuiTestCaseBase.create_test_app()

    # Configure the app for game end tests
    app.human_player = game_logic.PLAYER_X
    app.ai_player = game_logic.PLAYER_O
    app.current_turn = game_logic.PLAYER_X
    app.game_over = False
    app.winner = None
    app.draw_winning_line = MagicMock()

    # Reset mock calls
    app.status_label.setText.reset_mock()
    app.board_widget.update.reset_mock()

    yield app
    
    # Stop all patches
    for p in patches:
        p.stop()


@pytest.mark.skip(reason="PyQt initialization causes segmentation fault")
class TestGameEndCheck:
    """Tests for game end checking in TicTacToeApp."""

    def test_check_game_end_no_winner(self, app):
        """Test check_game_end method with no winner."""
        GameEndCheckTestUtils.test_check_game_end_no_winner(app)

    def test_check_game_end_human_wins(self, app):
        """Test check_game_end method with human winning."""
        GameEndCheckTestUtils.test_check_game_end_human_wins(app)

    def test_check_game_end_ai_wins(self, app):
        """Test check_game_end method with AI winning."""
        GameEndCheckTestUtils.test_check_game_end_ai_wins(app)

    def test_check_game_end_tie(self, app):
        """Test check_game_end method with tie."""
        GameEndCheckTestUtils.test_check_game_end_tie(app)

    def test_check_game_end_already_over(self, app):
        """Test check_game_end method when game is already over."""
        GameEndCheckTestUtils.test_check_game_end_already_over(app)

    def test_check_game_end_with_ai_win_drawing_issue(self, app):
        """Test check_game_end with AI win but drawing fails."""
        # Set up board with AI winning
        app.board_widget.board = [
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, 0],
            [game_logic.PLAYER_O, 0, 0]
        ]

        # Mock draw_winning_line to fail
        app.draw_winning_line.return_value = False

        # Setup winning line return value
        with patch('app.main.game_logic.get_winning_line', return_value=[(0, 0), (1, 0), (2, 0)]):
            # Call check_game_end
            app.check_game_end()

            # Check that game_over is True despite drawing failure
            assert app.game_over
            assert app.winner == game_logic.PLAYER_O

            # Draw was attempted but failed
            app.draw_winning_line.assert_called_once()

    def test_check_game_end_with_invalid_winning_line(self, app):
        """Test check_game_end when get_winning_line returns invalid line."""
        # Set up board with human winning
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, 0],
            [0, 0, 0]
        ]

        # Setup invalid winning line return value
        with patch('app.main.game_logic.get_winning_line', return_value=None):
            # Call check_game_end
            app.check_game_end()

            # Game should still be marked as over
            assert app.game_over
            assert app.winner == game_logic.PLAYER_X

            # Status should be updated for win
            app.status_label.setText.assert_called_once()

            # But winning line drawing shouldn't be attempted with invalid line
            assert app.board_widget.winning_line is None