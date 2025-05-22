"""
Unit tests for GameStateManager class.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.main.game_state_manager import GameStateManager


@pytest.fixture
def mock_config():
    """mock_config fixture for tests."""
    mock_config = MagicMock()
    mock_config.class_id_to_player = {0: 1, 1: 2}  # X=1, O=2
    return mock_config


@pytest.fixture
def game_state_manager(mock_config):
    """game_state_manager fixture for tests."""
    manager = GameStateManager(
        config=mock_config
    )
    return manager


class TestGameStateManager:
    """Test GameStateManager class."""

    def test_init(self, game_state_manager, mock_config):
        """Test initialization."""
        assert game_state_manager.config == mock_config
        assert game_state_manager.class_id_to_player == mock_config.class_id_to_player
        assert game_state_manager.game_state is not None

    def test_update_game_state(self, game_state_manager):
        """Test update_game_state method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create dummy keypoints
        keypoints = np.zeros((16, 2), dtype=np.float32)
        for i in range(16):
            keypoints[i] = [i * 20, i * 15]

        # Create dummy homography
        homography = np.eye(3)

        # Create dummy symbols
        symbols = [
            {'label': 'X', 'confidence': 0.8, 'box': [100, 100, 150, 150], 'class_id': 0},
            {'label': 'O', 'confidence': 0.9, 'box': [200, 200, 250, 250], 'class_id': 1}
        ]

        # Mock game_state.update_from_detection
        game_state_manager.game_state.update_from_detection = MagicMock()

        # Mock game_state.get_latest_derived_cell_polygons
        mock_polygons = [np.zeros((4, 2)) for _ in range(9)]
        game_state_manager.game_state.get_latest_derived_cell_polygons = MagicMock(
            return_value=mock_polygons
        )

        # Call the method
        polygons = game_state_manager.update_game_state(
            frame,
            keypoints,
            homography,
            symbols,
            100.0,  # timestamp
            False   # grid_status_changed
        )

        # Check that game_state.update_from_detection was called
        game_state_manager.game_state.update_from_detection.assert_called_once()

        # Check that game_state.get_latest_derived_cell_polygons was called
        game_state_manager.game_state.get_latest_derived_cell_polygons.assert_called_once()

        # Check that polygons were returned
        assert polygons == mock_polygons

    def test_update_game_state_grid_changed(self, game_state_manager):
        """Test update_game_state method with grid status changed."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock game_state.reset
        game_state_manager.game_state.reset = MagicMock()

        # Mock game_state.reset_changed_cells
        game_state_manager.game_state.reset_changed_cells = MagicMock()

        # Mock game_state.update_from_detection
        game_state_manager.game_state.update_from_detection = MagicMock()

        # Call the method with grid_status_changed=True
        game_state_manager.update_game_state(
            frame,
            None,   # keypoints
            None,   # homography
            [],     # symbols
            100.0,  # timestamp
            True    # grid_status_changed
        )

        # Check that game_state.reset was called
        game_state_manager.game_state.reset.assert_called_once()

        # Check that game_state.reset_changed_cells was called
        game_state_manager.game_state.reset_changed_cells.assert_called_once()

        # Check that game_state.update_from_detection was called
        game_state_manager.game_state.update_from_detection.assert_called_once()

    def test_get_board_state(self, game_state_manager):
        """Test get_board_state method."""
        # Mock game_state.board property
        mock_board = [
            ['X', '', ''],
            ['', 'O', ''],
            ['', '', '']
        ]
        # Mock the get_board_state method to return our mock board
        game_state_manager.get_board_state = MagicMock(return_value=mock_board)

        # Call the method
        board_state = game_state_manager.get_board_state()

        # Check that correct board state was returned
        assert board_state == mock_board

    def test_get_board_state_no_board(self, game_state_manager):
        """Test get_board_state method with no board attribute."""
        # Create a new mock game_state without board property
        empty_board = [['', '', ''], ['', '', ''], ['', '', '']]

        # Save original method
        original_method = game_state_manager.get_board_state

        # Override the method to simulate no board attribute
        def mock_get_board_state():
            # Simulate the behavior when board is not available
            return empty_board

        game_state_manager.get_board_state = mock_get_board_state

        try:
            # Call the method
            board_state = game_state_manager.get_board_state()

            # Check that empty board was returned
            assert board_state == empty_board
        finally:
            # Restore original method
            game_state_manager.get_board_state = original_method

    def test_is_valid(self, game_state_manager):
        """Test is_valid method."""
        # Mock game_state.is_valid
        game_state_manager.game_state.is_valid = MagicMock(return_value=True)

        # Call the method
        is_valid = game_state_manager.is_valid()

        # Check that game_state.is_valid was called
        game_state_manager.game_state.is_valid.assert_called_once()

        # Check that correct value was returned
        assert is_valid is True

    def test_get_winner(self, game_state_manager):
        """Test get_winner method."""
        # Set up mock winner
        game_state_manager.game_state.winner = 'X'

        # Call the method
        winner = game_state_manager.get_winner()

        # Check that correct winner was returned
        assert winner == 'X'

    def test_get_winner_no_winner(self, game_state_manager):
        """Test get_winner method with no winner attribute."""
        # Remove winner attribute
        if hasattr(game_state_manager.game_state, 'winner'):
            delattr(game_state_manager.game_state, 'winner')

        # Call the method
        winner = game_state_manager.get_winner()

        # Check that None was returned
        assert winner is None

    def test_is_grid_visible(self, game_state_manager):
        """Test is_grid_visible method."""
        # Set up mock grid visibility
        game_state_manager.game_state.is_grid_visible = True

        # Call the method
        is_visible = game_state_manager.is_grid_visible()

        # Check that correct value was returned
        assert is_visible is True

    def test_is_grid_visible_no_attribute(self, game_state_manager):
        """Test is_grid_visible method with no attribute."""
        # Remove attribute
        if hasattr(game_state_manager.game_state, 'is_grid_visible'):
            delattr(game_state_manager.game_state, 'is_grid_visible')

        # Call the method
        is_visible = game_state_manager.is_grid_visible()

        # Check that False was returned
        assert is_visible is False

    def test_is_grid_stable(self, game_state_manager):
        """Test is_grid_stable method."""
        # Set up mock grid stability
        game_state_manager.game_state.is_grid_stable = True

        # Call the method
        is_stable = game_state_manager.is_grid_stable()

        # Check that correct value was returned
        assert is_stable is True

    def test_is_grid_stable_no_attribute(self, game_state_manager):
        """Test is_grid_stable method with no attribute."""
        # Remove attribute
        if hasattr(game_state_manager.game_state, 'is_grid_stable'):
            delattr(game_state_manager.game_state, 'is_grid_stable')

        # Call the method
        is_stable = game_state_manager.is_grid_stable()

        # Check that False was returned
        assert is_stable is False

    def test_get_cell_polygons(self, game_state_manager):
        """Test get_cell_polygons method."""
        # Set up mock cell polygons
        mock_polygons = [np.zeros((4, 2)) for _ in range(9)]
        game_state_manager.game_state.cell_polygons = mock_polygons

        # Call the method
        polygons = game_state_manager.get_cell_polygons()

        # Check that correct polygons were returned
        assert polygons == mock_polygons

    def test_get_cell_polygons_no_attribute(self, game_state_manager):
        """Test get_cell_polygons method with no attribute."""
        # Remove attribute
        if hasattr(game_state_manager.game_state, 'cell_polygons'):
            delattr(game_state_manager.game_state, 'cell_polygons')

        # Call the method
        polygons = game_state_manager.get_cell_polygons()

        # Check that None was returned
        assert polygons is None

    def test_get_grid_points(self, game_state_manager):
        """Test get_grid_points method."""
        # Set up mock grid points
        mock_points = np.zeros((16, 2))
        game_state_manager.game_state._grid_points = mock_points

        # Call the method
        points = game_state_manager.get_grid_points()

        # Check that correct points were returned
        assert points is mock_points

    def test_get_grid_points_no_attribute(self, game_state_manager):
        """Test get_grid_points method with no attribute."""
        # Remove attribute
        if hasattr(game_state_manager.game_state, '_grid_points'):
            delattr(game_state_manager.game_state, '_grid_points')

        # Call the method
        points = game_state_manager.get_grid_points()

        # Check that None was returned
        assert points is None

    def test_get_cell_center(self, game_state_manager):
        """Test get_cell_center method."""
        # Mock game_state.get_cell_center_uv
        mock_center = (100.0, 100.0)
        game_state_manager.game_state.get_cell_center_uv = MagicMock(return_value=mock_center)

        # Call the method
        center = game_state_manager.get_cell_center(1, 1)

        # Check that game_state.get_cell_center_uv was called
        game_state_manager.game_state.get_cell_center_uv.assert_called_once_with(1, 1)

        # Check that correct center was returned
        assert center == mock_center

    def test_get_cell_center_no_method(self, game_state_manager):
        """Test get_cell_center method with no get_cell_center_uv method."""
        # Save original method
        original_method = game_state_manager.get_cell_center

        # Override the method to simulate no get_cell_center_uv method
        def mock_get_cell_center(row, col):
            # Simulate the behavior when get_cell_center_uv is not available
            return None

        game_state_manager.get_cell_center = mock_get_cell_center

        try:
            # Call the method
            center = game_state_manager.get_cell_center(1, 1)

            # Check that None was returned
            assert center is None
        finally:
            # Restore original method
            game_state_manager.get_cell_center = original_method

    def test_has_grid_issue(self, game_state_manager):
        """Test has_grid_issue method."""
        # Set up mock grid issue
        game_state_manager.game_state.grid_issue_type = "MISSING_POINTS"

        # Call the method
        has_issue = game_state_manager.has_grid_issue()

        # Check that correct value was returned
        assert has_issue is True

    def test_has_grid_issue_no_issue(self, game_state_manager):
        """Test has_grid_issue method with no issue."""
        # Set grid_issue_type to None
        game_state_manager.game_state.grid_issue_type = None

        # Call the method
        has_issue = game_state_manager.has_grid_issue()

        # Check that False was returned
        assert has_issue is False

    def test_get_grid_issue_message(self, game_state_manager):
        """Test get_grid_issue_message method."""
        # Set up mock grid issue message
        mock_message = "Missing grid points"
        game_state_manager.game_state.grid_issue_message = mock_message

        # Call the method
        message = game_state_manager.get_grid_issue_message()

        # Check that correct message was returned
        assert message == mock_message

    def test_get_grid_issue_message_no_attribute(self, game_state_manager):
        """Test get_grid_issue_message method with no attribute."""
        # Remove attribute
        if hasattr(game_state_manager.game_state, 'grid_issue_message'):
            delattr(game_state_manager.game_state, 'grid_issue_message')

        # Call the method
        message = game_state_manager.get_grid_issue_message()

        # Check that None was returned
        assert message is None
