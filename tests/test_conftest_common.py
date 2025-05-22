"""
Tests for conftest_common utilities.
"""
import pytest
from unittest.mock import MagicMock
from app.main import game_logic
from tests.conftest_common import (
    MockTicTacToeApp,
    AssertionUtils,
    GameEndCheckTestUtils,
    DrawingTestUtils,
    CameraTestUtils,
    UIComponentTestUtils
)

@pytest.mark.skip(reason="PyQt initialization causes segmentation fault")
def test_mock_tic_tac_toe_app():
    """Test creating MockTicTacToeApp."""
    app = MockTicTacToeApp()
    assert app is not None
    assert hasattr(app, 'board_widget')
    assert hasattr(app, 'status_label')
    assert hasattr(app, 'strategy_selector')

def test_assertion_utils():
    """Test AssertionUtils."""
    app = MagicMock()
    app.status_label = MagicMock()

    # Call setText first to make the assertion valid
    app.status_label.setText("Test message")

    # Test assertions
    AssertionUtils.assert_status_message(app, "Test message")

    app.status_label.setText.reset_mock()
    app.status_label.setText("Test message")
    AssertionUtils.assert_status_message_once(app, "Test message")

def test_camera_test_utils():
    """Test CameraTestUtils."""
    # Test creating a sample frame
    frame = CameraTestUtils.create_sample_frame()
    assert frame.shape == (480, 640, 3)
    
    # Test preparing a mock camera thread
    app = MagicMock()
    CameraTestUtils.prepare_mock_camera_thread(app)
    assert app.camera_thread is not None

def test_ui_component_test_utils():
    """Test UIComponentTestUtils."""
    # Test creating UI component mocks
    board_widget = UIComponentTestUtils.create_board_widget_mock()
    assert board_widget is not None
    assert hasattr(board_widget, 'board')
    
    status_label = UIComponentTestUtils.create_status_label_mock()
    assert status_label is not None
    assert hasattr(status_label, 'setText')
    
    # Test preparing UI components for an app
    app = MagicMock()
    UIComponentTestUtils.prepare_ui_components(app)
    assert app.board_widget is not None
    assert app.status_label is not None