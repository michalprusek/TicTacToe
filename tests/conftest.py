"""Pytest configuration file."""
import os
import sys
import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def empty_board():
    """Fixture for an empty game board."""
    from app.main import game_logic
    return game_logic.create_board()


@pytest.fixture
def mock_arm_controller():
    """Fixture for a mock arm controller."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.connected = True
    mock.draw_x.return_value = True
    mock.draw_o.return_value = True
    mock.park.return_value = True
    mock.go_to_position.return_value = True
    return mock


@pytest.fixture
def mock_camera():
    """Fixture for a mock camera."""
    from unittest.mock import MagicMock
    import numpy as np
    
    mock = MagicMock()
    mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock.isOpened.return_value = True
    return mock
