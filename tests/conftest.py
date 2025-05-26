# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Pytest configuration and shared fixtures for TicTacToe tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock PyQt5 to avoid GUI dependencies in tests
mock_qt_core = Mock()
mock_qt_core.PYQT_VERSION = 0x050F00  # Mock version 5.15.0
sys.modules['PyQt5'] = Mock()
sys.modules['PyQt5.QtCore'] = mock_qt_core
sys.modules['PyQt5.QtWidgets'] = Mock()
sys.modules['PyQt5.QtGui'] = Mock()

# Mock YOLO models to avoid loading actual models
@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for detection tests."""
    model = Mock()
    model.predict.return_value = [Mock()]
    return model

@pytest.fixture
def sample_board_empty():
    """Empty 3x3 game board."""
    return [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]

@pytest.fixture
def sample_board_with_moves():
    """Game board with some moves."""
    return [['X', ' ', 'O'], [' ', 'X', ' '], ['O', ' ', ' ']]

@pytest.fixture
def sample_board_winning_x():
    """Game board where X wins."""
    return [['X', 'X', 'X'], ['O', 'O', ' '], [' ', ' ', ' ']]

@pytest.fixture
def sample_board_full_tie():
    """Full game board resulting in tie."""
    return [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]@pytest.fixture
def sample_grid_points():
    """Sample 16 grid points in camera coordinates."""
    return np.array([
        [100, 100], [200, 100], [300, 100], [400, 100],
        [100, 200], [200, 200], [300, 200], [400, 200],
        [100, 300], [200, 300], [300, 300], [400, 300],
        [100, 400], [200, 400], [300, 400], [400, 400]
    ], dtype=np.float32)

@pytest.fixture
def sample_detection_results():
    """Sample YOLO detection results."""
    results = []
    result = Mock()
    result.boxes = Mock()
    result.boxes.xyxy = np.array([[100, 100, 150, 150], [200, 200, 250, 250]])
    result.boxes.conf = np.array([0.9, 0.8])
    result.boxes.cls = np.array([0, 1])  # 0=X, 1=O
    results.append(result)
    return results
