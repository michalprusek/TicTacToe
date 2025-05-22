"""Tests for the game detector symbol detection functionality."""
import pytest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import cv2

from app.main.game_detector import GameDetector
from app.core.game_state import GameState


@pytest.fixture
def detector():
    """detector fixture for tests."""
    detector = GameDetector(self.config)
    detector.logger = self.logger
    detector.detect_model = self.detect_model
    detector.game_state = GameState()
    detector.frame_width = 1920
    detector.frame_height = 1080
    return detector


@pytest.fixture
def detect_model():
    """detect_model fixture for tests."""
    detect_model = MagicMock()
    self.detector.detect_model = detect_model
    return detect_model



class TestGameDetectorSymbolDetection():
    """Test cases for the game detector symbol detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = MagicMock()
        self.config.game_detector.camera_index = 0
        self.config.game_detector.confidence_threshold = 0.5

        # Create a mock logger
        self.logger = MagicMock()

        # Create a mock detect model
        self.detect_model = MagicMock()

        # Create a patch for the GameDetector constructor
        self.patcher = patch('app.main.game_detector.GameDetector.__init__',
                            return_value=None)
        self.mock_init = self.patcher.start()

        # Create a GameDetector instance
        self.detector = GameDetector(self.config)
        self.detector.logger = self.logger
        self.detector.detect_model = self.detect_model
        self.detector.game_state = GameState()
        self.detector.frame_width = 1920
        self.detector.frame_height = 1080

    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher.stop()

    def test_get_cell_for_point(self):
        """Test the _get_cell_for_point method."""
        # Test points in different cells
        # Center of the grid
        point = [960, 540]
        cell = self.detector._get_cell_for_point(point)
        assert cell == (1, 1)

        # Top-left cell
        point = [500, 100]
        cell = self.detector._get_cell_for_point(point)
        assert cell == (0, 0)

        # Bottom-right cell
        point = [1400, 900]
        cell = self.detector._get_cell_for_point(point)
        assert cell == (2, 2)

        # Point outside the grid
        point = [100, 100]
        cell = self.detector._get_cell_for_point(point)
        assert cell is None

    def test_update_game_state(self):
        """Test the _update_game_state method."""
        # Create test symbols
        symbols = [
            {
                'label': 'X',
                'confidence': 0.95,
                'box': [500, 100, 600, 200]  # Top-left cell
            },
            {
                'label': 'O',
                'confidence': 0.90,
                'box': [900, 500, 1000, 600]  # Center cell
            }
        ]

        # Mock the _get_cell_for_point method
        with patch.object(self.detector, '_get_cell_for_point') as mock_get_cell:
            # Set up the mock to return specific cells for specific points
            def side_effect(point):
                x, y = point
                if 500 <= x <= 600 and 100 <= y <= 200:
                    return (0, 0)  # Top-left cell
                elif 900 <= x <= 1000 and 500 <= y <= 600:
                    return (1, 1)  # Center cell
                return None

            mock_get_cell.side_effect = side_effect

            # Call the method under test
            self.detector._update_game_state(symbols, None)

            # Check that the game state was updated correctly
            board = self.detector.game_state._board_state
        assert board[0][0] == 'X'
        assert board[1][1] == 'O'
        assert board[0][1] == ''  # Empty cell

    def test_detect_symbols(self):
        """Test the _detect_symbols method."""
        # Create a test frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Mock the detect_model.predict method
        mock_boxes = MagicMock()
        mock_boxes.data = MagicMock()
        mock_boxes.data.cpu.return_value.numpy.return_value = np.array([
            [500, 100, 600, 200, 0.95, 1],  # X symbol
            [900, 500, 1000, 600, 0.90, 0]  # O symbol
        ])

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        self.detect_model.predict.return_value = [mock_result]

        # Call the method under test
        processed_frame, symbols = self.detector._detect_symbols(frame)

        # Check that the symbols were detected correctly
        assert len(symbols) == 2
        assert symbols[0]['label'] == 'X'
        assert symbols[0]['confidence'] == 0.95
        assert symbols[0]['box'] == [500, 100, 600, 200]

        assert symbols[1]['label'] == 'O'
        assert symbols[1]['confidence'] == 0.90
        assert symbols[1]['box'] == [900, 500, 1000, 600]

    def test_draw_detection_results(self):
        """Test the _draw_detection_results method."""
        # Create a test frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Create test keypoints
        keypoints = np.array([
            [100, 100],  # Point 0
            [200, 100],  # Point 1
            [300, 100],  # Point 2
            [100, 200],  # Point 3
            [200, 200],  # Point 4
            [300, 200],  # Point 5
            [100, 300],  # Point 6
            [200, 300],  # Point 7
            [300, 300]   # Point 8
        ])

        # Create test symbols
        symbols = [
            [100, 100, 150, 150, 0.95, 1],  # X symbol
            [250, 250, 300, 300, 0.90, 0]   # O symbol
        ]

        # Create a mock homography matrix
        H = np.eye(3)

        # Set up class ID to player mapping
        self.detector.class_id_to_player = {0: 'O', 1: 'X'}
        self.detector.x_val = 'X'
        self.detector.o_val = 'O'

        # Set up detect model names
        self.detector.detect_model.names = {0: 'O', 1: 'X'}

        # Call the method under test
        result_frame = self.detector._draw_detection_results(
            frame, 30.0, keypoints, None, symbols, H
        )

        # Check that the frame was modified (not empty)
        assert np.any(result_frame)

        # We can't easily check the exact drawing operations,
        # but we can verify that the frame was modified
        assert not np.array_equal(frame, np.zeros_like(frame))

    def test_process_frame(self):
        """Test the process_frame method."""
        # Create a test frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Mock the _detect_symbols method
        with patch.object(self.detector, '_detect_symbols') as mock_detect_symbols:
            mock_detect_symbols.return_value = (frame, [
                {
                    'label': 'X',
                    'confidence': 0.95,
                    'box': [500, 100, 600, 200]  # Top-left cell
                }
            ])

            # Mock the _detect_grid method
            with patch.object(self.detector, '_detect_grid') as mock_detect_grid:
                mock_detect_grid.return_value = (frame, np.array([[100, 100]]))

                # Mock the _update_game_state method
                with patch.object(self.detector, '_update_game_state') as mock_update_game_state:
                    # Call the method under test
                    self.detector.process_frame(frame, 0.0)

                    # Check that the methods were called correctly
                    mock_detect_symbols.assert_called_once_with(frame)
                    mock_detect_grid.assert_called_once_with(frame)
                    mock_update_game_state.assert_called_once()
