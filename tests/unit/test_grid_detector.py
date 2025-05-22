"""
Unit tests for GridDetector class.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from app.main.grid_detector import GridDetector, GRID_POINTS_COUNT


@pytest.fixture
def mock_pose_model():
    """mock_pose_model fixture for tests."""
    mock_model = MagicMock()
    # Mock predict method to return a result with keypoints
    mock_result = MagicMock()
    mock_keypoints = MagicMock()
    mock_keypoints.xy = MagicMock()
    
    # Create a 16x2 array of keypoints
    kpts = np.zeros((1, GRID_POINTS_COUNT, 2), dtype=np.float32)
    for i in range(GRID_POINTS_COUNT):
        kpts[0, i] = [i * 10, i * 10]  # Simple pattern for testing
    
    # Set up the mock chain
    mock_keypoints.xy.cpu.return_value.numpy.return_value = kpts
    mock_result.keypoints = mock_keypoints
    mock_model.predict.return_value = [mock_result]
    
    return mock_model


@pytest.fixture
def mock_config():
    """mock_config fixture for tests."""
    mock_config = MagicMock()
    mock_config.pose_conf_threshold = 0.5
    mock_config.max_grid_detection_retries = 3
    mock_config.grid_lost_threshold_seconds = 2.0
    return mock_config


@pytest.fixture
def grid_detector(mock_pose_model, mock_config):
    """grid_detector fixture for tests."""
    detector = GridDetector(
        pose_model=mock_pose_model,
        config=mock_config
    )
    return detector


class TestGridDetector:
    """Test GridDetector class."""

    def test_init(self, grid_detector, mock_pose_model, mock_config):
        """Test initialization."""
        assert grid_detector.pose_model == mock_pose_model
        assert grid_detector.config == mock_config
        assert grid_detector.pose_conf_threshold == mock_config.pose_conf_threshold
        assert grid_detector.grid_detection_retries == 0
        assert grid_detector.last_valid_grid_time is None
        assert grid_detector.max_grid_detection_retries == mock_config.max_grid_detection_retries
        assert grid_detector.grid_lost_threshold_seconds == mock_config.grid_lost_threshold_seconds

    def test_detect_grid(self, grid_detector, mock_pose_model):
        """Test detect_grid method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Call the method
        _, keypoints = grid_detector.detect_grid(frame)
        
        # Check that pose_model.predict was called
        mock_pose_model.predict.assert_called_once()
        
        # Check that keypoints were extracted
        assert keypoints.shape == (GRID_POINTS_COUNT, 2)
        assert np.sum(keypoints) > 0  # Should have non-zero values

    def test_detect_grid_no_results(self, grid_detector, mock_pose_model):
        """Test detect_grid method with no results."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set up mock to return empty results
        mock_pose_model.predict.return_value = []
        
        # Call the method
        _, keypoints = grid_detector.detect_grid(frame)
        
        # Check that pose_model.predict was called
        mock_pose_model.predict.assert_called_once()
        
        # Check that keypoints are zeros
        assert keypoints.shape == (GRID_POINTS_COUNT, 2)
        assert np.sum(keypoints) == 0

    def test_sort_grid_points(self, grid_detector):
        """Test sort_grid_points method."""
        # Create test keypoints
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        for i in range(GRID_POINTS_COUNT):
            keypoints[i] = [np.random.randint(0, 100), np.random.randint(0, 100)]
        
        # Call the method
        sorted_keypoints = grid_detector.sort_grid_points(keypoints)
        
        # Check that output has same shape
        assert sorted_keypoints.shape == keypoints.shape
        
        # Check that all points are preserved (just reordered)
        assert np.sum(sorted_keypoints) == pytest.approx(np.sum(keypoints))

    def test_sort_grid_points_with_zeros(self, grid_detector):
        """Test sort_grid_points method with some zero points."""
        # Create test keypoints with some zeros
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        for i in range(10):  # Only 10 valid points
            keypoints[i] = [np.random.randint(1, 100), np.random.randint(1, 100)]
        
        # Call the method
        sorted_keypoints = grid_detector.sort_grid_points(keypoints)
        
        # Check that output has same shape
        assert sorted_keypoints.shape == keypoints.shape
        
        # Check that all points are preserved (just reordered)
        assert np.sum(sorted_keypoints) == pytest.approx(np.sum(keypoints))
        
        # Check that we still have 6 zero points
        zero_count = np.sum(np.sum(np.abs(sorted_keypoints), axis=1) == 0)
        assert zero_count == 6

    def test_is_valid_grid(self, grid_detector):
        """Test is_valid_grid method with valid grid."""
        # Create a valid grid (4x4 grid of evenly spaced points)
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                keypoints[i*4 + j] = [j*100, i*100]
        
        # Call the method
        is_valid = grid_detector.is_valid_grid(keypoints)
        
        # Should be valid
        assert is_valid

    def test_is_valid_grid_too_few_points(self, grid_detector):
        """Test is_valid_grid method with too few points."""
        # Create keypoints with only 3 valid points
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        keypoints[0] = [10, 10]
        keypoints[1] = [20, 20]
        keypoints[2] = [30, 30]
        
        # Call the method
        is_valid = grid_detector.is_valid_grid(keypoints)
        
        # Should be invalid
        assert not is_valid

    def test_compute_homography(self, grid_detector):
        """Test compute_homography method."""
        # Create a valid grid (4x4 grid of evenly spaced points)
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                keypoints[i*4 + j] = [j*100, i*100]
        
        # Mock cv2.findHomography
        with patch('cv2.findHomography') as mock_find_homography:
            # Set up mock to return a dummy homography matrix
            mock_find_homography.return_value = (np.eye(3), None)
            
            # Call the method
            H = grid_detector.compute_homography(keypoints)
            
            # Check that findHomography was called
            mock_find_homography.assert_called_once()
            
            # Check that H is the identity matrix
            assert np.array_equal(H, np.eye(3))

    def test_compute_homography_too_few_points(self, grid_detector):
        """Test compute_homography method with too few points."""
        # Create keypoints with only 3 valid points
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        keypoints[0] = [10, 10]
        keypoints[1] = [20, 20]
        keypoints[2] = [30, 30]
        
        # Call the method
        H = grid_detector.compute_homography(keypoints)
        
        # Should return None
        assert H is None

    def test_update_grid_status_first_valid(self, grid_detector):
        """Test update_grid_status method with first valid detection."""
        # Call the method with valid grid
        grid_status_changed = grid_detector.update_grid_status(True, 100.0)
        
        # Should report status changed
        assert grid_status_changed
        
        # Should update last_valid_grid_time
        assert grid_detector.last_valid_grid_time == 100.0
        
        # Should reset retry counter
        assert grid_detector.grid_detection_retries == 0

    def test_update_grid_status_still_valid(self, grid_detector):
        """Test update_grid_status method with continued valid detection."""
        # Set initial state
        grid_detector.last_valid_grid_time = 100.0
        
        # Call the method with valid grid
        grid_status_changed = grid_detector.update_grid_status(True, 101.0)
        
        # Should not report status changed
        assert not grid_status_changed
        
        # Should update last_valid_grid_time
        assert grid_detector.last_valid_grid_time == 101.0

    def test_update_grid_status_lost(self, grid_detector):
        """Test update_grid_status method with grid lost."""
        # Set initial state
        grid_detector.last_valid_grid_time = 100.0
        
        # Call the method with invalid grid and time beyond threshold
        grid_status_changed = grid_detector.update_grid_status(
            False, 
            100.0 + grid_detector.grid_lost_threshold_seconds + 0.1
        )
        
        # Should report status changed
        assert grid_status_changed
        
        # Should reset last_valid_grid_time
        assert grid_detector.last_valid_grid_time is None

    def test_update_grid_status_too_many_retries(self, grid_detector):
        """Test update_grid_status method with too many retries."""
        # Set initial state
        grid_detector.last_valid_grid_time = 100.0
        grid_detector.grid_detection_retries = grid_detector.max_grid_detection_retries
        
        # Call the method with invalid grid
        grid_status_changed = grid_detector.update_grid_status(False, 101.0)
        
        # Should report status changed
        assert grid_status_changed
        
        # Should reset last_valid_grid_time
        assert grid_detector.last_valid_grid_time is None
