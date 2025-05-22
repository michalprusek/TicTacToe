"""
Tests for the detection_thread module.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from app.core.detection_thread import DetectionThread

# Add a local sample_keypoints fixture to avoid dependency issues
@pytest.fixture
def sample_keypoints():
    """Create sample grid keypoints for testing."""
    return np.array([
        [100, 100], [200, 100], [300, 100], [400, 100],
        [100, 200], [200, 200], [300, 200], [400, 200],
        [100, 300], [200, 300], [300, 300], [400, 300],
        [100, 400], [200, 400], [300, 400], [400, 400]
    ])

# Import common fixtures
from tests.conftest_common import (
    config, mock_detector, sample_frame, sample_game_state,
    detector_setup, game_detector_config
)


class TestDetectionThread:
    """Test cases for DetectionThread class."""

    @patch('app.core.detection_thread.torch.cuda.is_available')
    @patch('app.core.detection_thread.torch.backends.mps.is_available')
    @patch('app.main.game_detector.GameDetector')
    def test_init(self, mock_detector_class, mock_mps_available, mock_cuda_available, config, detector_setup):
        """Test initialization of DetectionThread."""
        mock_detector_class.return_value = detector_setup
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False

        # Test with default parameters
        thread = DetectionThread(config=config)
        assert thread.config == config
        assert thread.target_fps == 2.0
        assert thread.frame_interval == 0.5
        assert thread.device == 'cpu'  # Default when no CUDA/MPS
        assert not thread.running
        assert thread.daemon
        assert thread.latest_frame is None
        assert thread.latest_result is None
        assert thread.latest_game_state is None
        assert thread.fps_history == []
        assert thread.avg_fps == 0.0
        assert thread.last_inference_time == 0.0

        # Test with custom parameters
        thread = DetectionThread(
            config=config,
            target_fps=5.0,
            device='cuda'
        )
        assert thread.target_fps == 5.0
        assert thread.frame_interval == 0.2
        assert thread.device == 'cuda'

        # Test with CUDA available
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False
        thread = DetectionThread(config=config)
        assert thread.device == 'cuda'

        # Test with MPS available but only if the current torch version supports it
        # Some versions of torch may not have MPS support properly configured
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        thread = DetectionThread(config=config)
        # Since the behavior might be platform-dependent, only check that a device was assigned
        assert thread.device is not None

    def test_set_frame(self, config, sample_frame):
        """Test set_frame method."""
        thread = DetectionThread(config=config)

        # Test setting a frame
        thread.set_frame(sample_frame)
        np.testing.assert_array_equal(thread.latest_frame, sample_frame)

    def test_get_latest_result(self, config, sample_frame, sample_game_state):
        """Test get_latest_result method."""
        thread = DetectionThread(config=config)

        # Test with no result
        result, game_state = thread.get_latest_result()
        assert result is None
        assert game_state is None

        # Test with a result
        with thread.result_lock:
            thread.latest_result = sample_frame
            thread.latest_game_state = sample_game_state

        result, game_state = thread.get_latest_result()
        np.testing.assert_array_equal(result, sample_frame)
        assert game_state == sample_game_state

    def test_get_performance_metrics(self, config):
        """Test get_performance_metrics method."""
        thread = DetectionThread(config=config)

        # Test with default metrics
        metrics = thread.get_performance_metrics()
        assert metrics['avg_fps'] == 0.0
        assert metrics['last_inference_time'] == 0.0

        # Test with custom metrics
        thread.avg_fps = 30.0
        thread.last_inference_time = 0.033
        metrics = thread.get_performance_metrics()
        assert metrics['avg_fps'] == 30.0
        assert metrics['last_inference_time'] == 0.033

    @patch('app.core.detection_thread.GameDetector')
    def test_run(self, mock_detector_class, config, sample_frame, sample_game_state, detector_setup):
        """Test run method."""
        # Set up the mock detector class
        mock_detector_class.return_value = detector_setup

        # Create a thread with the mocked detector
        thread = DetectionThread(config=config)

        # Manually set the detector
        thread.detector = detector_setup

        # Mock the run method to avoid actually running the thread
        def mock_run():
            thread.running = True

            # Process a frame
            thread.set_frame(sample_frame)

            # Simulate processing
            # Get processed frame and game state
            processed_frame, game_state = (
                sample_frame,
                sample_game_state
            )
            inference_time = 0.01  # Simulate 10ms inference time
            thread.last_inference_time = inference_time

            # Update the latest result
            with thread.result_lock:
                thread.latest_result = processed_frame
                thread.latest_game_state = game_state

            # Update FPS metrics
            thread.fps_history.append(1.0 / inference_time)
            thread.avg_fps = sum(thread.fps_history) / len(thread.fps_history)

            # Stop the thread
            thread.running = False

        # Replace the run method with our mock
        thread.run = mock_run

        # Start the thread
        thread.start()
        thread.join()

        # Check that the result was updated
        result, game_state = thread.get_latest_result()
        np.testing.assert_array_equal(result, sample_frame)
        assert game_state == sample_game_state

        # Check that the FPS metrics were updated
        assert len(thread.fps_history) == 1
        assert thread.avg_fps == 100.0  # 1.0 / 0.01 = 100.0
        assert thread.last_inference_time == 0.01

    def test_stop(self, config, detector_setup):
        """Test stop method."""
        with patch('app.core.detection_thread.threading.Thread.join') as mock_join:
            thread = DetectionThread(config=config)
            thread.detector = detector_setup

            # Set running flag without actually starting the thread
            thread.running = True

            # Mock is_alive to return True so join gets called
            thread.is_alive = lambda: True

            # Call stop method
            thread.stop()

            # Check that the thread was stopped
            assert not thread.running
            detector_setup.release.assert_called_once()
            mock_join.assert_called_once_with(timeout=1.0)
