"""
Comprehensive tests for DetectionThread module.
Tests detection processing, threading, and game state integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import numpy as np

from app.core.detection_thread import DetectionThread
from app.core.config import GameDetectorConfig


@pytest.fixture
def mock_config():
    """Create mock GameDetectorConfig."""
    config = Mock(spec=GameDetectorConfig)
    config.grid_model_path = "test_grid.pt"
    config.symbol_model_path = "test_symbol.pt"
    config.bbox_conf_threshold = 0.5
    config.grid_conf_threshold = 0.3
    return config


class TestDetectionThreadInit:
    """Test DetectionThread initialization."""

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_initialization_default_device(self, mock_game_detector, mock_torch, mock_config):
        """Test DetectionThread initialization with default device detection."""
        mock_torch.cuda.is_available.return_value = True
        mock_detector = Mock()
        mock_game_detector.return_value = mock_detector
        
        thread = DetectionThread(mock_config, target_fps=1.5)
        
        assert thread.config == mock_config
        assert thread.target_fps == 1.5
        assert abs(thread.frame_interval - (1.0 / 1.5)) < 0.001
        assert thread.device == 'cuda'
        assert thread.running is False
        assert thread.daemon is True

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_initialization_cpu_fallback(self, mock_game_detector, mock_torch, mock_config):
        """Test DetectionThread initialization with CPU fallback."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        
        thread = DetectionThread(mock_config)
        
        assert thread.device == 'cpu'

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_initialization_mps_device(self, mock_game_detector, mock_torch, mock_config):
        """Test DetectionThread initialization with MPS device."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        
        thread = DetectionThread(mock_config)
        
        assert thread.device == 'mps'

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_initialization_explicit_device(self, mock_game_detector, mock_torch, mock_config):
        """Test DetectionThread initialization with explicit device."""
        thread = DetectionThread(mock_config, device='cpu')
        
        assert thread.device == 'cpu'

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_inherits_from_thread(self, mock_game_detector, mock_torch, mock_config):
        """Test that DetectionThread inherits from threading.Thread."""
        thread = DetectionThread(mock_config)
        assert isinstance(thread, threading.Thread)

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_logger_setup(self, mock_game_detector, mock_torch, mock_config):
        """Test that DetectionThread sets up logger."""
        with patch('app.core.detection_thread.logging.getLogger') as mock_logger:
            thread = DetectionThread(mock_config)
            mock_logger.assert_called_once()


class TestDetectionThreadMethods:
    """Test DetectionThread method functionality."""

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_set_frame_valid_frame(self, mock_game_detector, mock_torch, mock_config):
        """Test setting a valid frame."""
        thread = DetectionThread(mock_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        thread.set_frame(frame)
        
        assert np.array_equal(thread.current_frame, frame)
        assert thread.frame_available.is_set()

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_set_frame_none(self, mock_game_detector, mock_torch, mock_config):
        """Test setting frame to None."""
        thread = DetectionThread(mock_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        thread.set_frame(frame)  # Set a frame first
        
        thread.set_frame(None)
        
        assert thread.current_frame is None
        assert not thread.frame_available.is_set()

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_get_latest_detection_with_result(self, mock_game_detector, mock_torch, mock_config):
        """Test getting latest detection when result available."""
        thread = DetectionThread(mock_config)
        mock_result = {"grid_points": [], "board": [0]*9}
        thread.latest_detection_result = mock_result
        
        result = thread.get_latest_detection()
        
        assert result == mock_result

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_get_latest_detection_no_result(self, mock_game_detector, mock_torch, mock_config):
        """Test getting latest detection when no result available."""
        thread = DetectionThread(mock_config)
        
        result = thread.get_latest_detection()
        
        assert result is None

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_stop_thread(self, mock_game_detector, mock_torch, mock_config):
        """Test stopping the thread."""
        thread = DetectionThread(mock_config)
        
        thread.stop()
        
        assert thread.running is False

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_is_grid_complete_true(self, mock_game_detector, mock_torch, mock_config):
        """Test is_grid_complete when grid is complete."""
        thread = DetectionThread(mock_config)
        thread.grid_complete = True
        
        result = thread.is_grid_complete()
        
        assert result is True

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_is_grid_complete_false(self, mock_game_detector, mock_torch, mock_config):
        """Test is_grid_complete when grid is incomplete."""
        thread = DetectionThread(mock_config)
        thread.grid_complete = False
        
        result = thread.is_grid_complete()
        
        assert result is False


class TestDetectionThreadConstants:
    """Test DetectionThread constants and attributes."""

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_has_required_attributes(self, mock_game_detector, mock_torch, mock_config):
        """Test that DetectionThread has all required attributes."""
        thread = DetectionThread(mock_config)
        
        required_attrs = [
            'config', 'target_fps', 'frame_interval', 'device', 'running',
            'daemon', 'current_frame', 'frame_available', 'latest_detection_result',
            'grid_complete'
        ]
        for attr in required_attrs:
            assert hasattr(thread, attr), f"DetectionThread missing attribute: {attr}"

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_has_required_methods(self, mock_game_detector, mock_torch, mock_config):
        """Test that DetectionThread has all required methods."""
        thread = DetectionThread(mock_config)
        
        required_methods = [
            'set_frame', 'get_latest_detection', 'stop', 'is_grid_complete', 'run'
        ]
        for method in required_methods:
            assert hasattr(thread, method), f"DetectionThread missing method: {method}"
            assert callable(getattr(thread, method)), f"DetectionThread.{method} is not callable"

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_frame_interval_calculation(self, mock_game_detector, mock_torch, mock_config):
        """Test frame interval calculation for different FPS values."""
        test_cases = [
            (1.0, 1.0),
            (2.0, 0.5),
            (5.0, 0.2),
            (10.0, 0.1)
        ]
        
        for fps, expected_interval in test_cases:
            thread = DetectionThread(mock_config, target_fps=fps)
            assert abs(thread.frame_interval - expected_interval) < 0.001


class TestDetectionThreadThreadingEvents:
    """Test DetectionThread threading events."""

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_frame_available_event_type(self, mock_game_detector, mock_torch, mock_config):
        """Test that frame_available is proper Event type."""
        thread = DetectionThread(mock_config)
        
        assert isinstance(thread.frame_available, threading.Event)
        assert not thread.frame_available.is_set()

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_frame_available_event_behavior(self, mock_game_detector, mock_torch, mock_config):
        """Test frame_available event behavior."""
        thread = DetectionThread(mock_config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Initially not set
        assert not thread.frame_available.is_set()
        
        # Set when frame is provided
        thread.set_frame(frame)
        assert thread.frame_available.is_set()
        
        # Cleared when frame is set to None
        thread.set_frame(None)
        assert not thread.frame_available.is_set()


class TestDetectionThreadGameDetectorIntegration:
    """Test DetectionThread integration with GameDetector."""

    @patch('app.core.detection_thread.torch')
    @patch('app.core.detection_thread.GameDetector')
    def test_game_detector_creation(self, mock_game_detector, mock_torch, mock_config):
        """Test that GameDetector is properly created."""
        mock_detector = Mock()
        mock_game_detector.return_value = mock_detector
        
        thread = DetectionThread(mock_config, device='cpu')
        
        mock_game_detector.assert_called_once_with(mock_config, device='cpu')
        assert thread.game_detector == mock_detector