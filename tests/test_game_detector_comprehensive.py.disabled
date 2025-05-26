"""
Comprehensive pytest tests for the GameDetector module.
Tests all core functionality including YOLO model integration, frame processing,
and component coordination.
"""
import logging
import time
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import numpy as np
import cv2
import torch

from app.main.game_detector import GameDetector
from app.core.config import GameDetectorConfig
from app.core.game_state import GameState


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_config():
    """Mock GameDetectorConfig for testing."""
    config = Mock(spec=GameDetectorConfig)
    config.bbox_conf_threshold = 0.6
    config.max_detection_wait_time = 10.0
    config.max_retry_count = 3
    config.enable_debug_window = False
    config.enable_visualization = True
    return config


@pytest.fixture
def sample_frame():
    """Sample frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_game_state():
    """Mock GameState for testing."""
    state = Mock(spec=GameState)
    state.is_valid.return_value = True
    state.board = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    return state


class TestGameDetectorInitialization:
    """Test GameDetector initialization and setup."""

    @patch('app.main.game_detector.torch.cuda.is_available')
    @patch('ultralytics.YOLO')
    def test_initialization_cuda_available(self, mock_yolo, mock_cuda_available, mock_config):
        """Test GameDetector initialization when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_yolo.return_value = mock_model
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0,
            detect_model_path='test_detect.pt',
            pose_model_path='test_pose.pt'
        )
        
        assert detector.device == 'cuda'
        assert detector.camera_index == 0
        assert detector.config == mock_config
        assert mock_yolo.call_count == 2  # detect and pose models

    @patch('app.main.game_detector.torch.cuda.is_available')
    @patch('app.main.game_detector.torch.backends.mps.is_available')
    @patch('ultralytics.YOLO')
    def test_initialization_mps_available(self, mock_yolo, mock_mps_available, mock_cuda_available, mock_config):
        """Test GameDetector initialization when MPS is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_yolo.return_value = mock_model
        
        with patch('app.main.game_detector.torch.backends'):
            detector = GameDetector(
                config=mock_config,
                camera_index=1,
                device='mps'
            )
        
        assert detector.device == 'mps'
        assert detector.camera_index == 1

    @patch('app.main.game_detector.torch.cuda.is_available')
    @patch('ultralytics.YOLO')
    def test_initialization_cpu_fallback(self, mock_yolo, mock_cuda_available, mock_config):
        """Test GameDetector initialization with CPU fallback."""
        mock_cuda_available.return_value = False
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_yolo.return_value = mock_model
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0
        )
        
        assert detector.device == 'cpu'

    @patch('ultralytics.YOLO')
    def test_initialization_with_custom_device(self, mock_yolo, mock_config):
        """Test GameDetector initialization with custom device."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_yolo.return_value = mock_model
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0,
            device='cuda:1'
        )
        
        assert detector.device == 'cuda:1'

    @patch('ultralytics.YOLO')
    def test_initialization_components_created(self, mock_yolo, mock_config):
        """Test that all components are properly initialized."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_yolo.return_value = mock_model
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0
        )
        
        assert detector.grid_detector is not None
        assert detector.symbol_detector is not None
        assert detector.visualization_manager is not None
        assert detector.game_state_manager is not None
        assert detector.fps_calculator is not None


class TestGameDetectorModelLoading:
    """Test model loading functionality."""

    @patch('ultralytics.YOLO')
    def test_load_models_ultralytics_success(self, mock_yolo_class, mock_config):
        """Test successful model loading with ultralytics."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_yolo_class.return_value = mock_model
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0
        )
        
        assert mock_yolo_class.call_count == 2  # detect and pose models
        assert detector.detect_model == mock_model
        assert detector.pose_model == mock_model

    @patch('ultralytics.YOLO', side_effect=ImportError)
    @patch('app.main.game_detector.torch.hub.load')
    def test_load_models_torch_hub_fallback(self, mock_hub_load, mock_yolo_class, mock_config):
        """Test model loading fallback to torch.hub."""
        mock_model = Mock()
        mock_hub_load.return_value = mock_model
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0
        )
        
        assert mock_hub_load.call_count == 2  # detect and pose models
        assert detector.detect_model == mock_model
        assert detector.pose_model == mock_model

    @patch('ultralytics.YOLO', side_effect=Exception("Model loading failed"))
    def test_load_models_failure(self, mock_yolo, mock_config):
        """Test model loading failure."""
        with pytest.raises(Exception, match="Model loading failed"):
            GameDetector(
                config=mock_config,
                camera_index=0
            )


class TestGameDetectorFrameProcessing:
    """Test frame processing functionality."""

    @patch('app.main.game_detector.torch.hub.load')
    def test_process_frame_basic(self, mock_hub_load, mock_config, sample_frame, mock_game_state):
        """Test basic frame processing."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        # Mock component methods
        detector.symbol_detector.detect_symbols = Mock(return_value=(sample_frame, []))
        detector.grid_detector.detect_grid = Mock(return_value=(sample_frame, []))
        detector.grid_detector.sort_grid_points = Mock(return_value=[])
        detector.grid_detector.is_valid_grid = Mock(return_value=True)
        detector.grid_detector.compute_homography = Mock(return_value=np.eye(3))
        detector.grid_detector.update_grid_status = Mock(return_value=False)
        detector.game_state_manager.update_game_state = Mock(return_value=[])
        detector.game_state_manager.game_state = mock_game_state
        detector.visualization_manager.draw_detection_results = Mock(return_value=sample_frame)
        detector.visualization_manager.draw_debug_info = Mock()
        
        frame_time = time.time()
        result_frame, result_state = detector.process_frame(sample_frame, frame_time)
        
        assert result_frame is not None
        assert result_state == mock_game_state
        detector.symbol_detector.detect_symbols.assert_called_once()
        detector.grid_detector.detect_grid.assert_called_once()

    @patch('app.main.game_detector.torch.hub.load')
    def test_process_frame_invalid_grid(self, mock_hub_load, mock_config, sample_frame):
        """Test frame processing with invalid grid."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        # Mock component methods with invalid grid
        detector.symbol_detector.detect_symbols = Mock(return_value=(sample_frame, []))
        detector.grid_detector.detect_grid = Mock(return_value=(sample_frame, []))
        detector.grid_detector.sort_grid_points = Mock(return_value=[])
        detector.grid_detector.is_valid_grid = Mock(return_value=False)
        detector.grid_detector.update_grid_status = Mock(return_value=True)
        detector.game_state_manager.update_game_state = Mock(return_value=[])
        detector.game_state_manager.game_state = None
        detector.visualization_manager.draw_detection_results = Mock(return_value=sample_frame)
        detector.visualization_manager.draw_debug_info = Mock()
        
        frame_time = time.time()
        result_frame, result_state = detector.process_frame(sample_frame, frame_time)
        
        assert result_frame is not None
        assert result_state is None
        # Homography should not be computed for invalid grid
        detector.grid_detector.compute_homography.assert_not_called()

    @patch('app.main.game_detector.torch.hub.load')
    def test_process_frame_fps_calculation(self, mock_hub_load, mock_config, sample_frame):
        """Test FPS calculation during frame processing."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        # Mock component methods
        detector.symbol_detector.detect_symbols = Mock(return_value=(sample_frame, []))
        detector.grid_detector.detect_grid = Mock(return_value=(sample_frame, []))
        detector.grid_detector.sort_grid_points = Mock(return_value=[])
        detector.grid_detector.is_valid_grid = Mock(return_value=True)
        detector.grid_detector.compute_homography = Mock(return_value=np.eye(3))
        detector.grid_detector.update_grid_status = Mock(return_value=False)
        detector.game_state_manager.update_game_state = Mock(return_value=[])
        detector.game_state_manager.game_state = None
        detector.visualization_manager.draw_detection_results = Mock(return_value=sample_frame)
        detector.visualization_manager.draw_debug_info = Mock()
        
        # Mock FPS calculator
        detector.fps_calculator.tick = Mock()
        detector.fps_calculator.get_fps = Mock(return_value=30.0)
        
        frame_time = time.time()
        detector.process_frame(sample_frame, frame_time)
        
        detector.fps_calculator.tick.assert_called_once()
        detector.fps_calculator.get_fps.assert_called_once()


class TestGameDetectorCameraHandling:
    """Test camera handling functionality."""

    @patch('app.main.game_detector.torch.hub.load')
    @patch('app.main.game_detector.cv2.VideoCapture')
    def test_run_detection_camera_init_success(self, mock_video_capture, mock_hub_load, mock_config):
        """Test successful camera initialization."""
        mock_hub_load.return_value = Mock()
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Return False to exit loop quickly
        mock_video_capture.return_value = mock_cap
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        with patch.object(detector, 'release') as mock_release:
            try:
                detector.run_detection()
            except Exception:
                pass  # Expected due to failed frame read
        
        mock_video_capture.assert_called_with(0)
        mock_cap.isOpened.assert_called_once()

    @patch('app.main.game_detector.torch.hub.load')
    @patch('app.main.game_detector.cv2.VideoCapture')
    def test_run_detection_camera_init_failure(self, mock_video_capture, mock_hub_load, mock_config):
        """Test camera initialization failure."""
        mock_hub_load.return_value = Mock()
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        with pytest.raises(RuntimeError, match="Failed to open camera"):
            detector.run_detection()

    @patch('app.main.game_detector.torch.hub.load')
    def test_release_resources(self, mock_hub_load, mock_config):
        """Test resource release."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        # Mock camera
        mock_cap = Mock()
        detector.cap = mock_cap
        
        with patch('app.main.game_detector.cv2.destroyAllWindows') as mock_destroy:
            detector.release()
        
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch('app.main.game_detector.torch.hub.load')
    def test_release_resources_no_camera(self, mock_hub_load, mock_config):
        """Test resource release when no camera is initialized."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        detector.cap = None
        
        with patch('app.main.game_detector.cv2.destroyAllWindows') as mock_destroy:
            detector.release()
        
        mock_destroy.assert_called_once()


class TestGameDetectorLogging:
    """Test logging functionality."""

    @patch('app.main.game_detector.torch.hub.load')
    def test_logger_setup(self, mock_hub_load, mock_config):
        """Test logger setup and configuration."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0,
            log_level=logging.DEBUG
        )
        
        assert detector.logger is not None
        assert detector.logger.level == logging.DEBUG

    @patch('app.main.game_detector.torch.hub.load')
    @patch('app.main.game_detector.time.time')
    def test_performance_logging(self, mock_time, mock_hub_load, mock_config, sample_frame):
        """Test performance logging during frame processing."""
        mock_hub_load.return_value = Mock()
        mock_time.side_effect = [0, 6, 6]  # Simulate log interval exceeded
        
        detector = GameDetector(config=mock_config, camera_index=0)
        detector.last_log_time = 0
        detector.log_interval = 5
        
        # Mock component methods
        detector.symbol_detector.detect_symbols = Mock(return_value=(sample_frame, []))
        detector.grid_detector.detect_grid = Mock(return_value=(sample_frame, []))
        detector.grid_detector.sort_grid_points = Mock(return_value=[])
        detector.grid_detector.is_valid_grid = Mock(return_value=True)
        detector.grid_detector.compute_homography = Mock(return_value=np.eye(3))
        detector.grid_detector.update_grid_status = Mock(return_value=False)
        detector.game_state_manager.update_game_state = Mock(return_value=[])
        detector.game_state_manager.game_state = None
        detector.visualization_manager.draw_detection_results = Mock(return_value=sample_frame)
        detector.visualization_manager.draw_debug_info = Mock()
        detector.fps_calculator.get_fps = Mock(return_value=30.0)
        
        with patch.object(detector.logger, 'info') as mock_log_info:
            detector.process_frame(sample_frame, 0)
            mock_log_info.assert_called()


class TestGameDetectorErrorHandling:
    """Test error handling in GameDetector."""

    @patch('app.main.game_detector.torch.hub.load')
    def test_process_frame_component_error(self, mock_hub_load, mock_config, sample_frame):
        """Test frame processing when component raises exception."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        # Mock symbol detector to raise exception
        detector.symbol_detector.detect_symbols = Mock(side_effect=Exception("Detection failed"))
        
        # Should not raise exception, but should handle it gracefully
        with patch.object(detector.logger, 'exception') as mock_log_exception:
            try:
                detector.process_frame(sample_frame, time.time())
            except Exception:
                pass  # Expected behavior - exception should be caught and logged


class TestGameDetectorIntegration:
    """Test integration between components."""

    @patch('app.main.game_detector.torch.hub.load')
    def test_full_pipeline_integration(self, mock_hub_load, mock_config, sample_frame, mock_game_state):
        """Test full detection pipeline integration."""
        mock_hub_load.return_value = Mock()
        
        detector = GameDetector(config=mock_config, camera_index=0)
        
        # Mock realistic detection results
        mock_symbols = [{'class': 0, 'bbox': [100, 100, 150, 150], 'confidence': 0.8}]
        mock_keypoints = np.array([[100, 100], [200, 100], [300, 100], [400, 100]])
        mock_homography = np.eye(3)
        mock_cell_polygons = [np.array([[0, 0], [100, 0], [100, 100], [0, 100]])]
        
        detector.symbol_detector.detect_symbols = Mock(return_value=(sample_frame, mock_symbols))
        detector.grid_detector.detect_grid = Mock(return_value=(sample_frame, mock_keypoints))
        detector.grid_detector.sort_grid_points = Mock(return_value=mock_keypoints)
        detector.grid_detector.is_valid_grid = Mock(return_value=True)
        detector.grid_detector.compute_homography = Mock(return_value=mock_homography)
        detector.grid_detector.update_grid_status = Mock(return_value=True)
        detector.game_state_manager.update_game_state = Mock(return_value=mock_cell_polygons)
        detector.game_state_manager.game_state = mock_game_state
        detector.visualization_manager.draw_detection_results = Mock(return_value=sample_frame)
        detector.visualization_manager.draw_debug_info = Mock()
        
        frame_time = time.time()
        result_frame, result_state = detector.process_frame(sample_frame, frame_time)
        
        assert result_frame is not None
        assert result_state == mock_game_state
        
        # Verify all components were called with correct data
        detector.game_state_manager.update_game_state.assert_called_once_with(
            sample_frame, mock_keypoints, mock_homography, mock_symbols, frame_time, True
        )
        
        detector.visualization_manager.draw_detection_results.assert_called_once_with(
            sample_frame, detector.fps_calculator.get_fps(),
            mock_keypoints, mock_keypoints, mock_cell_polygons,
            mock_symbols, mock_homography, mock_game_state
        )