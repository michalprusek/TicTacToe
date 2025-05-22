"""
Unit tests for refactored GameDetector class.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from app.main.refactored_game_detector import GameDetector
from app.core.config import GameDetectorConfig
from app.core.game_state import GameState


@pytest.fixture
def mock_config():
    """mock_config fixture for tests."""
    mock_config = MagicMock(spec=GameDetectorConfig)
    return mock_config


@pytest.fixture
def mock_torch_hub():
    """Mock torch.hub to avoid loading actual models."""
    with patch('torch.hub.load') as mock_load:
        # Set up mock models
        mock_detect_model = MagicMock()
        mock_pose_model = MagicMock()
        
        # Configure mock_load to return different models based on path
        def side_effect(repo, model_type, path, device):
            if 'detect' in path:
                return mock_detect_model
            else:
                return mock_pose_model
                
        mock_load.side_effect = side_effect
        
        yield mock_load


@pytest.fixture
def mock_components():
    """Mock all component classes."""
    with patch('app.main.refactored_game_detector.CameraManager') as mock_camera_manager, \
         patch('app.main.refactored_game_detector.GridDetector') as mock_grid_detector, \
         patch('app.main.refactored_game_detector.SymbolDetector') as mock_symbol_detector, \
         patch('app.main.refactored_game_detector.VisualizationManager') as mock_vis_manager, \
         patch('app.main.refactored_game_detector.GameStateManager') as mock_game_state_manager, \
         patch('app.main.refactored_game_detector.FPSCalculator') as mock_fps_calculator:
        
        # Set up mock instances
        mock_camera = MagicMock()
        mock_camera_manager.return_value = mock_camera
        
        mock_grid = MagicMock()
        mock_grid_detector.return_value = mock_grid
        
        mock_symbol = MagicMock()
        mock_symbol_detector.return_value = mock_symbol
        
        mock_vis = MagicMock()
        mock_vis_manager.return_value = mock_vis
        
        mock_game_state = MagicMock()
        mock_game_state_manager.return_value = mock_game_state
        
        mock_fps = MagicMock()
        mock_fps_calculator.return_value = mock_fps
        
        yield {
            'camera_manager': mock_camera,
            'grid_detector': mock_grid,
            'symbol_detector': mock_symbol,
            'visualization_manager': mock_vis,
            'game_state_manager': mock_game_state,
            'fps_calculator': mock_fps
        }


@pytest.fixture
def game_detector(mock_config, mock_torch_hub, mock_components):
    """game_detector fixture for tests."""
    # Mock torch.cuda.is_available to avoid actual CUDA check
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        
        detector = GameDetector(
            config=mock_config,
            camera_index=0,
            detect_model_path='mock_detect_model.pt',
            pose_model_path='mock_pose_model.pt',
            device='cpu'
        )
        
        yield detector


class TestGameDetector:
    """Test GameDetector class."""

    def test_init(self, game_detector, mock_config, mock_components):
        """Test initialization."""
        assert game_detector.config == mock_config
        assert game_detector.camera_index == 0
        assert game_detector.detect_model_path == 'mock_detect_model.pt'
        assert game_detector.pose_model_path == 'mock_pose_model.pt'
        assert game_detector.device == 'cpu'
        
        # Check that components were initialized
        assert game_detector.camera_manager == mock_components['camera_manager']
        assert game_detector.grid_detector == mock_components['grid_detector']
        assert game_detector.symbol_detector == mock_components['symbol_detector']
        assert game_detector.visualization_manager == mock_components['visualization_manager']
        assert game_detector.game_state_manager == mock_components['game_state_manager']
        assert game_detector.fps_calculator == mock_components['fps_calculator']

    def test_load_models(self, mock_torch_hub):
        """Test _load_models method."""
        # Mock torch.cuda.is_available to avoid actual CUDA check
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            
            # Create config
            mock_config = MagicMock(spec=GameDetectorConfig)
            
            # Create detector
            detector = GameDetector(
                config=mock_config,
                detect_model_path='mock_detect_model.pt',
                pose_model_path='mock_pose_model.pt',
                device='cpu'
            )
            
            # Check that torch.hub.load was called twice
            assert mock_torch_hub.call_count == 2
            
            # Check that models were loaded
            assert detector.detect_model is not None
            assert detector.pose_model is not None

    def test_process_frame(self, game_detector, mock_components):
        """Test process_frame method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set up mock returns
        mock_components['symbol_detector'].detect_symbols.return_value = (frame, [])
        mock_components['grid_detector'].detect_grid.return_value = (frame, np.zeros((16, 2)))
        mock_components['grid_detector'].sort_grid_points.return_value = np.zeros((16, 2))
        mock_components['grid_detector'].is_valid_grid.return_value = True
        mock_components['grid_detector'].compute_homography.return_value = np.eye(3)
        mock_components['grid_detector'].update_grid_status.return_value = False
        
        mock_components['game_state_manager'].update_game_state.return_value = []
        mock_components['game_state_manager'].game_state = MagicMock(spec=GameState)
        
        mock_components['fps_calculator'].update.return_value = 30.0
        
        mock_components['visualization_manager'].draw_detection_results.return_value = frame
        
        # Call the method
        result_frame, game_state = game_detector.process_frame(frame, 100.0)
        
        # Check that all component methods were called
        mock_components['symbol_detector'].detect_symbols.assert_called_once()
        mock_components['grid_detector'].detect_grid.assert_called_once()
        mock_components['grid_detector'].sort_grid_points.assert_called_once()
        mock_components['grid_detector'].is_valid_grid.assert_called_once()
        mock_components['grid_detector'].compute_homography.assert_called_once()
        mock_components['grid_detector'].update_grid_status.assert_called_once()
        
        mock_components['game_state_manager'].update_game_state.assert_called_once()
        mock_components['fps_calculator'].update.assert_called_once()
        
        mock_components['visualization_manager'].draw_detection_results.assert_called_once()
        mock_components['visualization_manager'].draw_debug_info.assert_called_once()
        
        # Check that correct values were returned
        assert result_frame is frame
        assert game_state is mock_components['game_state_manager'].game_state

    def test_process_frame_invalid_grid(self, game_detector, mock_components):
        """Test process_frame method with invalid grid."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set up mock returns
        mock_components['symbol_detector'].detect_symbols.return_value = (frame, [])
        mock_components['grid_detector'].detect_grid.return_value = (frame, np.zeros((16, 2)))
        mock_components['grid_detector'].sort_grid_points.return_value = np.zeros((16, 2))
        mock_components['grid_detector'].is_valid_grid.return_value = False  # Invalid grid
        mock_components['grid_detector'].update_grid_status.return_value = True
        
        mock_components['game_state_manager'].update_game_state.return_value = []
        mock_components['game_state_manager'].game_state = MagicMock(spec=GameState)
        
        mock_components['fps_calculator'].update.return_value = 30.0
        
        mock_components['visualization_manager'].draw_detection_results.return_value = frame
        
        # Call the method
        result_frame, game_state = game_detector.process_frame(frame, 100.0)
        
        # Check that compute_homography was not called
        mock_components['grid_detector'].compute_homography.assert_not_called()
        
        # Check that update_game_state was called with None for keypoints
        mock_components['game_state_manager'].update_game_state.assert_called_once()
        args, _ = mock_components['game_state_manager'].update_game_state.call_args
        assert args[1] is None  # keypoints should be None

    def test_run_detection(self, game_detector, mock_components):
        """Test run_detection method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set up mock returns
        mock_components['camera_manager'].read_frame.return_value = (True, frame)
        
        # Mock process_frame
        game_detector.process_frame = MagicMock(return_value=(frame, None))
        
        # Mock cv2.imshow and cv2.waitKey
        with patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey', return_value=27) as mock_waitkey:  # ESC key
            
            # Call the method
            game_detector.run_detection()
            
            # Check that camera_manager.read_frame was called
            mock_components['camera_manager'].read_frame.assert_called_once()
            
            # Check that process_frame was called
            game_detector.process_frame.assert_called_once()
            
            # Check that cv2.imshow was called
            mock_imshow.assert_called_once()
            
            # Check that cv2.waitKey was called
            mock_waitkey.assert_called_once()
            
            # Check that release was called
            mock_components['camera_manager'].release.assert_called_once()

    def test_release(self, game_detector, mock_components):
        """Test release method."""
        # Mock cv2.destroyAllWindows
        with patch('cv2.destroyAllWindows') as mock_destroy_windows:
            # Call the method
            game_detector.release()
            
            # Check that camera_manager.release was called
            mock_components['camera_manager'].release.assert_called_once()
            
            # Check that cv2.destroyAllWindows was called
            mock_destroy_windows.assert_called_once()
