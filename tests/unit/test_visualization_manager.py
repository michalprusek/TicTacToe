"""
Unit tests for VisualizationManager class.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from app.main.visualization_manager import VisualizationManager


@pytest.fixture
def mock_config():
    """mock_config fixture for tests."""
    mock_config = MagicMock()
    mock_config.show_detections = True
    mock_config.show_grid = True
    mock_config.show_debug_info = True
    mock_config.debug_window_scale_factor = 0.5
    mock_config.bbox_conf_threshold = 0.5
    return mock_config


@pytest.fixture
def visualization_manager(mock_config):
    """visualization_manager fixture for tests."""
    manager = VisualizationManager(
        config=mock_config
    )
    return manager


class TestVisualizationManager:
    """Test VisualizationManager class."""

    def test_init(self, visualization_manager, mock_config):
        """Test initialization."""
        assert visualization_manager.config == mock_config
        assert visualization_manager.show_detections == mock_config.show_detections
        assert visualization_manager.show_grid == mock_config.show_grid
        assert visualization_manager.show_debug_info == mock_config.show_debug_info
        assert visualization_manager.debug_window_scale_factor == mock_config.debug_window_scale_factor

    def test_draw_detection_results_basic(self, visualization_manager):
        """Test draw_detection_results method with minimal inputs."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Call the method with minimal inputs
        result_frame = visualization_manager.draw_detection_results(
            frame,
            30.0,  # fps
            None,  # pose_kpts_uv
            None,  # ordered_kpts_uv
            None,  # cell_polygons
            [],    # detected_symbols
            None,  # homography
            None   # game_state
        )
        
        # Check that result is not None and has same shape as input
        assert result_frame is not None
        assert result_frame.shape == frame.shape
        
        # Check that result is not the same object as input (should be a copy)
        assert result_frame is not frame

    def test_draw_detection_results_with_keypoints(self, visualization_manager):
        """Test draw_detection_results method with keypoints."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy keypoints
        pose_kpts_uv = np.zeros((16, 2), dtype=np.float32)
        for i in range(16):
            pose_kpts_uv[i] = [i * 20, i * 15]
        
        # Call the method with keypoints
        result_frame = visualization_manager.draw_detection_results(
            frame,
            30.0,        # fps
            pose_kpts_uv,  # pose_kpts_uv
            None,        # ordered_kpts_uv
            None,        # cell_polygons
            [],          # detected_symbols
            None,        # homography
            None         # game_state
        )
        
        # Check that result is not None and has same shape as input
        assert result_frame is not None
        assert result_frame.shape == frame.shape

    def test_draw_detection_results_with_cell_polygons(self, visualization_manager):
        """Test draw_detection_results method with cell polygons."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy cell polygons
        cell_polygons = []
        for i in range(3):
            for j in range(3):
                # Create a square cell
                cell = np.array([
                    [j*100, i*100],
                    [(j+1)*100, i*100],
                    [(j+1)*100, (i+1)*100],
                    [j*100, (i+1)*100]
                ])
                cell_polygons.append(cell)
        
        # Call the method with cell polygons
        result_frame = visualization_manager.draw_detection_results(
            frame,
            30.0,         # fps
            None,         # pose_kpts_uv
            None,         # ordered_kpts_uv
            cell_polygons,  # cell_polygons
            [],           # detected_symbols
            None,         # homography
            None          # game_state
        )
        
        # Check that result is not None and has same shape as input
        assert result_frame is not None
        assert result_frame.shape == frame.shape

    def test_draw_detection_results_with_symbols(self, visualization_manager):
        """Test draw_detection_results method with detected symbols."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy symbols
        symbols = [
            {'label': 'X', 'confidence': 0.8, 'box': [100, 100, 150, 150], 'class_id': 0},
            {'label': 'O', 'confidence': 0.9, 'box': [200, 200, 250, 250], 'class_id': 1}
        ]
        
        # Mock drawing_utils.draw_symbol_box to avoid actual drawing
        with patch('app.main.drawing_utils.draw_symbol_box') as mock_draw_box:
            # Call the method with symbols
            result_frame = visualization_manager.draw_detection_results(
                frame,
                30.0,    # fps
                None,    # pose_kpts_uv
                None,    # ordered_kpts_uv
                None,    # cell_polygons
                symbols,  # detected_symbols
                None,    # homography
                None     # game_state
            )
            
            # Check that draw_symbol_box was called twice (once for each symbol)
            assert mock_draw_box.call_count == 2
            
            # Check that result is not None and has same shape as input
            assert result_frame is not None
            assert result_frame.shape == frame.shape

    def test_draw_detection_results_with_game_state(self, visualization_manager):
        """Test draw_detection_results method with game state."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy game state
        game_state = MagicMock()
        game_state.status = "In Progress"
        game_state.winner = None
        
        # Call the method with game state
        result_frame = visualization_manager.draw_detection_results(
            frame,
            30.0,       # fps
            None,       # pose_kpts_uv
            None,       # ordered_kpts_uv
            None,       # cell_polygons
            [],         # detected_symbols
            None,       # homography
            game_state  # game_state
        )
        
        # Check that result is not None and has same shape as input
        assert result_frame is not None
        assert result_frame.shape == frame.shape

    def test_draw_debug_info(self, visualization_manager):
        """Test draw_debug_info method."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy game state
        game_state = MagicMock()
        game_state.status = "In Progress"
        game_state.winner = None
        game_state.is_grid_visible = True
        game_state.is_grid_stable = True
        game_state.grid_points = np.zeros((16, 2))
        game_state.cell_polygons = [np.zeros((4, 2)) for _ in range(9)]
        
        # Mock cv2.imshow to avoid actual window creation
        with patch('cv2.imshow') as mock_imshow:
            # Mock drawing_utils.draw_text_lines to avoid actual drawing
            with patch('app.main.drawing_utils.draw_text_lines') as mock_draw_text:
                # Call the method
                visualization_manager.draw_debug_info(
                    frame,
                    30.0,       # fps
                    game_state  # game_state
                )
                
                # Check that cv2.imshow was called
                mock_imshow.assert_called_once()
                
                # Check that draw_text_lines was called
                mock_draw_text.assert_called_once()

    def test_draw_debug_info_disabled(self, visualization_manager):
        """Test draw_debug_info method with debug info disabled."""
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Disable debug info
        visualization_manager.show_debug_info = False
        
        # Mock cv2.imshow to verify it's not called
        with patch('cv2.imshow') as mock_imshow:
            # Call the method
            visualization_manager.draw_debug_info(
                frame,
                30.0,  # fps
                None   # game_state
            )
            
            # Check that cv2.imshow was not called
            mock_imshow.assert_not_called()
