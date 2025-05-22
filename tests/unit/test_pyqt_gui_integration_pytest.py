"""
Integration tests for PyQt GUI with focus on complex interactions and signal handling using pytest
"""
import os
import sys
import json
import numpy as np
import time
import pytest
from unittest.mock import patch, MagicMock, mock_open, call, PropertyMock

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget
from PyQt5.QtCore import Qt, QSize, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPaintEvent, QResizeEvent

from app.main import game_logic
from app.main.pyqt_gui import (
    TicTacToeApp, CameraThread, TicTacToeBoard, CameraView,
    DEFAULT_SYMBOL_SIZE_MM, DEFAULT_DRAW_Z, DEFAULT_SAFE_Z, DRAWING_SPEED,
    CAMERA_REFRESH_RATE, MAX_SPEED, NEUTRAL_X, NEUTRAL_Y, NEUTRAL_Z, PARK_X, PARK_Y,
    CALIBRATION_FILE
)
from app.core.game_state import GameState
from app.core.detection_thread import DetectionThread
from app.core.arm_thread import ArmThread, ArmCommand


class MockGameState:
    """Mock for GameState class with required functionality"""
    
    def __init__(self):
        self._board_state = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        self._grid_corners = np.array([[100, 100], [400, 100], [100, 400], [400, 400]])
        self._grid_points = np.zeros((16, 2))
        for i in range(16):
            row, col = i // 4, i % 4
            self._grid_points[i] = [col * 100 + 100, row * 100 + 100]
    
    def is_valid(self):
        return True
    
    def get_cell_center_uv(self, row, col):
        if 0 <= row < 3 and 0 <= col < 3:
            return [col * 100 + 150, row * 100 + 150]
        return None

    def update_from_detection(self, symbols):
        return True
    
    @property
    def board(self):
        return [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    
    def draw_board_state(self, frame):
        pass
    
    def get_cell_points(self, row, col):
        if 0 <= row < 3 and 0 <= col < 3:
            top_left = row * 100 + 100, col * 100 + 100
            top_right = row * 100 + 100, (col + 1) * 100 + 100
            bottom_left = (row + 1) * 100 + 100, col * 100 + 100
            bottom_right = (row + 1) * 100 + 100, (col + 1) * 100 + 100
            return [top_left, top_right, bottom_left, bottom_right]
        return None


# Create QApplication instance
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def pyqt_gui_app(qapp):
    """Setup PyQt GUI application for testing"""
    # Patch QMainWindow.__init__ to avoid actual window creation
    with patch('PyQt5.QtWidgets.QMainWindow.__init__', return_value=None), \
         patch('PyQt5.QtWidgets.QVBoxLayout'), \
         patch('PyQt5.QtWidgets.QHBoxLayout'), \
         patch('PyQt5.QtWidgets.QLabel'), \
         patch('PyQt5.QtWidgets.QSlider'), \
         patch('PyQt5.QtWidgets.QPushButton'), \
         patch('PyQt5.QtWidgets.QWidget'), \
         patch('app.main.pyqt_gui.CameraThread'), \
         patch('app.core.detection_thread.DetectionThread'), \
         patch('app.core.arm_thread.ArmThread'), \
         patch('app.main.pyqt_gui.BernoulliStrategySelector'), \
         patch('app.main.arm_controller.SwiftAPI'):
        
        # Create the app instance
        app_instance = TicTacToeApp()
        
        # Setup mock objects for app_instance
        app_instance.board_widget = MagicMock()
        app_instance.board_widget.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        app_instance.board_widget.update = MagicMock()
        
        app_instance.camera_thread = MagicMock()
        app_instance.camera_thread.detector = MagicMock()
        app_instance.camera_thread.detector.game_state = MockGameState()
        app_instance.camera_thread.detection_thread = MagicMock()
        app_instance.camera_thread.detection_thread.game_state = MockGameState()
        
        app_instance.arm_thread = MagicMock()
        app_instance.arm_thread.connected = True
        app_instance.arm_thread.go_to_position = MagicMock(return_value=True)
        app_instance.arm_thread.draw_x = MagicMock(return_value=True)
        app_instance.arm_thread.draw_o = MagicMock(return_value=True)
        
        app_instance.arm_controller = MagicMock()
        app_instance.arm_controller.connected = True
        app_instance.arm_controller.go_to_position = MagicMock(return_value=True)
        app_instance.arm_controller.draw_x = MagicMock(return_value=True)
        app_instance.arm_controller.draw_o = MagicMock(return_value=True)
        
        app_instance.status_label = MagicMock()
        app_instance.fps_label = MagicMock()
        app_instance.difficulty_value_label = MagicMock()
        
        app_instance.strategy_selector = MagicMock()
        app_instance.strategy_selector.get_move = MagicMock(return_value=(1, 1))
        
        app_instance.debug_window = MagicMock()
        app_instance.camera_view = MagicMock()
        
        # Test state
        app_instance.human_player = game_logic.PLAYER_X
        app_instance.ai_player = game_logic.PLAYER_O
        app_instance.current_turn = game_logic.PLAYER_X
        
        yield app_instance


@pytest.fixture
def camera_thread(qapp):
    """Setup CameraThread for testing"""
    with patch('cv2.VideoCapture') as mock_video_capture, \
         patch('app.core.detection_thread.DetectionThread') as mock_detection_thread, \
         patch('torch.cuda.is_available', return_value=False):
        
        # Create mock cap
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Create mock detection thread
        mock_detection_thread_instance = MagicMock()
        mock_detection_thread_instance.get_latest_result.return_value = (
            np.zeros((480, 640, 3), dtype=np.uint8),
            MagicMock()
        )
        mock_detection_thread_instance.get_performance_metrics.return_value = {
            'avg_fps': 30.0
        }
        mock_detection_thread.return_value = mock_detection_thread_instance
        
        # Create CameraThread instance
        camera_thread = CameraThread()
        
        # Mock signals
        camera_thread.frame_ready = MagicMock()
        camera_thread.game_state_updated = MagicMock()
        camera_thread.fps_updated = MagicMock()
        
        yield camera_thread
        
        # Cleanup
        if camera_thread.running:
            camera_thread.stop()


def test_get_cell_coordinates_from_yolo_with_calibration_matrix(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with calibration matrix"""
    # Setup calibration data with uv_to_xy_matrix
    pyqt_gui_app.calibration_data = {
        'uv_to_xy_matrix': np.eye(3).tolist()
    }
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result (should be the same as input since we're using identity matrix)
    assert result is not None
    assert len(result) == 2
    
    # Test with matrix multiplication exception
    with patch('numpy.matmul', side_effect=Exception("Test exception")):
        result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
        assert result is not None  # Should still return a fallback value


def test_get_cell_coordinates_from_yolo_with_arm_workspace(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with arm_workspace but no matrix"""
    # Setup calibration data with arm_workspace only
    pyqt_gui_app.calibration_data = {
        'arm_workspace': {
            'min_x': 100,
            'max_x': 300,
            'min_y': -100,
            'max_y': 100
        }
    }
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result
    assert result is not None
    assert len(result) == 2


def test_get_cell_coordinates_from_yolo_without_calibration(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with no calibration data"""
    # Set no calibration data
    pyqt_gui_app.calibration_data = {}
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result (should use simplified transformation)
    assert result is not None
    assert len(result) == 2


def test_get_cell_coordinates_from_yolo_no_camera_thread(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with no camera thread"""
    # Set no camera thread
    pyqt_gui_app.camera_thread = None
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result (should return None)
    assert result == (None, None)


def test_get_cell_coordinates_from_yolo_no_detector(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with no detector"""
    # Set no detector
    pyqt_gui_app.camera_thread.detector = None
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result (should return None)
    assert result == (None, None)


def test_get_cell_coordinates_from_yolo_invalid_game_state(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with invalid game state"""
    # Set invalid game state
    pyqt_gui_app.camera_thread.detector.game_state.is_valid = MagicMock(return_value=False)
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result (should return None)
    assert result == (None, None)


def test_get_cell_coordinates_from_yolo_no_cell_center(pyqt_gui_app):
    """Test get_cell_coordinates_from_yolo with no cell center"""
    # Set no cell center
    pyqt_gui_app.camera_thread.detector.game_state.get_cell_center_uv = MagicMock(return_value=None)
    
    # Call the method
    result = pyqt_gui_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Check result (should return None)
    assert result == (None, None)


def test_load_calibration_file_missing(pyqt_gui_app):
    """Test load_calibration with missing file"""
    # Set up path.exists to return False
    with patch('os.path.exists', return_value=False):
        # Call the method
        result = pyqt_gui_app.load_calibration()
        
        # Check result (should return empty dict)
        assert result == {}


def test_load_calibration_json_error(pyqt_gui_app):
    """Test load_calibration with JSON error"""
    # Set up path.exists to return True and open to raise JSONDecodeError
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='invalid json')):
            with patch('json.load', side_effect=json.JSONDecodeError('Error', '', 0)):
                # Call the method
                result = pyqt_gui_app.load_calibration()
                
                # Check result (should return empty dict)
        assert result == {}


def test_load_calibration_general_exception(pyqt_gui_app):
    """Test load_calibration with general exception"""
    # Set up path.exists to return True and open to raise exception
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', side_effect=Exception("Test exception")):
            # Call the method
            result = pyqt_gui_app.load_calibration()
            
            # Check result (should return empty dict)
        assert result == {}


def test_load_calibration_no_calibration_points(pyqt_gui_app):
    """Test load_calibration with no calibration points"""
    # Set up path.exists to return True and json.load to return dict without calibration_points_raw
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value={}):
                # Call the method
                result = pyqt_gui_app.load_calibration()
                
                # Check result (should return empty dict)
        assert result == {}


def test_load_calibration_not_enough_points(pyqt_gui_app):
    """Test load_calibration with not enough calibration points"""
    # Set up data with only 2 points (less than required 4)
    calibration_data = {
        'calibration_points_raw': [
            {'target_uv': [100, 100], 'robot_xyz': [200, 0, 10]},
            {'target_uv': [300, 100], 'robot_xyz': [250, 0, 10]}
        ]
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value=calibration_data):
                # Call the method
                result = pyqt_gui_app.load_calibration()
                
                # Check result (should return data without transformation matrix)
        assert result == calibration_data
        assert 'uv_to_xy_matrix' not in result


def test_load_calibration_valid_points(pyqt_gui_app):
    """Test load_calibration with valid calibration points"""
    # Set up data with 4 valid points
    calibration_data = {
        'calibration_points_raw': [
            {'target_uv': [100, 100], 'robot_xyz': [200, 0, 10]},
            {'target_uv': [300, 100], 'robot_xyz': [250, 0, 10]},
            {'target_uv': [100, 300], 'robot_xyz': [200, 50, 10]},
            {'target_uv': [300, 300], 'robot_xyz': [250, 50, 10]}
        ]
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value=calibration_data):
                with patch('cv2.findHomography', return_value=(np.eye(3), np.ones(4))):
                    # Call the method
                    result = pyqt_gui_app.load_calibration()
                    
                    # Check result (should include transformation matrix)
        assert 'uv_to_xy_matrix' in result


def test_load_calibration_findHomography_none(pyqt_gui_app):
    """Test load_calibration when findHomography returns None"""
    # Set up data with 4 valid points
    calibration_data = {
        'calibration_points_raw': [
            {'target_uv': [100, 100], 'robot_xyz': [200, 0, 10]},
            {'target_uv': [300, 100], 'robot_xyz': [250, 0, 10]},
            {'target_uv': [100, 300], 'robot_xyz': [200, 50, 10]},
            {'target_uv': [300, 300], 'robot_xyz': [250, 50, 10]}
        ]
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value=calibration_data):
                with patch('cv2.findHomography', return_value=(None, None)):
                    # Call the method
                    result = pyqt_gui_app.load_calibration()
                    
                    # Check result (should not include transformation matrix)
        assert 'uv_to_xy_matrix' not in result


def test_load_calibration_findHomography_exception(pyqt_gui_app):
    """Test load_calibration when findHomography raises exception"""
    # Set up data with 4 valid points
    calibration_data = {
        'calibration_points_raw': [
            {'target_uv': [100, 100], 'robot_xyz': [200, 0, 10]},
            {'target_uv': [300, 100], 'robot_xyz': [250, 0, 10]},
            {'target_uv': [100, 300], 'robot_xyz': [200, 50, 10]},
            {'target_uv': [300, 300], 'robot_xyz': [250, 50, 10]}
        ]
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value=calibration_data):
                with patch('cv2.findHomography', side_effect=Exception("Test exception")):
                    # Call the method
                    result = pyqt_gui_app.load_calibration()
                    
                    # Check result (should not include transformation matrix)
        assert 'uv_to_xy_matrix' not in result


def test_load_calibration_default_values(pyqt_gui_app):
    """Test load_calibration fills in default values"""
    # Set up data with missing fields
    calibration_data = {
        'calibration_points_raw': [
            {'target_uv': [100, 100], 'robot_xyz': [200, 0, 10]},
            {'target_uv': [300, 100], 'robot_xyz': [250, 0, 10]},
            {'target_uv': [100, 300], 'robot_xyz': [200, 50, 10]},
            {'target_uv': [300, 300], 'robot_xyz': [250, 50, 10]}
        ],
        'touch_z': 10.0  # Only touch_z provided, not draw_z
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{}')):
            with patch('json.load', return_value=calibration_data):
                with patch('cv2.findHomography', return_value=(np.eye(3), np.ones(4))):
                    # Call the method
                    result = pyqt_gui_app.load_calibration()
                    
                    # Check result (should include draw_z from touch_z)
        assert 'draw_z' in result
        assert result['draw_z'] == 10.0
                    
                    # Check default value for symbol_size_mm
        assert 'symbol_size_mm' in result
        assert result['symbol_size_mm'] == DEFAULT_SYMBOL_SIZE_MM
                    
                    # Check default values for arm_workspace
        assert 'arm_workspace' in result
        assert result['arm_workspace']['min_x'] == 150
        assert result['arm_workspace']['max_x'] == 250
        assert result['arm_workspace']['min_y'] == -50
        assert result['arm_workspace']['max_y'] == 50


def test_update_game_state_game_over(pyqt_gui_app):
    """Test update_game_state when game is over"""
    # Set game_over to True
    pyqt_gui_app.game_over = True
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check no status update was made
    pyqt_gui_app.status_label.setText.assert_not_called()


def test_update_game_state_waiting_for_detection_symbol_detected(pyqt_gui_app):
    """Test update_game_state when waiting for detection and symbol is detected"""
    # Set state to waiting for detection
    pyqt_gui_app.waiting_for_detection = True
    pyqt_gui_app.ai_move_row = 1
    pyqt_gui_app.ai_move_col = 1
    
    # Set detected board with symbol at expected position
    pyqt_gui_app.camera_thread.last_board_state = [
        [0, 0, 0],
        [0, 2, 0],  # AI symbol (2) at position (1, 1)
        [0, 0, 0]
    ]
    
    # Mock check_game_end
    pyqt_gui_app.check_game_end = MagicMock()
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check waiting_for_detection is False
    assert not pyqt_gui_app.waiting_for_detection
    
    # Check current_turn changed to human player
    assert pyqt_gui_app.current_turn == pyqt_gui_app.human_player
    
    # Check the board was updated
    pyqt_gui_app.board_widget.update.assert_called()
    
    # Check check_game_end was called
    pyqt_gui_app.check_game_end.assert_called_once()


def test_update_game_state_waiting_for_detection_timeout(pyqt_gui_app):
    """Test update_game_state when waiting for detection and timeout occurs"""
    # Set state to waiting for detection with timeout
    pyqt_gui_app.waiting_for_detection = True
    pyqt_gui_app.detection_wait_time = 10.0  # More than max_detection_wait_time
    pyqt_gui_app.max_detection_wait_time = 5.0
    pyqt_gui_app.ai_move_retry_count = 1
    pyqt_gui_app.max_retry_count = 3
    pyqt_gui_app.ai_move_row = 1
    pyqt_gui_app.ai_move_col = 1
    
    # Mock draw_ai_symbol
    pyqt_gui_app.draw_ai_symbol = MagicMock(return_value=True)
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check retry count was incremented
    assert pyqt_gui_app.ai_move_retry_count == 2
    
    # Check draw_ai_symbol was called
    pyqt_gui_app.draw_ai_symbol.assert_called_once_with(1, 1)
    
    # Check waiting_for_detection is still True
    assert pyqt_gui_app.waiting_for_detection


def test_update_game_state_waiting_for_detection_max_retries(pyqt_gui_app):
    """Test update_game_state when waiting for detection and max retries reached"""
    # Set state to waiting for detection with max retries reached
    pyqt_gui_app.waiting_for_detection = True
    pyqt_gui_app.detection_wait_time = 10.0  # More than max_detection_wait_time
    pyqt_gui_app.max_detection_wait_time = 5.0
    pyqt_gui_app.ai_move_retry_count = 3
    pyqt_gui_app.max_retry_count = 3
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check waiting_for_detection is False
    assert not pyqt_gui_app.waiting_for_detection
    
    # Check current_turn changed to human player
    assert pyqt_gui_app.current_turn == pyqt_gui_app.human_player


def test_update_game_state_waiting_for_detection_draw_failure(pyqt_gui_app):
    """Test update_game_state when waiting for detection and drawing fails"""
    # Set state to waiting for detection with timeout
    pyqt_gui_app.waiting_for_detection = True
    pyqt_gui_app.detection_wait_time = 10.0  # More than max_detection_wait_time
    pyqt_gui_app.max_detection_wait_time = 5.0
    pyqt_gui_app.ai_move_retry_count = 1
    pyqt_gui_app.max_retry_count = 3
    pyqt_gui_app.ai_move_row = 1
    pyqt_gui_app.ai_move_col = 1
    
    # Mock draw_ai_symbol to fail
    pyqt_gui_app.draw_ai_symbol = MagicMock(return_value=False)
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check waiting_for_detection is False
    assert not pyqt_gui_app.waiting_for_detection
    
    # Check current_turn changed to human player
    assert pyqt_gui_app.current_turn == pyqt_gui_app.human_player


def test_update_game_state_ai_turn(pyqt_gui_app):
    """Test update_game_state when it's AI's turn"""
    # Set state to AI's turn
    pyqt_gui_app.current_turn = pyqt_gui_app.ai_player
    pyqt_gui_app.waiting_for_detection = False
    
    # Mock strategy_selector.get_move
    pyqt_gui_app.strategy_selector.get_move.return_value = (1, 1)
    
    # Mock draw_ai_symbol
    pyqt_gui_app.draw_ai_symbol = MagicMock(return_value=True)
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check ai_move_row and ai_move_col were set
    assert pyqt_gui_app.ai_move_row == 1
    assert pyqt_gui_app.ai_move_col == 1
    
    # Check draw_ai_symbol was called
    pyqt_gui_app.draw_ai_symbol.assert_called_once_with(1, 1)
    
    # Check waiting_for_detection is True
    assert pyqt_gui_app.waiting_for_detection


def test_update_game_state_ai_turn_draw_failure(pyqt_gui_app):
    """Test update_game_state when it's AI's turn and drawing fails"""
    # Set state to AI's turn
    pyqt_gui_app.current_turn = pyqt_gui_app.ai_player
    pyqt_gui_app.waiting_for_detection = False
    
    # Mock strategy_selector.get_move
    pyqt_gui_app.strategy_selector.get_move.return_value = (1, 1)
    
    # Mock draw_ai_symbol to fail
    pyqt_gui_app.draw_ai_symbol = MagicMock(return_value=False)
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check waiting_for_detection is False
    assert not pyqt_gui_app.waiting_for_detection


def test_update_game_state_ai_turn_no_arm(pyqt_gui_app):
    """Test update_game_state when it's AI's turn and no arm is connected"""
    # Set state to AI's turn
    pyqt_gui_app.current_turn = pyqt_gui_app.ai_player
    pyqt_gui_app.waiting_for_detection = False
    
    # Set no connected arm
    pyqt_gui_app.arm_thread.connected = False
    pyqt_gui_app.arm_controller.connected = False
    
    # Mock strategy_selector.get_move
    pyqt_gui_app.strategy_selector.get_move.return_value = (1, 1)
    
    # Call the method
    pyqt_gui_app.update_game_state()
    
    # Check board was updated directly
    assert pyqt_gui_app.board_widget.board[1][1] == pyqt_gui_app.ai_player
    
    # Check current_turn changed to human player
    assert pyqt_gui_app.current_turn == pyqt_gui_app.human_player


def test_closeEvent(pyqt_gui_app):
    """Test closeEvent method"""
    # Mock QApplication.processEvents
    with patch('app.main.pyqt_gui.QApplication.processEvents'):
        # Mock event
        event = MagicMock()
        
        # Call the method
        pyqt_gui_app.closeEvent(event)
        
        # Check event.accept was called
        event.accept.assert_called_once()
        
        # Check move_to_neutral_position was called
        pyqt_gui_app.move_to_neutral_position.assert_called_once()
        
        # Check arm_thread.stop was called
        pyqt_gui_app.arm_thread.stop.assert_called_once()


def test_closeEvent_with_exception(pyqt_gui_app):
    """Test closeEvent method with exception"""
    # Mock move_to_neutral_position to raise exception
    pyqt_gui_app.move_to_neutral_position = MagicMock(side_effect=Exception("Test exception"))
    
    # Mock QApplication.processEvents
    with patch('app.main.pyqt_gui.QApplication.processEvents'):
        # Mock event
        event = MagicMock()
        
        # Call the method
        pyqt_gui_app.closeEvent(event)
        
        # Check event.accept was still called despite exception
        event.accept.assert_called_once()


def test_init_ui(pyqt_gui_app):
    """Test init_ui method"""
    # Setup mock setCentralWidget
    pyqt_gui_app.setCentralWidget = MagicMock()
    
    # Call the method
    pyqt_gui_app.init_ui()
    
    # Check setCentralWidget was called
    pyqt_gui_app.setCentralWidget.assert_called_once()


def test_init_ui_no_setCentralWidget(pyqt_gui_app):
    """Test init_ui method without setCentralWidget"""
    # Setup app without setCentralWidget
    pyqt_gui_app.setCentralWidget = None
    
    # Call the method
    pyqt_gui_app.init_ui()
    
    # Should not crash


def test_init_ui_with_exception(pyqt_gui_app):
    """Test init_ui method with exception"""
    # Setup mock setCentralWidget that raises exception
    pyqt_gui_app.setCentralWidget = MagicMock(side_effect=Exception("Test exception"))
    
    # Call the method
    pyqt_gui_app.init_ui()
    
    # Should not crash


def test_handle_camera_changed(pyqt_gui_app):
    """Test handle_camera_changed method"""
    # Call the method
    pyqt_gui_app.handle_camera_changed(1)
    
    # Check camera_thread.stop was called
    pyqt_gui_app.camera_thread.stop.assert_called_once()


def test_handle_camera_changed_no_camera_thread(pyqt_gui_app):
    """Test handle_camera_changed method with no camera thread"""
    # Set no camera thread
    pyqt_gui_app.camera_thread = None
    
    # Call the method
    pyqt_gui_app.handle_camera_changed(1)
    
    # Should not crash


def test_handle_arm_connection_toggled_connect(pyqt_gui_app):
    """Test handle_arm_connection_toggled to connect arm"""
    # Setup arm_thread as not connected
    pyqt_gui_app.arm_thread.connected = False
    
    # Call the method to connect
    pyqt_gui_app.handle_arm_connection_toggled(True)
    
    # Check arm_thread.connect was called
    pyqt_gui_app.arm_thread.connect.assert_called_once()


def test_handle_arm_connection_toggled_disconnect(pyqt_gui_app):
    """Test handle_arm_connection_toggled to disconnect arm"""
    # Setup arm_thread as connected
    pyqt_gui_app.arm_thread.connected = True
    
    # Call the method to disconnect
    pyqt_gui_app.handle_arm_connection_toggled(False)
    
    # Check arm_thread.disconnect was called
    pyqt_gui_app.arm_thread.disconnect.assert_called_once()


def test_handle_arm_connection_toggled_no_arm_thread(pyqt_gui_app):
    """Test handle_arm_connection_toggled with no arm thread"""
    # Set no arm thread but keep arm controller
    pyqt_gui_app.arm_thread = None
    
    # Call the method
    pyqt_gui_app.handle_arm_connection_toggled(True)
    
    # Check arm_controller.connect was called as fallback
    pyqt_gui_app.arm_controller.connect.assert_called_once()


def test_handle_arm_connection_toggled_no_arm(pyqt_gui_app):
    """Test handle_arm_connection_toggled with no arm at all"""
    # Set no arm thread and no arm controller
    pyqt_gui_app.arm_thread = None
    pyqt_gui_app.arm_controller = None
    
    # Call the method
    pyqt_gui_app.handle_arm_connection_toggled(True)
    
    # Should not crash


def test_connect_signals_with_all_components(pyqt_gui_app):
    """Test connect_signals with all components available"""
    # Setup signal connections as MagicMocks
    pyqt_gui_app.camera_thread.frame_ready.connect = MagicMock()
    pyqt_gui_app.camera_thread.game_state_updated.connect = MagicMock()
    pyqt_gui_app.camera_thread.fps_updated.connect = MagicMock()
    
    pyqt_gui_app.board_widget.cell_clicked.connect = MagicMock()
    
    pyqt_gui_app.reset_button = MagicMock()
    pyqt_gui_app.reset_button.clicked.connect = MagicMock()
    
    pyqt_gui_app.debug_button = MagicMock()
    pyqt_gui_app.debug_button.clicked.connect = MagicMock()
    
    pyqt_gui_app.calibrate_button = MagicMock()
    pyqt_gui_app.calibrate_button.clicked.connect = MagicMock()
    
    pyqt_gui_app.park_button = MagicMock()
    pyqt_gui_app.park_button.clicked.connect = MagicMock()
    
    pyqt_gui_app.difficulty_slider = MagicMock()
    pyqt_gui_app.difficulty_slider.valueChanged.connect = MagicMock()
    
    # Call the method
    pyqt_gui_app.connect_signals()
    
    # Check all connect methods were called
    pyqt_gui_app.camera_thread.frame_ready.connect.assert_called_once()
    pyqt_gui_app.camera_thread.game_state_updated.connect.assert_called_once()
    pyqt_gui_app.camera_thread.fps_updated.connect.assert_called_once()
    
    pyqt_gui_app.board_widget.cell_clicked.connect.assert_called_once()
    
    pyqt_gui_app.reset_button.clicked.connect.assert_called_once()
    pyqt_gui_app.debug_button.clicked.connect.assert_called_once()
    pyqt_gui_app.calibrate_button.clicked.connect.assert_called_once()
    pyqt_gui_app.park_button.clicked.connect.assert_called_once()
    
    pyqt_gui_app.difficulty_slider.valueChanged.connect.assert_called_once()


def test_connect_signals_with_missing_components(pyqt_gui_app):
    """Test connect_signals with some components missing"""
    # Set no camera thread
    pyqt_gui_app.camera_thread = None
    
    # Set no board widget
    pyqt_gui_app.board_widget = None
    
    # Set no buttons
    pyqt_gui_app.reset_button = None
    pyqt_gui_app.debug_button = None
    pyqt_gui_app.calibrate_button = None
    pyqt_gui_app.park_button = None
    
    # Set no slider
    pyqt_gui_app.difficulty_slider = None
    
    # Call the method
    pyqt_gui_app.connect_signals()
    
    # Should not crash


def test_timer_setup(pyqt_gui_app):
    """Test timer_setup method"""
    # Setup mock timer
    pyqt_gui_app.update_timer = MagicMock()
    
    # Call the method
    pyqt_gui_app.timer_setup()
    
    # Check timer.timeout.connect and timer.start were called
    pyqt_gui_app.update_timer.timeout.connect.assert_called_once()
    pyqt_gui_app.update_timer.start.assert_called_once_with(100)  # 100ms interval


def test_update_camera_view_with_update_image(pyqt_gui_app):
    """Test update_camera_view with camera_view having update_image"""
    # Setup mock camera_view with update_image
    pyqt_gui_app.camera_view = MagicMock()
    pyqt_gui_app.camera_view.update_image = MagicMock()
    
    # Create a simple frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Call the method
    pyqt_gui_app.update_camera_view(frame)
    
    # Check update_image was called with the frame
    pyqt_gui_app.camera_view.update_image.assert_called_once_with(frame)


def test_update_camera_view_with_update_frame(pyqt_gui_app):
    """Test update_camera_view with camera_view having update_frame"""
    # Setup mock camera_view with update_frame
    pyqt_gui_app.camera_view = MagicMock()
    pyqt_gui_app.camera_view.update_frame = MagicMock()
    
    # Create a simple frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Call the method
    pyqt_gui_app.update_camera_view(frame)
    
    # Check update_frame was called with the frame
    pyqt_gui_app.camera_view.update_frame.assert_called_once_with(frame)


def test_update_camera_view_no_camera_view(pyqt_gui_app):
    """Test update_camera_view with no camera_view"""
    # Set no camera_view
    pyqt_gui_app.camera_view = None
    
    # Create a simple frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Call the method
    pyqt_gui_app.update_camera_view(frame)
    
    # Should not crash


# Camera Thread Tests

def test_camera_thread_init(camera_thread):
    """Test initialization of CameraThread"""
    # Check attributes
    assert camera_thread.camera_index == 0
    assert not camera_thread.running
    assert camera_thread.cap is None
    assert camera_thread.config.camera_index == 0
    assert camera_thread.config is not None
    assert camera_thread.config.device == 'cpu'
    assert camera_thread.last_board_state is None
    assert camera_thread.last_valid_time == 0
    assert camera_thread.detection_thread is not None


def test_camera_thread_init_with_custom_camera(qapp):
    """Test initialization with custom camera index"""
    with patch('app.core.detection_thread.DetectionThread'):
        camera_thread = CameraThread(camera_index=1)
        assert camera_thread.camera_index == 1
        assert camera_thread.config.camera_index == 1


def test_camera_thread_init_with_cuda(qapp):
    """Test initialization with CUDA available"""
    # Patch torch.cuda to return True
    with patch('torch.cuda.is_available', return_value=True), \
         patch('app.core.detection_thread.DetectionThread'):
        camera_thread = CameraThread()
        assert camera_thread.config.device == 'cuda'


def test_camera_thread_run_camera_not_opened(camera_thread):
    """Test run method when camera can't be opened"""
    # Mock cap.isOpened to return False
    camera_thread.cap = MagicMock()
    camera_thread.cap.isOpened.return_value = False
    
    # Call the method
    with pytest.raises(ConnectionError):
        camera_thread.run()


def test_camera_thread_run_with_read_failure(camera_thread):
    """Test run method when camera read fails"""
    # Set up cap
    camera_thread.cap = MagicMock()
    camera_thread.cap.isOpened.return_value = True
    
    # Set running to True initially to enter the loop
    camera_thread.running = True
    
    # Mock read to return (False, None) for first call, then stop thread
    camera_thread.stop = MagicMock(side_effect=lambda: setattr(camera_thread, 'running', False))
    camera_thread.cap.read.return_value = (False, None)
    
    # Call the method
    camera_thread.run()
    
    # No assertion needed since we're just testing that it doesn't crash


def test_camera_thread_cleanup(camera_thread):
    """Test cleanup method"""
    # Setup detection_thread and cap
    camera_thread.detection_thread = MagicMock()
    camera_thread.cap = MagicMock()
    camera_thread.cap.isOpened.return_value = True
    
    # Call the method
    camera_thread.cleanup()
    
    # Check detection_thread.stop and cap.release were called
    camera_thread.detection_thread.stop.assert_called_once()
    camera_thread.cap.release.assert_called_once()


def test_camera_thread_stop(camera_thread):
    """Test stop method"""
    # Mock cleanup
    camera_thread.cleanup = MagicMock()
    camera_thread.wait = MagicMock()
    
    # Set running to True
    camera_thread.running = True
    
    # Call the method
    camera_thread.stop()
    
    # Check running is False and cleanup and wait were called
    assert not camera_thread.running
    camera_thread.cleanup.assert_called_once()
    camera_thread.wait.assert_called_once()