"""
Unit tests for initialization of TicTacToeApp
"""
import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

from app.main import game_logic
from app.main.pyqt_gui import TicTacToeApp
from tests.conftest_common import MockTicTacToeApp, qt_app


@pytest.fixture
def mock_app():
    """mock_app fixture for tests."""
    app = MockTicTacToeApp()
    
    # Add signal connections for UI components
    app.reset_button.clicked.connect = MagicMock()
    app.park_button.clicked.connect = MagicMock()
    app.calibrate_button.clicked.connect = MagicMock()
    app.debug_button.clicked.connect = MagicMock()
    app.difficulty_slider.valueChanged.connect = MagicMock()
    app.timer.timeout.connect = MagicMock()
    app.timer.start = MagicMock()
    
    return app


class TestPyQtGuiInit:
    """Test cases for initialization of TicTacToeApp"""

    def test_init_default_values(self, mock_app):
        """Test default initialization values"""
        assert mock_app.debug_mode == False
        assert mock_app.camera_index == 0
        assert mock_app.difficulty == 0.5
        assert mock_app.game_over == False
        assert mock_app.winner is None
        assert mock_app.waiting_for_detection == False
        assert mock_app.ai_move_row is None
        assert mock_app.ai_move_col is None
        assert mock_app.ai_move_retry_count == 0
        assert mock_app.max_retry_count == 3
        assert mock_app.detection_wait_time == 0
        assert mock_app.max_detection_wait_time == 5.0
        assert mock_app.neutral_position == {"x": 200, "y": 0, "z": 100}

    @patch.object(TicTacToeApp, '_setup_ui')
    @patch.object(TicTacToeApp, '_setup_camera')
    @patch.object(TicTacToeApp, '_setup_arm')
    @patch.object(TicTacToeApp, '_load_calibration')
    @patch.object(TicTacToeApp, 'reset_game')
    def test_full_init(self, mock_reset, mock_load_calib, mock_setup_arm, 
                       mock_setup_camera, mock_setup_ui, qt_app):
        """Test full initialization with patched methods"""
        app = TicTacToeApp(debug_mode=True, camera_index=1, difficulty=0.8)
        
        # Verify constructor parameters were set
        assert app.debug_mode == True
        assert app.camera_index == 1
        assert app.difficulty == 0.8
        
        # Verify initialization methods were called
        mock_setup_ui.assert_called_once()
        mock_setup_camera.assert_called_once()
        mock_setup_arm.assert_called_once()
        mock_load_calib.assert_called_once()
        mock_reset.assert_called_once()

    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_calibration_success(self, mock_file, mock_json_load, mock_app):
        """Test successful loading of calibration data"""
        # Set up mock JSON data
        mock_json_load.return_value = {
            "arm_workspace": {
                "min_x": 100,
                "max_x": 300,
                "min_y": -100,
                "max_y": 100
            },
            "neutral_position": {
                "x": 200,
                "y": 0,
                "z": 100
            }
        }
        
        # Add _load_calibration method to our mock
        mock_app._load_calibration = lambda: TicTacToeApp._load_calibration(mock_app)
        
        # Call the method
        mock_app._load_calibration()
        
        # Verify calibration data was loaded
        assert mock_app.calibration_data["arm_workspace"]["min_x"] == 100
        assert mock_app.neutral_position["x"] == 200
        assert mock_file.call_count > 0

    @patch('builtins.open')
    def test_load_calibration_missing_file(self, mock_open, mock_app):
        """Test loading calibration when file doesn't exist"""
        # Make open raise FileNotFoundError
        mock_open.side_effect = FileNotFoundError()
        
        # Add _load_calibration method to our mock
        mock_app._load_calibration = lambda: TicTacToeApp._load_calibration(mock_app)
        mock_app.status_label = MagicMock()
        
        # Call the method
        mock_app._load_calibration()
        
        # Verify default values are used
        assert mock_app.calibration_data == {}
        assert mock_app.neutral_position == {"x": 200, "y": 0, "z": 100}
        mock_app.status_label.setText.assert_called_with("Kalibrační data nenalezena, používám výchozí hodnoty.")

    @patch('builtins.open')
    def test_load_calibration_json_error(self, mock_open, mock_app):
        """Test loading calibration with invalid JSON"""
        # Make json.load raise ValueError
        mock_open.return_value.__enter__.return_value.read.return_value = "{"
        
        # Add _load_calibration method to our mock
        mock_app._load_calibration = lambda: TicTacToeApp._load_calibration(mock_app)
        mock_app.status_label = MagicMock()
        
        # Call the method
        mock_app._load_calibration()
        
        # Verify default values are used
        assert mock_app.calibration_data == {}
        assert mock_app.neutral_position == {"x": 200, "y": 0, "z": 100}
        mock_app.status_label.setText.assert_called_with("Chyba při načítání kalibračních dat: .")

    def test_reset_game(self, mock_app):
        """Test game reset functionality"""
        # Set up initial state
        mock_app.board_widget.board = [
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 2]
        ]
        mock_app.game_over = True
        mock_app.winner = 1
        mock_app.human_player = 2
        mock_app.ai_player = 1
        
        # Add reset_game method
        mock_app.reset_game = lambda: TicTacToeApp.reset_game(mock_app)
        
        # Call reset method
        mock_app.reset_game()
        
        # Verify game was reset
        assert mock_app.game_over == False
        assert mock_app.winner is None
        assert mock_app.human_player == 1  # X
        assert mock_app.ai_player == 2     # O
        assert mock_app.current_turn == 1   # X starts
        
        # Verify board was reset
        mock_app.board_widget.reset_board.assert_called_once()
        
    @patch.object(TicTacToeApp, '_setup_strategy_selector')
    def test_setup_ui(self, mock_setup_strategy, mock_app, qt_app):
        """Test UI setup"""
        # Add _setup_ui method
        mock_app._setup_ui = lambda: TicTacToeApp._setup_ui(mock_app)
        
        # Call setup method
        mock_app._setup_ui()
        
        # Verify strategy selector was set up
        mock_setup_strategy.assert_called_once()
        
        # Verify signal connections were made
        assert mock_app.reset_button.clicked.connect.called
        assert mock_app.park_button.clicked.connect.called
        assert mock_app.calibrate_button.clicked.connect.called
        assert mock_app.debug_button.clicked.connect.called
        assert mock_app.difficulty_slider.valueChanged.connect.called
        assert mock_app.timer.timeout.connect.called
        
        # Verify timer was started
        assert mock_app.timer.start.called