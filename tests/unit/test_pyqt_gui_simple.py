"""
Simple unit tests for pyqt_gui.py
"""
import pytest
from unittest.mock import patch
import sys

from PyQt5.QtWidgets import QApplication
from app.main.pyqt_gui import TicTacToeApp, TicTacToeBoard, CameraView, CameraThread


class TestPyQtGuiSimple():
    """Simple test cases for pyqt_gui.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Create QApplication instance if it doesn't exist
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])

    def test_tic_tac_toe_board_init(self):
        """Test TicTacToeBoard initialization"""
        # Patch game_logic.create_board
        with patch('game_logic.create_board') as mock_create_board:
            # Set up mock to return empty board
            empty_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            mock_create_board.return_value = empty_board

            # Create board
            board = TicTacToeBoard()

            # Check default values
        assert board.board == empty_board
        assert board.cell_size == 150
        assert board.winning_line is None

            # Check minimum size
        assert board.minimumSize().width() == 450
        assert board.minimumSize().height() == 450

    def test_camera_view_init(self):
        """Test CameraView initialization"""
        # Create camera view
        camera_view = CameraView()

        # Check default values
        assert camera_view.minimumSize().width() == 320
        assert camera_view.minimumSize().height() == 240
        assert camera_view.text() == "Kamera nedostupná"

    def test_camera_thread_init(self):
        """Test CameraThread initialization"""
        # Patch cv2.VideoCapture
        with patch('pyqt_gui.cv2.VideoCapture'):
            # Patch DetectionThread
            with patch('pyqt_gui.DetectionThread'):
                # Patch torch
                with patch('pyqt_gui.torch') as mock_torch:
                    # Set up torch.cuda.is_available to return False
                    mock_torch.cuda.is_available.return_value = False

                    # Set up torch.backends.mps.is_available to return False
                    mock_torch.backends.mps.is_available.return_value = False

                    # Create thread
                    thread = CameraThread()

                    # Check default values
        assert thread.camera_index == 0  # Default index
        assert not thread.running
                    # CPU device when no GPU/MPS available
        assert thread.config.device == 'cpu'

    def test_tic_tac_toe_app_init(self):
        """Test TicTacToeApp initialization"""
        # Use the MockTicTacToeApp class from test_pyqt_gui_app_helper
        from tests.unit.test_pyqt_gui_app_helper import MockTicTacToeApp

        # Create app using the mock class
        app = MockTicTacToeApp()

        # Check default values
        assert app.human_player is None
        assert app.ai_player is None
        assert app.current_turn is None
        assert not app.game_over
        assert app.winner is None
        assert not app.waiting_for_detection



