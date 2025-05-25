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
        # Use the MockTicTacToeApp class from pyqt_gui_unified_helper
        from tests.unit.pyqt_gui_unified_helper import MockTicTacToeApp

        # Create app using the mock class
        app = MockTicTacToeApp()

        # Check default values
        assert app.human_player is None
        assert app.ai_player is None
        assert app.current_turn is None
        assert not app.game_over
        assert app.winner is None
        assert not app.waiting_for_detection

    def test_board_conversion_consolidation(self):
        """Test that board conversion consolidation works correctly"""
        # Use the MockTicTacToeApp class from pyqt_gui_unified_helper
        from tests.unit.pyqt_gui_unified_helper import MockTicTacToeApp

        # Create app using the mock class
        app = MockTicTacToeApp()

        # Test that the consolidated method is used throughout the codebase
        board_1d = ['X', 'O', '', 'X', 'O', '', '', 'X', 'O']
        expected_2d = [
            ['X', 'O', ''],
            ['X', 'O', ''],
            ['', 'X', 'O']
        ]

        # Test the consolidated method directly
        result = app._convert_board_1d_to_2d(board_1d)
        assert result == expected_2d

        # Test with None input
        result = app._convert_board_1d_to_2d(None)
        assert result is None

        # Test with empty list
        result = app._convert_board_1d_to_2d([])
        assert result == []

        # Test with string input (should return as-is)
        result = app._convert_board_1d_to_2d("invalid")
        assert result == "invalid"

        # Test with already 2D board (should return as-is)
        board_2d = [['X', 'O', ''], ['X', 'O', ''], ['', 'X', 'O']]
        result = app._convert_board_1d_to_2d(board_2d)
        assert result == board_2d

        # Test with invalid length 1D list (should return as-is)
        invalid_board = ['X', 'O']  # Wrong length
        result = app._convert_board_1d_to_2d(invalid_board)
        assert result == invalid_board



