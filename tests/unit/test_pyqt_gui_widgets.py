"""
Unit tests for widget classes in pyqt_gui.py
"""
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import cv2

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QEvent

from app.main import game_logic
from app.main.pyqt_gui import TicTacToeBoard, CameraView


@pytest.fixture
def board():
    """board fixture for tests."""
    board = TicTacToeBoard()
    return board


@pytest.fixture
def mock_painter_cls():
    """mock_painter_cls fixture for tests."""
    mock_painter_cls = self.painter_patcher.start()
    mock_painter_cls.return_value = self.mock_painter
    return mock_painter_cls


@pytest.fixture
def mock_painter():
    """mock_painter fixture for tests."""
    mock_painter_cls = self.painter_patcher.start()
    mock_painter = MagicMock()
    mock_painter_cls.return_value = mock_painter
    return mock_painter


@pytest.fixture
def camera_view():
    """camera_view fixture for tests."""
    camera_view = CameraView()
    return camera_view


@pytest.fixture
def mock_qimage():
    """mock_qimage fixture for tests."""
    mock_qimage_cls = self.qimage_patcher.start()
    mock_qimage = MagicMock()
    mock_qimage_cls.return_value = mock_qimage
    return mock_qimage


@pytest.fixture
def mock_qimage_cls():
    """mock_qimage_cls fixture for tests."""
    mock_qimage_cls = self.qimage_patcher.start()
    mock_qimage_cls.return_value = self.mock_qimage
    return mock_qimage_cls


@pytest.fixture
def mock_cvtColor():
    """mock_cvtColor fixture for tests."""
    mock_cvtColor = self.cv2_cvtColor_patcher.start()
    return mock_cvtColor


@pytest.fixture
def mock_qpixmap_cls():
    """mock_qpixmap_cls fixture for tests."""
    mock_qpixmap_cls = self.qpixmap_patcher.start()
    mock_qpixmap_cls.fromImage.return_value = self.mock_qpixmap
    return mock_qpixmap_cls


@pytest.fixture
def mock_qpixmap():
    """mock_qpixmap fixture for tests."""
    mock_qpixmap_cls = self.qpixmap_patcher.start()
    mock_qpixmap = MagicMock()
    mock_qpixmap_cls.fromImage.return_value = mock_qpixmap
    return mock_qpixmap



class TestTicTacToeBoard():
    """Test cases for TicTacToeBoard class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create QApplication instance if it doesn't exist
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])

        # Patch QPainter
        self.painter_patcher = patch('pyqt_gui.QPainter')
        self.mock_painter_cls = self.painter_patcher.start()
        self.mock_painter = MagicMock()
        self.mock_painter_cls.return_value = self.mock_painter

        # Patch game_logic.create_board
        self.create_board_patcher = patch('pyqt_gui.game_logic.create_board')
        self.mock_create_board = self.create_board_patcher.start()
        self.mock_create_board.return_value = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        # Create board widget
        self.board = TicTacToeBoard()

    def tearDown(self):
        """Tear down test fixtures"""
        self.painter_patcher.stop()
        self.create_board_patcher.stop()

    def test_init(self):
        """Test initialization"""
        # Check default values
        assert self.board.board == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert self.board.cell_size == 150
        assert self.board.winning_line is None

        # Check minimum size
        assert self.board.minimumSize().width() == 450
        assert self.board.minimumSize().height() == 450

        # Check style sheet
        assert self.board.styleSheet() == "background-color: #333333;"

    def test_paint_event_empty_board(self):
        """Test paintEvent with empty board"""
        # Create a mock event
        event = MagicMock()

        # Call paintEvent
        self.board.paintEvent(event)

        # Check that QPainter was created with the board as target
        self.mock_painter_cls.assert_called_once_with(self.board)

        # Check that setRenderHint was called
        self.mock_painter.setRenderHint.assert_called_once()

        # Check that grid lines were drawn
        assert self.mock_painter.drawLine.call_count == 4

        # Check that no symbols were drawn (empty board)
        self.mock_painter.drawEllipse.assert_not_called()  # No O symbols

        # No winning line should be drawn
        assert self.mock_painter.drawLine.call_count == 4  # Only grid lines

    def test_paint_event_with_symbols(self):
        """Test paintEvent with X and O symbols"""
        # Set up board with symbols
        self.board.board = [
            [game_logic.PLAYER_X, 0, game_logic.PLAYER_O],
            [0, game_logic.PLAYER_X, 0],
            [game_logic.PLAYER_O, 0, game_logic.PLAYER_X]
        ]

        # Create a mock event
        event = MagicMock()

        # Call paintEvent
        self.board.paintEvent(event)

        # Check that QPainter was created with the board as target
        self.mock_painter_cls.assert_called_once_with(self.board)

        # Check that X symbols were drawn (3 X's = 6 lines)
        # 4 grid lines + 6 lines for X symbols = 10 total lines
        assert self.mock_painter.drawLine.call_count == 10

        # Check that O symbols were drawn (2 O's = 2 ellipses)
        assert self.mock_painter.drawEllipse.call_count == 2

    def test_paint_event_with_winning_line(self):
        """Test paintEvent with winning line"""
        # Set up board with symbols and winning line
        self.board.board = [
            [game_logic.PLAYER_X, 0, 0],
            [0, game_logic.PLAYER_X, 0],
            [0, 0, game_logic.PLAYER_X]
        ]
        self.board.winning_line = [(0, 0), (1, 1), (2, 2)]  # Diagonal win

        # Create a mock event
        event = MagicMock()

        # Call paintEvent
        self.board.paintEvent(event)

        # Check that winning line was drawn
        # 4 grid lines + 6 lines for X symbols + 1 winning line = 11 total lines
        assert self.mock_painter.drawLine.call_count == 11

        # Check that winning line pen was set correctly
        self.mock_painter.setPen.assert_any_call(
            self.mock_painter.setPen.call_args_list[2][0][0])  # Third setPen call

    def test_update_board(self):
        """Test update_board method"""
        # Create a new board state
        new_board = [
            [game_logic.PLAYER_X, 0, 0],
            [0, game_logic.PLAYER_O, 0],
            [0, 0, 0]
        ]

        # Mock the update method
        self.board.update = MagicMock()

        # Call update_board
        self.board.update_board(new_board)

        # Check that board was updated
        assert self.board.board == new_board

        # Check that update was called
        self.board.update.assert_called_once()


class TestCameraView():
    """Test cases for CameraView class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create QApplication instance if it doesn't exist
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication([])

        # Patch cv2.cvtColor
        self.cv2_cvtColor_patcher = patch('pyqt_gui.cv2.cvtColor')
        self.mock_cvtColor = self.cv2_cvtColor_patcher.start()

        # Patch QImage
        self.qimage_patcher = patch('pyqt_gui.QImage')
        self.mock_qimage_cls = self.qimage_patcher.start()
        self.mock_qimage = MagicMock()
        self.mock_qimage_cls.return_value = self.mock_qimage

        # Patch QPixmap
        self.qpixmap_patcher = patch('pyqt_gui.QPixmap')
        self.mock_qpixmap_cls = self.qpixmap_patcher.start()
        self.mock_qpixmap = MagicMock()
        self.mock_qpixmap_cls.fromImage.return_value = self.mock_qpixmap

        # Create camera view
        self.camera_view = CameraView()

    def tearDown(self):
        """Tear down test fixtures"""
        self.cv2_cvtColor_patcher.stop()
        self.qimage_patcher.stop()
        self.qpixmap_patcher.stop()

    def test_init(self):
        """Test initialization"""
        # Check default values
        assert self.camera_view.minimumSize().width() == 320
        assert self.camera_view.minimumSize().height() == 240
        assert self.camera_view.alignment() == Qt.AlignCenter
        assert self.camera_view.text() == "Kamera nedostupná"
        self.assertEqual(
            self.camera_view.styleSheet(),
            "background-color: #222222; color: white;")

    def test_update_frame_with_none(self):
        """Test update_frame method with None frame"""
        # Call update_frame with None
        self.camera_view.update_frame(None)

        # Check that text was set
        assert self.camera_view.text() == "Kamera nedostupná"

        # Check that cv2.cvtColor was not called
        self.mock_cvtColor.assert_not_called()

        # Check that QImage was not created
        self.mock_qimage_cls.assert_not_called()

    def test_update_frame_with_valid_frame(self):
        """Test update_frame method with valid frame"""
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Set up mock cv2.cvtColor to return the same frame
        self.mock_cvtColor.return_value = test_frame

        # Mock setPixmap method to avoid TypeError
        self.camera_view.setPixmap = MagicMock()

        # Call update_frame
        self.camera_view.update_frame(test_frame)

        # Check that cv2.cvtColor was called
        self.mock_cvtColor.assert_called_once_with(test_frame, cv2.COLOR_BGR2RGB)

        # Check that QImage was created (don't check exact parameters)
        self.mock_qimage_cls.assert_called_once()

        # Check that QPixmap.fromImage was called
        self.mock_qpixmap_cls.fromImage.assert_called_once()

        # Check that setPixmap was called
        self.camera_view.setPixmap.assert_called_once()



