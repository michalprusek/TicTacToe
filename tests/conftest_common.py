"""
Common test fixtures for use across test modules.
This module helps reduce code duplication by providing shared fixtures.
"""
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import cv2

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

from app.core.config import GameDetectorConfig
from app.core.game_state import GameState
from app.main import game_logic
from app.main.pyqt_gui import TicTacToeApp


@pytest.fixture
def config():
    """Return a basic GameDetectorConfig for testing."""
    return GameDetectorConfig()


@pytest.fixture
def game_detector_config():
    """Return a basic GameDetectorConfig for testing."""
    return GameDetectorConfig(
        camera_index=0,
        disable_autofocus=True,
        detect_model_path="weights/best_detection.pt",
        pose_model_path="weights/best_pose.pt",
        device="cpu"
    )


@pytest.fixture
def mock_detector():
    """Create a mock detector for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_yolo():
    """Create a mock YOLO model for testing."""
    mock = MagicMock()
    mock.names = {0: 'O', 1: 'X'}
    mock.predict.return_value = []

    # For to() method
    mock.to.return_value = mock

    return mock


@pytest.fixture
def mock_cv2_videocapture():
    """Create a mock VideoCapture for testing."""
    mock = MagicMock()
    mock.isOpened.return_value = True
    mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_AUTOFOCUS: 0
    }.get(prop, 0)
    return mock


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_keypoints():
    """Create sample grid keypoints for testing."""
    return np.array([
        [100, 100], [200, 100], [300, 100], [400, 100],
        [100, 200], [200, 200], [300, 200], [400, 200],
        [100, 300], [200, 300], [300, 300], [400, 300],
        [100, 400], [200, 400], [300, 400], [400, 400]
    ])


@pytest.fixture
def sample_game_state(sample_frame, sample_keypoints):
    """Create a sample GameState for testing."""
    game_state = GameState()

    # Create sample homography
    sample_homography = np.eye(3)

    # Update the game state with the sample data
    game_state.update_from_detection(
        sample_frame,
        sample_keypoints,
        sample_homography,
        [],  # No symbols
        {1: 'X', 2: 'O'},  # Class ID to player mapping
        123.45  # Timestamp
    )

    return game_state


@pytest.fixture
def detector_setup(mock_detector, sample_frame, sample_game_state):
    """Set up a mock detector with sample frame and game state."""
    # Mock the detector's process_frame method
    mock_detector.process_frame.return_value = (sample_frame, sample_game_state)
    return mock_detector


@pytest.fixture
def qt_app():
    """
    QApplication fixture for PyQt GUI tests.
    
    This fixture creates a QApplication instance if one doesn't exist already,
    or returns the existing instance. It's essential for any test that uses PyQt
    widgets or signals.
    
    Returns:
        QApplication: A QApplication instance for the test.
    """
    app_instance = QApplication.instance()
    if app_instance is None:
        app_instance = QApplication(sys.argv)
    yield app_instance


class MockTicTacToeApp(QMainWindow):
    """
    Mock version of TicTacToeApp for testing.
    
    This class provides a standardized mock implementation of TicTacToeApp
    that can be used across different test modules. It initializes all the
    necessary attributes and mocks UI components to allow for testing without
    creating actual GUI elements.
    """

    def __init__(self, debug_mode=False, camera_index=0, difficulty=0.5):
        """
        Initialize the mock TicTacToeApp.
        
        Args:
            debug_mode (bool): Whether to enable debug mode
            camera_index (int): Camera index to use
            difficulty (float): AI difficulty level
        """
        # Initialize QMainWindow
        super().__init__()

        # Set up necessary attributes manually
        self.debug_mode = debug_mode
        self.camera_index = camera_index
        self.difficulty = difficulty
        self.human_player = None
        self.ai_player = None
        self.current_turn = None
        self.game_over = False
        self.winner = None
        self.waiting_for_detection = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.ai_move_retry_count = 0
        self.max_retry_count = 3
        self.detection_wait_time = 0
        self.max_detection_wait_time = 5.0
        self.arm_thread = None
        self.arm_controller = None
        self.camera_thread = None
        self.calibration_data = {}
        self.neutral_position = {"x": 200, "y": 0, "z": 100}

        # Mock UI components
        self.board_widget = MagicMock()
        self.board_widget.board = game_logic.create_board()
        self.status_label = MagicMock()
        self.difficulty_value_label = MagicMock()
        self.difficulty_slider = MagicMock()
        self.camera_view = MagicMock()
        self.reset_button = MagicMock()
        self.park_button = MagicMock()
        self.calibrate_button = MagicMock()
        self.debug_button = MagicMock()

        # Mock strategy selector
        self.strategy_selector = MagicMock()

        # Mock debug window
        self.debug_window = MagicMock()
        
        # Mock timer
        self.timer = MagicMock()


@pytest.fixture
def mock_tic_tac_toe_app():
    """
    Fixture providing a MockTicTacToeApp instance for testing.
    
    Returns:
        MockTicTacToeApp: A mock instance with all necessary attributes and mocked components.
    """
    return MockTicTacToeApp()


@pytest.fixture
def patched_tic_tac_toe_app():
    """
    Fixture providing a patched TicTacToeApp instance for testing.
    
    This fixture creates a TicTacToeApp instance with its __init__ method patched,
    and then manually sets up necessary attributes. This approach is useful when you
    want to test actual methods of TicTacToeApp but don't want to initialize GUI components.
    
    Returns:
        TicTacToeApp: A patched instance with attributes set up for testing.
    """
    with patch.object(TicTacToeApp, '__init__', return_value=None):
        app_instance = TicTacToeApp()
        
        # Set up necessary attributes manually
        app_instance.human_player = game_logic.PLAYER_X
        app_instance.ai_player = game_logic.PLAYER_O
        app_instance.current_turn = game_logic.PLAYER_O
        app_instance.game_over = False
        app_instance.winner = None
        app_instance.waiting_for_detection = False
        app_instance.ai_move_row = None
        app_instance.ai_move_col = None
        app_instance.ai_move_retry_count = 0
        app_instance.max_retry_count = 3
        app_instance.detection_wait_time = 0
        app_instance.max_detection_wait_time = 5.0
        
        # Mock board and status label
        app_instance.board_widget = MagicMock()
        app_instance.board_widget.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        app_instance.status_label = MagicMock()
        app_instance.camera_thread = MagicMock()
        app_instance.camera_view = MagicMock()
        
        # Mock strategy selector
        app_instance.strategy_selector = MagicMock()
        
        # Set up calibration data
        app_instance.calibration_data = {}
        app_instance.neutral_position = {"x": 200, "y": 0, "z": 100}
        
        # Mock arm controller
        app_instance.arm_controller = MagicMock()
        
        return app_instance


class PyQtGuiTestCaseBase:
    """
    Base class for PyQt GUI test cases.
    
    This class provides common methods for setting up Qt tests.
    It replaces the old PyQtGuiTestCase in pyqt_gui_unified_helper.py.
    """
    
    @staticmethod
    def create_app_instance():
        """Create a QApplication instance if one doesn't exist."""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app
    
    @staticmethod
    def setup_patches():
        """Set up common patches for testing."""
        patches = []
        
        # Patch cv2.VideoCapture to avoid actual camera usage
        cv2_patch = patch('cv2.VideoCapture')
        mock_video_capture = cv2_patch.start()
        mock_instance = mock_video_capture.return_value
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        patches.append(cv2_patch)
        
        # Patch arm controller to avoid hardware access
        arm_patch = patch('app.main.arm_controller.ArmController')
        mock_arm_controller = arm_patch.start()
        mock_arm_instance = mock_arm_controller.return_value
        mock_arm_instance.connected = True
        mock_arm_instance.move_to.return_value = True
        mock_arm_instance.go_to_position.return_value = True
        patches.append(arm_patch)
        
        # Patch game_logic.minimax to speed up tests
        minimax_patch = patch('app.main.game_logic.minimax')
        mock_minimax = minimax_patch.start()
        mock_minimax.return_value = (0, 0)
        patches.append(minimax_patch)
        
        return patches
        
    @classmethod
    def create_test_app(cls):
        """
        Create a test app with all necessary patches.
        
        Returns:
            tuple: (mock_app, patches) - the app instance and any patches that should be stopped
        """
        # Create QApplication instance
        cls.create_app_instance()
        
        # Set up patches
        patches = cls.setup_patches()
        
        # Create mock app
        app = MockTicTacToeApp()
        
        return app, patches


class AssertionUtils:
    """
    Common assertion utilities for testing.

    This class provides standardized assertion methods to reduce code duplication
    in test assertions across test files.
    """

    @staticmethod
    def assert_status_message(app, message):
        """
        Assert that the status label shows the expected message.

        Args:
            app: The app instance with a status_label mock
            message: The expected message
        """
        app.status_label.setText.assert_called_with(message)

    @staticmethod
    def assert_status_message_once(app, message):
        """
        Assert that the status label was called once with the expected message.

        Args:
            app: The app instance with a status_label mock
            message: The expected message
        """
        app.status_label.setText.assert_called_once_with(message)

    @staticmethod
    def assert_arm_not_connected_message(app):
        """
        Assert that the status label shows the arm not connected message.

        Args:
            app: The app instance with a status_label mock
        """
        app.status_label.setText.assert_called_with("Robotická ruka není připojena!")

    @staticmethod
    def assert_drawing_failed_message(app):
        """
        Assert that the status label shows the drawing failed message.

        Args:
            app: The app instance with a status_label mock
        """
        app.status_label.setText.assert_called_with("Chyba při kreslení symbolu!")

    @staticmethod
    def assert_neutral_position_success(app):
        """
        Assert that the status label shows the neutral position success message.

        Args:
            app: The app instance with a status_label mock
        """
        app.status_label.setText.assert_called_with("Ruka v neutrální pozici")

    @staticmethod
    def assert_neutral_position_failed(app):
        """
        Assert that the status label shows the neutral position failed message.

        Args:
            app: The app instance with a status_label mock
        """
        app.status_label.setText.assert_called_with("Nepodařilo se přesunout ruku do neutrální pozici")

    @staticmethod
    def assert_drawing_called(app, row, col, method_name='draw_ai_symbol'):
        """
        Assert that the drawing method was called with the expected coordinates.

        Args:
            app: The app instance
            row: The expected row
            col: The expected column
            method_name: The drawing method name to check
        """
        draw_method = getattr(app, method_name)
        draw_method.assert_called_once_with(row, col)


class GameEndCheckTestUtils:
    """
    Utility methods for testing game end check functionality.

    This class provides standardized test methods for common game end check scenarios
    to reduce code duplication across test files.
    """
    
    @staticmethod
    def test_check_game_end_no_winner(app):
        """Test check_game_end method with no winner."""
        # Set up board with no winner
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        # Call check_game_end
        app.check_game_end()

        # Check that game_over is still False
        assert not app.game_over

        # Check that winner is still None
        assert app.winner is None

        # Check that status was not updated
        app.status_label.setText.assert_not_called()

        # Check that draw_winning_line was not called
        app.draw_winning_line.assert_not_called()
    
    @staticmethod
    def test_check_game_end_human_wins(app):
        """Test check_game_end method with human winning."""
        # Set up board with human winning
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_O, 0],
            [0, 0, 0]
        ]

        # Call check_game_end
        app.check_game_end()

        # Check that game_over is True
        assert app.game_over

        # Check that winner is human player
        assert app.winner == game_logic.PLAYER_X

        # Check that status was updated
        app.status_label.setText.assert_called_once_with("Vyhráli jste!")

        # Check that winning_line was set
        assert app.board_widget.winning_line is not None

        # Check that draw_winning_line was not called (human win)
        app.draw_winning_line.assert_not_called()
    
    @staticmethod
    def test_check_game_end_ai_wins(app):
        """Test check_game_end method with AI winning."""
        # Set up board with AI winning
        app.board_widget.board = [
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, 0],
            [game_logic.PLAYER_O, 0, 0]
        ]

        # Call check_game_end
        app.check_game_end()

        # Check that game_over is True
        assert app.game_over

        # Check that winner is AI player
        assert app.winner == game_logic.PLAYER_O

        # Check that status was updated
        app.status_label.setText.assert_called_once_with("Vyhrál robot!")

        # Check that winning_line was set
        assert app.board_widget.winning_line is not None

        # Check that draw_winning_line was called (AI win)
        app.draw_winning_line.assert_called_once()
    
    @staticmethod
    def test_check_game_end_tie(app):
        """Test check_game_end method with tie."""
        # Set up board with tie
        app.board_widget.board = [
            [game_logic.PLAYER_X, game_logic.PLAYER_O, game_logic.PLAYER_X],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_O],
            [game_logic.PLAYER_O, game_logic.PLAYER_X, game_logic.PLAYER_O]
        ]

        # Call check_game_end
        app.check_game_end()

        # Check that game_over is True
        assert app.game_over

        # Check that winner is TIE
        assert app.winner == game_logic.TIE

        # Check that status was updated
        app.status_label.setText.assert_called_once_with("Remíza!")

        # In our implementation, we don't set winning_line to None for ties
        # so we don't check it here

        # Check that draw_winning_line was not called (tie)
        app.draw_winning_line.assert_not_called()
    
    @staticmethod
    def test_check_game_end_already_over(app):
        """Test check_game_end method when game is already over."""
        # Set up game as already over
        app.game_over = True
        app.winner = game_logic.PLAYER_X

        # Reset mocks
        app.status_label.setText.reset_mock()
        app.draw_winning_line.reset_mock()

        # Call check_game_end
        app.check_game_end()

        # Check that status was not updated
        app.status_label.setText.assert_not_called()

        # Check that draw_winning_line was not called
        app.draw_winning_line.assert_not_called()


class DrawingTestUtils:
    """
    Utility methods for testing drawing functionality.

    This class provides standardized test methods for common drawing scenarios
    to reduce code duplication across test files.
    """

    @staticmethod
    def prepare_draw_x_test(app, coords=(200, 0)):
        """
        Set up a test for drawing an X symbol.

        Args:
            app: The app instance
            coords: The coordinates to return from get_cell_coordinates_from_yolo
        """
        app.ai_player = game_logic.PLAYER_X
        app.get_cell_coordinates_from_yolo = MagicMock(return_value=coords)

    @staticmethod
    def prepare_draw_o_test(app, coords=(200, 0)):
        """
        Set up a test for drawing an O symbol.

        Args:
            app: The app instance
            coords: The coordinates to return from get_cell_coordinates_from_yolo
        """
        app.ai_player = game_logic.PLAYER_O
        app.get_cell_coordinates_from_yolo = MagicMock(return_value=coords)

    @staticmethod
    def prepare_arm_thread(app, success=True):
        """
        Set up an arm thread for drawing tests.

        Args:
            app: The app instance
            success: Whether drawing operations should succeed
        """
        app.arm_thread = MagicMock()
        app.arm_thread.connected = True
        app.arm_thread.draw_x = MagicMock(return_value=success)
        app.arm_thread.draw_o = MagicMock(return_value=success)

    @staticmethod
    def prepare_arm_controller(app, success=True):
        """
        Set up an arm controller for drawing tests.

        Args:
            app: The app instance
            success: Whether drawing operations should succeed
        """
        app.arm_controller = MagicMock()
        app.arm_controller.connected = True
        app.arm_controller.draw_x = MagicMock(return_value=success)
        app.arm_controller.draw_o = MagicMock(return_value=success)

    @staticmethod
    def verify_draw_x(app, x, y, size):
        """
        Verify that draw_x was called correctly.

        Args:
            app: The app instance
            x: The expected x coordinate
            y: The expected y coordinate
            size: The expected size
        """
        draw_method = app.arm_thread.draw_x if app.arm_thread else app.arm_controller.draw_x
        draw_method.assert_called_once()

        # Check that args match
        args = draw_method.call_args[0]
        assert args[0] == x
        assert args[1] == y

        # Check that size matches (if provided)
        if len(args) > 2:
            assert args[2] == size

    @staticmethod
    def verify_draw_o(app, x, y, size):
        """
        Verify that draw_o was called correctly.

        Args:
            app: The app instance
            x: The expected x coordinate
            y: The expected y coordinate
            size: The expected size
        """
        draw_method = app.arm_thread.draw_o if app.arm_thread else app.arm_controller.draw_o
        draw_method.assert_called_once()

        # Check that args match
        args = draw_method.call_args[0]
        assert args[0] == x
        assert args[1] == y

        # Check that size matches (if provided)
        if len(args) > 2:
            assert args[2] == size

    @staticmethod
    def verify_move_to_neutral(app):
        """
        Verify that move_to_neutral was called.

        Args:
            app: The app instance
        """
        app.move_to_neutral_position.assert_called_once()
class CameraTestUtils:
    """
    Utility methods for testing camera functionality.
    
    This class provides standardized test methods for camera and detection
    testing to reduce code duplication.
    """
    
    @staticmethod
    def create_sample_frame():
        """
        Create a sample frame for testing.
        
        Returns:
            numpy.ndarray: A black frame of 640x480 size
        """
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @staticmethod
    def prepare_mock_camera_thread(app):
        """
        Prepare a mock camera thread with common functionality.
        
        Args:
            app: The app instance
        """
        app.camera_thread = MagicMock()
        app.camera_thread.frame_ready = MagicMock()
        app.camera_thread.fps_updated = MagicMock()
        app.camera_thread.last_frame = CameraTestUtils.create_sample_frame()
        app.camera_thread.last_board_state = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        app.camera_thread.startThread = MagicMock()
        app.camera_thread.stopThread = MagicMock()
    
    @staticmethod
    def prepare_mock_detection_thread():
        """
        Prepare a mock detection thread with common functionality.
        
        Returns:
            MagicMock: A detection thread mock
        """
        mock_detection_thread = MagicMock()
        mock_detection_thread.process_frame = MagicMock()
        mock_detection_thread.process_frame.return_value = (
            CameraTestUtils.create_sample_frame(),
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        )
        mock_detection_thread.game_state = MagicMock()
        mock_detection_thread.game_state.is_valid.return_value = True
        
        return mock_detection_thread
    
    @staticmethod
    def verify_frame_processed(camera_thread, frame=None):
        """
        Verify that a frame was processed and emitted.
        
        Args:
            camera_thread: The camera thread instance
            frame: The expected frame (default: None, meaning any frame is acceptable)
        """
        camera_thread.frame_ready.emit.assert_called()
        
        if frame is not None:
            # Get the actual frame emitted
            actual_frame = camera_thread.frame_ready.emit.call_args[0][0]
            # Check shape matches
            assert actual_frame.shape == frame.shape
    
    @staticmethod
    def verify_fps_updated(camera_thread, fps=30.0):
        """
        Verify that FPS was updated and emitted.
        
        Args:
            camera_thread: The camera thread instance
            fps: The expected FPS value
        """
        camera_thread.fps_updated.emit.assert_called_with(fps)


class UIComponentTestUtils:
    """
    Utility methods for testing UI components.
    
    This class provides standardized test methods for UI component tests
    to reduce code duplication.
    """
    
    @staticmethod
    def create_board_widget_mock(board=None):
        """
        Create a mock board widget with standard attributes.
        
        Args:
            board: Optional board state, defaults to empty board
            
        Returns:
            MagicMock: A mock board widget
        """
        board_widget = MagicMock()
        
        if board is None:
            board = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            
        board_widget.board = board
        board_widget.winning_line = None
        board_widget.reset_board = MagicMock()
        board_widget.update = MagicMock()
        board_widget.draw_symbol = MagicMock()
        
        return board_widget
    
    @staticmethod
    def create_status_label_mock():
        """
        Create a mock status label.
        
        Returns:
            MagicMock: A mock status label
        """
        status_label = MagicMock()
        status_label.setText = MagicMock()
        
        return status_label
    
    @staticmethod
    def create_camera_view_mock():
        """
        Create a mock camera view.
        
        Returns:
            MagicMock: A mock camera view
        """
        camera_view = MagicMock()
        camera_view.update_image = MagicMock()
        
        return camera_view
    
    @staticmethod
    def create_strategy_selector_mock(difficulty=0.5):
        """
        Create a mock strategy selector.
        
        Args:
            difficulty: The initial difficulty
            
        Returns:
            MagicMock: A mock strategy selector
        """
        strategy_selector = MagicMock()
        strategy_selector.set_difficulty = MagicMock()
        strategy_selector.get_move = MagicMock(return_value=(1, 1))
        strategy_selector.select_strategy = MagicMock()
        
        return strategy_selector
    
    @staticmethod
    def prepare_ui_components(app):
        """
        Set up all UI components for an app instance.
        
        Args:
            app: The app instance
        """
        app.board_widget = UIComponentTestUtils.create_board_widget_mock()
        app.status_label = UIComponentTestUtils.create_status_label_mock()
        app.camera_view = UIComponentTestUtils.create_camera_view_mock()
        app.strategy_selector = UIComponentTestUtils.create_strategy_selector_mock()
        
        app.difficulty_value_label = MagicMock()
        app.difficulty_slider = MagicMock()
        app.reset_button = MagicMock()
        app.park_button = MagicMock()
        app.calibrate_button = MagicMock()
        app.debug_button = MagicMock()
        app.debug_window = MagicMock()
        app.timer = MagicMock()
    
    @staticmethod
    def prepare_game_state(app, human=None, ai=None, current_turn=None, game_over=False, winner=None):
        """
        Set up game state for an app instance.
        
        Args:
            app: The app instance
            human: Human player symbol (PLAYER_X or PLAYER_O)
            ai: AI player symbol (PLAYER_X or PLAYER_O)
            current_turn: Current turn (PLAYER_X or PLAYER_O)
            game_over: Whether the game is over
            winner: Winner (PLAYER_X, PLAYER_O, TIE, or None)
        """
        if human is not None:
            app.human_player = human
        if ai is not None:
            app.ai_player = ai
        if current_turn is not None:
            app.current_turn = current_turn
            
        app.game_over = game_over
        app.winner = winner
        
        if game_over and winner is None:
            # Default to a tie if game is over but no winner
            app.winner = game_logic.TIE


class EventHandlingTestUtils:
    """
    Utility methods for testing event handling functionality.

    This class provides standardized test methods for common event handling scenarios
    to reduce code duplication across test files.
    """

    @staticmethod
    def test_debug_button_enable(app):
        """
        Test enabling debug mode via debug button.

        Args:
            app: The app instance
        """
        # Set up debug_mode to False
        app.debug_mode = False

        # Call the method
        app.handle_debug_button_click()

        # Check that debug_mode was set to True
        assert app.debug_mode is True

        # Check that debug_button text was updated
        app.debug_button.setText.assert_called_once_with("Vypnout debug")

    @staticmethod
    def test_debug_button_disable(app):
        """
        Test disabling debug mode via debug button.

        Args:
            app: The app instance
        """
        # Set up debug_mode to True
        app.debug_mode = True
        app.debug_window = MagicMock()

        # Call the method
        app.handle_debug_button_click()

        # Check that debug_mode was set to False
        assert app.debug_mode is False

        # Check that debug_window was closed
        app.debug_window.close.assert_called_once()

        # Check that debug_button text was updated
        app.debug_button.setText.assert_called_once_with("Zapnout debug")

    @staticmethod
    def test_difficulty_changed(app, value=75):
        """
        Test handling of difficulty slider change.

        Args:
            app: The app instance
            value: The difficulty value to test with
        """
        # Call the method
        app.handle_difficulty_changed(value)

        # Check that difficulty was updated
        assert app.difficulty == value / 100.0

        # Check that difficulty_value_label was updated
        app.difficulty_value_label.setText.assert_called_once_with(f"{value}%")