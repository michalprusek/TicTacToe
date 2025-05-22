"""
Unit tests for CameraManager class.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from app.main.camera_manager import CameraManager


@pytest.fixture
def mock_config():
    """mock_config fixture for tests."""
    mock_config = MagicMock()
    mock_config.disable_autofocus = True
    mock_config.frame_width = 640
    mock_config.frame_height = 480
    return mock_config


@pytest.fixture
def camera_manager(mock_config):
    """camera_manager fixture for tests."""
    # Patch cv2.VideoCapture to avoid actual camera access
    with patch('cv2.VideoCapture') as mock_video_capture:
        # Set up mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640  # Default return value for get
        mock_cap.set.return_value = True
        mock_video_capture.return_value = mock_cap

        # Create camera manager with mocked VideoCapture
        manager = CameraManager(
            camera_index=0,
            config=mock_config
        )

        # Store mock_cap for tests to use
        manager._mock_cap = mock_cap

        yield manager


class TestCameraManager:
    """Test CameraManager class."""

    def test_init(self, camera_manager, mock_config):
        """Test initialization."""
        assert camera_manager.camera_index == 0
        assert camera_manager.config == mock_config
        assert camera_manager.disable_autofocus == mock_config.disable_autofocus
        assert camera_manager.frame_width == mock_config.frame_width
        assert camera_manager.frame_height == mock_config.frame_height
        assert camera_manager.cap is not None

    def test_setup_camera(self, camera_manager):
        """Test setup_camera method."""
        # Reset mock calls
        camera_manager._mock_cap.set.reset_mock()

        # Call the method
        result = camera_manager.setup_camera()

        # Check that VideoCapture.set was called for resolution
        camera_manager._mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, camera_manager.frame_width)
        camera_manager._mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, camera_manager.frame_height)

        # Check that VideoCapture.set was called for autofocus
        camera_manager._mock_cap.set.assert_any_call(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Check that method returned True
        assert result is True

    def test_setup_camera_failure(self, camera_manager):
        """Test setup_camera method with failure."""
        # Set up mock to simulate failure
        camera_manager._mock_cap.isOpened.return_value = False

        # Call the method
        result = camera_manager.setup_camera()

        # Check that method returned False
        assert result is False

    def test_read_frame(self, camera_manager):
        """Test read_frame method."""
        # Set up mock to return a frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera_manager._mock_cap.read.return_value = (True, mock_frame)

        # Call the method
        ret, frame = camera_manager.read_frame()

        # Check that VideoCapture.read was called
        camera_manager._mock_cap.read.assert_called_once()

        # Check that correct values were returned
        assert ret is True
        assert frame is mock_frame

    def test_read_frame_failure(self, camera_manager):
        """Test read_frame method with failure."""
        # Set up mock to simulate failure
        camera_manager._mock_cap.read.return_value = (False, None)

        # Call the method
        ret, frame = camera_manager.read_frame()

        # Check that VideoCapture.read was called
        camera_manager._mock_cap.read.assert_called_once()

        # Check that correct values were returned
        assert ret is False
        assert frame is None

    def test_read_frame_no_camera(self, camera_manager):
        """Test read_frame method with no camera."""
        # Set cap to None
        camera_manager.cap = None

        # Call the method
        ret, frame = camera_manager.read_frame()

        # Check that correct values were returned
        assert ret is False
        assert frame is None

    def test_release(self, camera_manager):
        """Test release method."""
        # Call the method
        camera_manager.release()

        # Check that VideoCapture.release was called
        camera_manager._mock_cap.release.assert_called_once()

    def test_is_opened(self, camera_manager):
        """Test is_opened method."""
        # Set up mock to return True
        camera_manager._mock_cap.isOpened.return_value = True

        # Reset mock to clear previous calls
        camera_manager._mock_cap.isOpened.reset_mock()

        # Call the method
        result = camera_manager.is_opened()

        # Check that VideoCapture.isOpened was called
        assert camera_manager._mock_cap.isOpened.called

        # Check that correct value was returned
        assert result is True

    def test_is_opened_no_camera(self, camera_manager):
        """Test is_opened method with no camera."""
        # Set cap to None
        camera_manager.cap = None

        # Call the method
        result = camera_manager.is_opened()

        # Check that correct value was returned
        assert result is False

    def test_get_camera_properties(self, camera_manager):
        """Test get_camera_properties method."""
        # Set up mock to return different values for different properties
        def mock_get_side_effect(prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            elif prop_id == cv2.CAP_PROP_FPS:
                return 30
            else:
                return 0

        camera_manager._mock_cap.get.side_effect = mock_get_side_effect

        # Call the method
        properties = camera_manager.get_camera_properties()

        # Check that properties were returned
        assert properties['width'] == 640
        assert properties['height'] == 480
        assert properties['fps'] == 30

    def test_get_camera_properties_no_camera(self, camera_manager):
        """Test get_camera_properties method with no camera."""
        # Set cap to None
        camera_manager.cap = None

        # Call the method
        properties = camera_manager.get_camera_properties()

        # Check that empty dict was returned
        assert properties == {}

    def test_set_camera_property(self, camera_manager):
        """Test set_camera_property method."""
        # Set up mock to return True
        camera_manager._mock_cap.set.return_value = True

        # Reset mock to clear previous calls
        camera_manager._mock_cap.set.reset_mock()

        # Call the method
        result = camera_manager.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, 50)

        # Check that VideoCapture.set was called with correct parameters
        camera_manager._mock_cap.set.assert_called_with(cv2.CAP_PROP_BRIGHTNESS, 50)

        # Check that correct value was returned
        assert result is True

    def test_set_camera_property_failure(self, camera_manager):
        """Test set_camera_property method with failure."""
        # Set up mock to return False
        camera_manager._mock_cap.set.return_value = False

        # Reset mock to clear previous calls
        camera_manager._mock_cap.set.reset_mock()

        # Call the method
        result = camera_manager.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, 50)

        # Check that VideoCapture.set was called with correct parameters
        camera_manager._mock_cap.set.assert_called_with(cv2.CAP_PROP_BRIGHTNESS, 50)

        # Check that correct value was returned
        assert result is False

    def test_set_camera_property_no_camera(self, camera_manager):
        """Test set_camera_property method with no camera."""
        # Set cap to None
        camera_manager.cap = None

        # Call the method
        result = camera_manager.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, 50)

        # Check that correct value was returned
        assert result is False
