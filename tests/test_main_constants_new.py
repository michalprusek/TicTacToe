"""
Tests for app/main/constants.py module.
"""
import pytest
from app.main.constants import (
    DEFAULT_CAMERA_INDEX, DEFAULT_DIFFICULTY, DEFAULT_SPEED, MAX_SPEED_FACTOR,
    DEFAULT_SAFE_Z, DEFAULT_DRAW_Z, DRAWING_SPEED, MAX_SPEED,
    BUTTON_FONT_SIZE, STATUS_FONT_SIZE, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    DETECTION_FPS, MIN_CONFIDENCE_THRESHOLD, CAMERA_THREAD_INTERVAL,
    ARM_THREAD_INTERVAL, WEIGHTS_DIR, DETECTION_MODEL, POSE_MODEL,
    CALIBRATION_FILE
)


class TestMainConstants:
    """Test class for main constants module."""

    def test_camera_constants(self):
        """Test camera related constants."""
        assert DEFAULT_CAMERA_INDEX == 0
        assert isinstance(DEFAULT_CAMERA_INDEX, int)
        assert DEFAULT_CAMERA_INDEX >= 0

    def test_game_constants(self):
        """Test game related constants."""
        assert DEFAULT_DIFFICULTY == 10
        assert isinstance(DEFAULT_DIFFICULTY, int)
        assert 1 <= DEFAULT_DIFFICULTY <= 10

    def test_arm_movement_constants(self):
        """Test arm movement constants."""
        assert DEFAULT_SPEED == 100000
        assert MAX_SPEED_FACTOR == 2
        assert DEFAULT_SAFE_Z == 15.0
        assert DEFAULT_DRAW_Z == 5.0
        assert DRAWING_SPEED == 50000
        assert MAX_SPEED == 100000
        
        assert isinstance(DEFAULT_SPEED, int)
        assert isinstance(MAX_SPEED_FACTOR, int)
        assert isinstance(DEFAULT_SAFE_Z, float)
        assert isinstance(DEFAULT_DRAW_Z, float)
        assert isinstance(DRAWING_SPEED, int)
        assert isinstance(MAX_SPEED, int)
        
        assert DEFAULT_SAFE_Z > DEFAULT_DRAW_Z
        assert DEFAULT_SPEED > 0
        assert DRAWING_SPEED > 0

    def test_gui_constants(self):
        """Test GUI related constants."""
        assert BUTTON_FONT_SIZE == 10
        assert STATUS_FONT_SIZE == 9
        assert WINDOW_MIN_WIDTH == 800
        assert WINDOW_MIN_HEIGHT == 600
        
        assert isinstance(BUTTON_FONT_SIZE, int)
        assert isinstance(STATUS_FONT_SIZE, int)
        assert isinstance(WINDOW_MIN_WIDTH, int)
        assert isinstance(WINDOW_MIN_HEIGHT, int)
        
        assert BUTTON_FONT_SIZE > 0
        assert STATUS_FONT_SIZE > 0
        assert WINDOW_MIN_WIDTH > 0
        assert WINDOW_MIN_HEIGHT > 0

    def test_detection_constants(self):
        """Test detection related constants."""
        assert DETECTION_FPS == 2
        assert MIN_CONFIDENCE_THRESHOLD == 0.5
        
        assert isinstance(DETECTION_FPS, int)
        assert isinstance(MIN_CONFIDENCE_THRESHOLD, float)
        
        assert DETECTION_FPS > 0
        assert 0.0 <= MIN_CONFIDENCE_THRESHOLD <= 1.0

    def test_threading_constants(self):
        """Test threading related constants."""
        assert CAMERA_THREAD_INTERVAL == 500
        assert ARM_THREAD_INTERVAL == 100
        
        assert isinstance(CAMERA_THREAD_INTERVAL, int)
        assert isinstance(ARM_THREAD_INTERVAL, int)
        
        assert CAMERA_THREAD_INTERVAL > 0
        assert ARM_THREAD_INTERVAL > 0

    def test_file_path_constants(self):
        """Test file path constants."""
        assert WEIGHTS_DIR == "weights"
        assert DETECTION_MODEL == "best_detection.pt"
        assert POSE_MODEL == "best_pose.pt"
        assert CALIBRATION_FILE == "hand_eye_calibration.json"
        
        assert isinstance(WEIGHTS_DIR, str)
        assert isinstance(DETECTION_MODEL, str)
        assert isinstance(POSE_MODEL, str)
        assert isinstance(CALIBRATION_FILE, str)
        
        assert DETECTION_MODEL.endswith(".pt")
        assert POSE_MODEL.endswith(".pt")
        assert CALIBRATION_FILE.endswith(".json")

    def test_constants_relationships(self):
        """Test logical relationships between constants."""
        assert MAX_SPEED >= DEFAULT_SPEED
        assert DEFAULT_SAFE_Z > DEFAULT_DRAW_Z
        assert WINDOW_MIN_WIDTH > 0
        assert WINDOW_MIN_HEIGHT > 0
        assert CAMERA_THREAD_INTERVAL > ARM_THREAD_INTERVAL
