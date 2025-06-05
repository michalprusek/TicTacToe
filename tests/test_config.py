# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Tests for config module.
"""
import pytest
import numpy as np

from app.core.config import (
    GameDetectorConfig, ArmControllerConfig, GameConfig, AppConfig
)


class TestGameDetectorConfig:
    """Test class for GameDetectorConfig."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = GameDetectorConfig()
        
        assert config.camera_index == 0
        assert config.disable_autofocus is True
        assert config.detect_model_path == "weights/best_detection.pt"
        assert config.pose_model_path == "weights/best_pose.pt"
        assert config.bbox_conf_threshold == 0.90
        assert config.pose_conf_threshold == 0.80
        assert config.keypoint_visible_threshold == 0.3
        assert config.min_points_for_homography == 4
        assert config.ransac_reproj_threshold == 10.0
        assert config.grid_points_count == 16
        assert config.target_fps == 30.0
        assert config.device is None
        assert config.show_game_state_on_frame is True
    
    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = GameDetectorConfig(
            camera_index=1,
            bbox_conf_threshold=0.8,
            target_fps=3.0,
            device="cuda:0"
        )
        
        assert config.camera_index == 1
        assert config.bbox_conf_threshold == 0.8
        assert config.target_fps == 3.0
        assert config.device == "cuda:0"    
    def test_ideal_grid_keypoints(self):
        """Test ideal grid keypoints array."""
        config = GameDetectorConfig()
        
        # Check that ideal_grid_keypoints_4x4 is properly initialized
        assert isinstance(config.ideal_grid_keypoints_4x4, np.ndarray)
        assert config.ideal_grid_keypoints_4x4.shape == (16, 2)
        assert config.ideal_grid_keypoints_4x4.dtype == np.float32
        
        # Check some specific values
        expected_first = np.array([0, 0], dtype=np.float32)
        expected_last = np.array([3, 3], dtype=np.float32)
        np.testing.assert_array_equal(config.ideal_grid_keypoints_4x4[0], expected_first)
        np.testing.assert_array_equal(config.ideal_grid_keypoints_4x4[-1], expected_last)


class TestArmControllerConfig:
    """Test class for ArmControllerConfig."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = ArmControllerConfig()
        
        assert config.port is None
        assert config.default_speed == 20000
        assert config.min_speed == 5000
        assert config.max_speed == 20000
        assert config.safe_z == 15.0
        assert config.draw_z == 5.0
        assert config.symbol_size_mm == 40.0
    
    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = ArmControllerConfig(
            port="/dev/ttyUSB0",
            default_speed=15000,
            safe_z=20.0,
            symbol_size_mm=35.0
        )
        
        assert config.port == "/dev/ttyUSB0"
        assert config.default_speed == 15000
        assert config.safe_z == 20.0
        assert config.symbol_size_mm == 35.0
class TestGameConfig:
    """Test class for GameConfig."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = GameConfig()
        
        assert config.default_difficulty == 10
        assert config.poll_interval_seconds == 1.0
        assert config.gui_window_title == "Piškvorky s Robotem"
        assert config.debug_window_title == "Ladění Detekce"
        assert config.language == "cs"
    
    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = GameConfig(
            default_difficulty=7,
            poll_interval_seconds=0.5,
            language="en"
        )
        
        assert config.default_difficulty == 7
        assert config.poll_interval_seconds == 0.5
        assert config.language == "en"


class TestAppConfig:
    """Test class for AppConfig."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = AppConfig()
        
        assert isinstance(config.game_detector, GameDetectorConfig)
        assert isinstance(config.arm_controller, ArmControllerConfig)
        assert isinstance(config.game, GameConfig)
        assert config.calibration_file == "hand_eye_calibration.json"
        assert config.debug_mode is False    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AppConfig()
        config.debug_mode = True
        config.game.default_difficulty = 8
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "game_detector" in config_dict
        assert "arm_controller" in config_dict
        assert "game" in config_dict
        assert "calibration_file" in config_dict
        assert "debug_mode" in config_dict
        assert config_dict["debug_mode"] is True
        assert config_dict["game"]["default_difficulty"] == 8
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "game_detector": {
                "camera_index": 2,
                "bbox_conf_threshold": 0.9
            },
            "arm_controller": {
                "default_speed": 15000,
                "safe_z": 20.0
            },
            "game": {
                "default_difficulty": 5,
                "language": "en"
            },
            "calibration_file": "custom_calibration.json",
            "debug_mode": True
        }
        
        config = AppConfig.from_dict(config_dict)
        
        assert config.game_detector.camera_index == 2
        assert config.game_detector.bbox_conf_threshold == 0.9
        assert config.arm_controller.default_speed == 15000
        assert config.arm_controller.safe_z == 20.0
        assert config.game.default_difficulty == 5
        assert config.game.language == "en"
        assert config.calibration_file == "custom_calibration.json"
        assert config.debug_mode is True    
    def test_from_dict_partial(self):
        """Test creation from partial dictionary."""
        config_dict = {
            "game": {
                "default_difficulty": 3
            },
            "debug_mode": True
        }
        
        config = AppConfig.from_dict(config_dict)
        
        # Modified values
        assert config.game.default_difficulty == 3
        assert config.debug_mode is True
        
        # Default values should remain
        assert config.game_detector.camera_index == 0
        assert config.arm_controller.default_speed == 20000
        assert config.game.language == "cs"
    
    def test_from_dict_invalid_keys(self):
        """Test handling of invalid keys in dictionary."""
        config_dict = {
            "game_detector": {
                "invalid_key": "should_be_ignored",
                "camera_index": 1
            }
        }
        
        config = AppConfig.from_dict(config_dict)
        
        # Valid key should be applied
        assert config.game_detector.camera_index == 1
        # Invalid key should be ignored (no error)
        assert not hasattr(config.game_detector, 'invalid_key')
