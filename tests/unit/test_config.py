"""
Unit tests for the config module.
"""
import pytest
from app.core.config import (
    GameDetectorConfig, ArmControllerConfig, GameConfig, AppConfig
)


class TestGameDetectorConfig():
    """Tests for the GameDetectorConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GameDetectorConfig()
        assert config.camera_index == 0
        assert config.disable_autofocus
        assert config.detect_model_path == "weights/best_detection.pt"
        assert config.pose_model_path == "weights/best_pose.pt"
        assert config.bbox_conf_threshold == 0.45
        assert config.pose_conf_threshold == 0.45
        assert config.keypoint_visible_threshold == 0.3
        assert config.min_points_for_homography == 4
        assert config.ransac_reproj_threshold == 10.0
        assert config.grid_points_count == 16
        assert config.grid_dist_std_dev_threshold == 100.0
        assert config.grid_angle_tolerance_deg == 30.0
        assert config.max_grid_detection_retries == 3

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = GameDetectorConfig(
            camera_index=1,
            disable_autofocus=False,
            detect_model_path="custom_detection.pt",
            pose_model_path="custom_pose.pt",
            bbox_conf_threshold=0.5,
            pose_conf_threshold=0.6,
            keypoint_visible_threshold=0.4,
            min_points_for_homography=5,
            ransac_reproj_threshold=15.0,
            grid_points_count=20,
            grid_dist_std_dev_threshold=120.0,
            grid_angle_tolerance_deg=45.0,
            max_grid_detection_retries=5
        )
        assert config.camera_index == 1
        assert not config.disable_autofocus
        assert config.detect_model_path == "custom_detection.pt"
        assert config.pose_model_path == "custom_pose.pt"
        assert config.bbox_conf_threshold == 0.5
        assert config.pose_conf_threshold == 0.6
        assert config.keypoint_visible_threshold == 0.4
        assert config.min_points_for_homography == 5
        assert config.ransac_reproj_threshold == 15.0
        assert config.grid_points_count == 20
        assert config.grid_dist_std_dev_threshold == 120.0
        assert config.grid_angle_tolerance_deg == 45.0
        assert config.max_grid_detection_retries == 5


class TestArmControllerConfig():
    """Tests for the ArmControllerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ArmControllerConfig()
        assert config.port is None
        assert config.default_speed == 20000
        assert config.min_speed == 5000
        assert config.max_speed == 20000
        assert config.safe_z == 15.0
        assert config.draw_z == 5.0
        assert config.symbol_size_mm == 40.0

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = ArmControllerConfig(
            port="/dev/ttyUSB0",
            default_speed=15000,
            min_speed=3000,
            max_speed=25000,
            safe_z=20.0,
            draw_z=3.0,
            symbol_size_mm=50.0
        )
        assert config.port == "/dev/ttyUSB0"
        assert config.default_speed == 15000
        assert config.min_speed == 3000
        assert config.max_speed == 25000
        assert config.safe_z == 20.0
        assert config.draw_z == 3.0
        assert config.symbol_size_mm == 50.0


class TestGameConfig():
    """Tests for the GameConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GameConfig()
        assert config.default_difficulty == 5
        assert config.poll_interval_seconds == 1.0
        assert config.gui_window_title == "Piškvorky s Robotem"
        assert config.debug_window_title == "Ladění Detekce"
        assert config.language == "cs"

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = GameConfig(
            default_difficulty=8,
            poll_interval_seconds=0.5,
            gui_window_title="Custom Title",
            debug_window_title="Debug Window",
            language="en"
        )
        assert config.default_difficulty == 8
        assert config.poll_interval_seconds == 0.5
        assert config.gui_window_title == "Custom Title"
        assert config.debug_window_title == "Debug Window"
        assert config.language == "en"


class TestAppConfig():
    """Tests for the AppConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AppConfig()
        assert isinstance(config.game_detector, GameDetectorConfig)
        assert isinstance(config.arm_controller, ArmControllerConfig)
        assert isinstance(config.game, GameConfig)
        assert config.calibration_file == "hand_eye_calibration.json"
        assert not config.debug_mode

    def test_custom_values(self):
        """Test that custom values can be set."""
        game_detector = GameDetectorConfig(camera_index=2)
        arm_controller = ArmControllerConfig(port="/dev/ttyUSB1")
        game = GameConfig(default_difficulty=7)
        
        config = AppConfig(
            game_detector=game_detector,
            arm_controller=arm_controller,
            game=game,
            calibration_file="custom_calibration.json",
            debug_mode=True
        )
        
        assert config.game_detector.camera_index == 2
        assert config.arm_controller.port == "/dev/ttyUSB1"
        assert config.game.default_difficulty == 7
        assert config.calibration_file == "custom_calibration.json"
        assert config.debug_mode

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AppConfig()
        config_dict = config.to_dict()
        
        assert "game_detector" in config_dict
        assert "arm_controller" in config_dict
        assert "game" in config_dict
        assert "calibration_file" in config_dict
        assert "debug_mode" in config_dict
        
        assert config_dict["calibration_file"] == "hand_eye_calibration.json"
        assert not config_dict["debug_mode"]
        
        # Check nested dictionaries
        assert "camera_index" in config_dict["game_detector"]
        assert "port" in config_dict["arm_controller"]
        assert "default_difficulty" in config_dict["game"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "game_detector": {
                "camera_index": 3,
                "disable_autofocus": False
            },
            "arm_controller": {
                "port": "/dev/ttyUSB2",
                "safe_z": 25.0
            },
            "game": {
                "default_difficulty": 9,
                "language": "en"
            },
            "calibration_file": "new_calibration.json",
            "debug_mode": True
        }
        
        config = AppConfig.from_dict(config_dict)
        
        # Check that values were properly set
        assert config.game_detector.camera_index == 3
        assert not config.game_detector.disable_autofocus
        assert config.arm_controller.port == "/dev/ttyUSB2"
        assert config.arm_controller.safe_z == 25.0
        assert config.game.default_difficulty == 9
        assert config.game.language == "en"
        assert config.calibration_file == "new_calibration.json"
        assert config.debug_mode
        
        # Check that unspecified values retain defaults
        assert config.game_detector.detect_model_path == "weights/best_detection.pt"
        assert config.arm_controller.default_speed == 20000
        assert config.game.poll_interval_seconds == 1.0

    def test_from_dict_partial(self):
        """Test creation from partial dictionary."""
        config_dict = {
            "game_detector": {
                "camera_index": 4
            },
            "debug_mode": True
        }
        
        config = AppConfig.from_dict(config_dict)
        
        # Check that specified values were set
        assert config.game_detector.camera_index == 4
        assert config.debug_mode
        
        # Check that unspecified values retain defaults
        assert config.game_detector.disable_autofocus
        assert config.arm_controller.port is None
        assert config.game.default_difficulty == 5
        assert config.calibration_file == "hand_eye_calibration.json"

    def test_from_dict_invalid_keys(self):
        """Test that invalid keys are ignored."""
        config_dict = {
            "game_detector": {
                "camera_index": 5,
                "invalid_key": "value"  # This should be ignored
            },
            "invalid_section": {  # This should be ignored
                "key": "value"
            },
            "debug_mode": True
        }
        
        config = AppConfig.from_dict(config_dict)
        
        # Check that valid values were set
        assert config.game_detector.camera_index == 5
        assert config.debug_mode
        
        # Check that invalid keys were ignored
        assert not hasattr(config.game_detector, "invalid_key")
        assert not hasattr(config, "invalid_section")



