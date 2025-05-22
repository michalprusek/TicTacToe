"""
Configuration module for the TicTacToe application.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class GameDetectorConfig:
    """Configuration for the game detector component."""
    # Camera settings
    camera_index: int = 0
    disable_autofocus: bool = True

    # Model paths
    detect_model_path: str = "weights/best_detection.pt"
    pose_model_path: str = "weights/best_pose.pt"

    # Detection thresholds
    bbox_conf_threshold: float = 0.45
    pose_conf_threshold: float = 0.45
    keypoint_visible_threshold: float = 0.3

    # Grid detection parameters
    min_points_for_homography: int = 4
    ransac_reproj_threshold: float = 10.0
    grid_points_count: int = 16

    # Validation thresholds
    grid_dist_std_dev_threshold: float = 100.0
    grid_angle_tolerance_deg: float = 30.0

    # Grid detection retry parameters
    max_grid_detection_retries: int = 3

    # Performance settings
    target_fps: float = 2.0  # Target frames per second for detection
    # Device to use for inference (None = auto-detect)
    device: Optional[str] = None
    
    # Visualization settings
    show_game_state_on_frame: bool = True  # Show game state information on frame
    
    # Ideal grid keypoints for visualization (4x4 grid in a 0-3 range)
    ideal_grid_keypoints_4x4: np.ndarray = field(default_factory=lambda: np.array([
        [x, y] for y in range(4) for x in range(4)
    ], dtype=np.float32))


@dataclass
class ArmControllerConfig:
    """Configuration for the robot arm controller."""
    # Port settings
    port: Optional[str] = None  # Auto-detect if None

    # Movement parameters
    default_speed: int = 20000
    min_speed: int = 5000
    max_speed: int = 20000

    # Z-axis heights
    safe_z: float = 15.0  # Height for safe travel moves (mm)
    draw_z: float = 5.0   # Height while drawing (mm)

    # Symbol dimensions
    symbol_size_mm: float = 40.0  # Size of X and O symbols in mm


@dataclass
class GameConfig:
    """Configuration for the game logic."""
    # Difficulty settings
    default_difficulty: int = 5  # Default difficulty (0-10)

    # Game parameters
    poll_interval_seconds: float = 1.0  # How often to check board state

    # GUI settings
    gui_window_title: str = "Piškvorky s Robotem"
    debug_window_title: str = "Ladění Detekce"

    # Language settings
    language: str = "cs"  # 'cs' for Czech, 'en' for English


@dataclass
class AppConfig:
    """Main application configuration."""
    game_detector: GameDetectorConfig = field(
        default_factory=GameDetectorConfig)
    arm_controller: ArmControllerConfig = field(
        default_factory=ArmControllerConfig)
    game: GameConfig = field(default_factory=GameConfig)

    # Calibration file path
    calibration_file: str = "hand_eye_calibration.json"

    # Additional settings
    debug_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "game_detector": self.game_detector.__dict__,
            "arm_controller": self.arm_controller.__dict__,
            "game": self.game.__dict__,
            "calibration_file": self.calibration_file,
            "debug_mode": self.debug_mode
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create config from dictionary."""
        config = cls()

        if "game_detector" in config_dict:
            for key, value in config_dict["game_detector"].items():
                if hasattr(config.game_detector, key):
                    setattr(config.game_detector, key, value)

        if "arm_controller" in config_dict:
            for key, value in config_dict["arm_controller"].items():
                if hasattr(config.arm_controller, key):
                    setattr(config.arm_controller, key, value)

        if "game" in config_dict:
            for key, value in config_dict["game"].items():
                if hasattr(config.game, key):
                    setattr(config.game, key, value)

        if "calibration_file" in config_dict:
            config.calibration_file = config_dict["calibration_file"]

        if "debug_mode" in config_dict:
            config.debug_mode = config_dict["debug_mode"]

        return config
