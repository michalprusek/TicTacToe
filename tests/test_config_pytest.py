"""
Pure pytest tests for config module.
"""
import pytest
from app.core.config import AppConfig, GameDetectorConfig, ArmControllerConfig


class TestAppConfig:
    """Pure pytest test class for AppConfig."""
    
    def test_app_config_creation(self):
        """Test creating AppConfig instance."""
        config = AppConfig()
        assert config is not None
        assert hasattr(config, 'game_detector')
        assert hasattr(config, 'arm_controller')
    
    def test_app_config_initialization(self):
        """Test AppConfig initialization."""
        config = AppConfig()
        assert isinstance(config.game_detector, GameDetectorConfig)
        assert isinstance(config.arm_controller, ArmControllerConfig)
    
    def test_app_config_dataclass(self):
        """Test that AppConfig is a dataclass."""
        assert hasattr(AppConfig, '__dataclass_fields__')
    
    def test_game_detector_config(self):
        """Test GameDetectorConfig."""
        config = GameDetectorConfig()
        assert config is not None
        assert hasattr(config, 'camera_index')
    
    def test_arm_controller_config(self):
        """Test ArmControllerConfig."""
        config = ArmControllerConfig()
        assert config is not None
        assert hasattr(config, 'port')
    
    def test_config_types(self):
        """Test configuration types."""
        config = AppConfig()
        assert isinstance(config.game_detector.camera_index, int)
        assert config.arm_controller.port is None or isinstance(config.arm_controller.port, str)