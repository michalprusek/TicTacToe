# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""Comprehensive tests for config_helper.py module."""
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from app.main.config_helper import ConfigHelper
from app.core.config import AppConfig


class TestConfigHelper:
    """Test ConfigHelper functionality."""
    
    def test_initialization_with_config(self):
        """Test initialization with provided config."""
        mock_config = Mock(spec=AppConfig)
        helper = ConfigHelper(config=mock_config)
        assert helper.config is mock_config
    
    def test_initialization_without_config(self):
        """Test initialization without config creates default."""
        helper = ConfigHelper()
        assert helper.config is not None
        assert isinstance(helper.config, AppConfig)
    
    def test_initialization_with_none_config(self):
        """Test initialization with None config creates default."""
        helper = ConfigHelper(config=None)
        assert helper.config is not None
        assert isinstance(helper.config, AppConfig)


class TestCameraConfig:
    """Test camera configuration methods."""
    
    def test_get_camera_config_with_attributes(self):
        """Test get_camera_config with all attributes present."""
        mock_config = Mock()
        mock_config.camera_index = 1
        mock_config.camera_fps = 60
        mock_config.flip_horizontal = True
        mock_config.flip_vertical = True
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_camera_config()
        
        expected = {
            'camera_index': 1,
            'fps': 60,
            'flip_horizontal': True,
            'flip_vertical': True
        }
        assert result == expected
    
    def test_get_camera_config_with_defaults(self):
        """Test get_camera_config with missing attributes uses defaults."""
        mock_config = Mock(spec=[])  # Empty spec = no attributes
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_camera_config()
        
        expected = {
            'camera_index': 0,
            'fps': 30,
            'flip_horizontal': False,
            'flip_vertical': False
        }
        assert result == expected
    
    def test_get_camera_config_partial_attributes(self):
        """Test get_camera_config with some attributes present."""
        # Create a simple object with partial attributes
        class PartialConfig:
            def __init__(self):
                self.camera_index = 2
                self.flip_horizontal = True
                # Missing: camera_fps, flip_vertical
        
        helper = ConfigHelper(config=PartialConfig())
        result = helper.get_camera_config()
        
        expected = {
            'camera_index': 2,
            'fps': 30,  # default
            'flip_horizontal': True,
            'flip_vertical': False  # default
        }
        
        assert result == expected


class TestArmConfig:
    """Test arm configuration methods."""
    
    def test_get_arm_config_with_attributes(self):
        """Test get_arm_config with all attributes present."""
        mock_config = Mock()
        mock_config.arm_port = "/dev/ttyUSB0"
        mock_config.arm_speed = 50000
        mock_config.safe_z = 20.0
        mock_config.draw_z = 3.0
        mock_config.park_position = (200, 100)
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_arm_config()
        
        expected = {
            'port': "/dev/ttyUSB0",
            'speed': 50000,
            'safe_z': 20.0,
            'draw_z': 3.0,
            'park_position': (200, 100)
        }
        assert result == expected
    
    def test_get_arm_config_with_defaults(self):
        """Test get_arm_config with missing attributes uses defaults."""
        mock_config = Mock(spec=[])
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_arm_config()
        
        expected = {
            'port': None,
            'speed': 100000,
            'safe_z': 15.0,
            'draw_z': 5.0,
            'park_position': (150, 0)
        }
        assert result == expected


class TestGameConfig:
    """Test game configuration methods."""
    
    def test_get_game_config_with_attributes(self):
        """Test get_game_config with all attributes present."""
        mock_config = Mock()
        mock_config.difficulty = 5
        mock_config.auto_move = False
        mock_config.show_debug = True
        mock_config.language = 'en'
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_game_config()
        
        expected = {
            'difficulty': 5,
            'auto_move': False,
            'show_debug': True,
            'language': 'en'
        }
        assert result == expected
    
    def test_get_game_config_with_defaults(self):
        """Test get_game_config with missing attributes uses defaults."""
        mock_config = Mock(spec=[])
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_game_config()
        
        expected = {
            'difficulty': 10,
            'auto_move': True,
            'show_debug': False,
            'language': 'cs'
        }
        assert result == expected


class TestGuiConfig:
    """Test GUI configuration methods."""
    
    def test_get_gui_config_with_attributes(self):
        """Test get_gui_config with all attributes present."""
        mock_config = Mock()
        mock_config.window_width = 1024
        mock_config.window_height = 768
        mock_config.font_size = 12
        mock_config.theme = 'dark'
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_gui_config()
        
        expected = {
            'window_width': 1024,
            'window_height': 768,
            'font_size': 12,
            'theme': 'dark'
        }
        assert result == expected
    
    def test_get_gui_config_with_defaults(self):
        """Test get_gui_config with missing attributes uses defaults."""
        mock_config = Mock(spec=[])
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_gui_config()
        
        expected = {
            'window_width': 800,
            'window_height': 600,
            'font_size': 10,
            'theme': 'default'
        }
        assert result == expected


class TestSafeValue:
    """Test get_safe_value method."""
    
    def test_get_safe_value_existing_attribute(self):
        """Test get_safe_value with existing attribute."""
        mock_config = Mock()
        mock_config.test_key = "test_value"
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("test_key", "default")
        
        assert result == "test_value"
    
    def test_get_safe_value_missing_attribute(self):
        """Test get_safe_value with missing attribute returns default."""
        mock_config = Mock(spec=[])
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("missing_key", "default_value")
        
        assert result == "default_value"
    
    def test_get_safe_value_no_default(self):
        """Test get_safe_value with missing attribute and no default."""
        mock_config = Mock(spec=[])
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("missing_key")
        
        assert result is None
    
    def test_get_safe_value_with_section_existing(self):
        """Test get_safe_value with section and existing attribute."""
        mock_section = Mock()
        mock_section.nested_key = "nested_value"
        
        mock_config = Mock()
        mock_config.test_section = mock_section
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("nested_key", "default", section="test_section")
        
        assert result == "nested_value"
    
    def test_get_safe_value_with_section_missing_key(self):
        """Test get_safe_value with section but missing key."""
        mock_section = Mock(spec=[])  # No nested_key attribute
        
        mock_config = Mock()
        mock_config.test_section = mock_section
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("missing_key", "default", section="test_section")
        
        assert result == "default"
    
    def test_get_safe_value_with_missing_section(self):
        """Test get_safe_value with missing section."""
        mock_config = Mock(spec=[])  # No test_section attribute
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("nested_key", "default", section="missing_section")
        
        assert result == "default"
    
    def test_get_safe_value_attribute_error_handling(self):
        """Test get_safe_value handles AttributeError gracefully."""
        # Create a config object that raises AttributeError when accessing test_key
        class BadConfig:
            def __getattribute__(self, name):
                if name == 'test_key':
                    raise AttributeError("Test error")
                return super().__getattribute__(name)
        
        helper = ConfigHelper(config=BadConfig())
        result = helper.get_safe_value("test_key", "fallback")
        
        assert result == "fallback"
    
    def test_get_safe_value_type_error_handling(self):
        """Test get_safe_value handles TypeError gracefully."""
        mock_config = Mock()
        # Configure mock to raise TypeError
        type(mock_config).test_key = PropertyMock(side_effect=TypeError("Test error"))
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("test_key", "fallback")
        
        assert result == "fallback"
    
    def test_get_safe_value_without_section_fallback(self):
        """Test get_safe_value without section falls back to main config."""
        mock_config = Mock()
        mock_config.global_key = "global_value"
        
        helper = ConfigHelper(config=mock_config)
        result = helper.get_safe_value("global_key", "default", section=None)
        
        assert result == "global_value"


class TestIntegration:
    """Test integration scenarios."""
    
    def test_all_config_methods_with_real_appconfig(self):
        """Test all config methods work with real AppConfig."""
        # Use real AppConfig to ensure compatibility
        real_config = AppConfig()
        helper = ConfigHelper(config=real_config)
        
        # Should not raise exceptions
        camera_config = helper.get_camera_config()
        arm_config = helper.get_arm_config()
        game_config = helper.get_game_config()
        gui_config = helper.get_gui_config()
        
        # All should be dicts
        assert isinstance(camera_config, dict)
        assert isinstance(arm_config, dict)
        assert isinstance(game_config, dict)
        assert isinstance(gui_config, dict)
        
        # Should have expected keys
        assert 'camera_index' in camera_config
        assert 'port' in arm_config
        assert 'difficulty' in game_config
        assert 'window_width' in gui_config





if __name__ == "__main__":
    pytest.main([__file__])
