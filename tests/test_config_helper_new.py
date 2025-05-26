# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Tests for app/main/config_helper.py module.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from app.main.config_helper import ConfigHelper
from app.core.config import AppConfig


class TestConfigHelper:
    """Test class for ConfigHelper."""

    def test_init_with_config(self):
        """Test initialization with provided config."""
        mock_config = Mock(spec=AppConfig)
        helper = ConfigHelper(mock_config)
        assert helper.config is mock_config

    def test_init_without_config(self):
        """Test initialization without config creates default."""
        helper = ConfigHelper()
        assert helper.config is not None
        assert isinstance(helper.config, AppConfig)

    def test_get_camera_config(self):
        """Test getting camera configuration."""
        mock_config = Mock()
        mock_config.camera_index = 1
        mock_config.camera_fps = 25
        mock_config.flip_horizontal = True
        mock_config.flip_vertical = False
        
        helper = ConfigHelper(mock_config)
        camera_config = helper.get_camera_config()
        
        assert camera_config['camera_index'] == 1
        assert camera_config['fps'] == 25
        assert camera_config['flip_horizontal'] is True
        assert camera_config['flip_vertical'] is False

    def test_get_camera_config_defaults(self):
        """Test camera config with default values."""
        mock_config = Mock(spec=[])  # Empty mock without attributes
        helper = ConfigHelper(mock_config)
        camera_config = helper.get_camera_config()
        
        assert camera_config['camera_index'] == 0
        assert camera_config['fps'] == 30
        assert camera_config['flip_horizontal'] is False
        assert camera_config['flip_vertical'] is False

    def test_get_arm_config(self):
        """Test getting arm configuration."""
        mock_config = Mock()
        mock_config.arm_port = "/dev/ttyUSB0"
        mock_config.arm_speed = 50000
        mock_config.safe_z = 20.0
        mock_config.draw_z = 3.0
        mock_config.park_position = (100, 50)
        
        helper = ConfigHelper(mock_config)
        arm_config = helper.get_arm_config()
        
        assert arm_config['port'] == "/dev/ttyUSB0"
        assert arm_config['speed'] == 50000
        assert arm_config['safe_z'] == 20.0
        assert arm_config['draw_z'] == 3.0
        assert arm_config['park_position'] == (100, 50)

    def test_get_arm_config_defaults(self):
        """Test arm config with default values."""
        mock_config = Mock(spec=[])
        helper = ConfigHelper(mock_config)
        arm_config = helper.get_arm_config()
        
        assert arm_config['port'] is None
        assert arm_config['speed'] == 100000
        assert arm_config['safe_z'] == 15.0
        assert arm_config['draw_z'] == 5.0
        assert arm_config['park_position'] == (150, 0)

    def test_get_game_config(self):
        """Test getting game configuration."""
        mock_config = Mock()
        mock_config.difficulty = 5
        mock_config.auto_move = False
        mock_config.show_debug = True
        mock_config.language = 'en'
        
        helper = ConfigHelper(mock_config)
        game_config = helper.get_game_config()
        
        assert game_config['difficulty'] == 5
        assert game_config['auto_move'] is False
        assert game_config['show_debug'] is True
        assert game_config['language'] == 'en'

    def test_get_game_config_defaults(self):
        """Test game config with default values."""
        mock_config = Mock(spec=[])
        helper = ConfigHelper(mock_config)
        game_config = helper.get_game_config()
        
        assert game_config['difficulty'] == 10
        assert game_config['auto_move'] is True
        assert game_config['show_debug'] is False
        assert game_config['language'] == 'cs'

    def test_get_gui_config(self):
        """Test getting GUI configuration."""
        mock_config = Mock()
        mock_config.window_width = 1200
        mock_config.window_height = 900
        mock_config.font_size = 12
        mock_config.theme = 'dark'
        
        helper = ConfigHelper(mock_config)
        gui_config = helper.get_gui_config()
        
        assert gui_config['window_width'] == 1200
        assert gui_config['window_height'] == 900
        assert gui_config['font_size'] == 12
        assert gui_config['theme'] == 'dark'

    def test_get_gui_config_defaults(self):
        """Test GUI config with default values."""
        mock_config = Mock(spec=[])
        helper = ConfigHelper(mock_config)
        gui_config = helper.get_gui_config()
        
        assert gui_config['window_width'] == 800
        assert gui_config['window_height'] == 600
        assert gui_config['font_size'] == 10
        assert gui_config['theme'] == 'default'

    def test_get_safe_value_existing_key(self):
        """Test getting existing configuration value safely."""
        mock_config = Mock()
        mock_config.test_key = "test_value"
        
        helper = ConfigHelper(mock_config)
        value = helper.get_safe_value('test_key', 'default')
        
        assert value == "test_value"

    def test_get_safe_value_missing_key(self):
        """Test getting missing configuration value returns default."""
        mock_config = Mock(spec=[])
        helper = ConfigHelper(mock_config)
        value = helper.get_safe_value('missing_key', 'default_value')
        
        assert value == 'default_value'

    def test_get_safe_value_with_section(self):
        """Test getting value from configuration section."""
        mock_section = Mock()
        mock_section.section_key = "section_value"
        
        mock_config = Mock()
        mock_config.test_section = mock_section
        
        helper = ConfigHelper(mock_config)
        value = helper.get_safe_value('section_key', 'default', 'test_section')
        
        assert value == "section_value"

    def test_get_safe_value_missing_section(self):
        """Test getting value from missing section returns default."""
        mock_config = Mock(spec=[])
        helper = ConfigHelper(mock_config)
        value = helper.get_safe_value('key', 'default', 'missing_section')
        
        assert value == 'default'

    def test_get_safe_value_exception_handling(self):
        """Test safe value retrieval handles exceptions."""
        # Create a config object that raises TypeError when accessing attributes
        class BadConfig:
            def __getattribute__(self, name):
                if name == 'test_key':
                    raise TypeError("Test error")
                return super().__getattribute__(name)
        
        helper = ConfigHelper(BadConfig())
        value = helper.get_safe_value('test_key', 'fallback')
        
        assert value == 'fallback'
