# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.main.config_helper module using pytest.
Tests ConfigHelper class with all methods and edge cases.
"""

import pytest
from unittest.mock import Mock, patch
from app.main.config_helper import ConfigHelper
from app.core.config import AppConfig


class TestConfigHelper:
    """Test ConfigHelper class functionality."""
    
    def test_init_with_default_config(self):
        """Test initialization with default config."""
        helper = ConfigHelper()
        assert helper.config is not None
        assert isinstance(helper.config, AppConfig)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        custom_config = AppConfig()
        helper = ConfigHelper(custom_config)
        assert helper.config is custom_config
    
    def test_init_with_none_config(self):
        """Test initialization with None config explicitly."""
        helper = ConfigHelper(None)
        assert helper.config is not None
        assert isinstance(helper.config, AppConfig)
    
    def test_get_camera_config_defaults(self):
        """Test get_camera_config with default values."""
        helper = ConfigHelper()
        camera_config = helper.get_camera_config()
        
        expected_keys = ['camera_index', 'fps', 'flip_horizontal', 'flip_vertical']
        assert all(key in camera_config for key in expected_keys)
        
        # Test default values
        assert camera_config['camera_index'] == 0
        assert camera_config['fps'] == 30
        assert camera_config['flip_horizontal'] is False
        assert camera_config['flip_vertical'] is False
    
    def test_get_camera_config_custom_values(self):
        """Test get_camera_config with custom config values."""
        config = Mock()
        config.camera_index = 1
        config.camera_fps = 60
        config.flip_horizontal = True
        config.flip_vertical = True
        
        helper = ConfigHelper(config)
        camera_config = helper.get_camera_config()
        
        assert camera_config['camera_index'] == 1
        assert camera_config['fps'] == 60
        assert camera_config['flip_horizontal'] is True
        assert camera_config['flip_vertical'] is True
    
    def test_get_arm_config_defaults(self):
        """Test get_arm_config with default values."""
        helper = ConfigHelper()
        arm_config = helper.get_arm_config()
        
        expected_keys = ['port', 'speed', 'safe_z', 'draw_z', 'park_position']
        assert all(key in arm_config for key in expected_keys)
        
        # Test default values
        assert arm_config['port'] is None
        assert arm_config['speed'] == 100000
        assert arm_config['safe_z'] == 15.0
        assert arm_config['draw_z'] == 5.0
        assert arm_config['park_position'] == (150, 0)
    
    def test_get_arm_config_custom_values(self):
        """Test get_arm_config with custom config values."""
        config = Mock()
        config.arm_port = '/dev/ttyUSB0'
        config.arm_speed = 50000
        config.safe_z = 20.0
        config.draw_z = 3.0
        config.park_position = (200, 50)
        
        helper = ConfigHelper(config)
        arm_config = helper.get_arm_config()
        
        assert arm_config['port'] == '/dev/ttyUSB0'
        assert arm_config['speed'] == 50000
        assert arm_config['safe_z'] == 20.0
        assert arm_config['draw_z'] == 3.0
        assert arm_config['park_position'] == (200, 50)
    
    def test_get_game_config_defaults(self):
        """Test get_game_config with default values."""
        helper = ConfigHelper()
        game_config = helper.get_game_config()
        
        expected_keys = ['difficulty', 'auto_move', 'show_debug', 'language']
        assert all(key in game_config for key in expected_keys)
        
        # Test default values
        assert game_config['difficulty'] == 10
        assert game_config['auto_move'] is True
        assert game_config['show_debug'] is False
        assert game_config['language'] == 'cs'
    
    def test_get_game_config_custom_values(self):
        """Test get_game_config with custom config values."""
        config = Mock()
        config.difficulty = 5
        config.auto_move = False
        config.show_debug = True
        config.language = 'en'
        
        helper = ConfigHelper(config)
        game_config = helper.get_game_config()
        
        assert game_config['difficulty'] == 5
        assert game_config['auto_move'] is False
        assert game_config['show_debug'] is True
        assert game_config['language'] == 'en'
    
    def test_get_gui_config_defaults(self):
        """Test get_gui_config with default values."""
        helper = ConfigHelper()
        gui_config = helper.get_gui_config()
        
        expected_keys = ['window_width', 'window_height', 'font_size', 'theme']
        assert all(key in gui_config for key in expected_keys)
        
        # Test default values
        assert gui_config['window_width'] == 800
        assert gui_config['window_height'] == 600
        assert gui_config['font_size'] == 10
        assert gui_config['theme'] == 'default'
    
    def test_get_gui_config_custom_values(self):
        """Test get_gui_config with custom config values."""
        config = Mock()
        config.window_width = 1024
        config.window_height = 768
        config.font_size = 12
        config.theme = 'dark'
        
        helper = ConfigHelper(config)
        gui_config = helper.get_gui_config()
        
        assert gui_config['window_width'] == 1024
        assert gui_config['window_height'] == 768
        assert gui_config['font_size'] == 12
        assert gui_config['theme'] == 'dark'
    
    def test_get_safe_value_simple_key(self):
        """Test get_safe_value with simple key."""
        config = Mock()
        config.test_key = 'test_value'
        
        helper = ConfigHelper(config)
        value = helper.get_safe_value('test_key', 'default')
        
        assert value == 'test_value'
    
    def test_get_safe_value_missing_key(self):
        """Test get_safe_value with missing key returns default."""
        config = Mock()
        # Simulate missing attribute
        del config.missing_key
        
        helper = ConfigHelper(config)
        value = helper.get_safe_value('missing_key', 'default_value')
        
        assert value == 'default_value'
    
    def test_get_safe_value_with_section(self):
        """Test get_safe_value with section parameter."""
        section_mock = Mock()
        section_mock.section_key = 'section_value'
        
        config = Mock()
        config.test_section = section_mock
        
        helper = ConfigHelper(config)
        value = helper.get_safe_value('section_key', 'default', 'test_section')
        
        assert value == 'section_value'
    
    def test_get_safe_value_missing_section(self):
        """Test get_safe_value with missing section returns default."""
        config = Mock()
        # Explicitly set that missing_section doesn't exist
        config.configure_mock(**{'missing_section': AttributeError()})
        
        helper = ConfigHelper(config)
        value = helper.get_safe_value('key', 'default', 'missing_section')
        
        assert value == 'default'
    
    def test_get_safe_value_none_default(self):
        """Test get_safe_value with None as default."""
        helper = ConfigHelper()
        value = helper.get_safe_value('missing_key')  # default=None
        
        assert value is None
    
    def test_all_config_methods_with_empty_config(self):
        """Test all config methods work with empty config object."""
        config = Mock()
        helper = ConfigHelper(config)
        
        # All methods should return defaults without errors
        camera_config = helper.get_camera_config()
        arm_config = helper.get_arm_config()
        game_config = helper.get_game_config()
        gui_config = helper.get_gui_config()
        
        assert isinstance(camera_config, dict)
        assert isinstance(arm_config, dict)
        assert isinstance(game_config, dict)
        assert isinstance(gui_config, dict)
    
    def test_config_methods_return_correct_types(self):
        """Test that config methods return correct data types."""
        helper = ConfigHelper()
        
        camera_config = helper.get_camera_config()
        arm_config = helper.get_arm_config()
        game_config = helper.get_game_config()
        gui_config = helper.get_gui_config()
        
        # Check specific type expectations
        assert isinstance(camera_config['camera_index'], int)
        assert isinstance(camera_config['fps'], int)
        assert isinstance(camera_config['flip_horizontal'], bool)
        assert isinstance(camera_config['flip_vertical'], bool)
        
        assert isinstance(arm_config['speed'], int)
        assert isinstance(arm_config['safe_z'], float)
        assert isinstance(arm_config['draw_z'], float)
        
        assert isinstance(game_config['difficulty'], int)
        assert isinstance(game_config['auto_move'], bool)
        assert isinstance(game_config['show_debug'], bool)
        assert isinstance(game_config['language'], str)
        
        assert isinstance(gui_config['window_width'], int)
        assert isinstance(gui_config['window_height'], int)
        assert isinstance(gui_config['font_size'], int)
        assert isinstance(gui_config['theme'], str)