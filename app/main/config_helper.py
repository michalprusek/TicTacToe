"""
Configuration helper utilities for standardized config access.
Consolidates repeated configuration access patterns.
"""

from typing import Any, Optional

from app.core.config import AppConfig


class ConfigHelper:
    """Helper for standardized configuration access."""

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize config helper.

        Args:
            config: Application configuration (creates default if None)
        """
        self.config = config if config is not None else AppConfig()

    def get_camera_config(self) -> dict:
        """Get camera-related configuration values."""
        return {
            'camera_index': getattr(self.config, 'camera_index', 0),
            'fps': getattr(self.config, 'camera_fps', 30),
            'flip_horizontal': getattr(self.config, 'flip_horizontal', False),
            'flip_vertical': getattr(self.config, 'flip_vertical', False)
        }

    def get_arm_config(self) -> dict:
        """Get arm-related configuration values."""
        return {
            'port': getattr(self.config, 'arm_port', None),
            'speed': getattr(self.config, 'arm_speed', 100000),
            'safe_z': getattr(self.config, 'safe_z', 15.0),
            'draw_z': getattr(self.config, 'draw_z', 5.0),
            'park_position': getattr(self.config, 'park_position', (150, 0))
        }

    def get_game_config(self) -> dict:
        """Get game-related configuration values."""
        return {
            'difficulty': getattr(self.config, 'difficulty', 10),
            'auto_move': getattr(self.config, 'auto_move', True),
            'show_debug': getattr(self.config, 'show_debug', False),
            'language': getattr(self.config, 'language', 'cs')
        }

    def get_gui_config(self) -> dict:
        """Get GUI-related configuration values."""
        return {
            'window_width': getattr(self.config, 'window_width', 800),
            'window_height': getattr(self.config, 'window_height', 600),
            'font_size': getattr(self.config, 'font_size', 10),
            'theme': getattr(self.config, 'theme', 'default')
        }

    def get_safe_value(self, key: str, default: Any = None,
                      section: Optional[str] = None) -> Any:
        """
        Safely get configuration value with fallback.

        Args:
            key: Configuration key
            default: Default value if key not found
            section: Optional section for nested configs

        Returns:
            Configuration value or default
        """
        try:
            if section and hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                return getattr(section_obj, key, default)
            return getattr(self.config, key, default)
        except (AttributeError, TypeError):
            return default
