"""
Extended tests for GameStateManager class.
Pure pytest implementation without unittest.
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from app.main.game_state_manager import GameStateManager
from app.core.config import GameDetectorConfig


class TestGameStateManagerExtended:
    """Extended tests for GameStateManager functionality."""

    def test_init_default(self):
        """Test GameStateManager initialization with defaults."""
        manager = GameStateManager()
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'logger')

    def test_init_with_config(self):
        """Test GameStateManager initialization with config."""
        config = GameDetectorConfig()
        manager = GameStateManager(config=config)
        assert hasattr(manager, 'config')

    def test_manager_attributes(self):
        """Test GameStateManager basic attributes."""
        manager = GameStateManager()
        assert manager.config is None  # Default config
        assert manager.logger is not None

    def test_config_attribute_access(self):
        """Test config attribute access when provided."""
        config = GameDetectorConfig()
        manager = GameStateManager(config=config)
        assert manager.config is not None
        assert isinstance(manager.config, GameDetectorConfig)