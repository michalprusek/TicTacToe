# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Tests for GameStateManager.
"""
import pytest
import numpy as np
from unittest.mock import Mock
from app.main.game_state_manager import GameStateManager
from app.core.game_state import GameState


class TestGameStateManager:
    """Test GameStateManager class."""

    def test_init_basic(self):
        """Test basic initialization."""
        manager = GameStateManager()
        assert manager.config is None
        assert manager.logger is not None
        assert isinstance(manager.game_state, GameState)
        assert manager.class_id_to_player == {0: 1, 1: 2}

    def test_init_with_config(self):
        """Test initialization with config."""
        config = Mock()
        config.class_id_to_player = {0: 'X', 1: 'O'}
        logger = Mock()
        
        manager = GameStateManager(config, logger)
        assert manager.config == config
        assert manager.logger == logger
        assert manager.class_id_to_player == {0: 'X', 1: 'O'}

    def test_class_id_mapping_default(self):
        """Test default class ID mapping."""
        manager = GameStateManager()
        
        # Test default mapping
        assert manager.class_id_to_player[0] == 1  # X
        assert manager.class_id_to_player[1] == 2  # O    def test_game_state_access(self):
        """Test game state access and modification."""
        manager = GameStateManager()
        
        # Test initial state
        assert manager.game_state is not None
        assert isinstance(manager.game_state, GameState)
        
        # Test that we can access game state properties
        initial_board = manager.game_state.board
        assert initial_board is not None

    def test_config_attribute_access(self):
        """Test config attribute access with getattr fallback."""
        # Test with config that doesn't have class_id_to_player attribute
        config_without_attr = Mock(spec=[])  # Empty spec means no attributes
        
        manager = GameStateManager(config_without_attr)
        
        # Should fall back to default mapping
        assert manager.class_id_to_player == {0: 1, 1: 2}
