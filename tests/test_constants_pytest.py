# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Pure pytest tests for constants module.
"""
import pytest
from app.core.constants import (
    ERROR_GRID_INCOMPLETE_PAUSE, GRID_PARTIALLY_VISIBLE_ERROR,
    MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR, PLAYER_X_COLOR, PLAYER_O_COLOR,
    SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR
)


class TestConstants:
    """Pure pytest test class for constants."""
    
    def test_error_constants(self):
        """Test error constants."""
        assert ERROR_GRID_INCOMPLETE_PAUSE == "grid_incomplete_pause"
        assert GRID_PARTIALLY_VISIBLE_ERROR == "grid_partially_visible"
    
    def test_message_colors(self):
        """Test message color constants."""
        assert MESSAGE_TEXT_COLOR == (255, 255, 255)
        assert MESSAGE_BG_COLOR == (25, 25, 150)
        assert isinstance(MESSAGE_TEXT_COLOR, tuple)
        assert isinstance(MESSAGE_BG_COLOR, tuple)
        assert len(MESSAGE_TEXT_COLOR) == 3
        assert len(MESSAGE_BG_COLOR) == 3
    
    def test_player_colors(self):
        """Test player color constants."""
        assert PLAYER_X_COLOR == (0, 0, 255)
        assert PLAYER_O_COLOR == (0, 255, 0)
        assert isinstance(PLAYER_X_COLOR, tuple)
        assert isinstance(PLAYER_O_COLOR, tuple)
        assert len(PLAYER_X_COLOR) == 3
        assert len(PLAYER_O_COLOR) == 3
    
    def test_symbol_confidence_color(self):
        """Test symbol confidence threshold text color."""
        assert SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR == (255, 255, 255)
        assert isinstance(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR, tuple)
        assert len(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR) == 3
