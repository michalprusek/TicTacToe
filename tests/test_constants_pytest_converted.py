# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Tests for constants module (converted to pytest).
"""
import pytest

from app.core.constants import (
    ERROR_GRID_INCOMPLETE_PAUSE, GRID_PARTIALLY_VISIBLE_ERROR,
    MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR,
    PLAYER_X_COLOR, PLAYER_O_COLOR,
    SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR
)


class TestConstants:
    """Test class for constants module using pytest."""
    
    def test_error_constants(self):
        """Test error message constants."""
        assert ERROR_GRID_INCOMPLETE_PAUSE == "grid_incomplete_pause"
        assert GRID_PARTIALLY_VISIBLE_ERROR == "grid_partially_visible"
        assert isinstance(ERROR_GRID_INCOMPLETE_PAUSE, str)
        assert isinstance(GRID_PARTIALLY_VISIBLE_ERROR, str)

    def test_color_constants(self):
        """Test color constants."""
        assert isinstance(MESSAGE_TEXT_COLOR, tuple)
        assert len(MESSAGE_TEXT_COLOR) == 3
        assert isinstance(MESSAGE_BG_COLOR, tuple)
        assert len(MESSAGE_BG_COLOR) == 3
        
        assert isinstance(PLAYER_X_COLOR, tuple)
        assert len(PLAYER_X_COLOR) == 3
        assert isinstance(PLAYER_O_COLOR, tuple)
        assert len(PLAYER_O_COLOR) == 3

    def test_player_colors_different(self):
        """Test that player colors are different."""
        assert PLAYER_X_COLOR != PLAYER_O_COLOR

    def test_symbol_confidence_color(self):
        """Test symbol confidence threshold text color."""
        assert isinstance(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR, tuple)
        assert len(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR) == 3

    def test_color_values_valid(self):
        """Test that color values are valid BGR format."""
        colors = [MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR, PLAYER_X_COLOR, 
                 PLAYER_O_COLOR, SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR]
        
        for color in colors:
            for channel in color:
                assert 0 <= channel <= 255
