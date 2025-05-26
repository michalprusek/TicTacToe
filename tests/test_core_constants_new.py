# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.core.constants module using pytest.
Tests all constants and their values.
"""

import pytest
from app.core.constants import (
    ERROR_GRID_INCOMPLETE_PAUSE, GRID_PARTIALLY_VISIBLE_ERROR,
    MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR, PLAYER_X_COLOR, PLAYER_O_COLOR,
    SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR
)


class TestConstants:
    """Test all constants in the module."""
    
    def test_error_constants(self):
        """Test error message constants."""
        assert ERROR_GRID_INCOMPLETE_PAUSE == "grid_incomplete_pause"
        assert GRID_PARTIALLY_VISIBLE_ERROR == "grid_partially_visible"
        assert isinstance(ERROR_GRID_INCOMPLETE_PAUSE, str)
        assert isinstance(GRID_PARTIALLY_VISIBLE_ERROR, str)
    
    def test_color_constants(self):
        """Test color constants are tuples with correct values."""
        # Message colors
        assert MESSAGE_TEXT_COLOR == (255, 255, 255)  # White
        assert MESSAGE_BG_COLOR == (25, 25, 150)      # Dark blue
        assert isinstance(MESSAGE_TEXT_COLOR, tuple)
        assert isinstance(MESSAGE_BG_COLOR, tuple)
        assert len(MESSAGE_TEXT_COLOR) == 3
        assert len(MESSAGE_BG_COLOR) == 3
        
        # Player colors (BGR format)
        assert PLAYER_X_COLOR == (0, 0, 255)    # Red in BGR
        assert PLAYER_O_COLOR == (0, 255, 0)    # Green in BGR
        assert isinstance(PLAYER_X_COLOR, tuple)
        assert isinstance(PLAYER_O_COLOR, tuple)
        assert len(PLAYER_X_COLOR) == 3
        assert len(PLAYER_O_COLOR) == 3
        
        # Symbol detection color
        assert SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR == (255, 255, 255)  # White
        assert isinstance(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR, tuple)
        assert len(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR) == 3
    
    def test_color_values_range(self):
        """Test color values are in valid RGB range (0-255)."""
        colors = [
            MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR, PLAYER_X_COLOR, 
            PLAYER_O_COLOR, SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR
        ]
        
        for color in colors:
            for component in color:
                assert 0 <= component <= 255, f"Color component {component} out of range in {color}"
                assert isinstance(component, int), f"Color component {component} is not int in {color}"
    
    def test_player_colors_different(self):
        """Test player colors are different."""
        assert PLAYER_X_COLOR != PLAYER_O_COLOR
    
    def test_constants_immutability(self):
        """Test that constants are immutable types."""
        # String constants should be immutable
        assert isinstance(ERROR_GRID_INCOMPLETE_PAUSE, str)
        assert isinstance(GRID_PARTIALLY_VISIBLE_ERROR, str)
        
        # Tuple constants should be immutable
        assert isinstance(MESSAGE_TEXT_COLOR, tuple)
        assert isinstance(MESSAGE_BG_COLOR, tuple)
        assert isinstance(PLAYER_X_COLOR, tuple)
        assert isinstance(PLAYER_O_COLOR, tuple)
        assert isinstance(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR, tuple)
    
    def test_constant_naming_conventions(self):
        """Test that constants follow naming conventions."""
        import app.core.constants as constants
        
        # Get all public constants (uppercase)
        constant_names = [name for name in dir(constants) 
                         if not name.startswith('_') and name.isupper()]
        
        expected_constants = {
            'ERROR_GRID_INCOMPLETE_PAUSE',
            'GRID_PARTIALLY_VISIBLE_ERROR', 
            'MESSAGE_TEXT_COLOR',
            'MESSAGE_BG_COLOR',
            'PLAYER_X_COLOR',
            'PLAYER_O_COLOR',
            'SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR'
        }
        
        assert set(constant_names) == expected_constants
    
    def test_color_semantic_meaning(self):
        """Test color constants have expected semantic meaning."""
        # White colors for text should be (255, 255, 255)
        assert MESSAGE_TEXT_COLOR == (255, 255, 255)
        assert SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR == (255, 255, 255)
        
        # Player X is red in BGR format
        assert PLAYER_X_COLOR[2] == 255  # Red component high
        assert PLAYER_X_COLOR[0] == 0    # Blue component low
        assert PLAYER_X_COLOR[1] == 0    # Green component low
        
        # Player O is green in BGR format  
        assert PLAYER_O_COLOR[1] == 255  # Green component high
        assert PLAYER_O_COLOR[0] == 0    # Blue component low
        assert PLAYER_O_COLOR[2] == 0    # Red component low