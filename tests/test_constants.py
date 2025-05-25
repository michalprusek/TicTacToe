"""
Tests for constants module.
"""
import unittest

from app.core.constants import (
    ERROR_GRID_INCOMPLETE_PAUSE, GRID_PARTIALLY_VISIBLE_ERROR,
    MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR,
    PLAYER_X_COLOR, PLAYER_O_COLOR,
    SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR
)


class TestConstants(unittest.TestCase):
    
    def test_error_constants(self):
        """Test error message constants."""
        self.assertEqual(ERROR_GRID_INCOMPLETE_PAUSE, "grid_incomplete_pause")
        self.assertEqual(GRID_PARTIALLY_VISIBLE_ERROR, "grid_partially_visible")
        self.assertIsInstance(ERROR_GRID_INCOMPLETE_PAUSE, str)
        self.assertIsInstance(GRID_PARTIALLY_VISIBLE_ERROR, str)
    
    def test_message_colors(self):
        """Test message color constants."""
        self.assertEqual(MESSAGE_TEXT_COLOR, (255, 255, 255))
        self.assertEqual(MESSAGE_BG_COLOR, (25, 25, 150))
        self.assertIsInstance(MESSAGE_TEXT_COLOR, tuple)
        self.assertIsInstance(MESSAGE_BG_COLOR, tuple)
        self.assertEqual(len(MESSAGE_TEXT_COLOR), 3)
        self.assertEqual(len(MESSAGE_BG_COLOR), 3)
    
    def test_player_colors(self):
        """Test player color constants."""
        self.assertEqual(PLAYER_X_COLOR, (0, 0, 255))  # Red in BGR
        self.assertEqual(PLAYER_O_COLOR, (0, 255, 0))  # Green in BGR
        self.assertIsInstance(PLAYER_X_COLOR, tuple)
        self.assertIsInstance(PLAYER_O_COLOR, tuple)
        self.assertEqual(len(PLAYER_X_COLOR), 3)
        self.assertEqual(len(PLAYER_O_COLOR), 3)
    
    def test_symbol_text_color(self):
        """Test symbol confidence text color."""
        self.assertEqual(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR, (255, 255, 255))
        self.assertIsInstance(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR, tuple)
        self.assertEqual(len(SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR), 3)
    
    def test_color_values_range(self):
        """Test that color values are in valid RGB range."""
        colors = [MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR, PLAYER_X_COLOR, 
                 PLAYER_O_COLOR, SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR]        
        for color in colors:
            for value in color:
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 255)
                self.assertIsInstance(value, int)


if __name__ == '__main__':
    unittest.main()