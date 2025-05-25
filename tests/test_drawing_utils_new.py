"""
Tests for app/main/drawing_utils.py module.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from app.main.drawing_utils import (
    draw_centered_text_message, draw_symbol_box, draw_text_lines
)


class TestDrawingUtils:
    """Test class for drawing utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_centered_text_message_empty_lines(self):
        """Test draw_centered_text_message with empty message lines."""
        # Should not raise an error and should return early
        draw_centered_text_message(self.test_frame, [])
        # Check that frame is unchanged (still all zeros)
        assert np.all(self.test_frame == 0)

    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_centered_text_message_single_line(self, mock_puttext, mock_rectangle, mock_gettextsize):
        """Test drawing a single line of text."""
        mock_gettextsize.return_value = ((100, 20), 5)
        
        draw_centered_text_message(self.test_frame, ["Test message"])
        
        mock_gettextsize.assert_called()
        mock_rectangle.assert_called_once()
        mock_puttext.assert_called_once()

    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_centered_text_message_multiple_lines(self, mock_puttext, mock_rectangle, mock_gettextsize):
        """Test drawing multiple lines of text."""
        mock_gettextsize.return_value = ((100, 20), 5)
        
        lines = ["Line 1", "Line 2", "Line 3"]
        draw_centered_text_message(self.test_frame, lines)
        
        # Function calls getTextSize once per line plus one additional call for baseline calculation
        assert mock_gettextsize.call_count == len(lines) + 1
        mock_rectangle.assert_called_once()
        assert mock_puttext.call_count == len(lines)

    @patch('cv2.getTextSize')
    def test_draw_centered_text_message_custom_parameters(self, mock_gettextsize):
        """Test draw_centered_text_message with custom parameters."""
        mock_gettextsize.return_value = ((100, 20), 5)
        
        draw_centered_text_message(
            self.test_frame, 
            ["Test"],
            font_scale=2.0,
            text_color=(255, 0, 0),
            bg_color=(0, 255, 0),
            font_thickness=3,
            padding=20,
            y_offset_percentage=0.3
        )
        
        mock_gettextsize.assert_called()

    def test_draw_symbol_box_invalid_box(self):
        """Test draw_symbol_box with invalid box coordinates."""
        # Box where x1 >= x2 or y1 >= y2 should be skipped
        invalid_box = [100, 100, 50, 50]  # x1 > x2, y1 > y2
        
        # Should not raise an error
        draw_symbol_box(self.test_frame, invalid_box, 0.95, 0, "X")

    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_symbol_box_valid_x_symbol(self, mock_gettextsize, mock_puttext, mock_rectangle):
        """Test drawing valid X symbol box."""
        mock_gettextsize.return_value = ((50, 15), 3)
        valid_box = [50, 50, 150, 150]
        
        draw_symbol_box(self.test_frame, valid_box, 0.95, 0, "X")
        
        mock_rectangle.assert_called()
        mock_puttext.assert_called()

    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_symbol_box_valid_o_symbol(self, mock_gettextsize, mock_puttext, mock_rectangle):
        """Test drawing valid O symbol box."""
        mock_gettextsize.return_value = ((50, 15), 3)
        valid_box = [50, 50, 150, 150]
        
        draw_symbol_box(self.test_frame, valid_box, 0.85, 1, "O")
        
        mock_rectangle.assert_called()
        mock_puttext.assert_called()

    @patch('cv2.getTextSize')
    def test_draw_symbol_box_gettextsize_exception(self, mock_gettextsize):
        """Test draw_symbol_box when getTextSize raises exception."""
        mock_gettextsize.side_effect = Exception("Test exception")
        valid_box = [50, 50, 150, 150]
        
        # Should not raise an error
        draw_symbol_box(self.test_frame, valid_box, 0.95, 0, "X")

    def test_draw_symbol_box_coordinates_clamping(self):
        """Test that box coordinates are properly clamped to frame bounds."""
        # Box that exceeds frame bounds
        oversized_box = [-50, -50, 700, 500]  # Exceeds 640x480 frame
        
        # Should not raise an error
        draw_symbol_box(self.test_frame, oversized_box, 0.95, 0, "X")

    def test_draw_symbol_box_confidence_types(self):
        """Test draw_symbol_box with different confidence value types."""
        valid_box = [50, 50, 150, 150]
        
        # Test with different numeric types
        draw_symbol_box(self.test_frame, valid_box, 0.95, 0, "X")  # float
        draw_symbol_box(self.test_frame, valid_box, 1, 0, "X")     # int
        draw_symbol_box(self.test_frame, valid_box, np.float32(0.8), 0, "X")  # numpy float
        draw_symbol_box(self.test_frame, valid_box, "invalid", 0, "X")  # invalid type

    def test_draw_symbol_box_unknown_class_id(self):
        """Test draw_symbol_box with unknown class_id."""
        valid_box = [50, 50, 150, 150]
        
        # Should use default green color for unknown class_id
        draw_symbol_box(self.test_frame, valid_box, 0.95, 99, "Unknown")

    @patch('cv2.putText')
    def test_draw_text_lines_empty_list(self, mock_puttext):
        """Test draw_text_lines with empty lines list."""
        draw_text_lines(self.test_frame, [], 10, 10)
        
        mock_puttext.assert_not_called()

    @patch('cv2.putText')
    def test_draw_text_lines_single_line(self, mock_puttext):
        """Test drawing single line of text."""
        draw_text_lines(self.test_frame, ["Single line"], 10, 30)
        
        mock_puttext.assert_called_once_with(
            self.test_frame, "Single line", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    @patch('cv2.putText')
    def test_draw_text_lines_multiple_lines(self, mock_puttext):
        """Test drawing multiple lines of text."""
        lines = ["Line 1", "Line 2", "Line 3"]
        draw_text_lines(self.test_frame, lines, 10, 30, y_offset=25)
        
        assert mock_puttext.call_count == 3
        # Check that y-coordinates increase by y_offset
        calls = mock_puttext.call_args_list
        assert calls[0][0][2] == (10, 30)    # First line at y=30
        assert calls[1][0][2] == (10, 55)    # Second line at y=55
        assert calls[2][0][2] == (10, 80)    # Third line at y=80

    @patch('cv2.putText')
    def test_draw_text_lines_custom_parameters(self, mock_puttext):
        """Test draw_text_lines with custom parameters."""
        draw_text_lines(
            self.test_frame, 
            ["Test line"],
            start_x=50, start_y=100, y_offset=30,
            font_face=cv2.FONT_HERSHEY_DUPLEX,
            font_scale=1.0,
            color=(0, 255, 0),
            thickness=2
        )
        
        mock_puttext.assert_called_once_with(
            self.test_frame, "Test line", (50, 100),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2
        )

    def test_frame_integrity(self):
        """Test that functions don't crash with edge case frames."""
        # Test with very small frame
        tiny_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        draw_centered_text_message(tiny_frame, ["Test"])
        draw_symbol_box(tiny_frame, [0, 0, 1, 1], 0.5, 0, "X")
        draw_text_lines(tiny_frame, ["Test"], 0, 0)
        
        # Test with large frame
        large_frame = np.zeros((2000, 2000, 3), dtype=np.uint8)
        draw_centered_text_message(large_frame, ["Test"])
        draw_symbol_box(large_frame, [100, 100, 200, 200], 0.5, 0, "X")
        draw_text_lines(large_frame, ["Test"], 100, 100)

    def test_luminance_calculation(self):
        """Test luminance calculation in draw_symbol_box."""
        # Test with different colors to ensure luminance calculation works
        colors_to_test = [
            (0, 0, 255),    # Red - low luminance
            (255, 255, 255), # White - high luminance  
            (0, 0, 0),      # Black - low luminance
            (128, 128, 128) # Gray - medium luminance
        ]
        
        valid_box = [50, 50, 150, 150]
        for color in colors_to_test:
            # Should not raise errors for any color
            draw_symbol_box(
                self.test_frame, valid_box, 0.95, 0, "X",
                player_x_color=color
            )
