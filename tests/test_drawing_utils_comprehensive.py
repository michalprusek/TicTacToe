"""
Comprehensive pytest test suite for drawing_utils.py module.
Tests drawing functions using mocked OpenCV operations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, call
from app.main.drawing_utils import (
    draw_centered_text_message,
    draw_symbol_box, 
    draw_text_lines
)


class TestDrawCenteredTextMessage:
    """Test draw_centered_text_message function."""
    
    def test_draw_centered_text_message_empty_lines(self):
        """Test with empty message lines should return early."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch('cv2.rectangle') as mock_rect:
            draw_centered_text_message(frame, [])
            mock_rect.assert_not_called()
    
    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_centered_text_message_single_line(self, mock_puttext, mock_rect, mock_textsize):
        """Test drawing single line of text."""
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        message_lines = ["Test Message"]
        
        # Mock text size calculation
        mock_textsize.return_value = ((100, 20), 5)  # (width, height), baseline
        
        draw_centered_text_message(frame, message_lines)
        
        # Verify text size was calculated
        mock_textsize.assert_called()
        
        # Verify background rectangle was drawn
        mock_rect.assert_called_once()
        
        # Verify text was drawn
        mock_puttext.assert_called_once()
    
    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_centered_text_message_multiple_lines(self, mock_puttext, mock_rect, mock_textsize):
        """Test drawing multiple lines of text."""
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        message_lines = ["Line 1", "Line 2", "Line 3"]
        
        # Mock text size calculation - different sizes for each line plus one extra for baseline
        mock_textsize.side_effect = [
            ((80, 20), 5),   # Line 1
            ((100, 20), 5),  # Line 2
            ((60, 20), 5),   # Line 3
            ((80, 20), 5)    # Extra call for baseline calculation
        ]
        
        draw_centered_text_message(frame, message_lines)
        
        # Should call getTextSize for each line plus one for baseline calculation
        assert mock_textsize.call_count == len(message_lines) + 1
        
        # Should draw background rectangle once
        mock_rect.assert_called_once()
        
        # Should draw text for each line
        assert mock_puttext.call_count == len(message_lines)
    
    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_centered_text_message_custom_parameters(self, mock_puttext, mock_rect, mock_textsize):
        """Test drawing with custom parameters."""
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        message_lines = ["Custom Text"]
        
        mock_textsize.return_value = ((100, 20), 5)
        
        draw_centered_text_message(
            frame, 
            message_lines,
            font_scale=1.5,
            text_color=(255, 0, 0),
            bg_color=(0, 255, 0),
            font_thickness=3,
            padding=20,
            y_offset_percentage=0.2
        )
        
        # Verify custom parameters were used in OpenCV calls
        mock_rect.assert_called_once()
        rect_args = mock_rect.call_args
        assert rect_args[0][3] == (0, 255, 0)  # bg_color
        
        mock_puttext.assert_called_once()
        text_args = mock_puttext.call_args
        assert text_args[0][4] == 1.5  # font_scale
        assert text_args[0][5] == (255, 0, 0)  # text_color
        assert text_args[0][6] == 3  # font_thickness
    
    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_centered_text_message_boundary_clipping(self, mock_puttext, mock_rect, mock_textsize):
        """Test text positioning with boundary clipping."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)  # Small frame
        message_lines = ["Very Long Text Message"]
        
        # Mock large text size that would exceed frame
        mock_textsize.return_value = ((200, 30), 5)
        
        draw_centered_text_message(frame, message_lines, y_offset_percentage=0.0)
        
        # Should still draw (clipped) without crashing
        mock_rect.assert_called_once()
        mock_puttext.assert_called_once()


class TestDrawSymbolBox:
    """Test draw_symbol_box function."""
    
    @patch('cv2.rectangle')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_draw_symbol_box_valid_x(self, mock_puttext, mock_textsize, mock_rect):
        """Test drawing symbol box for X symbol."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = [10, 20, 50, 60]
        confidence = 0.95
        class_id = 0  # X
        label = "X"
        
        mock_textsize.return_value = ((40, 15), 3)
        
        draw_symbol_box(frame, box, confidence, class_id, label)
        
        # Should draw bounding box
        mock_rect.assert_called()
        
        # Should calculate text size
        mock_textsize.assert_called_once()
        
        # Should draw text
        mock_puttext.assert_called_once()
    
    @patch('cv2.rectangle')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_draw_symbol_box_valid_o(self, mock_puttext, mock_textsize, mock_rect):
        """Test drawing symbol box for O symbol."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = [10, 20, 50, 60]
        confidence = 0.87
        class_id = 1  # O
        label = "O"
        
        mock_textsize.return_value = ((40, 15), 3)
        
        draw_symbol_box(frame, box, confidence, class_id, label)
        
        # Should draw bounding box
        mock_rect.assert_called()
        
        # Should calculate text size
        mock_textsize.assert_called_once()
        
        # Should draw text
        mock_puttext.assert_called_once()
    
    @patch('cv2.rectangle')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_draw_symbol_box_invalid_coordinates(self, mock_puttext, mock_textsize, mock_rect):
        """Test drawing symbol box with invalid coordinates should skip."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        invalid_box = [50, 60, 10, 20]  # x1 > x2, y1 > y2
        confidence = 0.95
        class_id = 0
        label = "X"
        
        draw_symbol_box(frame, invalid_box, confidence, class_id, label)
        
        # Should not draw anything for invalid box
        mock_rect.assert_not_called()
        mock_textsize.assert_not_called()
        mock_puttext.assert_not_called()
    
    @patch('cv2.rectangle')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_draw_symbol_box_boundary_clamping(self, mock_puttext, mock_textsize, mock_rect):
        """Test drawing symbol box with coordinates outside frame boundaries."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        box = [-10, -5, 150, 120]  # Extends beyond frame boundaries
        confidence = 0.75
        class_id = 0
        label = "X"
        
        mock_textsize.return_value = ((30, 12), 2)
        
        draw_symbol_box(frame, box, confidence, class_id, label)
        
        # Should still draw (clamped to frame bounds)
        mock_rect.assert_called()
        
        # Check that coordinates were clamped
        rect_calls = mock_rect.call_args_list
        # First call should be for the bounding box with clamped coordinates
        box_call = rect_calls[0]
        x1, y1 = box_call[0][1]  # First point
        x2, y2 = box_call[0][2]  # Second point
        
        # Coordinates should be clamped to frame bounds
        assert x1 >= 0 and x1 < 100
        assert y1 >= 0 and y1 < 100
        assert x2 >= 0 and x2 < 100
        assert y2 >= 0 and y2 < 100
    
    @patch('cv2.rectangle')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_draw_symbol_box_unknown_class_id(self, mock_puttext, mock_textsize, mock_rect):
        """Test drawing symbol box with unknown class_id."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = [10, 20, 50, 60]
        confidence = 0.85
        class_id = 5  # Unknown class_id
        label = "Unknown"
        
        mock_textsize.return_value = ((60, 15), 3)
        
        draw_symbol_box(frame, box, confidence, class_id, label)
        
        # Should still draw with default color
        mock_rect.assert_called()
        mock_textsize.assert_called_once()
        mock_puttext.assert_called_once()
    
    @patch('cv2.rectangle')
    @patch('cv2.getTextSize', side_effect=Exception("Text size error"))
    @patch('cv2.putText')
    def test_draw_symbol_box_text_size_error(self, mock_puttext, mock_textsize, mock_rect):
        """Test drawing symbol box when text size calculation fails."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = [10, 20, 50, 60]
        confidence = 0.95
        class_id = 0
        label = "X"
        
        draw_symbol_box(frame, box, confidence, class_id, label)
        
        # Should draw bounding box but skip text due to error
        mock_rect.assert_called_once()  # Only bounding box, not text background
        mock_textsize.assert_called_once()
        mock_puttext.assert_not_called()


class TestDrawTextLines:
    """Test draw_text_lines function."""
    
    @patch('cv2.putText')
    def test_draw_text_lines_single_line(self, mock_puttext):
        """Test drawing single line of text."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lines = ["Single line"]
        start_x, start_y = 10, 30
        
        draw_text_lines(frame, lines, start_x, start_y)
        
        mock_puttext.assert_called_once()
        call_args = mock_puttext.call_args
        assert call_args[0][1] == "Single line"
        assert call_args[0][2] == (start_x, start_y)
    
    @patch('cv2.putText')
    def test_draw_text_lines_multiple_lines(self, mock_puttext):
        """Test drawing multiple lines of text."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lines = ["Line 1", "Line 2", "Line 3"]
        start_x, start_y = 10, 30
        y_offset = 25
        
        draw_text_lines(frame, lines, start_x, start_y, y_offset=y_offset)
        
        assert mock_puttext.call_count == len(lines)
        
        # Check that each line is drawn at correct position
        calls = mock_puttext.call_args_list
        for i, call in enumerate(calls):
            expected_y = start_y + i * y_offset
            assert call[0][1] == lines[i]
            assert call[0][2] == (start_x, expected_y)
    
    @patch('cv2.putText')
    def test_draw_text_lines_empty_list(self, mock_puttext):
        """Test drawing empty list of lines."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lines = []
        start_x, start_y = 10, 30
        
        draw_text_lines(frame, lines, start_x, start_y)
        
        mock_puttext.assert_not_called()
    
    @patch('cv2.putText')
    def test_draw_text_lines_custom_parameters(self, mock_puttext):
        """Test drawing with custom parameters."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lines = ["Custom text"]
        start_x, start_y = 50, 100
        
        import cv2
        draw_text_lines(
            frame, lines, start_x, start_y,
            y_offset=30,
            font_face=cv2.FONT_HERSHEY_COMPLEX,
            font_scale=1.2,
            color=(0, 255, 255),
            thickness=2
        )
        
        mock_puttext.assert_called_once()
        call_args = mock_puttext.call_args
        assert call_args[0][3] == cv2.FONT_HERSHEY_COMPLEX  # font_face
        assert call_args[0][4] == 1.2  # font_scale
        assert call_args[0][5] == (0, 255, 255)  # color
        assert call_args[0][6] == 2  # thickness


class TestDrawingUtilsIntegration:
    """Integration tests for drawing utilities."""
    
    def test_all_functions_work_with_real_frame(self):
        """Test that all functions can be called without crashing."""
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        with patch('cv2.getTextSize', return_value=((50, 20), 5)):
            with patch('cv2.rectangle'):
                with patch('cv2.putText'):
                    # Test all functions
                    draw_centered_text_message(frame, ["Test message"])
                    draw_symbol_box(frame, [10, 10, 50, 50], 0.9, 0, "X")
                    draw_text_lines(frame, ["Line 1", "Line 2"], 10, 30)
    
    def test_numpy_array_compatibility(self):
        """Test compatibility with different numpy array types."""
        # Test with different dtypes
        frame_uint8 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_float32 = np.zeros((100, 100, 3), dtype=np.float32)
        
        with patch('cv2.getTextSize', return_value=((30, 15), 3)):
            with patch('cv2.rectangle'):
                with patch('cv2.putText'):
                    # Should work with both types
                    draw_centered_text_message(frame_uint8, ["Test"])
                    draw_centered_text_message(frame_float32, ["Test"])
                    
                    draw_symbol_box(frame_uint8, [5, 5, 25, 25], 0.8, 0, "X")
                    draw_symbol_box(frame_float32, [5, 5, 25, 25], 0.8, 1, "O")
                    
                    draw_text_lines(frame_uint8, ["Test"], 10, 20)
                    draw_text_lines(frame_float32, ["Test"], 10, 20)


if __name__ == "__main__":
    pytest.main([__file__])
