"""Drawing utility functions for the Tic Tac Toe application."""
import cv2
import numpy as np
from typing import List, Tuple

from app.core.constants import (
    MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR, PLAYER_X_COLOR, PLAYER_O_COLOR,
    SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR
)


def draw_centered_text_message(
    frame: np.ndarray,
    message_lines: list[str],
    font_scale: float = 1.0,
    text_color: tuple[int, int, int] = MESSAGE_TEXT_COLOR,
    bg_color: tuple[int, int, int] = MESSAGE_BG_COLOR,
    font_thickness: int = 2,
    padding: int = 10,
    y_offset_percentage: float = 0.5  # 0.5 for center, 0.1 for top etc.
) -> None:
    """Draws a list of text messages centered on the frame, with a background.

    Args:
        frame: The image (NumPy array) to draw on.
        message_lines: A list of strings, where each string is a line of text.
        font_scale: Font scale factor.
        text_color: Text color in BGR format.
        bg_color: Background color for the text box in BGR format.
        font_thickness: Thickness of the lines used to draw the text.
        padding: Padding around the text within the background box.
        y_offset_percentage: Vertical position of the text block
                             (0.0=top, 0.5=center, 1.0=bottom).
    """
    if not message_lines:
        return

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    line_heights = []
    text_widths = []

    for line in message_lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font_face, font_scale, font_thickness
        )
        line_heights.append(text_height + baseline)
        text_widths.append(text_width)

    max_text_width = max(text_widths) if text_widths else 0
    total_text_height = sum(line_heights)

    # Calculate background rectangle coordinates
    rect_width = max_text_width + 2 * padding
    rect_height = total_text_height + 2 * padding
    rect_x1 = (frame.shape[1] - rect_width) // 2
    rect_y1 = int((frame.shape[0] - rect_height) * y_offset_percentage)

    # Ensure rect_y1 is not negative if text block is too tall for y_offset_percentage
    rect_y1 = max(0, rect_y1)

    rect_x2 = rect_x1 + rect_width
    rect_y2 = rect_y1 + rect_height

    # Draw background rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Draw each line of text
    current_y_baseline_for_draw = rect_y1 + padding + line_heights[0] - (line_heights[0] - cv2.getTextSize(message_lines[0], font_face, font_scale, font_thickness)[1]) // 2

    for i, line in enumerate(message_lines):
        # Center each line horizontally within the background rectangle
        text_x = rect_x1 + padding + (max_text_width - text_widths[i]) // 2
        cv2.putText(
            frame, line, (text_x, current_y_baseline_for_draw),
            font_face, font_scale, text_color, font_thickness
        )
        if i < len(message_lines) - 1:
            current_y_baseline_for_draw += line_heights[i+1]


def draw_symbol_box(
    frame: np.ndarray,
    box: List[int],  # [x1, y1, x2, y2]
    confidence: float,
    class_id: int,
    label: str,
    player_x_color: Tuple[int, int, int] = PLAYER_X_COLOR,
    player_o_color: Tuple[int, int, int] = PLAYER_O_COLOR,
    text_color: Tuple[int, int, int] = SYMBOL_CONFIDENCE_THRESHOLD_TEXT_COLOR,
    font_scale: float = 0.7,
    thickness: int = 2
) -> None:
    """Draws a bounding box for a detected symbol (X or O) with label and confidence.

    Args:
        frame: The image (NumPy array) to draw on.
        box: A list containing the coordinates [x1, y1, x2, y2] of the bounding box.
        confidence: The confidence score of the detection.
        class_id: The class ID of the detected symbol (0 for 'O', 1 for 'X').
        label: The class label ('O' or 'X').
        player_x_color: Color for 'X' symbol bounding box.
        player_o_color: Color for 'O' symbol bounding box.
        text_color: Color for the confidence text.
        font_scale: Font scale for the confidence text.
        thickness: Thickness for the bounding box lines.
    """
    box_x1, box_y1, box_x2, box_y2 = map(int, box)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Determine color based on class_id
    if class_id == 0:  # Assuming 0 is 'O'
        color = player_o_color
    elif class_id == 1:  # Assuming 1 is 'X'
        color = player_x_color
    else:
        color = (0, 255, 0)  # Default to green if class_id is unexpected

    # Format the display text (label and confidence)
    # Ensure confidence is a float for formatting
    conf_float = float(confidence) if isinstance(confidence, (int, float, np.floating, np.integer)) else 0.0
    display_text = f"{label}: {conf_float:.2f}"

    # Draw bounding box
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, thickness)

    # Calculate text size to position it above the box
    (text_width, text_height), baseline = cv2.getTextSize(
        display_text, font, font_scale, thickness
    )

    # Position text background and text
    text_bg_y1 = box_y1 - text_height - baseline - 5
    text_bg_y2 = box_y1 - 5

    # Ensure text background does not go off-screen (top)
    if text_bg_y1 < 0:
        text_bg_y1 = box_y1 + 5
        text_bg_y2 = box_y1 + text_height + baseline + 5

    # Draw a filled rectangle as background for the text for better visibility
    cv2.rectangle(
        frame,
        (box_x1, text_bg_y1),
        (box_x1 + text_width, text_bg_y2),
        color,  # Use symbol color for text background
        cv2.FILLED
    )

    # Determine text color for contrast with background
    # Simple check: if background is dark, use white text, else black.
    # This assumes BGR format for color.
    # A more robust method would convert to a colorspace like HSL/HSV.
    luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_on_primary_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    cv2.putText(
        frame,
        display_text,
        (box_x1, box_y1 - baseline - 5 if text_bg_y1 >=0 else box_y1 + text_height + 5),
        font,
        font_scale,
        text_on_primary_color,
        thickness
    )


def draw_text_lines(
    frame: np.ndarray,
    lines: list[str],
    start_x: int,
    start_y: int,
    y_offset: int = 20,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1
) -> None:
    """Draws multiple lines of text sequentially on a frame.

    Args:
        frame: The image (NumPy array) to draw on.
        lines: A list of strings, where each string is a line of text.
        start_x: The x-coordinate for the start of the text lines.
        start_y: The y-coordinate for the start of the first text line.
        y_offset: The vertical offset between consecutive lines of text.
        font_face: OpenCV font type.
        font_scale: Font scale factor.
        color: Text color in BGR format.
        thickness: Thickness of the lines used to draw the text.
    """
    current_y = start_y
    for line in lines:
        cv2.putText(frame, line, (start_x, current_y),
                    font_face, font_scale, color, thickness)
        current_y += y_offset
