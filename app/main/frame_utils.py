"""
Frame processing utilities for the TicTacToe application.
Consolidates repeated frame processing patterns from multiple files.
"""

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class FrameConverter:
    """Utilities for frame format conversions and processing."""

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB format."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def frame_to_qimage(frame: np.ndarray) -> QImage:
        """
        Convert frame to QImage for PyQt display.

        Args:
            frame: Input frame in BGR format

        Returns:
            QImage object ready for PyQt display
        """
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_frame = FrameConverter.bgr_to_rgb(frame)

        return QImage(
            rgb_frame.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )

    @staticmethod
    def frame_to_pixmap(frame: np.ndarray, target_width: int,
                       target_height: int, keep_aspect: bool = True) -> QPixmap:
        """
        Convert frame to scaled QPixmap.

        Args:
            frame: Input frame
            target_width: Target width
            target_height: Target height
            keep_aspect: Whether to maintain aspect ratio

        Returns:
            Scaled QPixmap
        """
        qt_image = FrameConverter.frame_to_qimage(frame)

        if keep_aspect:
            return QPixmap.fromImage(qt_image).scaled(
                target_width, target_height,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            return QPixmap.fromImage(qt_image).scaled(
                target_width, target_height
            )

    @staticmethod
    def resize_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Resize frame by scale factor.

        Args:
            frame: Input frame
            scale_factor: Scale factor (e.g., 0.5 for half size)

        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        return cv2.resize(frame, (new_width, new_height))
