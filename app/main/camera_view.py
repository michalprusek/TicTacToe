"""
Camera view module for the TicTacToe application.
"""
import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class CameraView(QLabel):
    """Widget for displaying camera feed - moderní design"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Kamera nedostupná")
        self.setStyleSheet("""
            background-color: #1E1E1E;
            color: white;
            border: 2px solid #444444;
            border-radius: 5px;
            padding: 5px;
        """)

    def update_frame(self, frame):
        """Update the displayed frame"""
        if frame is not None:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(
                rgb_frame.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888)

            # Vytvoření pixmapy s rámečkem
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.width() - 10, self.height() - 10, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.setPixmap(pixmap)
        else:
            self.setText("📷 Kamera nedostupná")
