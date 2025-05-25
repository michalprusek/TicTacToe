"""
Camera view module for the TicTacToe application.
"""
import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from app.main.frame_utils import FrameConverter


class CameraView(QLabel):
    """Widget for displaying camera feed - moderní design"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("📷 Inicializace kamery...")
        self.camera_active = False
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
            try:
                # Vytvoření pixmapy s rámečkem pomocí FrameConverter
                pixmap = FrameConverter.frame_to_pixmap(
                    frame,
                    self.width() - 10,
                    self.height() - 10,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation)

                self.setPixmap(pixmap)

                # Mark camera as active
                if not self.camera_active:
                    self.camera_active = True

            except Exception as e:
                self.setText(f"📷 Chyba při zobrazení: {str(e)}")
        else:
            if self.camera_active:
                self.setText("📷 Kamera nedostupná")
            else:
                self.setText("📷 Inicializace kamery...")

    def set_camera_status(self, status_message):
        """Set camera status message"""
        self.setText(status_message)

    def set_camera_active(self, active):
        """Set camera active state"""
        self.camera_active = active
        if not active:
            self.setText("📷 Kamera nedostupná")
