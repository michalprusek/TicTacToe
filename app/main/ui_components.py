"""
UI components for the TicTacToe application.
"""
import os
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QCheckBox,
    QGraphicsOpacityEffect
)
from app.main.game_utils import setup_logger
from app.main.gui_factory import ButtonFactory, LayoutFactory, LabelFactory
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QIcon

from app.core.config import AppConfig


class StatusPanel(QWidget):
    """Status panel widget for displaying game status"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = setup_logger(__name__)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            background-color: #3498db;
            border-radius: 10px;
            border: 2px solid #2980b9;
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: bold;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.status_label)

    def set_text(self, text):
        """Set the status text"""
        self.status_label.setText(text)

    def set_style(self, style):
        """Set the panel style"""
        self.setStyleSheet(style)


class ControlPanel(QWidget):
    """Control panel widget for game controls"""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.logger = setup_logger(__name__)
        self.config = config if config is not None else AppConfig()
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        self.setStyleSheet("""
            background-color: #333740;
            border-radius: 10px;
            padding: 10px;
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        layout.addWidget(self.reset_button)

        # Difficulty slider
        difficulty_container = QWidget()
        difficulty_layout = QVBoxLayout(difficulty_container)
        difficulty_layout.setContentsMargins(0, 0, 0, 0)

        self.difficulty_label = QLabel("Obtížnost:")
        self.difficulty_label.setStyleSheet("color: white;")
        difficulty_layout.addWidget(self.difficulty_label)

        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)

        self.difficulty_slider = QSlider(Qt.Horizontal)
        self.difficulty_slider.setMinimum(1)
        self.difficulty_slider.setMaximum(10)
        self.difficulty_slider.setValue(self.config.game.default_difficulty)
        self.difficulty_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #555;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        slider_layout.addWidget(self.difficulty_slider)

        self.difficulty_value_label = QLabel(str(self.difficulty_slider.value()))
        self.difficulty_value_label.setStyleSheet("color: white; min-width: 20px;")
        slider_layout.addWidget(self.difficulty_value_label)

        difficulty_layout.addWidget(slider_container)
        layout.addWidget(difficulty_container)

        # Park button
        self.park_button = QPushButton("Park")
        self.park_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        layout.addWidget(self.park_button)

        # Calibrate button
        self.calibrate_button = QPushButton("Kalibrace")
        self.calibrate_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        layout.addWidget(self.calibrate_button)

        # Debug button
        self.debug_button = QPushButton("Debug")
        self.debug_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        layout.addWidget(self.debug_button)

        # Track checkbox
        self.track_checkbox = QCheckBox("Track")
        self.track_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #2ecc71;
                border: 2px solid #27ae60;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.track_checkbox)

        # Language button
        self.language_button = QPushButton("EN/CZ")
        self.language_button.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #16a085;
            }
        """)
        layout.addWidget(self.language_button)


def setup_window_icon(window, config):
    """Set up the window icon"""
    icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "resources", "app_icon.png")
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))


def create_fade_animation(widget, start_value=0.0, end_value=1.0, duration=500):
    """Create a fade animation for a widget"""
    opacity_effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(opacity_effect)

    animation = QPropertyAnimation(opacity_effect, b"opacity")
    animation.setDuration(duration)
    animation.setStartValue(start_value)
    animation.setEndValue(end_value)

    return animation
