"""
Debug window module for TicTacToe application.
"""
# pylint: disable=no-name-in-module,too-many-instance-attributes,too-many-statements
# pylint: disable=broad-exception-caught,unused-variable,invalid-name,wrong-import-order
# pylint: disable=line-too-long,unused-import,ungrouped-imports
import logging
from typing import Optional, List

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QSlider, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from app.core.config import AppConfig
from app.main.camera_view import CameraView
from app.main.game_utils import setup_logger


class DebugWindow(QMainWindow):
    """Debug window for TicTacToe application."""

    def __init__(self, config: Optional[AppConfig] = None, parent=None):
        """Initialize the debug window."""
        super().__init__(parent)
        self.logger = setup_logger(__name__)
        self.config = config if config is not None else AppConfig()

        # Window setup
        self.setWindowTitle(self.config.game.debug_window_title)
        self.resize(800, 600)

        # Initialize UI
        self.init_ui()

        # Connect signals
        self.connect_signals()

    def init_ui(self):
        """Initialize the UI components."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Camera view
        self.camera_view = CameraView()
        self.camera_view.set_camera_status("📷 Připojování kamery...")
        main_layout.addWidget(self.camera_view, 1)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Camera controls
        camera_group = QGroupBox("Kamera")
        camera_layout = QVBoxLayout(camera_group)

        # Camera selection
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("Kamera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Kamera {i}" for i in range(3)])  # Add available cameras
        camera_select_layout.addWidget(self.camera_combo)
        camera_layout.addLayout(camera_select_layout)

        # Camera settings
        self.autofocus_checkbox = QCheckBox("Vypnout autofocus")
        self.autofocus_checkbox.setChecked(True)
        camera_layout.addWidget(self.autofocus_checkbox)

        # Refresh button
        self.refresh_button = QPushButton("Obnovit kameru")
        camera_layout.addWidget(self.refresh_button)

        controls_layout.addWidget(camera_group)

        # Detection controls
        detection_group = QGroupBox("Detekce")
        detection_layout = QVBoxLayout(detection_group)

        # Grid confidence threshold (fixed)
        grid_conf_layout = QHBoxLayout()
        grid_conf_layout.addWidget(QLabel("Práh jistoty mřížky:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(45)  # Default 0.45
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setEnabled(False)  # Fixed for grid
        grid_conf_layout.addWidget(self.conf_slider)
        self.conf_value_label = QLabel("0.45")
        grid_conf_layout.addWidget(self.conf_value_label)
        detection_layout.addLayout(grid_conf_layout)

        # Symbol confidence threshold (adjustable)
        symbol_conf_layout = QHBoxLayout()
        symbol_conf_layout.addWidget(QLabel("Práh jistoty symbolů:"))
        self.symbol_conf_slider = QSlider(Qt.Horizontal)
        self.symbol_conf_slider.setMinimum(50)
        self.symbol_conf_slider.setMaximum(95)
        self.symbol_conf_slider.setValue(80)  # Default 0.80
        self.symbol_conf_slider.setTickPosition(QSlider.TicksBelow)
        self.symbol_conf_slider.setTickInterval(5)
        symbol_conf_layout.addWidget(self.symbol_conf_slider)
        self.symbol_conf_value_label = QLabel("0.80")
        symbol_conf_layout.addWidget(self.symbol_conf_value_label)
        detection_layout.addLayout(symbol_conf_layout)

        # Show detections checkbox
        self.show_detections_checkbox = QCheckBox("Zobrazit detekce")
        self.show_detections_checkbox.setChecked(True)
        detection_layout.addWidget(self.show_detections_checkbox)

        # Show grid checkbox
        self.show_grid_checkbox = QCheckBox("Zobrazit mřížku")
        self.show_grid_checkbox.setChecked(True)
        detection_layout.addWidget(self.show_grid_checkbox)

        controls_layout.addWidget(detection_group)

        # Status group
        status_group = QGroupBox("Stav")
        status_layout = QVBoxLayout(status_group)

        # Status label
        self.status_label = QLabel("Připraven")
        status_layout.addWidget(self.status_label)

        # FPS label
        self.fps_label = QLabel("FPS: 0.0")
        status_layout.addWidget(self.fps_label)

        # Board state label
        self.board_state_label = QLabel("Stav hry: Nedetekováno")
        status_layout.addWidget(self.board_state_label)

        controls_layout.addWidget(status_group)

        main_layout.addLayout(controls_layout)

    def connect_signals(self):
        """Connect UI signals to slots."""
        # Camera selection
        self.camera_combo.currentIndexChanged.connect(self.handle_camera_changed)

        # Confidence threshold
        self.conf_slider.valueChanged.connect(self.handle_conf_changed)

        # Symbol confidence threshold
        self.symbol_conf_slider.valueChanged.connect(self.handle_symbol_conf_changed)

        # Refresh button
        self.refresh_button.clicked.connect(self.handle_refresh_clicked)

        # Checkboxes
        self.autofocus_checkbox.stateChanged.connect(self.handle_autofocus_changed)
        self.show_detections_checkbox.stateChanged.connect(self.handle_show_detections_changed)
        self.show_grid_checkbox.stateChanged.connect(self.handle_show_grid_changed)

    @pyqtSlot(int)
    def handle_camera_changed(self, index: int):
        """Handle camera selection change."""
        self.logger.info("Změna kamery na index {index}")
        # Emit signal to parent to change camera
        if hasattr(self.parent(), "handle_camera_changed") and callable(self.parent().handle_camera_changed):
            self.parent().handle_camera_changed(index)

    @pyqtSlot(int)
    def handle_conf_changed(self, value: int):
        """Handle confidence threshold change."""
        conf = value / 100.0
        self.conf_value_label.setText(f"{conf:.2f}")
        self.logger.info("Změna prahu jistoty na {conf:.2f}")
        # Předání nastavení rodičovskému oknu
        if hasattr(self.parent(), "camera_thread") and self.parent().camera_thread:
            if hasattr(self.parent().camera_thread, "detection_thread") and self.parent().camera_thread.detection_thread:
                # Nastavení prahu jistoty v detection_thread
                detection_thread = self.parent().camera_thread.detection_thread
                if hasattr(detection_thread, "detector") and detection_thread.detector:
                    # Nastavení na GameDetector instanci
                    detection_thread.detector.bbox_conf_threshold = conf
                    detection_thread.detector.pose_conf_threshold = conf
                elif hasattr(detection_thread, "confidence_threshold"):
                    detection_thread.confidence_threshold = conf
                elif hasattr(detection_thread, "config"):
                    detection_thread.config.bbox_conf_threshold = conf
                    detection_thread.config.pose_conf_threshold = conf

    @pyqtSlot(int)
    def handle_symbol_conf_changed(self, value: int):
        """Handle symbol confidence threshold change."""
        conf = value / 100.0
        self.symbol_conf_value_label.setText(f"{conf:.2f}")
        self.logger.info("Změna prahu jistoty symbolů na {conf:.2f}")

        # Update symbol confidence in game state
        if hasattr(self.parent(), "camera_thread") and self.parent().camera_thread:
            if hasattr(self.parent().camera_thread, "detection_thread") and self.parent().camera_thread.detection_thread:
                detection_thread = self.parent().camera_thread.detection_thread
                if hasattr(detection_thread, "detector") and detection_thread.detector:
                    if hasattr(detection_thread.detector, "game_state") and detection_thread.detector.game_state:
                        detection_thread.detector.game_state.symbol_confidence_threshold = conf
                        self.logger.info("Updated symbol confidence threshold in game state to {conf:.2f}")

    @pyqtSlot()
    def handle_refresh_clicked(self):
        """Handle refresh button click."""
        self.logger.info("Obnovení kamery")
        try:
            # Předání požadavku na obnovení kamery rodičovskému oknu
            if hasattr(self.parent(), "handle_camera_changed") and callable(self.parent().handle_camera_changed):
                # Získáme aktuální kameru
                current_index = self.camera_combo.currentIndex()
                # Restartujeme kameru pomocí volání handle_camera_changed s aktuálním indexem
                self.parent().handle_camera_changed(current_index)
                # Aktualizujeme status
                self.status_label.setText(f"Kamera {current_index} obnovena")
        except Exception as e:
            self.logger.error("Chyba při obnovování kamery: {e}")
            self.status_label.setText(f"Chyba při obnovování kamery: {str(e)}")
            # Nastavení textu i když kamera selže
            self.camera_view.setText("Chyba kamery - zkuste restartovat aplikaci")

    @pyqtSlot(int)
    def handle_autofocus_changed(self, state: int):
        """Handle autofocus checkbox change."""
        enabled = state == Qt.Checked
        self.logger.info("Vypnutí autofocusu: {enabled}")
        # Předání nastavení rodičovskému oknu
        if hasattr(self.parent(), "camera_thread") and self.parent().camera_thread:
            # Zastavit a znovu spustit kameru s novým nastavením autofocusu
            if hasattr(self.parent().camera_thread, "cap") and self.parent().camera_thread.cap:
                try:
                    # Nastavení autofocusu přímo v kameře
                    # pylint: disable=no-member
                    self.parent().camera_thread.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if enabled else 1)
                except Exception as e:
                    self.logger.error("Chyba při nastavení autofocusu: {e}")

            # Nastavení konfigurace pro budoucí použití
            if hasattr(self.parent().camera_thread, "config"):
                self.parent().camera_thread.config.disable_autofocus = enabled

    @pyqtSlot(int)
    def handle_show_detections_changed(self, state: int):
        """Handle show detections checkbox change."""
        enabled = state == Qt.Checked
        self.logger.info("Zobrazení detekcí: {enabled}")
        # Předání nastavení rodičovskému oknu
        if hasattr(self.parent(), "camera_thread") and self.parent().camera_thread:
            if hasattr(self.parent().camera_thread, "detection_thread") and self.parent().camera_thread.detection_thread:
                # Nastavení zobrazení detekcí v detection_thread
                detection_thread = self.parent().camera_thread.detection_thread
                if hasattr(detection_thread, "show_detections"):
                    detection_thread.show_detections = enabled
                elif hasattr(detection_thread, "config"):
                    detection_thread.config.show_detections = enabled

    @pyqtSlot(int)
    def handle_show_grid_changed(self, state: int):
        """Handle show grid checkbox change."""
        enabled = state == Qt.Checked
        self.logger.info("Zobrazení mřížky: {enabled}")
        # Předání nastavení rodičovskému oknu
        if hasattr(self.parent(), "camera_thread") and self.parent().camera_thread:
            if hasattr(self.parent().camera_thread, "detection_thread") and self.parent().camera_thread.detection_thread:
                # Nastavení zobrazení mřížky v detection_thread
                detection_thread = self.parent().camera_thread.detection_thread
                if hasattr(detection_thread, "show_grid"):
                    detection_thread.show_grid = enabled
                elif hasattr(detection_thread, "config"):
                    detection_thread.config.show_grid = enabled

    @pyqtSlot(np.ndarray)
    def update_camera_view(self, frame: np.ndarray):
        """Update the camera view with a new frame."""
        self.camera_view.update_frame(frame)

    def update_status(self, status: str):
        """Update the status label."""
        self.status_label.setText(status)

    def update_fps(self, fps: float):
        """Update the FPS label."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_board_state(self, board_state: List[List[str]]):
        """Update the board state label."""
        if board_state:
            # Convert board state to string representation with proper formatting
            board_str = ""
            for row in board_state:
                row_str = " | ".join([cell if cell else " " for cell in row])
                board_str += row_str + "\n"
            self.board_state_label.setText(f"Stav hry:\n{board_str}")
            self.board_state_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
        else:
            self.board_state_label.setText("Stav hry: Nedetekováno")
            self.board_state_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")

    def update_camera_status(self, status_message):
        """Update camera status in the view."""
        self.camera_view.set_camera_status(status_message)

    def set_camera_active(self, active):
        """Set camera active state."""
        self.camera_view.set_camera_active(active)

    def closeEvent(self, event):
        """Handle window close event."""
        self.logger.info("Zavírání debug okna")
        event.accept()
