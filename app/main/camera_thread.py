# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Camera thread module for the TicTacToe application.
"""
import logging

# pylint: disable=no-name-in-module,too-many-instance-attributes,import-outside-toplevel,no-member,unused-import
import time

import numpy as np
import torch
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal

from app.core.config import GameDetectorConfig
from app.core.detection_thread import DetectionThread
from app.core.game_state import GameState
from app.main.constants import DEFAULT_CAMERA_INDEX


class CameraThread(QThread):
    """Thread for capturing camera frames and processing game state"""
    frame_ready = pyqtSignal(np.ndarray)
    game_state_updated = pyqtSignal(list)
    fps_updated = pyqtSignal(float)

    def __init__(self, camera_index=DEFAULT_CAMERA_INDEX, target_fps=2.0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.config = GameDetectorConfig()
        self.config.camera_index = camera_index
        self.config.target_fps = target_fps

        # Automatická detekce nejvhodnějšího zařízení
        if torch.cuda.is_available():
            self.config.device = 'cuda'
            print("Používám CUDA (GPU) pro detekci")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            self.config.device = 'mps'
            print("Používám MPS (Apple Silicon) pro detekci")
        else:
            self.config.device = 'cpu'
            print("Používám CPU pro detekci")

        # Inicializace detekčního vlákna
        self.detection_thread = DetectionThread(self.config)
        self.detection_thread.start()

        # Proměnné pro sledování stavu hry
        self.last_board_state = None
        self.last_board_update_time = 0
        self.board_state_stable_time = 0.5  # Čas v sekundách, po který musí být stav stabilní
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Main thread loop for capturing frames and processing game state."""
        self.running = True
        import cv2
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")

        # Nastavení rozlišení kamery (volitelné)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Pokus o vypnutí autofokusu (nemusí fungovat na všech kamerách)
        if self.config.disable_autofocus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.logger.info("Autofocus status: %s", self.cap.get(cv2.CAP_PROP_AUTOFOCUS))

        # Hlavní smyčka pro čtení snímků z kamery
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame from camera {self.camera_index}")

            # Předání snímku detekčnímu vláknu
            self.detection_thread.set_frame(frame)

            # Získání výsledků detekce
            processed_frame, game_state = self.detection_thread.get_latest_result()

            # Pokud máme zpracovaný snímek, odešleme ho do GUI
            if processed_frame is not None:
                self.frame_ready.emit(processed_frame)

                # Odeslání aktuální FPS
                metrics = self.detection_thread.get_performance_metrics()
                self.fps_updated.emit(metrics['avg_fps'])

            # Pokud máme platný stav hry, aktualizujeme
            if game_state and game_state.is_valid():
                current_time = time.time()
                current_board = game_state.board

                # Kontrola, zda se stav hry změnil
                if self._has_board_changed(current_board):
                    # Resetujeme čas stability
                    self.last_board_state = current_board
                    self.last_board_update_time = current_time
                elif (current_time - self.last_board_update_time) >= self.board_state_stable_time:
                    # Stav je stabilní po dostatečnou dobu, můžeme ho odeslat
                    # Převedeme 2D pole na 1D seznam pro signál
                    flat_board = [cell for row in current_board for cell in row]
                    self.game_state_updated.emit(flat_board)

            # Omezení FPS pro snížení zatížení CPU
            time.sleep(1.0 / self.config.target_fps)

    def stop(self):
        """Stops the camera thread and releases resources."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.stop()
            # DetectionThread is a threading.Thread, not QThread, so we use join() instead of wait()
            if self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _has_board_changed(self, current_board):
        """Checks if the board state has changed since the last update."""
        if self.last_board_state is None:
            return True

        for r in range(3):
            for c in range(3):
                if current_board[r][c] != self.last_board_state[r][c]:
                    return True
        return False
