"""
Detection thread module for TicTacToe application.
"""
import time
import logging
import threading
import json
from typing import Optional, Tuple, Dict

import numpy as np
import torch

from app.core.game_state import GameState
from app.core.config import GameDetectorConfig
from app.main.game_detector import GameDetector


class DetectionThread(threading.Thread):
    """Thread for running YOLO detection at a fixed frame rate."""

    def __init__(self,
                 config: GameDetectorConfig,
                 target_fps: float = 2.0,
                 device: Optional[str] = None):
        """Initialize the detection thread.

        Args:
            config: Configuration for the game detector
            target_fps: Target frames per second for detection (default: 2.0)
            device: Device to use for inference (default: auto-detect)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # Auto-detect best available device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.logger.info(f"Detection thread using device: {self.device}")

        # Thread control
        self.running = False
        self.daemon = True  # Thread will exit when main program exits

        # Shared resources
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.result_lock = threading.Lock()
        self.latest_result = None
        self.latest_game_state = None

        # Performance metrics
        self.fps_history = []
        self.avg_fps = 0.0
        self.last_inference_time = 0.0

        # Nastavení zobrazení
        self.show_detections = True  # Zobrazovat detekce symbolů
        self.show_grid = True  # Zobrazovat mřížku hrací plochy
        self.confidence_threshold = 0.45  # Výchozí práh jistoty pro detekce

        # Initialize detector
        self.detector = None
        
        # Load calibration data
        self.calibration_data = self._load_calibration_data()

    def set_frame(self, frame: np.ndarray) -> None:
        """Set the latest frame for processing.

        Args:
            frame: The frame to process
        """
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def get_latest_result(self) -> Tuple[Optional[np.ndarray], Optional[GameState]]:
        """Get the latest detection result.

        Returns:
            Tuple of (processed_frame, game_state)
        """
        with self.result_lock:
            if self.latest_result is not None and self.latest_game_state is not None:
                return self.latest_result.copy(), self.latest_game_state
            return None, None

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the detection thread.

        Returns:
            Dictionary with performance metrics
        """
        return {
            'avg_fps': self.avg_fps,
            'last_inference_time': self.last_inference_time
        }

    def run(self):
        """Main thread loop."""
        self.running = True

        # Initialize detector
        try:
            self.detector = GameDetector(
                config=self.config,
                camera_index=self.config.camera_index,
                device=self.device
            )
            self.logger.info("Detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            self.running = False
            return

        # Main detection loop
        while self.running:
            loop_start_time = time.time()

            # Get the latest frame
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.01)  # Short sleep to avoid busy waiting
                    continue
                frame = self.latest_frame.copy()

            # Process the frame
            try:
                inference_start = time.time()

                # Předáme nastavení detekce do detektoru
                # Správná cesta je přes konfiguraci detektoru (self.detector.config)
                if hasattr(self.detector.config, "show_detections"):
                    self.detector.config.show_detections = self.show_detections
                # Fallback, pokud by show_detections byl přímo na GameDetector instanci
                elif hasattr(self.detector, "show_detections"):
                    self.detector.show_detections = self.show_detections

                if hasattr(self.detector.config, "show_grid"):
                    self.detector.config.show_grid = self.show_grid
                elif hasattr(self.detector, "show_grid"):
                    self.detector.show_grid = self.show_grid

                if hasattr(self.detector.config, "bbox_conf_threshold"):
                    self.detector.config.bbox_conf_threshold = self.confidence_threshold
                if hasattr(self.detector.config, "pose_conf_threshold"):
                    self.detector.config.pose_conf_threshold = self.confidence_threshold
                
                # Pokud GameDetector má obecný confidence_threshold, nastavíme ho také,
                # ale prioritu má specifičtější nastavení v configu.
                # Tento blok by měl být až po nastavení specifičtějších prahů.
                if (not hasattr(self.detector.config, "bbox_conf_threshold") and
                    not hasattr(self.detector.config, "pose_conf_threshold") and
                    hasattr(self.detector, "confidence_threshold")):
                    self.detector.confidence_threshold = self.confidence_threshold

                processed_frame, game_state = self.detector.process_frame(frame, inference_start)
                inference_time = time.time() - inference_start
                self.last_inference_time = inference_time

                # Update the latest result - vždy použít zpracovaný snímek s detekcemi
                with self.result_lock:
                    self.latest_result = processed_frame
                    self.latest_game_state = game_state

                # Update FPS metrics
                self.fps_history.append(1.0 / inference_time if inference_time > 0 else 0)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
                self.avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

                # Log performance periodically
                if len(self.fps_history) % 10 == 0:
                    self.logger.debug(
                        f"Detection performance: {self.avg_fps:.2f} FPS, "
                        f"inference time: {inference_time*1000:.1f}ms"
                    )

            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")

            # Calculate sleep time to maintain target FPS
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """Stop the detection thread."""
        self.running = False
        if self.detector:
            self.detector.release()

        # Bezpečné ukončení - kontrola, zda vlákno běží před zavoláním join
        if self.is_alive():
            self.join(timeout=1.0)  # Wait for thread to finish
            self.logger.info("Detection thread stopped")
    
    def _load_calibration_data(self) -> Optional[Dict]:
        """Load calibration data from JSON file."""
        try:
            calibration_file = self.config.calibration_file
            if not calibration_file.startswith('/'):
                # Relative path, make it absolute based on project root
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                calibration_file = os.path.join(project_root, "app", "calibration", calibration_file)
            
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                self.logger.info(f"Loaded calibration data from {calibration_file}")
                return data
        except Exception as e:
            self.logger.warning(f"Could not load calibration data: {e}")
            return None
        else:
            self.logger.info("Detection thread not running")
