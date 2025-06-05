# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Detection thread module for TicTacToe application.
"""
import json
import logging
import threading
import time
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from app.core.config import GameDetectorConfig
from app.core.game_state import GameState
from app.main.game_detector import GameDetector


# pylint: disable=too-many-instance-attributes
class DetectionThread(threading.Thread):
    """Thread for running YOLO detection at a fixed frame rate."""

    def __init__(self,  # pylint: disable=too-many-arguments
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
        self.logger.info("Detection thread using device: %s", self.device)

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
        self.confidence_threshold = 0.8  # Výchozí práh jistoty pro detekce

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

    def get_latest_result(
            self) -> Tuple[Optional[np.ndarray], Optional[GameState]]:
        """Get the latest detection result.

        Returns:
            Tuple of (processed_frame, game_state)
        """
        with self.result_lock:
            if (self.latest_result is not None and
                    self.latest_game_state is not None):
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

        if not self._initialize_detector():
            return

        # Main detection loop
        while self.running:
            self._process_detection_loop()

    def _initialize_detector(self) -> bool:
        """Initialize the detector. Returns True if successful."""
        try:
            self.detector = GameDetector(
                detector_config=self.config,
                camera_index=self.config.camera_index,
                device=self.device
            )
            self.logger.info("Detector initialized successfully")
            return True
        except (RuntimeError, ImportError, ValueError) as e:
            self.logger.error("Failed to initialize detector: %s", e)
            self.running = False
            return False

    def _process_detection_loop(self):
        """Process one iteration of the detection loop."""
        loop_start_time = time.time()
        # Processing detection loop iteration

        # Get the latest frame
        frame = self._get_latest_frame()
        if frame is None:
            # No frame available, waiting...
            time.sleep(0.01)  # Short sleep to avoid busy waiting
            return

        # Process the frame
        try:
            self._process_frame(frame)
        except (RuntimeError, TypeError, ValueError) as e:
            self.logger.error("Error processing frame: %s", e)

        # Minimal sleep to prevent CPU overload
        time.sleep(0.001)  # 1ms sleep

    def _get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame for processing."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _process_frame(self, frame: np.ndarray):
        """Process a single frame."""
        inference_start = time.time()

        # Configure detector settings
        self._configure_detector_settings()

        # Run detection
        processed_frame, game_state = self.detector.process_frame(
            frame, inference_start)
        inference_time = time.time() - inference_start
        self.last_inference_time = inference_time

        # Update results
        self._update_results(processed_frame, game_state)

        # Update performance metrics
        self._update_performance_metrics(inference_time)

    def _configure_detector_settings(self):
        """Configure detector visualization and confidence settings."""
        # Configure show_detections
        if hasattr(self.detector.config, "show_detections"):
            self.detector.config.show_detections = self.show_detections
        elif hasattr(self.detector, "show_detections"):
            self.detector.show_detections = self.show_detections

        # Configure show_grid
        if hasattr(self.detector.config, "show_grid"):
            self.detector.config.show_grid = self.show_grid
        elif hasattr(self.detector, "show_grid"):
            self.detector.show_grid = self.show_grid

        # Configure confidence thresholds
        if hasattr(self.detector.config, "bbox_conf_threshold"):
            self.detector.config.bbox_conf_threshold = (
                self.confidence_threshold)
        if hasattr(self.detector.config, "pose_conf_threshold"):
            self.detector.config.pose_conf_threshold = (
                self.confidence_threshold)

        # Fallback for general confidence threshold
        if (not hasattr(self.detector.config, "bbox_conf_threshold") and
                not hasattr(self.detector.config, "pose_conf_threshold") and
                hasattr(self.detector, "confidence_threshold")):
            self.detector.confidence_threshold = self.confidence_threshold

    def _update_results(
            self, processed_frame: np.ndarray, game_state: GameState):
        """Update the latest detection results."""
        with self.result_lock:
            self.latest_result = processed_frame
            self.latest_game_state = game_state

    def _update_performance_metrics(self, inference_time: float):
        """Update performance metrics."""
        # Update FPS metrics
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)

        if self.fps_history:
            self.avg_fps = sum(self.fps_history) / len(self.fps_history)
        else:
            self.avg_fps = 0

        # Log performance periodically
        if len(self.fps_history) % 10 == 0:
            self.logger.info(
                "Detection performance: %.2f FPS, inference time: %.1fms",
                self.avg_fps, inference_time * 1000)

    def stop(self):
        """Stop the detection thread."""
        self.running = False
        if self.detector:
            self.detector.release()

        # Bezpečné ukončení - kontrola, zda vlákno běží pred zavoláním join
        if self.is_alive():
            self.join(timeout=1.0)  # Wait for thread to finish
        self.logger.info("Detection thread stopped")

    def _load_calibration_data(self) -> Optional[Dict]:
        """Load calibration data from JSON file."""
        try:
            calibration_file = self.config.calibration_file
            if not calibration_file.startswith('/'):
                # Relative path, make it absolute based on project root
                import os  # pylint: disable=import-outside-toplevel
                project_root = os.path.dirname(os.path.dirname(
                    os.path.dirname(__file__)))
                calibration_file = os.path.join(
                    project_root, "app", "calibration", calibration_file)

            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(
                    "Loaded calibration data from %s", calibration_file)
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            self.logger.warning("Could not load calibration data: %s", e)
            return None
