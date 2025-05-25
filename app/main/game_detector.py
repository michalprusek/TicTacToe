"""
Refactored game detector module for the TicTacToe application.
"""
import logging
import time
from typing import Tuple, Optional, List, Dict

import numpy as np
import cv2
import torch
from app.main.game_utils import setup_logger

from app.core.detector_constants import (
    DEFAULT_DETECT_MODEL_PATH,
    DEFAULT_POSE_MODEL_PATH
)
from app.core.config import GameDetectorConfig
from app.core.game_state import GameState
from app.core.utils import FPSCalculator

from app.main.grid_detector import GridDetector
from app.main.symbol_detector import SymbolDetector
from app.main.visualization_manager import VisualizationManager
from app.main.game_state_manager import GameStateManager


class GameDetector:
    """Detects Tic Tac Toe grid and symbols using YOLO models."""

    def __init__(
        self,
        config: GameDetectorConfig,
        camera_index: int = 0,
        detect_model_path: str = DEFAULT_DETECT_MODEL_PATH,
        pose_model_path: str = DEFAULT_POSE_MODEL_PATH,
        disable_autofocus: bool = True,
        device: Optional[str] = None,
        log_level=logging.INFO
    ):
        """Initializes the detector, loads models, and sets up the camera."""
        self.config = config
        self.logger = setup_logger(__name__)
        self.logger.setLevel(log_level)
        self.camera_index = camera_index
        self.detect_model_path = detect_model_path
        self.pose_model_path = pose_model_path
        self.disable_autofocus = disable_autofocus

        # Set device for models
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
            self.logger.info("Using CUDA for inference")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            self.logger.info("Using MPS (Apple Silicon) for inference")
        else:
            self.device = 'cpu'
            self.logger.info("Using CPU for inference")

        # Load models
        self._load_models()

        # Initialize camera directly (avoiding circular import)
        self.camera_index = camera_index
        self.cap = None

        self.grid_detector = GridDetector(
            pose_model=self.pose_model,
            config=config,
            logger=self.logger
        )

        self.symbol_detector = SymbolDetector(
            detect_model=self.detect_model,
            config=config,
            logger=self.logger
        )

        self.visualization_manager = VisualizationManager(
            config=config,
            logger=self.logger
        )

        self.game_state_manager = GameStateManager(
            config=config,
            logger=self.logger
        )

        # For FPS calculation
        self.fps_calculator = FPSCalculator(buffer_size=10)
        self.last_log_time = time.time()
        self.log_interval = 5  # Log performance every 5 seconds

    def _load_models(self):
        """Loads the YOLO models for detection and pose estimation."""
        try:
            # Load detection model for X and O symbols
            self.logger.info("Loading detection model from %s", self.detect_model_path)
            try:
                from ultralytics import YOLO
                self.detect_model = YOLO(self.detect_model_path)
                self.detect_model.to(self.device)
            except ImportError:
                # Fallback to torch.hub if ultralytics not available
                self.detect_model = torch.hub.load(
                    'ultralytics/yolov5', 'custom',
                    path=self.detect_model_path,
                    device=self.device,
                    force_reload=True
                )

            # Load pose model for grid detection
            self.logger.info("Loading pose model from %s", self.pose_model_path)
            try:
                from ultralytics import YOLO
                self.pose_model = YOLO(self.pose_model_path)
                self.pose_model.to(self.device)
            except ImportError:
                # Fallback to torch.hub if ultralytics not available
                self.pose_model = torch.hub.load(
                    'ultralytics/yolov5', 'custom',
                    path=self.pose_model_path,
                    device=self.device,
                    force_reload=True
                )

            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error("Error loading models: %s", e)
            raise

    def process_frame(self, frame: np.ndarray, frame_time: float) -> Tuple[np.ndarray, Optional[GameState]]:
        """Processes a single frame: detects grid, symbols, updates state."""
        # Start FPS counter for processing time
        start_time_processing = time.perf_counter()

        # --- 1. Detect Symbols (X/O) --- #
        _, detected_symbols = self.symbol_detector.detect_symbols(frame.copy())

        # --- 2. Detect Grid --- #
        _, raw_kpts = self.grid_detector.detect_grid(frame.copy())

        # --- 3. Sort and Validate Grid Keypoints --- #
        sorted_kpts = self.grid_detector.sort_grid_points(raw_kpts)
        grid_is_valid = self.grid_detector.is_valid_grid(sorted_kpts)

        # Compute homography if grid is valid
        current_H = None
        if grid_is_valid:
            current_H = self.grid_detector.compute_homography(sorted_kpts)

        # Update grid status and check if it changed significantly
        current_time = time.time()
        grid_status_changed = self.grid_detector.update_grid_status(
            grid_is_valid, current_time
        )

        # Determine final keypoints for processing
        final_kpts = sorted_kpts if grid_is_valid else None

        # --- 4. Update Game State --- #
        cell_polygons = self.game_state_manager.update_game_state(
            frame.copy(),
            final_kpts,
            current_H,
            detected_symbols,
            frame_time,
            grid_status_changed
        )

        # --- 5. Calculate FPS --- #
        processing_time = time.perf_counter() - start_time_processing
        self.fps_calculator.tick()
        fps = self.fps_calculator.get_fps()

        # Log performance periodically
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.logger.info("Processing FPS: %.2", fps)
            self.last_log_time = current_time

        # --- 6. Draw Detection Results --- #
        annotated_frame = self.visualization_manager.draw_detection_results(
            frame.copy(),
            fps,
            raw_kpts,
            final_kpts,
            cell_polygons,
            detected_symbols,
            current_H,
            self.game_state_manager.game_state
        )

        # Draw debug info in a separate window if enabled
        self.visualization_manager.draw_debug_info(
            frame.copy(),
            fps,
            self.game_state_manager.game_state
        )

        return annotated_frame, self.game_state_manager.game_state

    def run_detection(self):
        """Runs the detection loop on the camera feed."""
        self.logger.info("Starting detection loop")

        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")

        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()

                if not ret:
                    self.logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                # Process the frame
                frame_time = time.time()
                try:
                    processed_frame, current_game_state = self.process_frame(
                        frame, frame_time
                    )

                    # Display the processed frame
                    cv2.imshow('Tic Tac Toe Detection', processed_frame)

                    # Print board state if valid
                    if current_game_state and current_game_state.is_valid():
                        # Logging happens within game_state.update_from_detection
                        pass
                    else:
                        self.logger.debug("Waiting for valid grid detection...")

                except Exception as e:
                    self.logger.exception("Error during frame processing: %s", e)

                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' key
                    self.logger.info("User requested exit")
                    break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, exiting...")
        finally:
            self.release()

    def release(self):
        """Releases resources."""
        self.logger.info("Releasing resources")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# --- Integrated Test Block --- #
if __name__ == '__main__':
    # Configure logging
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    detector = None
    try:
        # Create detector with configuration
        config = GameDetectorConfig()
        detector = GameDetector(
            config=config,
            camera_index=0,
            log_level=log_level
        )

        # Run detection loop
        detector.run_detection()
    except Exception as e:
        logger.error("Error: %s", e)
    finally:
        if detector:
            detector.release()
        logger.info("Application exited.")
