"""
Camera Controller module for TicTacToe application.
This module handles camera integration, detection processing, and frame management.
Consolidates functionality from camera_manager.py.
"""

import logging
from PyQt5.QtCore import QObject, pyqtSignal

import cv2
import numpy as np

# Import required modules
from app.main.path_utils import setup_project_path
setup_project_path()

from app.main.camera_thread import CameraThread
from app.main.constants import DEFAULT_CAMERA_INDEX
from app.main.game_utils import setup_logger
from app.main.error_handler import ErrorHandler


class CameraController(QObject):
    """Controls camera integration and detection processing."""

    # Signals
    frame_ready = pyqtSignal(object)  # frame
    game_state_updated = pyqtSignal(object)  # detected_board
    fps_updated = pyqtSignal(float)  # fps
    grid_warning = pyqtSignal(str)  # warning_message
    grid_incomplete = pyqtSignal(bool)  # True=incomplete, False=complete

    def __init__(self, main_window, camera_index=0):
        super().__init__()

        self.main_window = main_window
        self.camera_index = camera_index
        self.logger = logging.getLogger(__name__)

        # Camera thread
        self.camera_thread = None

        # Grid warning state
        self.grid_warning_active = False

        # Initialize camera
        self._init_camera()

    def _init_camera(self):
        """Initialize camera thread."""
        self.logger.info("Creating camera thread with index {self.camera_index}.")

        self.camera_thread = CameraThread(camera_index=self.camera_index)

        # Connect signals
        self.camera_thread.frame_ready.connect(self._handle_frame_ready)
        self.camera_thread.game_state_updated.connect(self._handle_game_state_updated)
        self.camera_thread.fps_updated.connect(self._handle_fps_updated)

        self.logger.info("Camera thread for index {self.camera_index} created.")

    def start(self):
        """Start the camera thread."""
        if self.camera_thread:
            # Update debug window status
            if hasattr(self.main_window, 'debug_window') and self.main_window.debug_window:
                self.main_window.debug_window.update_camera_status("📷 Spouštění kamery...")

            self.camera_thread.start()
            self.logger.info("Camera thread started.")

            # Update debug window status after start
            if hasattr(self.main_window, 'debug_window') and self.main_window.debug_window:
                self.main_window.debug_window.update_camera_status("📷 Kamera aktivní")
                self.main_window.debug_window.set_camera_active(True)

    def stop(self):
        """Stop the camera thread."""
        if self.camera_thread:
            # Update debug window status
            if hasattr(self.main_window, 'debug_window') and self.main_window.debug_window:
                self.main_window.debug_window.update_camera_status("📷 Zastavování kamery...")
                self.main_window.debug_window.set_camera_active(False)

            self.camera_thread.stop()
            self.camera_thread.wait()
            self.logger.info("Camera thread stopped.")

            # Update debug window status after stop
            if hasattr(self.main_window, 'debug_window') and self.main_window.debug_window:
                self.main_window.debug_window.update_camera_status("📷 Kamera zastavena")

    def restart_camera(self, new_camera_index):
        """Restart camera with new index."""
        self.logger.info("Restarting camera with index {new_camera_index}.")

        # Stop current camera
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()

        # Update index and reinitialize
        self.camera_index = new_camera_index
        self._init_camera()

        # Start new camera
        self.start()

        self.logger.info("Camera restarted with index {new_camera_index}.")

    def _handle_frame_ready(self, frame):
        """Handle frame ready signal from camera thread."""
        if frame is None:
            return

        # Get detection data
        processed_frame, game_state_from_detection = self._get_detection_data()

        # Update main camera view (if needed)
        self._update_main_camera_view(frame)

        # Handle grid warnings
        if game_state_from_detection:
            self._handle_grid_warnings(game_state_from_detection)

        # Update debug window
        self._update_debug_window(
            processed_frame if processed_frame is not None else frame,
            frame,
            game_state_from_detection
        )

        # Emit frame ready signal
        self.frame_ready.emit(frame)

    def _handle_game_state_updated(self, detected_board):
        """Handle game state updated signal from camera thread."""
        # Emit to game controller
        self.game_state_updated.emit(detected_board)

    def _handle_fps_updated(self, fps):
        """Handle FPS updated signal from camera thread."""
        # Emit to interested components
        self.fps_updated.emit(fps)

    def _get_detection_data(self):
        """Get detection data from camera thread."""
        processed_frame = None
        game_state = None

        if self.camera_thread and hasattr(self.camera_thread, 'detection_thread'):
            if hasattr(self.camera_thread.detection_thread, 'latest_processed_frame'):
                processed_frame = self.camera_thread.detection_thread.latest_processed_frame
            if hasattr(self.camera_thread.detection_thread, 'latest_game_state'):
                game_state = self.camera_thread.detection_thread.latest_game_state

        return processed_frame, game_state

    def _update_main_camera_view(self, frame):
        """Update main camera view (placeholder for future implementation)."""
        # In this application, main window doesn't have direct CameraView
        # Camera view is in DebugWindow
        pass

    def _handle_grid_warnings(self, game_state_obj):
        """Handle grid warnings from detection."""
        if not game_state_obj:
            return

        # Check if grid is valid
        grid_valid = False
        if hasattr(game_state_obj, 'is_physical_grid_valid') and callable(game_state_obj.is_physical_grid_valid):
            grid_valid = game_state_obj.is_physical_grid_valid()

        # Check if game is paused due to incomplete grid
        grid_incomplete = False
        if hasattr(game_state_obj, 'is_game_paused_due_to_incomplete_grid'):
            grid_incomplete = game_state_obj.is_game_paused_due_to_incomplete_grid

        # Check if grid is actually incomplete (less than 16 points)
        grid_points_count = 0
        if hasattr(game_state_obj, '_grid_points') and game_state_obj._grid_points is not None:
            grid_points_count = len(game_state_obj._grid_points)

        is_grid_incomplete = grid_points_count < 16

        if not grid_valid or grid_incomplete or is_grid_incomplete:
            # Grid is not valid or incomplete - show warning
            if not self.grid_warning_active:
                self.grid_warning_active = True

                # Get warning message
                grid_issue_message = "Grid not visible!"
                if hasattr(game_state_obj, 'grid_issue_message'):
                    grid_issue_message = game_state_obj.grid_issue_message
                elif is_grid_incomplete:
                    grid_issue_message = f"Grid not completely visible! Detected {grid_points_count}/16 points."

                # Only emit warning if grid is actually incomplete
                if is_grid_incomplete:
                    self.grid_warning.emit(grid_issue_message)
                    self.logger.warning("Grid warning: {grid_issue_message}")

            # Emit grid incomplete signal for UI notification only if actually incomplete
            if is_grid_incomplete:
                self.grid_incomplete.emit(True)
        else:
            # Grid is valid - hide warning
            if self.grid_warning_active:
                self.grid_warning_active = False
                self.grid_warning.emit("")  # Empty message to hide warning
                self.logger.info("Grid is now valid - hiding warning.")

            # Emit grid complete signal
            self.grid_incomplete.emit(False)

    def _update_debug_window(self, processed_frame, raw_frame, game_state_obj):
        """Update debug window with camera data."""
        # Check if debug window exists and update it
        if hasattr(self.main_window, 'debug_window') and self.main_window.debug_window:
            self._update_debug_window_safe(processed_frame, raw_frame, game_state_obj)

    def _update_debug_window_safe(self, processed_frame, raw_frame, game_state_obj):
        """Safely update debug window with error handling."""
        try:
            # Update camera view in debug window
            if hasattr(self.main_window.debug_window, 'camera_view'):
                frame_to_show = processed_frame if processed_frame is not None else raw_frame
                self.main_window.debug_window.camera_view.update_frame(frame_to_show)

            # Update detection info
            if hasattr(self.main_window.debug_window, 'update_detection_info'):
                self.main_window.debug_window.update_detection_info(game_state_obj)
        except Exception as e:
            ErrorHandler.log_error(self.logger, "debug window update", e, False)

    def get_current_board_state(self):
        """Get current board state from camera."""
        if self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
            return self.camera_thread.last_board_state
        return None

    def get_calibration_data(self):
        """Get calibration data from camera thread."""
        if self.camera_thread and hasattr(self.camera_thread, 'detection_thread'):
            if hasattr(self.camera_thread.detection_thread, 'calibration_data'):
                return self.camera_thread.detection_thread.calibration_data
        return None

    def set_detection_threshold(self, threshold):
        """Set detection threshold."""
        if self.camera_thread and hasattr(self.camera_thread, 'detection_thread'):
            if hasattr(self.camera_thread.detection_thread, 'set_detection_threshold'):
                self.camera_thread.detection_thread.set_detection_threshold(threshold)
                self.logger.info("Detection threshold set to {threshold}")

    def get_detection_threshold(self):
        """Get current detection threshold."""
        if self.camera_thread and hasattr(self.camera_thread, 'detection_thread'):
            if hasattr(self.camera_thread.detection_thread, 'get_detection_threshold'):
                return self.camera_thread.detection_thread.get_detection_threshold()
        return 0.8  # Default threshold

    def calibrate_camera(self):
        """Trigger camera calibration."""
        if self.camera_thread and hasattr(self.camera_thread, 'detection_thread'):
            if hasattr(self.camera_thread.detection_thread, 'calibrate'):
                self.camera_thread.detection_thread.calibrate()
                self.logger.info("Camera calibration triggered")

    def is_camera_active(self):
        """Check if camera is active."""
        return self.camera_thread is not None and self.camera_thread.isRunning()

    def get_camera_info(self):
        """Get camera information."""
        info = {
            'camera_index': self.camera_index,
            'is_active': self.is_camera_active(),
            'grid_warning_active': self.grid_warning_active
        }

        # Add detection info if available
        _, game_state = self._get_detection_data()
        if game_state:
            info['grid_valid'] = False
            if hasattr(game_state, 'is_physical_grid_valid') and callable(game_state.is_physical_grid_valid):
                info['grid_valid'] = game_state.is_physical_grid_valid()

        return info

    # === Consolidated camera management functions from camera_manager.py ===

    def setup_camera_direct(self, camera_index=None, frame_width=640, frame_height=480, disable_autofocus=True):
        """Direct camera setup without thread (consolidated from camera_manager.py)."""
        if camera_index is not None:
            self.camera_index = camera_index

        try:
            cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                self.logger.error("Failed to open camera %s", self.camera_index)
                return False

            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

            # Disable autofocus if requested
            if disable_autofocus:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                autofocus_status = cap.get(cv2.CAP_PROP_AUTOFOCUS)
                self.logger.info("Autofocus status: %s", autofocus_status)

            # Log camera properties
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.logger.info("Camera initialized with resolution: %sx%s",
                            actual_width, actual_height)

            cap.release()  # Release for thread to use
            return True
        except Exception as e:
            self.logger.error("Error setting up camera: %s", e)
            return False

    def read_frame_direct(self, cap):
        """Read frame directly from OpenCV capture (consolidated from camera_manager.py)."""
        if not cap or not cap.isOpened():
            self.logger.error("Camera is not initialized")
            return False, None

        ret, frame = cap.read()

        if not ret:
            self.logger.warning("Failed to read frame from camera")
            return False, None

        return True, frame

    def get_camera_properties_direct(self, cap):
        """Get camera properties directly (consolidated from camera_manager.py)."""
        if not cap or not cap.isOpened():
            return {}

        properties = {
            'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'autofocus': cap.get(cv2.CAP_PROP_AUTOFOCUS),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': cap.get(cv2.CAP_PROP_SATURATION),
            'hue': cap.get(cv2.CAP_PROP_HUE),
            'gain': cap.get(cv2.CAP_PROP_GAIN),
            'exposure': cap.get(cv2.CAP_PROP_EXPOSURE)
        }

        return properties
