"""
Arm Movement Controller module for TicTacToe application.
This module provides centralized robotic arm control and movement coordination.
Refactored from pyqt_gui.py to consolidate arm control logic.
"""

import logging
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# Import required modules
from app.main.path_utils import setup_project_path
setup_project_path()

from app.main.arm_controller import ArmController
from app.core.arm_thread import ArmThread
from app.main import game_logic
from app.main.constants import (
    DEFAULT_SAFE_Z, DEFAULT_DRAW_Z, DRAWING_SPEED, MAX_SPEED
)
from app.main.game_utils import setup_logger
PARK_X = 150
PARK_Y = 0
DEFAULT_SYMBOL_SIZE = 15.0


class ArmMovementController(QObject):
    """Centralized controller for robotic arm movements and coordination."""

    # Signals
    arm_connected = pyqtSignal(bool)  # connection_status
    arm_move_completed = pyqtSignal(bool)  # success
    arm_status_changed = pyqtSignal(str)  # status_message

    def __init__(self, main_window, config):
        super().__init__()

        self.main_window = main_window
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Arm components
        self.arm_thread = None
        self.arm_controller = None

        # Configuration
        self.safe_z = DEFAULT_SAFE_Z
        self.draw_z = DEFAULT_DRAW_Z
        self.symbol_size = DEFAULT_SYMBOL_SIZE

        # Initialize arm
        self._init_arm_components()

    def _init_arm_components(self):
        """Initialize arm thread and controller."""
        # Get arm port from config
        arm_port = None
        if hasattr(self.config, 'arm') and hasattr(self.config.arm, 'port'):
            arm_port = self.config.arm.port

        # Get Z heights from config
        if hasattr(self.config, 'arm'):
            if hasattr(self.config.arm, 'safe_z'):
                self.safe_z = self.config.arm.safe_z
            if hasattr(self.config.arm, 'draw_z'):
                self.draw_z = self.config.arm.draw_z

        try:
            # Initialize arm thread
            self.arm_thread = ArmThread(port=arm_port)
            self.arm_thread.start()

            # Connect to arm
            if self.arm_thread.connect():
                self.logger.info("Robotic arm successfully connected via ArmThread.")
                self.arm_connected.emit(True)
                self.move_to_neutral_position()
            else:
                self.logger.error("Failed to connect to robotic arm via ArmThread.")
                self.arm_connected.emit(False)

            # Initialize arm controller (legacy/backup)
            self.arm_controller = ArmController(
                port=arm_port,
                draw_z=self.draw_z,
                safe_z=self.safe_z,
                speed=MAX_SPEED
            )
            # CRITICAL FIX: Sync connection status from ArmThread to ArmController
            if self.arm_thread and self.arm_thread.connected:
                self.arm_controller.connected = True
                self.arm_controller.swift = self.arm_thread.arm_controller.swift if hasattr(self.arm_thread.arm_controller, 'swift') else None

        except Exception as e:
            self.logger.error(f"Error initializing arm components: {e}")
            self.arm_connected.emit(False)

    def is_arm_available(self):
        """Check if arm is available for use."""
        # CRITICAL FIX: Sync connection status before checking
        self._sync_connection_status()
        arm_thread_available = (
            hasattr(self, 'arm_thread') and
            self.arm_thread and
            self.arm_thread.connected
        )
        return arm_thread_available

    def _sync_connection_status(self):
        """Sync connection status between ArmThread and ArmController."""
        if self.arm_thread and self.arm_controller:
            if self.arm_thread.connected and not self.arm_controller.connected:
                self.arm_controller.connected = True
                # Sync the swift instance if available
                if hasattr(self.arm_thread, 'arm_controller') and hasattr(self.arm_thread.arm_controller, 'swift'):
                    self.arm_controller.swift = self.arm_thread.arm_controller.swift

    def move_to_neutral_position(self):
        """Move arm to neutral position."""
        if not self.is_arm_available():
            self.logger.warning("Cannot move to neutral: arm not available.")
            return False

        try:
            success = self._unified_arm_command(
                'park',
                x=PARK_X,
                y=PARK_Y,
                z=self.safe_z,
                wait=True
            )

            if success:
                self.logger.info("Arm moved to neutral position.")
                self.arm_status_changed.emit("move_success")
            else:
                self.logger.error("Failed to move arm to neutral position.")
                self.arm_status_changed.emit("move_failed")

            return success

        except Exception as e:
            self.logger.error(f"Error moving to neutral position: {e}")
            self.arm_status_changed.emit("move_failed")
            return False

    def draw_ai_symbol(self, row, col, symbol_to_draw):
        """Draw AI symbol at specified position using consolidated arm control."""
        self.logger.info(f"Request to draw {symbol_to_draw} at ({row},{col})")

        if not self.is_arm_available():
            raise RuntimeError(f"Cannot draw symbol {symbol_to_draw}: robotic arm is not available")

        # Get target coordinates
        target_x, target_y = self._get_cell_coordinates_from_yolo(row, col)
        if target_x is None or target_y is None:
            raise RuntimeError(f"Cannot get coordinates for drawing at ({row},{col})")

        self.logger.info(f"Drawing {symbol_to_draw} at coordinates ({target_x:.1f}, {target_y:.1f})")

        try:
            # Use low-level arm controller for direct drawing (consolidated approach)
            if self.arm_controller and self.arm_controller.connected:
                if symbol_to_draw == game_logic.PLAYER_O:
                    success = self.arm_controller.draw_o(
                        center_x=target_x,
                        center_y=target_y,
                        radius=self.symbol_size / 2,
                        speed=DRAWING_SPEED
                    )
                elif symbol_to_draw == game_logic.PLAYER_X:
                    success = self.arm_controller.draw_x(
                        center_x=target_x,
                        center_y=target_y,
                        size=self.symbol_size,
                        speed=DRAWING_SPEED
                    )
                else:
                    raise ValueError(f"Unknown symbol: {symbol_to_draw}")
            else:
                # Fallback to thread-based approach
                success = self._unified_arm_command(
                    'draw_o' if symbol_to_draw == game_logic.PLAYER_O else 'draw_x',
                    x=target_x,
                    y=target_y,
                    radius=self.symbol_size / 2 if symbol_to_draw == game_logic.PLAYER_O else None,
                    size=self.symbol_size if symbol_to_draw == game_logic.PLAYER_X else None,
                    speed=DRAWING_SPEED
                )

            if success:
                self.logger.info(f"Successfully drew {symbol_to_draw} at ({row},{col})")
                self.arm_move_completed.emit(True)
                return True
            else:
                self.logger.error(f"Failed to draw {symbol_to_draw}")
                self.arm_move_completed.emit(False)
                return False

        except Exception as e:
            self.logger.error(f"Error drawing symbol: {e}")
            self.arm_move_completed.emit(False)
            return False

    def draw_winning_line(self):
        """Draw winning line through winning symbols."""
        if not hasattr(self.main_window, 'board_widget') or not self.main_window.board_widget.winning_line:
            raise RuntimeError("Cannot draw winning line: missing or invalid data")

        if not self.is_arm_available():
            raise RuntimeError("Cannot draw winning line: robotic arm is not connected")

        winning_line = self.main_window.board_widget.winning_line
        if len(winning_line) != 3:
            raise RuntimeError("Invalid winning line data")

        # Get start and end coordinates
        start_row, start_col = winning_line[0]
        end_row, end_col = winning_line[2]

        start_x, start_y = self._get_cell_coordinates_from_yolo(start_row, start_col)
        end_x, end_y = self._get_cell_coordinates_from_yolo(end_row, end_col)

        self.logger.info(f"Drawing winning line from ({start_x:.1f}, {start_y:.1f}) to ({end_x:.1f}, {end_y:.1f})")

        try:
            # Movement sequence
            # 1. Move above start position at safe_z
            if not self._unified_arm_command('go_to_position', x=start_x, y=start_y, z=self.safe_z, speed=MAX_SPEED, wait=True):
                raise RuntimeError("Failed to move arm to start position")

            # 2. Lower to drawing position
            if not self._unified_arm_command('go_to_position', x=start_x, y=start_y, z=self.draw_z, speed=DRAWING_SPEED, wait=True):
                raise RuntimeError("Failed to lower arm to drawing position")

            # 3. Draw line to end position
            if not self._unified_arm_command('go_to_position', x=end_x, y=end_y, z=self.draw_z, speed=DRAWING_SPEED, wait=True):
                raise RuntimeError("Failed to draw winning line")

            # 4. Lift to safe position
            if not self._unified_arm_command('go_to_position', x=end_x, y=end_y, z=self.safe_z, speed=MAX_SPEED, wait=True):
                raise RuntimeError("Failed to lift arm after drawing")

            self.logger.info("Winning line successfully drawn.")
            self.arm_status_changed.emit("Winning line drawn!")

            # Move to neutral after a delay
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1000, self.move_to_neutral_position)

            return True

        except Exception as e:
            self.logger.error(f"Error drawing winning line: {e}")
            raise

    def _unified_arm_command(self, command, *args, **kwargs):
        """Unified arm command interface."""
        if not self.is_arm_available():
            raise RuntimeError(f"Arm command '{command}' failed: robotic arm is not available")

        # Use ArmThread API
        try:
            if command == 'draw_o':
                success = self.arm_thread.draw_o(
                    center_x=kwargs.get('x'),
                    center_y=kwargs.get('y'),
                    radius=kwargs.get('radius'),
                    speed=kwargs.get('speed', DRAWING_SPEED)
                )
            elif command == 'draw_x':
                success = self.arm_thread.draw_x(
                    center_x=kwargs.get('x'),
                    center_y=kwargs.get('y'),
                    size=kwargs.get('size'),
                    speed=kwargs.get('speed', DRAWING_SPEED)
                )
            elif command == 'go_to_position':
                success = self.arm_thread.go_to_position(
                    x=kwargs.get('x'),
                    y=kwargs.get('y'),
                    z=kwargs.get('z'),
                    speed=kwargs.get('speed', MAX_SPEED),
                    wait=kwargs.get('wait', True)
                )
            elif command == 'park':
                success = self.arm_thread.go_to_position(
                    x=kwargs.get('x', PARK_X),
                    y=kwargs.get('y', PARK_Y),
                    z=kwargs.get('z', self.safe_z),
                    speed=MAX_SPEED // 2,
                    wait=kwargs.get('wait', True)
                )
            else:
                raise ValueError(f"Unknown arm command: {command}")

            if not success:
                raise RuntimeError(f"Arm command '{command}' failed to execute")

            self.logger.info(f"Arm command '{command}' executed successfully")
            return success

        except Exception as e:
            self.logger.error(f"Error executing arm command '{command}': {e}")
            raise

    def _get_cell_coordinates_from_yolo(self, row, col):
        """Get cell coordinates from YOLO detection with improved interpolation."""
        self.logger.info(f"🔍 COORDINATE TRANSFORMATION DEBUG for cell ({row},{col}):")

        # Get current detection state
        if hasattr(self.main_window, 'camera_controller'):
            camera_controller = self.main_window.camera_controller
            _, game_state_obj = camera_controller._get_detection_data()

            if not game_state_obj:
                raise RuntimeError(f"Cannot get detection state for cell ({row},{col})")

            # Get UV coordinates of cell center
            uv_center = game_state_obj.get_cell_center_uv(row, col)
            if uv_center is None:
                raise RuntimeError(f"Cannot get UV center for cell ({row},{col}) from current detection")

            self.logger.info(f"  📍 Step 1 - Grid position: ({row},{col})")
            self.logger.info(f"  📍 Step 2 - UV center from camera: ({uv_center[0]:.1f}, {uv_center[1]:.1f})")

            # DEBUG: Log all cell centers for comparison
            self.logger.info(f"  🗺️ ALL CELL CENTERS for reference:")
            for debug_row in range(3):
                for debug_col in range(3):
                    debug_center = game_state_obj.get_cell_center_uv(debug_row, debug_col)
                    if debug_center is not None:
                        self.logger.info(f"    Cell ({debug_row},{debug_col}): UV=({debug_center[0]:.1f}, {debug_center[1]:.1f})")

            # Get calibration data
            calibration_data = camera_controller.get_calibration_data()
            if calibration_data and "perspective_transform_matrix_xy_to_uv" in calibration_data:
                # Use matrix transformation with resolution scaling correction
                xy_to_uv_matrix = calibration_data["perspective_transform_matrix_xy_to_uv"]

                if xy_to_uv_matrix:
                    try:
                        # RESOLUTION SCALING CORRECTION
                        # Determine calibration resolution from calibration points
                        cal_points = calibration_data.get("calibration_points_raw", [])
                        if cal_points:
                            # Find max UV coordinates in calibration data
                            max_u = max(point["target_uv"][0] for point in cal_points)
                            max_v = max(point["target_uv"][1] for point in cal_points)
                            # Estimate calibration resolution (add margin)
                            calibration_width = max_u + 200  # Add margin
                            calibration_height = max_v + 200
                        else:
                            # Fallback to common resolution
                            calibration_width = 1920
                            calibration_height = 1080

                        # Current runtime resolution (assume camera resolution)
                        runtime_width = 1920  # Current camera width
                        runtime_height = 1080  # Current camera height

                        # Calculate scaling factors
                        scale_x = runtime_width / calibration_width
                        scale_y = runtime_height / calibration_height

                        # Scale UV coordinates from runtime to calibration resolution
                        uv_scaled = [
                            uv_center[0] / scale_x,
                            uv_center[1] / scale_y
                        ]

                        self.logger.info(f"  🔧 Resolution scaling:")
                        self.logger.info(f"    Calibration resolution: {calibration_width}x{calibration_height}")
                        self.logger.info(f"    Runtime resolution: {runtime_width}x{runtime_height}")
                        self.logger.info(f"    Scale factors: ({scale_x:.4f}, {scale_y:.4f})")
                        self.logger.info(f"    Original UV: ({uv_center[0]:.1f}, {uv_center[1]:.1f})")
                        self.logger.info(f"    Scaled UV: ({uv_scaled[0]:.1f}, {uv_scaled[1]:.1f})")

                        # Inverse matrix for UV->XY transformation
                        xy_to_uv_matrix = np.array(xy_to_uv_matrix, dtype=np.float32)
                        uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)

                        # Homogeneous coordinates for transformation using SCALED coordinates
                        uv_point_homogeneous = np.array([uv_scaled[0], uv_scaled[1], 1.0], dtype=np.float32).reshape(3, 1)
                        xy_transformed_homogeneous = np.dot(uv_to_xy_matrix, uv_point_homogeneous)

                        if xy_transformed_homogeneous[2, 0] != 0:
                            arm_x = xy_transformed_homogeneous[0, 0] / xy_transformed_homogeneous[2, 0]
                            arm_y = xy_transformed_homogeneous[1, 0] / xy_transformed_homogeneous[2, 0]
                            self.logger.info(f"  📍 Step 3 - Matrix transformation with scaling: ({arm_x:.1f}, {arm_y:.1f})")
                        else:
                            raise RuntimeError("Division by zero in UV->XY transformation")

                    except Exception as e:
                        raise RuntimeError(f"Error in UV->XY transformation for ({row},{col}): {e}")
                else:
                    raise RuntimeError("Missing transformation matrix in calibration data")
            else:
                raise RuntimeError("No perspective transformation matrix available in calibration data")

            # DEBUG: Log final coordinates
            self.logger.info(f"  🔍 CALIBRATION COMPARISON:")
            self.logger.info(f"    Current UV: ({uv_center[0]:.1f}, {uv_center[1]:.1f})")
            self.logger.info(f"    Transformed to: ({arm_x:.1f}, {arm_y:.1f})")
            self.logger.info(f"  🎯 FINAL COORDINATES for ({row},{col}): X={arm_x:.1f}, Y={arm_y:.1f}")

            return arm_x, arm_y
        else:
            raise RuntimeError("Camera controller not available")

    def calibrate_arm(self):
        """Calibrate arm (placeholder for future implementation)."""
        self.logger.info("Arm calibration requested (not implemented)")
        return True

    def park_arm(self):
        """Park arm in safe position."""
        return self.move_to_neutral_position()

    def get_arm_status(self):
        """Get current arm status."""
        return {
            'connected': self.is_arm_available(),
            'arm_thread_active': self.arm_thread is not None and self.arm_thread.isRunning() if self.arm_thread else False,
            'safe_z': self.safe_z,
            'draw_z': self.draw_z,
            'symbol_size': self.symbol_size
        }
