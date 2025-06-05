# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Arm Movement Controller module for TicTacToe application.
This module provides centralized robotic arm control and movement coordination.
Refactored from pyqt_gui.py to consolidate arm control logic.
"""
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,protected-access,line-too-long,wrong-import-position
# pylint: disable=broad-exception-caught,unused-variable,no-else-return,superfluous-parens
# pylint: disable=too-many-nested-blocks,no-member,unspecified-encoding,unused-argument
# pylint: disable=wrong-import-order,unused-import,no-name-in-module,import-outside-toplevel
# pylint: disable=raise-missing-from

import json
import logging

import numpy as np
from PyQt5.QtCore import QObject  # pylint: disable=no-name-in-module
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSignal

from app.core.arm_thread import ArmThread
from app.main import game_logic
from app.main.arm_controller import ArmController
from app.main.constants import DEFAULT_DRAW_Z
from app.main.constants import DEFAULT_SAFE_Z
from app.main.constants import DRAWING_SPEED
from app.main.constants import MAX_SPEED
from app.main.game_utils import setup_logger

# Import required modules
from app.main.path_utils import setup_project_path

setup_project_path()

DEFAULT_SYMBOL_SIZE = 15.0


class ArmMovementController(QObject):
    """Centralized controller for robotic arm movements and coordination."""

    # Signals
    arm_connected = pyqtSignal(bool)  # connection_status
    arm_move_completed = pyqtSignal(bool)  # success (DEPRECATED - use arm_turn_completed)
    arm_turn_completed = pyqtSignal(bool)  # success - emitted ONLY after neutral position reached
    arm_status_changed = pyqtSignal(str)  # status_message
    neutral_position_reached = pyqtSignal(bool)  # success - emitted when neutral position is reached

    def __init__(self, main_window, config):
        super().__init__()

        self.main_window = main_window
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Arm components
        self.arm_thread = None
        self.arm_controller = None

        # Configuration - will be loaded from calibration file
        self.safe_z = None
        self.draw_z = None
        self.symbol_size = DEFAULT_SYMBOL_SIZE

        # Turn completion tracking
        self._pending_turn_completion = False
        self._pending_turn_success = False

        # Load calibration data (neutral position, safe_z, draw_z)
        self.neutral_position, self.safe_z, self.draw_z = self._load_calibration_data()

        # Initialize arm
        self._init_arm_components()

    def _load_calibration_data(self):
        """Load neutral position, safe_z, and draw_z from calibration file.

        Returns:
            Tuple[Dict, float, float]: (neutral_position, safe_z, draw_z)

        Raises:
            RuntimeError: If calibration file is missing or invalid
        """
        calibration_path = "/Users/michalprusek/PycharmProjects/TicTacToe/app/calibration/hand_eye_calibration.json"

        try:
            with open(calibration_path, 'r') as f:
                calibration_data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"CRITICAL ERROR: Calibration file not found at {calibration_path}. "
                "The robotic arm cannot operate without proper calibration. "
                "Please run the calibration script first."
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"CRITICAL ERROR: Invalid JSON in calibration file {calibration_path}: {e}. "
                "The calibration file is corrupted. Please re-run the calibration script."
            )
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL ERROR: Cannot read calibration file {calibration_path}: {e}"
            )

        # Validate required fields
        required_fields = ["neutral_position", "safe_z", "touch_z"]
        missing_fields = [field for field in required_fields if field not in calibration_data]
        if missing_fields:
            raise RuntimeError(
                f"CRITICAL ERROR: Missing required fields in calibration file: {missing_fields}. "
                "The calibration file is incomplete. Please re-run the calibration script."
            )

        # Extract neutral position
        neutral_pos = calibration_data["neutral_position"]
        required_coords = ["x", "y", "z"]
        missing_coords = [coord for coord in required_coords if coord not in neutral_pos]
        if missing_coords:
            raise RuntimeError(
                f"CRITICAL ERROR: Missing neutral position coordinates: {missing_coords}. "
                "The calibration file is incomplete. Please re-run the calibration script."
            )

        neutral_position = {
            'x': float(neutral_pos['x']),
            'y': float(neutral_pos['y']),
            'z': float(neutral_pos['z'])
        }

        # Extract Z heights
        safe_z = float(calibration_data["safe_z"])
        draw_z = float(calibration_data["touch_z"])  # touch_z is the drawing height

        self.logger.info("✅ CALIBRATION LOADED SUCCESSFULLY:")
        self.logger.info(f"  Neutral Position: X={neutral_position['x']:.2f}, Y={neutral_position['y']:.2f}, Z={neutral_position['z']:.2f}")
        self.logger.info(f"  Safe Z: {safe_z:.2f}")
        self.logger.info(f"  Draw Z: {draw_z:.2f}")

        return neutral_position, safe_z, draw_z

    def _init_arm_components(self):
        """Initialize arm thread and controller."""
        # Get arm port from config
        arm_port = None
        if hasattr(self.config, 'arm_controller') and hasattr(self.config.arm_controller, 'port'):
            arm_port = self.config.arm_controller.port

        # Get Z heights from config
        if hasattr(self.config, 'arm_controller'):
            if hasattr(self.config.arm_controller, 'safe_z'):
                self.safe_z = self.config.arm_controller.safe_z
            if hasattr(self.config.arm_controller, 'draw_z'):
                self.draw_z = self.config.arm_controller.draw_z

        arm_connection_successful = False

        try:
            # Initialize arm thread
            self.arm_thread = ArmThread(port=arm_port)
            self.arm_thread.start()

            # Connect to arm
            if self.arm_thread.connect():
                self.logger.info(
                    "Robotic arm successfully connected via ArmThread.")
                arm_connection_successful = True
                self.move_to_neutral_position()
            else:
                self.logger.error(
                    "Failed to connect to robotic arm via ArmThread.")
                arm_connection_successful = False

            # Initialize arm controller (legacy/backup)
            self.arm_controller = ArmController(
                port=arm_port,
                draw_z=self.draw_z,
                safe_z=self.safe_z,
                speed=MAX_SPEED
            )
            # CRITICAL FIX: Sync connection
            # status from ArmThread to ArmController
            if self.arm_thread and self.arm_thread.connected:
                self.arm_controller.connected = True
                self.arm_controller.swift = self.arm_thread.arm_controller.swift if hasattr(self.arm_thread.arm_controller, 'swift') else None

        except Exception as e:
            self.logger.error("Error initializing arm components: {e}")
            arm_connection_successful = False
            # Initialize dummy arm controller for fallback
            self.arm_controller = ArmController(
                port=arm_port,
                draw_z=self.draw_z,
                safe_z=self.safe_z,
                speed=MAX_SPEED
            )

        # Store connection status - signal will
        # be emitted from GUI after connections are made
        self._initial_connection_status = arm_connection_successful
        self.logger.info(
            "Arm initialization complete. Connection status: %s", arm_connection_successful)

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
        """Move arm to neutral position and emit completion signal (NON-BLOCKING)."""
        if not self.is_arm_available():
            self.logger.warning("Cannot move to neutral: arm not available.")
            self.neutral_position_reached.emit(False)
            return False

        try:
            # REQUIREMENT: Use NON-BLOCKING operation to maintain GUI responsiveness
            success = self._unified_arm_command(
                'park',
                x=self.neutral_position['x'],
                y=self.neutral_position['y'],
                z=self.neutral_position['z'],
                wait=False  # NON-BLOCKING: Don't wait for completion
            )

            if success:
                self.logger.info("🔄 NEUTRAL POSITION: Non-blocking move initiated to calibrated neutral position.")
                self.arm_status_changed.emit("move_success")

                # Use QTimer to check completion asynchronously
                self._start_neutral_position_monitoring()
            else:
                self.logger.error("❌ NEUTRAL POSITION FAILED: Failed to initiate move to neutral position.")
                self.arm_status_changed.emit("move_failed")
                self.neutral_position_reached.emit(False)

            return success

        except Exception as e:
            self.logger.error("❌ NEUTRAL POSITION ERROR: Error moving to neutral position: %s", e)
            self.arm_status_changed.emit("move_failed")
            self.neutral_position_reached.emit(False)
            return False

    def _start_neutral_position_monitoring(self):
        """Start monitoring arm position to detect when neutral position is reached (NON-BLOCKING)."""
        self.logger.info("🔄 MONITORING: Starting neutral position monitoring")

        # Initialize monitoring variables
        self._neutral_monitoring_attempts = 0
        self._max_monitoring_attempts = 20  # 10 seconds at 500ms intervals

        # Start monitoring timer
        self._neutral_monitoring_timer = QTimer()
        self._neutral_monitoring_timer.timeout.connect(self._check_neutral_position_reached)
        self._neutral_monitoring_timer.start(500)  # Check every 500ms

    def _check_neutral_position_reached(self):
        """Check if arm has reached neutral position (called by QTimer)."""
        self._neutral_monitoring_attempts += 1

        try:
            # Get current arm position
            if self.arm_thread and self.arm_thread.connected:
                current_pos = self.arm_thread.get_position(cached=False)
                if current_pos:
                    current_x, current_y, current_z = current_pos
                    target_x = self.neutral_position['x']
                    target_y = self.neutral_position['y']
                    target_z = self.neutral_position['z']

                    # Check if position is close enough to neutral (tolerance: 2mm)
                    tolerance = 2.0
                    distance = ((current_x - target_x)**2 + (current_y - target_y)**2 + (current_z - target_z)**2)**0.5

                    if distance <= tolerance:
                        # Neutral position reached!
                        self.logger.info("✅ NEUTRAL POSITION REACHED: Distance=%.1fmm (tolerance=%.1fmm)", distance, tolerance)
                        self._stop_neutral_monitoring()
                        self.neutral_position_reached.emit(True)
                        return
                    else:
                        self.logger.debug("🔄 MONITORING: Distance to neutral=%.1fmm (attempt %d/%d)",
                                        distance, self._neutral_monitoring_attempts, self._max_monitoring_attempts)
                else:
                    self.logger.debug("🔄 MONITORING: Could not get arm position (attempt %d/%d)",
                                    self._neutral_monitoring_attempts, self._max_monitoring_attempts)
            else:
                self.logger.warning("🔄 MONITORING: Arm not connected (attempt %d/%d)",
                                  self._neutral_monitoring_attempts, self._max_monitoring_attempts)

        except Exception as e:
            self.logger.error("🔄 MONITORING ERROR: %s (attempt %d/%d)", e,
                            self._neutral_monitoring_attempts, self._max_monitoring_attempts)

        # Check timeout
        if self._neutral_monitoring_attempts >= self._max_monitoring_attempts:
            self.logger.warning("⏰ MONITORING TIMEOUT: Neutral position not reached within timeout")
            self._stop_neutral_monitoring()
            self.neutral_position_reached.emit(False)

    def _stop_neutral_monitoring(self):
        """Stop neutral position monitoring."""
        if hasattr(self, '_neutral_monitoring_timer') and self._neutral_monitoring_timer:
            self._neutral_monitoring_timer.stop()
            self._neutral_monitoring_timer = None
            self.logger.debug("🔄 MONITORING: Stopped neutral position monitoring")

    def draw_ai_symbol(self, row, col, symbol_to_draw):
        """Draw AI symbol at specified position using consolidated arm control."""
        self.logger.info("Request to draw {symbol_to_draw} at ({row},{col})")

        if not self.is_arm_available():
            raise RuntimeError(f"Cannot draw symbol {symbol_to_draw}: robotic arm is not available")

        # Get target coordinates
        target_x, target_y = self._get_cell_coordinates_from_yolo(row, col)
        if target_x is None or target_y is None:
            raise RuntimeError(f"Cannot get coordinates for drawing at ({row},{col})")

        self.logger.info(
            "Drawing {symbol_to_draw} at coordinates ({target_x:.1f}, {target_y:.1f})")

        try:
            # REQUIREMENT: Use NON-BLOCKING operations to maintain GUI responsiveness
            # Use thread-based approach for non-blocking execution
            success = self._unified_arm_command(
                'draw_o' if symbol_to_draw == game_logic.PLAYER_O else 'draw_x',
                x=target_x,
                y=target_y,
                radius=self.symbol_size / 2 if symbol_to_draw == game_logic.PLAYER_O else None,
                size=self.symbol_size if symbol_to_draw == game_logic.PLAYER_X else None,
                speed=DRAWING_SPEED,
                wait=False  # NON-BLOCKING: Don't wait for completion
            )

            if success:
                self.logger.info(
                    "Successfully drew {symbol_to_draw} at ({row},{col})")

                # REQUIREMENT: Return to neutral position after every move
                self.logger.info("🔄 TURN SEQUENCE: Drawing complete, returning to neutral position")

                # Store drawing success for later emission after neutral position reached
                self._pending_turn_completion = True
                self._pending_turn_success = True

                # Connect to neutral position signal for this turn
                self.neutral_position_reached.connect(self._handle_neutral_position_after_drawing)

                # Return to neutral position (will emit arm_turn_completed when reached)
                QTimer.singleShot(500, self.move_to_neutral_position)

                # Emit legacy signal for backward compatibility
                self.arm_move_completed.emit(True)
                return True
            else:
                self.logger.error("Failed to draw {symbol_to_draw}")
                self.arm_move_completed.emit(False)
                self.arm_turn_completed.emit(False)
                return False

        except Exception as e:
            self.logger.error("Error drawing symbol: %s", e)
            self.arm_move_completed.emit(False)
            self.arm_turn_completed.emit(False)
            return False

    def _handle_neutral_position_after_drawing(self, success):
        """Handle neutral position reached after drawing - emit turn completion signal."""
        # Disconnect the signal to avoid multiple connections
        try:
            self.neutral_position_reached.disconnect(self._handle_neutral_position_after_drawing)
        except TypeError:
            # Signal was not connected, ignore
            pass

        if self._pending_turn_completion:
            if success and self._pending_turn_success:
                self.logger.info("🎯 TURN COMPLETED: Arm returned to neutral, turn is complete")
                self.arm_turn_completed.emit(True)
            else:
                self.logger.error("❌ TURN FAILED: Failed to return to neutral position")
                self.arm_turn_completed.emit(False)

            # Reset pending flags
            self._pending_turn_completion = False
            self._pending_turn_success = False

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

        self.logger.info(
            "Drawing winning line from ({start_x:.1f}, {start_y:.1f}) to ({end_x:.1f}, {end_y:.1f})")

        try:
            # REQUIREMENT: Use NON-BLOCKING operations to maintain GUI responsiveness
            # Start winning line drawing sequence (non-blocking)
            self.logger.info("🔄 WINNING LINE: Starting non-blocking drawing sequence")

            # Store winning line success for later emission after neutral position reached
            self._pending_turn_completion = True
            self._pending_turn_success = True

            # Connect to neutral position signal for this winning line
            self.neutral_position_reached.connect(self._handle_neutral_position_after_drawing)

            # Start the winning line drawing sequence asynchronously
            self._start_winning_line_sequence(start_x, start_y, end_x, end_y)

            return True

        except Exception as e:
            self.logger.error("Error drawing winning line: %s", e)
            raise

    def _start_winning_line_sequence(self, start_x, start_y, end_x, end_y):
        """Start winning line drawing sequence asynchronously (NON-BLOCKING)."""
        self.logger.info("🔄 WINNING LINE SEQUENCE: Starting asynchronous drawing")

        # Initialize sequence variables
        self._winning_line_step = 0
        self._winning_line_coords = (start_x, start_y, end_x, end_y)

        # Start sequence timer
        self._winning_line_timer = QTimer()
        self._winning_line_timer.timeout.connect(self._execute_winning_line_step)
        self._winning_line_timer.start(100)  # Execute steps every 100ms

    def _execute_winning_line_step(self):
        """Execute next step in winning line sequence (called by QTimer)."""
        try:
            start_x, start_y, end_x, end_y = self._winning_line_coords

            if self._winning_line_step == 0:
                # Step 1: Move above start position at safe_z
                self.logger.info("🔄 WINNING LINE STEP 1: Moving to start position")
                success = self._unified_arm_command('go_to_position',
                                                  x=start_x, y=start_y, z=self.safe_z,
                                                  speed=MAX_SPEED, wait=False)
                if success:
                    self._winning_line_step = 1
                    # Continue to next step after delay
                    self._winning_line_timer.start(1000)  # Wait 1 second
                else:
                    self._stop_winning_line_sequence(False, "Failed to move to start position")

            elif self._winning_line_step == 1:
                # Step 2: Lower to drawing position
                self.logger.info("🔄 WINNING LINE STEP 2: Lowering to drawing position")
                success = self._unified_arm_command('go_to_position',
                                                  x=start_x, y=start_y, z=self.draw_z,
                                                  speed=DRAWING_SPEED, wait=False)
                if success:
                    self._winning_line_step = 2
                    self._winning_line_timer.start(1500)  # Wait 1.5 seconds
                else:
                    self._stop_winning_line_sequence(False, "Failed to lower to drawing position")

            elif self._winning_line_step == 2:
                # Step 3: Draw line to end position
                self.logger.info("🔄 WINNING LINE STEP 3: Drawing line to end position")
                success = self._unified_arm_command('go_to_position',
                                                  x=end_x, y=end_y, z=self.draw_z,
                                                  speed=DRAWING_SPEED, wait=False)
                if success:
                    self._winning_line_step = 3
                    self._winning_line_timer.start(2000)  # Wait 2 seconds for drawing
                else:
                    self._stop_winning_line_sequence(False, "Failed to draw winning line")

            elif self._winning_line_step == 3:
                # Step 4: Lift to safe position
                self.logger.info("🔄 WINNING LINE STEP 4: Lifting to safe position")
                success = self._unified_arm_command('go_to_position',
                                                  x=end_x, y=end_y, z=self.safe_z,
                                                  speed=MAX_SPEED, wait=False)
                if success:
                    self._winning_line_step = 4
                    self._winning_line_timer.start(1000)  # Wait 1 second
                else:
                    self._stop_winning_line_sequence(False, "Failed to lift arm after drawing")

            elif self._winning_line_step == 4:
                # Step 5: Complete sequence and return to neutral
                self.logger.info("✅ WINNING LINE COMPLETE: Moving to neutral position")
                self._stop_winning_line_sequence(True, "Winning line successfully drawn")

                # Move to neutral after a delay
                QTimer.singleShot(1000, self.move_to_neutral_position)

        except Exception as e:
            self.logger.error("🔄 WINNING LINE ERROR: %s", e)
            self._stop_winning_line_sequence(False, f"Error in winning line sequence: {e}")

    def _stop_winning_line_sequence(self, success, message):
        """Stop winning line sequence."""
        if hasattr(self, '_winning_line_timer') and self._winning_line_timer:
            self._winning_line_timer.stop()
            self._winning_line_timer = None

        if success:
            self.logger.info("✅ WINNING LINE SUCCESS: %s", message)
            self.arm_status_changed.emit("Winning line drawn!")
        else:
            self.logger.error("❌ WINNING LINE FAILED: %s", message)
            self.arm_status_changed.emit("Winning line failed!")

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
                    speed=kwargs.get('speed', DRAWING_SPEED),
                    wait=kwargs.get('wait', False)  # NON-BLOCKING by default
                )
            elif command == 'draw_x':
                success = self.arm_thread.draw_x(
                    center_x=kwargs.get('x'),
                    center_y=kwargs.get('y'),
                    size=kwargs.get('size'),
                    speed=kwargs.get('speed', DRAWING_SPEED),
                    wait=kwargs.get('wait', False)  # NON-BLOCKING by default
                )
            elif command == 'go_to_position':
                # REQUIREMENT: Default to NON-BLOCKING operations for GUI responsiveness
                success = self.arm_thread.go_to_position(
                    x=kwargs.get('x'),
                    y=kwargs.get('y'),
                    z=kwargs.get('z'),
                    speed=kwargs.get('speed', MAX_SPEED),
                    wait=kwargs.get('wait', False)  # NON-BLOCKING by default
                )
            elif command == 'park':
                # REQUIREMENT: Default to NON-BLOCKING operations for GUI responsiveness
                success = self.arm_thread.go_to_position(
                    x=kwargs.get('x', self.neutral_position['x']),
                    y=kwargs.get('y', self.neutral_position['y']),
                    z=kwargs.get('z', self.neutral_position['z']),
                    speed=MAX_SPEED // 2,
                    wait=kwargs.get('wait', False)  # NON-BLOCKING by default
                )
            else:
                raise ValueError(f"Unknown arm command: {command}")

            if not success:
                raise RuntimeError(f"Arm command '{command}' failed to execute")

            self.logger.info("Arm command '%s' executed successfully", command)
            return success

        except Exception as e:
            self.logger.error(
                "Error executing arm command '%s': %s", command, e)
            raise

    def _get_cell_coordinates_from_yolo(self, row, col):
        """Get cell coordinates from YOLO detection with improved interpolation."""
        self.logger.info(
            "🔍 COORDINATE TRANSFORMATION DEBUG for cell (%d,%d):", row, col)

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

            self.logger.info("  📍 Step 1 - Grid position: ({row},{col})")
            self.logger.info(
                "  📍 Step 2 - UV center from camera: ({uv_center[0]:.1f}, {uv_center[1]:.1f})")

            # ENHANCED DEBUG: Show grid points used for this cell
            self.logger.info(
                "  🔧 GRID DEBUG - Cell ({row},{col}) calculation:")
            if hasattr(game_state_obj, '_grid_points') and game_state_obj._grid_points is not None:
                # Show which grid points were used for this cell
                p_tl_idx = row * 4 + col          # top-left
                p_tr_idx = row * 4 + (col + 1)    # top-right
                p_bl_idx = (row + 1) * 4 + col    # bottom-left
                p_br_idx = (row + 1) * 4 + (col + 1)  # bottom-right

                grid_points = game_state_obj._grid_points
                if len(grid_points) > max(p_tl_idx, p_tr_idx, p_bl_idx, p_br_idx):
                    self.logger.info(
                        "    Grid points used: TL={p_tl_idx}, TR={p_tr_idx}, BL={p_bl_idx}, BR={p_br_idx}")
                    self.logger.info(
                        "    TL=({grid_points[p_tl_idx][0]:.0f},{grid_points[p_tl_idx][1]:.0f})")
                    self.logger.info(
                        "    TR=({grid_points[p_tr_idx][0]:.0f},{grid_points[p_tr_idx][1]:.0f})")
                    self.logger.info(
                        "    BL=({grid_points[p_bl_idx][0]:.0f},{grid_points[p_bl_idx][1]:.0f})")
                    self.logger.info(
                        "    BR=({grid_points[p_br_idx][0]:.0f},{grid_points[p_br_idx][1]:.0f})")

                    # Calculate expected center for verification
                    expected_center_u = (grid_points[p_tl_idx][0] + grid_points[p_tr_idx][0] +
                                         grid_points[p_bl_idx][0] + grid_points[p_br_idx][0]) / 4
                    expected_center_v = (grid_points[p_tl_idx][1] + grid_points[p_tr_idx][1] +
                                         grid_points[p_bl_idx][1] + grid_points[p_br_idx][1]) / 4
                    self.logger.info(
                        "    Expected center: ({expected_center_u:.1f},{expected_center_v:.1f})")
                    self.logger.info(
                        "    Actual center:   ({uv_center[0]:.1f}, {uv_center[1]:.1f})")

                    # Verify calculation
                    diff_u = abs(expected_center_u - uv_center[0])
                    diff_v = abs(expected_center_v - uv_center[1])
                    if diff_u > 5 or diff_v > 5:
                        self.logger.warning(
                            "    ⚠️ CENTER MISMATCH: diff=({diff_u:.1f},{diff_v:.1f})")

            # DEBUG: Log all cell centers for comparison
            self.logger.info("  🗺️ ALL CELL CENTERS for reference:")
            for debug_row in range(3):
                for debug_col in range(3):
                    debug_center = game_state_obj.get_cell_center_uv(debug_row, debug_col)
                    if debug_center is not None:
                        self.logger.info(
                            "    Cell ({debug_row},{debug_col}): UV=({debug_center[0]:.1f}, {debug_center[1]:.1f})")

            # Get calibration data
            calibration_data = camera_controller.get_calibration_data()
            if calibration_data and "perspective_transform_matrix_xy_to_uv" in calibration_data:
                # Use matrix transformation (direct, no
                # scaling needed since calibration is also 1920x1080)
                xy_to_uv_matrix = calibration_data["perspective_transform_matrix_xy_to_uv"]

                if xy_to_uv_matrix:
                    try:
                        # Inverse matrix for UV->XY transformation
                        xy_to_uv_matrix = np.array(xy_to_uv_matrix, dtype=np.float32)
                        uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)

                        # Homogeneous coordinates for
                        # transformation using original UV coordinates
                        uv_point_homogeneous = np.array([uv_center[0], uv_center[1], 1.0], dtype=np.float32).reshape(3, 1)
                        xy_transformed_homogeneous = np.dot(uv_to_xy_matrix, uv_point_homogeneous)

                        if xy_transformed_homogeneous[2, 0] != 0:
                            arm_x = xy_transformed_homogeneous[0, 0] / xy_transformed_homogeneous[2, 0]
                            arm_y = xy_transformed_homogeneous[1, 0] / xy_transformed_homogeneous[2, 0]
                            self.logger.info(
                                "  📍 Step 3 - Matrix transformation (direct): ({arm_x:.1f}, {arm_y:.1f})")

                            # COORDINATE VALIDATION -
                            # Check if coordinates are reasonable
                            self.logger.info("  🔍 COORDINATE VALIDATION:")

                            # Check against calibration data bounds
                            cal_points = calibration_data.get("calibration_points_raw", [])
                            if cal_points:
                                cal_x_coords = [point["robot_xyz"][0] for point in cal_points]
                                cal_y_coords = [point["robot_xyz"][1] for point in cal_points]
                                min_x, max_x = min(cal_x_coords), max(cal_x_coords)
                                min_y, max_y = min(cal_y_coords), max(cal_y_coords)

                                self.logger.info(
                                    "    Calibration X range: {min_x:.1f} to {max_x:.1f}")
                                self.logger.info(
                                    "    Calibration Y range: {min_y:.1f} to {max_y:.1f}")
                                self.logger.info(
                                    "    Current coordinates: X={arm_x:.1f}, Y={arm_y:.1f}")

                                if not (min_x <= arm_x <= max_x):
                                    self.logger.warning(
                                        "    ⚠️ X coordinate {arm_x:.1f} is outside calibration range [{min_x:.1f}, {max_x:.1f}]")
                                if not (min_y <= arm_y <= max_y):
                                    self.logger.warning(
                                        "    ⚠️ Y coordinate {arm_y:.1f} is outside calibration range [{min_y:.1f}, {max_y:.1f}]")

                                # Find closest calibration point for reference
                                distances = []
                                for point in cal_points:
                                    cal_uv = point["target_uv"]
                                    cal_xy = point["robot_xyz"]
                                    uv_dist = ((uv_center[0] - cal_uv[0])**2 + (uv_center[1] - cal_uv[1])**2)**0.5
                                    distances.append((uv_dist, cal_uv, cal_xy))

                                closest = min(distances, key=lambda x: x[0])
                                self.logger.info(
                                    "    📍 Closest calibration point:")
                                self.logger.info(
                                    "      UV: {closest[1]} -> XY: {closest[2][:2]}")
                                self.logger.info(
                                    "      Distance in UV space: {closest[0]:.1f} pixels")
                        else:
                            raise RuntimeError("Division by zero in UV->XY transformation")

                    except Exception as e:
                        raise RuntimeError(f"Error in UV->XY transformation for ({row},{col}): {e}") from e
                else:
                    raise RuntimeError("Missing transformation matrix in calibration data")
            else:
                raise RuntimeError("No perspective transformation matrix available in calibration data")

            # DEBUG: Log final coordinates
            self.logger.info("  🔍 CALIBRATION COMPARISON:")
            self.logger.info(
                "    Current UV: ({uv_center[0]:.1f}, {uv_center[1]:.1f})")
            self.logger.info("    Transformed to: ({arm_x:.1f}, {arm_y:.1f})")
            self.logger.info(
                "  🎯 FINAL COORDINATES for ({row},{col}): X={arm_x:.1f}, Y={arm_y:.1f}")

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
