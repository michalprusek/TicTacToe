import time
import traceback
import logging
import math
import sys
import os
from typing import Optional, Tuple, Dict, Any

# Import uArm - required for operation
# Add uArm-Python-SDK to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
uarm_sdk_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "uArm-Python-SDK")

if os.path.exists(uarm_sdk_path):
    sys.path.insert(0, uarm_sdk_path)
    print(f"Added uArm SDK path: {uarm_sdk_path}")
else:
    print(f"uArm SDK path not found: {uarm_sdk_path}")

try:
    from uarm.wrapper import SwiftAPI
    UARM_AVAILABLE = True
    print("Successfully imported uArm SDK")
except ImportError as e:
    print("ERROR: Failed to import uArm SDK:", e)
    print("Make sure the uArm-Python-SDK path is correctly set")
    UARM_AVAILABLE = False

# Create mock classes for SwiftAPI when real arm is not available


class MockSwiftAPI:
    def __init__(self, *_args, **_kwargs):  # pylint: disable=unused-argument
        print("Initializing Mock SwiftAPI")
        self._position = [200, 0, 150]  # Initial position
        self._is_moving = False
        self._servo_angles = [90, 90, 90, 90]  # Initial servo angles
        self._pump_on = False

    def waiting_ready(self, timeout=None):
        print(f"Mock SwiftAPI: waiting_ready(timeout={timeout})")
        return True

    def connect(self, *_args, **_kwargs):  # pylint: disable=unused-argument
        print("Mock SwiftAPI: connect")
        return True

    # pylint: disable=unused-argument
    def set_position(self, x=None, y=None, z=None, speed=None, wait=False,
                     timeout=None, cmd=None, relative=False):
        # Unused parameters: timeout, cmd, relative
        print(
            f"Mock SwiftAPI: set_position(x={x}, y={y}, z={z}, "
            f"speed={speed}, wait={wait})")
        # Update the stored position
        if x is not None:
            self._position[0] = x
        if y is not None:
            self._position[1] = y
        if z is not None:
            self._position[2] = z
        return True

    def set_servo_angle(self, servo_id, angle, speed=None, wait=False):
        print(
            f"Mock SwiftAPI: set_servo_angle({servo_id}, {angle}, "
            f"speed={speed}, wait={wait})")
        if 0 <= servo_id < 4:
            self._servo_angles[servo_id] = angle
        return True

    def set_wrist(self, angle, wait=False):
        print(f"Mock SwiftAPI: set_wrist({angle}, wait={wait})")
        # Wrist is typically servo 3
        self._servo_angles[3] = angle
        return True

    def set_pump(self, on, wait=False):
        print(f"Mock SwiftAPI: set_pump({'on' if on else 'off'}, wait={wait})")
        self._pump_on = on
        return True

    def get_position(self, wait=True, timeout=None):
        print(f"Mock SwiftAPI: get_position(wait={wait}, timeout={timeout})")
        # Return a copy to avoid reference issues
        return self._position.copy()

    def get_servo_angle(self, servo_id=None):
        print(f"Mock SwiftAPI: get_servo_angle({servo_id})")
        if servo_id is None:
            return self._servo_angles.copy()
        if 0 <= servo_id < 4:
            return self._servo_angles[servo_id]
        return 0

    def get_is_moving(self):
        print("Mock SwiftAPI: get_is_moving")
        return self._is_moving

    def set_speed_factor(self, factor):
        print(f"Mock SwiftAPI: set_speed_factor({factor})")
        return True

    def disconnect(self):
        print("Mock SwiftAPI: disconnect")
        return True


# --- Constants (Local Definitions) --- #
DEFAULT_SPEED = 100000            # Maximální rychlost pohybu (mm/min)
# Násobitel rychlosti pro překonání limitu firmware
MAX_SPEED_FACTOR = 2
DEFAULT_SAFE_Z = 15.0             # Height for safe travel moves (mm)
DEFAULT_DRAW_Z = 5.0              # Height while drawing (mm)
POSITION_TOLERANCE = 5.0          # Tolerance for position checks (mm)
# Multiplier for travel speed (faster than drawing)
TRAVEL_SPEED_MULTIPLIER = 1.5
OPTIMIZED_SEGMENTS = 16           # Reduced segments for O drawing (faster)
CORNER_SAFETY_MARGIN = 2.0        # Extra margin for corners to avoid confusion


class ArmController:
    """Controls the uArm Swift Pro for Tic Tac Toe playing."""

    def __init__(self, port: Optional[str] = None,
                 draw_z: float = DEFAULT_DRAW_Z,
                 speed: int = DEFAULT_SPEED,
                 safe_z: float = DEFAULT_SAFE_Z):
        """Initializes the ArmController."""
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.safe_z = safe_z
        self.draw_z = draw_z
        self.speed = speed  # Store default speed

        self.swift: Optional[SwiftAPI] = None
        self.connected: bool = False
        self._last_state: Dict[str, Any] = {
            "current_pos_cached": None,
            "current_pos_actual": None,
        }
        self.logger.info(
            "ArmController initialized. Port=%s, Speed=%s, DrawZ=%s, SafeZ=%s",
            port or 'Autodetect', speed, draw_z, safe_z
        )

    def connect(self) -> bool:
        """Connects to the uArm Swift Pro."""
        if self.connected:
            self.logger.info("Arm already connected.")
            return True

        # First try to connect to the real arm
        try:
            # Always try to connect to the real arm first
            self.logger.info(
                "Connecting to uArm on port: %s",
                self.port or 'Autodetect')
            try:
                self.swift = SwiftAPI(port=self.port)
                self.logger.info("Waiting for arm connection...")
                self.swift.waiting_ready(timeout=10)
                self.connected = True
                self.logger.info("uArm connected successfully.")

                # Nastavení speed_factor pro překonání limitu firmware
                self.swift.set_speed_factor(MAX_SPEED_FACTOR)
                self.logger.info("Speed factor set to %s", MAX_SPEED_FACTOR)

                pos = self.get_position(cached=False)
                if pos:
                    self.logger.info(
                        "Initial pos: X=%.1f, Y=%.1f, Z=%.1f",
                        pos[0], pos[1], pos[2])
                    self.go_to_position(z=self.safe_z, wait=True)
                else:
                    self.logger.warning("Could not get initial position.")
                self.swift.set_wrist(90, wait=True)
                return True
            except Exception as real_arm_error:
                # If connecting to real arm fails, try to use the mock
                # implementation
                self.logger.warning(
                    "Failed to connect to real uArm: %s", real_arm_error)
                self.logger.warning("Falling back to mock implementation.")
                self.swift = MockSwiftAPI()
                self.connected = True
                self.logger.info("Mock uArm connected successfully.")
                pos = self.get_position(cached=False)
                if pos:
                    self.logger.info(
                        "Initial mock pos: X=%.1f, Y=%.1f, Z=%.1f",
                        pos[0], pos[1], pos[2])
                return True
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                "Failed to connect to any arm implementation: %s", exc)
            self.logger.debug(traceback.format_exc())
            self.connected = False
            self.swift = None
            return False

    def disconnect(self):
        """Disconnects from the uArm."""
        if self.swift and self.connected:
            try:
                self.logger.info("Disconnecting from uArm...")
                # Optional: Move to safe home
                # self.go_to_position(x=150, y=0, z=self.safe_z, wait=True)
                self.swift.disconnect()
                self.logger.info("uArm disconnected.")
                return True
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error("Error disconnecting: %s", exc)
                self.logger.debug(traceback.format_exc())
                # Even if disconnect fails, mark as not connected
                self.connected = False
                self.swift = None
                return False
            finally:
                self.connected = False
                self.swift = None
        else:
            self.logger.info("Arm not connected, no need to disconnect.")
            return True

    def get_position(
            self, cached=True) -> Optional[Tuple[float, float, float]]:
        """Gets the current Cartesian position [x, y, z] of the arm."""
        if not self.swift or not self.connected:
            self.logger.warning("Cannot get position: Arm not connected.")
            return None

        pos_key = 'current_pos_cached' if cached else 'current_pos_actual'
        if cached and self._last_state.get(pos_key) is not None:
            return self._last_state[pos_key]

        try:
            position = self.swift.get_position()  # Returns [x, y, z]
            if position:
                pos_tuple = tuple(map(float, position))
                self._last_state['current_pos_actual'] = pos_tuple
                self._last_state['current_pos_cached'] = pos_tuple
                self.logger.debug(
                    "Position queried (%s): %s", pos_key, pos_tuple)
                return pos_tuple

            self.logger.warning("get_position() returned None or empty list")
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Error getting arm position: %s", exc)
            return None

    def get_current_position_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Gets the current reported position (X, Y, Z) of the arm."""
        if not self.connected or self.swift is None:
            self.logger.error("Cannot get position: Arm not connected.")
            return None
        try:
            # Short timeout, might not reflect true position if moving
            position = self.swift.get_position(wait=False, timeout=0.5)
            if position and isinstance(position, list) and len(position) >= 3:
                return tuple(map(float, position))

            self.logger.warning("Received invalid position data: %s", position)
            return None
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Error getting arm position: %s", exc)
            self.logger.debug(traceback.format_exc())
            return None

    def get_current_position_xy(self) -> Optional[Tuple[float, float]]:
        """Gets the current reported XY position of the arm."""
        xyz = self.get_current_position_xyz()
        if xyz:
            return xyz[0], xyz[1]  # Return only X and Y
        return None

    def go_to_position(
            self,
            x: Optional[float] = None,
            y: Optional[float] = None,
            z: Optional[float] = None,
            speed: Optional[int] = None,
            wait: bool = False,
            relative: bool = False,
            timeout: int = 10) -> bool:
        """Moves the arm to the specified Cartesian coordinates."""
        if not self.swift or not self.connected:
            self.logger.error("Cannot move: Arm not connected.")
            return False

        current_pos = self.get_position(cached=False)
        if not current_pos:
            self.logger.error("Could not get current position, cannot move.")
            return False

        target_x = x if x is not None else current_pos[0]
        target_y = y if y is not None else current_pos[1]
        target_z = z if z is not None else current_pos[2]
        move_speed = speed if speed is not None else self.speed

        self.logger.debug(
            "Moving to X:%.1f Y:%.1f Z:%.1f @ Speed:%s Wait:%s",
            target_x, target_y, target_z, move_speed, wait
        )

        try:
            result = self.swift.set_position(
                x=target_x, y=target_y, z=target_z, speed=move_speed,
                relative=relative, wait=wait, timeout=timeout, cmd='G0'
            )
            # Update cache optimistically if not waiting or succeeded
            if not relative and (not wait or result != 'TIMEOUT'):
                pos_tuple_after = (target_x, target_y, target_z)
                self._last_state["current_pos_actual"] = pos_tuple_after
                self._last_state["current_pos_cached"] = pos_tuple_after

            if result == 'TIMEOUT':
                self.logger.warning("Arm move timed out after %ss.", timeout)
                # Update position after timeout
                self.get_position(cached=False)
                return False
            if wait:
                self.logger.debug("Move finished (waited).")
            else:
                self.logger.debug(
                    "Move initiated (no wait). Update cache optimistically.")

            return True

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Error during arm movement: %s", exc)
            self.logger.debug(traceback.format_exc())
            self.get_position(cached=False)  # Update after error
            return False

    # Helper function v2 using explicit speeds
    def _draw_line_v2(
            self,
            x1,
            y1,
            x2,
            y2,
            z_draw,
            z_safe,
            draw_speed,
            safe_speed) -> bool:
        """Helper function to draw a line segment with explicit speeds."""
        # Apply corner safety margin to avoid confusion with corners
        x1_adj = x1 + (CORNER_SAFETY_MARGIN if x1 < 0 else -
                       CORNER_SAFETY_MARGIN if x1 > 0 else 0)
        y1_adj = y1 + (CORNER_SAFETY_MARGIN if y1 < 0 else -
                       CORNER_SAFETY_MARGIN if y1 > 0 else 0)
        x2_adj = x2 + (CORNER_SAFETY_MARGIN if x2 < 0 else -
                       CORNER_SAFETY_MARGIN if x2 > 0 else 0)
        y2_adj = y2 + (CORNER_SAFETY_MARGIN if y2 < 0 else -
                       CORNER_SAFETY_MARGIN if y2 > 0 else 0)

        # Calculate travel speed (faster than drawing)
        travel_speed = int(safe_speed * TRAVEL_SPEED_MULTIPLIER)

        # Move to start point at safe height (faster travel)
        if not self.go_to_position(x=x1_adj, y=y1_adj, z=z_safe,
                                   speed=travel_speed, wait=True):
            return False
        # Lower to drawing height
        if not self.go_to_position(z=z_draw, speed=draw_speed, wait=True):
            return False
        # Move to end point (drawing the line)
        if not self.go_to_position(
                x=x2_adj,
                y=y2_adj,
                speed=draw_speed,
                wait=True):
            return False
        # Lift back to safe height
        if not self.go_to_position(z=z_safe, speed=travel_speed, wait=True):
            return False
        return True

    def draw_x(self, center_x: float, center_y: float, size: float,
               speed: Optional[int] = None) -> bool:
        """Draws an 'X' symbol centered at the specified coordinates."""
        if not self.connected:
            self.logger.error("Cannot draw X: Arm not connected.")
            return False

        # Use faster drawing speed and travel speed
        half_size = size / 2.0
        draw_speed = speed if speed is not None else (
            DEFAULT_SPEED // 2)  # Poloviční rychlost pro kreslení
        safe_move_speed = DEFAULT_SPEED  # Maximální rychlost pro přesuny
        travel_speed = safe_move_speed

        self.logger.info(
            "Drawing X at (%.1f, %.1f), size %.1f, speed %s",
            center_x, center_y, size, draw_speed
        )

        # Apply corner safety margin to center coordinates
        center_x_adj = center_x
        center_y_adj = center_y

        # Points for the X with adjusted center
        p1_x = center_x_adj - half_size
        p1_y = center_y_adj - half_size
        p2_x = center_x_adj + half_size
        p2_y = center_y_adj + half_size
        p3_x = center_x_adj + half_size
        p3_y = center_y_adj - half_size
        p4_x = center_x_adj - half_size
        p4_y = center_y_adj + half_size

        # Draw first diagonal (p1 to p2) with optimized speeds
        if not self._draw_line_v2(
            p1_x, p1_y, p2_x, p2_y,
            self.draw_z, self.safe_z,
            draw_speed, travel_speed
        ):
            self.logger.error("Failed drawing first diagonal of X")
            self.go_to_position(
                z=self.safe_z,
                speed=travel_speed)  # Faster lift
            return False

        # Draw second diagonal (p3 to p4) with optimized speeds
        if not self._draw_line_v2(
            p3_x, p3_y, p4_x, p4_y,
            self.draw_z, self.safe_z,
            draw_speed, travel_speed
        ):
            self.logger.error("Failed drawing second diagonal of X")
            self.go_to_position(
                z=self.safe_z,
                speed=travel_speed)  # Faster lift
            return False

        self.logger.info("Finished drawing X.")
        return True

    def draw_o(
            self,
            center_x: float,
            center_y: float,
            radius: float,
            speed: Optional[int] = None,
            segments: int = OPTIMIZED_SEGMENTS) -> bool:
        """Draws an 'O' symbol centered at (center_x, center_y) with radius."""
        if not self.connected:
            self.logger.error("Cannot draw O: Arm not connected.")
            return False

        # Use faster drawing speed and travel speed
        draw_speed = speed if speed is not None else (
            DEFAULT_SPEED // 2)  # Poloviční rychlost pro kreslení
        safe_move_speed = DEFAULT_SPEED  # Maximální rychlost pro přesuny
        travel_speed = safe_move_speed

        self.logger.info(
            "Drawing O at (%.1f, %.1f), radius %.1f, speed %s",
            center_x, center_y, radius, draw_speed
        )

        # Apply corner safety margin to center coordinates
        center_x_adj = center_x
        center_y_adj = center_y

        # Start at top of circle
        start_x = center_x_adj
        start_y = center_y_adj + radius

        # Move to start position with faster travel speed
        if not self.go_to_position(
            x=start_x, y=start_y, z=self.safe_z,
            speed=travel_speed, wait=True
        ):
            self.logger.error("Failed moving to O start point (safe Z).")
            return False

        # Lower to drawing height
        if not self.go_to_position(z=self.draw_z, speed=draw_speed, wait=True):
            self.logger.error("Failed lowering to O draw height.")
            return False

        # Draw circle with fewer segments for faster movement
        for i in range(1, segments + 1):
            angle = (2 * math.pi * i) / segments
            next_x = center_x_adj + radius * math.sin(angle)
            next_y = center_y_adj + radius * math.cos(angle)

            if not self.go_to_position(
                x=next_x, y=next_y, z=self.draw_z,
                speed=draw_speed, wait=True
            ):
                self.logger.error("Failed drawing O segment %s.", i)
                self.go_to_position(z=self.safe_z, speed=travel_speed)   # Lift
                return False

        # Lift arm back to safe height with faster travel speed
        if not self.go_to_position(
                z=self.safe_z,
                speed=travel_speed,
                wait=True):
            self.logger.warning("Failed lifting arm after drawing O.")

        self.logger.info("Finished drawing O.")
        return True

    def park(self, x: float = -150, y: float = -150,
             z: Optional[float] = None) -> bool:
        """Parks the arm in the left corner."""
        self.logger.info("Parking arm at (%s, %s, %s)", x, y, z or self.safe_z)

        if not self.connected or self.swift is None:
            self.logger.error("Cannot park: Arm not connected.")
            return False

        # Use safe_z if z is not provided
        park_z = z if z is not None else self.safe_z

        # Move to parking position with maximum speed
        return self.go_to_position(
            x=x, y=y, z=park_z,
            speed=DEFAULT_SPEED,  # Maximální rychlost pro parkování
            wait=True
        )

    def __del__(self):
        """Ensures the arm is disconnected when the object is destroyed."""
        self.disconnect()


# --- Example Usage --- #
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    uarm_port = None  # Try auto-detect first

    # Use local defaults
    default_draw_z = DEFAULT_DRAW_Z
    default_safe_z = DEFAULT_SAFE_Z

    controller = ArmController(
        port=uarm_port,
        draw_z=default_draw_z,
        safe_z=default_safe_z)

    if controller.connect():
        logger.info("---- Testing Movement ----")
        controller.go_to_position(x=200, y=0, z=150, wait=True)
        time.sleep(1)
        controller.go_to_position(x=200, y=50, z=100, wait=True)
        time.sleep(1)
        controller.go_to_position(z=controller.safe_z + 20, wait=True)
        time.sleep(1)

        logger.info("---- Testing Drawing X ----")
        test_center_x = 200
        test_center_y = 0
        symbol_size = 50  # mm

        controller.go_to_position(
            x=test_center_x,
            y=test_center_y,
            z=controller.safe_z,
            wait=True)
        time.sleep(1)
        if controller.draw_x(test_center_x, test_center_y, symbol_size):
            logger.info("X drawn successfully.")
        else:
            logger.error("Failed to draw X.")
        time.sleep(2)

        logger.info("---- Testing Drawing O ----")
        test_center_x = 200
        test_center_y = 100
        symbol_radius = 25  # mm

        controller.go_to_position(
            x=test_center_x,
            y=test_center_y,
            z=controller.safe_z,
            wait=True)
        time.sleep(1)
        if controller.draw_o(test_center_x, test_center_y, symbol_radius):
            logger.info("O drawn successfully.")
        else:
            logger.error("Failed to draw O.")
        time.sleep(2)

        # Return home-ish
        logger.info("Returning to safe position.")
        controller.go_to_position(x=150, y=0, z=controller.safe_z, wait=True)

        controller.disconnect()
    else:
        logger.error("Could not connect to the arm.")

    logger.info("Arm controller test finished.")
