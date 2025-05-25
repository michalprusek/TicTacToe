"""
Arm control thread module for TicTacToe application.
"""
import time
import logging
import threading
import queue
from typing import Optional, Tuple, Dict, Any, List

from app.main.arm_controller import ArmController


class ArmCommand:
    """Command for the arm controller."""

    def __init__(self, command_type: str, params: Dict[str, Any] = None):
        """Initialize the arm command.

        Args:
            command_type: Type of command (e.g., 'draw_x', 'draw_o', 'move')
            params: Parameters for the command
        """
        self.command_type = command_type
        self.params = params or {}
        self.completed = threading.Event()
        self.success = False
        self.result = None

    def mark_completed(self, success: bool, result: Any = None):
        """Mark the command as completed.

        Args:
            success: Whether the command was successful
            result: Result of the command
        """
        self.success = success
        self.result = result
        self.completed.set()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for the command to complete.

        Args:
            timeout: Timeout in seconds

        Returns:
            Whether the command completed successfully
        """
        if self.completed.wait(timeout):
            return self.success
        return False


class ArmThread(threading.Thread):
    """Thread for controlling the uArm in the background."""

    def __init__(self, port: Optional[str] = None):
        """Initialize the arm thread.

        Args:
            port: Serial port for the uArm
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.port = port

        # Thread control
        self.running = False
        self.daemon = True  # Thread will exit when main program exits

        # Command queue
        self.command_queue = queue.Queue()

        # Arm controller
        self.arm_controller = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to the uArm.

        Returns:
            Whether the connection was successful
        """
        if self.connected:
            return True

        command = ArmCommand('connect')
        self.command_queue.put(command)
        return command.wait_for_completion(timeout=10)

    def disconnect(self) -> bool:
        """Disconnect from the uArm.

        Returns:
            Whether the disconnection was successful
        """
        if not self.connected:
            return True

        command = ArmCommand('disconnect')
        self.command_queue.put(command)
        return command.wait_for_completion(timeout=5)

    def draw_x(self, center_x: float, center_y: float, size: float,
               speed: Optional[int] = None) -> bool:
        """Draw an X symbol.

        Args:
            center_x: X coordinate of the center
            center_y: Y coordinate of the center
            size: Size of the X
            speed: Drawing speed

        Returns:
            Whether the command was successful
        """
        if not self.connected:
            self.logger.error("Cannot draw X: Arm not connected")
            return False

        command = ArmCommand('draw_x', {
            'center_x': center_x,
            'center_y': center_y,
            'size': size,
            'speed': speed
        })
        self.command_queue.put(command)
        return command.wait_for_completion()

    def draw_o(self, center_x: float, center_y: float, radius: float,
               speed: Optional[int] = None, segments: int = 16) -> bool:
        """Draw an O symbol.

        Args:
            center_x: X coordinate of the center
            center_y: Y coordinate of the center
            radius: Radius of the O
            speed: Drawing speed
            segments: Number of segments for the circle

        Returns:
            Whether the command was successful
        """
        if not self.connected:
            self.logger.error("Cannot draw O: Arm not connected")
            return False

        command = ArmCommand('draw_o', {
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'speed': speed,
            'segments': segments
        })
        self.command_queue.put(command)
        return command.wait_for_completion()

    def go_to_position(self, x: Optional[float] = None, y: Optional[float] = None,
                      z: Optional[float] = None, speed: Optional[int] = None,
                      wait: bool = True) -> bool:
        """Move the arm to a position.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            speed: Movement speed
            wait: Whether to wait for the movement to complete

        Returns:
            Whether the command was successful
        """
        if not self.connected:
            self.logger.error("Cannot move: Arm not connected")
            return False

        command = ArmCommand('move', {
            'x': x,
            'y': y,
            'z': z,
            'speed': speed,
            'wait': wait
        })
        self.command_queue.put(command)

        if wait:
            return command.wait_for_completion()
        else:
            # For non-waiting commands, return True immediately
            return True

    def get_position(self, cached: bool = True) -> Optional[Tuple[float, float, float]]:
        """Get the current position of the arm.

        Args:
            cached: Whether to use cached position

        Returns:
            Tuple of (x, y, z) coordinates or None if not available
        """
        if not self.connected:
            self.logger.error("Cannot get position: Arm not connected")
            return None

        command = ArmCommand('get_position', {'cached': cached})
        self.command_queue.put(command)

        if command.wait_for_completion(timeout=2):
            return command.result
        return None

    def run(self):
        """Main thread loop."""
        self.running = True

        # Initialize arm controller
        self.arm_controller = ArmController(port=self.port)

        # Main command processing loop
        while self.running:
            try:
                # Get command from queue with timeout to allow checking running flag
                try:
                    command = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process command
                if command.command_type == 'connect':
                    success = self.arm_controller.connect()
                    self.connected = success
                    command.mark_completed(success)

                elif command.command_type == 'disconnect':
                    success = self.arm_controller.disconnect()
                    self.connected = False
                    command.mark_completed(success)

                elif command.command_type == 'draw_x':
                    success = self.arm_controller.draw_x(
                        command.params['center_x'],
                        command.params['center_y'],
                        command.params['size'],
                        command.params.get('speed')
                    )
                    command.mark_completed(success)

                elif command.command_type == 'draw_o':
                    success = self.arm_controller.draw_o(
                        command.params['center_x'],
                        command.params['center_y'],
                        command.params['radius'],
                        command.params.get('speed'),
                        command.params.get('segments', 16)
                    )
                    command.mark_completed(success)

                elif command.command_type == 'move':
                    success = self.arm_controller.go_to_position(
                        x=command.params.get('x'),
                        y=command.params.get('y'),
                        z=command.params.get('z'),
                        speed=command.params.get('speed'),
                        wait=command.params.get('wait', True)
                    )
                    command.mark_completed(success)

                elif command.command_type == 'get_position':
                    result = self.arm_controller.get_position(
                        cached=command.params.get('cached', True)
                    )
                    command.mark_completed(result is not None, result)

                else:
                    self.logger.warning(f"Unknown command type: {command.command_type}")
                    command.mark_completed(False)

                # Mark task as done in the queue
                self.command_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing arm command: {e}")
                if 'command' in locals():
                    command.mark_completed(False)
                    self.command_queue.task_done()

        # Clean up
        if self.arm_controller and self.connected:
            self.arm_controller.disconnect()

    def stop_current_move(self):
        """Stop any current arm movement immediately."""
        if self.arm_controller and self.connected:
            try:
                # Clear the command queue to stop pending moves
                while not self.command_queue.empty():
                    try:
                        command = self.command_queue.get_nowait()
                        command.mark_completed(False)
                        self.command_queue.task_done()
                    except queue.Empty:
                        break
                self.logger.info("🛑 Cleared arm command queue and stopped current moves")
            except Exception as e:
                self.logger.error(f"Error stopping current arm move: {e}")

    def stop(self):
        """Stop the arm thread."""
        self.running = False
        self.join(timeout=5.0)  # Wait for thread to finish
        self.logger.info("Arm thread stopped")
