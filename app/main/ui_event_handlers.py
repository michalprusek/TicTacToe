"""
UI Event Handlers module for TicTacToe application.
This module handles user interface events and interactions.
Consolidates functionality from event_handlers.py.
"""
# pylint: disable=line-too-long,reimported,wrong-import-position,wrong-import-order
# pylint: disable=ungrouped-imports,broad-exception-caught,protected-access,too-many-branches
# pylint: disable=consider-using-with,too-many-nested-blocks,unused-import

import logging
import os
import subprocess
import sys

from PyQt5.QtCore import QObject, QTimer  # pylint: disable=no-name-in-module
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main.debug_window import DebugWindow
from app.main import game_logic
from app.core.arm_thread import ArmCommand
from app.main.game_utils import setup_logger


class UIEventHandlers(QObject):
    """Handles UI events and user interactions."""

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.logger = setup_logger(__name__)

        # Debug window reference
        self.debug_window = None



    def handle_cell_clicked(self, row, col):
        """Handle cell click from board widget."""
        self.logger.info("Cell clicked: (%s, %s)", row, col)

        # Delegate to game controller
        if hasattr(self.main_window, 'game_controller'):
            self.main_window.game_controller.handle_cell_clicked(row, col)

    def handle_reset_button_click(self):
        """Handle reset button click."""
        self.logger.info("Reset button clicked.")

        # Delegate to game controller
        if hasattr(self.main_window, 'game_controller'):
            self.main_window.game_controller.reset_game()

    def handle_debug_button_click(self):
        """Handle debug button click."""
        self.logger.info("Debug button clicked.")
        self.show_debug_window()

    def handle_difficulty_changed(self, value):
        """Handle difficulty slider change."""
        # Update strategy selector
        if hasattr(self.main_window, 'game_controller') and self.main_window.game_controller.strategy_selector:
            # Set difficulty directly (0-10), which internally converts to probability (0.0-1.0)
            self.main_window.game_controller.strategy_selector.difficulty = value
            new_p = self.main_window.game_controller.strategy_selector.p
            self.logger.info("Difficulty changed to %s/10 -> p=%.2f", value, new_p)

            # Log strategy distribution for verification
            if value == 10:
                self.logger.info("Difficulty 10: AI will always use optimal strategy (minimax)")
            elif value == 1:
                self.logger.info("Difficulty 1: AI will mostly use random strategy")
            else:
                self.logger.info("Difficulty %d: AI will use optimal strategy %.0f%% of the time", value, new_p*100)

    def change_language(self):
        """Handle language change button click."""
        if hasattr(self.main_window, 'status_manager'):
            self.main_window.status_manager.toggle_language()
            self._update_ui_texts()

        self.logger.info("Language changed")

    def show_debug_window(self):
        """Show or toggle debug window."""
        if not self.debug_window:
            self.debug_window = DebugWindow(config=self.main_window.config, parent=self.main_window)

            # Connect signals for debug window
            if hasattr(self.main_window, 'camera_controller') and self.main_window.camera_controller:
                # Connect FPS updates
                self.main_window.camera_controller.fps_updated.connect(
                    self.debug_window.update_fps
                )

                # Connect frame updates to debug window camera view
                self.main_window.camera_controller.frame_ready.connect(
                    self.debug_window.update_camera_view
                )

                # Connect camera restart signal
                if hasattr(self.debug_window, 'camera_index_changed'):
                    self.debug_window.camera_index_changed.connect(
                        self.main_window.camera_controller.restart_camera
                    )

                # Connect detection threshold changes
                if hasattr(self.debug_window, 'detection_threshold_changed'):
                    self.debug_window.detection_threshold_changed.connect(
                        self.main_window.camera_controller.set_detection_threshold
                    )

            # Connect arm control signals
            if hasattr(self.main_window, 'arm_controller') and self.main_window.arm_controller:
                if hasattr(self.debug_window, 'calibrate_arm_requested'):
                    self.debug_window.calibrate_arm_requested.connect(
                        self.main_window.arm_controller.calibrate_arm
                    )

                if hasattr(self.debug_window, 'park_arm_requested'):
                    self.debug_window.park_arm_requested.connect(
                        self.main_window.arm_controller.park_arm
                    )

        # Show debug window
        self.debug_window.show()
        self.debug_window.raise_()
        self.debug_window.activateWindow()

        self.logger.info("Debug window shown")



    def _update_ui_texts(self):
        """Update UI texts after language change."""
        try:
            # Update button texts
            if hasattr(self.main_window, 'reset_button'):
                self.main_window.reset_button.setText(
                    self.main_window.status_manager.tr("new_game")
                )

            if hasattr(self.main_window, 'debug_button'):
                self.main_window.debug_button.setText(
                    self.main_window.status_manager.tr("debug")
                )



            if hasattr(self.main_window, 'difficulty_label'):
                self.main_window.difficulty_label.setText(
                    self.main_window.status_manager.tr("difficulty")
                )

            # Update language button flag
            if hasattr(self.main_window, 'language_button'):
                if self.main_window.status_manager.is_czech:
                    self.main_window.language_button.setText("🇨🇿")
                else:
                    self.main_window.language_button.setText("🇺🇸")

            self.logger.info("UI texts updated")

        except Exception as e:
            self.logger.error("Error updating UI texts: %s", e)

    def handle_calibrate_button_click(self):
        """Handle calibrate button click."""
        self.logger.info("Calibrate button clicked.")

        if hasattr(self.main_window, 'arm_controller'):
            self.main_window.arm_controller.calibrate_arm()

    def handle_park_button_click(self):
        """Handle park button click."""
        self.logger.info("Park button clicked.")

        if hasattr(self.main_window, 'arm_controller'):
            self.main_window.arm_controller.park_arm()

    def handle_camera_index_changed(self, new_index):
        """Handle camera index change."""
        self.logger.info("Camera index changed to %s", new_index)

        if hasattr(self.main_window, 'camera_controller'):
            self.main_window.camera_controller.restart_camera(new_index)

    def handle_detection_threshold_changed(self, threshold):
        """Handle detection threshold change."""
        self.logger.info("Detection threshold changed to %s", threshold)

        if hasattr(self.main_window, 'camera_controller'):
            self.main_window.camera_controller.set_detection_threshold(threshold)

    def handle_window_close(self):
        """Handle main window close event."""
        self.logger.info("Main window closing")

        # Stop camera
        if hasattr(self.main_window, 'camera_controller'):
            self.main_window.camera_controller.stop()

        # Close debug window
        if self.debug_window:
            self.debug_window.close()



    def set_difficulty(self, difficulty):
        """Set difficulty value."""
        if hasattr(self.main_window, 'difficulty_slider'):
            self.main_window.difficulty_slider.setValue(difficulty)
        # The slider change will trigger handle_difficulty_changed

    # === Consolidated functions from event_handlers.py ===

    def handle_detected_game_state_extended(self, flat_board):
        """Extended detected game state handling (consolidated from event_handlers.py)."""
        # Convert flat list back to 2D board
        detected_board = [
            [flat_board[i*3 + j] for j in range(3)]
            for i in range(3)
        ]

        # DEBUG: Log what YOLO actually detected
        non_empty_symbols = []
        for r in range(3):
            for c in range(3):
                if detected_board[r][c] != ' ':
                    non_empty_symbols.append(f"({r},{c})={detected_board[r][c]}")

        if non_empty_symbols:
            self.logger.info("🔍 YOLO DETECTION: %s", ', '.join(non_empty_symbols))
        else:
            self.logger.debug("🔍 YOLO DETECTION: Empty board")

        # If game is over, ignore detection
        if hasattr(self.main_window, 'game_controller') and self.main_window.game_controller.game_over:
            return

        # If we're waiting for detection after arm move
        if (hasattr(self.main_window, 'game_controller') and
            self.main_window.game_controller.waiting_for_detection):

            gc = self.main_window.game_controller
            if (gc.ai_move_row is not None and gc.ai_move_col is not None and
                gc.expected_symbol is not None):

                expected_row = gc.ai_move_row
                expected_col = gc.ai_move_col
                expected_symbol = gc.expected_symbol

                if detected_board[expected_row][expected_col] == expected_symbol:
                    self.logger.info("✅ EXPECTED symbol %s detected at (%s, %s)", expected_symbol, expected_row, expected_col)

                    # Successfully detected arm move - reset detection flags
                    gc.waiting_for_detection = False
                    gc.arm_move_in_progress = False
                    gc.ai_move_row = None
                    gc.ai_move_col = None
                    gc.expected_symbol = None
                    gc.detection_wait_time = 0.0
                    gc.ai_move_retry_count = 0

                    # Switch turn to human
                    gc.current_turn = gc.human_player

                    # Update board with detected state
                    if hasattr(self.main_window, 'board_widget'):
                        self.main_window.board_widget.board = [row[:] for row in detected_board]
                        self.main_window.board_widget.update()
                        self.logger.info("GUI updated with detected board state")

                    # Check for game end, then continue to human turn if game not over
                    gc._check_game_end()
                    if not gc.game_over:
                        gc.status_changed.emit("your_turn", True)
                else:
                    self.logger.warning("❌ Expected %s at (%s, %s) but detected: %s", expected_symbol, expected_row, expected_col, detected_board[expected_row][expected_col])

                    # Reset detection flags
                    gc.waiting_for_detection = False
                    gc.ai_move_row = None
                    gc.ai_move_col = None

                    # Check for game end and continue
                    gc._check_game_end()
                    if not gc.game_over:
                        gc.current_turn = gc.human_player
                        gc.status_changed.emit("your_turn", True)

    def handle_arm_park(self):
        """Handle arm park (consolidated from event_handlers.py)."""
        self.logger.info("Park arm requested")
        if hasattr(self.main_window, 'arm_controller'):
            self.main_window.arm_controller.park_arm()

    def handle_arm_calibrate_extended(self):
        """Extended arm calibration (consolidated from event_handlers.py)."""
        self.logger.info("Calibrate arm requested")

        # Get the path to the calibration script
        calibration_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "app", "calibration", "calibration.py")

        # Check if the script exists
        if not os.path.exists(calibration_script):
            self.logger.error("Calibration script not found: %s", calibration_script)
            return

        # Launch the calibration script in a new process
        try:
            subprocess.Popen([sys.executable, calibration_script])
            self.logger.info("Calibration script launched")
        except Exception as e:
            self.logger.error("Failed to launch calibration script: %s", e)

    def handle_track_grid_center(self):
        """Track grid center (consolidated from event_handlers.py)."""
        if not hasattr(self, 'tracking_enabled') or not self.tracking_enabled:
            return

        # Get latest game state from detection
        if (hasattr(self.main_window, 'camera_controller') and
            hasattr(self.main_window.camera_controller, 'camera_thread')):

            camera_thread = self.main_window.camera_controller.camera_thread
            if hasattr(camera_thread, 'detection_thread'):
                detection_thread = camera_thread.detection_thread
                if hasattr(detection_thread, 'latest_game_state'):
                    game_state = detection_thread.latest_game_state

                    if (game_state and hasattr(game_state, 'is_valid') and
                        game_state.is_valid() and
                        hasattr(game_state, 'is_physical_grid_valid') and
                        game_state.is_physical_grid_valid()):

                        # Get cell centers
                        if hasattr(game_state, 'get_cell_centers_uv_transformed'):
                            cell_centers = game_state.get_cell_centers_uv_transformed()

                            if cell_centers is not None and len(cell_centers) == 9:
                                # Calculate center of the grid (center cell)
                                center_cell = cell_centers[4]  # Middle cell (index 4)

                                # Send command to arm to track this position
                                if hasattr(self.main_window, 'arm_controller'):
                                    # Track position command would go here
                                    self.logger.debug("Tracking grid center at %s", center_cell)

    def update_camera_view_extended(self, frame):
        """Extended camera view update (consolidated from event_handlers.py)."""
        # Update main camera view if it exists
        if hasattr(self.main_window, 'camera_view') and self.main_window.camera_view:
            self.main_window.camera_view.update_frame(frame)

        # Update debug window camera view if it exists
        if self.debug_window and hasattr(self.debug_window, 'camera_view'):
            self.debug_window.camera_view.update_frame(frame)

    def update_fps_display_extended(self, fps):
        """Extended FPS display update (consolidated from event_handlers.py)."""
        # Update debug window FPS display if it exists
        if self.debug_window and hasattr(self.debug_window, 'update_fps'):
            self.debug_window.update_fps(fps)
