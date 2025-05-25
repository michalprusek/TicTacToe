"""
Event handlers for the TicTacToe application.
"""
import logging
import subprocess
import os
import sys
import time
from PyQt5.QtCore import QTimer

from app.main import game_logic
from app.core.arm_thread import ArmCommand


class GameEventHandler:
    """Handles game-related events"""
    
    def __init__(self, app):
        """Initialize with reference to the main app"""
        self.app = app
        self.logger = logging.getLogger(__name__)
    
    def handle_cell_clicked(self, row, col):
        """Handle cell click event"""
        # Ignore clicks if game is over or it's not player's turn
        if self.app.game_over or self.app.current_turn != self.app.human_player:
            return
            
        # Ignore clicks on non-empty cells
        if self.app.board_widget.board[row][col] != game_logic.EMPTY:
            return
            
        # Update board with player's move
        self.app.board_widget.board[row][col] = self.app.human_player
        self.app.board_widget.update()
        
        # Increment move counter
        self.app.move_counter += 1
        
        # Check for game end
        self.app.check_game_end()
        
        # If game is not over, make AI move
        if not self.app.game_over:
            self.app.current_turn = self.app.ai_player
            
            # Update status
            self.app.update_status(self.app.tr("ai_turn"))
            self.app.main_status_panel.setStyleSheet("""
                background-color: #e74c3c;
                border-radius: 10px;
                border: 2px solid #c0392b;
            """)
            
            # Make AI move after a short delay
            QTimer.singleShot(500, self.app.make_ai_move)
    
    def handle_reset_button_click(self):
        """Handle reset button click event"""
        self.logger.info("Reset button clicked")
        
        # Reset game state
        self.app.reset_game()
        
        # Update status
        self.app.update_status(self.app.tr("game_reset"))
        self.app.main_status_panel.setStyleSheet("""
            background-color: #3498db;
            border-radius: 10px;
            border: 2px solid #2980b9;
        """)
        
        # Reset arm to neutral position if connected
        if hasattr(self.app, 'arm_thread') and self.app.arm_thread:
            self.app.arm_thread.add_command(ArmCommand.RESET_POSITION)
    
    def handle_difficulty_changed(self, value):
        """Handle difficulty slider change event"""
        self.logger.info(f"Difficulty changed to {value}")
        
        # Update difficulty value label
        if hasattr(self.app, 'difficulty_value_label'):
            self.app.difficulty_value_label.setText(str(value))
        
        # Update strategy selector difficulty
        if hasattr(self.app, 'strategy_selector'):
            self.app.strategy_selector.set_difficulty(value / 10.0)
    
    def handle_detected_game_state(self, flat_board):
        """Handle detected game state from camera"""
        # Convert flat list back to 2D board
        detected_board = [
            [flat_board[i*3 + j] for j in range(3)]
            for i in range(3)
        ]
        
        # If game is over, ignore detection
        if self.app.game_over:
            return
            
        # If we're waiting for detection after arm move
        if self.app.waiting_for_detection:
            # Check if the expected symbol is detected at the expected position
            if (self.app.ai_move_row is not None and 
                self.app.ai_move_col is not None and 
                self.app.arm_player_symbol is not None):
                
                expected_row = self.app.ai_move_row
                expected_col = self.app.ai_move_col
                expected_symbol = self.app.arm_player_symbol
                
                if detected_board[expected_row][expected_col] == expected_symbol:
                    self.logger.info(f"Symbol {expected_symbol} detected at ({expected_row}, {expected_col})")
                    
                    # Update board
                    self.app.board_widget.board = [row[:] for row in detected_board]
                    self.app.board_widget.update()
                    
                    # Reset detection flags
                    self.app.waiting_for_detection = False
                    self.app.ai_move_row = None
                    self.app.ai_move_col = None
                    
                    # Check for game end
                    self.app.check_game_end()
                    
                    # If game is not over, pass turn to player
                    if not self.app.game_over:
                        self.app.current_turn = self.app.human_player
                        self.app.update_status(self.app.tr("your_turn"))
        
        # If it's player's turn, check for changes on the board
        elif self.app.current_turn == self.app.human_player:
            # Compare current board with detected board
            diff = game_logic.get_board_diff(
                self.app.board_widget.board, detected_board)
            
            # If there's exactly one change and it's the player's symbol, update
            if len(diff) == 1:
                r, c, symbol = diff[0]
                if symbol == self.app.human_player:
                    # Update board
                    self.app.board_widget.board = [row[:] for row in detected_board]
                    self.app.board_widget.update()
                    
                    # Increment move counter
                    self.app.move_counter += 1
                    
                    # Check for game end
                    self.app.check_game_end()
                    
                    # If game is not over, make AI move
                    if not self.app.game_over:
                        self.app.current_turn = self.app.ai_player
                        self.app.update_status(self.app.tr("ai_turn"))
                        
                        # Make AI move after a short delay
                        QTimer.singleShot(500, self.app.make_ai_move)


class ArmEventHandler:
    """Handles arm-related events"""
    
    def __init__(self, app):
        """Initialize with reference to the main app"""
        self.app = app
        self.logger = logging.getLogger(__name__)
    
    def handle_park_button_click(self):
        """Handle park button click event"""
        self.logger.info("Park button clicked")
        
        if hasattr(self.app, 'arm_thread') and self.app.arm_thread:
            self.app.arm_thread.add_command(ArmCommand.PARK)
            self.app.update_status(self.app.tr("arm_parking"))
    
    def handle_calibrate_button_click(self):
        """Handle calibrate button click event"""
        self.logger.info("Calibrate button clicked")
        
        # Get the path to the calibration script
        calibration_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "app", "calibration", "calibration.py")
        
        # Check if the script exists
        if not os.path.exists(calibration_script):
            self.logger.error(f"Calibration script not found: {calibration_script}")
            return
        
        # Launch the calibration script in a new process
        try:
            subprocess.Popen([sys.executable, calibration_script])
            self.logger.info("Calibration script launched")
        except Exception as e:
            self.logger.error(f"Failed to launch calibration script: {e}")
    
    def handle_track_checkbox_changed(self, state):
        """Handle track checkbox state change event"""
        self.logger.info(f"Track checkbox state changed to {state}")
        
        # Update tracking state
        self.app.tracking_enabled = (state == 2)  # Qt.Checked = 2
        
        if self.app.tracking_enabled:
            # Pause the game while tracking
            self.app.game_paused = True
            
            # Start tracking timer
            self.app.tracking_timer.start(self.app.tracking_interval)
            
            # Update status
            self.app.update_status(self.app.tr("tracking"))
            self.app.main_status_panel.setStyleSheet("""
                background-color: #f39c12;
                border-radius: 10px;
                border: 2px solid #d35400;
            """)
        else:
            # Stop tracking timer
            self.app.tracking_timer.stop()
            
            # Resume the game
            self.app.game_paused = False
            
            # Update status based on current turn
            if self.app.current_turn == self.app.human_player:
                self.app.update_status(self.app.tr("your_turn"))
            else:
                self.app.update_status(self.app.tr("ai_turn"))
    
    def track_grid_center(self):
        """Track the center of the grid with the arm"""
        if not self.app.tracking_enabled or not hasattr(self.app, 'arm_thread'):
            return
            
        # Get the latest game state from detection thread
        if hasattr(self.app, 'camera_thread') and self.app.camera_thread:
            detection_thread = getattr(self.app.camera_thread, 'detection_thread', None)
            if detection_thread:
                _, game_state = detection_thread.get_latest_result()
                
                if game_state and game_state.is_valid() and game_state.is_physical_grid_valid():
                    # Get cell centers
                    cell_centers = game_state.get_cell_centers_uv_transformed()
                    
                    if cell_centers is not None and len(cell_centers) == 9:
                        # Calculate center of the grid (center cell)
                        center_cell = cell_centers[4]  # Middle cell (index 4)
                        
                        # Send command to arm to move to this position
                        self.app.arm_thread.add_command(
                            ArmCommand.TRACK_POSITION,
                            {"uv": (int(center_cell[0]), int(center_cell[1]))})


class UIEventHandler:
    """Handles UI-related events"""
    
    def __init__(self, app):
        """Initialize with reference to the main app"""
        self.app = app
        self.logger = logging.getLogger(__name__)
    
    def handle_debug_button_click(self):
        """Handle debug button click event"""
        self.logger.info("Debug button clicked")
        
        # Show debug window if it exists
        if hasattr(self.app, 'debug_window') and self.app.debug_window:
            self.app.debug_window.show()
    
    def handle_language_button_click(self):
        """Handle language button click event"""
        self.logger.info("Language button clicked")
        
        # Toggle language
        if self.app.is_czech:
            self.app.current_language = self.app.LANG_EN
            self.app.is_czech = False
        else:
            self.app.current_language = self.app.LANG_CS
            self.app.is_czech = True
        
        # Update UI text
        self.app.update_ui_text()
    
    def update_camera_view(self, frame):
        """Update camera view with the latest frame"""
        # Update main camera view if it exists
        if hasattr(self.app, 'camera_view') and self.app.camera_view:
            self.app.camera_view.update_frame(frame)
        
        # Update debug window camera view if it exists
        if hasattr(self.app, 'debug_window') and self.app.debug_window:
            self.app.debug_window.update_frame(frame)
    
    def update_fps_display(self, fps):
        """Update FPS display"""
        # Update debug window FPS display if it exists
        if hasattr(self.app, 'debug_window') and self.app.debug_window:
            self.app.debug_window.update_fps(fps)
