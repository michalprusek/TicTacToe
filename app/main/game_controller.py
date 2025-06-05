# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Game Controller module for TicTacToe application.
This module handles game logic, state management, and turn coordination.
Consolidates functionality from game_manager.py.
"""

import random
import time

from PyQt5.QtCore import QObject  # pylint: disable=no-name-in-module
# pylint: disable=no-name-in-module
from PyQt5.QtCore import QTimer
# pylint: disable=no-name-in-module
from PyQt5.QtCore import pyqtSignal

from app.core.strategy import BernoulliStrategySelector
from app.main import game_logic
from app.main.constants import DEFAULT_DIFFICULTY
from app.main.game_utils import convert_board_1d_to_2d
from app.main.game_utils import get_board_symbol_counts
from app.main.game_utils import setup_logger
from app.main.path_utils import setup_project_path

# Setup project path
setup_project_path()


class GameController(QObject):  # pylint: disable=too-many-instance-attributes
    """Controls game logic, state management, and turn coordination."""

    # Signals
    status_changed = pyqtSignal(str, bool)  # message, is_key
    game_ended = pyqtSignal(str)  # winner

    def __init__(self, main_window, config):
        super().__init__()

        self.main_window = main_window
        self.config = config
        self.logger = setup_logger(__name__)

        # Game state attributes
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = game_logic.PLAYER_X
        self.game_over = False
        self.winner = None
        self.move_counter = 0

        # UNIFIED BOARD STATE - single source of truth from camera detection
        self.authoritative_board = game_logic.create_board()  # This is the ONE board state

        # Arm control flags
        self.waiting_for_detection = False
        self.arm_move_in_progress = False
        self.waiting_for_arm_completion = False  # NEW: Track when waiting for arm to reach neutral
        self.robot_status_displayed = False  # NEW: Track if "ROBOT HRAJE" status is displayed
        self.last_arm_move_time = 0
        self.arm_move_cooldown = 3.0

        # Detection retry logic
        self.ai_move_row = None
        self.ai_move_col = None
        self.expected_symbol = None
        self.ai_move_retry_count = 0
        self.max_retry_count = 2
        self.detection_wait_time = 0.0
        self.max_detection_wait_time = 5.0

        # Strategy selector - set to maximum difficulty by default
        difficulty = DEFAULT_DIFFICULTY  # Now defaults to 10
        if hasattr(config, 'game') and hasattr(config.game, 'default_difficulty'):
            difficulty = config.game.default_difficulty
        self.strategy_selector = BernoulliStrategySelector(difficulty=difficulty)
        self.logger.info(
            "Strategy selector initialized with difficulty %s (p=%.2f)",
            difficulty, difficulty / 10.0
        )

        # Cache status tracking
        self.last_cache_status = False  # Track if we're using cached data

        # Arm controller reference (set by main window)
        self.arm_controller = None

    def set_arm_controller(self, arm_controller):
        """Set the arm controller reference."""
        self.arm_controller = arm_controller

    def handle_arm_turn_completed(self, success):
        """Handle arm turn completion - only emit your_turn after neutral position reached."""
        # Reset the waiting flag
        self.waiting_for_arm_completion = False

        # Reset robot status flag so "ROBOT HRAJE" can be displayed again next turn
        self.robot_status_displayed = False

        if success:
            self.logger.info("🎯 ARM TURN COMPLETED: Arm returned to neutral, emitting your_turn")
            # Only now emit the your_turn signal - arm is safely at neutral position
            self.status_changed.emit("your_turn", True)
        else:
            self.logger.error("❌ ARM TURN FAILED: Arm failed to return to neutral position")
            # Still emit your_turn to prevent game from getting stuck
            self.status_changed.emit("your_turn", True)

    def start_game(self):
        """Start a new game."""
        self.reset_game()

    def reset_game(self):
        """Reset the game to initial state with complete hard reset."""
        self.logger.info("HARD RESET: Performing complete game reset.")

        # HARD RESET: Reset authoritative board state
        self.authoritative_board = game_logic.create_board()

        # HARD RESET: Reset board widget completely
        if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
            empty_board = game_logic.create_board()
            self.main_window.board_widget.board = empty_board  # Direct board assignment
            self.main_window.board_widget.update_board(empty_board, None, highlight_changes=False)
            self.main_window.board_widget.winning_line = None
            self.main_window.board_widget.last_board = None  # Clear cached board
            self.main_window.board_widget.update()
            self.logger.info("HARD RESET: Board widget completely reset")

        # HARD RESET: Reset all game state variables
        self.game_over = False
        self.winner = None
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = game_logic.PLAYER_X
        self.move_counter = 0

        # HARD RESET: Reset arm flags and state
        self.waiting_for_detection = False
        self.arm_move_in_progress = False
        self.last_arm_move_time = 0

        # HARD RESET: Reset detection retry logic completely
        self.ai_move_row = None
        self.ai_move_col = None
        self.expected_symbol = None
        self.ai_move_retry_count = 0
        self.detection_wait_time = 0.0

        # HARD RESET: Reset celebration trigger
        if hasattr(self.main_window, '_celebration_triggered'):
            delattr(self.main_window, '_celebration_triggered')

        # HARD RESET: Reset core game state in detection system
        self._reset_detection_system_state()

        # HARD RESET: Force clear any cached detection results
        self._force_clear_detection_cache()

        self.status_changed.emit("new_game_detected", True)
        self.logger.info("HARD RESET: Game reset complete. Current turn: %s", self.current_turn)

        # CLEANUP: Immediate status update instead of 2-second delay
        self.status_changed.emit("your_turn", True)

        # Move arm to neutral position
        if self.arm_controller:
            self.arm_controller.move_to_neutral_position()

    def _reset_detection_system_state(self):
        """Reset the core game state in the detection system."""
        try:
            # Reset camera thread's board state tracking
            if (hasattr(self.main_window, 'camera_controller') and
                self.main_window.camera_controller and
                hasattr(self.main_window.camera_controller, 'camera_thread')):
                camera_thread = self.main_window.camera_controller.camera_thread
                if camera_thread:
                    camera_thread.last_board_state = None
                    camera_thread.last_board_update_time = 0
                    self.logger.info("HARD RESET: Camera thread state reset")

            # Reset detection thread's game state
            if (hasattr(self.main_window, 'camera_controller') and
                self.main_window.camera_controller and
                hasattr(self.main_window.camera_controller, 'camera_thread') and
                self.main_window.camera_controller.camera_thread and
                hasattr(self.main_window.camera_controller.camera_thread, 'detection_thread')):
                detection_thread = self.main_window.camera_controller.camera_thread.detection_thread
                if detection_thread and hasattr(detection_thread, 'detector'):
                    detector = detection_thread.detector
                    if detector and hasattr(detector, 'game_state_manager'):
                        detector.game_state_manager.game_state.reset_game()
                        self.logger.info("HARD RESET: Detection system game state reset")
        except Exception as e:
            self.logger.warning("Error resetting detection system state: %s", e)

    def _force_clear_detection_cache(self):
        """Force clear any cached detection results."""
        try:
            # Clear detection thread cache
            if (hasattr(self.main_window, 'camera_controller') and
                self.main_window.camera_controller and
                hasattr(self.main_window.camera_controller, 'camera_thread') and
                self.main_window.camera_controller.camera_thread and
                hasattr(self.main_window.camera_controller.camera_thread, 'detection_thread')):
                detection_thread = self.main_window.camera_controller.camera_thread.detection_thread
                if detection_thread:
                    with detection_thread.result_lock:
                        detection_thread.latest_result = None
                        detection_thread.latest_game_state = None
                    self.logger.info("HARD RESET: Detection cache cleared")
        except Exception as e:
            self.logger.warning("Error clearing detection cache: %s", e)

    def _is_board_empty(self, board):
        """Check if the board is completely empty with robust detection."""
        if not board:
            return True

        # Count non-empty cells
        non_empty_count = 0
        for row in board:
            for cell in row:
                if cell != game_logic.EMPTY and cell.strip():  # Also check for whitespace
                    non_empty_count += 1

        # Board is considered empty if it has 0 symbols
        is_empty = non_empty_count == 0

        if is_empty:
            self.logger.debug("EMPTY BOARD DETECTED: No symbols found on board")
        else:
            self.logger.debug("BOARD NOT EMPTY: Found %d symbols", non_empty_count)

        return is_empty

    def handle_cell_clicked(self, row, col):
        """Handle cell click from the board widget."""
        if self.game_over or self.current_turn != self.human_player:
            self.logger.warning("Click ignored: game_over=%s, current_turn=%s",
                                self.game_over, self.current_turn)
            return

        if self.arm_move_in_progress or self.waiting_for_detection:
            arm_busy = self.arm_move_in_progress or self.waiting_for_detection
            self.logger.warning("Click ignored: arm_busy=%s", arm_busy)
            return

        # Check if cell is empty
        if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
            if self.main_window.board_widget.board[row][col] != game_logic.EMPTY:
                self.logger.info("Cell is already occupied.")
                return

        # Human player intends to move - wait for camera detection
        self.logger.info(
            "Player (%s) intends to move to (%s,%s). Waiting for detection.",
            self.human_player, row, col)
        self.status_changed.emit("waiting_detection", True)

    def handle_detected_game_state(self, detected_board_from_camera):
        """Handle detected game state from camera."""
        if detected_board_from_camera is None:
            self.logger.debug("Detected empty board (None) from camera.")
            return

        detected_board = convert_board_1d_to_2d(detected_board_from_camera)
        if not detected_board:
            self.logger.warning(
                "Failed to convert detected board to 2D format.")
            return

        # CRITICAL FIX: Update authoritative board state from camera detection
        self.authoritative_board = [row[:] for row in detected_board]  # Deep copy
        self.logger.debug(
            "Updated authoritative board from camera: %s", self.authoritative_board)

        # Update board widget with detected board state
        if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
            self.main_window.board_widget.update_board(detected_board, None, highlight_changes=True)
            self.logger.debug(
                "Updated GUI board with detected state: %s", detected_board)

        # Handle game over state - improved empty board detection
        if self.game_over:
            is_empty_now = self._is_board_empty(detected_board)
            if is_empty_now:
                self.logger.info(
                    "AUTO NEW GAME: Detected empty board after game end - "
                    "automatically starting new game.")
                self.reset_game()
            return

        # Check for automatic new game detection even during active game
        # This handles cases where user replaces board during game
        if not self.game_over and self._is_board_empty(detected_board):
            # Only reset if we previously had symbols on the board
            if any(cell != game_logic.EMPTY for row in self.authoritative_board
                   for cell in row):
                self.logger.info(
                    "AUTO NEW GAME: Detected empty board during active game - "
                    "user replaced board, starting new game.")
                self.reset_game()
                return

        # Check for new moves and game progression
        self._process_detected_board(detected_board)

    def _process_detected_board(
            self, detected_board):  # pylint: disable=too-many-branches,too-many-statements
        """Process the detected board for game logic."""
        # Count symbols for potential debugging
        self._get_board_symbol_counts(detected_board)

        # Check if it's time for arm to play
        if not self.game_over:
            should_play, arm_symbol_to_play = self._should_arm_play_now(detected_board)
            if should_play and arm_symbol_to_play:
                self.logger.info(
                    "DECISION: Arm should play with symbol %s.", arm_symbol_to_play)
                self.ai_player = arm_symbol_to_play
                self.current_turn = self.ai_player

                # REQUIREMENT: Display "ROBOT HRAJE" immediately when robot's turn is determined
                if not self.robot_status_displayed:
                    self.status_changed.emit("arm_moving", True)
                    self.robot_status_displayed = True
                    self.logger.info("🔄 STATUS: Displaying 'ROBOT HRAJE' - robot turn determined")

                self.make_arm_move_with_symbol(arm_symbol_to_play)
            elif (self.current_turn == self.ai_player and not self.arm_move_in_progress
                  and not self.waiting_for_detection):
                self.status_changed.emit("arm_turn", True)
                print(f"DEBUG: Emitting arm_turn - current_turn={self.current_turn}, "
                      f"ai_player={self.ai_player}")
                self.logger.debug(
                    "Arm is on turn, but conditions for playing are not met.")
            elif (self.current_turn == self.human_player and not self.arm_move_in_progress
                  and not self.waiting_for_detection and not self.waiting_for_arm_completion):
                # REQUIREMENT: Only emit "your_turn" if robot is not currently playing
                # This prevents overriding "ROBOT HRAJE" status during robot's turn
                if not (self.current_turn == self.ai_player or self.arm_move_in_progress or
                       self.waiting_for_detection or self.waiting_for_arm_completion):
                    self.status_changed.emit("your_turn", True)
                    print(f"DEBUG: Emitting your_turn - current_turn={self.current_turn}, "
                          f"human_player={self.human_player}")
                else:
                    self.logger.debug("Skipping your_turn emission - robot is still playing")

    def _should_arm_play_now(self, current_board_state):
        """Determine if the arm should play now."""
        self.logger.debug(
            "Checking if arm should play. InProgress: %s, Cooldown: %s",
            self.arm_move_in_progress,
            time.time() - self.last_arm_move_time < self.arm_move_cooldown)

        if (self.game_over or self.arm_move_in_progress or
                (time.time() - self.last_arm_move_time < self.arm_move_cooldown)):
            return False, None

        # Check grid validity (simplified for now)

        # Count symbols to determine turn
        x_count, o_count, total_count = self._get_board_symbol_counts(current_board_state)

        # Arm plays when odd number of
        # total symbols (human played, now arm's turn)
        if total_count % 2 == 1:
            # Determine which symbol arm should play
            if x_count > o_count:
                return True, game_logic.PLAYER_O
            if o_count > x_count:
                return True, game_logic.PLAYER_X
            return True, game_logic.PLAYER_O  # Default to O

        return False, None

    def make_arm_move_with_symbol(self, symbol_to_play):
        """Make an arm move with the specified symbol."""
        self.logger.info("Starting arm move with symbol: %s", symbol_to_play)

        if self.game_over or self.arm_move_in_progress:
            self.logger.warning(
                "Arm move interrupted: game ended or arm already in progress.")
            return False

        # CRITICAL FIX: Use authoritative board state from camera detection
        current_board_for_strategy = self.authoritative_board

        if not current_board_for_strategy:
            self.logger.error(
                "Cannot get authoritative board state for strategy.")
            return False

        # Log the board state being used for strategy
        self.logger.info(
            "Using authoritative board for strategy: %s", current_board_for_strategy)

        # Set flags BEFORE starting movement
        self.arm_move_in_progress = True
        self.last_arm_move_time = time.time()

        # Get move from strategy using cached board state for occlusion handling
        if hasattr(self.main_window, 'game_state_manager') and self.main_window.game_state_manager:
            ai_board = self.main_window.game_state_manager.game_state.get_board_for_ai()
            move = self.strategy_selector.get_move(ai_board, symbol_to_play)
            self.logger.info("🤖 AI using cached board state for decision making")
        else:
            move = self.strategy_selector.get_move(current_board_for_strategy, symbol_to_play)
        if not move:
            self.logger.error("Strategy did not return a valid move.")
            self.arm_move_in_progress = False
            return False

        row, col = move
        self.logger.info(
            "Strategy selected move: (%s,%s) with symbol %s", row, col, symbol_to_play)

        # Store move for detection verification
        self.ai_move_row, self.ai_move_col = row, col
        self.expected_symbol = symbol_to_play
        self.ai_move_retry_count = 0
        self.detection_wait_time = 0.0

        # Execute arm drawing
        if self.arm_controller and self.arm_controller.draw_ai_symbol(row, col, symbol_to_play):
            self.logger.info(
                "Symbol %s successfully sent for drawing at (%s,%s). "
                "Waiting for detection.", symbol_to_play, row, col)

            self.waiting_for_detection = True
            self.waiting_for_arm_completion = True  # NEW: Wait for arm to reach neutral
            self.logger.info("🔄 TURN SEQUENCE: Waiting for arm to complete and return to neutral")
            return True

        self.logger.error(
            "Failed to start drawing symbol %s at (%s,%s).", symbol_to_play, row, col)
        self.arm_move_in_progress = False
        self.waiting_for_detection = False
        self.waiting_for_arm_completion = False  # Reset arm completion flag
        self.robot_status_displayed = False  # Reset robot status flag
        self.current_turn = self.human_player
        self.status_changed.emit("your_turn", True)  # Emit immediately on failure
        return False

    def update_game_state(
            self):  # pylint: disable=too-many-branches,too-many-statements
        """Update game state machine (called by timer)."""
        if self.game_over:
            return

        # Handle detection waiting
        if self.waiting_for_detection:
            self.detection_wait_time += 0.1  # Timer is every 100ms

            # Check for detection timeout
            if self.detection_wait_time >= self.max_detection_wait_time:
                self.ai_move_retry_count += 1
                self.logger.warning("Detection timeout. Retry %s/%s",
                                    self.ai_move_retry_count, self.max_retry_count)

                if self.ai_move_retry_count >= self.max_retry_count:
                    self.logger.error(
                        "Max retries reached. Giving up on arm move.")
                    self.waiting_for_detection = False
                    self.arm_move_in_progress = False
                    self.waiting_for_arm_completion = False  # Reset arm completion flag
                    self.robot_status_displayed = False  # Reset robot status flag
                    self.current_turn = self.human_player
                    # CLEANUP: Skip detection_failed message, proceed directly to your_turn
                    self.status_changed.emit("your_turn", True)

                # Reset for next attempt
                self.ai_move_row, self.ai_move_col, self.expected_symbol = None, None, None

        # Check for game end
        self._check_game_end()

        # Check cache status and update GUI feedback
        self._check_cache_status()

    def _check_game_end(
            self):  # pylint: disable=too-many-branches,too-many-statements
        """Check if the game has ended."""
        # CRITICAL FIX: Use authoritative board state for game end check
        board_to_check = self.authoritative_board

        if not board_to_check:
            return

        game_logic_winner = game_logic.check_winner(board_to_check)

        if game_logic_winner:
            # Get symbol count for logging and winner determination
            x_count, o_count, total_count = self._get_board_symbol_counts(board_to_check)
            self.logger.info("Symbol count at game end: X=%d, O=%d, Total=%d",
                             x_count, o_count, total_count)

            # For TIE, keep it as is
            if game_logic_winner == game_logic.TIE:
                self.winner = game_logic.TIE
            else:
                # Determine actual winner based on symbol count on board
                # Even number of symbols = arm won (arm moves second)
                # Odd number of symbols = human won (human moves first)
                print(f"DEBUG: Determining winner - total_count={total_count}, "
                      f"game_logic_winner={game_logic_winner}")
                if total_count % 2 == 0:
                    # Even count - arm (AI) won,
                    # but we show it from human perspective
                    self.winner = "ARM_WIN"
                    self.logger.info(
                        "ARM_WIN determined (even count: %s)", total_count)
                    print(f"DEBUG: Set winner to ARM_WIN (even count {total_count})")
                else:
                    # Odd count - human won
                    self.winner = "HUMAN_WIN"
                    self.logger.info(
                        "HUMAN_WIN determined (odd count: %s)", total_count)
                    print(f"DEBUG: Set winner to HUMAN_WIN (odd count {total_count})")

        if self.winner:
            self.game_over = True
            self.arm_move_in_progress = False
            self.waiting_for_detection = False

            # Get symbol count again for logging if not already done
            if 'total_count' not in locals():
                x_count, o_count, total_count = self._get_board_symbol_counts(board_to_check)

            self.logger.info(
                "GAME END! Winner: %s. Move count: %s. Total symbols: %s",
                self.winner, self.move_counter, total_count)

            # Show game end notification
            self._show_game_end_notification()

            # Handle winning line
            if self.winner != game_logic.TIE:
                if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
                    self.main_window.board_widget.winning_line = (
                        game_logic.get_winning_line(board_to_check))
                    self.main_window.board_widget.update()

                # Draw winning line if AI won
                if self.winner == "ARM_WIN" and self.arm_controller:
                    self.logger.info(
                        "AI (arm) won! Planning to draw winning line.")
                    QTimer.singleShot(1500, self.arm_controller.draw_winning_line)

            if self.winner == game_logic.TIE:
                self.status_changed.emit("draw", True)
            elif self.winner == "HUMAN_WIN":
                self.status_changed.emit("win", True)
            elif self.winner == "ARM_WIN":
                self.status_changed.emit("loss", True)

            # Move to neutral position
            if self.arm_controller:
                self.arm_controller.move_to_neutral_position()

        # Check for draw (board full)
        elif self.move_counter >= 9:
            self.game_over = True
            self.winner = game_logic.TIE
            self.logger.info(
                "GAME END! Draw. Move count: %s", self.move_counter)
            self.arm_move_in_progress = False
            self.waiting_for_detection = False
            self._show_game_end_notification()
            self.status_changed.emit("draw", True)

            if self.arm_controller:
                self.arm_controller.move_to_neutral_position()

    def _show_game_end_notification(self):
        """Show game end notification."""
        # Winner is already correctly determined in check_and_handle_game_end
        self.game_ended.emit(str(self.winner))

    def _get_board_symbol_counts(self, board):
        """Get symbol counts from board."""
        if board is None:
            return 0, 0, 0

        counts = get_board_symbol_counts(board)
        x_count = counts.get('X', 0)
        o_count = counts.get('O', 0)
        total_count = x_count + o_count

        self.logger.debug("Board symbol counts: X=%s, O=%s, Total=%s",
                          x_count, o_count, total_count)
        return x_count, o_count, total_count

    # === Consolidated game management functions from game_manager.py ===

    def make_ai_move_simple(self):
        """Simple AI move without arm (consolidated from game_manager.py)."""
        if self.game_over or self.current_turn != self.ai_player:
            return

        # Get valid moves
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
                    if self.main_window.board_widget.board[r][c] == game_logic.EMPTY:
                        valid_moves.append((r, c))

        if not valid_moves:
            self.logger.warning("No valid moves available for AI")
            return

        # Select move using cached board state for occlusion handling
        if (hasattr(self.main_window, 'game_state_manager') and
                self.main_window.game_state_manager):
            ai_board = self.main_window.game_state_manager.game_state.get_board_for_ai()
            move = self.strategy_selector.get_move(ai_board, self.ai_player)
            self.logger.info("🤖 AI using cached board state for simple move")
        elif hasattr(self.main_window, 'board_widget'):
            move = self.strategy_selector.get_move(
                self.main_window.board_widget.board, self.ai_player)
        else:
            move = random.choice(valid_moves)

        if move:
            row, col = move
            self.main_window.board_widget.board[row][col] = self.ai_player
            self.main_window.board_widget.update()
            self.move_counter += 1
            self._check_game_end()

            if not self.game_over:
                self.current_turn = self.human_player
                # Reset robot status flag for next turn
                self.robot_status_displayed = False
                # NOTE: Don't emit "your_turn" here - wait for arm_turn_completed signal
                self.logger.info("🔄 AI MOVE DETECTED: Waiting for arm to complete turn sequence")

    def make_arm_move_with_detection_timeout(
            self, symbol_to_play, cell_centers=None):
        """Arm move with detection timeout (consolidated from game_manager.py)."""
        if self.game_over or self.current_turn != self.ai_player:
            return False

        # Get valid moves and select one
        valid_moves = []
        for r in range(3):
            for c in range(3):
                has_board_widget = (hasattr(self.main_window, 'board_widget') and
                                    self.main_window.board_widget)
                if has_board_widget:
                    if self.main_window.board_widget.board[r][c] == game_logic.EMPTY:
                        valid_moves.append((r, c))

        if not valid_moves:
            return False

        # Use cached board state for AI decision making
        if (hasattr(self.main_window, 'game_state_manager') and
                self.main_window.game_state_manager):
            ai_board = self.main_window.game_state_manager.game_state.get_board_for_ai()
            move = self.strategy_selector.get_move(ai_board, symbol_to_play)
            self.logger.info("🤖 AI using cached board for timeout move")
        elif hasattr(self.main_window, 'board_widget'):
            move = self.strategy_selector.get_move(
                self.main_window.board_widget.board, symbol_to_play)
        else:
            move = random.choice(valid_moves)

        if not move:
            return False

        row, col = move
        self.ai_move_row, self.ai_move_col = row, col
        self.waiting_for_detection = True

        # Send arm command if available
        if self.arm_controller and cell_centers and len(cell_centers) == 9:
            # cell_index = row * 3 + col  # Available for future use
            # cell_center = cell_centers[cell_index]
            # # Available for future use

            if symbol_to_play == game_logic.PLAYER_X:
                # Send DRAW_X command
                pass
            else:
                # Send DRAW_O command
                pass

            self.move_counter += 1

            # Set timeout for detection
            QTimer.singleShot(5000, lambda: self._check_detection_timeout(row, col, symbol_to_play))
            return True

        return False

    def _check_detection_timeout(self, _row, _col, _symbol):
        """Check detection timeout (consolidated from game_manager.py)."""
        if not self.waiting_for_detection:
            return

        self.ai_move_retry_count += 1

        if self.ai_move_retry_count >= self.max_retry_count:
            self.logger.warning(
                "Detection timeout for move at (%s, %s) - symbol NOT added to GUI",
                _row, _col)
            self.logger.info("GUI will only show what YOLO actually detects")

            # Reset flags
            self.waiting_for_detection = False
            self.ai_move_row = None
            self.ai_move_col = None
            self.ai_move_retry_count = 0

            self._check_game_end()

            if not self.game_over:
                self.current_turn = self.human_player
                # Reset robot status flag for next turn
                self.robot_status_displayed = False
                # NOTE: Don't emit "your_turn" here - wait for arm_turn_completed signal
                self.logger.info("🔄 DETECTION TIMEOUT: Waiting for arm to complete turn sequence")

    def _check_cache_status(self):
        """Check symbol cache status and emit appropriate GUI feedback."""
        try:
            if (hasattr(self.main_window, 'game_state_manager') and
                    self.main_window.game_state_manager and
                    hasattr(self.main_window.game_state_manager.game_state, 'symbol_cache')):

                cache = self.main_window.game_state_manager.game_state.symbol_cache
                current_cache_status = cache.using_cached_data

                # Only emit signal if status changed
                if current_cache_status != self.last_cache_status:
                    if current_cache_status:
                        self.status_changed.emit("cached_symbols", True)
                        self.logger.info("🔄 GUI: Switched to cached symbol display")
                    else:
                        self.status_changed.emit("live_detection", True)
                        self.logger.info("📹 GUI: Switched to live detection display")

                    self.last_cache_status = current_cache_status

        except Exception as e:
            self.logger.debug("Error checking cache status: %s", e)
