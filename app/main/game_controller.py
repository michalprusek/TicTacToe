"""
Game Controller module for TicTacToe application.
This module handles game logic, state management, and turn coordination.
Consolidates functionality from game_manager.py.
"""

import logging
import time
import random
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Import required modules
from app.main.path_utils import setup_project_path
setup_project_path()

from app.main import game_logic
from app.core.strategy import BernoulliStrategySelector
from app.core.arm_thread import ArmCommand
from app.main.constants import DEFAULT_DIFFICULTY
from app.main.game_utils import setup_logger, convert_board_1d_to_2d, get_board_symbol_counts


class GameController(QObject):
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
        self.logger.info(f"Strategy selector initialized with difficulty {difficulty} (p={difficulty/10.0:.2f})")

        # Arm controller reference (set by main window)
        self.arm_controller = None

    def set_arm_controller(self, arm_controller):
        """Set the arm controller reference."""
        self.arm_controller = arm_controller

    def start_game(self):
        """Start a new game."""
        self.reset_game()

    def reset_game(self):
        """Reset the game to initial state."""
        self.logger.info("Resetting game.")

        # Reset authoritative board state
        self.authoritative_board = game_logic.create_board()

        # Reset board widget
        if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
            empty_board = game_logic.create_board()
            self.main_window.board_widget.update_board(empty_board, None, highlight_changes=False)
            self.main_window.board_widget.winning_line = None
            self.main_window.board_widget.update()

        # Reset game state
        self.game_over = False
        self.winner = None
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = game_logic.PLAYER_X
        self.move_counter = 0

        # Reset arm flags
        self.waiting_for_detection = False
        self.arm_move_in_progress = False
        self.last_arm_move_time = 0

        # Reset detection retry logic
        self.ai_move_row = None
        self.ai_move_col = None
        self.expected_symbol = None
        self.ai_move_retry_count = 0
        self.detection_wait_time = 0.0

        self.status_changed.emit("new_game_detected", True)
        self.logger.info(f"Game reset. Current turn: {self.current_turn}")

        # Set status to indicate it's human player's turn
        QTimer.singleShot(2000, lambda: self.status_changed.emit("your_turn", True))

        # Move arm to neutral position
        if self.arm_controller:
            self.arm_controller.move_to_neutral_position()

    def handle_cell_clicked(self, row, col):
        """Handle cell click from the board widget."""
        if self.game_over or self.current_turn != self.human_player:
            self.logger.warning(f"Click ignored: game_over={self.game_over}, current_turn={self.current_turn}")
            return

        if self.arm_move_in_progress or self.waiting_for_detection:
            self.logger.warning(f"Click ignored: arm_busy={self.arm_move_in_progress or self.waiting_for_detection}")
            return

        # Check if cell is empty
        if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
            if self.main_window.board_widget.board[row][col] != game_logic.EMPTY:
                self.logger.info("Cell is already occupied.")
                return

        # Human player intends to move - wait for camera detection
        self.logger.info(f"Player ({self.human_player}) intends to move to ({row},{col}). Waiting for detection.")
        self.status_changed.emit("waiting_detection", True)

    def handle_detected_game_state(self, detected_board_from_camera):
        """Handle detected game state from camera."""
        if detected_board_from_camera is None:
            self.logger.debug("Detected empty board (None) from camera.")
            return

        detected_board = convert_board_1d_to_2d(detected_board_from_camera)
        if not detected_board:
            self.logger.warning("Failed to convert detected board to 2D format.")
            return

        # CRITICAL FIX: Update authoritative board state from camera detection
        self.authoritative_board = [row[:] for row in detected_board]  # Deep copy
        self.logger.debug(f"Updated authoritative board from camera: {self.authoritative_board}")

        # Update board widget with detected board state
        if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
            self.main_window.board_widget.update_board(detected_board, None, highlight_changes=True)
            self.logger.debug(f"Updated GUI board with detected state: {detected_board}")

        # Handle game over state
        if self.game_over:
            is_empty_now = all(cell == game_logic.EMPTY for row in detected_board for cell in row)
            if is_empty_now:
                self.logger.info("Detected empty board after game end - resetting for new game.")
                self.reset_game()
            return

        # Check for new moves and game progression
        self._process_detected_board(detected_board)

    def _process_detected_board(self, detected_board):
        """Process the detected board for game logic."""
        # Count symbols
        x_count, o_count, total_count = self._get_board_symbol_counts(detected_board)

        # Check if it's time for arm to play
        if not self.game_over:
            should_play, arm_symbol_to_play = self._should_arm_play_now(detected_board)
            if should_play and arm_symbol_to_play:
                self.logger.info(f"DECISION: Arm should play with symbol {arm_symbol_to_play}.")
                self.ai_player = arm_symbol_to_play
                self.current_turn = self.ai_player
                self.status_changed.emit("arm_moving", True)
                self.make_arm_move_with_symbol(arm_symbol_to_play)
            elif self.current_turn == self.ai_player and not self.arm_move_in_progress and not self.waiting_for_detection:
                self.status_changed.emit("arm_turn", True)
                self.logger.debug("Arm is on turn, but conditions for playing are not met.")
            elif self.current_turn == self.human_player and not self.arm_move_in_progress and not self.waiting_for_detection:
                self.status_changed.emit("your_turn", True)

    def _should_arm_play_now(self, current_board_state):
        """Determine if the arm should play now."""
        self.logger.debug(f"Checking if arm should play. InProgress: {self.arm_move_in_progress}, "
                         f"Cooldown: {time.time() - self.last_arm_move_time < self.arm_move_cooldown}")

        if self.game_over or self.arm_move_in_progress or (time.time() - self.last_arm_move_time < self.arm_move_cooldown):
            return False, None

        # Check grid validity (simplified for now)
        # TODO: Implement proper grid validation

        # Count symbols to determine turn
        x_count, o_count, total_count = self._get_board_symbol_counts(current_board_state)

        # Arm plays when odd number of total symbols (after human move)
        if total_count % 2 == 1:
            # Determine which symbol arm should play
            if x_count > o_count:
                return True, game_logic.PLAYER_O
            elif o_count > x_count:
                return True, game_logic.PLAYER_X
            else:
                return True, game_logic.PLAYER_O  # Default to O

        return False, None

    def make_arm_move_with_symbol(self, symbol_to_play):
        """Make an arm move with the specified symbol."""
        self.logger.info(f"Starting arm move with symbol: {symbol_to_play}")

        if self.game_over or self.arm_move_in_progress:
            self.logger.warning("Arm move interrupted: game ended or arm already in progress.")
            return False

        # CRITICAL FIX: Use authoritative board state from camera detection
        current_board_for_strategy = self.authoritative_board

        if not current_board_for_strategy:
            self.logger.error("Cannot get authoritative board state for strategy.")
            return False

        # Log the board state being used for strategy
        self.logger.info(f"Using authoritative board for strategy: {current_board_for_strategy}")

        # Set flags BEFORE starting movement
        self.arm_move_in_progress = True
        self.last_arm_move_time = time.time()
        self.status_changed.emit("arm_moving", True)

        # Get move from strategy
        move = self.strategy_selector.get_move(current_board_for_strategy, symbol_to_play)
        if not move:
            self.logger.error("Strategy did not return a valid move.")
            self.arm_move_in_progress = False
            return False

        row, col = move
        self.logger.info(f"Strategy selected move: ({row},{col}) with symbol {symbol_to_play}")

        # Store move for detection verification
        self.ai_move_row, self.ai_move_col = row, col
        self.expected_symbol = symbol_to_play
        self.ai_move_retry_count = 0
        self.detection_wait_time = 0.0

        # Execute arm drawing
        if self.arm_controller and self.arm_controller.draw_ai_symbol(row, col, symbol_to_play):
            self.logger.info(f"Symbol {symbol_to_play} successfully sent for drawing at ({row},{col}). Waiting for detection.")
            self.waiting_for_detection = True
            return True
        else:
            self.logger.error(f"Failed to start drawing symbol {symbol_to_play} at ({row},{col}).")
            self.arm_move_in_progress = False
            self.waiting_for_detection = False
            self.current_turn = self.human_player
            self.status_changed.emit("your_turn", True)
            return False

    def update_game_state(self):
        """Update game state machine (called by timer)."""
        if self.game_over:
            return

        # Handle detection waiting
        if self.waiting_for_detection:
            self.detection_wait_time += 0.1  # Timer is every 100ms

            # Check for detection timeout
            if self.detection_wait_time >= self.max_detection_wait_time:
                self.ai_move_retry_count += 1
                self.logger.warning(f"Detection timeout. Retry {self.ai_move_retry_count}/{self.max_retry_count}")

                if self.ai_move_retry_count >= self.max_retry_count:
                    self.logger.error("Max retries reached. Giving up on arm move.")
                    self.waiting_for_detection = False
                    self.arm_move_in_progress = False
                    self.current_turn = self.human_player
                    self.status_changed.emit("detection_failed", True)
                    QTimer.singleShot(2000, lambda: self.status_changed.emit("your_turn", True))

                # Reset for next attempt
                self.ai_move_row, self.ai_move_col, self.expected_symbol = None, None, None

        # Check for game end
        self._check_game_end()

    def _check_game_end(self):
        """Check if the game has ended."""
        # CRITICAL FIX: Use authoritative board state for game end check
        board_to_check = self.authoritative_board

        if not board_to_check:
            return

        self.winner = game_logic.check_winner(board_to_check)

        if self.winner:
            self.game_over = True
            self.arm_move_in_progress = False
            self.waiting_for_detection = False
            self.logger.info(f"GAME END! Winner: {self.winner}. Move count: {self.move_counter}")

            # Show game end notification
            self._show_game_end_notification()

            # Handle winning line
            if self.winner != game_logic.TIE:
                if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
                    self.main_window.board_widget.winning_line = game_logic.get_winning_line(board_to_check)
                    self.main_window.board_widget.update()

                # Draw winning line if AI won
                if self.winner == self.ai_player and self.arm_controller:
                    self.logger.info("AI (arm) won! Planning to draw winning line.")
                    QTimer.singleShot(1500, self.arm_controller.draw_winning_line)

            if self.winner == game_logic.TIE:
                self.status_changed.emit("draw", True)
            else:
                self.status_changed.emit("win", True)

            # Move to neutral position
            if self.arm_controller:
                self.arm_controller.move_to_neutral_position()

        # Check for draw (board full)
        elif self.move_counter >= 9:
            self.game_over = True
            self.winner = game_logic.TIE
            self.logger.info(f"GAME END! Draw. Move count: {self.move_counter}")
            self.arm_move_in_progress = False
            self.waiting_for_detection = False
            self._show_game_end_notification()
            self.status_changed.emit("draw", True)

            if self.arm_controller:
                self.arm_controller.move_to_neutral_position()

    def _show_game_end_notification(self):
        """Show game end notification."""
        # This will be handled by the status manager
        self.game_ended.emit(str(self.winner))

    def _get_board_symbol_counts(self, board):
        """Get symbol counts from board."""
        if board is None:
            return 0, 0, 0

        board_2d = convert_board_1d_to_2d(board)
        if not isinstance(board_2d, list) or not all(isinstance(row, list) for row in board_2d):
            return 0, 0, 0

        x_count = sum(row.count(game_logic.PLAYER_X) for row in board_2d)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in board_2d)
        return x_count, o_count, x_count + o_count

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

        # Select move
        move = self.strategy_selector.get_move(
            self.main_window.board_widget.board, self.ai_player) if hasattr(self.main_window, 'board_widget') else random.choice(valid_moves)

        if move:
            row, col = move
            self.main_window.board_widget.board[row][col] = self.ai_player
            self.main_window.board_widget.update()
            self.move_counter += 1
            self._check_game_end()

            if not self.game_over:
                self.current_turn = self.human_player
                self.status_changed.emit("your_turn", True)

    def make_arm_move_with_detection_timeout(self, symbol_to_play, cell_centers=None):
        """Arm move with detection timeout (consolidated from game_manager.py)."""
        if self.game_over or self.current_turn != self.ai_player:
            return False

        # Get valid moves and select one
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if hasattr(self.main_window, 'board_widget') and self.main_window.board_widget:
                    if self.main_window.board_widget.board[r][c] == game_logic.EMPTY:
                        valid_moves.append((r, c))

        if not valid_moves:
            return False

        move = self.strategy_selector.get_move(
            self.main_window.board_widget.board, symbol_to_play) if hasattr(self.main_window, 'board_widget') else random.choice(valid_moves)

        if not move:
            return False

        row, col = move
        self.ai_move_row, self.ai_move_col = row, col
        self.waiting_for_detection = True

        # Send arm command if available
        if self.arm_controller and cell_centers and len(cell_centers) == 9:
            cell_index = row * 3 + col
            cell_center = cell_centers[cell_index]

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

    def _check_detection_timeout(self, row, col, symbol):
        """Check detection timeout (consolidated from game_manager.py)."""
        if not self.waiting_for_detection:
            return

        self.ai_move_retry_count += 1

        if self.ai_move_retry_count >= self.max_retry_count:
            self.logger.warning(f"Detection timeout for move at ({row}, {col}) - symbol NOT added to GUI")
            self.logger.info("GUI will only show what YOLO actually detects")

            # Reset flags
            self.waiting_for_detection = False
            self.ai_move_row = None
            self.ai_move_col = None
            self.ai_move_retry_count = 0

            self._check_game_end()

            if not self.game_over:
                self.current_turn = self.human_player
                self.status_changed.emit("your_turn", True)
