"""
Game manager for the TicTacToe application.
"""
import logging
import time
import random
from PyQt5.QtCore import QTimer

from app.main import game_logic
from app.core.arm_thread import ArmCommand


class GameManager:
    """Manages the game state and logic"""
    
    def __init__(self, app):
        """Initialize with reference to the main app"""
        self.app = app
        self.logger = logging.getLogger(__name__)
        
        # Game state
        self.game_over = False
        self.winner = None
        self.waiting_for_detection = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.ai_move_retry_count = 0
        self.move_counter = 0
        self.arm_player_symbol = None
        
        # Player symbols
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = self.human_player
        
        # Detection timeout
        self.detection_timeout_counter = 0
        self.max_detection_attempts = 3
    
    def reset_game(self):
        """Reset the game state"""
        self.logger.info("Resetting game")
        
        # Reset board widget
        if hasattr(self.app, 'board_widget'):
            self.app.board_widget.board = game_logic.create_board()
            self.app.board_widget.winning_line = None
            self.app.board_widget.update()
        
        # Reset game state
        self.game_over = False
        self.winner = None
        self.waiting_for_detection = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.ai_move_retry_count = 0
        self.move_counter = 0
        self.arm_player_symbol = None
        
        # Reset player symbols
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = self.human_player
        
        # Reset detection timeout
        self.detection_timeout_counter = 0
    
    def make_ai_move(self):
        """Make AI move based on current board state"""
        if self.game_over or self.current_turn != self.ai_player:
            return
        
        # Ensure the status is set to AI's turn
        self.app._status_lock = True
        self.app.update_status(self.app.tr("ai_turn"))
        
        # Make sure AI player symbol is valid
        if not self.ai_player or self.ai_player == game_logic.EMPTY:
            self.ai_player = game_logic.PLAYER_O  # Default to O for AI
            self.logger.warning(f"Invalid AI player symbol, using default: {self.ai_player}")
        
        # Log current board state
        self.logger.info("=== Board state before AI move ===")
        for r in range(3):
            row_str = ""
            for c in range(3):
                cell = self.app.board_widget.board[r][c]
                if cell == game_logic.EMPTY:
                    row_str += "[ ]"
                else:
                    row_str += f"[{cell}]"
            self.logger.info(row_str)
        self.logger.info("===================================")
        
        # Get valid moves
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if self.app.board_widget.board[r][c] == game_logic.EMPTY:
                    valid_moves.append((r, c))
        
        if not valid_moves:
            self.logger.warning("No valid moves available for AI")
            return
        
        # Select move based on difficulty
        if hasattr(self.app, 'strategy_selector'):
            # Use strategy selector for move selection
            board_copy = [row[:] for row in self.app.board_widget.board]
            move = self.app.strategy_selector.select_move(
                board_copy, self.ai_player)
        else:
            # Fallback to random move
            move = random.choice(valid_moves)
        
        self.logger.info(f"AI selected move: {move}")
        
        # Make the move
        row, col = move
        self.app.board_widget.board[row][col] = self.ai_player
        self.app.board_widget.update()
        
        # Increment move counter
        self.move_counter += 1
        
        # Check for game end
        self.check_game_end()
        
        # If game is not over, pass turn to player
        if not self.game_over:
            self.current_turn = self.human_player
            self.app.update_status(self.app.tr("your_turn"))
            self.app.main_status_panel.setStyleSheet("""
                background-color: #9b59b6;
                border-radius: 10px;
                border: 2px solid #8e44ad;
            """)
    
    def make_arm_move(self):
        """Make move with robotic arm"""
        if self.game_over or self.current_turn != self.ai_player:
            return
        
        # Determine which symbol the arm should draw
        symbol = self.arm_player_symbol
        if not symbol or symbol == game_logic.EMPTY:
            symbol = self.ai_player
        
        self.logger.info(f"Arm will draw symbol: {symbol}")
        
        # Ensure the status is set to arm's turn
        self.app._status_lock = True
        self.app.update_status(self.app.tr("arm_turn"))
        self.app.main_status_panel.setStyleSheet("""
            background-color: #9b59b6;
            border-radius: 10px;
            border: 2px solid #8e44ad;
        """)
        
        # Get valid moves
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if self.app.board_widget.board[r][c] == game_logic.EMPTY:
                    valid_moves.append((r, c))
        
        self.logger.info(f"Valid moves for arm: {valid_moves}")
        
        if not valid_moves:
            self.logger.warning("No valid moves available for arm")
            return
        
        # Select move based on difficulty
        if hasattr(self.app, 'strategy_selector'):
            # Use strategy selector for move selection
            board_copy = [row[:] for row in self.app.board_widget.board]
            move = self.app.strategy_selector.select_move(
                board_copy, symbol)
        else:
            # Fallback to random move
            move = random.choice(valid_moves)
        
        self.logger.info(f"Arm selected move: {move}")
        
        # Make the move
        row, col = move
        
        # Store the move for detection verification
        self.ai_move_row = row
        self.ai_move_col = col
        self.waiting_for_detection = True
        
        # Draw the symbol with the arm
        if hasattr(self.app, 'arm_thread') and self.app.arm_thread:
            # Get cell centers from game state
            cell_centers = None
            if hasattr(self.app, 'camera_thread') and self.app.camera_thread:
                detection_thread = getattr(self.app.camera_thread, 'detection_thread', None)
                if detection_thread:
                    _, game_state = detection_thread.get_latest_result()
                    if game_state and game_state.is_valid() and game_state.is_physical_grid_valid():
                        cell_centers = game_state.get_cell_centers_uv_transformed()
            
            if cell_centers is not None and len(cell_centers) == 9:
                # Calculate cell index (row * 3 + col)
                cell_index = row * 3 + col
                
                # Get cell center coordinates
                cell_center = cell_centers[cell_index]
                
                # Send command to arm to draw symbol
                if symbol == game_logic.PLAYER_X:
                    self.app.arm_thread.add_command(
                        ArmCommand.DRAW_X,
                        {"uv": (int(cell_center[0]), int(cell_center[1]))})
                else:
                    self.app.arm_thread.add_command(
                        ArmCommand.DRAW_O,
                        {"uv": (int(cell_center[0]), int(cell_center[1]))})
                
                # Increment move counter
                self.move_counter += 1
                
                # After a delay, check if the move was detected
                QTimer.singleShot(5000, lambda: self.check_detection_timeout(row, col, symbol))
            else:
                # If cell centers not available, restore turn to player
                self.logger.error("Cell centers not available for arm move")
                self.current_turn = self.human_player
                self.app.update_status(self.app.tr("your_turn"))
        else:
            # Simulate the arm move if arm is not available
            self.logger.info(f"Simulating arm move: {symbol} at ({row}, {col})")
            self.app.board_widget.board[row][col] = symbol
            self.app.board_widget.update()
            
            # Increment move counter
            self.move_counter += 1
            
            # Check game end
            self.check_game_end()
            
            # If game is not over, pass turn to player
            if not self.game_over:
                self.current_turn = self.human_player
                self.app.update_status(self.app.tr("your_turn"))
    
    def check_detection_timeout(self, row, col, symbol):
        """Check if the arm move was detected"""
        if not self.waiting_for_detection:
            return
        
        self.detection_timeout_counter += 1
        
        if self.detection_timeout_counter >= self.max_detection_attempts:
            self.logger.warning(f"Detection timeout for move at ({row}, {col})")
            
            # Manually update the board
            self.app.board_widget.board[row][col] = symbol
            self.app.board_widget.update()
            
            # Reset detection flags
            self.waiting_for_detection = False
            self.ai_move_row = None
            self.ai_move_col = None
            self.detection_timeout_counter = 0
            
            # Check for game end
            self.check_game_end()
            
            # If game is not over, pass turn to player
            if not self.game_over:
                self.current_turn = self.human_player
                self.app.update_status(self.app.tr("your_turn"))
        else:
            # Try again after a delay
            QTimer.singleShot(1000, lambda: self.check_detection_timeout(row, col, symbol))
    
    def check_game_end(self):
        """Check if the game has ended (win or draw)"""
        self.winner = game_logic.check_winner(self.app.board_widget.board)
        
        if self.winner:
            self.game_over = True
            
            # Get winning line
            winning_line = None
            if self.winner != game_logic.TIE:
                winning_line = game_logic.get_winning_line(self.app.board_widget.board)
            
            # Update board widget
            self.app.board_widget.update_board(self.app.board_widget.board, winning_line)
            
            # Update status
            if self.winner == game_logic.TIE:
                self.app.update_status(self.app.tr("game_draw"))
            elif self.winner == self.human_player:
                self.app.update_status(self.app.tr("you_win"))
            else:
                self.app.update_status(self.app.tr("ai_wins"))
            
            # Update status panel style
            if self.winner == game_logic.TIE:
                self.app.main_status_panel.setStyleSheet("""
                    background-color: #f39c12;
                    border-radius: 10px;
                    border: 2px solid #d35400;
                """)
            elif self.winner == self.human_player:
                self.app.main_status_panel.setStyleSheet("""
                    background-color: #2ecc71;
                    border-radius: 10px;
                    border: 2px solid #27ae60;
                """)
            else:
                self.app.main_status_panel.setStyleSheet("""
                    background-color: #e74c3c;
                    border-radius: 10px;
                    border: 2px solid #c0392b;
                """)
            
            return True
        
        return False
