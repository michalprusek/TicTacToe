"""
Game state module for the TicTacToe application.
"""
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

# Constants for game state
EMPTY = ''
PLAYER_X = 'X'
PLAYER_O = 'O'
TIE = "TIE"

# GUI Colors for visualization
GUI_GRID_COLOR = (255, 255, 255)  # White for grid lines
GUI_X_COLOR = (0, 0, 255)         # Red for X (BGR format)
GUI_O_COLOR = (0, 255, 0)         # Green for O (BGR format)
GUI_LINE_THICKNESS = 2
GUI_SYMBOL_THICKNESS = 3

GRID_POINTS_COUNT = 16

# Canonical ideal grid coordinates (4x4 grid, points 0-15)
# (col, row) format, ranging from (0,0) to (3,3) for a 3x3 cell grid.
IDEAL_GRID_POINTS_CANONICAL = np.array([
    (c, r) for r in range(4) for c in range(4)
], dtype=np.float32)


class GameState:
    """Class to represent and manage the game state."""
    ERROR_GRID_INCOMPLETE_PAUSE = "GRID_INCOMPLETE_PAUSE_STATE"  # Unique state identifier

    def __init__(self):
        """Initialize an empty game state."""
        self.logger = logging.getLogger(__name__)
        self._board_state = [[EMPTY for _ in range(3)] for _ in range(3)]
        self._grid_points = None
        self._homography = None
        self._detection_results = []
        self._timestamp = 0
        self._is_valid_grid = False
        self._changed_cells_this_turn: List[Tuple[int, int]] = []
        self.error_message = None  # Added for error handling
        self.game_paused_due_to_incomplete_grid: bool = False
        self.grid_fully_visible: bool = False  # True if all grid points are detected
        self.missing_grid_points_count: int = 0  # Number of missing grid points

        self._last_move_timestamp: Optional[float] = None
        self._move_cooldown_seconds: float = 1.0

        # Game result attributes
        self.winner: Optional[str] = None
        self.winning_line_indices: Optional[List[Tuple[int, int]]] = None

        # Initialize previous rotation angle for smoothing
        self._previous_rotation_angle = None

        # Points for drawing the orthogonal grid, derived from homography
        self._transformed_grid_points_for_drawing: Optional[np.ndarray] = None

        # Cell centers in UV space, transformed from ideal grid for game logic
        self._cell_centers_uv_transformed = None
        self._cell_polygons_uv_transformed: Optional[List[np.ndarray]] = None  # Added initialization

    def reset_game(self):
        """Resets game state, keeping grid points if valid."""
        self.logger.info("Resetting game state")
        self._board_state = [[EMPTY for _ in range(3)] for _ in range(3)]
        self._detection_results = []
        self._changed_cells_this_turn = []
        # Keep grid_points and homography if still valid
        self.error_message = None  # Reset error message
        self._last_move_timestamp = None  # Reset cooldown
        self.game_paused_due_to_incomplete_grid = False # Reset pause state
        self.grid_fully_visible = False # Reset grid visibility status
        self.missing_grid_points_count = 0 # Reset missing points count
        self.winner = None
        self.winning_line_indices = None

    @property
    def board(self) -> List[List[str]]:
        """Get the current board state as a 2D list."""
        return [row[:] for row in self._board_state]

    @property
    def grid_points(self) -> Optional[np.ndarray]:
        """Get the detected grid points."""
        return self._grid_points

    @property
    def detection_results(self) -> List:
        """Get the raw detection results."""
        return self._detection_results

    @property
    def changed_cells_this_turn(self) -> List[Tuple[int, int]]:
        """Get the list of cells changed in the current detection cycle."""
        return self._changed_cells_this_turn

    def reset_changed_cells(self) -> None:
        """Reset the list of changed cells for the current turn."""
        self._changed_cells_this_turn = []

    def is_physical_grid_valid(self) -> bool:
        """Check if the detected physical grid is valid."""
        return self._is_valid_grid

    def is_game_paused_due_to_incomplete_grid(self) -> bool:
        """Check if the game is paused because the grid is not fully visible."""
        return self.game_paused_due_to_incomplete_grid

    def is_valid(self) -> bool:
        """Check if the current logical game board state is valid (symbols)."""
        for r in range(3):
            for c in range(3):
                symbol = self._board_state[r][c]
                if symbol not in [PLAYER_X, PLAYER_O, EMPTY]:
                    self.logger.warning(
                        "Invalid symbol '%s' found at (%d,%d)", symbol, r, c
                    )
                    return False
        return True

    def count_symbols(self) -> Tuple[int, int]:
        """Count the number of X and O symbols on the board."""
        x_count = 0
        o_count = 0
        for r in range(3):
            for c in range(3):
                if self._board_state[r][c] == PLAYER_X:
                    x_count += 1
                elif self._board_state[r][c] == PLAYER_O:
                    o_count += 1
        return x_count, o_count

    def is_valid_turn_sequence(self) -> bool:
        """Check if the sequence of turns is valid (X >= O and diff <= 1)."""
        x_count, o_count = self.count_symbols()
        diff = x_count - o_count
        # X always starts, so x_count must be >= o_count
        # The difference can be at most 1 (e.g., X, XOX, XOXOX)
        return 0 <= diff <= 1

    def _update_board_with_symbols(
            self,
            detected_symbols: List[Dict],
            cell_centers_uv: np.ndarray,
            class_id_to_player: Dict[int, str]
    ) -> List[Tuple[int, int]]:
        """
        Updates the board state based on detected symbols and their proximity
        to cell centers. Only updates empty cells.

        Args:
            detected_symbols: A list of dictionaries, each representing a
                              detected symbol. Expected keys: 'center_uv',
                              'player', 'confidence'.
            cell_centers_uv: A NumPy array of (u, v) coordinates for the 9
                             cell centers.
            class_id_to_player: Mapping from class ID to player ('X' or 'O').

        Returns:
            A list of (row, col) tuples for cells that were changed in this update.
        """
        if cell_centers_uv is None or len(cell_centers_uv) != 9:
            self.logger.warning(
                "Cannot update board: cell_centers_uv is invalid or not 9 points."
            )
            return []

        changed_cells = []
        
        # Debug: Log current board state
        board_str = str([row[:] for row in self._board_state])
        self.logger.info(f"Current board state before symbol update: {board_str}")
        # Create a copy of the board to check for changes
        # original_board_state = [row[:] for row in self.board]

        # Sort symbols by confidence to prioritize more certain detections
        # sorted_symbols = sorted(
        #    detected_symbols, key=lambda s: s.get('confidence', 0), reverse=True
        # )

        for symbol_info in detected_symbols: # Use unsorted for now
            symbol_center_uv = symbol_info.get('center_uv')
            player = symbol_info.get('player')

            if symbol_center_uv is None or player is None:
                self.logger.warning(
                    f"Skipping symbol due to missing data: {symbol_info}"
                )
                continue

            # Find the closest cell center to this symbol
            distances = np.linalg.norm(cell_centers_uv - symbol_center_uv, axis=1)
            closest_cell_idx = np.argmin(distances)

            row, col = divmod(closest_cell_idx, 3)

            # Check if the cell is empty before placing the symbol
            if self.board[row][col] == EMPTY:
                self.board[row][col] = player
                changed_cells.append((row, col))
                self.logger.info(
                    f"Placed '{player}' at ({row},{col}) based on symbol at {symbol_center_uv}"
                )
            else:
                self.logger.info(
                    f"Cell ({row},{col}) already occupied by {self.board[row][col]}. Symbol {player} not placed."
                )
            # else:
            #     self.logger.debug(
            #         f"Cell ({row},{col}) already occupied by {self.board[row][col]}. Symbol {player} not placed."
            #     )

        # if not changed_cells:
        #     self.logger.debug("No cells were changed in this update cycle.")

        return changed_cells

    def is_game_over(self) -> bool:
        """Check if the game is over (win or draw)."""
        return self.winner is not None

    def is_game_over_due_to_error(self) -> bool:
        """Check if the game is effectively over due to a persistent error."""
        return self.error_message is not None and \
               self.error_message.startswith("FATAL:")

    def get_winner(self) -> Optional[str]:
        """Get the winner of the game, if any."""
        return self.winner

    def get_winning_line_indices(self) -> Optional[List[Tuple[int, int]]]:
        """Get the indices of the cells forming the winning line."""
        return self.winning_line_indices

    def set_error(self, message: str) -> None:
        """
        Set an error message. Prepends 'FATAL:' if it's a game-ending error.
        Avoids overwriting a FATAL error with a non-FATAL one.
        """
        if self.error_message and \
           self.error_message.startswith("FATAL:") and \
           not message.startswith("FATAL:"):
            self.logger.debug(
                "Skipping non-fatal error '%s' as fatal error '%s' is active.",
                message, self.error_message
            )
            return
        self.error_message = message
        self.logger.error("Error set: %s", self.error_message)

    def clear_error_message(self) -> None:
        """Clear the current error message."""
        if self.error_message:
            self.logger.info(
                "Clearing error: %s",
                self.error_message
            )
            self.error_message = None

    def get_error(self) -> Optional[str]:
        """Get the current error message."""
        return self.error_message
        
    def get_error_message(self) -> Optional[str]:
        """Get the current error message, alias for get_error()."""
        return self.error_message

    def is_error_active(self) -> bool:
        """Check if there is an active error message."""
        return self.error_message is not None

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current processed frame."""
        return self._frame

    def get_timestamp(self) -> float:
        """Get the timestamp of the current state."""
        return self._timestamp

    def get_homography(self) -> Optional[np.ndarray]:
        """Get the homography matrix (H_ideal_to_uv)."""
        return self._homography

    def get_transformed_grid_points_for_drawing(self) -> Optional[np.ndarray]:
        """
        Get the 16 grid points transformed by homography, intended for drawing
        an orthogonal grid. These points represent the corners of an ideal grid
        projected into the camera's view.
        """
        return self._transformed_grid_points_for_drawing

    def get_cell_centers_uv_transformed(self) -> Optional[np.ndarray]:
        """
        Get the 9 cell centers in UV space, transformed from the ideal grid.
        These are used for game logic, e.g., determining which cell a
        detected symbol belongs to.
        """
        return self._cell_centers_uv_transformed

    def update_from_detection(
        self,
        frame: np.ndarray,
        ordered_kpts_uv: Optional[np.ndarray],
        homography: Optional[np.ndarray],
        detected_symbols: List[Dict],
        class_id_to_player: Dict[int, str],
        timestamp: float
    ) -> None:
        """Update game state based on detection results.

        This method now also checks if the physical grid is valid before
        updating the board or checking for win/draw conditions.
        It also handles pausing the game if the grid is not fully detected.
        """
        self._timestamp = timestamp
        self._detection_results = detected_symbols  # Store raw detections
        self._changed_cells_this_turn = [] # Reset changed cells for this update cycle

        # Check for complete grid detection first
        if ordered_kpts_uv is None or len(ordered_kpts_uv) != GRID_POINTS_COUNT:
            self._is_valid_grid = False # Physical grid is not valid
            self.game_paused_due_to_incomplete_grid = True
            self.error_message = self.ERROR_GRID_INCOMPLETE_PAUSE
            # Clear any existing grid-dependent data as it's no longer valid
            self._grid_points = None
            self._homography = None
            self._transformed_grid_points_for_drawing = None
            self._cell_centers_uv_transformed = None
            self._cell_polygons_uv_transformed = None
            self.logger.warning(
                "Grid not fully detected (%s points). Game paused.",
                len(ordered_kpts_uv) if ordered_kpts_uv is not None else 0
            )
            return # Stop further processing for this frame
        
        # Grid is fully detected, proceed with normal logic
        self._is_valid_grid = True # Physical grid is now considered valid
        self._grid_points = ordered_kpts_uv # Store the valid grid points

        if self.game_paused_due_to_incomplete_grid: # If game was paused, now it's resuming
            self.logger.info("Full grid detected. Resuming game logic.")
            if self.error_message == self.ERROR_GRID_INCOMPLETE_PAUSE:
                self.clear_error_message() # Clear the specific pause message
        self.game_paused_due_to_incomplete_grid = False

        if homography is not None:
            self._homography = homography

        # If the physical grid is not valid (e.g. due to ordering issues, though covered above),
        # do not proceed with game logic updates.
        # This check is somewhat redundant now due to the explicit check above but kept for safety.
        if not self.is_physical_grid_valid():
            self.logger.debug(
                "Physical grid not valid. Skipping board update & win/draw checks."
            )
            # Do not reset winner; game might be paused with a winner displayed
            return

        # Compute cell centers in UV space if grid points are valid
        # This should only happen if is_physical_grid_valid() is true
        if self._grid_points is not None and len(self._grid_points) == GRID_POINTS_COUNT:
            success_transform_logic = self._compute_grid_transformation()
            if not success_transform_logic:
                self.logger.warning(
                    "Failed to compute grid transformation for game logic."
                )
                return  # Cannot proceed without cell centers for logic
        else:
            # This case should ideally not be reached if the top check is working
            self.logger.warning(
                "Grid points are not valid for computing cell centers (should have been caught)."
            )
            return

        # Update board with symbols only if cell centers are available
        if (self._cell_centers_uv_transformed is not None and
                len(self._cell_centers_uv_transformed) == 9):
            # Check for cooldown before allowing a move
            if self._last_move_timestamp is None or \
               (timestamp - self._last_move_timestamp) >= self._move_cooldown_seconds:
                changed_cells = self._update_board_with_symbols(
                    detected_symbols,
                    self._cell_centers_uv_transformed,
                    class_id_to_player
                )
                if changed_cells: # If a move was made
                    self._changed_cells_this_turn = changed_cells
                    self._last_move_timestamp = timestamp # Update last move time
                    self.logger.info(f"Move made. Cooldown started. Changed cells: {changed_cells}")
                else:
                    self.logger.debug("No valid move detected or board unchanged.")
            else:
                self.logger.debug(
                    f"Move cooldown active. {(timestamp - self._last_move_timestamp):.2f}s / "
                    f"{self._move_cooldown_seconds:.2f}s"
                )
        else:
            self.logger.warning(
                "Cell centers for game logic not computed. "
                "Cannot update board with symbols."
            )
            # No return here; win/draw check on existing board might be desired

        # Check for win/draw conditions only if the game is not paused and board is valid
        if not self.is_game_paused_due_to_incomplete_grid and self.is_physical_grid_valid():
            self._check_win_conditions()

    def _check_win_conditions(self) -> None:
        """Check for win conditions (rows, columns, diagonals) and draw."""
        if not self.is_physical_grid_valid():
            self.logger.debug(
                "Skipping win/draw check: physical grid not valid."
            )
            return

        if self.winner:  # Game already has a winner
            return

        # Check rows
        lines = [
            # Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]

        for line_indices in lines:
            r1, c1 = line_indices[0]
            r2, c2 = line_indices[1]
            r3, c3 = line_indices[2]

            if (self.board[r1][c1] == self.board[r2][c2] == self.board[r3][c3] and
                    self.board[r1][c1] != EMPTY):
                self.winner = self.board[r1][c1]
                self.winning_line_indices = line_indices
                self.logger.info(
                    f"Winner: {self.winner}, Line: {self.winning_line_indices}"
                )
                return

        # Check for a draw
        if all(self.board[r][c] != EMPTY for r in range(3) for c in range(3)) \
           and self.winner is None:
            self.winner = "Draw"
            self.logger.info("Game is a Draw.")

    def get_latest_derived_cell_polygons(self) -> Optional[List[np.ndarray]]:
        """
        Returns the latest computed cell polygons based on transformed grid points.
        These are suitable for drawing the grid cells.
        Returns None if not available.
        """
        if self._cell_polygons_uv_transformed is None:
            self.logger.debug(
                "Cell polygons for drawing not available in GameState."
            )
        return self._cell_polygons_uv_transformed

    def get_cell_center_uv(self, row: int, col: int) -> Optional[np.ndarray]:
        """Get the UV coordinates of the center of a specific cell.

        Args:
            row: The row index of the cell (0-2).
            col: The column index of the cell (0-2).

        Returns:
            A NumPy array [x, y] for the cell center, or None if not available.
        """
        if self._cell_centers_uv_transformed is not None:
            idx = row * 3 + col
            if 0 <= idx < len(self._cell_centers_uv_transformed):
                return self._cell_centers_uv_transformed[idx]
            else:
                self.logger.warning(
                    f"Invalid cell index {idx} for get_cell_center_uv."
                )
                return None
        else:
            self.logger.warning(
                "Cell centers not available for get_cell_center_uv."
            )
            return None

    def _compute_grid_transformation(self) -> bool:
        """
        Compute cell centers in UV space if grid points are valid.
        This should only happen if is_physical_grid_valid() is true.
        Updates self._cell_centers_uv_transformed.
        """
        if self._grid_points is None or len(self._grid_points) != GRID_POINTS_COUNT:
            self.logger.debug(
                "Not enough grid points for transformation. Points: %s",
                len(self._grid_points) if self._grid_points is not None else 'None'
            )
            return False

        try:
            cell_centers = []
            for r_cell in range(3):  # cell row
                for c_cell in range(3):  # cell col
                    p_tl_idx = r_cell * 4 + c_cell
                    p_tr_idx = r_cell * 4 + (c_cell + 1)
                    p_bl_idx = (r_cell + 1) * 4 + c_cell
                    p_br_idx = (r_cell + 1) * 4 + (c_cell + 1)

                    if not (0 <= p_tl_idx < GRID_POINTS_COUNT and
                            0 <= p_tr_idx < GRID_POINTS_COUNT and
                            0 <= p_bl_idx < GRID_POINTS_COUNT and
                            0 <= p_br_idx < GRID_POINTS_COUNT):
                        self.logger.error(
                            f"Invalid point indices for cell ({r_cell},{c_cell}) "
                            f"when calculating centers."
                        )
                        raise ValueError(
                            "Invalid grid point indices for cell center calculation"
                        )

                    center_x = (
                        self._grid_points[p_tl_idx][0] +
                        self._grid_points[p_tr_idx][0] +
                        self._grid_points[p_bl_idx][0] +
                        self._grid_points[p_br_idx][0]
                    ) / 4.0
                    center_y = (
                        self._grid_points[p_tl_idx][1] +
                        self._grid_points[p_tr_idx][1] +
                        self._grid_points[p_bl_idx][1] +
                        self._grid_points[p_br_idx][1]
                    ) / 4.0
                    cell_centers.append([center_x, center_y])
            
            if len(cell_centers) == 9:
                self._cell_centers_uv_transformed = np.array(
                    cell_centers, dtype=np.float32
                )
                self.logger.info(
                    "Computed %s cell centers for game logic from _grid_points.",
                    len(self._cell_centers_uv_transformed)
                )
                return True
            else:
                self.logger.error(
                    "Failed to compute all 9 cell centers for logic. Got %s.",
                    len(cell_centers)
                )
                return False
        except ValueError as ve:
            self.logger.error(f"ValueError during cell center calculation: {ve}")
            return False
        except IndexError as e:
            self.logger.error(
                "IndexError computing logic cell centers from _grid_points: %s", e,
                exc_info=True
            )
            return False
        except Exception as e:
            self.logger.error(
                "Error computing logic cell centers from _grid_points: %s - %s",
                type(e).__name__, e, exc_info=True
            )
            return False
