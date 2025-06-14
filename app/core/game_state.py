# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Game state module for the TicTacToe application.
"""
# pylint: disable=logging-too-many-args
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from .grid_utils import robust_sort_grid_points
from .symbol_cache import SymbolCache

# Constants for game state
EMPTY = ' '
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


# pylint: disable=too-many-public-methods,too-many-instance-attributes
class GameState:
    """Class to represent and manage the game state."""
    ERROR_GRID_INCOMPLETE_PAUSE = "GRID_INCOMPLETE_PAUSE_STATE"  # Unique ID

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
        self.grid_fully_visible: bool = False  # All grid points detected
        self.missing_grid_points_count: int = 0  # Missing grid points

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
        # Cell polygons in UV space, transformed from ideal grid
        self._cell_polygons_uv_transformed: Optional[List[np.ndarray]] = None

        # Current frame for processing
        self._frame: Optional[np.ndarray] = None

        # Symbol confidence threshold for filtering
        self.symbol_confidence_threshold: float = 0.90

        # Symbol caching system for handling occlusion
        self.symbol_cache = SymbolCache(self.logger)
        
        # Flag to ignore symbols after grid reset
        self._frames_since_reset = 0
        self._ignore_symbols_frames = 5  # Ignore symbols for 5 frames after reset

    def reset_game(self):
        """Resets game state, keeping grid points if valid."""
        self.logger.info("Resetting game state")
        self._board_state = [[EMPTY for _ in range(3)] for _ in range(3)]
        self._detection_results = []
        self._changed_cells_this_turn = []
        # Keep grid_points and homography if still valid
        self.error_message = None  # Reset error message
        self._last_move_timestamp = None  # Reset cooldown
        self.game_paused_due_to_incomplete_grid = False  # Reset pause state
        self.grid_fully_visible = False  # Reset grid visibility status
        self.missing_grid_points_count = 0  # Reset missing points count
        self.winner = None
        self.winning_line_indices = None

        # Clear symbol cache on game reset
        self.symbol_cache.clear_cache()
        
        # Reset frame counter to ignore symbols after reset
        self._frames_since_reset = 0

    @property
    def board(self) -> List[List[str]]:
        """Get the current board state as a 2D list."""
        return [row[:] for row in self._board_state]

    def get_board_for_ai(self) -> List[List[str]]:
        """
        Get board state optimized for AI decision making.

        This method returns the cached board state which includes symbols
        that may be temporarily occluded but are still part of the game state.
        This ensures AI makes decisions based on complete information.

        Returns:
            Board state with cached symbol information
        """
        # If we have cached symbols, use them for AI decisions
        if (hasattr(self.symbol_cache, 'cached_symbols') and
                self.symbol_cache.cached_symbols):
            cached_board = self.symbol_cache._build_board_from_cache()
            self.logger.debug(
                "🤖 AI using cached board state with %d symbols",
                len(self.symbol_cache.cached_symbols)
            )
            return cached_board

        # Otherwise use current board state
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

    # pylint: disable=too-many-branches
    def is_physical_grid_valid(self) -> bool:
        """Check if the detected physical grid is valid."""
        return self._is_valid_grid

    def is_game_paused_due_to_incomplete_grid(self) -> bool:
        """Check if the game is paused because grid is not fully visible."""
        return self.game_paused_due_to_incomplete_grid

    def is_valid(self) -> bool:
        """Check if the current logical game board state is valid (symbols)."""
        for r in range(3):
            for c in range(3):
                symbol = self._board_state[r][c]
                if symbol not in [PLAYER_X, PLAYER_O, EMPTY]:
                    self.logger.warning(
                        "Invalid symbol '%s' found at (%d,%d)",
                        symbol, r, c
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

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves (empty cells) on the board."""
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if self._board_state[r][c] == EMPTY:
                    valid_moves.append((r, c))
        return valid_moves

    def check_winner(self) -> Optional[str]:
        """Check if there is a winner on the board."""
        # Check rows
        for row in range(3):
            if (self._board_state[row][0] == self._board_state[row][1] ==
                    self._board_state[row][2] != EMPTY):
                return self._board_state[row][0]

        # Check columns
        for col in range(3):
            if (self._board_state[0][col] == self._board_state[1][col] ==
                    self._board_state[2][col] != EMPTY):
                return self._board_state[0][col]

        # Check diagonals
        if (self._board_state[0][0] == self._board_state[1][1] ==
                self._board_state[2][2] != EMPTY):
            return self._board_state[0][0]
        if (self._board_state[0][2] == self._board_state[1][1] ==
                self._board_state[2][0] != EMPTY):
            return self._board_state[0][2]

        return None

    def is_board_full(self) -> bool:
        """Check if the board is full."""
        return all(
            self._board_state[r][c] != EMPTY
            for r in range(3) for c in range(3)
        )

    def board_to_string(self) -> str:
        """Convert the board to a string representation."""
        result = ""
        for row in self._board_state:
            for cell in row:
                result += cell if cell != EMPTY else " "
            result += "\n"
        return result.strip()

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
            A list of (row, col) tuples for cells that were changed in this
            update.
        """
        if cell_centers_uv is None or len(cell_centers_uv) != 9:
            self.logger.warning(
                "Cannot update board: cell_centers_uv is invalid or not 9 "
                "points."
            )
            return []

        # Convert symbols to expected format
        detected_symbols = self._convert_symbols_to_expected_format(
            detected_symbols, class_id_to_player)

        changed_cells = []

        # Debug: Log current board state
        board_str = str([row[:] for row in self._board_state])
        self.logger.info(
            "Current board state before symbol update: %s", board_str
        )

        # ✅ CRITICAL DEBUG: Log all YOLO detections for verification
        # (fallback method)
        self.logger.info(
            "🔍 FALLBACK METHOD - YOLO DETECTED %d SYMBOLS",
            len(detected_symbols)
        )

        # Filter out low-confidence detections (fallback method)
        detected_symbols = self._filter_high_confidence_symbols(
            detected_symbols
        )
        self.logger.info(
            "🔍 FALLBACK AFTER FILTERING: %d HIGH-CONFIDENCE SYMBOLS",
            len(detected_symbols)
        )

        for symbol_info in detected_symbols:  # Use unsorted for now
            symbol_center_uv = symbol_info.get('center_uv')
            player = symbol_info.get('player')

            if symbol_center_uv is None or player is None:
                self.logger.warning(
                    "Skipping symbol due to missing data: %s", symbol_info
                )
                continue

            # Find the closest cell center to this symbol
            distances = np.linalg.norm(
                cell_centers_uv - symbol_center_uv, axis=1
            )
            closest_cell_idx = np.argmin(distances)

            row, col = divmod(closest_cell_idx, 3)

            # Check if the cell is empty before placing the symbol
            if self._board_state[row][col] == EMPTY:
                self._board_state[row][col] = player
                changed_cells.append((row, col))
                self.logger.info(
                    "Placed '%s' at (%d,%d) based on symbol at %s",
                    player, row, col, symbol_center_uv
                )
            else:
                self.logger.info(
                    "Cell (%d,%d) occupied by %s. Symbol %s not placed.",
                    row, col, self._board_state[row][col], player
                )
            # else:
            #     self.logger.debug(
            #         f"Cell ({row},{col}) already occupied by "
            #         f"{self.board[row][col]}. Symbol {player} not placed."
            #     )

        # if not changed_cells:
        #     self.logger.debug("No cells were changed in this update cycle.")

        return changed_cells

    def _convert_symbols_to_expected_format(
            self,
            detected_symbols: List[Dict],
            # pylint: disable=unused-argument
            class_id_to_player: Dict[int, str]
    ) -> List[Dict]:
        """Convert symbols from detector format to expected format.

        Args:
            detected_symbols: Symbols from detector with 'box', 'label',
                'confidence', 'class_id'
            class_id_to_player: Mapping from class ID to player symbol

        Returns:
            Symbols with 'center_uv', 'player', 'confidence' format
        """
        converted_symbols = []

        for symbol in detected_symbols:
            # Check if symbol has the expected detector format
            if all(
                key in symbol
                for key in ['box', 'label', 'confidence', 'class_id']
            ):
                # Calculate center from bounding box
                box = symbol['box']
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2

                # Convert class_id to player symbol
                player = symbol['label']  # Use label directly (X or O)

                converted_symbol = {
                    'center_uv': np.array([center_x, center_y]),
                    'player': player,
                    'confidence': symbol['confidence']
                }
                converted_symbols.append(converted_symbol)
            elif all(
                key in symbol
                for key in ['center_uv', 'player', 'confidence']
            ):
                # Already in expected format
                converted_symbols.append(symbol)
            else:
                self.logger.warning("Symbol has unexpected format: %s", symbol)

        return converted_symbols

    def _filter_high_confidence_symbols(
            self, detected_symbols: List[Dict]
    ) -> List[Dict]:
        """Filter symbols based on confidence threshold."""
        # Use dynamic threshold if available, otherwise default to 0.90
        min_confidence_threshold = getattr(
            self, 'symbol_confidence_threshold', 0.90
        )
        filtered_symbols = []

        for i, symbol_info in enumerate(detected_symbols):
            center = symbol_info.get('center_uv', 'Unknown')
            player = symbol_info.get('player', 'Unknown')
            confidence = symbol_info.get('confidence', 0.0)

            self.logger.info(
                "  Symbol %d: %s at %s (confidence: %.3f)",
                i + 1, player, center, confidence
            )

            # Only accept high-confidence detections
            if confidence >= min_confidence_threshold:
                filtered_symbols.append(symbol_info)
                self.logger.info(
                    "    ✅ ACCEPTED (confidence >= %.3f)",
                    min_confidence_threshold
                )
            else:
                self.logger.warning(
                    "    ❌ REJECTED (confidence %.3f < %.3f)",
                    confidence, min_confidence_threshold
                )

        self.logger.info(
            "🔍 AFTER FILTERING: %d HIGH-CONFIDENCE SYMBOLS",
            len(filtered_symbols)
        )
        return filtered_symbols

    def _process_symbol_placement(
            self, symbol_info: Dict, homography: np.ndarray, cell_size: int
    ) -> Optional[Tuple[int, int]]:
        """Process symbol placement and return changed cell if any."""
        symbol_center_uv = symbol_info.get('center_uv')
        player = symbol_info.get('player')

        if symbol_center_uv is None or player is None:
            self.logger.warning(
                "Skipping symbol due to missing data: %s", symbol_info
            )
            return None

        # Transform symbol center to normalized space
        symbol_center_img = np.array(
            [[[symbol_center_uv[0], symbol_center_uv[1]]]],
            dtype=np.float32
        )
        # pylint: disable=no-member
        symbol_center_norm_final = cv2.perspectiveTransform(
            symbol_center_img, homography
        )
        nx, ny = symbol_center_norm_final[0][0]

        # Map to grid positions (0-3)
        grid_col = int(nx / cell_size)
        grid_row = int(ny / cell_size)

        # Clamp to valid grid indices (0-3)
        grid_col = max(0, min(3, grid_col))
        grid_row = max(0, min(3, grid_row))

        # Map from grid indices to game cell indices (0-2)
        if grid_row <= 2 and grid_col <= 2:
            final_row = grid_row
            final_col = grid_col
        else:
            # Symbol is outside game area (on grid edge)
            self.logger.warning(
                "ROBUST: Symbol at %s mapped to grid (%d,%d) which is "
                "outside game area. Normalized coords: (%.1f,%.1f)",
                symbol_center_uv, grid_row, grid_col, nx, ny
            )
            return None

        # Validate final coordinates
        if 0 <= final_row < 3 and 0 <= final_col < 3:
            # Check if cell is empty
            if self._board_state[final_row][final_col] == EMPTY:
                self._board_state[final_row][final_col] = player
                self.logger.info(
                    "ROBUST: Placed '%s' at (%d,%d) based on symbol "
                    "at %s -> normalized (%.1f,%.1f) -> grid (%d,%d)",
                    player, final_row, final_col, symbol_center_uv,
                    nx, ny, grid_row, grid_col
                )
                return (final_row, final_col)
            self.logger.info(
                "ROBUST: Cell (%d,%d) already occupied by %s. "
                "Symbol %s not placed.",
                final_row, final_col,
                self._board_state[final_row][final_col], player
            )
            return None
        self.logger.warning(
            "ROBUST: Symbol at %s mapped to invalid cell (%d,%d). "
            "Normalized coords: (%.1f,%.1f)",
            symbol_center_uv, final_row, final_col, nx, ny
        )
        return None

    def _synchronize_board_with_detections(
            self,
            detected_symbols: List[Dict],
            class_id_to_player: Dict[int, str]
    ) -> List[Tuple[int, int]]:
        """
        Synchronize board state with current YOLO detections.

        This method ensures that the board state only contains symbols that are
        currently detected by YOLO with sufficient confidence. It rebuilds the
        board state from scratch based on current detections, preventing
        phantom symbols from persisting in the GUI.

        Args:
            detected_symbols: List of symbols detected by YOLO in current frame
            class_id_to_player: Mapping from class ID to player symbol

        Returns:
            List of (row, col) tuples representing cells that changed
        """
        # Store old board state for comparison
        old_board_state = [row[:] for row in self._board_state]

        if not detected_symbols:
            # No symbols detected - clear the board completely
            self.logger.info("🧹 No symbols detected - clearing board state")
            self._board_state = [[EMPTY for _ in range(3)] for _ in range(3)]
            return self._get_changed_cells(old_board_state, self._board_state)

        # Convert symbols to expected format and filter by confidence
        converted_symbols = self._convert_symbols_to_expected_format(
            detected_symbols, class_id_to_player
        )
        filtered_symbols = self._filter_high_confidence_symbols(converted_symbols)

        if not filtered_symbols:
            # No high-confidence symbols - clear the board
            self.logger.info("🧹 No high-confidence symbols - clearing board state")
            self._board_state = [[EMPTY for _ in range(3)] for _ in range(3)]
            return self._get_changed_cells(old_board_state, self._board_state)

        # Rebuild board state from current detections only
        self.logger.info("🔄 Rebuilding board state from %d current detections",
                        len(filtered_symbols))

        # Clear board first
        new_board_state = [[EMPTY for _ in range(3)] for _ in range(3)]

        # Map each detected symbol to board position
        for symbol_info in filtered_symbols:
            symbol_center_uv = symbol_info.get('center_uv')
            player = symbol_info.get('player')
            confidence = symbol_info.get('confidence', 0.0)

            if symbol_center_uv is None or player is None:
                continue

            # Find closest cell center
            min_distance = float('inf')
            closest_cell = None

            for i, cell_center in enumerate(self._cell_centers_uv_transformed):
                distance = np.linalg.norm(
                    np.array(symbol_center_uv) - np.array(cell_center)
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_cell = i

            if closest_cell is not None:
                row, col = closest_cell // 3, closest_cell % 3
                # Only place symbol if cell is empty in new board
                if new_board_state[row][col] == EMPTY:
                    new_board_state[row][col] = player
                    self.logger.info(
                        "🎯 Placed %s at (%d,%d) from detection (conf: %.3f)",
                        player, row, col, confidence
                    )
                else:
                    self.logger.warning(
                        "⚠️ Cell (%d,%d) already occupied in new board state",
                        row, col
                    )

        # Update board state
        self._board_state = new_board_state

        # Log final board state
        board_str = str([row[:] for row in self._board_state])
        self.logger.info("🎮 Synchronized board state: %s", board_str)

        # Return changed cells
        return self._get_changed_cells(old_board_state, self._board_state)

    def _synchronize_board_with_cache(
            self,
            detected_symbols: List[Dict],
            class_id_to_player: Dict[int, str],
            timestamp: float
    ) -> List[Tuple[int, int]]:
        """
        Synchronize board state using intelligent symbol caching.

        This method uses the symbol cache to maintain symbol memory even when
        they become temporarily occluded by the player's hand, ensuring
        consistent AI decision-making based on complete board state.

        Args:
            detected_symbols: List of symbols detected by YOLO in current frame
            class_id_to_player: Mapping from class ID to player symbol
            timestamp: Current timestamp

        Returns:
            List of (row, col) tuples representing cells that changed
        """
        # Store old board state for comparison
        old_board_state = [row[:] for row in self._board_state]

        # Use symbol cache to get board state with occlusion handling
        if self._cell_centers_uv_transformed is not None:
            cached_board, using_cached_data = self.symbol_cache.update_cache(
                detected_symbols,
                self._cell_centers_uv_transformed,
                timestamp,
                class_id_to_player
            )

            # Update board state from cache
            self._board_state = cached_board

            # Log cache usage for debugging
            if using_cached_data:
                cache_status = self.symbol_cache.get_cache_status()
                self.logger.info(
                    "🔄 Using cached symbol data: %d cached symbols at %s",
                    cache_status['cached_count'],
                    cache_status['cached_positions']
                )
            else:
                self.logger.debug("🎯 Using live detection data only")
        else:
            # Fallback to old synchronization if no cell centers available
            self.logger.warning(
                "No cell centers available, falling back to basic synchronization")
            return self._synchronize_board_with_detections(
                detected_symbols, class_id_to_player
            )

        # Return changed cells
        return self._get_changed_cells(old_board_state, self._board_state)

    def _get_changed_cells(
            self,
            old_board: List[List[str]],
            new_board: List[List[str]]
    ) -> List[Tuple[int, int]]:
        """
        Compare two board states and return list of changed cells.

        Args:
            old_board: Previous board state
            new_board: Current board state

        Returns:
            List of (row, col) tuples representing cells that changed
        """
        changed_cells = []
        for row in range(3):
            for col in range(3):
                if old_board[row][col] != new_board[row][col]:
                    changed_cells.append((row, col))
                    self.logger.debug(
                        "Cell (%d,%d) changed: '%s' → '%s'",
                        row, col, old_board[row][col], new_board[row][col]
                    )
        return changed_cells

    def _update_board_with_symbols_robust(
            self,
            detected_symbols: List[Dict],
            grid_points: np.ndarray,
            class_id_to_player: Dict[int, str]
    ) -> List[Tuple[int, int]]:
        """
        Robustní aktualizace herní desky pomocí homografie.

        Používá robustní homografii pro přesné mapování symbolů na buňky,
        nezávisle na rotaci nebo perspektivě herní plochy.

        Args:
            detected_symbols: Seznam detekovaných symbolů
            grid_points: 16 seřazených grid points z kamery
            class_id_to_player: Mapování class ID na hráče

        Returns:
            Seznam (row, col) tuples pro změněné buňky
        """
        if grid_points is None or len(grid_points) != 16:
            self.logger.debug(
                "Cannot use robust symbol mapping: invalid grid points"
            )
            return []

        if not detected_symbols:
            return []

        # Convert symbols to expected format
        detected_symbols = self._convert_symbols_to_expected_format(
            detected_symbols, class_id_to_player)

        try:
            # Použijeme robustní funkci pro získání homografie
            _, h_final = robust_sort_grid_points(
                grid_points, self.logger
            )

            if h_final is None:
                self.logger.debug(
                    "Robust homography failed, falling back to center-based "
                    "mapping"
                )
                return []

            # Konstanta pro velikost buňky v normalizovaném prostoru
            cell_size_final = 100  # Stejná jako v robust_sort_grid_points

            changed_cells = []

            # Debug: Log current board state
            board_str = str([row[:] for row in self._board_state])
            self.logger.info(
                "Current board state before robust symbol update: %s",
                board_str
            )

            # ✅ CRITICAL DEBUG: Log all YOLO detections for verification
            self.logger.info(
                "🔍 YOLO DETECTED %d SYMBOLS", len(detected_symbols)
            )

            # Filter out low-confidence detections
            detected_symbols = self._filter_high_confidence_symbols(
                detected_symbols
            )

            for symbol_info in detected_symbols:
                changed_cell = self._process_symbol_placement(
                    symbol_info, h_final, cell_size_final
                )
                if changed_cell:
                    changed_cells.append(changed_cell)

            return changed_cells

        except (ValueError, IndexError, TypeError) as e:
            self.logger.error("Error in robust symbol mapping: %s", e)
            return []

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
        Set an error message. Prepends FATAL: if it is a game-ending error.
        Avoids overwriting a FATAL error with a non-FATAL one.
        """
        if (self.error_message and
                self.error_message.startswith("FATAL:") and
                not message.startswith("FATAL:")):
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
            self.logger.info("Clearing error: %s", self.error_message)
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

    def update_from_detection(  # pylint: disable=too-many-arguments
        self,
        frame: np.ndarray,  # pylint: disable=unused-argument
        ordered_kpts_uv: Optional[np.ndarray],
        homography: Optional[np.ndarray],
        *,
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
        # Reset changed cells for this update cycle
        self._changed_cells_this_turn = []
        
        # Increment frame counter
        self._frames_since_reset += 1

        # Check for complete grid detection first
        if (ordered_kpts_uv is None or
                len(ordered_kpts_uv) != GRID_POINTS_COUNT):
            self._is_valid_grid = False  # Physical grid is not valid
            self.game_paused_due_to_incomplete_grid = True
            self.error_message = self.ERROR_GRID_INCOMPLETE_PAUSE
            # Clear any existing grid-dependent data as it's no longer valid
            self._grid_points = None
            self._homography = None
            self._transformed_grid_points_for_drawing = None
            self._cell_centers_uv_transformed = None
            self._cell_polygons_uv_transformed = None
            points_count = (len(ordered_kpts_uv) if ordered_kpts_uv is not None
                            else 0)
            self.logger.warning(
                "Grid not fully detected (%d points). Game paused.",
                points_count
            )
            return  # Stop further processing for this frame

        # Grid is fully detected, proceed with normal logic
        self._is_valid_grid = True  # Physical grid is now considered valid
        self._grid_points = ordered_kpts_uv  # Store the valid grid points

        # If game was paused, now it's resuming
        if self.game_paused_due_to_incomplete_grid:
            self.logger.info("Full grid detected. Resuming game logic.")
            if self.error_message == self.ERROR_GRID_INCOMPLETE_PAUSE:
                self.clear_error_message()  # Clear the specific pause message
        self.game_paused_due_to_incomplete_grid = False

        if homography is not None:
            self._homography = homography

        # If the physical grid is not valid (e.g. due to ordering issues,
        # though covered above), do not proceed with game logic updates.
        # This check is somewhat redundant now due to the explicit check
        # above but kept for safety.
        if not self.is_physical_grid_valid():
            self.logger.debug(
                "Physical grid not valid. Skipping board update & win/draw "
                "checks."
            )
            # Do not reset winner; game might be paused with a winner displayed
            return

        # Compute cell centers in UV space if grid points are valid
        # This should only happen if is_physical_grid_valid() is true
        if (self._grid_points is not None and
                len(self._grid_points) == GRID_POINTS_COUNT):
            success_transform_logic = self._compute_grid_transformation()
            if not success_transform_logic:
                self.logger.warning(
                    "Failed to compute grid transformation for game logic."
                )
                return  # Cannot proceed without cell centers for logic
        else:
            # This case should ideally not be reached if the top check is
            # working
            self.logger.warning(
                "Grid points are not valid for computing cell centers "
                "(should have been caught)."
            )
            return

        # Update board with symbols using robust homography if available
        if (self._cell_centers_uv_transformed is not None and
                len(self._cell_centers_uv_transformed) == 9):
            # Ignore symbols for a few frames after reset to avoid false detections
            if self._frames_since_reset <= self._ignore_symbols_frames:
                self.logger.debug(
                    "Ignoring symbols - frame %d/%d after reset",
                    self._frames_since_reset, self._ignore_symbols_frames
                )
            # Check for cooldown before allowing a move
            elif (self._last_move_timestamp is None or
                    (timestamp - self._last_move_timestamp) >=
                    self._move_cooldown_seconds):

                # CRITICAL FIX: Use symbol cache for intelligent symbol persistence
                # This handles occlusion and maintains symbol memory during hand movements
                changed_cells = self._synchronize_board_with_cache(
                    detected_symbols, class_id_to_player, timestamp
                )

                # Log synchronization results
                if changed_cells:
                    self.logger.info(
                        "Board synchronized with %d cell changes: %s",
                        len(changed_cells), changed_cells
                    )
                else:
                    self.logger.debug(
                        "Board synchronization completed - no changes detected"
                    )

                if changed_cells:  # If a move was made
                    self._changed_cells_this_turn = changed_cells
                    # Update last move time
                    self._last_move_timestamp = timestamp
                    self.logger.info(
                        "Move made. Cooldown started. Changed cells: %s",
                        changed_cells
                    )
                else:
                    self.logger.debug(
                        "No valid move detected or board unchanged."
                    )
            else:
                cooldown_elapsed = timestamp - self._last_move_timestamp
                self.logger.debug(
                    "Move cooldown active. %.2fs / %.2fs",
                    cooldown_elapsed, self._move_cooldown_seconds
                )
        else:
            self.logger.warning(
                "Cell centers for game logic not computed. "
                "Cannot update board with symbols."
            )
            # No return here; win/draw check on existing board might be desired

        # Check for win/draw conditions only if the game is not paused and
        # board is valid
        if (not self.is_game_paused_due_to_incomplete_grid and
                self.is_physical_grid_valid()):
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

            if (self._board_state[r1][c1] == self._board_state[r2][c2] ==
                    self._board_state[r3][c3] and
                    self._board_state[r1][c1] != EMPTY):
                self.winner = self._board_state[r1][c1]
                self.winning_line_indices = line_indices
                self.logger.info(
                    "Winner: %s, Line: %s", self.winner,
                    self.winning_line_indices
                )
                return

        # Check for a draw
        if (all(self._board_state[r][c] != EMPTY for r in range(3)
                for c in range(3)) and self.winner is None):
            self.winner = "Draw"
            self.logger.info("Game is a Draw.")

    def get_latest_derived_cell_polygons(self) -> Optional[List[np.ndarray]]:
        """
        Returns the latest computed cell polygons based on transformed
        grid points.
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
            self.logger.warning(
                "Invalid cell index %d for get_cell_center_uv.", idx
            )
            return None
        self.logger.warning(
            "Cell centers not available for get_cell_center_uv."
        )
        return None

    def _calculate_cell_centers_from_grid(self) -> List[List[float]]:
        """Calculate cell centers as average of surrounding grid points."""
        cell_centers = []
        for r_cell in range(3):  # cell row
            for c_cell in range(3):  # cell col
                # Calculate grid point indices that surround this cell
                p_tl_idx = r_cell * 4 + c_cell          # top-left
                p_tr_idx = r_cell * 4 + (c_cell + 1)    # top-right
                p_bl_idx = (r_cell + 1) * 4 + c_cell    # bottom-left
                p_br_idx = (r_cell + 1) * 4 + (c_cell + 1)  # bottom-right

                # Validate indices
                if not (0 <= p_tl_idx < GRID_POINTS_COUNT and
                        0 <= p_tr_idx < GRID_POINTS_COUNT and
                        0 <= p_bl_idx < GRID_POINTS_COUNT and
                        0 <= p_br_idx < GRID_POINTS_COUNT):
                    self.logger.error(
                        "Invalid point indices for cell (%d,%d): "
                        "TL=%d, TR=%d, BL=%d, BR=%d",
                        r_cell, c_cell, p_tl_idx, p_tr_idx,
                        p_bl_idx, p_br_idx
                    )
                    raise ValueError(
                        "Invalid grid indices for cell calculation"
                    )

                # Get the four corner points of the cell
                p_tl = self._grid_points[p_tl_idx]
                p_tr = self._grid_points[p_tr_idx]
                p_bl = self._grid_points[p_bl_idx]
                p_br = self._grid_points[p_br_idx]

                # Calculate center as average of four corners
                center_x = (p_tl[0] + p_tr[0] + p_bl[0] + p_br[0]) / 4.0
                center_y = (p_tl[1] + p_tr[1] + p_bl[1] + p_br[1]) / 4.0

                cell_centers.append([center_x, center_y])

                self.logger.debug(
                    "  Cell (%d,%d): TL=%d(%.1f,%.1f) TR=%d(%.1f,%.1f) "
                    "BL=%d(%.1f,%.1f) BR=%d(%.1f,%.1f) → center=(%.1f, %.1f)",
                    r_cell, c_cell, p_tl_idx, p_tl[0], p_tl[1],
                    p_tr_idx, p_tr[0], p_tr[1], p_bl_idx, p_bl[0], p_bl[1],
                    p_br_idx, p_br[0], p_br[1], center_x, center_y
                )

        return cell_centers

    def _compute_grid_transformation(self) -> bool:
        """
        Compute cell centers in UV space if grid points are valid.
        This should only happen if is_physical_grid_valid() is true.
        Updates self._cell_centers_uv_transformed.

        Calculates cell centers as the average of the 4 grid points that
        surround each cell.
        """
        if (self._grid_points is None or
                len(self._grid_points) != GRID_POINTS_COUNT):
            points_count = (
                len(self._grid_points) if self._grid_points is not None
                else 'None'
            )
            self.logger.debug(
                "Not enough grid points for transformation. Points: %s",
                points_count
            )
            return False

        try:
            # Calculate cell centers from grid points
            cell_centers = self._calculate_cell_centers_from_grid()

            if len(cell_centers) == 9:
                self._cell_centers_uv_transformed = np.array(
                    cell_centers, dtype=np.float32
                )
                self.logger.info(
                    "Computed %d cell centers for game logic.",
                    len(self._cell_centers_uv_transformed)
                )

                # DEBUG: Log all cell centers for coordinate debugging
                self.logger.info("🗺️ CELL CENTERS DEBUG:")
                for i, center in enumerate(self._cell_centers_uv_transformed):
                    row, col = i // 3, i % 3
                    self.logger.info(
                        "Cell (%d,%d): UV=(%.1f, %.1f)",
                        row, col, center[0], center[1]
                    )

                return True
            self.logger.error(
                "Failed to compute all 9 cell centers for logic. Got %d.",
                len(cell_centers)
            )
            return False
        except ValueError as ve:
            self.logger.error(
                "ValueError during cell center calculation: %s", ve
            )
            return False
        except IndexError as e:
            self.logger.error(
                "IndexError computing cell centers: %s",
                e, exc_info=True
            )
            return False
        except (TypeError, AttributeError) as e:
            self.logger.error(
                "Error computing cell centers: %s - %s",
                type(e).__name__, e, exc_info=True
            )
            return False
