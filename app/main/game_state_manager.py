"""
Game state manager module for the TicTacToe application.
"""
import logging
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import cv2

from app.core.game_state import GameState


class GameStateManager:
    """Manages the game state based on detection results."""

    def __init__(self, config=None, logger=None):
        """Initialize the game state manager.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize game state
        self.game_state = GameState()
        
        # Class ID to player mapping
        self.class_id_to_player = getattr(config, 'class_id_to_player', {0: 1, 1: 2})  # X=1, O=2

    def update_game_state(
        self,
        frame: np.ndarray,
        ordered_kpts_uv: Optional[np.ndarray],
        homography: Optional[np.ndarray],
        detected_symbols: List[Any],
        timestamp: float,
        grid_status_changed: bool = False
    ) -> Optional[List[np.ndarray]]:
        """Updates the game state based on detection results.
        
        Args:
            frame: Current video frame
            ordered_kpts_uv: Ordered keypoints in UV coordinates
            homography: Homography matrix from ideal grid to image
            detected_symbols: List of detected symbols
            timestamp: Current timestamp
            grid_status_changed: Whether grid status changed significantly
            
        Returns:
            List of cell polygons derived from game state, or None
        """
        # If grid status changed significantly, reset game state
        if grid_status_changed:
            self.logger.info("Grid status changed significantly, resetting game state")
            self.game_state.reset()
        
        # Always reset changed cells for the current detection cycle
        self.game_state.reset_changed_cells()
        
        # Update game state with detection results
        self.game_state.update_from_detection(
            frame,
            ordered_kpts_uv,
            homography,
            detected_symbols,
            self.class_id_to_player,
            timestamp
        )
        
        # Retrieve derived cell polygons from game state if available
        if hasattr(self.game_state, 'get_latest_derived_cell_polygons'):
            polygons = self.game_state.get_latest_derived_cell_polygons()
            if polygons is not None:
                return polygons
        
        return None

    def get_board_state(self) -> List[List[str]]:
        """Gets the current board state.
        
        Returns:
            2D list representing the board state
        """
        if self.game_state and hasattr(self.game_state, 'board'):
            return self.game_state.board
        
        # Return empty board if game state is not available
        return [['', '', ''], ['', '', ''], ['', '', '']]

    def is_valid(self) -> bool:
        """Checks if the game state is valid.
        
        Returns:
            True if the game state is valid, False otherwise
        """
        if self.game_state:
            return self.game_state.is_valid()
        return False

    def get_winner(self) -> Optional[str]:
        """Gets the winner of the game.
        
        Returns:
            Winner symbol ('X' or 'O') or None if no winner
        """
        if self.game_state and hasattr(self.game_state, 'winner'):
            return self.game_state.winner
        return None

    def is_grid_visible(self) -> bool:
        """Checks if the grid is visible.
        
        Returns:
            True if the grid is visible, False otherwise
        """
        if self.game_state and hasattr(self.game_state, 'is_grid_visible'):
            return self.game_state.is_grid_visible
        return False

    def is_grid_stable(self) -> bool:
        """Checks if the grid is stable.
        
        Returns:
            True if the grid is stable, False otherwise
        """
        if self.game_state and hasattr(self.game_state, 'is_grid_stable'):
            return self.game_state.is_grid_stable
        return False

    def get_cell_polygons(self) -> Optional[List[np.ndarray]]:
        """Gets the cell polygons.
        
        Returns:
            List of cell polygons or None if not available
        """
        if self.game_state and hasattr(self.game_state, 'cell_polygons'):
            return self.game_state.cell_polygons
        return None

    def get_grid_points(self) -> Optional[np.ndarray]:
        """Gets the grid points.
        
        Returns:
            Array of grid points or None if not available
        """
        if self.game_state and hasattr(self.game_state, '_grid_points'):
            return self.game_state._grid_points
        return None

    def get_cell_center(self, row: int, col: int) -> Optional[Tuple[float, float]]:
        """Gets the center coordinates of a cell.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Tuple of (x, y) coordinates or None if not available
        """
        if self.game_state and hasattr(self.game_state, 'get_cell_center_uv'):
            return self.game_state.get_cell_center_uv(row, col)
        return None

    def has_grid_issue(self) -> bool:
        """Checks if there is an issue with the grid.
        
        Returns:
            True if there is a grid issue, False otherwise
        """
        if self.game_state:
            return (hasattr(self.game_state, 'grid_issue_type') and 
                    self.game_state.grid_issue_type is not None)
        return False

    def get_grid_issue_message(self) -> Optional[str]:
        """Gets the grid issue message.
        
        Returns:
            Grid issue message or None if no issue
        """
        if self.game_state and hasattr(self.game_state, 'grid_issue_message'):
            return self.game_state.grid_issue_message
        return None
