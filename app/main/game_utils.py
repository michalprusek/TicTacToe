# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Shared utility functions for the TicTacToe application.
Consolidates duplicated utility functions from multiple files.
"""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Union


def convert_board_1d_to_2d(board_1d: Union[List, Any]) -> Union[List[List], Any]:
    """
    Convert 1D board representation to 2D.

    Args:
        board_1d: Board in 1D format (list of 9 elements)

    Returns:
        Board in 2D format (3x3 nested list) or original if not valid 1D board
    """
    if isinstance(board_1d, list) and len(board_1d) == 9:
        return [board_1d[i:i + 3] for i in range(0, 9, 3)]
    return board_1d


def get_board_symbol_counts(board: Union[List[List], List]) -> Dict[str, int]:
    """
    Count symbols on the board.

    Args:
        board: Board in 1D or 2D format

    Returns:
        Dictionary with counts of each symbol
    """
    # Convert to 1D if needed
    if isinstance(board, list) and len(board) == 3 and isinstance(board[0], list):
        board_1d = [cell for row in board for cell in row]
    else:
        board_1d = board if isinstance(board, list) else []

    counts = {'X': 0, 'O': 0, ' ': 0}
    for cell in board_1d:
        if cell in counts:
            counts[cell] += 1
        else:
            counts[' '] += 1  # Count unknown as empty

    return counts


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup standardized logger configuration.

    Args:
        name: Logger name (usually __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
    return logger
