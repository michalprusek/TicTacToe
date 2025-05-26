"""
Shared minimax algorithm implementation for TicTacToe.
This module contains the common minimax logic to avoid code duplication.
"""
# pylint: disable=too-many-return-statements,too-many-arguments
import math
from typing import List, Tuple, Optional

EMPTY = " "


def get_available_moves(board: List[List[str]]) -> List[Tuple[int, int]]:
    """Get all available moves on the board."""
    moves = []
    for r in range(3):
        for c in range(3):
            if board[r][c] == EMPTY:
                moves.append((r, c))
    return moves


def is_board_full(board: List[List[str]]) -> bool:
    """Check if the board is full."""
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    return True


def evaluate_board(board: List[List[str]], ai_player: str) -> int:
    """Evaluate the board state for the AI player."""
    human_player = "O" if ai_player == "X" else "X"

    # Check rows
    for row in board:
        if all(cell == ai_player for cell in row):
            return 1
        if all(cell == human_player for cell in row):
            return -1

    # Check columns
    for col in range(3):
        if all(board[row][col] == ai_player for row in range(3)):
            return 1
        if all(board[row][col] == human_player for row in range(3)):
            return -1

    # Check diagonals
    if all(board[i][i] == ai_player for i in range(3)):
        return 1
    if all(board[i][i] == human_player for i in range(3)):
        return -1

    if all(board[i][2-i] == ai_player for i in range(3)):
        return 1
    if all(board[i][2-i] == human_player for i in range(3)):
        return -1

    return 0  # Draw or game continues


def minimax_maximize(board: List[List[str]], available_moves: List[Tuple[int, int]],
                    depth: int, *, alpha: float, beta: float, ai_player: str,
                    minimax_func) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Handle maximizing player logic in minimax."""
    best_score = -math.inf
    best_move = None
    human_player = "O" if ai_player == "X" else "X"

    for move in available_moves:
        r, c = move
        board[r][c] = ai_player
        score, _ = minimax_func(
            board, human_player, depth + 1, alpha=alpha, beta=beta, ai_player=ai_player
        )
        board[r][c] = EMPTY  # Undo move

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if beta <= alpha:
            break  # Beta cut-off

    return best_score, best_move


def minimax_minimize(board: List[List[str]], available_moves: List[Tuple[int, int]],
                    depth: int, *, alpha: float, beta: float, ai_player: str,
                    minimax_func) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Handle minimizing player logic in minimax."""
    best_score = math.inf
    best_move = None
    human_player = "O" if ai_player == "X" else "X"

    for move in available_moves:
        r, c = move
        board[r][c] = human_player
        score, _ = minimax_func(
            board, ai_player, depth + 1, alpha=alpha, beta=beta, ai_player=ai_player
        )
        board[r][c] = EMPTY  # Undo move

        if score < best_score:
            best_score = score
            best_move = move

        beta = min(beta, score)
        if beta <= alpha:
            break  # Alpha cut-off

    return best_score, best_move


def get_optimal_move_with_heuristics(board: List[List[str]]) -> Optional[Tuple[int, int]]:
    """Get optimal move with common heuristics applied."""
    available_moves = get_available_moves(board)
    if not available_moves:
        return None

    # Handle case where only one move is left (optimization)
    if len(available_moves) == 1:
        return available_moves[0]

    # If it's the first move, play center for speed
    if len(available_moves) == 9:
        return (1, 1)  # Center

    # If it's the second move and center is available, take it
    if len(available_moves) == 8 and board[1][1] == EMPTY:
        return (1, 1)

    return None  # No heuristic applies, use full minimax
