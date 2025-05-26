# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Game logic for TicTacToe game.
"""
import math
import random
from typing import List
from typing import Tuple

from app.core.minimax_algorithm import (
    get_available_moves as get_available_moves_shared
)
from app.core.minimax_algorithm import get_optimal_move_with_heuristics
from app.core.minimax_algorithm import minimax_maximize
from app.core.minimax_algorithm import minimax_minimize

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
# Constant for representing a tie
TIE = "TIE"


def create_board():
    """Creates an empty 3x3 Tic Tac Toe board."""
    return [[EMPTY for _ in range(3)] for _ in range(3)]


def get_available_moves(game_board):
    """Returns a list of available moves (row, col) on the board."""
    moves = []
    for r in range(3):
        for c in range(3):
            if game_board[r][c] == EMPTY:
                moves.append((r, c))
    return moves


def get_valid_moves(game_board):
    """Alias for get_available_moves for backward compatibility."""
    return get_available_moves(game_board)


def check_winner(game_board):
    """
    Checks if there is a winner.
    Returns: PLAYER_X, PLAYER_O, TIE, or None
    """
    # Rows
    for row in range(3):
        if (game_board[row][0] == game_board[row][1] ==
                game_board[row][2] != EMPTY):
            return game_board[row][0]
    # Columns
    for col in range(3):
        if (game_board[0][col] == game_board[1][col] ==
                game_board[2][col] != EMPTY):
            return game_board[0][col]
    # Diagonals
    if game_board[0][0] == game_board[1][1] == game_board[2][2] != EMPTY:
        return game_board[0][0]
    if game_board[0][2] == game_board[1][1] == game_board[2][0] != EMPTY:
        return game_board[0][2]
    # Tie check
    if is_board_full(game_board):
        return TIE
    return None


def get_winning_line(game_board):
    """
    Returns the winning line coordinates if there is a winner.
    Returns: List of (row, col) tuples representing the winning line, or None
    """
    # Rows
    for row in range(3):
        if (game_board[row][0] == game_board[row][1] ==
                game_board[row][2] != EMPTY):
            return [(row, 0), (row, 1), (row, 2)]
    # Columns
    for col in range(3):
        if (game_board[0][col] == game_board[1][col] ==
                game_board[2][col] != EMPTY):
            return [(0, col), (1, col), (2, col)]
    # Diagonals
    if game_board[0][0] == game_board[1][1] == game_board[2][2] != EMPTY:
        return [(0, 0), (1, 1), (2, 2)]
    if game_board[0][2] == game_board[1][1] == game_board[2][0] != EMPTY:
        return [(0, 2), (1, 1), (2, 0)]
    return None


def is_board_full(game_board):
    """Checks if the board is full."""
    return all(game_board[r][c] != EMPTY for r in range(3) for c in range(3))


def is_game_over(game_board):
    """Checks if the game is over (win or draw)."""
    return check_winner(
        game_board) is not None  # Winner check also handles TIE


def print_board(game_board):
    """Prints the board to the console."""
    print("-" * 13)
    for row in game_board:
        print(f"| {row[0] if row[0] else ' '} | "
              f"{row[1] if row[1] else ' '} | "
              f"{row[2] if row[2] else ' '} |")
        print("-" * 13)


# --- AI Strategies ---


def get_random_move(
        game_board, player=None):  # pylint: disable=unused-argument
    """Chooses a random valid move."""
    available_moves = get_available_moves(game_board)
    if not available_moves:
        return None
    return random.choice(available_moves)


def get_other_player(player):
    """Returns the opposing player."""
    return PLAYER_O if player == PLAYER_X else PLAYER_X


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def minimax(game_board, player_or_depth, is_maximizing_or_player=None,
            alpha_or_depth=None, beta=None, ai_player=None):
    """
    Minimax algorithm with Alpha-Beta Pruning.
    Supports both old and new parameter formats for backward compatibility:

    Old format: minimax(game_board, depth, is_maximizing_player, alpha, beta,
                        ai_player)
    New format: minimax(game_board, player, depth, alpha, beta, ai_player)

    Returns: (score, move) - move is None here, only score matters for
             recursion
    """
    # Handle both old and new parameter formats
    if isinstance(player_or_depth, int):
        # Old format: minimax(game_board, depth, is_maximizing_player, alpha,
        # beta, ai_player)
        depth = player_or_depth
        is_maximizing_player = is_maximizing_or_player
        alpha = alpha_or_depth
        player = ai_player  # In old format, ai_player is the player
    else:
        # New format: minimax(game_board, player, depth, alpha, beta,
        # ai_player)
        player = player_or_depth
        depth = is_maximizing_or_player
        alpha = alpha_or_depth
        is_maximizing_player = player == ai_player

    winner = check_winner(game_board)
    human_player = get_other_player(ai_player)

    if winner == ai_player:
        return 10 - depth, None  # AI wins, prioritize faster wins
    if winner == human_player:
        return depth - 10, None  # Human wins, block slower losses
    if winner == TIE:  # is_board_full check is implicit in check_winner
        return 0, None          # Draw

    available_moves = get_available_moves_shared(game_board)

    # Create a wrapper for recursive calls
    def minimax_recursive(board_state, current_player, depth, *, alpha, beta,
                          ai_player):
        return minimax(
            board_state, depth,
            is_maximizing_or_player=current_player == ai_player,
            alpha_or_depth=alpha, beta=beta, ai_player=ai_player)

    minimax_args = {
        'board': game_board, 'available_moves': available_moves,
        'depth': depth, 'alpha': alpha, 'beta': beta, 'ai_player': ai_player,
        'minimax_func': minimax_recursive
    }

    if is_maximizing_player:
        return minimax_maximize(**minimax_args)
    return minimax_minimize(**minimax_args)


def get_best_move(game_board, player):
    """
    Finds the best move using the Minimax algorithm with Alpha-Beta Pruning.
    """
    if is_board_full(game_board):
        return None

    # Try heuristics first
    heuristic_move = get_optimal_move_with_heuristics(game_board)
    if heuristic_move:
        return heuristic_move

    # Call minimax to find the best move
    _, best_move = minimax(
        game_board, player,
        is_maximizing_or_player=True,
        alpha_or_depth=-math.inf,
        beta=math.inf,
        ai_player=player
    )

    return best_move


def board_to_string(game_board):
    """Converts the board list of lists into a simple string representation."""
    # For unit tests, we need to remove empty spaces
    result = ""
    for row in game_board:
        for cell in row:
            if cell != EMPTY:
                result += cell
    return result


def get_board_diff(prev_board: List[List[str]],
                   curr_board: List[List[str]]) -> List[Tuple[int, int, str]]:
    """Compares two boards and returns a list of changes (r, c, new_symbol)."""
    diff = []
    for r in range(3):
        for c in range(3):
            if prev_board[r][c] != curr_board[r][c]:
                # Allow change only if previous was empty
                if prev_board[r][c] == EMPTY and curr_board[r][c] != EMPTY:
                    diff.append((r, c, curr_board[r][c]))
                # Ignore other changes (like removals or AI overwriting itself)
    return diff


# --- Example Usage (can be removed later) ---
if __name__ == '__main__':
    board = create_board()
    board[0][0] = PLAYER_X  # Example move

    print("Initial Board:")
    print_board(board)

    print("\nAvailable Moves:", get_available_moves(board))

    print("\nRandom move for O:")
    move_o_random = get_random_move(board, PLAYER_O)
    if move_o_random:
        board[move_o_random[0]][move_o_random[1]] = PLAYER_O
        print_board(board)
    else:
        print("No available moves.")

    # Test minimax
    board = create_board()
    board[1][1] = PLAYER_X  # Human plays center
    print("\nTest Minimax for O after human plays center:")
    print_board(board)
    best_move_o = get_best_move(board, PLAYER_O)
    print(f"Minimax suggests O plays at: {best_move_o}")
    if best_move_o:
        board[best_move_o[0]][best_move_o[1]] = PLAYER_O
        print_board(board)

    # Test win check
    board[0][0] = PLAYER_X
    board[0][1] = PLAYER_X
    board[0][2] = PLAYER_X
    print("\nTesting Win Check (X wins row 0):")
    print_board(board)
    print(f"Winner: {check_winner(board)}")
    print(f"Game Over: {is_game_over(board)}")

    # Test draw check
    board = [
        [PLAYER_X, PLAYER_O, PLAYER_X],
        [PLAYER_X, PLAYER_O, PLAYER_O],
        [PLAYER_O, PLAYER_X, PLAYER_X]
    ]
    print("\nTesting Draw Check:")
    print_board(board)
    print(f"Winner: {check_winner(board)}")
    print(f"Board Full: {is_board_full(board)}")
    print(f"Game Over: {is_game_over(board)}")
