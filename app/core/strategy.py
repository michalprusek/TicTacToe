# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Strategy module for TicTacToe game AI.
"""
import logging
import math
import random
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

from .game_state import EMPTY
from .game_state import PLAYER_O
from .game_state import PLAYER_X
from .game_state import GameState
from .minimax_algorithm import get_available_moves
from .minimax_algorithm import get_optimal_move_with_heuristics
from .minimax_algorithm import minimax_maximize
from .minimax_algorithm import minimax_minimize


class Strategy(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for Tic Tac Toe strategies."""

    def __init__(self, player: str):
        self.player = player
        # Note: Keep f-string for logger name setup as it's evaluated once at init
        logger_name = f"{__name__}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)

        self.opponent = PLAYER_O if player == PLAYER_X else PLAYER_X

    @abstractmethod
    def suggest_move(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        """Suggest the next best move based on the current game state."""
        raise NotImplementedError


class RandomStrategy(Strategy):  # pylint: disable=too-few-public-methods
    """A strategy that suggests a random valid move."""

    def suggest_move(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        """Suggest a random valid move."""
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            self.logger.debug("RandomStrategy: No valid moves available.")
            return None
        move = random.choice(valid_moves)
        self.logger.debug("RandomStrategy: Suggesting move %s", move)
        return move


class MinimaxStrategy(Strategy):  # pylint: disable=too-few-public-methods
    """A strategy that uses the minimax algorithm with alpha-beta pruning for optimal play."""

    def suggest_move(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        """Suggest the optimal move using minimax algorithm with alpha-beta pruning."""
        self.logger.debug(
            "MinimaxStrategy: Suggesting move for board:\n%s", game_state.board_to_string()
        )

        board = game_state.board
        # Check if game is finished
        winner = game_state.check_winner() if hasattr(game_state, 'check_winner') else None
        is_full = game_state.is_board_full() if hasattr(game_state, 'is_board_full') else False

        if winner or is_full:
            self.logger.debug("MinimaxStrategy: Game already finished.")
            return None

        # Convert GameState board to the format expected by minimax
        board_copy = [row[:] for row in board]

        # Get the best move using minimax with alpha-beta pruning
        best_move = self._get_best_move(board_copy, self.player)

        if best_move:
            self.logger.debug("MinimaxStrategy: Selected optimal move: %s", best_move)
        else:
            self.logger.warning("MinimaxStrategy: No valid move found")

        return best_move

    def _get_best_move(self, board: List[List[str]], player: str) -> Optional[Tuple[int, int]]:
        """Find the best move using minimax algorithm with alpha-beta pruning."""
        if self._is_board_full(board):
            return None

        # Try heuristics first
        heuristic_move = get_optimal_move_with_heuristics(board)
        if heuristic_move:
            return heuristic_move

        # Call minimax to find the best move
        _, best_move = self._minimax(
            board, player, 0, alpha=-math.inf, beta=math.inf, ai_player=player
        )
        return best_move

    def _minimax(self, board: List[List[str]], current_player: str, depth: int,  # pylint: disable=too-many-arguments
                 *, alpha: float, beta: float,
                 ai_player: str) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            board: Current game board
            current_player: Player whose turn it is
            depth: Current depth in the game tree
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            ai_player: The AI player (maximizing player)

        Returns:
            Tuple of (score, best_move)
        """
        # Check for terminal states
        terminal_score = self._evaluate_terminal_state(board, depth, ai_player)
        if terminal_score is not None:
            return terminal_score, None

        available_moves = get_available_moves(board)
        is_maximizing = current_player == ai_player

        minimax_args = {
            'board': board, 'available_moves': available_moves, 'depth': depth,
            'alpha': alpha, 'beta': beta, 'ai_player': ai_player,
            'minimax_func': self._minimax
        }

        if is_maximizing:
            return minimax_maximize(**minimax_args)
        return minimax_minimize(**minimax_args)

    def _evaluate_terminal_state(self, board: List[List[str]], depth: int,
                                 ai_player: str) -> Optional[float]:
        """Evaluate terminal game states."""
        winner = self._check_winner(board)
        human_player = self._get_other_player(ai_player)

        if winner == ai_player:
            return 10 - depth  # AI wins, prioritize faster wins
        if winner == human_player:
            return depth - 10  # Human wins, penalize slower losses
        if winner == "TIE":
            return 0  # Draw
        return None  # Not a terminal state

    def _is_board_full(self, board: List[List[str]]) -> bool:
        """Check if the board is full."""
        return all(board[r][c] != EMPTY for r in range(3) for c in range(3))

    def _check_winner(self, board: List[List[str]]) -> Optional[str]:
        """Check if there is a winner on the board."""
        # Check rows
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2] != EMPTY:
                return board[row][0]

        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != EMPTY:
                return board[0][col]

        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != EMPTY:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != EMPTY:
            return board[0][2]

        # Check for tie
        if self._is_board_full(board):
            return "TIE"

        return None

    def _get_other_player(self, player: str) -> str:
        """Get the opposing player."""
        return PLAYER_O if player == PLAYER_X else PLAYER_X


class StrategySelector(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for selecting a strategy."""

    @abstractmethod
    def select_strategy(self) -> str:
        """Selects a strategy type ('basic' or 'advanced')."""
        raise NotImplementedError


class FixedStrategySelector(StrategySelector):  # pylint: disable=too-few-public-methods
    """Selects a fixed strategy."""

    def __init__(self, strategy_type: str = 'minimax'):
        self.strategy_type = strategy_type
        if strategy_type.lower() not in ['minimax', 'random']:
            raise ValueError(f"Invalid fixed strategy type specified: {strategy_type}")
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initialized FixedStrategySelector with type: %s", self.strategy_type)

    def select_strategy(self) -> str:
        return self.strategy_type.lower()


class BernoulliStrategySelector(StrategySelector):
    """Selects between random and minimax strategy based on Bernoulli trial."""

    def __init__(self, p: float = 0.5, difficulty: int = None):
        """p: probability of selecting 'minimax' strategy."""
        self.logger = logging.getLogger(__name__)

        # If difficulty is provided, use it to set p
        if difficulty is not None:
            clamped_value = max(0, min(10, difficulty))
            self._p = clamped_value / 10.0
        else:
            self._p = max(0.0, min(1.0, p))

        self.logger.debug("Initialized BernoulliStrategySelector with p=%.2f", self._p)

    @property
    def p(self) -> float:
        """Get the probability value."""
        return self._p

    @p.setter
    def p(self, value: float) -> None:
        """Set the probability value, clamped between 0 and 1."""
        self._p = max(0.0, min(1.0, value))

    @property
    def difficulty(self) -> int:
        """Get the difficulty level (0-10) derived from probability."""
        return int(round(self._p * 10))

    @difficulty.setter
    def difficulty(self, value: int) -> None:
        """Set the difficulty level (0-10), which updates probability."""
        clamped_value = max(0, min(10, value))
        self._p = clamped_value / 10.0

    def select_strategy(self) -> str:
        """
        Select a strategy based on probability.
        Returns 'minimax' or 'random' with 50/50 probability when p=0.5.
        When random value is 1: select 'minimax'
        When random value is 0: select 'random'
        """
        # Generate random value between 0 and 1
        random_value = random.random()

        # Select strategy based on probability threshold
        # If random_value < p, select minimax (intelligent play)
        # If random_value >= p, select random (random play)
        selected_strategy = 'minimax' if random_value < self._p else 'random'

        self.logger.info("🎯 STRATEGY SELECTION: p=%.2f, random=%.3f, selected='%s'",
                         self._p, random_value, selected_strategy)

        return selected_strategy

    def get_move(self, board, player):
        """
        Get a move using the selected strategy.

        Args:
            board: The current game board
            player: The current player

        Returns:
            A move (row, col) tuple
        """
        # pylint: disable=import-outside-toplevel
        from app.main import game_logic  # Import here to avoid circular imports

        strategy = self.select_strategy()

        # Use the selected strategy to get the move
        if strategy == 'minimax':
            self.logger.info("🧠 USING MINIMAX STRATEGY for player %s", player)
            move = game_logic.get_best_move(board, player)
            self.logger.info("🎯 MINIMAX SELECTED MOVE: %s", move)
            return move

        self.logger.info("🎲 USING RANDOM STRATEGY for player %s", player)
        move = game_logic.get_random_move(board, player)
        self.logger.info("🎯 RANDOM SELECTED MOVE: %s", move)
        return move


STRATEGY_MAP: Dict[str, Type[Strategy]] = {
    'random': RandomStrategy,
    'minimax': MinimaxStrategy,
}


def create_strategy(strategy_type: str, player: str) -> Strategy:
    """Create a strategy instance based on strategy type and player.

    Args:
        strategy_type: Type of strategy ('random' or 'minimax')
        player: Player symbol ('X' or 'O')

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy_type is unknown
    """
    strategy_class = STRATEGY_MAP.get(strategy_type.lower())
    if strategy_class:
        return strategy_class(player)
    raise ValueError(f"Unknown strategy type: {strategy_type}")
