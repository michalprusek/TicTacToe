"""
Game manager compatibility module.
DEPRECATED: Functionality moved to game_controller.py
"""

# Re-export for backward compatibility
from app.main.game_controller import GameController as GameManager

__all__ = ['GameManager']