"""
Refactored game detector compatibility module.
DEPRECATED: Functionality moved to game_detector.py
"""

# Re-export for backward compatibility
from app.main.game_detector import GameDetector

__all__ = ['GameDetector']