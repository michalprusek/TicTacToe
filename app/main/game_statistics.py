"""
Game Statistics module for TicTacToe application.
Handles tracking and persistence of game wins, losses, and ties.
"""

import os
import json
import logging
from typing import Dict, Any
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from app.main.game_utils import setup_logger


class GameStatistics:
    """Handles game statistics tracking and persistence."""

    def __init__(self, stats_file: str = "game_statistics.json"):
        self.logger = setup_logger(__name__)
        self.stats_file = stats_file
        self.stats = {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_games": 0
        }
        self.load_statistics()

    def load_statistics(self) -> None:
        """Load statistics from file."""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    loaded_stats = json.load(f)
                    # Validate and merge loaded stats
                    for key in self.stats.keys():
                        if key in loaded_stats and isinstance(loaded_stats[key], int):
                            self.stats[key] = loaded_stats[key]
                    self.logger.info(f"Statistics loaded: {self.stats}")
            else:
                self.logger.info("No statistics file found, starting with empty stats")
        except Exception as e:
            self.logger.error(f"Error loading statistics: {e}")
            # Keep default stats on error

    def save_statistics(self) -> None:
        """Save statistics to file."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
            self.logger.debug(f"Statistics saved: {self.stats}")
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")

    def record_win(self) -> None:
        """Record a player win."""
        self.stats["wins"] += 1
        self.stats["total_games"] += 1
        self.save_statistics()
        self.logger.info(f"Win recorded. Stats: {self.stats}")

    def record_loss(self) -> None:
        """Record a player loss."""
        self.stats["losses"] += 1
        self.stats["total_games"] += 1
        self.save_statistics()
        self.logger.info(f"Loss recorded. Stats: {self.stats}")

    def record_tie(self) -> None:
        """Record a tie game."""
        self.stats["ties"] += 1
        self.stats["total_games"] += 1
        self.save_statistics()
        self.logger.info(f"Tie recorded. Stats: {self.stats}")

    def reset_statistics(self) -> None:
        """Reset all statistics to zero."""
        self.stats = {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_games": 0
        }
        self.save_statistics()
        self.logger.info("Statistics reset")

    def get_statistics(self) -> Dict[str, int]:
        """Get current statistics."""
        return self.stats.copy()

    def get_win_rate(self) -> float:
        """Get win rate as percentage."""
        if self.stats["total_games"] == 0:
            return 0.0
        return (self.stats["wins"] / self.stats["total_games"]) * 100


class GameStatisticsWidget(QWidget):
    """Widget for displaying game statistics."""

    # Signals
    reset_requested = pyqtSignal()

    def __init__(self, status_manager, parent=None):
        super().__init__(parent)
        self.status_manager = status_manager
        self.logger = setup_logger(__name__)

        # Initialize statistics
        self.statistics = GameStatistics()

        # Setup UI
        self.setup_ui()

        # Connect to language changes
        if hasattr(status_manager, 'language_changed'):
            status_manager.language_changed.connect(self.update_labels)

    def setup_ui(self):
        """Setup the statistics widget UI."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Title
        self.title_label = QLabel()
        self.title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 8px;
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        # Statistics container
        stats_container = QWidget()
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setSpacing(8)

        # Individual stat rows
        self.wins_layout = self.create_stat_row()
        self.losses_layout = self.create_stat_row()
        self.ties_layout = self.create_stat_row()
        self.total_layout = self.create_stat_row()
        self.winrate_layout = self.create_stat_row()

        stats_layout.addLayout(self.wins_layout)
        stats_layout.addLayout(self.losses_layout)
        stats_layout.addLayout(self.ties_layout)
        stats_layout.addLayout(self.total_layout)
        stats_layout.addLayout(self.winrate_layout)

        main_layout.addWidget(stats_container)

        # Reset button
        self.reset_button = QPushButton()
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.reset_button.clicked.connect(self.reset_statistics)
        main_layout.addWidget(self.reset_button)

        # Widget styling
        self.setStyleSheet("""
            QWidget {
                background-color: #34495e;
                border-radius: 8px;
                color: #ecf0f1;
            }
        """)

        # Initialize labels
        self.update_labels()
        self.update_statistics_display()

    def create_stat_row(self):
        """Create a row for displaying a statistic."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QLabel()
        label.setStyleSheet("font-size: 15px; color: #bdc3c7;")
        label.setMinimumWidth(120)
        layout.addWidget(label)

        # Value
        value = QLabel("0")
        value.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        value.setAlignment(Qt.AlignRight)
        layout.addWidget(value)

        # Store references
        layout.label = label
        layout.value = value

        return layout

    def update_labels(self):
        """Update all text labels based on current language."""
        # Title
        self.title_label.setText(self.tr("statistics_title"))

        # Statistics labels
        self.wins_layout.label.setText(f"🏆 {self.tr('wins')}:")
        self.losses_layout.label.setText(f"❌ {self.tr('losses')}:")
        self.ties_layout.label.setText(f"🤝 {self.tr('ties')}:")
        self.total_layout.label.setText(f"🎮 {self.tr('total_games')}:")
        self.winrate_layout.label.setText(f"📊 {self.tr('win_rate')}:")

        # Reset button
        self.reset_button.setText(self.tr("reset_stats"))

    def tr(self, key):
        """Translate key using status manager."""
        # Extended translation keys for statistics
        stats_translations = {
            "statistics_title": "STATISTIKY" if self.status_manager.is_czech else "STATISTICS",
            "wins": "Výhry" if self.status_manager.is_czech else "Wins",
            "losses": "Prohry" if self.status_manager.is_czech else "Losses",
            "ties": "Remízy" if self.status_manager.is_czech else "Ties",
            "total_games": "Celkem" if self.status_manager.is_czech else "Total",
            "win_rate": "Úspěšnost" if self.status_manager.is_czech else "Win Rate",
            "reset_stats": "Resetovat" if self.status_manager.is_czech else "Reset"
        }

        return stats_translations.get(key, self.status_manager.tr(key))

    def update_statistics_display(self):
        """Update the displayed statistics values."""
        stats = self.statistics.get_statistics()
        win_rate = self.statistics.get_win_rate()

        self.wins_layout.value.setText(str(stats["wins"]))
        self.losses_layout.value.setText(str(stats["losses"]))
        self.ties_layout.value.setText(str(stats["ties"]))
        self.total_layout.value.setText(str(stats["total_games"]))
        self.winrate_layout.value.setText(f"{win_rate:.1f}%")

        # Color coding for win rate
        if win_rate >= 70:
            color = "#27ae60"  # Green
        elif win_rate >= 50:
            color = "#f39c12"  # Orange
        else:
            color = "#e74c3c"  # Red

        self.winrate_layout.value.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")

    def record_game_result(self, winner, human_player):
        """Record the result of a game."""
        if winner == "HUMAN_WIN":
            self.statistics.record_win()
        elif winner == "TIE" or winner == "Draw":
            self.statistics.record_tie()
        elif winner == "ARM_WIN":
            self.statistics.record_loss()
        else:
            # Fallback for old logic
            if winner == human_player:
                self.statistics.record_win()
            else:
                self.statistics.record_loss()

        self.update_statistics_display()
        self.logger.info(f"Game result recorded: winner={winner}, human_player={human_player}")

    def reset_statistics(self):
        """Reset all statistics."""
        self.statistics.reset_statistics()
        self.update_statistics_display()
        self.reset_requested.emit()
        self.logger.info("Statistics reset by user")

    def get_current_statistics(self):
        """Get current statistics for external use."""
        return self.statistics.get_statistics()