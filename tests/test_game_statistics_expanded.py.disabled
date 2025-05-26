"""
Expanded comprehensive tests for GameStatistics module.
Tests statistics tracking, persistence, and UI widget functionality.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open, MagicMock
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt
import sys

from app.main.game_statistics import GameStatistics, GameStatisticsWidget


@pytest.fixture
def qt_app():
    """Create QApplication if it doesn't exist."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def temp_stats_file():
    """Create temporary stats file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"wins": 5, "losses": 3, "ties": 2, "total_games": 10}')
        temp_file = f.name
    yield temp_file
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture 
def game_stats():
    """Create GameStatistics instance with temp file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    stats = GameStatistics(temp_file)
    yield stats
    if os.path.exists(temp_file):
        os.unlink(temp_file)


class TestGameStatisticsInit:
    """Test GameStatistics initialization."""

    def test_initialization_default_file(self):
        """Test GameStatistics initialization with default file."""
        with patch('app.main.game_statistics.setup_logger') as mock_logger:
            stats = GameStatistics()
            mock_logger.assert_called_once()
            assert stats.stats_file == "game_statistics.json"
            assert stats.stats == {
                "wins": 0, "losses": 0, "ties": 0, "total_games": 0
            }

    def test_initialization_custom_file(self):
        """Test GameStatistics initialization with custom file."""
        with patch('app.main.game_statistics.setup_logger'):
            stats = GameStatistics("custom_stats.json")
            assert stats.stats_file == "custom_stats.json"

    @patch('app.main.game_statistics.setup_logger')
    @patch('app.main.game_statistics.os.path.exists')
    def test_initialization_no_existing_file(self, mock_exists, mock_logger):
        """Test initialization when stats file doesn't exist."""
        mock_exists.return_value = False
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics("nonexistent.json")
        
        mock_logger_instance.info.assert_called_with(
            "No statistics file found, starting with empty stats"
        )


class TestGameStatisticsLoading:
    """Test GameStatistics file loading functionality."""

    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_existing_file(self, mock_logger, temp_stats_file):
        """Test loading statistics from existing file."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics(temp_stats_file)
        
        assert stats.stats["wins"] == 5
        assert stats.stats["losses"] == 3
        assert stats.stats["ties"] == 2
        assert stats.stats["total_games"] == 10

    @patch('app.main.game_statistics.setup_logger')
    @patch('builtins.open')
    @patch('app.main.game_statistics.os.path.exists')
    def test_load_statistics_json_error(self, mock_exists, mock_open_func, mock_logger):
        """Test loading statistics when JSON is invalid."""
        mock_exists.return_value = True
        mock_open_func.return_value.__enter__.return_value.read.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics("invalid.json")
        
        # Should use default stats on JSON error
        assert stats.stats == {"wins": 0, "losses": 0, "ties": 0, "total_games": 0}

    @patch('app.main.game_statistics.setup_logger')
    @patch('builtins.open')
    @patch('app.main.game_statistics.os.path.exists')
    def test_load_statistics_partial_data(self, mock_exists, mock_open_func, mock_logger):
        """Test loading statistics with partial data."""
        mock_exists.return_value = True
        mock_file_content = '{"wins": 8, "losses": 4}'  # Missing ties and total_games
        mock_open_func.return_value = mock_open(read_data=mock_file_content)
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics("partial.json")
        
        assert stats.stats["wins"] == 8
        assert stats.stats["losses"] == 4
        assert stats.stats["ties"] == 0  # Default
        assert stats.stats["total_games"] == 0  # Default


class TestGameStatisticsSaving:
    """Test GameStatistics file saving functionality."""

    def test_save_statistics(self, game_stats):
        """Test saving statistics to file."""
        game_stats.stats["wins"] = 7
        game_stats.stats["losses"] = 2
        game_stats.stats["ties"] = 1
        game_stats.stats["total_games"] = 10
        
        game_stats.save_statistics()
        
        # Verify file was written correctly
        with open(game_stats.stats_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["wins"] == 7
        assert saved_data["losses"] == 2
        assert saved_data["ties"] == 1
        assert saved_data["total_games"] == 10

    @patch('builtins.open')
    def test_save_statistics_error(self, mock_open_func, game_stats):
        """Test save statistics when file write fails."""
        mock_open_func.side_effect = IOError("Permission denied")
        
        with patch.object(game_stats.logger, 'error') as mock_error:
            game_stats.save_statistics()
            mock_error.assert_called_once()


class TestGameStatisticsGameResults:
    """Test recording game results."""

    def test_record_win(self, game_stats):
        """Test recording a win."""
        initial_wins = game_stats.stats["wins"]
        initial_total = game_stats.stats["total_games"]
        
        with patch.object(game_stats, 'save_statistics') as mock_save:
            game_stats.record_win()
            
            assert game_stats.stats["wins"] == initial_wins + 1
            assert game_stats.stats["total_games"] == initial_total + 1
            mock_save.assert_called_once()

    def test_record_loss(self, game_stats):
        """Test recording a loss."""
        initial_losses = game_stats.stats["losses"]
        initial_total = game_stats.stats["total_games"]
        
        with patch.object(game_stats, 'save_statistics') as mock_save:
            game_stats.record_loss()
            
            assert game_stats.stats["losses"] == initial_losses + 1
            assert game_stats.stats["total_games"] == initial_total + 1
            mock_save.assert_called_once()

    def test_record_tie(self, game_stats):
        """Test recording a tie."""
        initial_ties = game_stats.stats["ties"]
        initial_total = game_stats.stats["total_games"]
        
        with patch.object(game_stats, 'save_statistics') as mock_save:
            game_stats.record_tie()
            
            assert game_stats.stats["ties"] == initial_ties + 1
            assert game_stats.stats["total_games"] == initial_total + 1
            mock_save.assert_called_once()


class TestGameStatisticsGetters:
    """Test getter methods."""

    def test_get_wins(self, game_stats):
        """Test get_wins method."""
        game_stats.stats["wins"] = 15
        assert game_stats.get_wins() == 15

    def test_get_losses(self, game_stats):
        """Test get_losses method."""
        game_stats.stats["losses"] = 8
        assert game_stats.get_losses() == 8

    def test_get_ties(self, game_stats):
        """Test get_ties method."""
        game_stats.stats["ties"] = 3
        assert game_stats.get_ties() == 3

    def test_get_total_games(self, game_stats):
        """Test get_total_games method."""
        game_stats.stats["total_games"] = 26
        assert game_stats.get_total_games() == 26

    def test_get_win_rate_with_games(self, game_stats):
        """Test get_win_rate with games played."""
        game_stats.stats["wins"] = 7
        game_stats.stats["total_games"] = 10
        
        win_rate = game_stats.get_win_rate()
        assert abs(win_rate - 70.0) < 0.001

    def test_get_win_rate_no_games(self, game_stats):
        """Test get_win_rate with no games played."""
        game_stats.stats["wins"] = 0
        game_stats.stats["total_games"] = 0
        
        win_rate = game_stats.get_win_rate()
        assert win_rate == 0.0


class TestGameStatisticsReset:
    """Test reset functionality."""

    def test_reset_statistics(self, game_stats):
        """Test resetting statistics."""
        # Set some values first
        game_stats.stats["wins"] = 10
        game_stats.stats["losses"] = 5
        game_stats.stats["ties"] = 2
        game_stats.stats["total_games"] = 17
        
        with patch.object(game_stats, 'save_statistics') as mock_save:
            game_stats.reset_statistics()
            
            assert game_stats.stats["wins"] == 0
            assert game_stats.stats["losses"] == 0
            assert game_stats.stats["ties"] == 0
            assert game_stats.stats["total_games"] == 0
            mock_save.assert_called_once()


class TestGameStatisticsWidget:
    """Test GameStatisticsWidget functionality."""

    def test_widget_initialization(self, qt_app):
        """Test GameStatisticsWidget initialization."""
        mock_stats = Mock(spec=GameStatistics)
        mock_stats.get_wins.return_value = 5
        mock_stats.get_losses.return_value = 3
        mock_stats.get_ties.return_value = 2
        mock_stats.get_total_games.return_value = 10
        mock_stats.get_win_rate.return_value = 50.0
        
        widget = GameStatisticsWidget(mock_stats)
        
        assert widget.game_stats == mock_stats
        assert isinstance(widget, QWidget)

    def test_widget_has_required_components(self, qt_app):
        """Test that widget has required UI components."""
        mock_stats = Mock(spec=GameStatistics)
        mock_stats.get_wins.return_value = 0
        mock_stats.get_losses.return_value = 0
        mock_stats.get_ties.return_value = 0
        mock_stats.get_total_games.return_value = 0
        mock_stats.get_win_rate.return_value = 0.0
        
        widget = GameStatisticsWidget(mock_stats)
        
        # Check that widget has labels for statistics
        labels = widget.findChildren(QLabel)
        assert len(labels) > 0
        
        # Check that widget has reset button
        buttons = widget.findChildren(QPushButton)
        reset_buttons = [btn for btn in buttons if "reset" in btn.text().lower()]
        assert len(reset_buttons) > 0

    def test_update_display(self, qt_app):
        """Test updating the display."""
        mock_stats = Mock(spec=GameStatistics)
        mock_stats.get_wins.return_value = 12
        mock_stats.get_losses.return_value = 7
        mock_stats.get_ties.return_value = 1
        mock_stats.get_total_games.return_value = 20
        mock_stats.get_win_rate.return_value = 60.0
        
        widget = GameStatisticsWidget(mock_stats)
        widget.update_display()
        
        # Verify that the stats methods were called
        mock_stats.get_wins.assert_called()
        mock_stats.get_losses.assert_called()
        mock_stats.get_ties.assert_called()
        mock_stats.get_total_games.assert_called()
        mock_stats.get_win_rate.assert_called()


class TestGameStatisticsIntegration:
    """Test integration scenarios."""

    def test_full_game_cycle(self, game_stats):
        """Test a full cycle of recording different game results."""
        # Record multiple games
        with patch.object(game_stats, 'save_statistics'):
            game_stats.record_win()
            game_stats.record_win()
            game_stats.record_loss()
            game_stats.record_tie()
        
        assert game_stats.get_wins() == 2
        assert game_stats.get_losses() == 1
        assert game_stats.get_ties() == 1
        assert game_stats.get_total_games() == 4
        assert abs(game_stats.get_win_rate() - 50.0) < 0.001

    def test_statistics_persistence(self):
        """Test that statistics persist across instances."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create first instance and record some games
            stats1 = GameStatistics(temp_file)
            stats1.record_win()
            stats1.record_loss()
            stats1.save_statistics()
            
            # Create second instance and verify data persisted
            stats2 = GameStatistics(temp_file)
            assert stats2.get_wins() == 1
            assert stats2.get_losses() == 1
            assert stats2.get_total_games() == 2
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)