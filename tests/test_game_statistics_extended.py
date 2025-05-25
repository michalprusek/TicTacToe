"""
Extended tests for GameStatistics module.
"""
import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, mock_open
from app.main.game_statistics import GameStatistics


class TestGameStatistics:
    """Test GameStatistics class."""

    def test_init_default(self):
        """Test basic initialization with default file."""
        with patch('os.path.exists', return_value=False):
            stats = GameStatistics()
            assert stats.stats_file == "game_statistics.json"
            assert stats.stats == {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_games": 0
            }

    def test_init_custom_file(self):
        """Test initialization with custom file."""
        with patch('os.path.exists', return_value=False):
            stats = GameStatistics("custom_stats.json")
            assert stats.stats_file == "custom_stats.json"

    def test_load_statistics_file_exists(self):
        """Test loading statistics when file exists."""
        mock_data = {
            "wins": 5,
            "losses": 3,
            "ties": 2,
            "total_games": 10
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
            stats = GameStatistics()
            assert stats.stats == mock_data

    def test_load_statistics_file_not_exists(self):
        """Test loading statistics when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            stats = GameStatistics()
            assert stats.stats == {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_games": 0
            }

    def test_load_statistics_invalid_json(self):
        """Test loading statistics with invalid JSON."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")):
            stats = GameStatistics()
            # Should fall back to default stats
            assert stats.stats == {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_games": 0
            }

    def test_save_statistics(self):
        """Test saving statistics to file."""
        with patch('os.path.exists', return_value=False):
            stats = GameStatistics()
            stats.stats = {"wins": 1, "losses": 0, "ties": 0, "total_games": 1}
            
            with patch('builtins.open', mock_open()) as mock_file:
                stats.save_statistics()
                mock_file.assert_called_once_with("game_statistics.json", 'w', encoding='utf-8')

    def test_save_statistics_exception(self):
        """Test save statistics handles exceptions."""
        with patch('os.path.exists', return_value=False):
            stats = GameStatistics()
            
            with patch('builtins.open', side_effect=IOError("Permission denied")):
                # Should not raise exception
                stats.save_statistics()