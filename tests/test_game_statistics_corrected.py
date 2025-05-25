"""
Corrected comprehensive tests for GameStatistics module.
Tests statistics tracking, persistence, and actual implementation.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open

from app.main.game_statistics import GameStatistics


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

    @patch('app.main.game_statistics.os.path.exists')
    def test_initialization_no_existing_file(self, mock_exists):
        """Test initialization when stats file doesn't exist."""
        mock_exists.return_value = False
        
        with patch('app.main.game_statistics.setup_logger'):
            stats = GameStatistics("nonexistent.json")
            assert stats.stats == {
                "wins": 0, "losses": 0, "ties": 0, "total_games": 0
            }

    def test_initialization_custom_file(self):
        """Test GameStatistics initialization with custom file."""
        with patch('app.main.game_statistics.setup_logger'):
            with patch('app.main.game_statistics.os.path.exists', return_value=False):
                stats = GameStatistics("custom_stats.json")
                assert stats.stats_file == "custom_stats.json"


class TestGameStatisticsLoading:
    """Test GameStatistics file loading functionality."""

    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_existing_file(self, mock_logger):
        """Test loading statistics from existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"wins": 5, "losses": 3, "ties": 2, "total_games": 10}, f)
            temp_file = f.name
        
        try:
            stats = GameStatistics(temp_file)
            assert stats.stats["wins"] == 5
            assert stats.stats["losses"] == 3
            assert stats.stats["ties"] == 2
            assert stats.stats["total_games"] == 10
        finally:
            os.unlink(temp_file)

    @patch('app.main.game_statistics.setup_logger')
    @patch('builtins.open')
    @patch('app.main.game_statistics.os.path.exists')
    def test_load_statistics_json_error(self, mock_exists, mock_open_func, mock_logger):
        """Test loading statistics when JSON is invalid."""
        mock_exists.return_value = True
        mock_open_func.side_effect = json.JSONDecodeError("msg", "doc", 0)
        
        stats = GameStatistics("invalid.json")
        
        # Should use default stats on JSON error
        assert stats.stats == {"wins": 0, "losses": 0, "ties": 0, "total_games": 0}


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

    def test_get_statistics(self, game_stats):
        """Test get_statistics method."""
        game_stats.stats["wins"] = 15
        game_stats.stats["losses"] = 8
        game_stats.stats["ties"] = 3
        game_stats.stats["total_games"] = 26
        
        result = game_stats.get_statistics()
        
        assert result["wins"] == 15
        assert result["losses"] == 8
        assert result["ties"] == 3
        assert result["total_games"] == 26
        
        # Verify it returns a copy, not the original
        assert result is not game_stats.stats

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

    def test_get_win_rate_all_losses(self, game_stats):
        """Test get_win_rate with all losses."""
        game_stats.stats["wins"] = 0
        game_stats.stats["losses"] = 5
        game_stats.stats["total_games"] = 5
        
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
        
        stats = game_stats.get_statistics()
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["ties"] == 1
        assert stats["total_games"] == 4
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
            
            # Create second instance and verify data persisted
            stats2 = GameStatistics(temp_file)
            stats2_data = stats2.get_statistics()
            assert stats2_data["wins"] == 1
            assert stats2_data["losses"] == 1
            assert stats2_data["total_games"] == 2
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_win_rate_calculation_edge_cases(self, game_stats):
        """Test win rate calculation with various scenarios."""
        # Perfect win rate
        game_stats.stats["wins"] = 10
        game_stats.stats["total_games"] = 10
        assert game_stats.get_win_rate() == 100.0
        
        # Mixed results
        game_stats.stats["wins"] = 3
        game_stats.stats["losses"] = 2
        game_stats.stats["ties"] = 1
        game_stats.stats["total_games"] = 6
        assert abs(game_stats.get_win_rate() - 50.0) < 0.001


class TestGameStatisticsErrorHandling:
    """Test error handling scenarios."""

    @patch('app.main.game_statistics.setup_logger')
    def test_load_corrupted_file(self, mock_logger):
        """Test loading from corrupted file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"wins": "invalid", "losses": 3}')  # Invalid data type
            temp_file = f.name
        
        try:
            stats = GameStatistics(temp_file)
            # Should use defaults for invalid data
            assert stats.stats["wins"] == 0
            assert stats.stats["losses"] == 0
        finally:
            os.unlink(temp_file)

    def test_file_permission_error(self, game_stats):
        """Test handling file permission errors during save."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch.object(game_stats.logger, 'error') as mock_error:
                game_stats.save_statistics()
                mock_error.assert_called_once()


class TestGameStatisticsConstants:
    """Test GameStatistics constants and validation."""

    def test_has_required_attributes(self, game_stats):
        """Test that GameStatistics has all required attributes."""
        required_attrs = ['stats', 'stats_file', 'logger']
        for attr in required_attrs:
            assert hasattr(game_stats, attr), f"Missing attribute: {attr}"

    def test_has_required_methods(self, game_stats):
        """Test that GameStatistics has all required methods."""
        required_methods = [
            'load_statistics', 'save_statistics', 'record_win', 'record_loss',
            'record_tie', 'reset_statistics', 'get_statistics', 'get_win_rate'
        ]
        for method in required_methods:
            assert hasattr(game_stats, method), f"Missing method: {method}"
            assert callable(getattr(game_stats, method)), f"Method {method} not callable"

    def test_stats_dict_structure(self, game_stats):
        """Test that stats dictionary has required keys."""
        required_keys = ['wins', 'losses', 'ties', 'total_games']
        for key in required_keys:
            assert key in game_stats.stats, f"Missing stats key: {key}"
            assert isinstance(game_stats.stats[key], int), f"Stats key {key} should be int"