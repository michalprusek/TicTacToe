"""
Comprehensive pytest test suite for game_statistics.py module.
Tests GameStatistics functionality including file I/O operations.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open, MagicMock
from app.main.game_statistics import GameStatistics


class TestGameStatistics:
    """Test GameStatistics class functionality."""
    
    def test_game_statistics_initialization_default(self):
        """Test GameStatistics initialization with default file."""
        with patch.object(GameStatistics, 'load_statistics'):
            stats = GameStatistics()
            assert stats.stats_file == "game_statistics.json"
            assert stats.stats == {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_games": 0
            }
    
    def test_game_statistics_initialization_custom_file(self):
        """Test GameStatistics initialization with custom file."""
        with patch.object(GameStatistics, 'load_statistics'):
            stats = GameStatistics(stats_file="custom_stats.json")
            assert stats.stats_file == "custom_stats.json"
    
    @patch('app.main.game_statistics.setup_logger')
    def test_game_statistics_logger_setup(self, mock_setup_logger):
        """Test that logger is properly set up."""
        mock_logger = Mock()
        mock_setup_logger.return_value = mock_logger
        
        with patch.object(GameStatistics, 'load_statistics'):
            stats = GameStatistics()
            
        mock_setup_logger.assert_called_once()
        assert stats.logger == mock_logger


class TestLoadStatistics:
    """Test load_statistics method."""
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_file_exists_valid_data(self, mock_logger, mock_json_load, mock_file, mock_exists):
        """Test loading statistics from existing file with valid data."""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "wins": 5,
            "losses": 3,
            "ties": 2,
            "total_games": 10
        }
        mock_logger.return_value = Mock()
        
        stats = GameStatistics()
        
        mock_exists.assert_called_once_with("game_statistics.json")
        mock_file.assert_called_once_with("game_statistics.json", 'r', encoding='utf-8')
        mock_json_load.assert_called_once()
        
        assert stats.stats == {
            "wins": 5,
            "losses": 3,
            "ties": 2,
            "total_games": 10
        }
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_file_exists_partial_data(self, mock_logger, mock_json_load, mock_file, mock_exists):
        """Test loading statistics with partial data (missing keys)."""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "wins": 3,
            "ties": 1
            # missing "losses" and "total_games"
        }
        mock_logger.return_value = Mock()
        
        stats = GameStatistics()
        
        # Should keep defaults for missing keys
        assert stats.stats == {
            "wins": 3,
            "losses": 0,  # default
            "ties": 1,
            "total_games": 0  # default
        }
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_file_exists_invalid_data_types(self, mock_logger, mock_json_load, mock_file, mock_exists):
        """Test loading statistics with invalid data types."""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "wins": "five",  # string instead of int
            "losses": 3.5,   # float instead of int
            "ties": None,    # None instead of int
            "total_games": 10
        }
        mock_logger.return_value = Mock()
        
        stats = GameStatistics()
        
        # Should only keep valid integer values, use defaults for invalid
        assert stats.stats == {
            "wins": 0,        # default (string not accepted)
            "losses": 0,      # default (float not accepted)
            "ties": 0,        # default (None not accepted)
            "total_games": 10 # valid
        }
    
    @patch('os.path.exists')
    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_file_not_exists(self, mock_logger, mock_exists):
        """Test loading statistics when file doesn't exist."""
        mock_exists.return_value = False
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics()
        
        # Should keep default stats
        assert stats.stats == {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_games": 0
        }
        mock_logger_instance.info.assert_called_with("No statistics file found, starting with empty stats")
    
    @patch('os.path.exists')
    @patch('builtins.open', side_effect=IOError("File read error"))
    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_file_read_error(self, mock_logger, mock_file, mock_exists):
        """Test loading statistics when file read fails."""
        mock_exists.return_value = True
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics()
        
        # Should keep default stats on error
        assert stats.stats == {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_games": 0
        }
        mock_logger_instance.error.assert_called_once()
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0))
    @patch('app.main.game_statistics.setup_logger')
    def test_load_statistics_invalid_json(self, mock_logger, mock_json_load, mock_file, mock_exists):
        """Test loading statistics with invalid JSON."""
        mock_exists.return_value = True
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        stats = GameStatistics()
        
        # Should keep default stats on JSON error
        assert stats.stats == {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_games": 0
        }
        mock_logger_instance.error.assert_called_once()


class TestSaveStatistics:
    """Test save_statistics method."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('app.main.game_statistics.setup_logger')
    def test_save_statistics_success(self, mock_logger, mock_json_dump, mock_file):
        """Test successful saving of statistics."""
        mock_logger.return_value = Mock()
        
        with patch.object(GameStatistics, 'load_statistics'):
            stats = GameStatistics()
            stats.stats = {
                "wins": 5,
                "losses": 3,
                "ties": 2,
                "total_games": 10
            }
        
        stats.save_statistics()
        
        mock_file.assert_called_once_with("game_statistics.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_once_with(stats.stats, mock_file().__enter__(), indent=2)
    
    @patch('builtins.open', side_effect=IOError("File write error"))
    @patch('app.main.game_statistics.setup_logger')
    def test_save_statistics_write_error(self, mock_logger, mock_file):
        """Test saving statistics when file write fails."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        with patch.object(GameStatistics, 'load_statistics'):
            stats = GameStatistics()
        
        stats.save_statistics()
        
        # Should log error
        mock_logger_instance.error.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump', side_effect=TypeError("JSON serialization error"))
    @patch('app.main.game_statistics.setup_logger')
    def test_save_statistics_json_error(self, mock_logger, mock_json_dump, mock_file):
        """Test saving statistics with JSON serialization error."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        with patch.object(GameStatistics, 'load_statistics'):
            stats = GameStatistics()
        
        stats.save_statistics()
        
        # Should log error
        mock_logger_instance.error.assert_called_once()


class TestGameStatisticsIntegration:
    """Integration tests for GameStatistics."""
    
    def test_load_save_roundtrip(self):
        """Test loading and saving statistics in a roundtrip."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            test_file = tf.name
            # Write initial data
            json.dump({
                "wins": 7,
                "losses": 4,
                "ties": 1,
                "total_games": 12
            }, tf, indent=2)
        
        try:
            # Load statistics
            with patch('app.main.game_statistics.setup_logger'):
                stats = GameStatistics(stats_file=test_file)
            
            # Verify loaded correctly
            assert stats.stats == {
                "wins": 7,
                "losses": 4,
                "ties": 1,
                "total_games": 12
            }
            
            # Modify and save
            stats.stats["wins"] = 8
            stats.stats["total_games"] = 13
            stats.save_statistics()
            
            # Load again to verify changes persisted
            with patch('app.main.game_statistics.setup_logger'):
                stats2 = GameStatistics(stats_file=test_file)
            
            assert stats2.stats == {
                "wins": 8,
                "losses": 4,
                "ties": 1,
                "total_games": 13
            }
        
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent file."""
        nonexistent_file = "nonexistent_stats_file.json"
        
        # Ensure file doesn't exist
        if os.path.exists(nonexistent_file):
            os.unlink(nonexistent_file)
        
        try:
            with patch('app.main.game_statistics.setup_logger'):
                stats = GameStatistics(stats_file=nonexistent_file)
            
            # Should have default stats
            assert stats.stats == {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_games": 0
            }
            
            # Save should create the file
            stats.stats["wins"] = 1
            stats.save_statistics()
            
            # File should now exist and be readable
            assert os.path.exists(nonexistent_file)
            
            with open(nonexistent_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["wins"] == 1
        
        finally:
            # Clean up
            if os.path.exists(nonexistent_file):
                os.unlink(nonexistent_file)


if __name__ == "__main__":
    pytest.main([__file__])