"""
Tests for app/main/status_manager.py module.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from PyQt5.QtCore import QObject
from app.main.status_manager import StatusManager, LANG_CS, LANG_EN


class TestStatusManager:
    """Test class for StatusManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_main_window = Mock()
        
    @patch('app.main.status_manager.setup_logger')
    def test_init(self, mock_setup_logger):
        """Test StatusManager initialization."""
        mock_logger = Mock()
        mock_setup_logger.return_value = mock_logger
        
        status_manager = StatusManager(self.mock_main_window)
        
        assert status_manager.main_window is self.mock_main_window
        assert status_manager.current_language is LANG_CS
        assert status_manager.is_czech is True
        assert status_manager._current_style_key is None
        assert status_manager._status_lock_time == 0
        assert status_manager._current_status_text == ""
        assert status_manager.main_status_panel is None
        assert status_manager.main_status_message is None
        mock_setup_logger.assert_called_once()

    def test_language_constants(self):
        """Test that language dictionaries contain required keys."""
        required_keys = [
            "your_turn", "ai_turn", "arm_turn", "arm_moving", "place_symbol",
            "waiting_detection", "win", "draw", "new_game", "reset", "debug",
            "camera", "difficulty", "arm_connect", "arm_disconnect", "game_over"
        ]
        
        for key in required_keys:
            assert key in LANG_CS, f"Key '{key}' missing from LANG_CS"
            assert key in LANG_EN, f"Key '{key}' missing from LANG_EN"
            assert isinstance(LANG_CS[key], str)
            assert isinstance(LANG_EN[key], str)

    @patch('app.main.status_manager.setup_logger')
    def test_tr_existing_key(self, mock_setup_logger):
        """Test translation of existing key."""
        status_manager = StatusManager(self.mock_main_window)
        
        # Test Czech translation
        result = status_manager.tr("your_turn")
        assert result == "VÁŠ TAH"
        
        # Switch to English and test
        status_manager.current_language = LANG_EN
        result = status_manager.tr("your_turn")
        assert result == "YOUR TURN"

    @patch('app.main.status_manager.setup_logger')
    def test_tr_missing_key(self, mock_setup_logger):
        """Test translation of missing key returns the key itself."""
        status_manager = StatusManager(self.mock_main_window)
        
        result = status_manager.tr("nonexistent_key")
        assert result == "nonexistent_key"

    @patch('app.main.status_manager.setup_logger')
    def test_toggle_language_czech_to_english(self, mock_setup_logger):
        """Test toggling from Czech to English."""
        status_manager = StatusManager(self.mock_main_window)
        status_manager.language_changed = Mock()
        
        # Start with Czech (default)
        assert status_manager.is_czech is True
        assert status_manager.current_language is LANG_CS
        
        # Toggle to English
        status_manager.toggle_language()
        
        assert status_manager.is_czech is False
        assert status_manager.current_language is LANG_EN
        status_manager.language_changed.emit.assert_called_once_with("en")

    @patch('app.main.status_manager.setup_logger')
    def test_toggle_language_english_to_czech(self, mock_setup_logger):
        """Test toggling from English to Czech."""
        status_manager = StatusManager(self.mock_main_window)
        status_manager.language_changed = Mock()
        
        # Set to English first
        status_manager.current_language = LANG_EN
        status_manager.is_czech = False
        
        # Toggle to Czech
        status_manager.toggle_language()
        
        assert status_manager.is_czech is True
        assert status_manager.current_language is LANG_CS
        status_manager.language_changed.emit.assert_called_once_with("cs")

    @patch('app.main.status_manager.setup_logger')
    def test_multiple_language_toggles(self, mock_setup_logger):
        """Test multiple language toggles."""
        status_manager = StatusManager(self.mock_main_window)
        status_manager.language_changed = Mock()
        
        # Start with Czech
        assert status_manager.is_czech is True
        
        # Toggle to English
        status_manager.toggle_language()
        assert status_manager.is_czech is False
        
        # Toggle back to Czech
        status_manager.toggle_language()
        assert status_manager.is_czech is True
        
        # Toggle to English again
        status_manager.toggle_language()
        assert status_manager.is_czech is False

    def test_language_dictionaries_structure(self):
        """Test that language dictionaries have consistent structure."""
        cs_keys = set(LANG_CS.keys())
        en_keys = set(LANG_EN.keys())
        
        # Both dictionaries should have the same keys
        assert cs_keys == en_keys, f"Language dictionaries have different keys: CS={cs_keys - en_keys}, EN={en_keys - cs_keys}"
        
        # All values should be strings
        for key, value in LANG_CS.items():
            assert isinstance(value, str), f"LANG_CS['{key}'] is not a string: {type(value)}"
        
        for key, value in LANG_EN.items():
            assert isinstance(value, str), f"LANG_EN['{key}'] is not a string: {type(value)}"

    def test_language_dictionaries_not_empty(self):
        """Test that language dictionaries are not empty."""
        assert len(LANG_CS) > 0, "LANG_CS is empty"
        assert len(LANG_EN) > 0, "LANG_EN is empty"
        
        # Check that values are not empty strings
        for key, value in LANG_CS.items():
            assert value.strip() != "", f"LANG_CS['{key}'] is empty or whitespace"
        
        for key, value in LANG_EN.items():
            assert value.strip() != "", f"LANG_EN['{key}'] is empty or whitespace"

    @patch('app.main.status_manager.setup_logger')
    def test_inheritance(self, mock_setup_logger):
        """Test that StatusManager inherits from QObject."""
        status_manager = StatusManager(self.mock_main_window)
        assert isinstance(status_manager, QObject)

    @patch('app.main.status_manager.setup_logger')
    def test_initial_status_state(self, mock_setup_logger):
        """Test initial status state values."""
        status_manager = StatusManager(self.mock_main_window)
        
        assert status_manager._current_style_key is None
        assert status_manager._status_lock_time == 0
        assert status_manager._current_status_text == ""
        assert status_manager.main_status_panel is None
        assert status_manager.main_status_message is None

    def test_special_characters_in_translations(self):
        """Test that translations handle special characters correctly."""
        # Test some Czech specific characters
        czech_special_chars = "ěščřžýáíéóúůďťň"
        
        # Check if any Czech translations contain special characters
        czech_translations_with_special = []
        for key, value in LANG_CS.items():
            if any(char in value.lower() for char in czech_special_chars):
                czech_translations_with_special.append((key, value))
        
        # Should have at least some Czech translations with special characters
        assert len(czech_translations_with_special) > 0, "No Czech translations with special characters found"

    @patch('app.main.status_manager.setup_logger')
    def test_tr_with_different_languages_set(self, mock_setup_logger):
        """Test translation behavior when language is explicitly set."""
        status_manager = StatusManager(self.mock_main_window)
        
        # Test with Czech
        status_manager.current_language = LANG_CS
        result_cs = status_manager.tr("win")
        assert result_cs == "VÝHRA"
        
        # Test with English
        status_manager.current_language = LANG_EN
        result_en = status_manager.tr("win")
        assert result_en == "WIN"
        
        # Results should be different
        assert result_cs != result_en