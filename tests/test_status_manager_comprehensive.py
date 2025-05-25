"""
Comprehensive tests for StatusManager module.
Tests status updates, language management, and UI state.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel
from PyQt5.QtCore import QObject, QTimer
import sys

from app.main.status_manager import (
    StatusManager, LANG_CS, LANG_EN
)


@pytest.fixture
def qt_app():
    """Create QApplication if it doesn't exist."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def mock_main_window():
    """Mock main window for testing."""
    mock_window = Mock(spec=QMainWindow)
    mock_window.style_manager = Mock()
    mock_window.style_manager.update_status_style = Mock()
    return mock_window


@pytest.fixture
def status_manager(qt_app, mock_main_window):
    """Create StatusManager instance for testing."""
    return StatusManager(mock_main_window)


class TestStatusManagerInit:
    """Test StatusManager initialization."""

    def test_initialization(self, status_manager, mock_main_window):
        """Test proper initialization of StatusManager."""
        assert status_manager.main_window == mock_main_window
        assert status_manager.current_language == LANG_CS
        assert status_manager.is_czech is True
        assert status_manager._current_style_key is None
        assert status_manager._status_lock_time == 0
        assert status_manager._current_status_text == ""
        assert status_manager.main_status_panel is None
        assert status_manager.main_status_message is None

    def test_inherits_qobject(self, status_manager):
        """Test that StatusManager inherits from QObject."""
        assert isinstance(status_manager, QObject)

    def test_has_required_signals(self, status_manager):
        """Test that StatusManager has required signals."""
        assert hasattr(status_manager, 'language_changed')
        assert hasattr(status_manager, 'status_updated')


class TestLanguageManagement:
    """Test language management functionality."""

    def test_tr_method_czech_existing_key(self, status_manager):
        """Test translation method with existing Czech key."""
        result = status_manager.tr("your_turn")
        assert result == "VÁŠ TAH"

    def test_tr_method_missing_key(self, status_manager):
        """Test translation method with missing key."""
        result = status_manager.tr("nonexistent_key")
        assert result == "nonexistent_key"

    def test_toggle_language_czech_to_english(self, status_manager):
        """Test language toggle from Czech to English."""
        # Start in Czech
        assert status_manager.is_czech is True
        assert status_manager.current_language == LANG_CS
        
        # Mock the signal emission
        with patch.object(status_manager, 'language_changed') as mock_signal:
            status_manager.toggle_language()
            mock_signal.emit.assert_called_once_with("en")
        
        # Check state after toggle
        assert status_manager.is_czech is False
        assert status_manager.current_language == LANG_EN

    def test_toggle_language_english_to_czech(self, status_manager):
        """Test language toggle from English to Czech."""
        # Set to English first
        status_manager.current_language = LANG_EN
        status_manager.is_czech = False
        
        # Mock the signal emission
        with patch.object(status_manager, 'language_changed') as mock_signal:
            status_manager.toggle_language()
            mock_signal.emit.assert_called_once_with("cs")
        
        # Check state after toggle
        assert status_manager.is_czech is True
        assert status_manager.current_language == LANG_CS

    def test_tr_method_english(self, status_manager):
        """Test translation method with English language."""
        # Switch to English
        status_manager.current_language = LANG_EN
        status_manager.is_czech = False
        
        result = status_manager.tr("your_turn")
        assert result == "YOUR TURN"


class TestLanguageConstants:
    """Test language dictionary constants."""

    def test_lang_cs_contains_required_keys(self):
        """Test that Czech language dict contains required keys."""
        required_keys = [
            "your_turn", "ai_turn", "arm_turn", "win", "draw", "new_game",
            "reset", "debug", "camera", "difficulty", "language"
        ]
        for key in required_keys:
            assert key in LANG_CS, f"Missing key '{key}' in LANG_CS"

    def test_lang_en_contains_required_keys(self):
        """Test that English language dict contains required keys."""
        required_keys = [
            "your_turn", "ai_turn", "arm_turn", "win", "draw", "new_game",
            "reset", "debug", "camera", "difficulty", "language"
        ]
        for key in required_keys:
            assert key in LANG_EN, f"Missing key '{key}' in LANG_EN"

    def test_language_dicts_have_same_keys(self):
        """Test that both language dicts have the same keys."""
        cs_keys = set(LANG_CS.keys())
        en_keys = set(LANG_EN.keys())
        assert cs_keys == en_keys, "Language dictionaries should have identical keys"

    def test_language_values_are_strings(self):
        """Test that all language values are strings."""
        for key, value in LANG_CS.items():
            assert isinstance(value, str), f"LANG_CS['{key}'] should be string, got {type(value)}"
        
        for key, value in LANG_EN.items():
            assert isinstance(value, str), f"LANG_EN['{key}'] should be string, got {type(value)}"


class TestStatusManagerMethods:
    """Test other StatusManager methods."""

    @patch('app.main.status_manager.setup_logger')
    def test_logger_setup(self, mock_setup_logger, qt_app, mock_main_window):
        """Test that logger is properly set up."""
        mock_logger = Mock()
        mock_setup_logger.return_value = mock_logger
        
        manager = StatusManager(mock_main_window)
        
        mock_setup_logger.assert_called_once_with('app.main.status_manager')
        assert manager.logger == mock_logger
