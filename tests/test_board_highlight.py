import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

# Přidání cesty k projektu do PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main.pyqt_gui import TicTacToeBoard
from app.main import game_logic


@pytest.fixture
def app():
    """Fixture pro vytvoření QApplication instance"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def board(app):
    """Fixture pro vytvoření instance TicTacToeBoard"""
    board = TicTacToeBoard()
    yield board


def test_board_initialization(board):
    """Test inicializace herní desky"""
    assert board.board == game_logic.create_board()
    assert board.winning_line is None
    assert board.highlighted_cells == []
    assert board.highlight_alpha == 0
    assert board.highlight_fade_in is True


def test_highlight_cells(board):
    """Test zvýraznění buněk"""
    # Příprava testovacích dat
    cells_to_highlight = [(0, 0), (1, 1)]

    # Mockování QTimer.start
    original_start = board.highlight_timer.start
    board.highlight_timer.start = MagicMock()

    # Volání testované metody
    board.highlight_cells(cells_to_highlight)

    # Ověření výsledků
    assert board.highlighted_cells == cells_to_highlight
    assert board.highlight_alpha == 0
    assert board.highlight_fade_in is True
    board.highlight_timer.start.assert_called_once_with(50)

    # Obnovení původní metody
    board.highlight_timer.start = original_start


def test_update_highlight(board):
    """Test aktualizace zvýraznění"""
    # Příprava testovacích dat
    board.highlighted_cells = [(0, 0), (1, 1)]
    board.highlight_alpha = 0
    board.highlight_fade_in = True

    # Mockování QTimer.stop a board.update
    original_stop = board.highlight_timer.stop
    board.highlight_timer.stop = MagicMock()
    original_update = board.update
    board.update = MagicMock()

    # Test fade in
    board.update_highlight()
    assert board.highlight_alpha == 15
    assert board.highlight_fade_in is True
    board.update.assert_called_once()

    # Reset mocku
    board.update.reset_mock()

    # Nastavení maximální průhlednosti
    board.highlight_alpha = 180

    # Test přepnutí na fade out
    board.update_highlight()
    assert board.highlight_alpha == 180  # Zůstává na maximální hodnotě
    assert board.highlight_fade_in is False
    board.update.assert_called_once()

    # Reset mocku
    board.update.reset_mock()

    # Nastavení minimální průhlednosti
    board.highlight_alpha = 10

    # Test ukončení animace
    board.update_highlight()
    assert board.highlight_alpha == 0
    assert len(board.highlighted_cells) == 0
    board.highlight_timer.stop.assert_called_once()
    board.update.assert_called_once()

    # Obnovení původních metod
    board.highlight_timer.stop = original_stop
    board.update = original_update


def test_update_board_with_highlight(board):
    """Test aktualizace stavu desky se zvýrazněním změn"""
    # Příprava testovacích dat
    original_board = [
        [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
        [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY],
        [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
    ]

    new_board = [
        [game_logic.PLAYER_X, game_logic.EMPTY, game_logic.EMPTY],
        [game_logic.EMPTY, game_logic.PLAYER_O, game_logic.EMPTY],
        [game_logic.EMPTY, game_logic.EMPTY, game_logic.EMPTY]
    ]

    # Nastavení počátečního stavu
    board.board = original_board.copy()

    # Mockování metody highlight_cells
    original_highlight_cells = board.highlight_cells
    board.highlight_cells = MagicMock()

    # Volání testované metody
    board.update_board(new_board, None, highlight_changes=True)

    # Ověření výsledků
    assert board.board == new_board
    board.highlight_cells.assert_called_once_with([(0, 0), (1, 1)])

    # Obnovení původní metody
    board.highlight_cells = original_highlight_cells
