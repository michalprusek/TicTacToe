import sys
import time
import json
import os
import logging
import numpy as np
import cv2
from unittest.mock import MagicMock
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QCheckBox,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QThread
from PyQt5.QtGui import QIcon

from app.main import game_logic
from app.main.arm_controller import ArmController
from app.main.debug_window import DebugWindow
from app.main.camera_view import CameraView
from app.core.config import AppConfig
from app.core.strategy import BernoulliStrategySelector
from app.core.arm_thread import ArmThread
from app.core.game_state import GameState

# Import refactored modules
from app.main.camera_thread import CameraThread
from app.main.board_widget import TicTacToeBoard

# Constants
DEFAULT_SAFE_Z = 15.0
DEFAULT_DRAW_Z = 5.0
DEFAULT_SYMBOL_SIZE_MM = 40.0
DEFAULT_CAMERA_INDEX = 0
DEFAULT_DIFFICULTY = 10  # Maximum difficulty on 0-10 scale
CAMERA_REFRESH_RATE = 30  # ms
PARK_X = -150  # X coordinate for parking position (mm)
PARK_Y = -150  # Y coordinate for parking position (mm)
NEUTRAL_X = 200  # Default X coordinate for neutral position (mm)
NEUTRAL_Y = 0    # Default Y coordinate for neutral position (mm)
NEUTRAL_Z = 15   # Default Z coordinate for neutral position (mm)
# Absolutní cesta ke kalibračnímu souboru
CALIBRATION_FILE = "/Users/michalprusek/PycharmProjects/TicTacToe/app/calibration/hand_eye_calibration.json"
MAX_SPEED = 100000  # Maximální rychlost pohybu ruky (uArm Swift Pro)
DRAWING_SPEED = MAX_SPEED // 2  # Poloviční rychlost pro kreslení

# Language dictionaries for localization
LANG_CS = {
    "your_turn": "VÁŠ TAH",
    "ai_turn": "TAH AI",
    "arm_turn": "TAH RUKY",
    "arm_moving": "RUKA SE POHYBUJE",
    "place_symbol": "POLOŽTE SYMBOL",
    "waiting_detection": "ČEKÁM NA DETEKCI",
    "win": "VÝHRA",
    "draw": "REMÍZA",
    "new_game": "Nová hra",
    "reset": "Reset",
    "debug": "Debug",
    "camera": "Kamera",
    "difficulty": "Obtížnost",
    "arm_connect": "Připojit ruku",
    "arm_disconnect": "Odpojit ruku",
    "game_over": "KONEC HRY",
    "grid_not_visible": "⚠️ MŘÍŽKA NENÍ VIDITELNÁ!",
    "grid_visible": "✅ MŘÍŽKA VIDITELNÁ",
    "move_to_neutral": "PŘESUN DO NEUTRÁLNÍ POZICE",
    "new_game_detected": "NOVÁ HRA DETEKOVÁNA",
    "move_success": "Ruka v neutrální pozici",
    "move_failed": "Nepodařilo se přesunout ruku do neutrální pozice",
    "waiting_for_symbol": "⏳ Čekám na detekci symbolu {}...",
    "detection_failed": "Detekce tahu selhala.",
    "detection_attempt": "Čekám na detekci tahu... (pokus {}/{})",
    "language": "Jazyk",
    "tracking": "SLEDOVÁNÍ HRACÍ PLOCHY"
}

LANG_EN = {
    "your_turn": "YOUR TURN",
    "ai_turn": "AI TURN",
    "arm_turn": "ARM TURN",
    "arm_moving": "ARM MOVING",
    "place_symbol": "PLACE SYMBOL",
    "waiting_detection": "WAITING FOR DETECTION",
    "win": "WIN",
    "draw": "DRAW",
    "new_game": "New Game",
    "reset": "Reset",
    "debug": "Debug",
    "camera": "Camera",
    "difficulty": "Difficulty",
    "arm_connect": "Connect arm",
    "arm_disconnect": "Disconnect arm",
    "game_over": "GAME OVER",
    "grid_not_visible": "⚠️ GRID NOT VISIBLE!",
    "grid_visible": "✅ GRID VISIBLE",
    "move_to_neutral": "MOVING TO NEUTRAL POSITION",
    "new_game_detected": "NEW GAME DETECTED",
    "move_success": "Arm in neutral position",
    "move_failed": "Failed to move arm to neutral position",
    "waiting_for_symbol": "⏳ Waiting for symbol {} detection...",
    "detection_failed": "Symbol detection failed.",
    "detection_attempt": "Waiting for symbol detection... (attempt {}/{})",
    "language": "Language",
    "tracking": "TRACKING GAME BOARD"
}


# CameraThread class has been moved to app/main/camera_thread.py


# TicTacToeBoard class has been moved to app/main/board_widget.py


# CameraView class has been moved to app/main/debug_window.py


class TicTacToeApp(QMainWindow):
    """Main application window"""

    def __init__(self, config=None):
        super().__init__()

        # Inicializace loggeru
        self.logger = logging.getLogger(__name__)

        # Use provided config or create a default one
        self.config = config if config is not None else AppConfig()

        # Inicializace jazyka (výchozí je čeština)
        self.current_language = LANG_CS
        self.is_czech = True

        self.setWindowTitle(self.config.game.gui_window_title)

        # Nastavení ikony aplikace
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 "resources", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Zobrazení na celou obrazovku - v testech chceme normální velikost
        if 'pytest' not in sys.modules:
            self.showFullScreen()

        # Initialize game state
        self.game_state = GameState()

        # Game state
        self.human_player = None
        self.ai_player = None
        self.current_turn = None
        self.game_over = False
        self.winner = None

        # 🤖 UNIFIED ARM MOVE SYSTEM - Single source of truth for arm moves
        self.waiting_for_detection = False
        self.arm_move_in_progress = False  # Flag to prevent multiple simultaneous moves
        self.arm_move_scheduled = False   # Flag to prevent duplicate scheduling
        self.last_arm_move_time = 0       # Timestamp of last arm move
        self.arm_move_cooldown = 3.0      # Cooldown between arm moves (seconds)

        # Legacy flags for compatibility (will be phased out)
        self.waiting_for_valid_moves = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.ai_move_retry_count = 0
        self.max_retry_count = 3
        self.detection_wait_time = 0
        self.max_detection_wait_time = 5.0

        # Atributy pro sledování hrací plochy
        self.tracking_enabled = False
        self.game_paused = False
        self.tracking_timer = QTimer(self)
        self.tracking_timer.timeout.connect(self.track_grid_center)
        self.tracking_interval = 200  # Interval sledování v ms - sníženo pro rychlejší odezvu

        # Debug window - vytvoříme jen při potřebě, ne automaticky
        self.debug_mode = self.config.debug_mode
        self.debug_window = None
        # Nevytváříme debug window automaticky - jen při kliknutí na tlačítko

        # Strategy selector with configured difficulty
        self.strategy_selector = BernoulliStrategySelector(
            difficulty=self.config.game.default_difficulty)

        # Initialize components
        self.init_game_components()
        self.init_ui()

        # Start camera thread with specified camera index (external camera - 0)
        self.camera_thread = CameraThread(camera_index=0)  # Vždy použijeme kameru 0
        self.camera_thread.game_state_updated.connect(
            self.handle_detected_game_state)

        # Připojíme signál kamery přímo k update_camera_view metody hlavního okna
        # Tato metoda se postará o aktualizaci jak hlavního okna, tak debug okna
        self.camera_thread.frame_ready.connect(self.update_camera_view)

        # Debug window signály připojíme jen když se debug window vytvoří
        # Nepřipojujeme automaticky při startu

        self.camera_thread.start()

        # Debug okno se otevře jen při kliknutí na debug tlačítko
        # V debug mode jen zapneme debug okno při prvním kliknutí
        if self.debug_mode:
            # Automaticky otevřeme debug okno jen v debug mode
            QTimer.singleShot(1000, self.show_debug_window)

        # Timer for periodic updates
        self.timer_setup()

    def timer_setup(self):
        """Set up timers for periodic updates"""
        # Update UI texts with current language
        self.update_ui_texts()

        # Timer for game state updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_game_state)
        self.update_timer.start(100)  # 10 FPS for game logic updates

    def tr(self, key):
        """Translate text based on current language"""
        return self.current_language.get(key, key)

    def set_status_style_safe(self, style_key, style_css):
        """Safely set status panel style only if it's different from current style"""
        if not hasattr(self, '_current_style'):
            self._current_style = None

        if self._current_style != style_key:
            self._current_style = style_key
            if hasattr(self, 'main_status_panel') and self.main_status_panel:
                self.main_status_panel.setStyleSheet(style_css)

    def update_status(self, message):
        """Update status message with message"""
        # Check if we need to initialize the status lock
        if not hasattr(self, '_status_lock'):
            self._status_lock = False
            self._current_status = None
            self._last_status_change = 0
            self._status_update_count = 0
            self._current_style = None  # Track current style to prevent redundant setStyleSheet calls

        # Get color codes for different statuses for later consistency
        arm_color = "#9b59b6"
        arm_border = "#8e44ad"
        ai_color = "#3498db"
        ai_border = "#2980b9"
        player_color = "#e74c3c"
        player_border = "#c0392b"

        # Only allow status updates every 3 seconds for locked statuses
        current_time = time.time()
        if self._status_lock and current_time - self._last_status_change < 3.0:
            # Don't update if locked and not enough time has passed, but ensure consistent styling
            if hasattr(self, 'main_status_message') and self.main_status_message:
                if message == self.tr("arm_turn") or message == self.tr("arm_moving"):
                    self.set_status_style_safe("arm", f"""
                        background-color: {arm_color};
                        border-radius: 10px;
                        border: 2px solid {arm_border};
                    """)
                elif message == self.tr("ai_turn"):
                    self.set_status_style_safe("ai", f"""
                        background-color: {ai_color};
                        border-radius: 10px;
                        border: 2px solid {ai_border};
                    """)
                elif message == self.tr("your_turn"):
                    self.set_status_style_safe("player", f"""
                        background-color: {player_color};
                        border-radius: 10px;
                        border: 2px solid {player_border};
                    """)
            return

        # Don't update if it's the same message
        if message == self._current_status:
            return

        # Prevent rapid status updates (max 1 per second)
        current_time = time.time()
        if current_time - self._last_status_change < 1.0:
            self._status_update_count += 1
            if self._status_update_count > 3:  # Allow max 3 rapid updates, then block
                return
        else:
            self._status_update_count = 0  # Reset counter after 1 second

        # Update status with enhanced message
        if hasattr(self, 'main_status_message') and self.main_status_message:
            # Enhanced messages with clear instructions
            enhanced_message = message

            # Add symbol count info to help user understand whose turn it is
            if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
                board = self.camera_thread.last_board_state
                if isinstance(board, list) and len(board) == 9:
                    board_2d = [
                        [board[0], board[1], board[2]],
                        [board[3], board[4], board[5]],
                        [board[6], board[7], board[8]]
                    ]
                    x_count = sum(row.count(game_logic.PLAYER_X) for row in board_2d)
                    o_count = sum(row.count(game_logic.PLAYER_O) for row in board_2d)
                    total = x_count + o_count

                    if message == self.tr("your_turn"):
                        if total % 2 == 0:
                            enhanced_message = f"VÁŠ TAH (počet symbolů: {total} - sudý)"
                        else:
                            enhanced_message = f"ČEKEJTE - RUKA HRAJE (počet: {total} - lichý)"
                    elif message == self.tr("arm_turn"):
                        enhanced_message = f"RUKA HRAJE (počet symbolů: {total} - lichý)"
                    elif message == self.tr("new_game_detected"):
                        enhanced_message = "NOVÁ HRA DETEKOVÁNA - ZAČNĚTE UMÍSTĚNÍM SYMBOLU"

            self.main_status_message.setText(enhanced_message.upper())
            self._current_status = message
            self._last_status_change = current_time

            # Set appropriate styling and lock status based on message type
            if message == self.tr("arm_turn") or message == self.tr("arm_moving"):
                self._status_lock = True
                self.set_status_style_safe("arm", f"""
                    background-color: {arm_color};
                    border-radius: 10px;
                    border: 2px solid {arm_border};
                """)
            elif message == self.tr("ai_turn"):
                self._status_lock = True
                self.set_status_style_safe("ai", f"""
                    background-color: {ai_color};
                    border-radius: 10px;
                    border: 2px solid {ai_border};
                """)
            elif message == self.tr("your_turn"):
                self._status_lock = False
                self.set_status_style_safe("player", f"""
                    background-color: {player_color};
                    border-radius: 10px;
                    border: 2px solid {player_border};
                """)
            else:
                self._status_lock = False

        # Nastavíme status_label na prázdný text, aby se nic nezobrazovalo dole
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.setText("")

    def change_language(self):
        """Toggle between Czech and English"""
        self.is_czech = not self.is_czech
        self.current_language = LANG_CS if self.is_czech else LANG_EN

        # Update UI texts
        self.update_ui_texts()

    def update_ui_texts(self):
        """Update all UI texts based on current language"""
        # Update button texts
        if hasattr(self, 'reset_button'):
            self.reset_button.setText(self.tr("new_game"))

        if hasattr(self, 'debug_button'):
            self.debug_button.setText(self.tr("debug"))

        if hasattr(self, 'language_button'):
            self.language_button.setText("🇨🇿" if self.is_czech else "🇬🇧")

        # Update labels
        if hasattr(self, 'difficulty_label'):
            self.difficulty_label.setText(self.tr("difficulty"))

        # Update main status based on current game state
        self.update_game_status()

    def update_game_status(self):
        """Update main status message based on current game state"""
        # Pokud je hra ve stavu výhra/remíza
        if hasattr(self, 'game_over') and self.game_over:
            winner = None
            if hasattr(self, 'board_widget') and hasattr(self.board_widget, 'winner'):
                winner = self.board_widget.winner

            if winner:
                if winner == self.human_player:
                    self.update_status(self.tr("win"))
                else:
                    self.update_status(self.tr("ai_turn") + " - " + self.tr("win"))
            else:
                self.update_status(self.tr("draw"))
        # Pokud se čeká na tah hráče
        elif hasattr(self, 'current_turn') and self.current_turn == self.human_player:
            self.update_status(self.tr("your_turn"))
        # Pokud hraje AI
        elif hasattr(self, 'current_turn') and self.current_turn == self.ai_player:
            self.update_status(self.tr("ai_turn"))

    def reset_status_panel_style(self):
        """Reset status panel style to default"""
        self.set_status_style_safe("default", """
            background-color: #3498db;
            border-radius: 10px;
            border: 2px solid #2980b9;
        """)

    def update_fps_display(self, fps):
        """Update FPS display"""
        if hasattr(self, 'fps_label') and self.fps_label:
            self.fps_label.setText(f"FPS: {fps}")

    def connect_signals(self):
        """Connect signals to slots"""
        # V testech přeskočíme připojení signálů
        if not hasattr(self, 'camera_thread') or not self.camera_thread:
            return

        # Camera thread signals
        self.camera_thread.frame_ready.connect(self.update_camera_view)
        self.camera_thread.game_state_updated.connect(self.handle_detected_game_state)
        self.camera_thread.fps_updated.connect(self.update_fps_display)

        # Board signals
        if hasattr(self, 'board') and self.board:
            self.board.board_clicked.connect(self.handle_cell_clicked)
        elif hasattr(self, 'board_widget') and self.board_widget:
            self.board_widget.cell_clicked.connect(self.handle_cell_clicked)

        # Button signals
        if hasattr(self, 'reset_button') and self.reset_button:
            self.reset_button.clicked.connect(self.handle_reset_button_click)
        if hasattr(self, 'debug_button') and self.debug_button:
            self.debug_button.clicked.connect(self.handle_debug_button_click)
        if hasattr(self, 'calibrate_button') and self.calibrate_button:
            self.calibrate_button.clicked.connect(self.handle_calibrate_button_click)
        if hasattr(self, 'park_button') and self.park_button:
            self.park_button.clicked.connect(self.handle_park_button_click)
        if hasattr(self, 'track_checkbox') and self.track_checkbox:
            self.track_checkbox.stateChanged.connect(self.handle_track_checkbox_changed)

        # Slider signals
        if hasattr(self, 'difficulty_slider') and self.difficulty_slider:
            self.difficulty_slider.valueChanged.connect(self.handle_difficulty_changed)

    def update_camera_view(self, frame):
        """Update camera view with new frame"""
        # Ujistíme se, že máme platný snímek
        if frame is None:
            return

        # Získáme zpracovaný snímek a stav hry z detection_thread pokud je k dispozici
        processed_frame = None
        game_state = None

        if hasattr(self, 'camera_thread') and self.camera_thread:
            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                # Získání posledního zpracovaného snímku (s detekcemi)
                result = self.camera_thread.detection_thread.get_latest_result()
                if result and result[0] is not None:
                    processed_frame = result[0]

                # Získání stavu hry
                if hasattr(self.camera_thread.detection_thread, 'latest_game_state'):
                    game_state = self.camera_thread.detection_thread.latest_game_state

        # Aktualizace snímku v hlavním okně (použijeme nezpracovaný snímek)
        if hasattr(self, 'camera_view') and self.camera_view:
            if hasattr(self.camera_view, 'update_image'):
                self.camera_view.update_image(frame)
            elif hasattr(self.camera_view, 'update_frame'):
                self.camera_view.update_frame(frame)

        # Kontrola problémů s mřížkou - zobrazit varování, když je mřížka neviditelná nebo mimo záběr
        if game_state:
            # Nejprve zkontrolujeme dynamicky přidané atributy z game_detector.py
            has_grid_issue = hasattr(game_state, 'grid_issue_type') and hasattr(game_state, 'grid_issue_message')

            # Pokud game_detector.py nepřidal atributy, zkontrolujeme taky tradiční způsob
            if hasattr(game_state, '_grid_points') and game_state._grid_points is not None:
                # Počítáme body, které jsou viditelné (nemají nulové souřadnice)
                non_zero_count = np.count_nonzero(np.sum(np.abs(game_state._grid_points), axis=1))
                if non_zero_count < 16:  # 16 je počet bodů v mřížce
                    # Dynamicky přidáme atributy, pokud chybí
                    setattr(game_state, 'grid_issue_type', 'incomplete_visibility')
                    setattr(game_state, 'grid_issue_message',
                            f"CHYBA: Mřížka není kompletně viditelná!\nPouze {non_zero_count}/16 bodů viditelných.\nUmístěte mřížku plně do záběru kamery.")
                    has_grid_issue = True
                elif non_zero_count == 16:
                    # Mřížka je plně viditelná, odstraníme případné příznaky problémů
                    if hasattr(game_state, 'grid_issue_type'):
                        # Log the action for debugging
                        if hasattr(self, 'logger'):
                            self.logger.info("Grid points are all visible in _grid_points. Clearing grid_issue_type attribute.")
                        delattr(game_state, 'grid_issue_type')
                    if hasattr(game_state, 'grid_issue_message'):
                        delattr(game_state, 'grid_issue_message')
                    has_grid_issue = False

            # Zpracování varování (ať už z dynamických atributů nebo z kontroly bodů)
            if has_grid_issue:
                # Vytvoříme nebo zobrazíme výstražný panel, pokud ještě neexistuje
                if not hasattr(self, 'warning_panel') or not self.warning_panel.isVisible():
                    # Vytvoříme panel pro varovné zprávy, pokud neexistuje
                    if not hasattr(self, 'warning_panel'):
                        self.warning_panel = QWidget(self)
                        self.warning_panel.setStyleSheet("""
                            background-color: #A93226;
                            border-radius: 10px;
                            border: 2px solid #E74C3C;
                        """)
                        warning_layout = QVBoxLayout(self.warning_panel)

                        self.warning_icon = QLabel("⚠️")
                        self.warning_icon.setStyleSheet("""
                            color: #FFFFFF;
                            font-size: 32px;
                            margin: 0px;
                        """)
                        self.warning_icon.setAlignment(Qt.AlignCenter)
                        warning_layout.addWidget(self.warning_icon)

                        self.warning_text = QLabel(game_state.grid_issue_message)
                        self.warning_text.setStyleSheet("""
                            color: #FFFFFF;
                            font-size: 18px;
                            font-weight: bold;
                            padding: 10px;
                        """)
                        self.warning_text.setAlignment(Qt.AlignCenter)
                        self.warning_text.setWordWrap(True)
                        warning_layout.addWidget(self.warning_text)

                        # Velikost a pozice varování - nad herní plochou
                        self.warning_panel.setFixedSize(500, 150)
                        # Použijeme relativní pozici v rámci hlavního okna
                        board_x = self.board_widget.x()
                        board_y = self.board_widget.y()
                        board_width = self.board_widget.width()

                        # Umístíme varování nad herní plochu
                        warn_x = board_x + (board_width // 2) - 250
                        warn_y = max(0, board_y - 170)  # Zajistíme, že nezmizí mimo okno
                        self.warning_panel.move(warn_x, warn_y)
                    else:
                        # Aktualizujeme text s aktuálním problémem
                        self.warning_text.setText(game_state.grid_issue_message)

                    # Zobrazíme panel
                    self.warning_panel.show()
                    self.warning_panel.raise_()

                # Zastavíme hru, dokud nebude mřížka viditelná
                if not hasattr(self, 'grid_warning_active') or not self.grid_warning_active:
                    self.grid_warning_active = True
                    self.waiting_for_detection = False
                    self.ai_move_retry_count = 0

                    # Aktualizujeme hlavní stavovou zprávu
                    self.update_status(self.tr("grid_not_visible"))
                    self.set_status_style_safe("error", """
                        background-color: #e74c3c;
                        border-radius: 10px;
                        border: 2px solid #c0392b;
                    """)

                    # Pro zpětnou kompatibilitu
                    if hasattr(self, 'status_label') and self.status_label:
                        self.status_label.setText("")
                        self.status_label.setStyleSheet("color: #E74C3C; font-size: 24px; font-weight: bold; margin: 10px;")
            else:
                # Když není žádný problém s mřížkou, skryjeme varování
                if hasattr(self, 'grid_warning_active') and self.grid_warning_active:
                    # Když je mřížka opět viditelná, skryjeme varování
                    self.grid_warning_active = False

                    # Skryjeme varovný panel pokud existuje
                    if hasattr(self, 'warning_panel'):
                        self.warning_panel.hide()

                    # Aktualizujeme hlavní stavovou zprávu
                    self.update_status(self.tr("grid_visible"))
                    self.set_status_style_safe("success", """
                        background-color: #2ecc71;
                        border-radius: 10px;
                        border: 2px solid #27ae60;
                    """)
                    # Po 3 sekundách vrátíme původní barvu
                    QTimer.singleShot(3000, self.reset_status_panel_style)

                    # Pro zpětnou kompatibilitu
                    if hasattr(self, 'status_label') and self.status_label:
                        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #e0e0e0;")
                        self.status_label.setText("")

            # Pro aktualizaci herní desky z detekovaného stavu
            # Tím zobrazíme aktuální stav hry i když je aktivní varování o chybějících bodech mřížky
            if hasattr(game_state, 'board') and game_state.board is not None:
                self.update_board_from_detection(game_state.board)

            # Aktualizace stavu hry v hlavním GUI podle detekovaného stavu
            if hasattr(game_state, '_board_state') and hasattr(self, 'board_widget'):
                detected_board = game_state._board_state
                # Aktualizujeme přímo herní desku v hlavním GUI podle detekovaného stavu
                # ale provedeme pouze vizuální aktualizaci, ne herní logiku (highlight_changes=False)
                self.board_widget.update_board(detected_board, None, highlight_changes=False)
                # Disable the excessive logging
                # self.logger.info("Aktualizován stav herní desky v hlavním GUI podle detekce z kamery")

        # Aktualizace snímku v debug okně (použijeme zpracovaný snímek s detekcemi)
        if hasattr(self, 'debug_window') and self.debug_window:
            # Přímo aktualizujeme snímek v debug okně - bezpečnější přímá aktualizace
            display_frame = processed_frame if processed_frame is not None else frame
            try:
                self.debug_window.camera_view.update_frame(display_frame.copy())

                # Aktualizace stavu hry v debug okně
                if game_state and hasattr(game_state, '_board_state'):
                    self.debug_window.update_board_state(game_state._board_state)
            except Exception as e:
                print(f"Chyba při aktualizaci debug okna: {e}")

    def handle_cell_clicked(self, row, col):
        """Handle cell click event"""
        # V testech přeskočíme obsluhu kliknutí
        if not hasattr(self, 'board_widget') or not self.board_widget:
            return

        # Pokud je hra ukončena nebo není hráčův tah, ignorujeme kliknutí
        if self.game_over or self.current_turn != self.human_player:
            return

        # Pokud je buňka již obsazena, ignorujeme kliknutí
        if self.board_widget.board[row][col] != game_logic.EMPTY:
            return

        # NEAKTUALIZUJEME BOARD AUTOMATICKY - čekáme na YOLO detekci!
        # self.board_widget.board[row][col] = self.human_player
        self.board_widget.update()

        # Increment move counter to track even/odd turns
        self.move_counter += 1

        # On first move, remember the player's symbol for arm moves
        if self.move_counter == 1:
            self.arm_player_symbol = self.human_player
            self.logger.info(f"Player is using symbol {self.arm_player_symbol}")

        # Kontrola konce hry
        self.check_game_end()

        # Pokud hra neskončila, předáme tah AI
        if not self.game_over:
            self.current_turn = self.ai_player
            self.status_label.setText("")

            # For even-numbered moves, the robotic arm should play instead of the player
            # Check if this was an odd-numbered move (player's turn)
            if self.move_counter % 2 == 1:
                # The next move (even-numbered) should be played by the arm using the player's symbol
                self.logger.info(f"Move #{self.move_counter+1}: Arm will play using player symbol {self.arm_player_symbol}")
                # We still need to update the status to indicate it's player's (arm) turn
                self.update_status(self.tr("arm_turn"))
                self.main_status_panel.setStyleSheet("""
                    background-color: #9b59b6;
                    border-radius: 10px;
                    border: 2px solid #8e44ad;
                """)

                # Schedule the arm's move to happen after a short delay
                # CRITICAL: Use ai_player symbol (the one with fewer pieces)
                arm_symbol = self.ai_player
                self.logger.info(f"Scheduling arm move with symbol {arm_symbol} (ai_player)")
                QTimer.singleShot(1000, lambda: self.make_arm_move(arm_symbol))

    def handle_reset_button_click(self):
        """Handle reset button click event"""
        self.reset_game()

    def reset_game(self):
        """Reset game state"""
        # V testech přeskočíme reset hry
        if not hasattr(self, 'board_widget') or not self.board_widget:
            return

        # CRITICAL: Do not reset board_widget.board here!
        # Board should only be updated from YOLO detections in update_board_from_detection
        # Just clear the visual display
        empty_board = game_logic.create_board()
        self.board_widget.update_board(empty_board, None, highlight_changes=False)
        self.board_widget.winning_line = None
        self.board_widget.update()

        # Reset stavu hry
        self.game_over = False
        self.winner = None
        self.waiting_for_detection = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.ai_move_retry_count = 0

        # Reset counter for tracking even/odd turns
        self.move_counter = 0

        # Store player symbol for arm moves
        self.arm_player_symbol = None

        # Výběr hráče
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = self.human_player

        # Aktualizace stavu
        self.status_label.setText("")

    def handle_debug_button_click(self):
        """Handle debug button click event"""
        self.show_debug_window()

    def show_debug_window(self):
        """Show debug window"""
        # V testech přeskočíme zobrazení debug okna
        if not hasattr(self, 'debug_window') or not self.debug_window:
            # Vytvoříme debug okno, pokud neexistuje
            self.debug_window = DebugWindow(config=self.config, parent=self)

        # Zobrazení debug okna
        self.debug_window.show()

    def handle_calibrate_button_click(self):
        """Handle calibrate button click event"""
        self.calibrate_arm()

    def calibrate_arm(self):
        """Calibrate robotic arm"""
        # V testech přeskočíme kalibraci ruky
        if not hasattr(self, 'arm_controller') or not self.arm_controller:
            return

        # Kontrola připojení ruky
        if not self.arm_controller.connected:
            self.status_label.setText("")
            return

        # Kalibrace ruky
        self.status_label.setText("")
        # TODO: Implementace kalibrace
        self.status_label.setText("")

    def handle_park_button_click(self):
        """Handle park button click event"""
        self.park_arm()

    def park_arm(self):
        """Park robotic arm"""
        # V testech přeskočíme parkování ruky
        if not hasattr(self, 'arm_controller') or not self.arm_controller:
            return

        # Kontrola připojení ruky
        if not self.arm_controller.connected:
            self.status_label.setText("")
            return

        # Parkování ruky
        self.status_label.setText("")
        if hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected:
            # Použití arm_thread pro parkování
            self.arm_thread.go_to_position(x=PARK_X, y=PARK_Y, wait=True)
        elif hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected:
            # Záložní použití arm_controller
            self.arm_controller.park(x=PARK_X, y=PARK_Y)
        self.status_label.setText("")

    def handle_difficulty_changed(self, value):
        """Handle difficulty slider value change"""
        if hasattr(self, 'difficulty_value_label') and self.difficulty_value_label:
            self.difficulty_value_label.setText(f"{value}")

        # Aktualizace obtížnosti AI
        if hasattr(self, 'strategy_selector') and self.strategy_selector:
            # BernoulliStrategySelector has a difficulty property that accepts 0-10
            self.strategy_selector.difficulty = value
            self.logger.info(f"Difficulty set to {value} (p={self.strategy_selector.p:.2f})")

    def handle_track_checkbox_changed(self, state):
        """Handle track checkbox state change"""
        self.tracking_enabled = state == Qt.Checked

        if self.tracking_enabled:
            # Pozastavit hru a zablokovat veškerou interakci s herní deskou
            self.waiting_for_detection = True
            self.game_paused = True

            # Informovat uživatele o sledování hrací plochy
            self.update_status(self.tr("Sledování středu hrací plochy aktivováno"))

            # Zablokovat tlačítka hry
            if hasattr(self, 'start_game_button'):
                self.start_game_button.setEnabled(False)

            # Spustit timer pro kontinuální sledování
            self.tracking_timer.start(self.tracking_interval)
            self.logger.info("Sledování hrací plochy aktivováno")

            # Okamžitě zkusit sledovat střed hrací plochy
            self.track_grid_center()
        else:
            # Zastavit timer pro sledování
            self.tracking_timer.stop()

            # Obnovit hru a umožnit interakci s herní deskou
            self.waiting_for_detection = False
            self.game_paused = False

            # Přesunout ruku do neutrální pozice
            self.move_to_neutral_position()

            # Aktivovat tlačítka hry
            if hasattr(self, 'start_game_button'):
                self.start_game_button.setEnabled(True)

            # Obnovit stav hry
            self.update_status(self.tr("your_turn"))
            self.logger.info("Sledování hrací plochy deaktivováno")

    def track_grid_center(self):
        """Sleduje střed hrací plochy a pohybuje rukou podle něj, i když se hýbe hrací plochou"""
        if not self.tracking_enabled:
            return

        # Kontrola, zda je robotická ruka připojena
        arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and hasattr(self.arm_thread, 'connected') and self.arm_thread.connected
        arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and hasattr(self.arm_controller, 'connected') and self.arm_controller.connected

        if not (arm_thread_available or arm_controller_available):
            self.logger.warning("Robotická ruka není připojena pro sledování")
            return

        # Získání stavu hry z kamery - stejný postup jako v draw_ai_symbol
        game_state = None

        # Zkusíme získat stav hry ze všech možných zdrojů
        if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
            try:
                result = self.camera_thread.detection_thread.get_latest_result()
                if result and len(result) >= 2 and result[1] is not None:
                    game_state = result[1]
                    self.logger.debug("Získán stav hry z detection_thread")
                elif hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                    if hasattr(self.camera_thread.detection_thread.detector, 'game_state'):
                        game_state = self.camera_thread.detection_thread.detector.game_state
                        self.logger.debug("Získán stav hry z detektoru v detection_thread")
            except Exception as e:
                self.logger.debug(f"Chyba při získávání stavu z detection_thread: {e}")

        # Záložní zdroj: game_state z camera_thread.detector (již nepoužíváno)
        # Všechny přístupy by měly být přes detection_thread.detector

        # Záložní zdroj: last_board_state z camera_thread
        if game_state is None and hasattr(self.camera_thread, 'last_board_state') and self.camera_thread.last_board_state is not None:
            try:
                from app.core.game_state import GameState
                game_state = GameState()
                game_state.board = self.camera_thread.last_board_state
                self.logger.debug("Vytvořen nový GameState z last_board_state")
            except Exception as e:
                self.logger.debug(f"Chyba při vytváření GameState z last_board_state: {e}")

        # Bez game_state nemůžeme pokračovat
        if game_state is None:
            self.update_status(self.tr("Čekám na detekci herní plochy..."))
            self.logger.warning("Stav hry není k dispozici pro sledování")
            return

        # Kontrola, zda máme body mřížky
        if not hasattr(game_state, '_grid_points') or game_state._grid_points is None:
            self.update_status(self.tr("Čekám na detekci herní plochy..."))
            self.logger.warning("Mřížka není detekována pro sledování")
            return

        # Kontrola počtu bodů mřížky
        grid_points = game_state._grid_points
        if len(grid_points) < 16:
            self.update_status(self.tr(f"Detekováno jen {len(grid_points)}/16 bodů mřížky"))
            self.logger.warning(f"Nedostatek bodů mřížky pro sledování: {len(grid_points)}/16")
            return

        # Výpočet středu mřížky - průměrujeme všechny x a y souřadnice
        grid_center = np.mean(grid_points, axis=0)
        self.logger.info(f"Střed mřížky vypočten jako průměr {len(grid_points)} bodů: {grid_center}")

        # Použijeme stejný přístup jako v get_cell_coordinates_from_yolo, ale pro střed
        # Získání rozměrů snímku - nejprve z detektoru, pak výchozí hodnoty
        frame_width = 640  # Výchozí hodnota
        frame_height = 480  # Výchozí hodnota

        detector = None
        if hasattr(self.camera_thread, 'detector'):
            detector = self.camera_thread.detector

        if detector and hasattr(detector, 'frame_width') and hasattr(detector, 'frame_height'):
            frame_width = detector.frame_width or frame_width
            frame_height = detector.frame_height or frame_height

        # Převedeme souřadnice středu mřížky na normalizované souřadnice (0-1)
        norm_u = grid_center[0] / frame_width
        norm_v = grid_center[1] / frame_height
        self.logger.info(f"Normalizované souřadnice středu mřížky: u={norm_u:.3f}, v={norm_v:.3f}")

        # Převedeme normalizované souřadnice stejným způsobem, jakým se převádí souřadnice buněk
        # Nejprve zkusíme použít transformační matici, pokud existuje
        target_x = None
        target_y = None

        if hasattr(self, 'calibration_data') and self.calibration_data:
            if "uv_to_xy_matrix" in self.calibration_data:
                try:
                    # Převedeme souřadnice pomocí homografie
                    uv_to_xy_matrix = np.array(self.calibration_data["uv_to_xy_matrix"])

                    # Příprava bodu pro transformaci (potřebujeme homogenní souřadnice)
                    uv_point = np.array([[grid_center[0], grid_center[1], 1.0]], dtype=np.float32).T

                    # Aplikace transformace
                    xy_point = np.matmul(uv_to_xy_matrix, uv_point)

                    # Normalizace homogenních souřadnic
                    if xy_point[2, 0] != 0:
                        target_x = xy_point[0, 0] / xy_point[2, 0]
                        target_y = xy_point[1, 0] / xy_point[2, 0]

                        self.logger.info(f"Transformované souřadnice středu mřížky: UV({grid_center[0]:.1f}, {grid_center[1]:.1f}) -> XY({target_x:.1f}, {target_y:.1f})")
                except Exception as e:
                    self.logger.error(f"Chyba při transformaci souřadnic: {e}")

        # Pokud transformace pomocí matice selhala, použijeme zjednodušenou metodu
        if target_x is None or target_y is None:
            self.logger.info("Používám zjednodušenou transformaci souřadnic (bez kalibrace)")

            # Definice pracovního prostoru
            arm_min_x = 150
            arm_max_x = 300
            arm_min_y = -50
            arm_max_y = 50

            # Načtení pracovního prostoru z kalibračních dat
            if hasattr(self, 'calibration_data') and self.calibration_data:
                if "arm_workspace" in self.calibration_data:
                    workspace = self.calibration_data["arm_workspace"]
                    arm_min_x = workspace.get("min_x", arm_min_x)
                    arm_max_x = workspace.get("max_x", arm_max_x)
                    arm_min_y = workspace.get("min_y", arm_min_y)
                    arm_max_y = workspace.get("max_y", arm_max_y)
                    self.logger.info(f"Použity hodnoty arm_workspace: X({arm_min_x}-{arm_max_x}), Y({arm_min_y}-{arm_max_y})")

            # Převedeme normalizované souřadnice na souřadnice robotické ruky
            # Invertujeme osu Y, protože v obraze je osa Y směrem dolů, ale v
            # robotické ruce je směrem nahoru
            target_x = arm_min_x + norm_u * (arm_max_x - arm_min_x)
            target_y = arm_min_y + (1 - norm_v) * (arm_max_y - arm_min_y)
            self.logger.info(f"Vypočtené souřadnice pro střed mřížky: ({target_x:.1f}, {target_y:.1f})")

        # Použití safe_z z kalibračních dat (nebo výchozí hodnota)
        safe_z = 50
        if hasattr(self, 'calibration_data') and self.calibration_data:
            safe_z = self.calibration_data.get("safe_z", safe_z)
            self.logger.info(f"Použita bezpečná výška z kalibrace: {safe_z}")

        # Aktualizace status labelu
        self.update_status(self.tr(f"Sledování středu mřížky na ({target_x:.1f}, {target_y:.1f}, {safe_z})"))

        # Pohyb ruky na vypočtené souřadnice
        if arm_thread_available:
            self.logger.info(f"Sledování středu mřížky pomocí arm_thread: ({target_x:.1f}, {target_y:.1f}, {safe_z})")
            success = self.arm_thread.go_to_position(x=target_x, y=target_y, z=safe_z, wait=False)
            if not success:
                self.logger.warning("Nepodařilo se odeslat příkaz pro pohyb ruky pomocí arm_thread")
        elif arm_controller_available:
            self.logger.info(f"Sledování středu mřížky pomocí arm_controller: ({target_x:.1f}, {target_y:.1f}, {safe_z})")
            success = self.arm_controller.go_to_position(x=target_x, y=target_y, z=safe_z, wait=False)
            if not success:
                self.logger.warning("Nepodařilo se odeslat příkaz pro pohyb ruky pomocí arm_controller")

    def handle_camera_changed(self, camera_index):
        """Handle camera change event"""
        # V testech přeskočíme obsluhu změny kamery
        if not hasattr(self, 'camera_thread') or not self.camera_thread:
            return

        # Zastavení stávajícího vlákna kamery
        self.camera_thread.stop()
        self.camera_thread.wait()

        # Vytvoření nového vlákna kamery
        self.camera_thread = CameraThread(camera_index=camera_index)
        self.camera_thread.frame_ready.connect(self.update_camera_view)
        self.camera_thread.game_state_updated.connect(self.handle_detected_game_state)
        self.camera_thread.fps_updated.connect(self.update_fps_display)
        self.camera_thread.start()

    def handle_arm_connection_toggled(self, connected):
        """Handle arm connection toggle event"""
        # Unified method - check both arm_thread and arm_controller
        if hasattr(self, 'arm_thread') and self.arm_thread:
            if connected and not self.arm_thread.connected:
                self.arm_thread.connect()
            elif not connected and self.arm_thread.connected:
                self.arm_thread.disconnect()
        # Fallback to arm_controller
        elif hasattr(self, 'arm_controller') and self.arm_controller:
            if connected and not self.arm_controller.connected:
                self.arm_controller.connect()
            elif not connected and self.arm_controller.connected:
                self.arm_controller.disconnect()

    def update_board_from_detection(self, board):
        """Update board visualization without triggering game logic"""
        # V testech přeskočíme aktualizaci stavu herní desky
        if not hasattr(self, 'board_widget') or not self.board_widget:
            return

        self.logger.debug(f"📊 update_board_from_detection called. Current turn: {self.current_turn}, "
                         f"Human: {self.human_player}, AI: {self.ai_player}, "
                         f"waiting_for_detection: {self.waiting_for_detection}, "
                         f"arm_move_in_progress: {self.arm_move_in_progress}")

        # Aktualizace stavu herní desky v GUI bez zvýraznění změn
        # Pouze vizuální aktualizace bez spouštění herní logiky
        if hasattr(self.board_widget, 'update_board'):
            self.board_widget.update_board(board, None, highlight_changes=False)

    def handle_detected_game_state(self, board):
        """Handle detected game state event"""
        # V testech přeskočíme obsluhu detekovaného stavu hry
        if not hasattr(self, 'board_widget') or not self.board_widget:
            return

        # Zjistíme změny mezi starým a novým stavem
        changes = []
        for r in range(3):
            for c in range(3):
                if self.board_widget.board[r][c] != board[r][c] and board[r][c] != game_logic.EMPTY:
                    changes.append((r, c))
                    if hasattr(self, 'logger'):
                        self.logger.info(f"Detekována změna na pozici ({r}, {c}): {self.board_widget.board[r][c]} -> {board[r][c]}")

        # Aktualizace stavu herní desky v GUI
        # Zvýrazníme změny pouze pokud existují nové symboly
        has_new_symbols = len(changes) > 0

        # CRITICAL: Only update board_widget.board from YOLO detections in update_board_from_detection
        # DO NOT update board here - this causes GUI to show moves not yet detected by camera
        # The board parameter here is just for analysis, not for updating GUI

        # Aktualizujeme vizuální reprezentaci herní desky
        self.board_widget.update_board(board, None, highlight_changes=has_new_symbols)

        # Kontrola, zda čekáme na platné tahy
        if hasattr(self, 'waiting_for_valid_moves') and self.waiting_for_valid_moves:
            # Zjistíme, zda jsou k dispozici platné tahy
            valid_moves = []
            for r in range(3):
                for c in range(3):
                    if board[r][c] == game_logic.EMPTY:
                        valid_moves.append((r, c))

            if valid_moves:
                self.logger.info(f"Platné tahy jsou nyní k dispozici: {valid_moves}")
                self.waiting_for_valid_moves = False
                self.current_turn = self.ai_player
                self.make_ai_move()
                return

        # Pokračujeme s herní logikou pouze pokud existují nové symboly
        # To zabrání opakovanému spouštění herní logiky pro každou aktualizaci z kamery
        if has_new_symbols:
            # Detekce prvního tahu hráče pro určení symbolů
            if self.move_counter == 0:
                # Zjistíme, kolik symbolů každého typu je na desce
                x_count = sum(row.count(game_logic.PLAYER_X) for row in board)
                o_count = sum(row.count(game_logic.PLAYER_O) for row in board)

                self.logger.info(f"První detekce: X={x_count}, O={o_count}")

                # Debug - zkontrolovat hodnoty
                self.logger.info(f"DEBUG: x_count={x_count}, o_count={o_count}, x_count < o_count = {x_count < o_count}, o_count < x_count = {o_count < x_count}")

                # FIXED: Ruka hraje symbol, kterého je méně!
                if x_count < o_count:
                    # Méně X než O → ruka hraje X, hráč hraje O
                    self.human_player = game_logic.PLAYER_O
                    self.ai_player = game_logic.PLAYER_X
                    self.arm_player_symbol = game_logic.PLAYER_X
                    self.logger.info(f"První detekovaný tah: méně X ({x_count}) než O ({o_count}) → hráč hraje O, ruka bude kreslit X")
                elif o_count < x_count:
                    # Méně O než X → ruka hraje O, hráč hraje X
                    self.human_player = game_logic.PLAYER_X
                    self.ai_player = game_logic.PLAYER_O
                    self.arm_player_symbol = game_logic.PLAYER_O
                    self.logger.info(f"První detekovaný tah: méně O ({o_count}) než X ({x_count}) → hráč hraje X, ruka bude kreslit O")
                else:
                    # Stejný počet symbolů nebo žádný symbol
                    if x_count == 0 and o_count == 0:
                        # Prázdná deska - čekáme na první tah
                        self.logger.info("Prázdná deska, čekám na první tah hráče")
                        return
                    else:
                        # Stejný počet symbolů - použijeme výchozí (X začíná, takže hráč je asi X)
                        self.logger.warning(f"Stejný počet symbolů X={x_count}, O={o_count}, používám výchozí")
                        self.human_player = game_logic.PLAYER_X
                        self.ai_player = game_logic.PLAYER_O
                        self.arm_player_symbol = game_logic.PLAYER_O

                # Inkrementujeme počítadlo tahů
                self.move_counter = 1

                # CRITICAL: DO NOT update board here!
                # Board updates should ONLY come from update_board_from_detection
                # This prevents GUI from showing moves not yet detected by YOLO

                # Logování aktuálního stavu herní desky
                self.logger.info("=== Detekovaný stav herní desky ===")
                for r in range(3):
                    row_str = ""
                    for c in range(3):
                        cell = board[r][c]
                        if cell == game_logic.EMPTY:
                            row_str += "[ ]"
                        else:
                            row_str += f"[{cell}]"
                    self.logger.info(row_str)
                self.logger.info("======================================")

                # Zjistíme, zda jsou k dispozici platné tahy
                valid_moves = []
                for r in range(3):
                    for c in range(3):
                        if board[r][c] == game_logic.EMPTY:
                            valid_moves.append((r, c))

                if valid_moves:
                        # Určíme, kdo je na tahu podle počtu symbolů
                    total_symbols = x_count + o_count
                    if total_symbols % 2 == 0:
                        # Sudý počet symbolů → hráč je na tahu
                        self.current_turn = self.human_player
                        self.logger.info(f"Sudý počet symbolů ({total_symbols}) → hráč je na tahu")
                        self.update_status(self.tr("your_turn"))
                    else:
                        # Lichý počet symbolů → ruka je na tahu
                        self.current_turn = self.ai_player
                        self.logger.info(f"Lichý počet symbolů ({total_symbols}) → ruka je na tahu")
                        # Update status to show it's arm's turn
                        self.update_status(self.tr("arm_turn"))
                        self.logger.info(f"Starting AI move immediately after first detection, valid moves: {valid_moves}")
                        self.make_ai_move()
                else:
                    self.logger.warning(f"No valid moves available for AI, waiting for next detection")
                    # Nastavíme příznak, že čekáme na další detekci
                    self.waiting_for_valid_moves = True
                return

            # Kontrola konce hry
            self.check_game_end()

            # Pokud hra neskončila a je tah AI, provedeme ho
            if not self.game_over and self.current_turn == self.ai_player:
                self.make_ai_move()
            # Pokud hra neskončila a je tah hráče, přesuneme ruku do neutrální pozice
            elif not self.game_over and self.current_turn == self.human_player:
                # Přesun ruky do neutrální pozice, když čekáme na tah hráče
                self.move_to_neutral_position()

    def make_arm_move_with_symbol(self, symbol):
        """Unified method for making arm moves with a specific symbol"""
        import time

        self.logger.info(f"🤖 ===== MAKE_ARM_MOVE_WITH_SYMBOL STARTED =====")
        self.logger.info(f"🤖 symbol={symbol}")
        self.logger.info(f"🤖 current_turn={self.current_turn}")
        self.logger.info(f"🤖 ai_player={self.ai_player}")
        self.logger.info(f"🤖 human_player={self.human_player}")
        self.logger.info(f"🤖 waiting_for_detection={self.waiting_for_detection}")
        self.logger.info(f"🤖 game_over={self.game_over}")
        self.logger.info(f"🤖 arm_move_in_progress={self.arm_move_in_progress}")
        self.logger.info(f"🤖 arm_move_scheduled={self.arm_move_scheduled}")

        # Check if game is over
        if self.game_over:
            self.logger.warning("🤖 Game is over, cannot make move")
            return False

        # Check if arm is busy or scheduled
        if self.arm_move_in_progress or self.arm_move_scheduled:
            self.logger.warning(f"🤖 ARM BUSY: in_progress={self.arm_move_in_progress}, scheduled={self.arm_move_scheduled}")

            # Check if we're stuck - if waiting_for_detection is True but we're not actually waiting
            if self.waiting_for_detection and hasattr(self, 'detection_wait_time'):
                if self.detection_wait_time > self.max_detection_wait_time:
                    self.logger.error("🚨 ARM STUCK - resetting all flags!")
                    self.reset_arm_flags()
                    return False

            return False

        # Check for duplicate calls and enforce minimum time between moves
        if not hasattr(self, 'last_arm_move_time'):
            self.last_arm_move_time = 0

        current_time = time.time()
        min_time_between_moves = 5.0  # Increased to 5 seconds to ensure proper detection

        if current_time - self.last_arm_move_time < min_time_between_moves:
            remaining_time = min_time_between_moves - (current_time - self.last_arm_move_time)
            self.logger.info(f"🤖 Too soon for next arm move. Please wait {remaining_time:.1f} more seconds.")
            return False

        # Get the latest detected board from camera
        detected_board = None
        if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
            detected_board = self.camera_thread.last_board_state
            # Convert 1D to 2D if needed
            if isinstance(detected_board, list) and len(detected_board) == 9:
                detected_board = [
                    [detected_board[0], detected_board[1], detected_board[2]],
                    [detected_board[3], detected_board[4], detected_board[5]],
                    [detected_board[6], detected_board[7], detected_board[8]]
                ]

        if not detected_board:
            self.logger.error("🤖 No detected board available")
            return False

        # Count symbols to ensure it's arm's turn
        x_count = sum(row.count(game_logic.PLAYER_X) for row in detected_board)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in detected_board)
        total_symbols = x_count + o_count

        self.logger.info(f"🤖 Board state before arm move: X={x_count}, O={o_count}, total={total_symbols}")

        # Check if it's really arm's turn based on symbol count
        # Arm should only play when there's an odd number of symbols (after human plays)
        if total_symbols % 2 == 0:
            self.logger.warning(f"🤖 Not arm's turn! Even number of symbols ({total_symbols}). Waiting for human move.")
            return False

        # Check grid validity
        valid_grid = False
        if hasattr(self, 'camera_thread') and self.camera_thread:
            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                if hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                    game_state = self.camera_thread.detection_thread.detector.game_state
                    if game_state and hasattr(game_state, 'is_physical_grid_valid'):
                        valid_grid = game_state.is_physical_grid_valid()

        if not valid_grid:
            self.logger.warning("🤖 Cannot make arm move - grid is not valid!")
            self.update_status("Umístěte hrací plochu do záběru kamery")
            return False

        # Set flags to prevent duplicate moves ONLY after all checks pass
        self.arm_move_in_progress = True
        self.arm_move_scheduled = True
        self.last_arm_move_time = current_time

        # Kontrola, zda je robotická ruka připojena
        arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected
        arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected

        self.logger.info(f"🤖 arm_thread_available={arm_thread_available}, arm_controller_available={arm_controller_available}")

        if not (arm_thread_available or arm_controller_available):
            self.logger.warning("🚫 Robotická ruka není připojena")
            self.current_turn = self.human_player
            self.update_status(self.tr("your_turn"))
            return False

        # Získání tahu od AI strategie
        self.logger.info(f"🤖 Získávám tah od AI strategie pro symbol {symbol}")
        move = self.strategy_selector.get_move(detected_board, symbol)
        if not move:
            self.logger.warning(f"🚫 Žádný platný tah pro symbol {symbol}")
            self.current_turn = self.human_player
            self.update_status(self.tr("your_turn"))
            return False

        row, col = move
        self.logger.info(f"🎯 AI vybrala tah: ({row}, {col}) pro symbol {symbol}")

        # Nakreslíme symbol
        self.logger.info(f"🤖 Volám draw_ai_symbol({row}, {col}, {symbol})")
        if self.draw_ai_symbol(row, col, symbol):
            # Začneme čekat na detekci symbolu
            self.waiting_for_detection = True
            self.detection_wait_time = 0
            self.ai_move_retry_count = 0
            self.expected_symbol = symbol
            self.ai_move_row = row
            self.ai_move_col = col

            self.logger.info(f"✅ Ruka úspěšně nakreslila {symbol} na ({row}, {col}), čekám na detekci")
            self.logger.info(f"📍 Uložené souřadnice pro detekci: ai_move_row={self.ai_move_row}, ai_move_col={self.ai_move_col}")

            # Clear busy flags after successful draw - we're now waiting for detection
            self.arm_move_in_progress = False
            self.arm_move_scheduled = False
            self.logger.info("🤖 Arm move flags cleared, now waiting for detection")

            # Don't schedule cleanup here - let check_detection_timeout handle it
            # The flags will be cleared when move is detected or timeout occurs
            return True
        else:
            self.logger.error(f"❌ Nepodařilo se nakreslit {symbol} na ({row}, {col})")
            # Reset flags immediately on failure
            self.arm_move_in_progress = False
            self.arm_move_scheduled = False
            self.waiting_for_detection = False
            self.current_turn = self.human_player
            self.update_status(self.tr("your_turn"))
            return False

    def make_ai_move(self):
        """Make AI move"""
        # V testech přeskočíme provedení tahu AI
        if not hasattr(self, 'strategy_selector') or not self.strategy_selector:
            return

        # Kontrola validity mřížky před AI tahem
        valid_grid = False
        if hasattr(self, 'camera_thread') and self.camera_thread:
            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                if hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                    game_state = self.camera_thread.detection_thread.detector.game_state
                    if game_state and hasattr(game_state, 'is_physical_grid_valid'):
                        valid_grid = game_state.is_physical_grid_valid()

        if not valid_grid:
            self.logger.warning("AI tah přeskočen - mřížka není validní!")
            self.update_status("Umístěte hrací plochu do záběru kamery")
            return

        # Kontrola, zda je na řadě AI
        if hasattr(self, 'current_turn') and self.current_turn != self.ai_player:
            self.logger.warning(f"Ignoruji tah AI, protože není na řadě AI (current_turn={self.current_turn})")
            return

        # Ignore duplicate calls to make_ai_move that happen close together
        if not hasattr(self, 'last_ai_move_time'):
            self.last_ai_move_time = 0
        current_time = time.time()
        if current_time - self.last_ai_move_time < 2.0:
            self.logger.info(f"Ignoring duplicate AI move within 2 seconds, last move at {self.last_ai_move_time:.1f}, current time {current_time:.1f}")
            return
        self.last_ai_move_time = current_time

        # Ensure the status is set to AI's turn and lock status changes
        self._status_lock = True
        self.update_status(self.tr("ai_turn"))

        # Make sure AI player symbol is valid
        if not self.ai_player or self.ai_player == game_logic.EMPTY:
            self.ai_player = game_logic.PLAYER_O  # Default to O for AI
            self.logger.warning(f"Invalid AI player symbol, using default: {self.ai_player}")

        self.logger.info(f"🤖 make_ai_move: current_turn={self.current_turn}, ai_player={self.ai_player}, human_player={self.human_player}")

        # Get the latest detected board from camera
        detected_board = None
        if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
            detected_board = self.camera_thread.last_board_state
            # Convert 1D to 2D if needed
            if isinstance(detected_board, list) and len(detected_board) == 9:
                detected_board = [
                    [detected_board[0], detected_board[1], detected_board[2]],
                    [detected_board[3], detected_board[4], detected_board[5]],
                    [detected_board[6], detected_board[7], detected_board[8]]
                ]

        if not detected_board:
            self.logger.error("No detected board available for AI move")
            return

        # Logování stavu herní desky před výběrem tahu
        self.logger.info("=== Detekovaný stav herní desky před výběrem tahu AI ===")
        for r in range(3):
            row_str = ""
            for c in range(3):
                cell = detected_board[r][c]
                if cell == game_logic.EMPTY:
                    row_str += "[ ]"
                else:
                    row_str += f"[{cell}]"
            self.logger.info(row_str)
        self.logger.info("==========================================")

        # Logování dostupných tahů před získáním tahu AI
        valid_moves = []
        for r in range(3):
            for c in range(3):
                if detected_board[r][c] == game_logic.EMPTY:
                    valid_moves.append((r, c))

        self.logger.info(f"Dostupné tahy před výběrem: {valid_moves}")

        # Kontrola, zda jsou k dispozici platné tahy
        if not valid_moves:
            self.logger.error("Žádné platné tahy nejsou k dispozici!")
            return

        # Získání tahu AI
        self.logger.info(f"Získávám tah AI pro hráče {self.ai_player}...")
        move = self.strategy_selector.get_move(detected_board, self.ai_player)
        if not move:
            self.logger.warning("No valid move found for AI")
            self.logger.warning(f"Dostupné tahy: {valid_moves}")

            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("")

            # Pokud nejsou žádné platné tahy, ale herní deska není plná,
            # zkusíme vybrat náhodný prázdný tah
            if valid_moves:
                self.logger.info("Vybírám náhodný tah z dostupných prázdných polí")
                import random
                move = random.choice(valid_moves)
                self.logger.info(f"Vybrán náhodný tah: {move}")
            else:
                return

        # Provedení tahu AI
        row, col = move
        self.logger.info(f"AI hraje tah na pozici ({row}, {col}) se symbolem {self.ai_player}")
        # NEAKTUALIZUJEME BOARD AUTOMATICKY - čekáme na YOLO detekci!
        # self.board_widget.board[row][col] = self.ai_player
        # self.board_widget.update()

        # Increment move counter
        self.move_counter += 1

        # Kontrola konce hry
        self.check_game_end()

        # Pokud hra neskončila, předáme tah hráči
        if not self.game_over:
            self.current_turn = self.human_player

            # Aktualizujeme hlavní stavovou zprávu
            self.update_status(self.tr("your_turn"))
            self.main_status_panel.setStyleSheet("""
                background-color: #9b59b6;
                border-radius: 10px;
                border: 2px solid #8e44ad;
            """)

            # Pro zpětnou kompatibilitu
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("")

    def make_arm_move(self, symbol):
        """Make a move using the robotic arm with the player's symbol"""
        # DEPRECATED: Use make_arm_move_with_symbol instead
        # This method is kept for backward compatibility
        return self.make_arm_move_with_symbol(symbol)

    def check_detection_timeout(self, row, col, symbol=None):
        """Check if the arm move was detected and handle timeout if needed"""
        # If we're still waiting for detection after the timeout
        if self.waiting_for_detection:
            self.logger.warning(f"Detection timeout for arm move at ({row}, {col})")

            # ✅ CRITICAL FIX: Complete flag reset and turn management
            self.waiting_for_detection = False
            self.arm_move_in_progress = False
            self.arm_move_scheduled = False
            self._move_in_progress = False  # Legacy flag
            self.ai_move_row = None
            self.ai_move_col = None
            self.detection_wait_time = 0
            self.last_arm_move_time = 0  # Reset cooldown timer

            # Reset legacy scheduled flag if it exists
            if hasattr(self, '_arm_move_scheduled'):
                delattr(self, '_arm_move_scheduled')

            self.logger.info(f"🤖 All flags reset after detection timeout")

            # ✅ CRITICAL FIX: Properly determine next turn based on current board state
            # Don't just pass to human - determine correct turn based on symbol count
            if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
                detected_board = self.camera_thread.last_board_state
                if detected_board:
                    x_count = sum(row.count('X') for row in detected_board)
                    o_count = sum(row.count('O') for row in detected_board)
                    total_symbols = x_count + o_count

                    # Determine whose turn it should be based on symbol count
                    if total_symbols % 2 == 0:
                        # Even number of symbols - X's turn (first player)
                        self.current_turn = 'X'
                    else:
                        # Odd number of symbols - O's turn (second player)
                        self.current_turn = 'O'

                    self.logger.info(f"🔄 Turn determined from board state: X={x_count}, O={o_count}, total={total_symbols}, next_turn={self.current_turn}")
                else:
                    # Fallback to human player
                    self.current_turn = self.human_player
            else:
                # Fallback to human player
                self.current_turn = self.human_player

            self.update_status(self.tr("your_turn") if self.current_turn == self.human_player else self.tr("ai_turn"))

            self.logger.info(f"🔄 Detection timeout handled - all flags reset, next turn: {self.current_turn}")

    def reset_arm_flags(self):
        """🤖 EMERGENCY: Reset all arm-related flags to recover from stuck states"""
        self.logger.warning("🚨 EMERGENCY ARM FLAG RESET - clearing all stuck states")

        # Reset all arm-related flags
        self.waiting_for_detection = False
        self.arm_move_in_progress = False
        self.arm_move_scheduled = False
        self._move_in_progress = False
        self.ai_move_row = None
        self.ai_move_col = None
        self.detection_wait_time = 0
        self.last_arm_move_time = 0

        # Reset legacy flags
        if hasattr(self, '_arm_move_scheduled'):
            delattr(self, '_arm_move_scheduled')

        # Determine correct turn from current board state
        if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
            detected_board = self.camera_thread.last_board_state
            if detected_board:
                x_count = sum(row.count('X') for row in detected_board)
                o_count = sum(row.count('O') for row in detected_board)
                total_symbols = x_count + o_count

                # Determine whose turn it should be
                if total_symbols % 2 == 0:
                    self.current_turn = 'X'
                else:
                    self.current_turn = 'O'

                self.logger.info(f"🚨 Turn reset from board state: X={x_count}, O={o_count}, next_turn={self.current_turn}")
            else:
                self.current_turn = self.human_player
        else:
            self.current_turn = self.human_player

        self.update_status(self.tr("your_turn") if self.current_turn == self.human_player else self.tr("ai_turn"))
        self.logger.warning("🚨 EMERGENCY RESET COMPLETE - arm should be able to move again")

    def init_ui(self):
        """Initialize the user interface"""
        # V testech přeskočíme inicializaci UI
        try:
            # Nastavíme mock atributy pro testy
            if not hasattr(self, 'board_widget'):
                self.board_widget = MagicMock()
            if not hasattr(self, 'status_label'):
                self.status_label = MagicMock()
            if not hasattr(self, 'difficulty_slider'):
                self.difficulty_slider = MagicMock()
            if not hasattr(self, 'difficulty_value_label'):
                self.difficulty_value_label = MagicMock()
            if not hasattr(self, 'reset_button'):
                self.reset_button = MagicMock()
            if not hasattr(self, 'debug_button'):
                self.debug_button = MagicMock()
            if not hasattr(self, 'language_button'):
                self.language_button = MagicMock()
            if not hasattr(self, 'difficulty_label'):
                self.difficulty_label = MagicMock()

            # Vytvoření centrálního widgetu
            if hasattr(self, 'setCentralWidget') and callable(self.setCentralWidget):
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
            else:
                return
        except Exception as e:
            print(f"Chyba při inicializaci UI: {e}")
            return

        # Nastavení moderního tmavého vzhledu
        self.setStyleSheet("""
            QWidget {
                background-color: #2D2D30;
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084E3;
            }
            QPushButton:pressed {
                background-color: #0067B8;
            }
            QLabel {
                color: #E0E0E0;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #3D3D3D;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #0078D7;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
        """)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)  # Přidání okrajů
        main_layout.setSpacing(15)  # Větší mezery mezi prvky

        # Panel se stavovou zprávou - velký a výrazný
        self.main_status_panel = QWidget()
        self.main_status_panel.setStyleSheet("""
            background-color: #3498db;
            border-radius: 10px;
            border: 2px solid #2980b9;
        """)
        status_layout = QVBoxLayout(self.main_status_panel)

        # Velká stavová zpráva pro uživatele
        self.main_status_message = QLabel("ZAČNĚTE HRU")
        self.main_status_message.setStyleSheet("""
            color: #FFFFFF;
            font-size: 32px;
            font-weight: bold;
            padding: 15px;
        """)
        self.main_status_message.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.main_status_message)

        main_layout.addWidget(self.main_status_panel)

        # Game board - vycentrovaná a zvětšená
        board_container = QWidget()
        board_container.setStyleSheet("background-color: #333740; border-radius: 10px; padding: 10px;")
        board_layout = QHBoxLayout(board_container)

        # Přidání pružných spacerů pro vycentrování
        board_layout.addStretch(1)

        self.board_widget = TicTacToeBoard()
        self.board_widget.cell_clicked.connect(self.handle_cell_clicked)
        self.board_widget.setMinimumSize(450, 450)  # Zvětšení herní desky
        board_layout.addWidget(self.board_widget)

        # Přidání pružných spacerů pro vycentrování
        board_layout.addStretch(1)

        main_layout.addWidget(board_container, 1)

        # Kamerový náhled je pouze v debug okně, tady jen vytvoříme instanci
        self.camera_view = CameraView()  # Skryté, bude viditelné pouze v debug okně

        # Controls panel - moderní design s panelem
        controls_panel = QWidget()
        controls_panel.setStyleSheet("background-color: #333740; border-radius: 10px; padding: 15px;")
        controls_layout = QVBoxLayout(controls_panel)

        # Title for controls section
        controls_title = QLabel("Ovládání hry")
        controls_title.setAlignment(Qt.AlignCenter)
        controls_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        controls_layout.addWidget(controls_title)

        # Difficulty controls in a horizontal layout
        difficulty_container = QWidget()
        difficulty_layout = QHBoxLayout(difficulty_container)
        difficulty_layout.setContentsMargins(0, 0, 0, 0)

        self.difficulty_label = QLabel(self.tr("difficulty"))
        self.difficulty_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.difficulty_slider = QSlider(Qt.Horizontal)
        self.difficulty_slider.setMinimum(0)
        self.difficulty_slider.setMaximum(10)
        self.difficulty_slider.setValue(DEFAULT_DIFFICULTY)
        self.difficulty_slider.setTickPosition(QSlider.TicksBelow)
        self.difficulty_slider.setTickInterval(1)
        self.difficulty_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #3d4351;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 2px solid #2980b9;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::groove:horizontal {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #2ecc71, stop: 1 #e74c3c);
                height: 8px;
                border-radius: 4px;
            }
            /* Zajistí, že sub-page (aktivní část) je průhledná a vidí se základní gradient */
            QSlider::sub-page:horizontal {
                background: transparent;
                border-radius: 4px;
            }
        """)
        self.difficulty_slider.valueChanged.connect(self.handle_difficulty_changed)

        self.difficulty_value_label = QLabel(f"{DEFAULT_DIFFICULTY}")
        self.difficulty_value_label.setStyleSheet("font-weight: bold; min-width: 20px;")

        difficulty_layout.addWidget(self.difficulty_label)
        difficulty_layout.addWidget(self.difficulty_slider, 1)  # Přidání stretche pro slider
        difficulty_layout.addWidget(self.difficulty_value_label)

        controls_layout.addWidget(difficulty_container)

        # Button row in a horizontal layout
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)

        # Hlavní panel tlačítek
        main_button_container = QWidget()
        main_button_layout = QHBoxLayout(main_button_container)
        main_button_layout.setContentsMargins(0, 0, 0, 0)

        # Reset button with icon
        self.reset_button = QPushButton("🔄 " + self.tr("new_game"))
        self.reset_button.clicked.connect(self.reset_game)
        self.reset_button.setMinimumHeight(40)  # Vyšší tlačítka
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                border-radius: 5px;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)

        # Přidáme tlačítko do hlavního layout
        main_button_layout.addWidget(self.reset_button)

        # Pravá část s Debug a Language tlačítky
        right_button_container = QWidget()
        right_button_layout = QHBoxLayout(right_button_container)
        right_button_layout.setContentsMargins(0, 0, 0, 0)
        right_button_layout.setSpacing(5)

        # Language button
        self.language_button = QPushButton("🇨🇿")
        self.language_button.setToolTip(self.tr("language"))
        self.language_button.clicked.connect(self.change_language)
        self.language_button.setFixedSize(40, 40)
        self.language_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        # Debug button with wrench icon
        self.debug_button = QPushButton("⚙️")  # Unicode znak pro ozubené kolo - nejbližší náhrada za francouzský klíč
        self.debug_button.setToolTip(self.tr("debug"))
        self.debug_button.clicked.connect(self.show_debug_window)
        self.debug_button.setFixedSize(40, 40)
        self.debug_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)

        # Track checkbox
        self.track_checkbox = QCheckBox("Track")
        self.track_checkbox.setToolTip(self.tr("Sledování hrací plochy"))
        self.track_checkbox.stateChanged.connect(self.handle_track_checkbox_changed)
        self.track_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid #3498db;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
            }
        """)

        right_button_layout.addWidget(self.language_button)
        right_button_layout.addWidget(self.debug_button)
        right_button_layout.addWidget(self.track_checkbox)

        # Přidáme oba kontejnery do hlavního layout tlačítek
        button_layout.addWidget(main_button_container, 3)  # 75% šířky pro reset tlačítko
        button_layout.addStretch(1)  # Vloží mezeru mezi tlačítka
        button_layout.addWidget(right_button_container)  # 25% pro pravá tlačítka

        controls_layout.addWidget(button_container)

        main_layout.addWidget(controls_panel)

        # Vytvoříme status_label, ale bez zobrazení v GUI (zachování kompatibility se starým kódem)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: transparent;")
        self.status_label.setVisible(False)  # Skryjeme label kompletně

    def init_game_components(self):
        """Initialize game components (arm controller, etc.)"""
        try:
            # Načtení kalibračního souboru
            self.calibration_data = self.load_calibration()

            # Získání hodnot z kalibrace nebo použití hodnot z konfigurace
            draw_z = self.calibration_data.get("draw_z", self.config.arm_controller.draw_z)
            safe_z = self.calibration_data.get("safe_z", self.config.arm_controller.safe_z)

            # Načtení nebo nastavení neutrálního stavu
            self.neutral_position = self.load_neutral_position()

            # Vytvoření vlákna pro ovládání robotické ruky
            self.arm_thread = ArmThread(port=self.config.arm_controller.port)
            self.arm_thread.start()

            # Vytvoření arm_controller pro zpětnou kompatibilitu
            self.arm_controller = ArmController(
                port=self.config.arm_controller.port,
                draw_z=draw_z,
                safe_z=safe_z,
                speed=MAX_SPEED  # Nastavení maximální rychlosti
            )

            # Připojení k robotické ruce
            if self.arm_thread.connect():
                # Přesun do neutrálního stavu po připojení
                self.move_to_neutral_position()
                self.arm_controller.connected = True
            else:
                print("Nepodařilo se připojit k robotické ruce")
                self.arm_controller.connected = False

        except Exception as e:
            print(f"Chyba při inicializaci: {str(e)}")

    def load_calibration(self):
        """Načte kalibrační data ze souboru a vypočítá transformační matici"""
        try:
            if not os.path.exists(CALIBRATION_FILE):
                print(
                    f"Kalibrační soubor {CALIBRATION_FILE} nenalezen, používám výchozí hodnoty")
                return {}

            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
            print(f"Kalibrace načtena z {CALIBRATION_FILE}")

            # 1. Výpočet transformační matice UV -> XY
            if "calibration_points_raw" not in data:
                print("'calibration_points_raw' nenalezeno v kalibračním souboru")
                return data

            raw_points = data["calibration_points_raw"]
            if not isinstance(raw_points, list) or len(raw_points) < 4:
                print(
                    f"'calibration_points_raw' musí být seznam s alespoň 4 body. Nalezeno {len(raw_points)}.")
                return data

            print(
                f"Nalezeno {len(raw_points)} kalibračních bodů. Počítám transformaci UV->XY.")
            points_uv = []
            points_xy = []
            valid_points_count = 0

            try:
                for p_idx, p in enumerate(raw_points):
                    if ('target_uv' in p and len(p['target_uv']) == 2 and
                            'robot_xyz' in p and len(p['robot_xyz']) >= 2):
                        points_uv.append(p['target_uv'])
                        # Potřebujeme jen XY
                        points_xy.append(p['robot_xyz'][:2])
                        valid_points_count += 1
                    else:
                        print(
                            f"Přeskakuji neplatný/neúplný bod na indexu {p_idx}")

                if valid_points_count < 4:
                    print(
                        f"Nedostatek platných bodů ({valid_points_count} < 4) pro výpočet transformace UV->XY.")
                    return data

                np_points_uv = np.array(points_uv, dtype=np.float32)
                np_points_xy = np.array(points_xy, dtype=np.float32)

                uv_to_xy_matrix, mask = cv2.findHomography(
                    np_points_uv, np_points_xy, method=cv2.RANSAC,
                    ransacReprojThreshold=10.0
                )

                if uv_to_xy_matrix is None:
                    print("Nepodařilo se vypočítat transformační matici UV->XY")
                    return data

                data["uv_to_xy_matrix"] = uv_to_xy_matrix.tolist()
                num_inliers = np.sum(mask) if mask is not None else 0
                print(
                    f"Transformační matice UV->XY úspěšně vypočítána s {num_inliers}/{valid_points_count} inliery.")

                if num_inliers < 4:
                    print("Varování: Nízký počet inlierů (<4) pro transformaci UV->XY.")

            except Exception as e:
                print(f"Chyba při zpracování kalibračních bodů: {e}")
                import traceback
                traceback.print_exc()
                return data

            # 2. Zpracování draw_z vs touch_z
            if "draw_z" not in data and "touch_z" in data:
                data["draw_z"] = data["touch_z"]
                print("Klíč 'draw_z' nenalezen, používám hodnotu 'touch_z'.")

            # 3. Zpracování symbol_size_mm
            if "symbol_size_mm" not in data:
                data["symbol_size_mm"] = DEFAULT_SYMBOL_SIZE_MM
                print(
                    f"Klíč 'symbol_size_mm' nenalezen, používám výchozí hodnotu: {DEFAULT_SYMBOL_SIZE_MM}mm")

            # 4. Definice pracovního prostoru ruky
            if "arm_workspace" not in data:
                data["arm_workspace"] = {
                    "min_x": 150,
                    "max_x": 250,
                    "min_y": -50,
                    "max_y": 50
                }
                print("Klíč 'arm_workspace' nenalezen, používám výchozí hodnoty.")

            print("Kalibrace úspěšně načtena a zpracována.")
            return data

        except FileNotFoundError:
            print(f"Kalibrační soubor '{CALIBRATION_FILE}' nenalezen.")
            return {}
        except json.JSONDecodeError:
            print(f"Chyba při zpracování '{CALIBRATION_FILE}'. Neplatný JSON?")
            return {}
        except Exception as e:
            print(f"Neočekávaná chyba při načítání kalibrace: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_calibration_data(self):
        """Načte kalibrační data ze souboru"""
        try:
            if not os.path.exists(CALIBRATION_FILE):
                print(f"Kalibrační soubor {CALIBRATION_FILE} nenalezen")
                return False

            with open(CALIBRATION_FILE, 'r') as f:
                self.calibration_data = json.load(f)

            # Převod matice na numpy array, pokud existuje
            if "calibration_matrix" in self.calibration_data:
                self.calibration_matrix = np.array(self.calibration_data["calibration_matrix"])
            else:
                self.calibration_matrix = None

            # Načtení neutrální pozice
            if "neutral_position" in self.calibration_data:
                self.neutral_position = self.calibration_data["neutral_position"]

            # Načtení pracovního prostoru ruky
            if "arm_workspace" in self.calibration_data:
                self.arm_workspace = self.calibration_data["arm_workspace"]

            return True
        except Exception as e:
            print(f"Chyba při načítání kalibračních dat: {e}")
            return False

    def save_calibration_data(self):
        """Uloží kalibrační data do souboru"""
        try:
            if not hasattr(self, 'calibration_data') or not self.calibration_data:
                return False

            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)

            return True
        except Exception as e:
            print(f"Chyba při ukládání kalibračních dat: {e}")
            return False

    def get_cell_coordinates(self, row, col):
        """Získá souřadnice buňky pro robotickou ruku"""
        # Kontrola, zda máme kameru a detektor
        if not hasattr(self, 'camera_thread') or not self.camera_thread:
            return None

        if not hasattr(self, 'camera_thread') or not hasattr(self.camera_thread, 'detection_thread') or not self.camera_thread.detection_thread:
            return None

        # Pokud máme game_state jako atribut, použijeme ho (pro testy)
        if hasattr(self, 'game_state'):
            game_state = self.game_state
        else:
            # Získání stavu hry z detection_thread
            game_state = self.camera_thread.detection_thread.game_state

        if not game_state or not game_state.is_valid():
            return None

        # Získání souřadnic středu buňky v pixelech
        try:
            cell_center_uv = game_state.get_cell_center_uv(row, col)
        except Exception:
            return None

        if not cell_center_uv:
            return None

        # Pokud máme kalibrační matici, použijeme ji pro transformaci
        if hasattr(self, 'calibration_matrix') and self.calibration_matrix is not None:
            try:
                # Příprava bodu pro transformaci (homogenní souřadnice)
                uv_point = np.array([cell_center_uv[0], cell_center_uv[1], 1.0])

                # Aplikace transformace
                xy_point = np.dot(self.calibration_matrix, uv_point)

                # Vrácení transformovaných souřadnic
                return (xy_point[0], xy_point[1])
            except Exception as e:
                print(f"Chyba při transformaci souřadnic: {e}")
                return None

        # Pokud nemáme kalibrační matici, ale máme definovaný pracovní prostor,
        # použijeme zjednodušenou transformaci
        elif hasattr(self, 'arm_workspace') and self.arm_workspace:
            # Výpočet středu pracovního prostoru
            center_x = (self.arm_workspace["min_x"] + self.arm_workspace["max_x"]) / 2
            center_y = (self.arm_workspace["min_y"] + self.arm_workspace["max_y"]) / 2

            # Vrácení středu pracovního prostoru
            return (center_x, center_y)

        # Pokud nemáme ani kalibrační matici, ani pracovní prostor,
        # vrátíme výchozí souřadnice
        else:
            return (200, 0)  # Výchozí souřadnice

    def load_neutral_position(self):
        """Načte nebo nastaví neutrální pozici ruky"""
        try:
            # Nejprve zkusíme načíst neutrální pozici z kalibračního souboru
            if hasattr(
                    self,
                    'calibration_data') and self.calibration_data and "neutral_position" in self.calibration_data:
                neutral_pos = self.calibration_data["neutral_position"]
                print(
                    f"Neutrální pozice načtena z kalibračního souboru: {neutral_pos}")
                return neutral_pos
            else:
                # Výchozí neutrální pozice
                neutral_pos = {
                    "x": NEUTRAL_X,
                    "y": NEUTRAL_Y,
                    "z": NEUTRAL_Z
                }
                print(
                    f"Neutrální pozice nenalezena v kalibračním souboru, používám výchozí: {neutral_pos}")
                return neutral_pos
        except Exception as e:
            print(f"Chyba při načítání neutrální pozice: {e}")
            return {"x": NEUTRAL_X, "y": NEUTRAL_Y, "z": NEUTRAL_Z}

    def move_to_neutral_position(self):
        """Přesune ruku do neutrální pozice"""
        # V testech přeskočíme přesun ruky, pokud nemáme neutral_position
        try:
            if not hasattr(self, 'neutral_position'):
                self.neutral_position = {
                    "x": NEUTRAL_X,
                    "y": NEUTRAL_Y,
                    "z": NEUTRAL_Z
                }

            x = self.neutral_position.get("x", NEUTRAL_X)
            y = self.neutral_position.get("y", NEUTRAL_Y)
            z = self.neutral_position.get("z", NEUTRAL_Z)
        except Exception as e:
            print(f"Chyba při získávání neutrální pozice: {e}")
            x = NEUTRAL_X
            y = NEUTRAL_Y
            z = NEUTRAL_Z

        # Kontrola, zda již existuje status_label
        if hasattr(self, 'status_label'):
            self.status_label.setText(
                f"Přesouvám ruku do neutrální pozice ({x}, {y}, {z})...")
            # Přidáme vizuální styl pro lepší viditelnost
            self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #3498db;")
        else:
            print(f"Přesouvám ruku do neutrální pozice ({x}, {y}, {z})...")

        # Použití arm_thread, pokud je k dispozici
        if hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected:
            success = self.arm_thread.go_to_position(
                x=x, y=y, z=z, speed=MAX_SPEED, wait=True)
        # Záložní použití arm_controller
        elif hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected:
            success = self.arm_controller.go_to_position(
                x=x, y=y, z=z, speed=MAX_SPEED, wait=True)
        else:
            success = False

        if hasattr(self, 'status_label'):
            if success:
                self.status_label.setText("Ruka v neutrální pozici")
                # Po 1 sekundě skryjeme zprávu
                QTimer.singleShot(1000, lambda: self.status_label.setText(""))
                # Vrátíme původní styl
                QTimer.singleShot(1000, lambda: self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #e0e0e0;"))
            else:
                self.status_label.setText("Nepodařilo se přesunout ruku do neutrální pozice")
                self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #e74c3c;")

        return success

    def draw_winning_line(self):
        """🎨 Nakreslí výherní čáru přes tři symboly v řadě"""
        self.logger.info("🎨 ===== DRAW_WINNING_LINE SPUŠTĚNO =====")

        if not hasattr(self, 'board_widget') or not self.board_widget.winning_line:
            self.logger.warning("❌ No winning line available to draw")
            return False

        # Kontrola, zda je robotická ruka připojena
        arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected
        arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected

        if not (arm_thread_available or arm_controller_available):
            self.logger.warning("❌ Robotic arm not connected - cannot draw winning line")
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("Robotická ruka není připojena!")
            return False

        # Získání souřadnic výherní čáry
        winning_line = self.board_widget.winning_line
        if len(winning_line) != 3:
            self.logger.warning(f"❌ Invalid winning line length: {len(winning_line)}")
            return False

        self.logger.info(f"🎯 Drawing winning line through: {winning_line}")

        # Získání souřadnic prvního a posledního bodu výherní čáry
        start_row, start_col = winning_line[0]
        end_row, end_col = winning_line[2]

        # Získání souřadnic pro robotickou ruku
        start_coords = self.get_cell_coordinates_from_yolo(start_row, start_col)
        end_coords = self.get_cell_coordinates_from_yolo(end_row, end_col)

        if start_coords is None or end_coords is None:
            self.logger.warning("❌ Cannot draw winning line: cell coordinates not available from YOLO")
            # Try fallback method
            start_coords = self.get_cell_coordinates(start_row, start_col)
            end_coords = self.get_cell_coordinates(end_row, end_col)

            if start_coords is None or end_coords is None:
                self.logger.warning("❌ Cannot draw winning line: fallback coordinates also not available")
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("Nelze získat souřadnice pro výherní čáru")
                return False

        start_x, start_y = start_coords
        end_x, end_y = end_coords

        if start_x is None or start_y is None or end_x is None or end_y is None:
            self.logger.warning(f"❌ Invalid coordinates: start=({start_x}, {start_y}), end=({end_x}, {end_y})")
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("Neplatné souřadnice pro výherní čáru")
            return False

        # Nastavení výšky pro kreslení
        draw_z = DEFAULT_DRAW_Z
        safe_z = DEFAULT_SAFE_Z

        self.logger.info(f"🎨 Drawing celebration line from ({start_x:.1f}, {start_y:.1f}) to ({end_x:.1f}, {end_y:.1f})")

        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.setText(
                f"🎉 Kreslím výherní čáru z ({start_x:.1f}, {start_y:.1f}) do ({end_x:.1f}, {end_y:.1f})")

        success = False

        try:
            # Použití arm_thread, pokud je k dispozici
            if arm_thread_available:
                self.logger.info("🤖 Using arm_thread for celebration line")

                # Přesun na začátek čáry
                self.logger.info(f"📍 Moving to start position: ({start_x:.1f}, {start_y:.1f}, {safe_z})")
                self.arm_thread.go_to_position(
                    x=start_x, y=start_y, z=safe_z, speed=MAX_SPEED, wait=True)

                self.logger.info(f"📍 Lowering to draw height: {draw_z}")
                self.arm_thread.go_to_position(
                    x=start_x, y=start_y, z=draw_z, speed=DRAWING_SPEED, wait=True)

                # Kreslení čáry
                self.logger.info(f"✏️ Drawing line to end position: ({end_x:.1f}, {end_y:.1f})")
                self.arm_thread.go_to_position(
                    x=end_x, y=end_y, z=draw_z, speed=DRAWING_SPEED, wait=True)

                # Zvednutí pera
                self.logger.info(f"⬆️ Lifting to safe height: {safe_z}")
                self.arm_thread.go_to_position(
                    z=safe_z, speed=MAX_SPEED, wait=True)

                success = True
                self.logger.info("✅ Celebration line drawn successfully with arm_thread!")

            # Záložní použití arm_controller
            elif arm_controller_available:
                self.logger.info("🤖 Using arm_controller for celebration line")

                # Přesun na začátek čáry
                self.logger.info(f"📍 Moving to start position: ({start_x:.1f}, {start_y:.1f}, {safe_z})")
                self.arm_controller.go_to_position(
                    x=start_x, y=start_y, z=safe_z, speed=MAX_SPEED, wait=True)

                self.logger.info(f"📍 Lowering to draw height: {draw_z}")
                self.arm_controller.go_to_position(
                    x=start_x, y=start_y, z=draw_z, speed=DRAWING_SPEED, wait=True)

                # Kreslení čáry
                self.logger.info(f"✏️ Drawing line to end position: ({end_x:.1f}, {end_y:.1f})")
                self.arm_controller.go_to_position(
                    x=end_x, y=end_y, z=draw_z, speed=DRAWING_SPEED, wait=True)

                # Zvednutí pera
                self.logger.info(f"⬆️ Lifting to safe height: {safe_z}")
                self.arm_controller.go_to_position(
                    z=safe_z, speed=MAX_SPEED, wait=True)

                success = True
                self.logger.info("✅ Celebration line drawn successfully with arm_controller!")

        except Exception as e:
            self.logger.error(f"❌ Error drawing celebration line: {e}")
            success = False

        # Přesun do neutrální pozice
        self.logger.info("🏠 Moving to neutral position after celebration")
        self.move_to_neutral_position()

        if success:
            self.logger.info("🎉 WIN CELEBRATION COMPLETE!")
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("🏆 Výherní čára nakreslena!")
            return True
        else:
            self.logger.error("❌ Failed to draw celebration line")
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("❌ Chyba při kreslení výherní čáry")
            return False

    def handle_difficulty_changed(self, value):
        """Handle changes to the difficulty slider"""
        self.strategy_selector.difficulty = value
        self.difficulty_value_label.setText(f"{value}")
        # Update status in debug window if it exists and is visible
        if hasattr(self, 'debug_window') and self.debug_window is not None and hasattr(self.debug_window, 'status_label'):
            self.debug_window.status_label.setText(
                f"Obtížnost nastavena na {value}/10 (p={value / 10:.1f})")

    def handle_cell_clicked(self, row, col):
        """Handle clicks on the game board cells"""
        # Only allow clicks if it's the human's turn and the cell is empty
        if (not self.game_over and
            (self.current_turn is None or self.current_turn == self.human_player) and
                self.board_widget.board[row][col] == game_logic.EMPTY):

            # First move determines human player
            if self.human_player is None:
                self.human_player = game_logic.PLAYER_X
                self.ai_player = game_logic.PLAYER_O
                self.current_turn = self.ai_player

                # Initialize move counter for the first player move
                self.move_counter = 1

                # Remember the player's symbol for arm moves
                self.arm_player_symbol = self.human_player
                self.logger.info(f"First click: player is using symbol {self.human_player}")

                # For even-numbered moves (next turn = 2), the arm should play
                self.update_status(self.tr("arm_turn"))
                self.main_status_panel.setStyleSheet("""
                    background-color: #9b59b6;
                    border-radius: 10px;
                    border: 2px solid #8e44ad;
                """)

                # Pro zpětnou kompatibilitu
                self.status_label.setText("")
            else:
                self.current_turn = self.ai_player

                # Increment move counter
                self.move_counter += 1
                self.logger.info(f"Player clicked, move counter now: {self.move_counter}")

                # Decide whether to use arm or AI based on move counter
                if self.move_counter % 2 == 0:
                    # For even-numbered moves, the arm plays next
                    self.update_status(self.tr("arm_turn"))
                else:
                    # For odd-numbered moves, AI plays next
                    self.update_status(self.tr("ai_turn"))

                # Pro zpětnou kompatibilitu
                self.status_label.setText("")

            # NEAKTUALIZUJEME BOARD AUTOMATICKY - čekáme na YOLO detekci!
            # self.board_widget.board[row][col] = self.human_player
            # self.board_widget.update()

            # Check for game end
            self.check_game_end()

            # If game not over, determine whether to trigger arm or AI move
            if not self.game_over:
                if self.move_counter % 2 == 0:
                    # Even-numbered turns - arm's turn (after player)
                    # Pause slightly longer to let UI update fully
                    # CRITICAL: Use ai_player symbol (the one with fewer pieces)
                    arm_symbol = self.ai_player
                    self.logger.info(f"Even turn - arm plays {arm_symbol} (ai_player)")
                    QTimer.singleShot(300, lambda: self.make_arm_move(arm_symbol))
                else:
                    # Odd-numbered turns - AI's turn (after player)
                    # Pause slightly longer to let UI update fully
                    QTimer.singleShot(300, self.make_ai_move)

    def handle_detected_game_state(self, detected_board):
        """Handle game state detected from camera"""
        if not detected_board:
            return

        # CRITICAL: If game is over, do not process any more moves
        if self.game_over:
            self.logger.debug("🛑 Game is over - ignoring board detection updates")
            return

        # Logování aktuálního stavu pro debugging
        self.logger.debug(f"Detekovaný stav hry: {detected_board}")
        self.logger.debug(f"Aktuální stav: turn={self.current_turn}, waiting_for_detection={getattr(self, 'waiting_for_detection', False)}, game_over={self.game_over}")

        # Convert 1D list to 2D board if necessary
        if isinstance(detected_board, list) and len(detected_board) == 9:
            # Convert flat list to 2D board
            detected_board = [
                [detected_board[0], detected_board[1], detected_board[2]],
                [detected_board[3], detected_board[4], detected_board[5]],
                [detected_board[6], detected_board[7], detected_board[8]]
            ]

        # 🏆 REAL-TIME WIN DETECTION: Check for wins in every frame
        self._check_real_time_win_detection(detected_board)

        # Kontrola, zda je deska prázdná (začátek nové hry)
        is_empty_board = True
        for r in range(3):
            for c in range(3):
                if detected_board[r][c] != game_logic.EMPTY:
                    is_empty_board = False
                    break
            if not is_empty_board:
                break

        # Pokud je detekována prázdná deska, resetujeme hru
        if is_empty_board:
            # Resetujeme hru pouze pokud aktuální stav není prázdný
            # (abychom předešli zbytečnému resetování)
            current_is_empty = True
            for r in range(3):
                for c in range(3):
                    if self.board_widget.board[r][c] != game_logic.EMPTY:
                        current_is_empty = False
                        break
                if not current_is_empty:
                    break

            # OPRAVA: Resetujeme hru pokud je prázdná deska detekována a buď:
            # 1. Aktuální stav není prázdný (normální případ)
            # 2. Máme výherní čáru (potřebujeme ji vymazat)
            # 3. Hra je ukončená (potřebujeme začít novou)
            should_reset = (not current_is_empty or
                          (hasattr(self, 'board_widget') and self.board_widget and self.board_widget.winning_line) or
                          (hasattr(self, 'game_over') and self.game_over))

            if should_reset:
                self.logger.info("🆕 Prázdná hrací plocha detekována - resetuji hru a čistím výherní čáru")

                # OPRAVA 1: Vyčistit výherní čáru před resetem hry
                if hasattr(self, 'board_widget') and self.board_widget:
                    self.board_widget.winning_line = None
                    self.board_widget.update()
                    self.logger.info("✅ Výherní čára vymazána")

                # OPRAVA 2: Reset všech arm flags pro novou hru
                self.reset_arm_flags()
                self.logger.info("✅ Všechny arm flags resetovány pro novou hru")

                self.reset_game()
                self.status_label.setText("")
            return

        # Pokud hra ještě nezačala, zkontrolujeme, zda je na desce nějaký
        # symbol
        if self.human_player is None:
            # Hledáme první symbol na desce
            for r in range(3):
                for c in range(3):
                    if detected_board[r][c] != game_logic.EMPTY:
                        # První symbol určuje hráče
                        self.human_player = detected_board[r][c]

                        # Make sure we have a valid detected symbol
                        if not self.human_player or self.human_player == game_logic.EMPTY:
                            self.human_player = game_logic.PLAYER_X
                            self.logger.warning(f"Failed to detect valid symbol, using default X")

                        # Count symbols to determine which one arm should play
                        x_count = sum(row.count(game_logic.PLAYER_X) for row in detected_board)
                        o_count = sum(row.count(game_logic.PLAYER_O) for row in detected_board)
                        total_symbols = x_count + o_count

                        # OPRAVA: Pokud je sudý počet symbolů, čekáme na tah hráče
                        # Lichý počet symbolů = ruka hraje (po tahu hráče)
                        if total_symbols % 2 == 0:
                            self.logger.info(f"Symbol count: X={x_count}, O={o_count}, total={total_symbols} (sudý) → čekáme na tah hráče")
                            # Určíme, kdo je hráč a kdo je AI podle toho, kterého symbolu je méně
                            if x_count < o_count:
                                self.human_player = game_logic.PLAYER_X
                                self.ai_player = game_logic.PLAYER_O
                            elif o_count < x_count:
                                self.human_player = game_logic.PLAYER_O
                                self.ai_player = game_logic.PLAYER_X
                            else:
                                # Stejný počet - výchozí nastavení
                                self.human_player = game_logic.PLAYER_X
                                self.ai_player = game_logic.PLAYER_O

                            self.current_turn = self.human_player
                            self.update_status(self.tr("your_turn"))
                            return

                        # Arm plays the symbol with fewer pieces
                        if x_count < o_count:
                            self.ai_player = game_logic.PLAYER_X
                        elif o_count < x_count:
                            self.ai_player = game_logic.PLAYER_O
                        else:
                            # Equal count - use opposite of human player as fallback
                            self.ai_player = game_logic.PLAYER_O if self.human_player == game_logic.PLAYER_X else game_logic.PLAYER_X

                        self.logger.info(f"Symbol count: X={x_count}, O={o_count} → AI plays {self.ai_player}")
                        self.current_turn = self.ai_player

                        # Inicializuj počítadlo tahů (první tah hráče)
                        self.move_counter = 1

                        # Zapamatuj si symbol hráče pro tahy ruky
                        self.arm_player_symbol = self.human_player
                        # Set a default symbol if detection failed to identify it
                        if not self.arm_player_symbol:
                            self.arm_player_symbol = game_logic.PLAYER_X  # Fallback to X
                            self.logger.warning(f"Failed to detect player symbol, using fallback: {self.arm_player_symbol}")
                        else:
                            self.logger.info(f"První detekovaný tah: hráč je {self.arm_player_symbol}")

                        # Store the last move time to force a move if detection keeps happening without action
                        if not hasattr(self, 'last_move_time'):
                            self.last_move_time = 0
                        current_time = time.time()

                        # OPRAVA: Pouze při lichém počtu symbolů spustíme tah ruky
                        # Lichý počet znamená, že hráč právě hrál a teď je na tahu ruka
                        self.update_status(self.tr("arm_turn"))
                        self.main_status_panel.setStyleSheet("""
                            background-color: #9b59b6;
                            border-radius: 10px;
                            border: 2px solid #8e44ad;
                        """)

                        # Pro zpětnou kompatibilitu
                        if hasattr(self, 'status_label') and self.status_label:
                            self.status_label.setText("")

                        # Set a flag to indicate we're waiting for AI/arm to move
                        # This will prevent status flickering during continuous detection
                        self._status_lock = True

                        # Ensure we don't execute duplicate moves too quickly
                        if current_time - self.last_move_time > 3.0:
                            self.last_move_time = current_time

                            # Make the ARM move immediately for better reliability
                            self.logger.info(f"Starting ARM move immediately after first detection with symbol {self.ai_player}")
                            self.make_arm_move_with_symbol(self.ai_player)

                        return

        # Pokud je hra v průběhu a je tah hráče, kontrolujeme změny na desce
        elif self.current_turn == self.human_player:
            self.logger.info(f"🎮 Checking for human move. Current turn: {self.current_turn}, Human: {self.human_player}")

            # Debug: Log both boards
            self.logger.debug(f"Widget board: {self.board_widget.board}")
            self.logger.debug(f"Detected board: {detected_board}")

            # Porovnáme aktuální stav desky s detekovaným stavem
            diff = game_logic.get_board_diff(
                self.board_widget.board, detected_board)

            self.logger.info(f"🔍 Board diff: {diff}")
            # Pokud je přesně jedna změna a je to symbol hráče, aktualizujeme
            # stav
            if len(diff) == 1:
                r, c, symbol = diff[0]
                self.logger.info(f"🎯 Single change detected at ({r},{c}): {symbol}")
                if symbol == self.human_player:
                    # ✅ CRITICAL FIX: ONLY update board with YOLO detections
                    # Never artificially place symbols - only show what YOLO actually sees
                    self.board_widget.board = [
                        row[:] for row in detected_board]  # Deep copy of YOLO-detected board
                    self.board_widget.update()

                    self.logger.info(f"📋 PLAYER MOVE DETECTED - BOARD UPDATED FROM YOLO: {detected_board}")

                    # Increment move counter
                    self.move_counter += 1
                    self.logger.info(f"Detekován tah hráče, počítadlo tahů: {self.move_counter}")

                    # Kontrola konce hry
                    self.check_game_end()

                    # Pokud hra neskončila, předáme tah AI nebo ruce
                    if not self.game_over:
                        # Kontrola, zda máme všech 16 grid points detekovaných
                        valid_grid = False
                        if hasattr(self, 'camera_thread') and self.camera_thread:
                            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                                if hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                                    game_state = self.camera_thread.detection_thread.detector.game_state
                                    if game_state and hasattr(game_state, 'is_physical_grid_valid'):
                                        valid_grid = game_state.is_physical_grid_valid()

                        # FIXED LOGIC: Arm plays ONLY when there is an odd number of total symbols (1, 3, 5, 7, 9)
                        x_count = sum(row.count(game_logic.PLAYER_X) for row in detected_board)
                        o_count = sum(row.count(game_logic.PLAYER_O) for row in detected_board)
                        total_symbols = x_count + o_count

                        self.logger.info(f"🔍 TURN ANALYSIS: X={x_count}, O={o_count}, total={total_symbols}, valid_grid={valid_grid}")

                        if not valid_grid:
                            self.logger.debug("AI turn skipped - grid is not valid (missing grid points)")
                            return

                        # Check if we're waiting for detection after arm move or if arm is busy
                        if self.waiting_for_detection or self.arm_move_in_progress or self.arm_move_scheduled or (hasattr(self, '_arm_move_scheduled') and self._arm_move_scheduled):
                            self.logger.info(f"🕐 Arm busy state: waiting_for_detection={self.waiting_for_detection}, "
                                            f"arm_move_in_progress={self.arm_move_in_progress}, "
                                            f"arm_move_scheduled={self.arm_move_scheduled}, "
                                            f"_arm_move_scheduled={getattr(self, '_arm_move_scheduled', False)}")

                            # Check if we've been waiting too long (stuck state)
                            if hasattr(self, 'last_arm_busy_log_time'):
                                if time.time() - self.last_arm_busy_log_time > 5.0:
                                    self.logger.warning("⚠️ Arm has been busy for too long, might be stuck!")
                                    # Reset flags if stuck
                                    self.reset_arm_flags()
                            else:
                                self.last_arm_busy_log_time = time.time()
                            return

                        # FIXED: Check if arm should play based on symbol count
                        # Arm plays when there's an odd number of symbols (after human plays)
                        if total_symbols % 2 == 0:
                            self.logger.info(f"🚫 ARM DOES NOT PLAY - symbol count is even ({total_symbols}), waiting for player move")
                            # Set turn to human player
                            self.current_turn = self.human_player
                            self.update_status(self.tr("your_turn"))
                            # Also update label with clear instruction
                            if hasattr(self, 'status_label') and self.status_label:
                                self.status_label.setText(f"Váš tah - počet symbolů: {total_symbols} (sudý)")
                            return

                        # ARM PLAYS when there is an odd number of symbols (human just played)
                        self.logger.info(f"🤖 ARM'S TURN - symbol count is odd ({total_symbols})")

                        # FIXED: Determine arm symbol based on which appears less frequently
                        if x_count < o_count:
                            # Fewer X than O → arm plays X
                            arm_symbol = game_logic.PLAYER_X
                            self.logger.info(f"✅ Arm plays X (X={x_count} < O={o_count})")
                        elif o_count < x_count:
                            # Fewer O than X → arm plays O
                            arm_symbol = game_logic.PLAYER_O
                            self.logger.info(f"✅ Arm plays O (O={o_count} < X={x_count})")
                        else:
                            # Equal count → arm plays X (default, since X goes first)
                            arm_symbol = game_logic.PLAYER_X
                            self.logger.info(f"✅ Arm plays X (X={x_count} = O={o_count}, default to X)")

                        # VALIDATION: Check if the game state is valid (proper turn order)
                        if not self._validate_game_state(detected_board):
                            self.logger.warning("⚠️ Invalid game state detected - skipping arm move")
                            return

                        # Nastavíme AI symbol pro strategii
                        self.ai_player = arm_symbol
                        self.current_turn = self.ai_player

                        self.logger.info(f"🎯 PŘIPRAVUJI TAH RUKY: symbol={arm_symbol}, current_turn={self.current_turn}")

                        # Store the last move time to force a move if detection keeps happening without action
                        if not hasattr(self, 'last_move_time'):
                            self.last_move_time = 0
                        current_time = time.time()

                        if hasattr(self, 'status_label') and self.status_label:
                            self.status_label.setText("")

                        # Nastavíme čas posledního tahu
                        self.last_move_time = current_time

                        # Aktualizujeme hlavní stavovou zprávu pro tah ruky
                        self.update_status(self.tr("arm_turn"))
                        self.main_status_panel.setStyleSheet("""
                            background-color: #9b59b6;
                            border-radius: 10px;
                            border: 2px solid #8e44ad;
                        """)

                        # Pro zpětnou kompatibilitu
                        if hasattr(self, 'status_label') and self.status_label:
                            self.status_label.setText("")

                        # Check if arm move is already scheduled or in progress
                        if (hasattr(self, '_arm_move_scheduled') and self._arm_move_scheduled) or self.arm_move_in_progress or self.arm_move_scheduled:
                            self.logger.info(f"🤖 Tah ruky je již naplánován nebo probíhá, přeskakuji duplikátní plánování")
                            self.logger.info(f"🤖 DEBUG: _arm_move_scheduled={getattr(self, '_arm_move_scheduled', False)}")
                            self.logger.info(f"🤖 DEBUG: arm_move_in_progress={self.arm_move_in_progress}")
                            self.logger.info(f"🤖 DEBUG: arm_move_scheduled={self.arm_move_scheduled}")
                            return

                        # Provedeme tah ruky s vybraným symbolem
                        self.logger.info(f"🤖 SPOUŠTÍM TIMER PRO TAH RUKY se symbolem {arm_symbol}")

                        # Mark that arm move is scheduled to prevent duplicates
                        self._arm_move_scheduled = True
                        def execute_arm_move():
                            self.logger.info(f"⏰ TIMER SPUŠTĚN - volám make_arm_move_with_symbol({arm_symbol})")
                            # Don't set flags here - let make_arm_move_with_symbol handle all flag management
                            self.make_arm_move_with_symbol(arm_symbol)

                            # Clear the scheduled flag after executing
                            if hasattr(self, '_arm_move_scheduled'):
                                delattr(self, '_arm_move_scheduled')

                        # Increased delay to ensure proper game flow
                        QTimer.singleShot(2000, execute_arm_move)  # 2 second delay

        # CRITICAL: If game is over, check for new game
        elif self.game_over:
            # Check if board is empty (new game started)
            is_empty = all(cell == game_logic.EMPTY for row in detected_board for cell in row)

            if is_empty:
                self.logger.info("🆕 Empty board detected after game over - starting new game!")
                # Reset the game
                self.reset_game()
                # Update status for new game
                self.update_status(self.tr("new_game_detected"))
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("Nová hra! Začněte umístěním symbolu.")
                return
            else:
                self.logger.debug("🛑 Game is over - ignoring all board updates")
                return

        # Pokud je hra resetována, ale na desce jsou symboly,
        # aktualizujeme GUI podle detekovaného stavu
        elif self.current_turn is None:
            # Zjistíme, kolik symbolů X a O je na desce
            x_count = sum(row.count(game_logic.PLAYER_X)
                          for row in detected_board)
            o_count = sum(row.count(game_logic.PLAYER_O)
                          for row in detected_board)

            # Pokud je na desce alespoň jeden symbol, aktualizujeme GUI
            if x_count > 0 or o_count > 0:
                # Určíme symboly podle toho, kterého je méně (ruka hraje ten vzácnější)
                if x_count < o_count:
                    # Méně X → ruka hraje X, hráč hraje O
                    self.human_player = game_logic.PLAYER_O
                    self.ai_player = game_logic.PLAYER_X
                    self.logger.info(f"Reset detection: X={x_count} < O={o_count} → arm plays X")
                elif o_count < x_count:
                    # Méně O → ruka hraje O, hráč hraje X
                    self.human_player = game_logic.PLAYER_X
                    self.ai_player = game_logic.PLAYER_O
                    self.logger.info(f"Reset detection: O={o_count} < X={x_count} → arm plays O")
                else:
                    # Stejný počet - určíme podle toho, kdo je na tahu
                    if (x_count + o_count) % 2 == 0:
                        # Sudý počet symbolů → X je na tahu
                        self.current_turn = game_logic.PLAYER_X
                        self.human_player = game_logic.PLAYER_X
                        self.ai_player = game_logic.PLAYER_O
                    else:
                        # Lichý počet symbolů → O je na tahu
                        self.current_turn = game_logic.PLAYER_O
                        self.human_player = game_logic.PLAYER_O
                        self.ai_player = game_logic.PLAYER_X
                    self.logger.info(f"Reset detection: X={x_count} = O={o_count} → using turn order")

                # Nastavíme, kdo je na tahu podle počtu symbolů
                total_symbols = x_count + o_count
                if total_symbols % 2 == 0:
                    # Sudý počet → hráč je na tahu
                    self.current_turn = self.human_player
                else:
                    # Lichý počet → ruka (AI) je na tahu
                    self.current_turn = self.ai_player

                # ✅ CRITICAL FIX: ONLY update board with actual YOLO detections
                # Never artificially place symbols - only show what YOLO actually sees
                self.board_widget.board = [
                    row[:] for row in detected_board]  # Deep copy of YOLO-detected board
                self.board_widget.update()

                self.logger.info(f"📋 BOARD UPDATED FROM YOLO DETECTION ONLY: {detected_board}")

                # Kontrola konce hry
                self.check_game_end()

                if not self.game_over:
                    # Aktualizujeme hlavní stavovou zprávu
                    self.update_status("AI PŘEMÝŠLÍ...")
                    self.main_status_panel.setStyleSheet("""
                        background-color: #3498db;
                        border-radius: 10px;
                        border: 2px solid #2980b9;
                    """)

                    # Pro zpětnou kompatibilitu
                    if hasattr(self, 'status_label') and self.status_label:
                        self.status_label.setText("")

                    # Spustíme tah ruky pokud je na tahu AI, s výrazným zpožděním pro stabilnější UI
                    if self.current_turn == self.ai_player:
                        self.logger.info(f"🤖 Spouštím tah ruky se symbolem {self.ai_player} po reset detekci")
                        QTimer.singleShot(1000, lambda: self.make_arm_move_with_symbol(self.ai_player))

        # 🤖 UNIFIED ARM MOVE LOGIC - Single point of control
        # DISABLED - This duplicates arm move logic already handled in the update_board_from_detection method
        # if detected_board and not self.game_over:
        #     should_play, arm_symbol = self._should_arm_play_now(detected_board)
        #
        #     if should_play and arm_symbol:
        #         self.logger.info(f"🤖 UNIFIED LOGIC: Executing arm move with {arm_symbol}")
        #         self._execute_arm_move(arm_symbol)

    def update_game_state(self):
        """Periodic update for game state and AI moves"""
        # Zkontrolujeme, zda máme aktivní varování o špatně viditelné mřížce
        if hasattr(self, 'grid_warning_active') and self.grid_warning_active:
            # Pokud máme aktivní varování, přerušíme aktualizaci stavů hry
            return

        # Kontrola, zda je robotická ruka připojena
        arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected
        arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected

        # Park the arm when it's human's turn or game is over
        if (self.game_over or self.current_turn == self.human_player or self.current_turn is None) and \
           (arm_thread_available or arm_controller_available):
            # We could park the arm here, but it's not necessary to do it on every update
            # We'll only park when explicitly requested or when closing the app
            pass

        # Pokud čekáme na detekci nakresleného symbolu
        if self.waiting_for_detection:
            self.logger.debug(f"⏳ WAITING_FOR_DETECTION=True, ai_move_row={getattr(self, 'ai_move_row', None)}, ai_move_col={getattr(self, 'ai_move_col', None)}, expected_symbol={getattr(self, 'expected_symbol', None)}")
            # Zvýšíme čas čekání
            self.detection_wait_time += 0.1  # Předpokládáme, že timer se volá každých 100ms

            # Kontrola, zda byl symbol detekován
            if hasattr(self, 'camera_thread') and self.camera_thread.last_board_state:
                detected_board = self.camera_thread.last_board_state

                # Převedeme 1D list na 2D board pokud je potřeba
                if isinstance(detected_board, list) and len(detected_board) == 9:
                    detected_board_2d = [
                        [detected_board[0], detected_board[1], detected_board[2]],
                        [detected_board[3], detected_board[4], detected_board[5]],
                        [detected_board[6], detected_board[7], detected_board[8]]
                    ]
                else:
                    detected_board_2d = detected_board

                # Kontrola, zda byl symbol detekován na správné pozici
                if (0 <= self.ai_move_row < 3 and 0 <= self.ai_move_col < 3):
                    expected_symbol = getattr(self, 'expected_symbol', self.ai_player)
                    detected_symbol = detected_board_2d[self.ai_move_row][self.ai_move_col]

                    self.logger.info(f"🔍 Kontrola detekce: pozice ({self.ai_move_row},{self.ai_move_col}), "
                                   f"očekávaný symbol: {expected_symbol}, detekovaný: {detected_symbol}")

                    # Debug: vypsat celou desku
                    self.logger.debug("🎮 Aktuální detekovaná deska:")
                    for r in range(3):
                        row_str = ""
                        for c in range(3):
                            cell = detected_board_2d[r][c]
                            if cell == game_logic.EMPTY:
                                row_str += "[ ]"
                            else:
                                row_str += f"[{cell}]"
                        self.logger.debug(f"  {row_str}")

                    if detected_symbol == expected_symbol and detected_symbol != game_logic.EMPTY:
                        # Symbol byl detekován, můžeme pokračovat
                        self.logger.info(f"✅ Symbol {expected_symbol} úspěšně detekován na pozici ({self.ai_move_row},{self.ai_move_col})")
                        self.waiting_for_detection = False
                        self.detection_wait_time = 0
                        self.ai_move_retry_count = 0
                        self._move_in_progress = False  # Reset move flag
                        # Clear all arm move flags
                        self.arm_move_in_progress = False
                        self.arm_move_scheduled = False
                        # Clear the scheduled flag
                        if hasattr(self, '_arm_move_scheduled'):
                            delattr(self, '_arm_move_scheduled')
                        # Also clear any move-related state
                        self.ai_move_row = None
                        self.ai_move_col = None
                        self.expected_symbol = None
                        self.detection_wait_time = 0
                        self.ai_move_retry_count = 0
                        self.logger.info(f"🤖 All arm move flags and state cleared after successful detection")

                        # ✅ CRITICAL: DO NOT update board directly here - ONLY YOLO detection updates board
                        # The detection system will handle board updates through proper channels
                        self.logger.info("✅ Symbol detection confirmed - board will be updated by YOLO detection only")

                        # Update the board widget to show the detected state
                        self.board_widget.board = [row[:] for row in detected_board_2d]
                        self.board_widget.update()

                        # Kontrola konce hry
                        self.check_game_end()

                        if not self.game_over:
                            # After arm move is detected, we need to check whose turn it should be
                            # based on the total number of symbols on the board
                            x_count = sum(row.count(game_logic.PLAYER_X) for row in detected_board_2d)
                            o_count = sum(row.count(game_logic.PLAYER_O) for row in detected_board_2d)
                            total_symbols = x_count + o_count

                            self.logger.info(f"🔄 After arm move detection: X={x_count}, O={o_count}, total={total_symbols}")

                            # OPRAVA: Po tahu ruky je vždy na tahu hráč
                            # Ruka hraje pouze když je lichý počet symbolů (po tahu hráče)
                            # Po tahu ruky je sudý počet symbolů, takže je na tahu hráč
                            self.current_turn = self.human_player
                            self.logger.info(f"🎯 After arm move: turn passed to human player ({self.human_player})")

                            # Update status for human turn
                            self.update_status(self.tr("your_turn"))
                            if hasattr(self, 'status_label') and self.status_label:
                                self.status_label.setText(f"Váš tah ({self.human_player})")

                            # Don't force board update - let the natural detection flow handle it
                            self.logger.info("✅ Arm move detected successfully. Waiting for human player's move.")

                        return
                    else:
                        self.logger.debug(f"❌ Symbol {expected_symbol} ještě nebyl detekován na pozici ({self.ai_move_row},{self.ai_move_col}). "
                                        f"Detekovaný symbol: {detected_symbol}")
                else:
                    self.logger.warning(f"⚠️ Neplatná pozice pro kontrolu detekce: ({self.ai_move_row},{self.ai_move_col})")

            # Pokud vypršel čas čekání a symbol nebyl detekován
            if self.detection_wait_time >= self.max_detection_wait_time:
                self.logger.warning(f"⏰ Detection timeout! Waited {self.detection_wait_time}s for symbol {getattr(self, 'expected_symbol', '?')} at ({getattr(self, 'ai_move_row', '?')},{getattr(self, 'ai_move_col', '?')})")
                self.detection_wait_time = 0
                self.waiting_for_detection = False

                # Clear all arm move flags
                self.arm_move_in_progress = False
                self.arm_move_scheduled = False
                if hasattr(self, '_arm_move_scheduled'):
                    delattr(self, '_arm_move_scheduled')

                # Pokud jsme nepřekročili maximální počet pokusů, zkusíme
                # nakreslit symbol znovu
                if self.ai_move_retry_count < self.max_retry_count:
                    self.ai_move_retry_count += 1
                    self.status_label.setText(
                        f"⚠️ Symbol nebyl detekován, zkouším znovu (pokus {self.ai_move_retry_count}/{self.max_retry_count})...")
                    self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #FFA500;")

                    # Kontrola, zda je robotická ruka připojena
                    arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected
                    arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected

                    # Zkusíme nakreslit symbol znovu
                    if arm_thread_available or arm_controller_available:
                        if self.draw_ai_symbol(
                                self.ai_move_row, self.ai_move_col, self.ai_player):
                            # Začneme znovu čekat na detekci
                            self.waiting_for_detection = True
                        else:
                            # Pokud kreslení selhalo, vzdáme to
                            self.status_label.setText(
                                "❌ Chyba při kreslení, vzdávám to.")
                            self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #FF5555;")
                            self.current_turn = self.human_player
                else:
                    # Pokud jsme vyčerpali všechny pokusy, vzdáme to
                    self.status_label.setText(
                        "⚠️ Symbol se nepodařilo nakreslit po několika pokusech. Pokračujeme dál.")
                    self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #FFA500;")
                    self.current_turn = self.human_player
                    self.update_status(self.tr("your_turn"))

                    # Reset all move-related variables
                    self.ai_move_row = None
                    self.ai_move_col = None
                    self.expected_symbol = None
                    self.ai_move_retry_count = 0

            # Pokud stále čekáme, nepokračujeme dál
            return

        # DUPLICITNÍ LOGIKA ODSTRANĚNA - tahy ruky se nyní provádějí pouze v update_board_from_detection()
        # Tato sekce byla nahrazena novou logikou v update_board_from_detection()
        pass

    def draw_ai_symbol(self, row, col, symbol=None):
        """Make the robot arm draw the AI's symbol"""
        # Kontrola, zda je robotická ruka připojena
        arm_thread_available = hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected
        arm_controller_available = hasattr(self, 'arm_controller') and self.arm_controller and self.arm_controller.connected

        if not (arm_thread_available or arm_controller_available):
            self.status_label.setText("")
            print(f"Robot by nyní nakreslil {symbol or self.ai_player} na pozici ({row}, {col})")
            return False

        # Zkontrolujeme, zda máme platnou mřížku před provedením tahu
        valid_grid = False

        # Získáme poslední detekovaný stav z kamery
        if hasattr(self, 'camera_thread') and self.camera_thread:
            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                if hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                    game_state = self.camera_thread.detection_thread.detector.game_state
                    if game_state and hasattr(game_state, 'is_physical_grid_valid'):
                        valid_grid = game_state.is_physical_grid_valid()

        if not valid_grid:
            self.logger.warning("Nelze nakreslit symbol - mřížka není validní!")
            self.update_status("Umístěte hrací plochu do záběru kamery")
            self.set_status_style_safe("error", """
                background-color: #e74c3c;
                border-radius: 10px;
                border: 2px solid #c0392b;
            """)
            return False

        # Pokud není zadán symbol, použijeme symbol AI
        if symbol is None:
            symbol = self.ai_player

        # Logování pro debugging
        self.logger.info(f"draw_ai_symbol: Kreslím symbol {symbol} na pozici ({row}, {col})")

        # Získání souřadnic z YOLO detekcí
        target_x, target_y = self.get_cell_coordinates_from_yolo(row, col)

        if target_x is None or target_y is None:
            # Pokud nemáme souřadnice z YOLO, použijeme výchozí hodnoty
            # Hardcoded coordinates for the 3x3 grid
            # These values should be calibrated for your specific setup
            grid_center_x = 200  # Center X coordinate of the grid in mm
            grid_center_y = 0    # Center Y coordinate of the grid in mm
            cell_size = 50       # Size of each cell in mm

            # Calculate target coordinates based on row and column
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size

            self.status_label.setText(
                f"Kreslím {symbol} na pozici ({row}, {col}) s výchozími souřadnicemi...")
        else:
            self.status_label.setText(
                f"Kreslím {symbol} na pozici ({row}, {col}) se souřadnicemi z YOLO...")

        # Draw the appropriate symbol
        success = False

        # Použití arm_thread, pokud je k dispozici
        if arm_thread_available:
            if symbol == game_logic.PLAYER_O:
                self.logger.info(f"Kreslím O pomocí arm_thread na ({target_x}, {target_y})")
                success = self.arm_thread.draw_o(
                    target_x, target_y, DEFAULT_SYMBOL_SIZE_MM / 2, speed=DRAWING_SPEED)
            else:
                self.logger.info(f"Kreslím X pomocí arm_thread na ({target_x}, {target_y})")
                success = self.arm_thread.draw_x(
                    target_x, target_y, DEFAULT_SYMBOL_SIZE_MM, speed=DRAWING_SPEED)
        # Záložní použití arm_controller
        elif arm_controller_available:
            if symbol == game_logic.PLAYER_O:
                self.logger.info(f"Kreslím O pomocí arm_controller na ({target_x}, {target_y})")
                success = self.arm_controller.draw_o(
                    target_x, target_y, DEFAULT_SYMBOL_SIZE_MM / 2, speed=DRAWING_SPEED)
            else:
                self.logger.info(f"Kreslím X pomocí arm_controller na ({target_x}, {target_y})")
                success = self.arm_controller.draw_x(
                    target_x, target_y, DEFAULT_SYMBOL_SIZE_MM, speed=DRAWING_SPEED)

        if success:
            self.status_label.setText(
                f"Symbol {symbol} nakreslen na souřadnicích ({target_x:.1f}, {target_y:.1f}).")
            # Přesun do neutrálního stavu po nakreslení symbolu
            self.move_to_neutral_position()

            # Wait 2 seconds after moving to neutral position to allow camera detection
            self.logger.info("⏱️ Waiting 2 seconds for camera to detect the new symbol...")
            import time
            time.sleep(2.0)

            return True
        else:
            self.status_label.setText("")
            return False

    def get_cell_coordinates_from_yolo(self, row, col):
        """Získá souřadnice buňky z YOLO detekcí a aplikuje kalibraci"""
        # Nejprve zkusíme použít přímé mapování z kalibračních dat
        if hasattr(self, 'calibration_data') and self.calibration_data and "grid_positions" in self.calibration_data:
            grid_positions = self.calibration_data["grid_positions"]
            cell_key = f"{row}_{col}"
            if cell_key in grid_positions:
                target_x = grid_positions[cell_key]["x"]
                target_y = grid_positions[cell_key]["y"]
                self.logger.info(f"Používám kalibrované souřadnice pro buňku ({row}, {col}): ({target_x}, {target_y})")
                return target_x, target_y

        # Pokud nemáme přímé mapování, zkusíme získat souřadnice z kamery
        if not hasattr(self, 'camera_thread') or not self.camera_thread:
            # Pokud nemáme kameru, použijeme výchozí souřadnice
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            self.logger.warning(f"Kamera není k dispozici, používám výchozí souřadnice: ({target_x}, {target_y})")
            return target_x, target_y

        # Získáme poslední detekovaný stav z kamery - debug kontroly
        if not hasattr(self.camera_thread, 'detection_thread'):
            self.logger.warning(f"camera_thread nemá detection_thread, používám výchozí souřadnice")
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            return target_x, target_y

        if not self.camera_thread.detection_thread:
            self.logger.warning(f"detection_thread je None, používám výchozí souřadnice")
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            return target_x, target_y

        if not hasattr(self.camera_thread.detection_thread, 'detector'):
            self.logger.warning(f"detection_thread nemá detector, používám výchozí souřadnice")
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            return target_x, target_y

        if not self.camera_thread.detection_thread.detector:
            self.logger.warning(f"detector je None, používám výchozí souřadnice")
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            return target_x, target_y

        # Získáme objekt GameState z detektoru
        game_state = None
        if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
            if hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                game_state = self.camera_thread.detection_thread.detector.game_state

        if not game_state or not game_state.is_valid():
            # Pokud nemáme platný stav hry, použijeme výchozí souřadnice
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            self.logger.warning(f"Stav hry není platný, používám výchozí souřadnice: ({target_x}, {target_y})")
            return target_x, target_y

        # Získáme souřadnice středu buňky v pixelech
        cell_center_uv = game_state.get_cell_center_uv(row, col)

        if cell_center_uv is None or len(cell_center_uv) == 0:
            self.logger.warning(f"Nepodařilo se získat souřadnice středu buňky ({row}, {col})")
            # Pokud nemáme souřadnice buňky, použijeme výchozí souřadnice
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            self.logger.warning(f"Používám výchozí souřadnice: ({target_x}, {target_y})")
            return target_x, target_y

        self.logger.info(f"Získány souřadnice buňky ({row}, {col}) v pixelech: ({cell_center_uv[0]:.1f}, {cell_center_uv[1]:.1f})")

        # Použijeme kalibrační data pro převod z UV (pixely) na XY (milimetry)
        if hasattr(self, 'calibration_data') and self.calibration_data:
            # Kontrola, zda máme transformační matici
            if "uv_to_xy_matrix" in self.calibration_data:
                try:
                    # Převedeme souřadnice pomocí homografie
                    uv_to_xy_matrix = np.array(
                        self.calibration_data["uv_to_xy_matrix"])

                    # Příprava bodu pro transformaci (potřebujeme homogenní
                    # souřadnice)
                    uv_point = np.array(
                        [[cell_center_uv[0], cell_center_uv[1], 1.0]], dtype=np.float32).T

                    # Aplikace transformace
                    xy_point = np.matmul(uv_to_xy_matrix, uv_point)

                    # Normalizace homogenních souřadnic
                    if xy_point[2, 0] != 0:
                        arm_x = xy_point[0, 0] / xy_point[2, 0]
                        arm_y = xy_point[1, 0] / xy_point[2, 0]

                        self.logger.info(
                            f"Transformované souřadnice: UV({cell_center_uv[0]:.1f}, {cell_center_uv[1]:.1f}) -> XY({arm_x:.1f}, {arm_y:.1f})")
                        return arm_x, arm_y
                except Exception as e:
                    self.logger.error(f"Chyba při transformaci souřadnic: {e}")

        # Pokud nemáme kalibrační data nebo transformace selhala, použijeme
        # zjednodušenou metodu
        self.logger.info("Používám zjednodušenou transformaci souřadnic (bez kalibrace)")

        # Získáme rozměry snímku z kamery
        frame_width = self.camera_thread.detection_thread.detector.frame_width
        frame_height = self.camera_thread.detection_thread.detector.frame_height

        if not frame_width or not frame_height:
            self.logger.warning("Neznámé rozměry snímku z kamery")
            # Pokud nemáme rozměry snímku, použijeme výchozí souřadnice
            grid_center_x = 200
            grid_center_y = 0
            cell_size = 50
            target_x = grid_center_x + (col - 1) * cell_size
            target_y = grid_center_y + (row - 1) * cell_size
            self.logger.warning(f"Používám výchozí souřadnice: ({target_x}, {target_y})")
            return target_x, target_y

        # Převedeme souřadnice z pixelů na normalizované souřadnice (0-1)
        norm_u = cell_center_uv[0] / frame_width
        norm_v = cell_center_uv[1] / frame_height

        # Pracovní prostor robotické ruky - přesnější hodnoty z kalibrace
        arm_min_x = 150
        arm_max_x = 250
        arm_min_y = -50
        arm_max_y = 50

        # Pokud máme kalibrační data, použijeme je
        if hasattr(self, 'calibration_data') and self.calibration_data:
            if "arm_workspace" in self.calibration_data:
                workspace = self.calibration_data["arm_workspace"]
                arm_min_x = workspace.get("min_x", arm_min_x)
                arm_max_x = workspace.get("max_x", arm_max_x)
                arm_min_y = workspace.get("min_y", arm_min_y)
                arm_max_y = workspace.get("max_y", arm_max_y)
                self.logger.info(f"Použity kalibrační hodnoty pro pracovní prostor: X({arm_min_x}-{arm_max_x}), Y({arm_min_y}-{arm_max_y})")
            else:
                self.logger.warning("Chybí kalibrační data pro pracovní prostor, používám výchozí hodnoty")

        # Převedeme normalizované souřadnice na souřadnice robotické ruky
        # Invertujeme osu Y, protože v obraze je osa Y směrem dolů, ale v
        # robotické ruce je směrem nahoru
        # Použití správného mapování z kalibračních dat
        arm_x = arm_min_x + norm_u * (arm_max_x - arm_min_x)
        arm_y = arm_min_y + (1 - norm_v) * (arm_max_y - arm_min_y)

        self.logger.info(f"Vypočtené souřadnice pro ruku: ({arm_x:.1f}, {arm_y:.1f})")
        return arm_x, arm_y

    def process_game_state(self):
        """Process the current game state and make AI moves if needed"""
        # If game is over, do nothing
        if self.game_over:
            return

        # Check for game end
        # For tests, use the mocked check_winner method
        if hasattr(self.game_state, 'check_winner') and callable(self.game_state.check_winner):
            winner = self.game_state.check_winner()
        else:
            winner = game_logic.check_winner(self.game_state.board)

        if winner:
            self.game_over = True
            self.winner = winner

            # Update board
            winning_line = None
            if winner != game_logic.TIE:
                # For tests, use the mocked get_winning_line method
                if hasattr(self.game_state, 'get_winning_line') and callable(self.game_state.get_winning_line):
                    winning_line = self.game_state.get_winning_line()
                else:
                    winning_line = game_logic.get_winning_line(self.game_state.board)

            # Update board widget
            # Use self.board for tests and self.board_widget for actual app
            board_widget = getattr(self, 'board', None) or getattr(self, 'board_widget', None)
            if board_widget:
                board_widget.update_board(self.game_state.board, winning_line)

            # Update status
            if winner == game_logic.TIE:
                self.status_label.setText("")
            elif winner == game_logic.PLAYER_X:
                self.status_label.setText("")
            elif winner == game_logic.PLAYER_O:
                self.status_label.setText("")
            return

        # If no current turn, set it to X (first move)
        if self.current_turn is None:
            self.current_turn = game_logic.PLAYER_X
            self.status_label.setText("")
            # Use self.board for tests and self.board_widget for actual app
            board_widget = getattr(self, 'board', None) or getattr(self, 'board_widget', None)
            if board_widget:
                board_widget.update_board(self.game_state.board, None)
            return

        # If it's human's turn, update status and wait for input
        if self.current_turn == self.human_player:
            self.status_label.setText(f"Váš tah ({self.human_player})")
            # Use self.board for tests and self.board_widget for actual app
            board_widget = getattr(self, 'board', None) or getattr(self, 'board_widget', None)
            if board_widget:
                board_widget.update_board(self.game_state.board, None)
            return

        # If it's AI's turn, make a move
        if self.current_turn == self.ai_player:
            # Get AI move
            move = self.strategy_selector.get_move(self.game_state.board, self.ai_player)
            if move:
                row, col = move
                # Get coordinates for the move
                coords = self.get_cell_coordinates(row, col)
                if coords:
                    # Draw the symbol
                    if self.draw_ai_symbol(row, col, self.ai_player):
                        # Wait for detection
                        self.current_turn = None
                        self.status_label.setText("")
                    else:
                        self.status_label.setText("")
                else:
                    self.status_label.setText("")
            else:
                self.status_label.setText("")

    def handle_detection_timeout(self):
        """Handle timeout for detection of AI moves"""
        # If not waiting for detection, do nothing
        if self.current_turn is not None:
            return

        # Check if timeout has occurred
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_timeout:
            return

        # Increment retry counter
        self.detection_timeout_counter += 1

        # Update status
        self.status_label.setText(f"Čekám na detekci tahu... (pokus {self.detection_timeout_counter}/{self.max_detection_retries})")

        # If max retries reached, give up and let human play
        if self.detection_timeout_counter >= self.max_detection_retries:
            self.update_game_state(detection_timeout=True)

    def update_game_state(self, detection_timeout=False):
        """Update game state after detection or timeout"""
        # Reset detection timeout counter
        self.detection_timeout_counter = 0

        # If detection timeout occurred, let human play
        if detection_timeout:
            self.current_turn = self.human_player
            self.status_label.setText(f"Detekce tahu selhala. Váš tah ({self.human_player})")
            return

        # Check for game end
        winner = game_logic.check_winner(self.game_state.board)
        if winner:
            self.game_over = True
            self.winner = winner

            # Update status
            if winner == game_logic.TIE:
                self.status_label.setText("")
            elif winner == self.human_player:
                self.status_label.setText("")
            else:
                self.status_label.setText("")
            return

        # Switch turns based on the move counter
        if self.current_turn == self.human_player:
            # Determine if it's AI or Arm turn
            is_arm_turn = hasattr(self, 'move_counter') and self.move_counter % 2 == 0

            # Update turn status
            self.current_turn = self.ai_player

            if is_arm_turn and hasattr(self, 'arm_player_symbol') and self.arm_player_symbol:
                # Arm's turn - use player's symbol with AI strategy
                # Only update status if not already set to prevent flickering
                if not hasattr(self, '_last_status') or self._last_status != "arm_turn":
                    self.update_status(self.tr("arm_turn"))
                    self._last_status = "arm_turn"

                # Pro zpětnou kompatibilitu
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("")

                # Schedule arm move with a longer delay to ensure UI stability
                # CRITICAL: Use ai_player symbol (the one with fewer pieces)
                # But only if game is not over!
                if not self.game_over:
                    arm_symbol = self.ai_player
                    self.logger.info(f"Scheduling arm move after celebration with symbol {arm_symbol}")
                    QTimer.singleShot(1000, lambda: self.make_arm_move(arm_symbol) if not self.game_over else None)
                else:
                    self.logger.info("🛑 Game is over - not scheduling arm move")
            else:
                # AI's turn - use AI symbol
                # Only update status if not already set to prevent flickering
                if not hasattr(self, '_last_status') or self._last_status != "ai_turn":
                    self.update_status(self.tr("ai_turn"))
                    self._last_status = "ai_turn"

                # Pro zpětnou kompatibilitu
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("")
        else:
            # Human's turn
            self.current_turn = self.human_player
            # Only update status if not already set to prevent flickering
            if not hasattr(self, '_last_status') or self._last_status != "your_turn":
                self.update_status(self.tr("your_turn"))
                self._last_status = "your_turn"

            # Pro zpětnou kompatibilitu
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText(f"Váš tah ({self.human_player})")

    def check_game_end(self):
        """Check if the game has ended (win or draw)"""
        # Skip if game is already over or celebration already triggered
        if self.game_over or hasattr(self, '_celebration_triggered'):
            return

        # Get the latest detected board from camera for checking winner
        detected_board = None
        if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
            detected_board = self.camera_thread.last_board_state
            # Convert 1D to 2D if needed
            if isinstance(detected_board, list) and len(detected_board) == 9:
                detected_board = [
                    [detected_board[0], detected_board[1], detected_board[2]],
                    [detected_board[3], detected_board[4], detected_board[5]],
                    [detected_board[6], detected_board[7], detected_board[8]]
                ]

        # If no detected board, use the widget board as fallback
        if not detected_board:
            detected_board = self.board_widget.board

        self.winner = game_logic.check_winner(detected_board)

        if self.winner:
            self.game_over = True

            # Set flag to prevent multiple celebrations
            self._celebration_triggered = True

            # Reset all arm flags when game ends
            self.reset_arm_flags()
            self.logger.info("🏁 Game ended - all arm flags reset")

            # Stop any pending arm moves immediately
            if hasattr(self, 'arm_thread') and self.arm_thread:
                self.arm_thread.stop_current_move()
                self.logger.info("🛑 Stopped any pending arm moves")

            # Vytvoření animovaného oznámení o konci hry
            self.show_game_end_notification()

            if self.winner == game_logic.TIE:
                # Aktualizujeme hlavní stavovou zprávu
                self.update_status("REMÍZA!")
                self.main_status_panel.setStyleSheet("""
                    background-color: #f1c40f;
                    border-radius: 10px;
                    border: 2px solid #f39c12;
                """)

                # Pro zpětnou kompatibilitu
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("")
                    self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #FFCC00;")

            elif self.winner == self.human_player:
                # Aktualizujeme hlavní stavovou zprávu
                self.update_status("VYHRÁLI JSTE!")
                self.main_status_panel.setStyleSheet("""
                    background-color: #2ecc71;
                    border-radius: 10px;
                    border: 2px solid #27ae60;
                """)

                # Pro zpětnou kompatibilitu
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("")
                    self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #55FF55;")
                # Získání výherní čáry pro vykreslení
                self.board_widget.winning_line = game_logic.get_winning_line(
                    self.board_widget.board)
                self.board_widget.update()
            else:
                # Aktualizujeme hlavní stavovou zprávu
                self.update_status("AI VYHRÁLA!")
                self.main_status_panel.setStyleSheet("""
                    background-color: #3498db;
                    border-radius: 10px;
                    border: 2px solid #2980b9;
                """)

                # Pro zpětnou kompatibilitu
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("")
                    self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #55AAFF;")

                # Získání výherní čáry pro vykreslení
                self.board_widget.winning_line = game_logic.get_winning_line(
                    self.board_widget.board)
                self.board_widget.update()

                # WIN CELEBRATION: If the robotic arm wins, draw a celebration line through the winning symbols
                self.logger.info(f"🏆 ARM WINS! Winner: {self.winner}, AI Player: {self.ai_player}")

                # Check if the arm was the one playing (current AI player) and won
                if (hasattr(self, 'ai_player') and self.winner == self.ai_player and
                    hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected):

                    self.logger.info("🎉 ARM WIN CELEBRATION - Drawing winning line!")

                    # Update status to show celebration
                    self.update_status("🏆 RUKA VYHRÁLA! 🎉")

                    # Schedule the winning line drawing after a short delay for dramatic effect
                    QTimer.singleShot(1000, self.draw_winning_line)
                else:
                    # Regular AI win (not arm), still draw the line but without special celebration
                    self.draw_winning_line()

    def show_game_end_notification(self):
        """Zobrazí animované oznámení o konci hry"""
        # Vytvoření widgetu pro oznámení
        notification = QWidget(self)
        notification.setObjectName("game_end_notification")
        notification.setStyleSheet("""
            #game_end_notification {
                background-color: rgba(0, 0, 0, 0.8);
                border-radius: 15px;
                border: 2px solid white;
            }
        """)

        # Layout pro oznámení
        layout = QVBoxLayout(notification)

        # Ikona podle výsledku hry
        if self.winner == game_logic.TIE:
            icon_text = "🤝"
            message = "REMÍZA!"
            color = "#f1c40f"  # Žlutá
        elif self.winner == self.human_player:
            icon_text = "🏆"
            message = "VYHRÁLI JSTE!"
            color = "#2ecc71"  # Zelená
        else:
            # Check if it was the robotic arm that won
            arm_won = (hasattr(self, 'ai_player') and self.winner == self.ai_player and
                       hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected)
            if arm_won:
                icon_text = "🤖"
                message = "RUKA VYHRÁLA!"
                color = "#9b59b6"  # Fialová pro ruku
            else:
                icon_text = "🤖"
                message = "AI VYHRÁLA!"
                color = "#3498db"  # Modrá pro AI

        # Ikona
        icon = QLabel(icon_text)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"""
            font-size: 72px;
            color: {color};
            margin: 10px;
        """)
        layout.addWidget(icon)

        # Text oznámení
        text = QLabel(message)
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet(f"""
            font-size: 36px;
            font-weight: bold;
            color: {color};
            margin: 10px;
        """)
        layout.addWidget(text)

        # Instrukce pro novou hru
        instruction_text = QLabel("Pro novou hru vymažte hrací plochu")
        instruction_text.setAlignment(Qt.AlignCenter)
        instruction_text.setStyleSheet(f"""
            font-size: 14px;
            color: {color};
            margin: 10px;
            font-style: italic;
        """)
        layout.addWidget(instruction_text)

        # Nastavení velikosti a pozice
        notification.setFixedSize(300, 250)
        notification.move(
            (self.width() - notification.width()) // 2,
            (self.height() - notification.height()) // 2
        )

        # Animace pro zobrazení
        self.notification_opacity = QGraphicsOpacityEffect(notification)
        self.notification_opacity.setOpacity(0)
        notification.setGraphicsEffect(self.notification_opacity)

        # Zobrazení widgetu
        notification.show()
        notification.raise_()

        # Animace fade-in
        self.notification_animation = QPropertyAnimation(self.notification_opacity, b"opacity")
        self.notification_animation.setDuration(500)
        self.notification_animation.setStartValue(0)
        self.notification_animation.setEndValue(1)
        self.notification_animation.start()

        # Automatické skrytí po 5 sekundách
        QTimer.singleShot(5000, notification.hide)

    def reset_game(self):
        """Reset the game to initial state"""
        # CRITICAL: Do not reset board_widget.board here!
        # Board should only be updated from YOLO detections in update_board_from_detection
        # Just clear the visual display
        empty_board = game_logic.create_board()
        self.board_widget.update_board(empty_board, None, highlight_changes=False)
        self.board_widget.winning_line = None  # Vymazání výherní čáry
        self.board_widget.update()
        self.human_player = None
        self.ai_player = None
        self.current_turn = None
        self.game_over = False
        self.winner = None
        self.waiting_for_detection = False
        self.ai_move_retry_count = 0
        self.detection_wait_time = 0

        # Reset move counter for tracking even/odd turns
        self.move_counter = 0

        # Reset player symbol for arm moves
        self.arm_player_symbol = None

        # Reset status tracking to prevent flickering
        self._last_status = None
        self._last_periodic_status = None
        self._move_in_progress = False
        self._status_lock = False
        self._current_status = None
        self._last_status_change = 0
        self._status_update_count = 0
        self._current_style = None

        # Reset varování o mřížce
        if hasattr(self, 'grid_warning_active'):
            self.grid_warning_active = False

        # Reset celebration flag to allow new celebrations
        if hasattr(self, '_celebration_triggered'):
            delattr(self, '_celebration_triggered')

        # 🤖 UNIFIED ARM SYSTEM: Reset all arm move flags
        self.arm_move_in_progress = False
        self.arm_move_scheduled = False
        self.waiting_for_detection = False
        self.last_arm_move_time = 0

        # Skrytí varovného panelu, pokud existuje
        if hasattr(self, 'warning_panel') and self.warning_panel.isVisible():
            self.warning_panel.hide()

        # Aktualizujeme hlavní stavovou zprávu
        self.update_status("ZAČNĚTE HRU")
        self.reset_status_panel_style()

        # Pro zpětnou kompatibilitu
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.setText("")
            self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #e0e0e0;")

        # Update debug window if it exists and is visible
        if hasattr(self, 'debug_window') and self.debug_window is not None and hasattr(self.debug_window, 'status_label'):
            self.debug_window.status_label.setText("")

        # OPRAVA: Nebudeme volat handle_detected_game_state po resetu
        # protože to může znovu nastavit výherní čáru
        # Kamera bude pokračovat v detekci automaticky

    def show_debug_window(self):
        """Show the debug window"""
        if not hasattr(self, 'debug_window') or self.debug_window is None:
            # Vytvoříme debug okno, pokud neexistuje
            self.debug_window = DebugWindow(config=self.config, parent=self)

            # Připojíme signály jen při prvním vytvoření
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.fps_updated.connect(self.debug_window.update_fps)
                self.camera_thread.game_state_updated.connect(
                    lambda board: self.debug_window.update_board_state(board))
                # Nastav výchozí kameru v debug window
                if hasattr(self.debug_window, 'camera_combo'):
                    self.debug_window.camera_combo.setCurrentIndex(0)

        # Zobrazení debug okna
        self.debug_window.show()

    def handle_camera_changed(self, camera_index):
        """Handle camera selection change from debug window"""
        try:
            self.logger.info(f"Přepínám na kameru {camera_index}")

            # Validate camera index (only use 0 or 1)
            if camera_index < 0 or camera_index > 1:
                camera_index = 0
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"Kamera {camera_index} není dostupná, používám kameru 0")
                self.logger.info(f"Neplatný index kamery, používám kameru 0")

            # Bezpečné zastavení a uvolnění stávajícího vlákna kamery
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.logger.info("Zastavuji stávající vlákno kamery...")

                # Nejprve odpojíme všechny signály, aby nedocházelo k volání callbacků během uvolňování
                try:
                    if hasattr(self.camera_thread, 'frame_ready'):
                        self.camera_thread.frame_ready.disconnect()
                    if hasattr(self.camera_thread, 'game_state_updated'):
                        self.camera_thread.game_state_updated.disconnect()
                    if hasattr(self.camera_thread, 'fps_updated'):
                        self.camera_thread.fps_updated.disconnect()
                except Exception as e:
                    self.logger.warning(f"Chyba při odpojování signálů: {e}")

                # Zastavíme vlákno a počkáme na jeho ukončení
                try:
                    # Nastavíme running na False, aby vlákno vědělo, že má skončit
                    self.camera_thread.running = False

                    # Explicitně zavoláme cleanup pro uvolnění zdrojů
                    if hasattr(self.camera_thread, 'cleanup'):
                        self.logger.info("Uvolňuji zdroje kamery...")
                        self.camera_thread.cleanup()

                    # Počkáme na ukončení vlákna s timeoutem
                    if self.camera_thread.isRunning():
                        self.logger.info("Čekám na ukončení vlákna...")
                        self.camera_thread.wait(2000)  # Počkáme až 2 sekundy

                        # Pokud vlákno stále běží, zkusíme ho ukončit znovu
                        if self.camera_thread.isRunning():
                            self.logger.warning("Vlákno stále běží, zkouším ukončit znovu...")
                            self.camera_thread.terminate()  # Poslední možnost - násilné ukončení
                            self.camera_thread.wait(1000)  # Počkáme ještě 1 sekundu
                except Exception as e:
                    self.logger.error(f"Chyba při zastavování camera_thread: {e}")

                # Explicitně nastavíme na None, aby GC mohl uvolnit zdroje
                self.camera_thread = None

                # Krátká pauza pro jistotu, že všechny zdroje byly uvolněny
                QThread.msleep(100)

            self.logger.info(f"Vytvářím nové vlákno kamery s indexem {camera_index}...")

            # Create new camera thread with selected camera
            self.camera_thread = CameraThread(camera_index=camera_index)

            # Připojení signálů
            self.camera_thread.frame_ready.connect(self.update_camera_view)
            self.camera_thread.game_state_updated.connect(self.handle_detected_game_state)
            if hasattr(self.camera_thread, 'fps_updated'):
                self.camera_thread.fps_updated.connect(self.update_fps_display)

            # Start new camera thread
            self.logger.info("Spouštím nové vlákno kamery...")
            self.camera_thread.start()

            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Kamera {camera_index} aktivována")

            self.logger.info(f"Kamera {camera_index} úspěšně aktivována")

        except Exception as e:
            self.logger.error(f"Chyba při přepínání kamery: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Chyba při přepínání kamery: {e}")


    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # V testech přeskočíme pohyb ruky
        try:
            if hasattr(self, 'status_label') and hasattr(self.status_label, 'setText'):
                # Nejprve přesuneme ruku do neutrálního stavu
                self.status_label.setText("")
                if hasattr(self, 'move_to_neutral_position') and callable(self.move_to_neutral_position):
                    self.move_to_neutral_position()
                QApplication.processEvents()

                # Pak parkujeme ruku
                self.status_label.setText("")
                if hasattr(self, 'arm_thread') and self.arm_thread and hasattr(self.arm_thread, 'connected') and self.arm_thread.connected:
                    # Použití arm_thread pro parkování
                    self.arm_thread.go_to_position(x=PARK_X, y=PARK_Y, wait=True)
                elif hasattr(self, 'arm_controller') and self.arm_controller and hasattr(self.arm_controller, 'connected') and self.arm_controller.connected:
                    # Záložní použití arm_controller
                    self.arm_controller.park(x=PARK_X, y=PARK_Y)
                # Give it a moment to complete the parking
                QApplication.processEvents()
        except Exception as e:
            print(f"Chyba při parkování ruky: {e}")

        # Stop camera thread
        if hasattr(self, 'camera_thread') and self.camera_thread:
            self.camera_thread.stop()

        # Stop arm thread
        if hasattr(self, 'arm_thread') and self.arm_thread:
            if hasattr(self.arm_thread, 'disconnect') and callable(self.arm_thread.disconnect):
                self.arm_thread.disconnect()
            self.arm_thread.stop()

        # Disconnect arm controller (záložní)
        if hasattr(self, 'arm_controller') and self.arm_controller:
            if hasattr(self.arm_controller, 'disconnect') and callable(self.arm_controller.disconnect):
                self.arm_controller.disconnect()

        # Close debug window
        if hasattr(self, 'debug_window') and self.debug_window:
            self.debug_window.close()

        event.accept()

    def _validate_game_state(self, board):
        """Validate that the game state follows proper tic-tac-toe rules.

        Args:
            board: 2D list representing the current board state

        Returns:
            bool: True if the game state is valid, False otherwise
        """
        # Count X and O symbols
        x_count = sum(row.count(game_logic.PLAYER_X) for row in board)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in board)

        # RULE 1: X always goes first, so X count should be equal to O count or one more
        # Valid states: X=O (even total, O's turn) or X=O+1 (odd total, X's turn)
        if abs(x_count - o_count) > 1:
            self.logger.warning(f"❌ Invalid game state: X={x_count}, O={o_count} (difference > 1)")
            return False

        # RULE 2: X should never have fewer symbols than O (since X goes first)
        if x_count < o_count:
            self.logger.warning(f"❌ Invalid game state: X={x_count} < O={o_count} (X should go first)")
            return False

        # RULE 3: Check if there's already a winner but game continues
        winner = game_logic.check_winner(board)
        if winner and winner != game_logic.TIE:
            # If there's a winner, the game should have stopped
            # But we allow this for detection purposes
            self.logger.info(f"ℹ️ Game has winner {winner} but continuing for detection")

        self.logger.info(f"✅ Valid game state: X={x_count}, O={o_count}")
        return True

    def _check_real_time_win_detection(self, detected_board):
        """🏆 Real-time win detection in every camera frame"""
        if not detected_board or self.game_over:
            return

        # Check for winner using game logic
        winner = game_logic.check_winner(detected_board)

        if winner and winner != game_logic.TIE and not self.game_over:
            self.logger.info(f"🏆 REAL-TIME WIN DETECTED: {winner} wins!")

            # Set game over state immediately to prevent multiple celebrations
            self.game_over = True
            self.winner = winner

            # Set flag to prevent multiple celebrations
            if not hasattr(self, '_celebration_triggered'):
                self._celebration_triggered = True

                # Get winning line for celebration
                winning_line = game_logic.get_winning_line(detected_board)
                if winning_line:
                    self.board_widget.winning_line = winning_line
                    self.board_widget.update()

                # Update board to detected state
                self.board_widget.board = [row[:] for row in detected_board]
                self.board_widget.update()

                # Determine who won and trigger appropriate celebration
                self._trigger_win_celebration(winner, winning_line)

        elif winner == game_logic.TIE and not self.game_over:
            self.logger.info("🤝 REAL-TIME DRAW DETECTED!")

            # Set game over state immediately to prevent multiple celebrations
            self.game_over = True
            self.winner = winner

            # Set flag to prevent multiple celebrations
            if not hasattr(self, '_celebration_triggered'):
                self._celebration_triggered = True

                # Update board
                self.board_widget.board = [row[:] for row in detected_board]
                self.board_widget.update()

                # Show draw notification
                self._trigger_draw_celebration()

    def _trigger_win_celebration(self, winner, winning_line):
        """🎉 Trigger appropriate win celebration based on who won"""
        # Determine if the robotic arm won
        arm_won = (hasattr(self, 'ai_player') and winner == self.ai_player and
                   hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected)

        if arm_won:
            # 🤖 ROBOTIC ARM WIN CELEBRATION
            self.logger.info("🎉 ROBOTIC ARM WIN CELEBRATION!")

            # Update status with special arm win message
            self.update_status("🏆 RUKA VYHRÁLA! 🎉")
            self.main_status_panel.setStyleSheet("""
                background-color: #9b59b6;
                border-radius: 10px;
                border: 2px solid #8e44ad;
                animation: pulse 1s infinite;
            """)

            # Show animated notification
            self.show_game_end_notification()

            # Draw celebration line immediately (no delay)
            if winning_line:
                self.logger.info("🎨 Drawing celebration line through winning symbols!")
                QTimer.singleShot(100, self.draw_winning_line)  # Very short delay for dramatic effect

            # Park arm after celebration
            QTimer.singleShot(3000, self._park_arm_after_win)

        else:
            # 👤 HUMAN PLAYER WIN CELEBRATION
            self.logger.info("🎉 HUMAN PLAYER WIN CELEBRATION!")

            # Update status with human win message
            self.update_status("🏆 VYHRÁLI JSTE! 🎉")
            self.main_status_panel.setStyleSheet("""
                background-color: #2ecc71;
                border-radius: 10px;
                border: 2px solid #27ae60;
                animation: pulse 1s infinite;
            """)

            # Show animated notification
            self.show_game_end_notification()

            # No physical line drawing for human wins (they can't control the arm)
            self.logger.info("ℹ️ Human win - no physical line drawing")

    def _trigger_draw_celebration(self):
        """🤝 Trigger draw celebration"""
        self.logger.info("🤝 DRAW CELEBRATION!")

        # Update status with draw message
        self.update_status("🤝 REMÍZA!")
        self.main_status_panel.setStyleSheet("""
            background-color: #f1c40f;
            border-radius: 10px;
            border: 2px solid #f39c12;
        """)

        # Show animated notification
        self.show_game_end_notification()

    def _park_arm_after_win(self):
        """Park the robotic arm after a win"""
        try:
            if hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected:
                self.logger.info("🅿️ Parking arm after win celebration")
                # Move to neutral position from calibration
                if hasattr(self, 'calibration_data') and self.calibration_data:
                    neutral_x = self.calibration_data.get('neutral_x', 200)
                    neutral_y = self.calibration_data.get('neutral_y', 0)
                    self.arm_thread.go_to_position(x=neutral_x, y=neutral_y, wait=False)
                else:
                    # Default neutral position
                    self.arm_thread.go_to_position(x=200, y=0, wait=False)
        except Exception as e:
            self.logger.error(f"Error parking arm after win: {e}")

    def _should_arm_play_now(self, detected_board):
        """🤖 UNIFIED ARM LOGIC: Determine if arm should play right now"""
        import time

        # Basic checks
        if not detected_board or self.game_over:
            return False, None

        # ✅ CRITICAL FIX: Only check if arm is physically moving or scheduled
        if self.arm_move_in_progress or self.arm_move_scheduled:
            self.logger.debug(f"🤖 Arm busy: in_progress={self.arm_move_in_progress}, scheduled={self.arm_move_scheduled}")
            return False, None

        # ✅ CRITICAL FIX: Reduced cooldown and better logic
        current_time = time.time()
        if current_time - self.last_arm_move_time < 3.0:  # Increased back to 3.0 seconds for safety
            remaining = 3.0 - (current_time - self.last_arm_move_time)
            self.logger.debug(f"🤖 Arm move cooldown active: {remaining:.1f}s remaining")
            return False, None

        # Check if waiting for detection
        if self.waiting_for_detection:
            self.logger.debug("🤖 Still waiting for detection of previous move")
            return False, None

        # Check grid validity
        valid_grid = False
        if hasattr(self, 'camera_thread') and self.camera_thread:
            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                if hasattr(self.camera_thread.detection_thread, 'detector') and self.camera_thread.detection_thread.detector:
                    game_state = self.camera_thread.detection_thread.detector.game_state
                    if game_state and hasattr(game_state, 'is_physical_grid_valid'):
                        valid_grid = game_state.is_physical_grid_valid()

        if not valid_grid:
            self.logger.debug("🤖 Grid not valid for arm move")
            return False, None

        # Analyze symbol counts
        x_count = sum(row.count(game_logic.PLAYER_X) for row in detected_board)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in detected_board)
        total_symbols = x_count + o_count

        # Arm should play when there's an odd number of symbols (1,3,5,7,9)
        if total_symbols % 2 != 1:
            self.logger.debug(f"🤖 Even symbol count ({total_symbols}) - arm should not play")
            return False, None

        # Determine which symbol arm should play (less frequent one)
        if x_count < o_count:
            arm_symbol = game_logic.PLAYER_X
        elif o_count < x_count:
            arm_symbol = game_logic.PLAYER_O
        else:
            arm_symbol = game_logic.PLAYER_X  # Default to X if equal

        self.logger.info(f"🤖 ARM SHOULD PLAY: {arm_symbol} (X={x_count}, O={o_count}, total={total_symbols})")
        return True, arm_symbol

    def _execute_arm_move(self, arm_symbol):
        """🤖 UNIFIED ARM EXECUTION: Execute arm move with proper state management"""
        # DEPRECATED: This is now just a wrapper
        # All logic moved to make_arm_move_with_symbol
        return self.make_arm_move_with_symbol(arm_symbol)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TicTacToeApp()
    window.show()
    sys.exit(app.exec_())
