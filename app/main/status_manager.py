# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Status Manager module for TicTacToe application.
This module handles status updates, language management, and UI state.
Refactored from pyqt_gui.py to separate concerns.
"""

# import logging  # unused
import time

# pylint: disable=no-name-in-module
from PyQt5.QtCore import QObject
# pylint: disable=no-name-in-module
from PyQt5.QtCore import Qt
# pylint: disable=no-name-in-module
from PyQt5.QtCore import QTimer
# pylint: disable=no-name-in-module
from PyQt5.QtCore import pyqtSignal
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import QLabel
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import QVBoxLayout
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import QWidget

from app.main.game_utils import convert_board_1d_to_2d
from app.main.game_utils import setup_logger

# from app.main.gui_factory import LabelFactory, LayoutFactory  # unused

# Language dictionaries
LANG_CS = {
    "your_turn": "VÁŠ TAH", "ai_turn": "TAH AI", "arm_turn": "TAH RUKY",
    "arm_moving": "ROBOT HRAJE", "place_symbol": "POLOŽTE SYMBOL",
    "waiting_detection": "ČEKÁM NA DETEKCI", "win": "VÝHRA", "draw": "REMÍZA",
    "new_game": "Nová hra", "reset": "Reset", "debug": "Debug", "camera": "Kamera",
    "difficulty": "Obtížnost", "arm_connect": "Připojit ruku",
    "arm_disconnect": "Odpojit ruku", "game_over": "KONEC HRY",
    "grid_not_visible": "⚠️ MŘÍŽKA NENÍ VIDITELNÁ!", "grid_visible": "✅ MŘÍŽKA VIDITELNÁ",
    "move_to_neutral": "PŘESUN DO NEUTRÁLNÍ POZICE", "new_game_detected": "NOVÁ HRA DETEKOVÁNA",
    "move_success": "Ruka v neutrální pozici", "move_failed": "Nepodařilo se přesunout ruku",
    "waiting_for_symbol": "⏳ Čekám na detekci symbolu {}...",
    "detection_attempt": "Čekám na detekci tahu... (pokus {}/{})", "language": "Jazyk",
    "debug_tooltip": "Otevřít ladící okno", "loss": "PROHRA",
    "game_instructions": "Klikněte na políčko pro umístění vašeho symbolu "
                         "nebo použijte fyzickou herní desku",
    "grid_incomplete_notification": "Umístěte celou hrací plochu do záběru kamery "
    "tak, aby byly viditelné všechny průsečíky mřížky.",
    "grid_incomplete_title": "⚠️ NEÚPLNÁ DETEKCE HRACÍ PLOCHY",
    "new_game_instruction": "Pro novou hru vymažte hrací plochu nebo stiskněte Reset.",
    "arm_connected": "✅ RUKA PŘIPOJENA", "arm_disconnected": "❌ RUKA ODPOJENA",
    "arm_connection_notification": "Robotická ruka není připojena. "
    "Hra bude v režimu pouze s kamerou.",
    "arm_disconnection_title": "⚠️ RUKA ODPOJENA",
    "cached_symbols": "🔄 POUŽÍVÁM CACHE SYMBOLŮ",
    "live_detection": "📹 ŽIVÁ DETEKCE"
}

LANG_EN = {
    "your_turn": "YOUR TURN", "ai_turn": "AI TURN", "arm_turn": "ARM TURN",
    "arm_moving": "ROBOT PLAYING", "place_symbol": "PLACE SYMBOL",
    "waiting_detection": "WAITING FOR DETECTION", "win": "WIN", "draw": "DRAW",
    "new_game": "New Game", "reset": "Reset", "debug": "Debug", "camera": "Camera",
    "difficulty": "Difficulty", "arm_connect": "Connect arm",
    "arm_disconnect": "Disconnect arm", "game_over": "GAME OVER",
    "grid_not_visible": "⚠️ GRID NOT VISIBLE!", "grid_visible": "✅ GRID VISIBLE",
    "move_to_neutral": "MOVING TO NEUTRAL POSITION", "new_game_detected": "NEW GAME DETECTED",
    "move_success": "Arm in neutral position", "move_failed": "Failed to move arm",
    "waiting_for_symbol": "⏳ Waiting for symbol {} detection...",
    "detection_attempt": "Waiting for symbol detection... (attempt {}/{})", "language": "Language",
    "debug_tooltip": "Open debug window", "loss": "LOSS",
    "game_instructions": "Click on a cell to place your symbol "
                         "or use the physical game board",
    "grid_incomplete_notification": "Please position the entire game board within "
                                   "the camera view so all grid points are visible.",
    "grid_incomplete_title": "⚠️ INCOMPLETE GAME BOARD DETECTION",
    "new_game_instruction": "To start a new game, clear the game board or press Reset.",
    "arm_connected": "✅ ARM CONNECTED", "arm_disconnected": "❌ ARM DISCONNECTED",
    "arm_connection_notification": "Robotic arm is not connected. "
    "Game will run in camera-only mode.",
    "arm_disconnection_title": "⚠️ ARM DISCONNECTED",
    "cached_symbols": "🔄 USING CACHED SYMBOLS",
    "live_detection": "📹 LIVE DETECTION"
}


# pylint: disable=too-many-instance-attributes
class StatusManager(QObject):
    """Manages status updates, language, and UI state."""

    # Signals
    language_changed = pyqtSignal(str)  # language_code
    status_updated = pyqtSignal(str, bool)  # message, is_key

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.logger = setup_logger(__name__)

        # Language state
        self.current_language = LANG_CS
        self.is_czech = True

        # Status state
        self._current_style_key = None
        self._status_lock_time = 0
        self._current_status_text = ""

        # UI components
        self.main_status_panel = None
        self.main_status_message = None

    def tr(self, key):
        """Translate key to current language."""
        return self.current_language.get(key, key)

    def toggle_language(self):
        """Toggle between Czech and English."""
        if self.is_czech:
            self.current_language = LANG_EN
            self.is_czech = False
            language_code = "en"
        else:
            self.current_language = LANG_CS
            self.is_czech = True
            language_code = "cs"

        self.language_changed.emit(language_code)
        self.logger.info("Language changed to {language_code}")

    def create_status_panel(self):
        """Create the main status panel."""
        self.main_status_panel = QWidget()
        self.main_status_panel.setStyleSheet(
            "background-color: #333740; border-radius: 10px; padding: 10px; margin-bottom: 10px;"
        )

        status_layout = QVBoxLayout(self.main_status_panel)
        self.main_status_message = QLabel("START")
        self.main_status_message.setStyleSheet(
            "color: #FFFFFF; font-size: 28px; font-weight: bold; padding: 12px;"
        )
        self.main_status_message.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.main_status_message)

        return self.main_status_panel

    def update_status(self, message_key_or_text, is_key=True):
        """Update the main status display."""
        message = self.tr(message_key_or_text) if is_key else message_key_or_text

        current_time = time.time()

        # Simple status locking - update if message changes
        if message == self._current_status_text and current_time - self._status_lock_time < 1.0:
            return

        self._current_status_text = message
        self._status_lock_time = current_time

        if self.main_status_message:
            # Get board info for status context
            board_for_status = None
            if hasattr(self.main_window, 'camera_controller'):
                board_for_status = self.main_window.camera_controller.get_current_board_state()
            elif hasattr(self.main_window, 'board_widget'):
                board_for_status = self.main_window.board_widget.board

            _ = self._get_board_symbol_counts(board_for_status)  # unused variables

            status_text_to_show = message.upper()
            # style_key = "default"  # unused

            # Set style based on message type
            if message_key_or_text == "your_turn":
                self.set_status_style_safe("player", self._get_status_style("player"))
            elif message_key_or_text in ("arm_turn", "arm_moving"):
                self.set_status_style_safe("ai", self._get_status_style("ai"))
            elif message_key_or_text == "win":
                self.set_status_style_safe("win", self._get_status_style("win"))
                # Show win notification popup
                QTimer.singleShot(1000, lambda: self.show_game_end_notification("HUMAN_WIN"))
            elif message_key_or_text == "loss":
                self.set_status_style_safe("loss", self._get_status_style("loss"))
                # Show loss notification popup
                QTimer.singleShot(1000, lambda: self.show_game_end_notification("ARM_WIN"))
            elif message_key_or_text == "draw":
                self.set_status_style_safe("draw", self._get_status_style("draw"))
                # Show draw notification popup
                # pylint: disable=import-outside-toplevel
                from app.main import game_logic
                QTimer.singleShot(1000, lambda: self.show_game_end_notification(game_logic.TIE))
            elif message_key_or_text == "new_game_detected":
                self.set_status_style_safe("new_game", self._get_status_style("new_game"))
            elif message_key_or_text == "grid_not_visible":
                self.set_status_style_safe("error", self._get_status_style("error"))
            elif message_key_or_text == "grid_visible":
                self.set_status_style_safe("success", self._get_status_style("success"))
                QTimer.singleShot(2000, self.reset_status_panel_style)
            elif message_key_or_text == "cached_symbols":
                self.set_status_style_safe("warning", self._get_status_style("warning"))
            elif message_key_or_text == "live_detection":
                self.set_status_style_safe("success", self._get_status_style("success"))

            self.main_status_message.setText(status_text_to_show)

        # Emit signal for other components
        self.status_updated.emit(message, is_key)

        self.logger.debug("Status updated: {message}")

    def set_status_style_safe(self, style_key, style_css):
        """Safely set status panel style."""
        if self._current_style_key != style_key:
            self._current_style_key = style_key
            if self.main_status_panel:
                self.main_status_panel.setStyleSheet(style_css)

    def reset_status_panel_style(self):
        """Reset status panel to default style."""
        self.set_status_style_safe("default", self._get_status_style("default"))

    def _get_status_style(self, style_type):
        """Get CSS style for status panel."""
        base_style = "border-radius: 10px; padding: 10px; margin-bottom: 10px;"

        styles = {
            "default": f"background-color: #333740; {base_style}",
            "player": f"background-color: #2980b9; {base_style}",  # Blue for player
            "ai": f"background-color: #e74c3c; {base_style}",      # Red for AI
            "win": f"background-color: #27ae60; {base_style}",     # Green for win
            "loss": f"background-color: #e74c3c; {base_style}",    # Red for loss
            "draw": f"background-color: #f39c12; {base_style}",    # Orange for draw
            "new_game": f"background-color: #9b59b6; {base_style}",  # Purple for new game
            "error": f"background-color: #e74c3c; {base_style}",   # Red for error
            "success": f"background-color: #27ae60; {base_style}",  # Green for success
            "warning": f"background-color: #f39c12; {base_style}",  # Orange for warning
        }

        return styles.get(style_type, styles["default"])

    def _get_board_symbol_counts(self, board):
        """Get symbol counts from board."""
        if board is None:
            return 0, 0, 0

        board_2d = convert_board_1d_to_2d(board)
        if not isinstance(board_2d, list) or not all(isinstance(row, list) for row in board_2d):
            return 0, 0, 0

        # Import game_logic here to avoid circular imports
        # pylint: disable=import-outside-toplevel
        from app.main import game_logic
        x_count = sum(row.count(game_logic.PLAYER_X) for row in board_2d)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in board_2d)
        return x_count, o_count, x_count + o_count

    # pylint: disable=too-many-statements
    def show_game_end_notification(self, winner):
        """Show game end notification."""
        # pylint: disable=import-outside-toplevel
        from PyQt5.QtCore import QPropertyAnimation
        # pylint: disable=import-outside-toplevel
        from PyQt5.QtWidgets import QGraphicsOpacityEffect

        # Prevent multiple notifications
        if hasattr(self.main_window, '_celebration_triggered'):
            return
        self.main_window._celebration_triggered = True

        notification_widget = QWidget(self.main_window)
        notification_widget.setObjectName("game_end_notification")
        notification_widget.setStyleSheet("""
            QWidget#game_end_notification {
                background-color: rgba(45, 45, 48, 0.95);
                border-radius: 15px;
                border: 2px solid #0078D7;
            }
        """)

        layout = QVBoxLayout(notification_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Determine message and color based on winner
        # pylint: disable=import-outside-toplevel
        from app.main import game_logic
        print(f"DEBUG: show_game_end_notification called with winner='{winner}'")  # Debug

        icon_text, message_text, color = "", "", ""
        if winner == game_logic.TIE:
            icon_text, message_text, color = "🤝", self.tr("draw"), "#f1c40"
        elif winner == "HUMAN_WIN":
            icon_text, message_text, color = "🏆", self.tr("win"), "#2ecc71"
            print("DEBUG: Setting win text for HUMAN_WIN")  # Debug
        elif winner == "ARM_WIN":
            icon_text, message_text, color = "🤖", self.tr("loss"), "#e74c3c"
            print("DEBUG: Setting loss text for ARM_WIN")  # Debug
        else:
            icon_text, message_text, color = "🏁", self.tr("game_over"), "#95a5a6"
            print(f"DEBUG: Unknown winner '{winner}', using game_over")  # Debug

        # Create notification content
        icon_label = QLabel(icon_text)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 60px; color: {color};")
        layout.addWidget(icon_label)

        message_label = QLabel(message_text.upper())
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")
        layout.addWidget(message_label)

        instruction_label = QLabel(self.tr("new_game_instruction"))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("font-size: 12px; color: #bdc3c7; margin-top: 10px;")
        layout.addWidget(instruction_label)

        # Position notification
        notification_widget.resize(300, 200)
        notification_widget.move(
            (self.main_window.width() - notification_widget.width()) // 2,
            (self.main_window.height() - notification_widget.height()) // 2
        )

        # Add fade-in animation
        opacity_effect = QGraphicsOpacityEffect(notification_widget)
        notification_widget.setGraphicsEffect(opacity_effect)

        notification_widget.show()
        notification_widget.raise_()

        anim = QPropertyAnimation(opacity_effect, b"opacity")
        anim.setDuration(500)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

        # Store reference and auto-hide after 4 seconds
        self.main_window._active_notification = notification_widget
        QTimer.singleShot(4000, self._hide_notification)

    def _hide_notification(self):
        """Hide the active notification."""
        if (hasattr(self.main_window, '_active_notification') and
                self.main_window._active_notification):
            self.main_window._active_notification.hide()

    def show_grid_incomplete_notification(self):
        """Show grid incomplete notification."""
        # pylint: disable=import-outside-toplevel
        from PyQt5.QtCore import QPropertyAnimation
        # pylint: disable=import-outside-toplevel
        from PyQt5.QtWidgets import QGraphicsOpacityEffect

        # Hide any existing notification first
        self._hide_notification()

        notification_widget = QWidget(self.main_window)
        notification_widget.setObjectName("grid_incomplete_notification")
        notification_widget.setStyleSheet("""
            QWidget#grid_incomplete_notification {
                background-color: rgba(231, 76, 60, 0.95);
                border-radius: 15px;
                border: 3px solid #c0392b;
            }
        """)

        layout = QVBoxLayout(notification_widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Warning icon
        icon_label = QLabel("⚠️")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 50px; color: #ffffff;")
        layout.addWidget(icon_label)

        # Title
        title_label = QLabel(self.tr("grid_incomplete_title"))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title_label)

        # Instruction message
        message_label = QLabel(self.tr("grid_incomplete_notification"))
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("font-size: 14px; color: #ffffff; line-height: 1.4;")
        layout.addWidget(message_label)

        # Position notification at top center
        notification_widget.resize(400, 180)
        notification_widget.move(
            (self.main_window.width() - notification_widget.width()) // 2,
            50  # Position near top of screen
        )

        # Add fade-in animation
        opacity_effect = QGraphicsOpacityEffect(notification_widget)
        notification_widget.setGraphicsEffect(opacity_effect)

        notification_widget.show()
        notification_widget.raise_()

        anim = QPropertyAnimation(opacity_effect, b"opacity")
        anim.setDuration(300)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

        # Store reference - this notification stays until grid is complete
        self.main_window._active_notification = notification_widget

    def hide_grid_incomplete_notification(self):
        """Hide grid incomplete notification."""
        self._hide_notification()

    def show_arm_disconnected_notification(self):
        """Show arm disconnected notification."""
        # Hide any existing notification first
        self._hide_notification()

        notification_widget = QWidget(self.main_window)
        notification_widget.setParent(self.main_window)
        notification_widget.setObjectName("arm_disconnected_notification")
        notification_widget.setStyleSheet("""
            QWidget#arm_disconnected_notification {
                background-color: rgba(243, 156, 18, 0.95);
                border-radius: 15px;
                border: 3px solid #e67e22;
            }
        """)

        layout = QVBoxLayout(notification_widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # Warning icon
        icon_label = QLabel("🤖❌")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 45px; color: #ffffff;")
        layout.addWidget(icon_label)

        # Title
        title_label = QLabel(self.tr("arm_disconnection_title"))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title_label)

        # Instruction message
        message_label = QLabel(self.tr("arm_connection_notification"))
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("font-size: 13px; color: #ffffff; line-height: 1.4;")
        layout.addWidget(message_label)

        # Position notification at top center
        notification_widget.resize(450, 180)
        notification_widget.move(
            (self.main_window.width() - notification_widget.width()) // 2,
            50  # Position near top of screen
        )

        # Show notification directly - no animation
        notification_widget.show()
        notification_widget.raise_()
        notification_widget.activateWindow()

        # Store reference and auto-hide after 8 seconds for arm notification
        self.main_window._active_notification = notification_widget
        QTimer.singleShot(8000, self._hide_notification)

    def hide_arm_disconnected_notification(self):
        """Hide arm disconnected notification."""
        self._hide_notification()

    def get_current_language_code(self):
        """Get current language code."""
        return "cs" if self.is_czech else "en"

    def set_language(self, language_code):
        """Set language by code."""
        if language_code == "cs" and not self.is_czech:
            self.toggle_language()
        elif language_code == "en" and self.is_czech:
            self.toggle_language()
