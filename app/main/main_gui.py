"""
Main GUI module for TicTacToe application.
This module contains the main window setup, layout, and basic UI components.
Refactored from pyqt_gui.py to separate concerns.
"""

import sys
import os
import logging
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
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon

# Add project root to path if not already there
from app.main.path_utils import setup_project_path
setup_project_path()

# Import refactored modules
from app.main.game_controller import GameController
from app.main.camera_controller import CameraController
from app.main.arm_movement_controller import ArmMovementController
from app.main.ui_event_handlers import UIEventHandlers
from app.main.status_manager import StatusManager
from app.main.board_widget import TicTacToeBoard
from app.main.game_statistics import GameStatisticsWidget
from app.main.constants import DEFAULT_CAMERA_INDEX, DEFAULT_DIFFICULTY
from app.main.game_utils import setup_logger
from app.core.config import AppConfig


class TicTacToeApp(QMainWindow):
    """Main TicTacToe application window with refactored architecture."""

    def __init__(self, config=None):
        super().__init__()

        self.logger = setup_logger(__name__)

        # Configuration
        self.config = config if config is not None else AppConfig()

        # Initialize controllers
        self.status_manager = StatusManager(self)
        self.game_controller = GameController(self, self.config)
        self.camera_controller = CameraController(self, DEFAULT_CAMERA_INDEX)
        self.arm_controller = ArmMovementController(self, self.config)
        self.event_handlers = UIEventHandlers(self)

        # Set window properties
        self._setup_window()

        # Initialize UI
        self.init_ui()

        # Connect controllers
        self._connect_controllers()

        # Start application
        self._start_application()

    def _setup_window(self):
        """Setup main window properties."""
        title = "Piškvorky"
        if hasattr(self.config, 'game') and hasattr(self.config.game, 'gui_window_title'):
            title = self.config.game.gui_window_title
        self.setWindowTitle(title)

        # Set window icon
        try:
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "resources", "app_icon.png"
            )
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                self.logger.warning("Icon file not found: {icon_path}")
        except Exception as e:
            self.logger.error("Error setting icon: {e}")

        # Show fullscreen if not in test mode
        if 'pytest' not in sys.modules:
            self.showFullScreen()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Apply stylesheet
        self._apply_stylesheet()

        # Create main layout with better spacing for 3x3 grid
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Status panel
        self.main_status_panel = self.status_manager.create_status_panel()
        main_layout.addWidget(self.main_status_panel)

        # Board and statistics container
        board_stats_container = self._create_board_with_statistics_container()
        main_layout.addWidget(board_stats_container, 2)  # Give more stretch to board area

        # Controls panel
        controls_panel = self._create_controls_panel()
        main_layout.addWidget(controls_panel)

    def _apply_stylesheet(self):
        """Apply the main application stylesheet."""
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
            QPushButton:hover { background-color: #1084E3; }
            QPushButton:pressed { background-color: #0067B8; }
            QLabel { color: #E0E0E0; }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #3D3D3D;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

    def _create_board_with_statistics_container(self):
        """Create container with game board and statistics side by side."""
        container = QWidget()
        container.setStyleSheet(
            "background-color: #333740; border-radius: 15px; padding: 20px;"
        )

        layout = QHBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(30)

        # Left side: Game board
        board_container = self._create_board_container()
        layout.addWidget(board_container, 1)

        # Right side: Statistics
        self.statistics_widget = GameStatisticsWidget(self.status_manager)
        self.statistics_widget.setFixedWidth(280)
        layout.addWidget(self.statistics_widget, 0)

        return container

    def _create_board_container(self):
        """Create the game board container with proper 3x3 grid layout."""
        board_container = QWidget()
        board_layout = QHBoxLayout(board_container)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addStretch(1)

        self.board_widget = TicTacToeBoard()
        # Removed cell_clicked connection - no manual interaction allowed

        # Set proper size for 3x3 grid - ensure it's fully visible
        board_size = 480  # Increased size for better visibility
        self.board_widget.setFixedSize(board_size, board_size)
        self.board_widget.cell_size = board_size // 3  # 160px per cell

        # Center the board
        board_layout.addWidget(self.board_widget, 0, Qt.AlignCenter)
        board_layout.addStretch(1)

        return board_container

    def _create_controls_panel(self):
        """Create the controls panel."""
        controls_panel = QWidget()
        controls_panel.setStyleSheet(
            "background-color: #333740; border-radius: 10px; padding: 15px;"
        )
        controls_layout = QVBoxLayout(controls_panel)

        # Difficulty slider
        difficulty_container = self._create_difficulty_slider()
        controls_layout.addWidget(difficulty_container)

        # Buttons
        button_container = self._create_buttons()
        controls_layout.addWidget(button_container)

        return controls_panel

    def _create_difficulty_slider(self):
        """Create difficulty slider container."""
        difficulty_container = QWidget()
        difficulty_layout = QVBoxLayout(difficulty_container)

        # Label and value display
        header_layout = QHBoxLayout()
        self.difficulty_label = QLabel(self.status_manager.tr("difficulty"))
        self.difficulty_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")

        self.difficulty_value_label = QLabel(str(DEFAULT_DIFFICULTY))
        self.difficulty_value_label.setStyleSheet("""
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
            font-size: 16px;
        """)

        header_layout.addWidget(self.difficulty_label)
        header_layout.addStretch()
        header_layout.addWidget(self.difficulty_value_label)
        difficulty_layout.addLayout(header_layout)

        # Slider
        self.difficulty_slider = QSlider(Qt.Horizontal)
        self.difficulty_slider.setRange(1, 10)
        self.difficulty_slider.setValue(DEFAULT_DIFFICULTY)
        self.difficulty_slider.valueChanged.connect(self._on_difficulty_changed)
        difficulty_layout.addWidget(self.difficulty_slider)

        return difficulty_container

    def _on_difficulty_changed(self, value):
        """Handle difficulty slider change and update value display."""
        self.difficulty_value_label.setText(str(value))
        self.event_handlers.handle_difficulty_changed(value)

    def _create_buttons(self):
        """Create button container."""
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)

        self.reset_button = QPushButton(self.status_manager.tr("new_game"))
        self.reset_button.clicked.connect(self.event_handlers.handle_reset_button_click)
        self.reset_button.setStyleSheet("background-color: #27ae60;")

        self.language_button = QPushButton("🇨🇿")
        self.language_button.clicked.connect(self.event_handlers.change_language)
        self.language_button.setFixedSize(40, 40)

        self.debug_button = QPushButton("🔧")
        self.debug_button.clicked.connect(self.event_handlers.handle_debug_button_click)
        self.debug_button.setFixedSize(50, 50)
        self.debug_button.setToolTip(self.status_manager.tr("debug_tooltip"))
        self.debug_button.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                border: 2px solid #2c3e50;
                border-radius: 25px;
                font-size: 18px;
                color: #ecf0f1;
            }
            QPushButton:hover {
                background-color: #3498db;
                border-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """)

        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.language_button)
        button_layout.addWidget(self.debug_button)
        button_layout.addStretch()

        return button_container

    def _connect_controllers(self):
        """Connect signals between controllers."""
        # Connect camera to game controller
        self.camera_controller.game_state_updated.connect(
            self.game_controller.handle_detected_game_state
        )

        # Connect game controller to status manager
        self.game_controller.status_changed.connect(
            self.status_manager.update_status
        )

        # Connect camera grid incomplete signal to status manager
        self.camera_controller.grid_incomplete.connect(
            self._handle_grid_incomplete
        )

        # Connect arm connection signal to status manager
        self.arm_controller.arm_connected.connect(
            self._handle_arm_connection_changed
        )

        # Force emit current arm connection status after GUI is connected
        current_arm_status = self.arm_controller.is_arm_available()
        self.logger.info("Forcing arm connection status emit: {current_arm_status}")
        self._handle_arm_connection_changed(current_arm_status)

        # Connect arm controller to game controller
        self.game_controller.set_arm_controller(self.arm_controller)

        # Connect game ended signal to statistics
        self.game_controller.game_ended.connect(
            self._handle_game_ended
        )

        # Connect statistics reset to show confirmation
        self.statistics_widget.reset_requested.connect(
            self._handle_statistics_reset
        )

    def _handle_grid_incomplete(self, is_incomplete):
        """Handle grid incomplete signal from camera controller."""
        if is_incomplete:
            # Show grid incomplete notification
            self.status_manager.show_grid_incomplete_notification()
        else:
            # Hide grid incomplete notification
            self.status_manager.hide_grid_incomplete_notification()

    def _handle_arm_connection_changed(self, is_connected):
        """Handle arm connection status change."""
        if is_connected:
            self.logger.info("Arm connected successfully")
        else:
            # Show arm disconnected notification - independent of main status
            self.status_manager.show_arm_disconnected_notification()
            self.logger.warning("Arm connection lost or failed")

    def _start_application(self):
        """Start the application components."""
        # Start camera
        self.camera_controller.start()

        # Start game
        self.game_controller.start_game()

        # Setup timers
        self._setup_timers()

    def _handle_game_ended(self, winner):
        """Handle game end and record statistics."""
        human_player = self.game_controller.human_player
        self.statistics_widget.record_game_result(winner, human_player)
        # Show game end notification is already handled by status_manager
        self.logger.info("Game ended: winner={winner}, recorded in statistics")

    def _handle_statistics_reset(self):
        """Handle statistics reset request."""
        self.logger.info("Statistics reset requested by user")
        # Could add confirmation dialog here if needed

    def _setup_timers(self):
        """Setup application timers."""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.game_controller.update_game_state)
        self.update_timer.start(100)  # 10 FPS


if __name__ == "__main__":
    # Basic logger configuration if not set elsewhere
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )

    app = QApplication(sys.argv)
    window = TicTacToeApp(config=None)
    window.show()
    sys.exit(app.exec_())
