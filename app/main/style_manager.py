"""
Centralized style management for GUI components.
Consolidates repeated styling patterns from multiple files.
"""

from PyQt5.QtWidgets import QPushButton, QLabel, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class StyleManager:
    """Centralized styling for consistent UI appearance."""

    # Base styles
    BASE_BUTTON_STYLE = """
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1f618d;
        }
    """

    STATUS_STYLES = {
        "default": "color: #333333; font-weight: bold; font-size: 14px;",
        "info": "color: #3498db; font-weight: bold; font-size: 14px;",
        "warning": "color: #f39c12; font-weight: bold; font-size: 14px;",
        "error": "color: #e74c3c; font-weight: bold; font-size: 14px;",
        "success": "color: #27ae60; font-weight: bold; font-size: 14px;"
    }

    @staticmethod
    def style_button(button: QPushButton, style_type: str = "default") -> None:
        """Apply standardized button styling."""
        styles = {
            "default": StyleManager.BASE_BUTTON_STYLE,
            "primary": StyleManager.BASE_BUTTON_STYLE,
            "danger": StyleManager.BASE_BUTTON_STYLE.replace("#3498db", "#e74c3c")
                                                   .replace("#2980b9", "#c0392b")
                                                   .replace("#1f618d", "#a93226"),
            "success": StyleManager.BASE_BUTTON_STYLE.replace("#3498db", "#27ae60")
                                                    .replace("#2980b9", "#229954")
                                                    .replace("#1f618d", "#1e8449")
        }
        button.setStyleSheet(styles.get(style_type, styles["default"]))

    @staticmethod
    def style_status_label(label: QLabel, status_type: str = "default") -> None:
        """Apply standardized status label styling."""
        style = StyleManager.STATUS_STYLES.get(status_type,
                                              StyleManager.STATUS_STYLES["default"])
        label.setStyleSheet(style)

    @staticmethod
    def create_status_panel_style() -> str:
        """Get standardized status panel styling."""
        return """
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """