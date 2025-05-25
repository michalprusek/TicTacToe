"""
GUI Factory for standardized widget and layout creation.
Consolidates repeated GUI creation patterns from multiple files.
"""

from PyQt5.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QSlider, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from app.main.style_manager import StyleManager
from typing import Optional, List, Callable


class ButtonFactory:
    """Factory for creating standardized buttons."""

    @staticmethod
    def create_button(text: str, style_type: str = "default",
                     click_handler: Optional[Callable] = None,
                     font_size: int = 10, min_width: int = 100) -> QPushButton:
        """
        Create standardized button with consistent styling.

        Args:
            text: Button text
            style_type: Style type (default, primary, danger, success)
            click_handler: Click event handler
            font_size: Font size
            min_width: Minimum button width

        Returns:
            Configured QPushButton
        """
        button = QPushButton(text)
        button.setMinimumWidth(min_width)
        button.setFont(QFont("Arial", font_size, QFont.Bold))

        StyleManager.style_button(button, style_type)

        if click_handler:
            button.clicked.connect(click_handler)

        return button

    @staticmethod
    def create_control_button(text: str, click_handler: Callable,
                            enabled: bool = True) -> QPushButton:
        """Create control button with standard control styling."""
        button = ButtonFactory.create_button(
            text, "primary", click_handler,
            font_size=10, min_width=120
        )
        button.setEnabled(enabled)
        return button

    @staticmethod
    def create_danger_button(text: str, click_handler: Callable) -> QPushButton:
        """Create danger/warning button with red styling."""
        return ButtonFactory.create_button(
            text, "danger", click_handler,
            font_size=10, min_width=120
        )


class LayoutFactory:
    """Factory for creating standardized layouts."""

    @staticmethod
    def create_vbox_layout(spacing: int = 5, margins: tuple = (5, 5, 5, 5)) -> QVBoxLayout:
        """
        Create vertical box layout with standard spacing and margins.

        Args:
            spacing: Spacing between widgets
            margins: Margins (left, top, right, bottom)

        Returns:
            Configured QVBoxLayout
        """
        layout = QVBoxLayout()
        layout.setSpacing(spacing)
        layout.setContentsMargins(*margins)
        return layout    @staticmethod
    def create_hbox_layout(spacing: int = 5, margins: tuple = (5, 5, 5, 5)) -> QHBoxLayout:
        """Create horizontal box layout with standard spacing and margins."""
        layout = QHBoxLayout()
        layout.setSpacing(spacing)
        layout.setContentsMargins(*margins)
        return layout

    @staticmethod
    def create_button_row(buttons: List[QPushButton], spacing: int = 10) -> QHBoxLayout:
        """Create horizontal layout with buttons."""
        layout = LayoutFactory.create_hbox_layout(spacing)
        for button in buttons:
            layout.addWidget(button)
        layout.addStretch()
        return layout

    @staticmethod
    def create_label_value_row(label_text: str, value_widget: QWidget) -> QHBoxLayout:
        """Create horizontal row with label and value widget."""
        layout = LayoutFactory.create_hbox_layout()

        label = QLabel(label_text + ":")
        label.setMinimumWidth(100)
        layout.addWidget(label)
        layout.addWidget(value_widget)
        layout.addStretch()

        return layout


class LabelFactory:
    """Factory for creating standardized labels."""

    @staticmethod
    def create_status_label(text: str = "", status_type: str = "default") -> QLabel:
        """Create status label with standardized styling."""
        label = QLabel(text)
        StyleManager.style_status_label(label, status_type)
        label.setAlignment(Qt.AlignCenter)
        return label

    @staticmethod
    def create_info_label(text: str, font_size: int = 10) -> QLabel:
        """Create informational label."""
        label = QLabel(text)
        label.setFont(QFont("Arial", font_size))
        label.setWordWrap(True)
        return label