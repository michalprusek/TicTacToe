# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""Shared styles for the TicTacToe application."""


def create_button_style():
    """Create default button style."""
    return """
        QPushButton {
            background-color: #2d2d30;
            color: #f0f0f0;
            border: 1px solid #3f3f46;
            border-radius: 5px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #3e3e42;
            border-color: #0078D7;
        }
        QPushButton:pressed {
            background-color: #1f1f23;
        }
        QPushButton:disabled {
            background-color: #2d2d30;
            color: #6e6e6e;
            border-color: #3f3f46;
        }
    """


def create_reset_button_style():
    """Create reset button style."""
    return """
        QPushButton {
            background-color: #C0392B;
            color: white;
            border: 2px solid #A93226;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #E74C3C;
            border-color: #C0392B;
        }
        QPushButton:pressed {
            background-color: #A93226;
        }
    """
