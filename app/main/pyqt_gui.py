"""
TicTacToe GUI Module - Refactored Architecture Entry Point

This module serves as a backward-compatibility wrapper for the refactored TicTacToe application.
The original monolithic pyqt_gui.py has been refactored into a modular architecture:

Refactored Modules:
- main_gui.py: Main window, layout, and UI setup
- game_controller.py: Game logic, state management, turn coordination  
- camera_controller.py: Camera integration and detection processing
- arm_movement_controller.py: Centralized robotic arm control
- ui_event_handlers.py: UI event handling and user interactions
- status_manager.py: Status updates, language management, UI state

This wrapper maintains 100% backward compatibility with existing code that imports
from pyqt_gui.py while providing access to the new modular architecture.

Usage:
    from app.main.pyqt_gui import TicTacToeApp
    
    # All existing usage patterns continue to work unchanged
    app = TicTacToeApp(config=config)
    app.show()
"""

import sys
import os
import logging

# Add project root to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new modular TicTacToe application
from app.main.main_gui import TicTacToeApp

# Export for backward compatibility
__all__ = ['TicTacToeApp']


def main():
    """Entry point for direct execution of the TicTacToe application."""
    from PyQt5.QtWidgets import QApplication
    
    # Configure logging if not already configured
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
    
    # Create and run the application
    app = QApplication(sys.argv)
    window = TicTacToeApp(config=None)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
