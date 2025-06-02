#!/usr/bin/env python3
# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Main entry point for the PyQt-based TicTacToe application.
"""
import argparse
import logging

# pylint: disable=no-name-in-module
import sys

from PyQt5.QtWidgets import QApplication

from app.core.config import AppConfig
from app.main.main_gui import TicTacToeApp

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TicTacToe")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Robot Tic Tac Toe Game with PyQt GUI")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index to use (default: 0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with additional logging and visualization"
    )
    parser.add_argument(
        "--difficulty", type=int, default=10, choices=range(11),
        help="Initial difficulty level (0-10, default: 10)"
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="Serial port for uArm connection (e.g., COM3, /dev/ttyUSB0). Auto-detects if not specified."
    )
    args = parser.parse_args()

    # Create application configuration
    config = AppConfig()
    config.game_detector.camera_index = args.camera
    config.game.default_difficulty = args.difficulty
    config.debug_mode = args.debug
    if args.port:
        config.arm_controller.port = args.port

    # Configure logging level based on debug mode
    if config.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    # Create and run the Qt application
    app = QApplication(sys.argv)
    window = TicTacToeApp(config=config)
    window.show()

    # Start the application event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
