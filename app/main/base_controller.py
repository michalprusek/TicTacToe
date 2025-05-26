# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Base controller class for common controller patterns.
Consolidates repeated initialization and setup patterns.
"""
# pylint: disable=no-name-in-module,unused-argument,wrong-import-order,unused-import

import logging
from typing import Optional

from PyQt5.QtCore import QObject

from app.core.config import AppConfig
from app.main.game_utils import setup_logger


class BaseController(QObject):
    """Base class for all controllers with common functionality."""

    def __init__(self, main_window, config: Optional[AppConfig] = None):
        """
        Initialize base controller.

        Args:
            main_window: Reference to main application window
            config: Application configuration (optional)
        """
        super().__init__()

        self.main_window = main_window
        self.config = config if config is not None else AppConfig()
        self.logger = setup_logger(self.__class__.__name__)

        # Common initialization flags
        self._initialized = False
        self._connected = False

    def initialize(self) -> bool:
        """
        Initialize the controller. Override in subclasses.

        Returns:
            True if initialization successful, False otherwise
        """
        self._initialized = True
        self.logger.info("{self.__class__.__name__} initialized successfully")
        return True

    def cleanup(self) -> None:
        """
        Cleanup controller resources. Override in subclasses.
        """
        self._connected = False
        self._initialized = False
        self.logger.info("{self.__class__.__name__} cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._initialized

    @property
    def is_connected(self) -> bool:
        """Check if controller is connected (for external devices)."""
        return self._connected

    def _log_status_change(self, status: str) -> None:
        """Log status changes with standardized format."""
        self.logger.info("{self.__class__.__name__} status: {status}")

    def _emit_status_change(self, message: str, is_key: bool = False) -> None:
        """Emit status change if signal available."""
        if hasattr(self, 'status_changed'):
            self.status_changed.emit(message, is_key)
