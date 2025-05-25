"""
Centralized error handling utilities for the TicTacToe application.
Consolidates repeated error handling patterns from multiple files.
"""

import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps


class ErrorHandler:
    """Centralized error handling for common operations."""
    
    @staticmethod
    def log_error(logger: logging.Logger, operation: str, error: Exception, 
                  include_traceback: bool = True) -> None:
        """
        Standardized error logging.
        
        Args:
            logger: Logger instance
            operation: Description of the operation that failed
            error: The exception that occurred
            include_traceback: Whether to include full traceback
        """
        error_msg = f"Error in {operation}: {error}"
        
        if include_traceback:
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
        else:
            logger.error(error_msg)
    
    @staticmethod
    def safe_operation(logger: logging.Logger, operation_name: str, 
                      default_return: Any = None, log_traceback: bool = False):
        """
        Decorator for safe operation execution with standardized error handling.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    ErrorHandler.log_error(logger, operation_name, e, log_traceback)
                    return default_return
            return wrapper
        return decorator    @staticmethod
    def camera_operation_handler(logger: logging.Logger, operation: str) -> Callable:
        """Specialized decorator for camera operations."""
        return ErrorHandler.safe_operation(
            logger, f"camera {operation}", 
            default_return=False, 
            log_traceback=False
        )
    
    @staticmethod  
    def arm_operation_handler(logger: logging.Logger, operation: str) -> Callable:
        """Specialized decorator for arm operations."""
        return ErrorHandler.safe_operation(
            logger, f"arm {operation}",
            default_return=False,
            log_traceback=True
        )
    
    @staticmethod
    def gui_operation_handler(logger: logging.Logger, operation: str) -> Callable:
        """Specialized decorator for GUI operations."""
        return ErrorHandler.safe_operation(
            logger, f"GUI {operation}",
            default_return=None,
            log_traceback=False
        )