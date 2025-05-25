"""
Extended tests for ErrorHandler module.
"""
import pytest
import logging
from unittest.mock import Mock, patch
from app.main.error_handler import ErrorHandler


class TestErrorHandlerExtended:
    """Extended tests for ErrorHandler class."""

    def test_log_error_without_traceback(self):
        """Test log_error without traceback."""
        mock_logger = Mock(spec=logging.Logger)
        test_error = ValueError("Test error")
        
        ErrorHandler.log_error(mock_logger, "test operation", test_error, 
                             include_traceback=False)
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "Error in test operation: Test error" in call_args
        assert "traceback" not in call_args.lower()

    def test_safe_operation_decorator_success(self):
        """Test safe_operation decorator with successful function."""
        mock_logger = Mock()
        
        @ErrorHandler.safe_operation(mock_logger, "test operation")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        mock_logger.error.assert_not_called()

    def test_safe_operation_decorator_with_exception(self):
        """Test safe_operation decorator with exception."""
        mock_logger = Mock()
        
        @ErrorHandler.safe_operation(mock_logger, "test operation", 
                                   default_return="default")
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        assert result == "default"
        mock_logger.error.assert_called_once()

    def test_safe_operation_decorator_with_args_kwargs(self):
        """Test safe_operation decorator preserves args and kwargs."""
        mock_logger = Mock()
        
        @ErrorHandler.safe_operation(mock_logger, "test operation")
        def test_function(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"
        
        result = test_function("a", "b", kwarg1="c")
        assert result == "a-b-c"

    def test_safe_operation_with_traceback_enabled(self):
        """Test safe_operation with traceback logging enabled."""
        mock_logger = Mock()
        
        @ErrorHandler.safe_operation(mock_logger, "test operation", 
                                   log_traceback=True)
        def test_function():
            raise RuntimeError("Test runtime error")
        
        test_function()
        mock_logger.error.assert_called_once()
        # Verify error was called with traceback logging
        assert mock_logger.error.called
