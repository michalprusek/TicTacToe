"""Comprehensive tests for error_handler.py module."""
import pytest
import logging
from unittest.mock import Mock, MagicMock, patch
import traceback

from app.main.error_handler import ErrorHandler


class TestErrorHandlerLogError:
    """Test ErrorHandler.log_error static method."""
    
    def test_log_error_with_traceback(self):
        """Test log_error with traceback enabled."""
        mock_logger = Mock(spec=logging.Logger)
        test_error = ValueError("Test error message")
        
        with patch('traceback.format_exc', return_value="Mock traceback"):
            ErrorHandler.log_error(mock_logger, "test operation", test_error, include_traceback=True)
        
        # Should call logger.error with traceback
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "Error in test operation: Test error message" in call_args
        assert "Mock traceback" in call_args
    
    def test_log_error_without_traceback(self):
        """Test log_error with traceback disabled."""
        mock_logger = Mock(spec=logging.Logger)
        test_error = RuntimeError("Runtime error")
        
        ErrorHandler.log_error(mock_logger, "runtime operation", test_error, include_traceback=False)
        
        # Should call logger.error without traceback
        mock_logger.error.assert_called_once_with("Error in runtime operation: Runtime error")
    
    def test_log_error_default_traceback(self):
        """Test log_error with default traceback setting (True)."""
        mock_logger = Mock(spec=logging.Logger)
        test_error = ConnectionError("Connection failed")
        
        with patch('traceback.format_exc', return_value="Connection traceback"):
            ErrorHandler.log_error(mock_logger, "connection", test_error)
        
        # Should include traceback by default
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "Error in connection: Connection failed" in call_args
        assert "Connection traceback" in call_args


class TestErrorHandlerSafeOperation:
    """Test ErrorHandler.safe_operation decorator."""
    
    def test_safe_operation_success(self):
        """Test safe_operation decorator with successful function."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "test operation")
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
        mock_logger.error.assert_not_called()
    
    def test_safe_operation_with_exception(self):
        """Test safe_operation decorator with function that raises exception."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "failing operation", default_return="fallback")
        def failing_function():
            raise ValueError("Test exception")
        
        result = failing_function()
        assert result == "fallback"
        mock_logger.error.assert_called_once()
    
    def test_safe_operation_default_return_none(self):
        """Test safe_operation decorator with default return value None."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "operation")
        def failing_function():
            raise RuntimeError("Error")
        
        result = failing_function()
        assert result is None
        mock_logger.error.assert_called_once()
    
    def test_safe_operation_with_traceback(self):
        """Test safe_operation decorator with traceback logging enabled."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "traced operation", log_traceback=True)
        def failing_function():
            raise KeyError("Key not found")
        
        with patch('traceback.format_exc', return_value="Mock traceback"):
            result = failing_function()
        
        assert result is None
        mock_logger.error.assert_called_once()
        # Check that traceback was included
        call_args = mock_logger.error.call_args[0][0]
        assert "Mock traceback" in call_args
    
    def test_safe_operation_preserves_function_metadata(self):
        """Test that safe_operation preserves original function metadata."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "operation")
        def documented_function():
            """This is a test function."""
            return "success"
        
        # Should preserve function name and docstring
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function."
    
    def test_safe_operation_with_args_kwargs(self):
        """Test safe_operation decorator preserves function arguments."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "complex operation")
        def complex_function(a, b, c=3, d=4):
            return a + b + c + d
        
        result = complex_function(1, 2, d=5)
        assert result == 11  # 1 + 2 + 3 + 5
        mock_logger.error.assert_not_called()


class TestErrorHandlerSpecializedHandlers:
    """Test specialized error handler decorators."""
    
    def test_camera_operation_handler_success(self):
        """Test camera operation handler with successful operation."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.camera_operation_handler(mock_logger, "capture")
        def camera_function():
            return True
        
        result = camera_function()
        assert result is True
        mock_logger.error.assert_not_called()
    
    def test_camera_operation_handler_failure(self):
        """Test camera operation handler with failed operation."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.camera_operation_handler(mock_logger, "capture")
        def failing_camera_function():
            raise OSError("Camera not found")
        
        result = failing_camera_function()
        assert result is False  # Default return for camera operations
        mock_logger.error.assert_called_once()
        
        # Should log without traceback
        call_args = mock_logger.error.call_args[0][0]
        assert "Error in camera capture: Camera not found" in call_args
        assert "Traceback" not in call_args
    
    def test_arm_operation_handler_success(self):
        """Test arm operation handler with successful operation."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.arm_operation_handler(mock_logger, "move")
        def arm_function():
            return True
        
        result = arm_function()
        assert result is True
        mock_logger.error.assert_not_called()
    
    def test_arm_operation_handler_failure(self):
        """Test arm operation handler with failed operation."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.arm_operation_handler(mock_logger, "move")
        def failing_arm_function():
            raise ConnectionError("Arm disconnected")
        
        with patch('traceback.format_exc', return_value="Arm traceback"):
            result = failing_arm_function()
        
        assert result is False  # Default return for arm operations
        mock_logger.error.assert_called_once()
        
        # Should log with traceback
        call_args = mock_logger.error.call_args[0][0]
        assert "Error in arm move: Arm disconnected" in call_args
        assert "Arm traceback" in call_args
    
    def test_gui_operation_handler_success(self):
        """Test GUI operation handler with successful operation."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.gui_operation_handler(mock_logger, "update")
        def gui_function():
            return "updated"
        
        result = gui_function()
        assert result == "updated"
        mock_logger.error.assert_not_called()
    
    def test_gui_operation_handler_failure(self):
        """Test GUI operation handler with failed operation."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.gui_operation_handler(mock_logger, "render")
        def failing_gui_function():
            raise AttributeError("Widget not found")
        
        result = failing_gui_function()
        assert result is None  # Default return for GUI operations
        mock_logger.error.assert_called_once()
        
        # Should log without traceback
        call_args = mock_logger.error.call_args[0][0]
        assert "Error in GUI render: Widget not found" in call_args
        assert "Traceback" not in call_args


class TestErrorHandlerIntegration:
    """Test integration scenarios and edge cases."""
    
    def test_nested_decorated_functions(self):
        """Test that nested decorated functions work correctly."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "outer operation")
        def outer_function():
            @ErrorHandler.safe_operation(mock_logger, "inner operation")
            def inner_function():
                raise ValueError("Inner error")
            return inner_function()
        
        result = outer_function()
        assert result is None
        # Should log error from inner function
        assert mock_logger.error.call_count >= 1
    
    def test_different_exception_types(self):
        """Test handling of different exception types."""
        mock_logger = Mock(spec=logging.Logger)
        
        exceptions_to_test = [
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            ConnectionError("Connection error"),
            FileNotFoundError("File not found"),
            KeyError("Key error"),
            AttributeError("Attribute error")
        ]
        
        for exception in exceptions_to_test:
            mock_logger.reset_mock()
            
            @ErrorHandler.safe_operation(mock_logger, "test", default_return="handled")
            def test_function():
                raise exception
            
            result = test_function()
            assert result == "handled"
            mock_logger.error.assert_called_once()
    
    def test_error_message_formatting(self):
        """Test that error messages are formatted correctly."""
        mock_logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(mock_logger, "formatting test")
        def test_function():
            raise ValueError("Test message with special chars: !@#$%^&*()")
        
        test_function()
        
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "Error in formatting test: Test message with special chars: !@#$%^&*()" in error_message
    
    def test_real_logger_integration(self):
        """Test with real logger to ensure compatibility."""
        # Create real logger
        logger = logging.getLogger("test_error_handler")
        logger.setLevel(logging.ERROR)
        
        # Add handler to capture logs
        from io import StringIO
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
        
        try:
            @ErrorHandler.safe_operation(logger, "real logger test")
            def test_function():
                raise RuntimeError("Real error")
            
            result = test_function()
            assert result is None
            
            # Check that something was logged
            log_output = log_capture.getvalue()
            assert "Error in real logger test: Real error" in log_output
            
        finally:
            logger.removeHandler(handler)


if __name__ == "__main__":
    pytest.main([__file__])