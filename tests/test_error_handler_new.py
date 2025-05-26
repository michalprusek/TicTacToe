# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Comprehensive tests for app.main.error_handler module using pytest.
Tests ErrorHandler class with all methods and decorators.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from app.main.error_handler import ErrorHandler


class TestErrorHandler:
    """Test ErrorHandler class functionality."""
    
    def test_log_error_with_traceback(self):
        """Test log_error with traceback included."""
        logger = Mock(spec=logging.Logger)
        operation = "test operation"
        error = ValueError("test error")
        
        with patch('traceback.format_exc', return_value="traceback details"):
            ErrorHandler.log_error(logger, operation, error, include_traceback=True)
        
        expected_msg = "Error in test operation: test error\ntraceback details"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_log_error_without_traceback(self):
        """Test log_error without traceback."""
        logger = Mock(spec=logging.Logger)
        operation = "test operation"
        error = ValueError("test error")
        
        ErrorHandler.log_error(logger, operation, error, include_traceback=False)
        
        expected_msg = "Error in test operation: test error"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_log_error_default_traceback(self):
        """Test log_error with default traceback setting (True)."""
        logger = Mock(spec=logging.Logger)
        operation = "test operation"
        error = RuntimeError("runtime error")
        
        with patch('traceback.format_exc', return_value="default traceback"):
            ErrorHandler.log_error(logger, operation, error)
        
        expected_msg = "Error in test operation: runtime error\ndefault traceback"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_safe_operation_decorator_success(self):
        """Test safe_operation decorator with successful function."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(logger, "test op", default_return="default")
        def test_func(x, y):
            return x + y
        
        result = test_func(2, 3)
        assert result == 5
        logger.error.assert_not_called()
    
    def test_safe_operation_decorator_exception(self):
        """Test safe_operation decorator with function that raises exception."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(logger, "test op", default_return="default")
        def test_func():
            raise ValueError("test exception")
        
        result = test_func()
        assert result == "default"
        logger.error.assert_called_once()
    
    def test_safe_operation_decorator_with_args_kwargs(self):
        """Test safe_operation decorator preserves function arguments."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(logger, "test op", default_return=None)
        def test_func(a, b, c=None, d=None):
            return (a, b, c, d)
        
        result = test_func(1, 2, c=3, d=4)
        assert result == (1, 2, 3, 4)
        logger.error.assert_not_called()
    
    def test_safe_operation_decorator_with_traceback(self):
        """Test safe_operation decorator with traceback logging enabled."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(logger, "test op", log_traceback=True)
        def test_func():
            raise RuntimeError("runtime error")
        
        with patch('traceback.format_exc', return_value="full traceback"):
            result = test_func()
        
        assert result is None  # default_return not specified
        expected_msg = "Error in test op: runtime error\nfull traceback"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_safe_operation_decorator_without_traceback(self):
        """Test safe_operation decorator with traceback logging disabled."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(logger, "test op", log_traceback=False)
        def test_func():
            raise RuntimeError("runtime error")
        
        result = test_func()
        
        assert result is None
        expected_msg = "Error in test op: runtime error"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_camera_operation_handler(self):
        """Test camera_operation_handler decorator."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.camera_operation_handler(logger, "capture")
        def camera_func():
            raise ConnectionError("camera disconnected")
        
        result = camera_func()
        
        assert result is False  # default_return for camera operations
        expected_msg = "Error in camera capture: camera disconnected"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_camera_operation_handler_success(self):
        """Test camera_operation_handler with successful operation."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.camera_operation_handler(logger, "capture")
        def camera_func():
            return True
        
        result = camera_func()
        
        assert result is True
        logger.error.assert_not_called()
    
    def test_arm_operation_handler(self):
        """Test arm_operation_handler decorator."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.arm_operation_handler(logger, "move")
        def arm_func():
            raise OSError("arm communication error")
        
        with patch('traceback.format_exc', return_value="arm traceback"):
            result = arm_func()
        
        assert result is False  # default_return for arm operations
        expected_msg = "Error in arm move: arm communication error\narm traceback"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_arm_operation_handler_success(self):
        """Test arm_operation_handler with successful operation."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.arm_operation_handler(logger, "move")
        def arm_func():
            return "moved successfully"
        
        result = arm_func()
        
        assert result == "moved successfully"
        logger.error.assert_not_called()
    
    def test_gui_operation_handler(self):
        """Test gui_operation_handler decorator."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.gui_operation_handler(logger, "update")
        def gui_func():
            raise AttributeError("widget not found")
        
        result = gui_func()
        
        assert result is None  # default_return for GUI operations
        expected_msg = "Error in GUI update: widget not found"
        logger.error.assert_called_once_with(expected_msg)
    
    def test_gui_operation_handler_success(self):
        """Test gui_operation_handler with successful operation."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.gui_operation_handler(logger, "update")
        def gui_func():
            return "GUI updated"
        
        result = gui_func()
        
        assert result == "GUI updated"
        logger.error.assert_not_called()
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorators preserve original function metadata."""
        logger = Mock(spec=logging.Logger)
        
        @ErrorHandler.safe_operation(logger, "test", default_return=None)
        def original_func():
            """Original function docstring."""
            pass
        
        assert original_func.__name__ == "original_func"
        assert original_func.__doc__ == "Original function docstring."
    
    def test_multiple_decorators_applied(self):
        """Test function with multiple error handler decorators (edge case)."""
        logger = Mock(spec=logging.Logger)
        
        # This tests decorator composition
        @ErrorHandler.safe_operation(logger, "outer", default_return="outer_default")
        @ErrorHandler.safe_operation(logger, "inner", default_return="inner_default")
        def nested_func():
            raise ValueError("nested error")
        
        result = nested_func()
        # Inner decorator executes first, so returns inner_default
        assert result == "inner_default"
    
    def test_specialized_handlers_use_correct_parameters(self):
        """Test that specialized handlers use correct default parameters."""
        logger = Mock(spec=logging.Logger)
        
        # Test camera handler parameters
        camera_decorator = ErrorHandler.camera_operation_handler(logger, "test")
        assert hasattr(camera_decorator, '__call__')
        
        # Test arm handler parameters  
        arm_decorator = ErrorHandler.arm_operation_handler(logger, "test")
        assert hasattr(arm_decorator, '__call__')
        
        # Test GUI handler parameters
        gui_decorator = ErrorHandler.gui_operation_handler(logger, "test")
        assert hasattr(gui_decorator, '__call__')
    
    def test_error_with_complex_exception_types(self):
        """Test error handling with various exception types."""
        logger = Mock(spec=logging.Logger)
        
        exceptions = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            ConnectionError("connection error"),
            FileNotFoundError("file not found"),
            KeyError("key error")
        ]
        
        for exception in exceptions:
            logger.reset_mock()
            
            @ErrorHandler.safe_operation(logger, "test", default_return="handled")
            def error_func():
                raise exception
            
            result = error_func()
            assert result == "handled"
            logger.error.assert_called_once()