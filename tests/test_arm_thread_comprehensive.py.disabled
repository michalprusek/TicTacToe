"""
Comprehensive tests for ArmThread module.
Tests command structure, threading, and arm control integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import queue
import time

from app.core.arm_thread import ArmCommand, ArmThread


class TestArmCommand:
    """Test ArmCommand functionality."""

    def test_arm_command_initialization(self):
        """Test ArmCommand initialization."""
        cmd = ArmCommand("draw_x", {"position": (100, 100)})
        
        assert cmd.command_type == "draw_x"
        assert cmd.params == {"position": (100, 100)}
        assert isinstance(cmd.completed, threading.Event)
        assert cmd.success is False
        assert cmd.result is None

    def test_arm_command_initialization_no_params(self):
        """Test ArmCommand initialization without parameters."""
        cmd = ArmCommand("move_to_neutral")
        
        assert cmd.command_type == "move_to_neutral"
        assert cmd.params == {}
        assert cmd.success is False

    def test_mark_completed_success(self):
        """Test marking command as completed successfully."""
        cmd = ArmCommand("draw_o")
        result_data = {"status": "success"}
        
        cmd.mark_completed(True, result_data)
        
        assert cmd.success is True
        assert cmd.result == result_data
        assert cmd.completed.is_set()

    def test_mark_completed_failure(self):
        """Test marking command as completed with failure."""
        cmd = ArmCommand("draw_x")
        
        cmd.mark_completed(False, "Connection error")
        
        assert cmd.success is False
        assert cmd.result == "Connection error"
        assert cmd.completed.is_set()

    def test_wait_for_completion_immediate(self):
        """Test waiting for completion when already completed."""
        cmd = ArmCommand("test")
        cmd.mark_completed(True)
        
        result = cmd.wait_for_completion(timeout=1.0)
        
        assert result is True

    def test_wait_for_completion_timeout(self):
        """Test waiting for completion with timeout."""
        cmd = ArmCommand("test")
        
        start_time = time.time()
        result = cmd.wait_for_completion(timeout=0.1)
        end_time = time.time()
        
        assert result is False
        assert 0.09 <= (end_time - start_time) <= 0.2  # Allow some timing variance

    def test_wait_for_completion_no_timeout(self):
        """Test waiting for completion without timeout."""
        cmd = ArmCommand("test")
        
        # Mark completed in separate thread after short delay
        def complete_after_delay():
            time.sleep(0.05)
            cmd.mark_completed(True)
        
        thread = threading.Thread(target=complete_after_delay)
        thread.start()
        
        result = cmd.wait_for_completion()
        thread.join()
        
        assert result is True
        assert cmd.success is True


class TestArmThreadInit:
    """Test ArmThread initialization and basic functionality."""

    @patch('app.core.arm_thread.ArmController')
    def test_arm_thread_initialization(self, mock_arm_controller):
        """Test ArmThread initialization."""
        mock_controller = Mock()
        mock_arm_controller.return_value = mock_controller
        
        arm_thread = ArmThread()
        
        assert arm_thread.arm_controller == mock_controller
        assert isinstance(arm_thread.command_queue, queue.Queue)
        assert arm_thread.shutdown_event is not None
        assert isinstance(arm_thread.shutdown_event, threading.Event)
        assert arm_thread.daemon is True

    @patch('app.core.arm_thread.ArmController')
    def test_arm_thread_inherits_from_thread(self, mock_arm_controller):
        """Test that ArmThread inherits from threading.Thread."""
        arm_thread = ArmThread()
        assert isinstance(arm_thread, threading.Thread)

    @patch('app.core.arm_thread.ArmController')
    def test_arm_thread_logger_setup(self, mock_arm_controller):
        """Test that ArmThread sets up logger."""
        with patch('app.core.arm_thread.logging.getLogger') as mock_logger:
            arm_thread = ArmThread()
            mock_logger.assert_called_once()


class TestArmThreadCommands:
    """Test ArmThread command handling."""

    @patch('app.core.arm_thread.ArmController')
    def test_add_command(self, mock_arm_controller):
        """Test adding command to queue."""
        arm_thread = ArmThread()
        cmd = ArmCommand("test_command")
        
        arm_thread.add_command(cmd)
        
        # Command should be in queue
        assert not arm_thread.command_queue.empty()
        queued_cmd = arm_thread.command_queue.get_nowait()
        assert queued_cmd == cmd

    @patch('app.core.arm_thread.ArmController')
    def test_stop_thread(self, mock_arm_controller):
        """Test stopping the thread."""
        arm_thread = ArmThread()
        
        arm_thread.stop()
        
        assert arm_thread.shutdown_event.is_set()

    @patch('app.core.arm_thread.ArmController')
    def test_is_connected_delegates_to_controller(self, mock_arm_controller):
        """Test that is_connected delegates to arm controller."""
        mock_controller = Mock()
        mock_controller.is_connected.return_value = True
        mock_arm_controller.return_value = mock_controller
        
        arm_thread = ArmThread()
        result = arm_thread.is_connected()
        
        mock_controller.is_connected.assert_called_once()
        assert result is True


class TestArmThreadConstants:
    """Test ArmThread constants and class attributes."""

    def test_arm_command_has_required_attributes(self):
        """Test that ArmCommand has all required attributes."""
        cmd = ArmCommand("test")
        
        required_attrs = ['command_type', 'params', 'completed', 'success', 'result']
        for attr in required_attrs:
            assert hasattr(cmd, attr), f"ArmCommand missing attribute: {attr}"

    def test_arm_command_methods_exist(self):
        """Test that ArmCommand has required methods."""
        cmd = ArmCommand("test")
        
        assert callable(cmd.mark_completed)
        assert callable(cmd.wait_for_completion)

    @patch('app.core.arm_thread.ArmController')
    def test_arm_thread_has_required_attributes(self, mock_arm_controller):
        """Test that ArmThread has all required attributes."""
        arm_thread = ArmThread()
        
        required_attrs = ['arm_controller', 'command_queue', 'shutdown_event', 'daemon']
        for attr in required_attrs:
            assert hasattr(arm_thread, attr), f"ArmThread missing attribute: {attr}"

    @patch('app.core.arm_thread.ArmController')
    def test_arm_thread_methods_exist(self, mock_arm_controller):
        """Test that ArmThread has required methods."""
        arm_thread = ArmThread()
        
        required_methods = ['add_command', 'stop', 'is_connected', 'run']
        for method in required_methods:
            assert hasattr(arm_thread, method), f"ArmThread missing method: {method}"
            assert callable(getattr(arm_thread, method)), f"ArmThread.{method} is not callable"


class TestArmThreadIntegration:
    """Test ArmThread integration with arm controller."""

    @patch('app.core.arm_thread.ArmController')
    def test_arm_controller_integration(self, mock_arm_controller):
        """Test that ArmThread properly integrates with ArmController."""
        mock_controller = Mock()
        mock_arm_controller.return_value = mock_controller
        
        arm_thread = ArmThread()
        
        # Verify controller is created and stored
        mock_arm_controller.assert_called_once()
        assert arm_thread.arm_controller == mock_controller

    @patch('app.core.arm_thread.ArmController')
    def test_command_queue_type(self, mock_arm_controller):
        """Test that command queue is proper Queue type."""
        arm_thread = ArmThread()
        
        assert isinstance(arm_thread.command_queue, queue.Queue)
        assert arm_thread.command_queue.empty()

    @patch('app.core.arm_thread.ArmController')
    def test_shutdown_event_type(self, mock_arm_controller):
        """Test that shutdown event is proper Event type."""
        arm_thread = ArmThread()
        
        assert isinstance(arm_thread.shutdown_event, threading.Event)
        assert not arm_thread.shutdown_event.is_set()
