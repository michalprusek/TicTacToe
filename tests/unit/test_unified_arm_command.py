"""
Test unified arm command functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.main.pyqt_gui import TicTacToeApp
from app.main import game_logic


class TestUnifiedArmCommand:
    """Test the unified arm command interface"""

    @pytest.fixture
    def mock_app(self):
        """Create a mock PyQt application"""
        with patch('app.main.pyqt_gui.QApplication') as mock_qapp:
            mock_qapp.instance.return_value = None
            mock_qapp.return_value = Mock()
            yield mock_qapp

    @pytest.fixture
    def gui(self, mock_app):
        """Create a GUI instance with mocked dependencies"""
        with patch('app.main.pyqt_gui.QMainWindow.__init__'):
            with patch('app.main.pyqt_gui.TicTacToeApp.init_game_components'):
                with patch('app.main.pyqt_gui.TicTacToeApp.init_ui'):
                    with patch('app.main.pyqt_gui.CameraThread'):
                        gui = TicTacToeApp()

                        # Mock the logger
                        gui.logger = Mock()

                        # Mock arm components
                        gui.arm_thread = Mock()
                        gui.arm_thread.connected = True
                        gui.arm_controller = Mock()
                        gui.arm_controller.connected = True

                        return gui

    def test_unified_arm_command_with_arm_thread(self, gui):
        """Test unified arm command uses arm_thread when available"""
        # Setup
        gui.arm_thread.draw_x = Mock(return_value=True)
        gui.arm_controller.draw_x = Mock(return_value=True)

        # Execute
        result = gui._unified_arm_command('draw_x', 100, 200, 50, speed=5)

        # Verify
        assert result is True
        gui.arm_thread.draw_x.assert_called_once_with(100, 200, 50, speed=5)
        gui.arm_controller.draw_x.assert_not_called()

    def test_unified_arm_command_fallback_to_controller(self, gui):
        """Test unified arm command falls back to arm_controller"""
        # Setup - arm_thread not available
        gui.arm_thread = None
        gui.arm_controller.draw_o = Mock(return_value=True)

        # Execute
        result = gui._unified_arm_command('draw_o', 150, 250, 25, speed=3)

        # Verify
        assert result is True
        gui.arm_controller.draw_o.assert_called_once_with(150, 250, 25, speed=3)

    def test_unified_arm_command_no_arm_available(self, gui):
        """Test unified arm command when no arm is available"""
        # Setup - no arm available
        gui.arm_thread = None
        gui.arm_controller = None

        # Execute
        result = gui._unified_arm_command('draw_x', 100, 200, 50)

        # Verify
        assert result is False
        gui.logger.warning.assert_called_once()

    def test_unified_arm_command_with_kwargs(self, gui):
        """Test unified arm command passes keyword arguments correctly"""
        # Setup
        gui.arm_thread.go_to_position = Mock(return_value=True)

        # Execute
        result = gui._unified_arm_command('go_to_position', x=200, y=100, wait=True)

        # Verify
        assert result is True
        gui.arm_thread.go_to_position.assert_called_once_with(x=200, y=100, wait=True)

    def test_unified_arm_command_method_not_found(self, gui):
        """Test unified arm command when method doesn't exist"""
        # Setup
        gui.arm_thread.invalid_method = None
        delattr(gui.arm_thread, 'invalid_method') if hasattr(gui.arm_thread, 'invalid_method') else None

        # Execute
        result = gui._unified_arm_command('invalid_method', 100, 200)

        # Verify
        assert result is False
        gui.logger.warning.assert_called()

    def test_unified_arm_command_exception_handling(self, gui):
        """Test unified arm command handles exceptions gracefully"""
        # Setup
        gui.arm_thread.draw_x = Mock(side_effect=Exception("Test error"))
        gui.arm_controller.draw_x = Mock(return_value=True)

        # Execute
        result = gui._unified_arm_command('draw_x', 100, 200, 50)

        # Verify - should fallback to controller
        assert result is True
        gui.arm_thread.draw_x.assert_called_once()
        gui.arm_controller.draw_x.assert_called_once()
        gui.logger.warning.assert_called()

    def test_unified_arm_command_park_method(self, gui):
        """Test unified arm command with park method"""
        # Setup
        gui.arm_thread.park = Mock(return_value=True)

        # Execute
        result = gui._unified_arm_command('park', x=200, y=0, wait=True)

        # Verify
        assert result is True
        gui.arm_thread.park.assert_called_once_with(x=200, y=0, wait=True)

    def test_unified_arm_command_connected_check(self, gui):
        """Test unified arm command checks connection status"""
        # Setup - arm_thread not connected
        gui.arm_thread.connected = False
        gui.arm_controller.connected = True
        gui.arm_controller.draw_x = Mock(return_value=True)

        # Execute
        result = gui._unified_arm_command('draw_x', 100, 200, 50)

        # Verify - should use controller since arm_thread not connected
        assert result is True
        gui.arm_controller.draw_x.assert_called_once()

    def test_unified_arm_command_both_disconnected(self, gui):
        """Test unified arm command when both arms are disconnected"""
        # Setup
        gui.arm_thread.connected = False
        gui.arm_controller.connected = False

        # Execute
        result = gui._unified_arm_command('draw_x', 100, 200, 50)

        # Verify
        assert result is False
        gui.logger.warning.assert_called()
