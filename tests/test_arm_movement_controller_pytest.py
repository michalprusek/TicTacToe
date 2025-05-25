"""
Comprehensive pytest tests for ArmMovementController module.
Tests robotic arm movement control and coordination functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import numpy as np
from PyQt5.QtCore import QObject

from app.main.arm_movement_controller import ArmMovementController, DEFAULT_SYMBOL_SIZE
from app.main.constants import DEFAULT_SAFE_Z, DEFAULT_DRAW_Z, DRAWING_SPEED, MAX_SPEED
from app.main import game_logic


class TestArmMovementController:
    """Test class for ArmMovementController."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.arm = Mock()
        config.arm.port = '/dev/ttyUSB0'
        config.arm.safe_z = 100
        config.arm.draw_z = 20
        return config

    @pytest.fixture
    def mock_main_window(self):
        """Create mock main window."""
        main_window = Mock()
        main_window.board_widget = Mock()
        main_window.board_widget.winning_line = [(0, 0), (0, 1), (0, 2)]
        
        # Mock camera controller for coordinate transformation
        camera_controller = Mock()
        main_window.camera_controller = camera_controller
        
        return main_window

    @pytest.fixture
    def arm_controller(self, mock_main_window, mock_config):
        """Create ArmMovementController instance with mocked dependencies."""
        with patch('app.main.arm_movement_controller.ArmThread') as mock_arm_thread, \
             patch('app.main.arm_movement_controller.ArmController') as mock_arm_controller, \
             patch('builtins.open', mock_open(read_data='{"neutral_position": {"x": 150, "y": 0, "z": 100}}')):
            
            controller = ArmMovementController(mock_main_window, mock_config)
            
            # Mock the arm thread instance
            controller.arm_thread = Mock()
            controller.arm_thread.connected = True
            controller.arm_thread.connect.return_value = True
            controller.arm_thread.isRunning.return_value = True
            
            # Mock the arm controller instance
            controller.arm_controller = Mock()
            controller.arm_controller.connected = True
            
            return controller

    def test_init_basic(self, mock_main_window, mock_config):
        """Test basic initialization."""
        with patch('app.main.arm_movement_controller.ArmThread') as mock_arm_thread, \
             patch('app.main.arm_movement_controller.ArmController') as mock_arm_controller, \
             patch('builtins.open', mock_open(read_data='{"neutral_position": {"x": 150, "y": 0, "z": 100}}')):
            
            controller = ArmMovementController(mock_main_window, mock_config)
            
            assert controller.main_window == mock_main_window
            assert controller.config == mock_config
            assert controller.safe_z == mock_config.arm.safe_z
            assert controller.draw_z == mock_config.arm.draw_z
            assert controller.symbol_size == DEFAULT_SYMBOL_SIZE

    def test_load_neutral_position_success(self):
        """Test successful loading of neutral position from calibration."""
        calibration_data = {
            "neutral_position": {
                "x": 200,
                "y": 50,
                "z": 120
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(calibration_data))):
            with patch('app.main.arm_movement_controller.ArmThread'), \
                 patch('app.main.arm_movement_controller.ArmController'):
                
                controller = ArmMovementController(Mock(), Mock())
                
                assert controller.neutral_position['x'] == 200
                assert controller.neutral_position['y'] == 50
                assert controller.neutral_position['z'] == 120

    def test_load_neutral_position_file_not_found(self):
        """Test neutral position loading with missing calibration file."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with patch('app.main.arm_movement_controller.ArmThread'), \
                 patch('app.main.arm_movement_controller.ArmController'):
                
                controller = ArmMovementController(Mock(), Mock())
                
                # Should use defaults
                assert controller.neutral_position['x'] == 150
                assert controller.neutral_position['y'] == 0
                assert controller.neutral_position['z'] == DEFAULT_SAFE_Z

    def test_load_neutral_position_invalid_json(self):
        """Test neutral position loading with invalid JSON."""
        with patch('builtins.open', mock_open(read_data='invalid json')):
            with patch('app.main.arm_movement_controller.ArmThread'), \
                 patch('app.main.arm_movement_controller.ArmController'):
                
                controller = ArmMovementController(Mock(), Mock())
                
                # Should use defaults
                assert controller.neutral_position['x'] == 150
                assert controller.neutral_position['y'] == 0
                assert controller.neutral_position['z'] == DEFAULT_SAFE_Z

    def test_is_arm_available_true(self, arm_controller):
        """Test arm availability check when arm is available."""
        arm_controller.arm_thread.connected = True
        result = arm_controller.is_arm_available()
        assert result is True

    def test_is_arm_available_false_no_thread(self, arm_controller):
        """Test arm availability check when no arm thread."""
        arm_controller.arm_thread = None
        result = arm_controller.is_arm_available()
        assert result is False

    def test_is_arm_available_false_not_connected(self, arm_controller):
        """Test arm availability check when arm not connected."""
        arm_controller.arm_thread.connected = False
        result = arm_controller.is_arm_available()
        assert result is False

    def test_sync_connection_status(self, arm_controller):
        """Test synchronization of connection status between components."""
        # Setup initial state
        arm_controller.arm_thread.connected = True
        arm_controller.arm_controller.connected = False
        arm_controller.arm_thread.arm_controller = Mock()
        arm_controller.arm_thread.arm_controller.swift = Mock()
        
        arm_controller._sync_connection_status()
        
        assert arm_controller.arm_controller.connected is True
        assert arm_controller.arm_controller.swift == arm_controller.arm_thread.arm_controller.swift

    def test_move_to_neutral_position_success(self, arm_controller):
        """Test successful move to neutral position."""
        arm_controller._unified_arm_command = Mock(return_value=True)
        
        result = arm_controller.move_to_neutral_position()
        
        assert result is True
        arm_controller._unified_arm_command.assert_called_once_with(
            'park',
            x=arm_controller.neutral_position['x'],
            y=arm_controller.neutral_position['y'],
            z=arm_controller.neutral_position['z'],
            wait=True
        )

    def test_move_to_neutral_position_arm_not_available(self, arm_controller):
        """Test move to neutral when arm not available."""
        arm_controller.arm_thread.connected = False
        
        result = arm_controller.move_to_neutral_position()
        
        assert result is False

    def test_move_to_neutral_position_failure(self, arm_controller):
        """Test move to neutral position failure."""
        arm_controller._unified_arm_command = Mock(return_value=False)
        
        result = arm_controller.move_to_neutral_position()
        
        assert result is False

    def test_draw_ai_symbol_o_success(self, arm_controller):
        """Test successful drawing of O symbol."""
        arm_controller._get_cell_coordinates_from_yolo = Mock(return_value=(100, 50))
        arm_controller.arm_controller.draw_o = Mock(return_value=True)
        
        result = arm_controller.draw_ai_symbol(1, 1, game_logic.PLAYER_O)
        
        assert result is True
        arm_controller.arm_controller.draw_o.assert_called_once_with(
            center_x=100,
            center_y=50,
            radius=DEFAULT_SYMBOL_SIZE / 2,
            speed=DRAWING_SPEED
        )

    def test_draw_ai_symbol_x_success(self, arm_controller):
        """Test successful drawing of X symbol."""
        arm_controller._get_cell_coordinates_from_yolo = Mock(return_value=(100, 50))
        arm_controller.arm_controller.draw_x = Mock(return_value=True)
        
        result = arm_controller.draw_ai_symbol(1, 1, game_logic.PLAYER_X)
        
        assert result is True
        arm_controller.arm_controller.draw_x.assert_called_once_with(
            center_x=100,
            center_y=50,
            size=DEFAULT_SYMBOL_SIZE,
            speed=DRAWING_SPEED
        )

    def test_draw_ai_symbol_arm_not_available(self, arm_controller):
        """Test drawing symbol when arm not available."""
        arm_controller.arm_thread.connected = False
        
        with pytest.raises(RuntimeError, match="robotic arm is not available"):
            arm_controller.draw_ai_symbol(1, 1, game_logic.PLAYER_O)

    def test_draw_ai_symbol_invalid_coordinates(self, arm_controller):
        """Test drawing symbol with invalid coordinates."""
        arm_controller._get_cell_coordinates_from_yolo = Mock(return_value=(None, None))
        
        with pytest.raises(RuntimeError, match="Cannot get coordinates for drawing"):
            arm_controller.draw_ai_symbol(1, 1, game_logic.PLAYER_O)

    def test_draw_ai_symbol_unknown_symbol(self, arm_controller):
        """Test drawing unknown symbol."""
        arm_controller._get_cell_coordinates_from_yolo = Mock(return_value=(100, 50))
        
        with pytest.raises(ValueError, match="Unknown symbol: Z"):
            arm_controller.draw_ai_symbol(1, 1, "Z")

    def test_draw_winning_line_success(self, arm_controller):
        """Test successful drawing of winning line."""
        arm_controller._get_cell_coordinates_from_yolo = Mock(side_effect=[(100, 50), (200, 50)])
        arm_controller._unified_arm_command = Mock(return_value=True)
        
        result = arm_controller.draw_winning_line()
        
        assert result is True
        assert arm_controller._unified_arm_command.call_count == 4  # 4 movement commands

    def test_draw_winning_line_no_board_widget(self, arm_controller):
        """Test drawing winning line with no board widget."""
        arm_controller.main_window.board_widget = None
        
        with pytest.raises(RuntimeError, match="missing or invalid data"):
            arm_controller.draw_winning_line()

    def test_draw_winning_line_arm_not_available(self, arm_controller):
        """Test drawing winning line when arm not available."""
        arm_controller.arm_thread.connected = False
        
        with pytest.raises(RuntimeError, match="robotic arm is not connected"):
            arm_controller.draw_winning_line()

    def test_draw_winning_line_invalid_data(self, arm_controller):
        """Test drawing winning line with invalid data."""
        arm_controller.main_window.board_widget.winning_line = [(0, 0), (0, 1)]  # Only 2 points
        
        with pytest.raises(RuntimeError, match="Invalid winning line data"):
            arm_controller.draw_winning_line()

    def test_unified_arm_command_draw_o(self, arm_controller):
        """Test unified arm command for drawing O."""
        arm_controller.arm_thread.draw_o = Mock(return_value=True)
        
        result = arm_controller._unified_arm_command(
            'draw_o', x=100, y=50, radius=15, speed=DRAWING_SPEED
        )
        
        assert result is True
        arm_controller.arm_thread.draw_o.assert_called_once_with(
            center_x=100, center_y=50, radius=15, speed=DRAWING_SPEED
        )

    def test_unified_arm_command_draw_x(self, arm_controller):
        """Test unified arm command for drawing X."""
        arm_controller.arm_thread.draw_x = Mock(return_value=True)
        
        result = arm_controller._unified_arm_command(
            'draw_x', x=100, y=50, size=20, speed=DRAWING_SPEED
        )
        
        assert result is True
        arm_controller.arm_thread.draw_x.assert_called_once_with(
            center_x=100, center_y=50, size=20, speed=DRAWING_SPEED
        )

    def test_unified_arm_command_go_to_position(self, arm_controller):
        """Test unified arm command for position movement."""
        arm_controller.arm_thread.go_to_position = Mock(return_value=True)
        
        result = arm_controller._unified_arm_command(
            'go_to_position', x=100, y=50, z=80, speed=MAX_SPEED, wait=True
        )
        
        assert result is True
        arm_controller.arm_thread.go_to_position.assert_called_once_with(
            x=100, y=50, z=80, speed=MAX_SPEED, wait=True
        )

    def test_unified_arm_command_park(self, arm_controller):
        """Test unified arm command for parking."""
        arm_controller.arm_thread.go_to_position = Mock(return_value=True)
        
        result = arm_controller._unified_arm_command('park')
        
        assert result is True
        arm_controller.arm_thread.go_to_position.assert_called_once_with(
            x=arm_controller.neutral_position['x'],
            y=arm_controller.neutral_position['y'],
            z=arm_controller.neutral_position['z'],
            speed=MAX_SPEED // 2,
            wait=True
        )

    def test_unified_arm_command_unknown(self, arm_controller):
        """Test unified arm command with unknown command."""
        with pytest.raises(ValueError, match="Unknown arm command: invalid"):
            arm_controller._unified_arm_command('invalid')

    def test_unified_arm_command_arm_not_available(self, arm_controller):
        """Test unified arm command when arm not available."""
        arm_controller.arm_thread.connected = False
        
        with pytest.raises(RuntimeError, match="robotic arm is not available"):
            arm_controller._unified_arm_command('draw_o')

    def test_unified_arm_command_execution_failure(self, arm_controller):
        """Test unified arm command execution failure."""
        arm_controller.arm_thread.draw_o = Mock(return_value=False)
        
        with pytest.raises(RuntimeError, match="failed to execute"):
            arm_controller._unified_arm_command('draw_o', x=100, y=50, radius=15)

    def test_get_cell_coordinates_basic_transformation(self, arm_controller):
        """Test basic coordinate transformation."""
        # Mock game state object
        game_state_obj = Mock()
        game_state_obj.get_cell_center_uv.return_value = (500, 400)
        game_state_obj._grid_points = [[0, 0]] * 16  # Mock 16 grid points
        
        # Mock camera controller
        arm_controller.main_window.camera_controller._get_detection_data = Mock(
            return_value=(None, game_state_obj)
        )
        
        # Mock calibration data with transformation matrix
        calibration_data = {
            "perspective_transform_matrix_xy_to_uv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "calibration_points_raw": [
                {"robot_xyz": [100, 100, 0], "target_uv": [400, 300]}
            ]
        }
        arm_controller.main_window.camera_controller.get_calibration_data = Mock(
            return_value=calibration_data
        )
        
        result = arm_controller._get_cell_coordinates_from_yolo(1, 1)
        
        assert result is not None
        assert len(result) == 2
        assert isinstance(result[0], (int, float))
        assert isinstance(result[1], (int, float))

    def test_get_cell_coordinates_no_detection_state(self, arm_controller):
        """Test coordinate transformation with no detection state."""
        arm_controller.main_window.camera_controller._get_detection_data = Mock(
            return_value=(None, None)
        )
        
        with pytest.raises(RuntimeError, match="Cannot get detection state"):
            arm_controller._get_cell_coordinates_from_yolo(1, 1)

    def test_get_cell_coordinates_no_uv_center(self, arm_controller):
        """Test coordinate transformation with no UV center."""
        game_state_obj = Mock()
        game_state_obj.get_cell_center_uv.return_value = None
        
        arm_controller.main_window.camera_controller._get_detection_data = Mock(
            return_value=(None, game_state_obj)
        )
        
        with pytest.raises(RuntimeError, match="Cannot get UV center"):
            arm_controller._get_cell_coordinates_from_yolo(1, 1)

    def test_get_cell_coordinates_no_calibration_data(self, arm_controller):
        """Test coordinate transformation with no calibration data."""
        game_state_obj = Mock()
        game_state_obj.get_cell_center_uv.return_value = (500, 400)
        
        arm_controller.main_window.camera_controller._get_detection_data = Mock(
            return_value=(None, game_state_obj)
        )
        arm_controller.main_window.camera_controller.get_calibration_data = Mock(
            return_value=None
        )
        
        with pytest.raises(RuntimeError, match="No perspective transformation matrix available"):
            arm_controller._get_cell_coordinates_from_yolo(1, 1)

    def test_calibrate_arm(self, arm_controller):
        """Test arm calibration placeholder."""
        result = arm_controller.calibrate_arm()
        assert result is True

    def test_park_arm(self, arm_controller):
        """Test arm parking."""
        arm_controller.move_to_neutral_position = Mock(return_value=True)
        
        result = arm_controller.park_arm()
        
        assert result is True
        arm_controller.move_to_neutral_position.assert_called_once()

    def test_get_arm_status(self, arm_controller):
        """Test getting arm status."""
        arm_controller.arm_thread.isRunning = Mock(return_value=True)
        
        status = arm_controller.get_arm_status()
        
        assert isinstance(status, dict)
        assert 'connected' in status
        assert 'arm_thread_active' in status
        assert 'safe_z' in status
        assert 'draw_z' in status
        assert 'symbol_size' in status
        assert status['safe_z'] == arm_controller.safe_z
        assert status['draw_z'] == arm_controller.draw_z
        assert status['symbol_size'] == arm_controller.symbol_size

    def test_get_arm_status_no_thread(self, arm_controller):
        """Test getting arm status with no thread."""
        arm_controller.arm_thread = None
        
        status = arm_controller.get_arm_status()
        
        assert status['arm_thread_active'] is False

    @pytest.mark.parametrize("command,expected_method", [
        ('draw_o', 'draw_o'),
        ('draw_x', 'draw_x'),
        ('go_to_position', 'go_to_position'),
        ('park', 'go_to_position'),
    ])
    def test_unified_arm_command_all_commands(self, arm_controller, command, expected_method):
        """Test all unified arm commands."""
        # Mock the expected method
        mock_method = Mock(return_value=True)
        setattr(arm_controller.arm_thread, expected_method, mock_method)
        
        if command == 'park':
            result = arm_controller._unified_arm_command(command)
        else:
            result = arm_controller._unified_arm_command(command, x=100, y=50)
        
        assert result is True
        mock_method.assert_called_once()

    def test_coordinate_transformation_matrix_operations(self, arm_controller):
        """Test coordinate transformation matrix operations."""
        # Create a realistic transformation scenario
        game_state_obj = Mock()
        game_state_obj.get_cell_center_uv.return_value = (640, 480)  # Center of 1280x960
        game_state_obj._grid_points = np.array([[100, 100]] * 16)  # Mock grid points
        
        arm_controller.main_window.camera_controller._get_detection_data = Mock(
            return_value=(None, game_state_obj)
        )
        
        # Create a realistic transformation matrix (identity-like for testing)
        transform_matrix = np.array([
            [0.5, 0.0, 50.0],
            [0.0, 0.5, 25.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        calibration_data = {
            "perspective_transform_matrix_xy_to_uv": transform_matrix.tolist(),
            "calibration_points_raw": [
                {"robot_xyz": [150, 75, 0], "target_uv": [640, 480]}
            ]
        }
        
        arm_controller.main_window.camera_controller.get_calibration_data = Mock(
            return_value=calibration_data
        )
        
        result = arm_controller._get_cell_coordinates_from_yolo(1, 1)
        
        # Verify result is reasonable
        assert result is not None
        assert len(result) == 2
        x, y = result
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        # The transformation should map (640, 480) -> approximately (1230, 930)
        assert 1000 < x < 1500  # Reasonable range for X
        assert 700 < y < 1200   # Reasonable range for Y


class TestArmMovementControllerEdgeCases:
    """Test edge cases and error conditions for ArmMovementController."""

    def test_initialization_with_missing_config_attributes(self, mock_main_window):
        """Test initialization with incomplete config."""
        config = Mock()
        # No arm attribute
        
        with patch('app.main.arm_movement_controller.ArmThread'), \
             patch('app.main.arm_movement_controller.ArmController'), \
             patch('builtins.open', mock_open(read_data='{}')):
            
            controller = ArmMovementController(mock_main_window, config)
            
            # Should use defaults
            assert controller.safe_z == DEFAULT_SAFE_Z
            assert controller.draw_z == DEFAULT_DRAW_Z

    def test_coordinate_transformation_division_by_zero(self):
        """Test coordinate transformation with division by zero."""
        mock_main_window = Mock()
        config = Mock()
        
        with patch('app.main.arm_movement_controller.ArmThread'), \
             patch('app.main.arm_movement_controller.ArmController'), \
             patch('builtins.open', mock_open(read_data='{}')):
            
            controller = ArmMovementController(mock_main_window, config)
            
            # Mock components for coordinate transformation
            game_state_obj = Mock()
            game_state_obj.get_cell_center_uv.return_value = (500, 400)
            game_state_obj._grid_points = [[0, 0]] * 16
            
            controller.main_window.camera_controller._get_detection_data = Mock(
                return_value=(None, game_state_obj)
            )
            
            # Create transformation matrix that results in zero denominator
            transform_matrix = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0]  # This will cause division by zero
            ], dtype=np.float32)
            
            calibration_data = {
                "perspective_transform_matrix_xy_to_uv": transform_matrix.tolist(),
                "calibration_points_raw": []
            }
            
            controller.main_window.camera_controller.get_calibration_data = Mock(
                return_value=calibration_data
            )
            
            with pytest.raises(RuntimeError, match="Division by zero"):
                controller._get_cell_coordinates_from_yolo(1, 1)

    def test_signal_emissions(self, mock_main_window, mock_config):
        """Test that signals are properly emitted."""
        with patch('app.main.arm_movement_controller.ArmThread'), \
             patch('app.main.arm_movement_controller.ArmController'), \
             patch('builtins.open', mock_open(read_data='{}')):
            
            controller = ArmMovementController(mock_main_window, mock_config)
            
            # Test signal attributes exist
            assert hasattr(controller, 'arm_connected')
            assert hasattr(controller, 'arm_move_completed')
            assert hasattr(controller, 'arm_status_changed')

    def test_initialization_arm_connection_failure(self, mock_main_window, mock_config):
        """Test initialization when arm connection fails."""
        with patch('app.main.arm_movement_controller.ArmThread') as mock_arm_thread_class, \
             patch('app.main.arm_movement_controller.ArmController'), \
             patch('builtins.open', mock_open(read_data='{}')):
            
            # Mock arm thread to fail connection
            mock_arm_thread_instance = Mock()
            mock_arm_thread_instance.connect.return_value = False
            mock_arm_thread_class.return_value = mock_arm_thread_instance
            
            controller = ArmMovementController(mock_main_window, mock_config)
            
            assert hasattr(controller, '_initial_connection_status')
            assert controller._initial_connection_status is False