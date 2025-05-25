# Robot Control Documentation

Comprehensive documentation for robotic arm control and movement coordination.

## Overview

The robot control system manages a uArm Swift Pro robotic arm for physical TicTacToe gameplay, including symbol drawing, movement coordination, and safety features.

## Hardware Setup

### uArm Swift Pro Specifications
- **Workspace**: 320mm radius, 230mm height
- **Precision**: ±0.2mm repeatability
- **Communication**: USB serial connection
- **Control**: G-code based command protocol
- **Safety**: Software-enforced workspace limits

### Physical Configuration

```
     Y+
     ↑
     │
     │     Game Board (9 cells)
     │   ┌─────┬─────┬─────┐
     │   │(0,0)│(0,1)│(0,2)│
     │   ├─────┼─────┼─────┤
     │   │(1,0)│(1,1)│(1,2)│
     │   ├─────┼─────┼─────┤
     │   │(2,0)│(2,1)│(2,2)│
     │   └─────┴─────┴─────┘
     │
     └─────────────────────→ X+
    Robot Base              
```

### Workspace Calibration

```python
# Typical calibration coordinates (mm)
WORKSPACE_BOUNDS = {
    'x_min': 50,   'x_max': 250,
    'y_min': -100, 'y_max': 100,
    'z_min': 10,   'z_max': 150
}

# Game board corners (example)
BOARD_CORNERS = {
    'top_left':     (80, 80),
    'top_right':    (220, 80),
    'bottom_left':  (80, -80),
    'bottom_right': (220, -80)
}
```

## Control Architecture

### Component Hierarchy

```
ArmMovementController (High-level coordination)
    ↓
ArmController (Movement primitives)
    ↓
ArmThread (Asynchronous execution)
    ↓
pyuarm Library (Hardware communication)
    ↓
uArm Swift Pro (Physical robot)
```

### Key Classes

#### `ArmMovementController`
- **Purpose**: High-level movement coordination and game integration
- **Responsibilities**: Symbol drawing, coordinate transformation, safety validation
- **Location**: `app/main/arm_movement_controller.py`

#### `ArmController`
- **Purpose**: Basic movement primitives and robot communication
- **Responsibilities**: G-code generation, path planning, hardware interface
- **Location**: `app/main/arm_controller.py`

#### `ArmThread`
- **Purpose**: Asynchronous command execution
- **Responsibilities**: Command queuing, timeout handling, thread safety
- **Location**: `app/core/arm_thread.py`

## Movement Primitives

### Basic Operations

#### Position Movement
```python
def go_to_position(self, x: float, y: float, z: float, speed: int = 1000, wait: bool = True) -> bool:
    """Move to absolute position in robot coordinates"""
    # Validate workspace bounds
    if not self._is_position_safe(x, y, z):
        raise ValueError(f"Position ({x}, {y}, {z}) outside safe workspace")
    
    # Send G-code command
    command = f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{speed}"
    success = self._send_command(command, wait)
    
    if wait:
        self._wait_for_completion()
    
    return success
```

#### Height Control
```python
# Safety heights
DEFAULT_SAFE_Z = 100.0   # Safe travel height (mm)
DEFAULT_DRAW_Z = 20.0    # Drawing height (mm)

def move_to_safe_height(self):
    """Move to safe Z height for travel"""
    current_pos = self.get_current_position()
    return self.go_to_position(current_pos.x, current_pos.y, DEFAULT_SAFE_Z)

def lower_to_drawing_height(self, x: float, y: float):
    """Lower to drawing height at specific position"""
    return self.go_to_position(x, y, DEFAULT_DRAW_Z)
```

### Symbol Drawing

#### X Symbol Drawing
```python
def draw_x(self, center_x: float, center_y: float, size: float = 15.0, speed: int = 500) -> bool:
    """Draw X symbol at specified position"""
    half_size = size / 2
    
    # Calculate X endpoints
    points = [
        (center_x - half_size, center_y - half_size),  # Bottom-left
        (center_x + half_size, center_y + half_size),  # Top-right
        (center_x + half_size, center_y - half_size),  # Bottom-right
        (center_x - half_size, center_y + half_size)   # Top-left
    ]
    
    try:
        # Draw first diagonal
        self.go_to_position(points[0][0], points[0][1], DEFAULT_SAFE_Z, speed=speed)
        self.go_to_position(points[0][0], points[0][1], DEFAULT_DRAW_Z, speed=speed//2)
        self.go_to_position(points[1][0], points[1][1], DEFAULT_DRAW_Z, speed=speed//2)
        self.go_to_position(points[1][0], points[1][1], DEFAULT_SAFE_Z, speed=speed//2)
        
        # Move to second diagonal start
        self.go_to_position(points[2][0], points[2][1], DEFAULT_SAFE_Z, speed=speed)
        self.go_to_position(points[2][0], points[2][1], DEFAULT_DRAW_Z, speed=speed//2)
        self.go_to_position(points[3][0], points[3][1], DEFAULT_DRAW_Z, speed=speed//2)
        self.go_to_position(points[3][0], points[3][1], DEFAULT_SAFE_Z, speed=speed//2)
        
        return True
    except Exception as e:
        self.logger.error(f"Failed to draw X: {e}")
        return False
```

#### O Symbol Drawing
```python
def draw_o(self, center_x: float, center_y: float, radius: float = 7.5, speed: int = 500) -> bool:
    """Draw O symbol (circle) at specified position"""
    num_points = 16  # Number of points for circle approximation
    
    try:
        # Calculate circle points
        circle_points = []
        for i in range(num_points + 1):  # +1 to close the circle
            angle = 2 * np.pi * i / num_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            circle_points.append((x, y))
        
        # Move to start position
        start_x, start_y = circle_points[0]
        self.go_to_position(start_x, start_y, DEFAULT_SAFE_Z, speed=speed)
        self.go_to_position(start_x, start_y, DEFAULT_DRAW_Z, speed=speed//2)
        
        # Draw circle
        for x, y in circle_points[1:]:
            self.go_to_position(x, y, DEFAULT_DRAW_Z, speed=speed//2)
        
        # Lift up
        self.go_to_position(start_x, start_y, DEFAULT_SAFE_Z, speed=speed//2)
        
        return True
    except Exception as e:
        self.logger.error(f"Failed to draw O: {e}")
        return False
```

### Advanced Drawing Operations

#### Winning Line Drawing
```python
def draw_winning_line(self, start_cell: Tuple[int, int], end_cell: Tuple[int, int]) -> bool:
    """Draw line through winning symbols"""
    # Get cell coordinates
    start_x, start_y = self._get_cell_coordinates(start_cell[0], start_cell[1])
    end_x, end_y = self._get_cell_coordinates(end_cell[0], end_cell[1])
    
    try:
        # Move to start position
        self.go_to_position(start_x, start_y, DEFAULT_SAFE_Z, speed=MAX_SPEED)
        self.go_to_position(start_x, start_y, DEFAULT_DRAW_Z, speed=DRAWING_SPEED)
        
        # Draw line to end position
        self.go_to_position(end_x, end_y, DEFAULT_DRAW_Z, speed=DRAWING_SPEED)
        
        # Lift to safe height
        self.go_to_position(end_x, end_y, DEFAULT_SAFE_Z, speed=MAX_SPEED)
        
        return True
    except Exception as e:
        self.logger.error(f"Failed to draw winning line: {e}")
        return False
```

## Coordinate Transformation

### Camera-to-Robot Mapping

The system uses hand-eye calibration to transform camera coordinates to robot coordinates:

```python
def transform_camera_to_robot(self, uv_point: Tuple[float, float]) -> Tuple[float, float]:
    """Transform camera UV coordinates to robot XY coordinates"""
    # Load calibration matrix
    calibration_data = self._load_calibration_data()
    xy_to_uv_matrix = np.array(calibration_data["perspective_transform_matrix_xy_to_uv"])
    
    # Invert to get UV-to-XY transformation
    uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)
    
    # Apply transformation using homogeneous coordinates
    uv_homogeneous = np.array([uv_point[0], uv_point[1], 1.0])
    xy_transformed = np.dot(uv_to_xy_matrix, uv_homogeneous)
    
    # Convert back to 2D coordinates
    if xy_transformed[2] != 0:
        robot_x = xy_transformed[0] / xy_transformed[2]
        robot_y = xy_transformed[1] / xy_transformed[2]
        return (robot_x, robot_y)
    else:
        raise ValueError("Invalid transformation (division by zero)")
```

### Calibration Process

```python
def calibrate_hand_eye(self, calibration_points: List[Dict]) -> np.ndarray:
    """Perform hand-eye calibration using known point correspondences"""
    camera_points = []
    robot_points = []
    
    for point in calibration_points:
        camera_points.append(point['camera_uv'])
        robot_points.append(point['robot_xy'])
    
    camera_points = np.array(camera_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)
    
    # Compute homography
    H, _ = cv2.findHomography(robot_points, camera_points, cv2.RANSAC)
    
    # Save calibration
    calibration_data = {
        "perspective_transform_matrix_xy_to_uv": H.tolist(),
        "calibration_points_raw": calibration_points,
        "calibration_date": datetime.now().isoformat()
    }
    
    with open("app/calibration/hand_eye_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=2)
    
    return H
```

## Safety Features

### Workspace Validation

```python
def _is_position_safe(self, x: float, y: float, z: float) -> bool:
    """Validate position is within safe workspace bounds"""
    return (
        WORKSPACE_BOUNDS['x_min'] <= x <= WORKSPACE_BOUNDS['x_max'] and
        WORKSPACE_BOUNDS['y_min'] <= y <= WORKSPACE_BOUNDS['y_max'] and
        WORKSPACE_BOUNDS['z_min'] <= z <= WORKSPACE_BOUNDS['z_max']
    )

def _validate_movement(self, start_pos: Tuple[float, float, float], 
                      end_pos: Tuple[float, float, float]) -> bool:
    """Validate movement path doesn't exceed limits"""
    # Check both endpoints
    if not self._is_position_safe(*start_pos) or not self._is_position_safe(*end_pos):
        return False
    
    # Check movement distance isn't excessive
    distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    if distance > MAX_MOVEMENT_DISTANCE:
        return False
    
    return True
```

### Emergency Stop

```python
def emergency_stop(self) -> bool:
    """Immediately stop all robot movement"""
    try:
        # Send emergency stop command
        self._send_command("M112", wait=False)  # Emergency stop G-code
        
        # Clear command queue
        self._clear_command_queue()
        
        # Move to safe position if possible
        current_pos = self.get_current_position()
        if current_pos and current_pos.z < DEFAULT_SAFE_Z:
            self.go_to_position(current_pos.x, current_pos.y, DEFAULT_SAFE_Z)
        
        return True
    except Exception as e:
        self.logger.error(f"Emergency stop failed: {e}")
        return False
```

### Collision Avoidance

```python
def _check_collision_path(self, waypoints: List[Tuple[float, float, float]]) -> bool:
    """Check if planned path avoids known obstacles"""
    for waypoint in waypoints:
        # Check against known obstacle positions
        for obstacle in self.known_obstacles:
            distance = np.linalg.norm(np.array(waypoint[:2]) - np.array(obstacle[:2]))
            if distance < COLLISION_THRESHOLD:
                return False
    
    return True
```

## Communication Protocol

### G-Code Commands

```python
# Movement commands
GCODE_COMMANDS = {
    'move_linear': 'G1 X{x} Y{y} Z{z} F{speed}',
    'move_rapid': 'G0 X{x} Y{y} Z{z}',
    'home_all': 'G28',
    'home_z': 'G28 Z',
    'set_absolute': 'G90',
    'set_relative': 'G91'
}

# Robot status commands
STATUS_COMMANDS = {
    'get_position': 'M114',
    'get_endstops': 'M119',
    'get_temperature': 'M105'
}
```

### Command Execution

```python
def _send_command(self, command: str, wait: bool = True, timeout: float = 30.0) -> bool:
    """Send G-code command to robot"""
    try:
        if not self.is_connected():
            raise RuntimeError("Robot not connected")
        
        # Send command
        self.swift.send_cmd(command)
        
        if wait:
            # Wait for completion
            start_time = time.time()
            while not self._is_movement_complete():
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Command timeout: {command}")
                time.sleep(0.1)
        
        return True
    except Exception as e:
        self.logger.error(f"Command failed: {command}, Error: {e}")
        return False
```

### Status Monitoring

```python
def get_robot_status(self) -> Dict:
    """Get comprehensive robot status"""
    try:
        status = {
            'connected': self.is_connected(),
            'position': self.get_current_position(),
            'is_moving': self._is_movement_complete(),
            'workspace_valid': True,
            'last_command_time': self.last_command_time,
            'error_count': self.error_count
        }
        
        # Validate current position
        if status['position']:
            pos = status['position']
            status['workspace_valid'] = self._is_position_safe(pos.x, pos.y, pos.z)
        
        return status
    except Exception as e:
        return {'connected': False, 'error': str(e)}
```

## Threading and Concurrency

### Asynchronous Command Execution

```python
class ArmThread(QThread):
    """Thread for asynchronous robot command execution"""
    
    command_completed = pyqtSignal(bool)  # success
    position_changed = pyqtSignal(object)  # position
    error_occurred = pyqtSignal(str)       # error message
    
    def __init__(self):
        super().__init__()
        self.command_queue = queue.Queue()
        self.running = False
        
    def run(self):
        """Main thread execution loop"""
        self.running = True
        
        while self.running:
            try:
                # Get next command from queue
                command = self.command_queue.get(timeout=0.1)
                
                # Execute command
                success = self._execute_command(command)
                
                # Emit completion signal
                self.command_completed.emit(success)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.error_occurred.emit(str(e))
```

### Thread-Safe Communication

```python
def execute_async_command(self, command_type: str, **kwargs) -> None:
    """Queue command for asynchronous execution"""
    command = {
        'type': command_type,
        'parameters': kwargs,
        'timestamp': time.time()
    }
    
    # Thread-safe queue operation
    self.command_queue.put(command)

def wait_for_completion(self, timeout: float = 30.0) -> bool:
    """Wait for current command to complete"""
    start_time = time.time()
    
    while self.is_busy():
        if time.time() - start_time > timeout:
            return False
        
        # Process Qt events to handle signals
        QApplication.processEvents()
        time.sleep(0.01)
    
    return True
```

## Error Handling and Recovery

### Error Categories

```python
class RobotError(Exception):
    """Base class for robot control errors"""
    pass

class CommunicationError(RobotError):
    """Error in robot communication"""
    pass

class MovementError(RobotError):
    """Error in robot movement"""
    pass

class CalibrationError(RobotError):
    """Error in coordinate calibration"""
    pass

class SafetyError(RobotError):
    """Safety violation error"""
    pass
```

### Recovery Strategies

```python
def handle_communication_error(self, error: CommunicationError) -> bool:
    """Attempt to recover from communication errors"""
    try:
        # Reconnect to robot
        self.disconnect()
        time.sleep(1.0)
        
        if self.connect():
            # Re-home robot
            self.home_robot()
            return True
        else:
            return False
            
    except Exception as e:
        self.logger.error(f"Recovery failed: {e}")
        return False

def handle_movement_error(self, error: MovementError) -> bool:
    """Attempt to recover from movement errors"""
    try:
        # Stop current movement
        self.emergency_stop()
        
        # Move to safe position
        self.move_to_neutral_position()
        
        return True
    except Exception as e:
        self.logger.error(f"Movement recovery failed: {e}")
        return False
```

## Performance Optimization

### Movement Optimization

```python
def optimize_movement_path(self, waypoints: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Optimize movement path for speed and safety"""
    optimized = []
    
    for i, waypoint in enumerate(waypoints):
        if i == 0:
            optimized.append(waypoint)
            continue
        
        prev_point = optimized[-1]
        
        # Check if direct movement is safe
        if self._is_direct_path_safe(prev_point, waypoint):
            optimized.append(waypoint)
        else:
            # Add intermediate safe point
            safe_point = self._find_safe_intermediate_point(prev_point, waypoint)
            optimized.extend([safe_point, waypoint])
    
    return optimized
```

### Speed Profiles

```python
# Movement speed configurations
SPEED_PROFILES = {
    'precise': {
        'drawing_speed': 200,
        'travel_speed': 500,
        'approach_speed': 100
    },
    'normal': {
        'drawing_speed': 500,
        'travel_speed': 1000,
        'approach_speed': 300
    },
    'fast': {
        'drawing_speed': 800,
        'travel_speed': 1500,
        'approach_speed': 500
    }
}
```

## Troubleshooting

### Common Issues

#### Robot Not Responding
- **Check USB connection and power**
- **Verify correct COM port**
- **Restart robot and reconnect**

#### Coordinate Accuracy Issues
- **Re-run hand-eye calibration**
- **Check camera position stability**
- **Verify workspace calibration**

#### Movement Stuttering
- **Reduce movement speed**
- **Check for USB communication issues**
- **Verify adequate power supply**

### Debug Tools

```python
def test_robot_connectivity(self) -> Dict:
    """Test robot connection and basic functionality"""
    results = {
        'connection': False,
        'movement': False,
        'position_accuracy': False,
        'errors': []
    }
    
    try:
        # Test connection
        if self.connect():
            results['connection'] = True
        
        # Test basic movement
        home_pos = (150, 0, 100)
        if self.go_to_position(*home_pos):
            results['movement'] = True
        
        # Test position accuracy
        reported_pos = self.get_current_position()
        if reported_pos:
            accuracy = np.linalg.norm(np.array([reported_pos.x, reported_pos.y, reported_pos.z]) - np.array(home_pos))
            results['position_accuracy'] = accuracy < 2.0  # 2mm tolerance
        
    except Exception as e:
        results['errors'].append(str(e))
    
    return results
```

---

This robot control system provides safe, precise, and reliable robotic arm operation for interactive TicTacToe gameplay.
