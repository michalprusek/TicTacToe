# Architecture Documentation

Detailed technical architecture of the Robotic TicTacToe application.

## System Overview

The application follows a modular, signal-based architecture that separates concerns across different layers:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Control Layer  │    │ Hardware Layer  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ MainGUI     │◄┼────┼─│GameController│◄┼────┼─│CameraThread │ │
│ │ BoardWidget │ │    │ │StatusManager│ │    │ │DetectionThr.│ │
│ │ DebugWindow │ │    │ │ErrorHandler │ │    │ │ArmThread    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Detection Pipeline (`app/core/`)

#### `detection_thread.py`
- **Purpose**: Continuous computer vision processing
- **Frequency**: ~2 FPS for optimal balance of responsiveness and CPU usage
- **Models**: YOLO-based grid detection and symbol recognition
- **Output**: Game state updates via signals

#### `game_state.py`
- **Purpose**: Authoritative game state management
- **Features**: 
  - Board state validation
  - Coordinate transformation (camera → normalized grid)
  - Symbol-to-cell mapping with confidence thresholds
- **Critical**: Single source of truth for board state

#### `arm_thread.py`
- **Purpose**: Asynchronous robot control
- **Safety**: Movement validation and collision avoidance
- **Communication**: Command queuing with status feedback

### 2. Control Layer (`app/main/`)

#### `game_controller.py`
- **Purpose**: Central game orchestration
- **Responsibilities**:
  - Turn management (human vs AI)
  - Move validation and coordination
  - Timeout and retry handling
- **State Machine**: Manages game flow states

#### `camera_controller.py`
- **Purpose**: Camera management and detection coordination
- **Features**:
  - Multi-camera support with runtime switching
  - Frame processing pipeline
  - Grid visibility monitoring

#### `arm_movement_controller.py`
- **Purpose**: High-level robot movement coordination
- **Features**:
  - Symbol drawing (X, O)
  - Winning line drawing
  - Coordinate transformation integration

### 3. UI Layer (`app/main/`)

#### `main_gui.py`
- **Purpose**: Primary application window
- **Architecture**: PyQt5 with signal-slot communication
- **Update Frequency**: 100ms timer for responsive UI

#### `board_widget.py`
- **Purpose**: Visual game board representation
- **Features**:
  - Real-time board state display
  - Move animations and highlighting
  - Click interaction for manual control

#### `debug_window.py`
- **Purpose**: Development and debugging interface
- **Features**:
  - Live camera feed with detection overlays
  - Performance metrics (FPS, detection confidence)
  - Coordinate transformation visualization

## Signal-Based Communication

### Core Signals Flow

```
CameraController.game_state_updated
    ↓
GameController.handle_detected_game_state()
    ↓
BoardWidget.update_board_state()

GameController.ai_move_ready
    ↓
ArmMovementController.draw_ai_symbol()
    ↓
ArmMovementController.arm_move_completed
```

### Key Signal Chains

#### Detection → Game Logic
```python
CameraController.game_state_updated.emit(detected_board)
GameController.handle_detected_game_state(detected_board)
```

#### Game Logic → UI Updates
```python
GameController.status_changed.emit(status_message)
StatusManager.update_status(status_message)
```

#### Robot Control Flow
```python
GameController.ai_move_ready.emit(row, col, symbol)
ArmMovementController.draw_ai_symbol(row, col, symbol)
ArmMovementController.arm_move_completed.emit(success)
```

## State Management

### Authoritative Board State

The `GameController.authoritative_board` serves as the single source of truth:

```python
# Only updated from camera detection
authoritative_board = [
    ['X', ' ', 'O'],
    [' ', 'X', ' '],
    ['O', ' ', 'X']
]
```

### Turn Coordination Flags

```python
class GameController:
    arm_move_in_progress: bool      # Robot is physically moving
    waiting_for_detection: bool     # Waiting for camera confirmation
    current_turn: str              # 'human' or 'ai'
    game_active: bool              # Game is running (not paused/ended)
```

### State Transitions

```
Game Start → Human Turn → Move Detection → AI Turn → Robot Movement → Move Confirmation → Human Turn
     ↑                                                                                           ↓
     └─── Game End ← Win Detection ← Move Validation ← Detection Timeout ← Retry Logic ←────────┘
```

## Threading Architecture

### Thread Responsibilities

#### Main Thread (GUI)
- **Responsibilities**: PyQt5 event loop, UI updates, user interaction
- **Update Rate**: 100ms timer for status updates
- **Critical**: All GUI operations must happen in main thread

#### Detection Thread
- **Responsibilities**: Computer vision processing, model inference
- **Rate**: ~2 FPS (configurable)
- **Communication**: Signals to main thread (thread-safe)

#### Camera Thread
- **Responsibilities**: Frame capture, camera management
- **Rate**: 30 FPS capture, buffered processing
- **Buffering**: Handles frame rate mismatch between capture and processing

#### Arm Thread
- **Responsibilities**: Robot command execution, movement coordination
- **Safety**: Independent safety checking and timeout handling
- **Queuing**: Command queue with priority and cancellation support

### Thread Communication

```python
# Detection Thread → Main Thread
detection_thread.game_state_updated.connect(
    game_controller.handle_detected_game_state,
    Qt.QueuedConnection  # Thread-safe signal delivery
)

# Main Thread → Arm Thread
arm_thread.execute_command(command, parameters)
arm_thread.command_completed.connect(
    game_controller.handle_arm_command_completed
)
```

## Computer Vision Pipeline

### Two-Stage Detection Architecture

#### Stage 1: Grid Detection (Pose Estimation)
```python
# Input: Camera frame (1920x1080)
# Model: YOLO pose estimation
# Output: 16 keypoints forming 4x4 grid
grid_points = pose_model.predict(frame)
```

#### Stage 2: Symbol Detection (Object Detection)
```python
# Input: Same camera frame
# Model: YOLO object detection
# Output: X/O symbols with bounding boxes and confidence
symbols = detection_model.predict(frame)
```

### Coordinate Transformation Pipeline

```
Camera Pixels → Homography Matrix → Normalized Grid → Cell Mapping → Robot Coordinates
     (u,v)     →    H * [u,v,1]    →   (grid_x,y)   →  cell(r,c)   →     (x,y,z)
```

#### Homography Computation
```python
# From detected grid points to ideal 4x4 grid
ideal_grid = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0],
    [0, 1], [1, 1], [2, 1], [3, 1],
    [0, 2], [1, 2], [2, 2], [3, 2],
    [0, 3], [1, 3], [2, 3], [3, 3]
])
H, _ = cv2.findHomography(ideal_grid, detected_points, cv2.RANSAC)
```

### Robust Grid Sorting Algorithm

The grid sorting algorithm handles rotation and perspective distortion:

```python
def sort_grid_points(self, keypoints):
    # 1. Filter valid points (non-zero)
    valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]
    
    # 2. Sort by Y coordinate (top to bottom)
    y_sorted = valid_points[np.argsort(valid_points[:, 1])]
    
    # 3. Group into rows and sort each row by X
    rows = np.array_split(y_sorted, 4)
    sorted_points = []
    for row in rows:
        x_sorted = row[np.argsort(row[:, 0])]
        sorted_points.extend(x_sorted)
    
    return np.array(sorted_points)
```

## Configuration System

### Hierarchical Configuration

```python
@dataclass
class AppConfig:
    detector: GameDetectorConfig
    arm: ArmConfig
    game: GameConfig
    ui: UIConfig
```

### Configuration Sources
1. **Default Values**: Hardcoded in dataclass definitions
2. **Configuration Files**: JSON/YAML overrides
3. **Command Line**: Runtime parameter overrides
4. **Calibration Data**: Hardware-specific calibration

### Example Configuration
```python
@dataclass
class GameDetectorConfig:
    camera_index: int = 0
    frame_width: int = 1920
    frame_height: int = 1080
    detection_confidence: float = 0.45
    pose_confidence: float = 0.45
    processing_fps: float = 2.0
    
@dataclass
class ArmConfig:
    port: str = None  # Auto-detect
    safe_z: float = 100.0
    draw_z: float = 20.0
    drawing_speed: int = 500
    movement_speed: int = 1000
```

## AI Strategy Architecture

### Strategy Pattern Implementation

```python
class StrategySelector:
    def select_strategy(self, difficulty: int, board_state: List[List[str]]) -> str:
        # Returns 'random' or 'minimax' based on difficulty
        probability = self._calculate_minimax_probability(difficulty)
        return np.random.choice(['random', 'minimax'], p=[1-probability, probability])

class MinimaxStrategy:
    def get_best_move(self, board: List[List[str]], player: str) -> Tuple[int, int]:
        # Alpha-beta pruning minimax implementation
        _, best_move = self._minimax(board, player, True, float('-inf'), float('inf'))
        return best_move
```

### Difficulty Scaling

```python
def _calculate_minimax_probability(self, difficulty: int) -> float:
    # Exponential scaling: difficulty 1 = 5% optimal, difficulty 10 = 95% optimal
    return 0.05 + 0.9 * ((difficulty - 1) / 9) ** 2
```

## Error Handling and Recovery

### Hierarchical Error Handling

```python
class ErrorHandler:
    @staticmethod
    def handle_detection_error(error: Exception, retry_count: int) -> bool:
        # Returns True if should retry, False if should abort
        
    @staticmethod
    def handle_robot_error(error: Exception, context: str) -> None:
        # Robot-specific error recovery
        
    @staticmethod
    def handle_camera_error(error: Exception) -> bool:
        # Camera reconnection logic
```

### Retry Logic

```python
class GameController:
    def _handle_detection_timeout(self):
        if self.retry_count < self.max_retry_count:
            self.retry_count += 1
            self._retry_detection()
        else:
            self._fallback_to_manual_mode()
```

### Graceful Degradation

1. **Camera Failure**: Fall back to manual board input
2. **Robot Failure**: Continue with detection-only mode
3. **Detection Failure**: Retry with adjusted thresholds
4. **Calibration Issues**: Use default coordinate mapping

## Performance Considerations

### Optimization Strategies

#### Computer Vision
- **Model Optimization**: Use TensorRT or ONNX for GPU acceleration
- **Frame Skipping**: Process every nth frame during non-critical periods
- **ROI Processing**: Focus detection on game area only

#### Threading
- **Lock-Free Communication**: Use Qt signals for thread communication
- **Buffering**: Frame buffers to handle processing speed variations
- **Priority Queuing**: Critical commands get priority in robot queue

#### Memory Management
- **Frame Pooling**: Reuse frame buffers to reduce allocations
- **Model Caching**: Keep models loaded in GPU memory
- **Garbage Collection**: Explicit cleanup for large objects

### Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.fps_calculator = FPSCalculator(buffer_size=30)
        self.detection_times = collections.deque(maxlen=100)
        self.robot_command_times = collections.deque(maxlen=50)
    
    def log_performance_metrics(self):
        avg_detection_time = np.mean(self.detection_times)
        current_fps = self.fps_calculator.get_fps()
        # Log metrics for optimization analysis
```

## Security Considerations

### Hardware Safety
- **Movement Validation**: All robot movements validated before execution
- **Emergency Stop**: Immediate halt capability for robot operations
- **Workspace Boundaries**: Software-enforced movement limits

### Software Security
- **Input Validation**: All external inputs validated and sanitized
- **Error Isolation**: Exceptions contained within components
- **Resource Limits**: Memory and processing time limits enforced

### Network Security (Future)
- **Communication Encryption**: For remote operation capabilities
- **Authentication**: User access control for robot operations
- **Audit Logging**: Complete operation history for debugging

---

This architecture supports modular development, hardware abstraction, and reliable real-time operation for robotic gameplay applications.