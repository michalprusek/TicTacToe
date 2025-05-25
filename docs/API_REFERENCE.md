# API Reference

Complete API documentation for the Robotic TicTacToe application.

## Core Classes

### GameState (`app/core/game_state.py`)

Central game state management with board validation and coordinate transformation.

```python
class GameState:
    """Authoritative game state with detection integration"""
    
    def __init__(self, board_size: int = 3):
        """Initialize game state
        
        Args:
            board_size: Size of the game board (default: 3 for 3x3)
        """
```

#### Properties
```python
@property
def board(self) -> List[List[str]]:
    """Current board state as 2D list"""

@property
def is_valid_grid(self) -> bool:
    """True if grid detection is valid"""

@property
def grid_points(self) -> Optional[np.ndarray]:
    """Detected grid points (16 points for 4x4 grid)"""
```

#### Methods
```python
def update_from_detection(self, grid_points: np.ndarray, symbols: List[Dict], 
                         frame_time: float) -> bool:
    """Update state from computer vision detection
    
    Args:
        grid_points: Detected grid intersection points
        symbols: Detected X/O symbols with positions
        frame_time: Timestamp of frame
        
    Returns:
        True if state was updated
    """

def get_cell_center_uv(self, row: int, col: int) -> Optional[Tuple[float, float]]:
    """Get UV coordinates of cell center
    
    Args:
        row: Board row (0-2)
        col: Board column (0-2)
        
    Returns:
        (u, v) coordinates or None if invalid
    """

def is_valid(self) -> bool:
    """Check if current state is valid for gameplay"""

def get_empty_cells(self) -> List[Tuple[int, int]]:
    """Get list of empty cell coordinates"""

def check_winner(self) -> Optional[str]:
    """Check for winner
    
    Returns:
        'X', 'O', 'tie', or None
    """

def get_winning_line(self) -> Optional[List[Tuple[int, int]]]:
    """Get winning line coordinates if game is won"""
```

---

### GameController (`app/main/game_controller.py`)

Central game orchestration and turn management.

```python
class GameController(QObject):
    """Main game controller managing turns and state"""
    
    # Signals
    status_changed = pyqtSignal(str)          # status_message
    ai_move_ready = pyqtSignal(int, int, str) # row, col, symbol
    game_ended = pyqtSignal(str)              # winner
    board_updated = pyqtSignal(object)        # board_state
```

#### Methods
```python
def start_new_game(self) -> None:
    """Start a new game"""

def handle_detected_game_state(self, detected_board: GameState) -> None:
    """Process game state from computer vision
    
    Args:
        detected_board: Detected game state
    """

def handle_user_move(self, row: int, col: int) -> bool:
    """Process manual user move
    
    Args:
        row: Board row (0-2)
        col: Board column (0-2)
        
    Returns:
        True if move was valid
    """

def handle_ai_move_completed(self, success: bool) -> None:
    """Handle completion of AI robot move
    
    Args:
        success: True if move completed successfully
    """

def pause_game(self) -> None:
    """Pause the current game"""

def resume_game(self) -> None:
    """Resume paused game"""

def get_game_status(self) -> Dict:
    """Get current game status information"""
```

---

### ArmMovementController (`app/main/arm_movement_controller.py`)

High-level robotic arm movement coordination.

```python
class ArmMovementController(QObject):
    """Centralized controller for robotic arm movements"""
    
    # Signals
    arm_connected = pyqtSignal(bool)      # connection_status
    arm_move_completed = pyqtSignal(bool) # success
    arm_status_changed = pyqtSignal(str)  # status_message
```

#### Methods
```python
def is_arm_available(self) -> bool:
    """Check if arm is available for use"""

def move_to_neutral_position(self) -> bool:
    """Move arm to neutral position"""

def draw_ai_symbol(self, row: int, col: int, symbol_to_draw: str) -> bool:
    """Draw AI symbol at specified position
    
    Args:
        row: Board row (0-2)
        col: Board column (0-2)
        symbol_to_draw: 'X' or 'O'
        
    Returns:
        True if drawing succeeded
    """

def draw_winning_line(self) -> bool:
    """Draw winning line through winning symbols"""

def calibrate_arm(self) -> bool:
    """Calibrate arm (placeholder for future implementation)"""

def park_arm(self) -> bool:
    """Park arm in safe position"""

def get_arm_status(self) -> Dict:
    """Get current arm status information"""
```

---

### CameraController (`app/main/camera_controller.py`)

Camera management and detection processing coordination.

```python
class CameraController(QObject):
    """Controls camera integration and detection processing"""
    
    # Signals
    frame_ready = pyqtSignal(object)        # frame
    game_state_updated = pyqtSignal(object) # detected_board
    fps_updated = pyqtSignal(float)         # fps
    grid_warning = pyqtSignal(str)          # warning_message
    grid_incomplete = pyqtSignal(bool)      # True=incomplete, False=complete
```

#### Methods
```python
def start(self) -> None:
    """Start the camera thread"""

def stop(self) -> None:
    """Stop the camera thread"""

def restart_camera(self, new_camera_index: int) -> None:
    """Restart camera with new index
    
    Args:
        new_camera_index: New camera device index
    """

def get_current_board_state(self) -> Optional[GameState]:
    """Get current board state from camera"""

def get_calibration_data(self) -> Optional[Dict]:
    """Get calibration data from camera thread"""

def set_detection_threshold(self, threshold: float) -> None:
    """Set detection threshold
    
    Args:
        threshold: Detection confidence threshold (0.0-1.0)
    """

def get_detection_threshold(self) -> float:
    """Get current detection threshold"""

def calibrate_camera(self) -> None:
    """Trigger camera calibration"""

def is_camera_active(self) -> bool:
    """Check if camera is active"""

def get_camera_info(self) -> Dict:
    """Get camera information and status"""
```

---

### GameDetector (`app/main/game_detector.py`)

Computer vision game detection pipeline.

```python
class GameDetector:
    """Detects Tic Tac Toe grid and symbols using YOLO models"""
    
    def __init__(self, config: GameDetectorConfig, camera_index: int = 0,
                 detect_model_path: str = DEFAULT_DETECT_MODEL_PATH,
                 pose_model_path: str = DEFAULT_POSE_MODEL_PATH,
                 device: Optional[str] = None):
        """Initialize detector with models and camera
        
        Args:
            config: Detector configuration
            camera_index: Camera device index
            detect_model_path: Path to symbol detection model
            pose_model_path: Path to grid detection model
            device: Processing device ('cpu', 'cuda', 'mps')
        """
```

#### Methods
```python
def process_frame(self, frame: np.ndarray, frame_time: float) -> Tuple[np.ndarray, Optional[GameState]]:
    """Process single frame for detection
    
    Args:
        frame: Input camera frame
        frame_time: Frame timestamp
        
    Returns:
        (annotated_frame, game_state)
    """

def run_detection(self) -> None:
    """Run detection loop on camera feed"""

def release(self) -> None:
    """Release detector resources"""
```

---

### GridDetector (`app/main/grid_detector.py`)

Grid detection and processing functionality.

```python
class GridDetector:
    """Detects and processes the Tic Tac Toe grid"""
    
    def __init__(self, pose_model, config=None, logger=None):
        """Initialize grid detector
        
        Args:
            pose_model: YOLO pose model for grid detection
            config: Configuration object
            logger: Logger instance
        """
```

#### Methods
```python
def detect_grid(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect grid in frame
    
    Args:
        frame: Input camera frame
        
    Returns:
        (processed_frame, keypoints) where keypoints is (16, 2) array
    """

def sort_grid_points(self, keypoints: np.ndarray) -> np.ndarray:
    """Sort grid points into consistent order
    
    Args:
        keypoints: Detected grid points
        
    Returns:
        Sorted keypoints array
    """

def is_valid_grid(self, keypoints: np.ndarray) -> bool:
    """Check if detected grid is valid
    
    Args:
        keypoints: Grid points to validate
        
    Returns:
        True if grid is valid
    """

def compute_homography(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
    """Compute homography matrix from ideal grid to image coordinates
    
    Args:
        keypoints: Detected grid points
        
    Returns:
        Homography matrix or None if computation fails
    """

def update_grid_status(self, is_valid: bool, current_time: float) -> bool:
    """Update grid status based on validity and timing
    
    Args:
        is_valid: Whether current grid detection is valid
        current_time: Current timestamp
        
    Returns:
        True if grid status changed significantly
    """
```

---

### SymbolDetector (`app/main/symbol_detector.py`)

Symbol detection functionality.

```python
class SymbolDetector:
    """Detects X and O symbols in the game"""
    
    def __init__(self, detect_model, config=None, logger=None):
        """Initialize symbol detector
        
        Args:
            detect_model: YOLO detection model
            config: Configuration object
            logger: Logger instance
        """
```

#### Methods
```python
def detect_symbols(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Detect symbols in frame
    
    Args:
        frame: Input camera frame
        
    Returns:
        (processed_frame, symbols_list)
    """

def filter_symbols_by_confidence(self, symbols: List[Dict]) -> List[Dict]:
    """Filter symbols by confidence threshold
    
    Args:
        symbols: List of detected symbols
        
    Returns:
        Filtered symbols list
    """
```

---

### Strategy (`app/core/strategy.py`)

AI strategy implementation with minimax algorithm.

```python
class BernoulliStrategySelector:
    """Strategy selector using Bernoulli distribution"""
    
    def __init__(self, difficulty: int = 5):
        """Initialize strategy selector
        
        Args:
            difficulty: Game difficulty (1-10)
        """

def select_strategy(self, board: List[List[str]]) -> str:
    """Select strategy for current move
    
    Args:
        board: Current board state
        
    Returns:
        'random' or 'minimax'
    """

class MinimaxStrategy:
    """Minimax algorithm with alpha-beta pruning"""
    
    def get_best_move(self, board: List[List[str]], player: str) -> Optional[Tuple[int, int]]:
        """Get best move using minimax
        
        Args:
            board: Current board state
            player: Current player ('X' or 'O')
            
    Returns:
        (row, col) of best move or None
    """
```

## Configuration Classes

### AppConfig (`app/core/config.py`)

Main application configuration.

```python
@dataclass
class AppConfig:
    """Main application configuration"""
    detector: GameDetectorConfig
    arm: ArmConfig  
    game: GameConfig
    ui: UIConfig

@dataclass
class GameDetectorConfig:
    """Computer vision detection configuration"""
    camera_index: int = 0
    frame_width: int = 1920
    frame_height: int = 1080
    detection_confidence: float = 0.45
    pose_confidence: float = 0.45
    processing_fps: float = 2.0
    
@dataclass  
class ArmConfig:
    """Robot arm configuration"""
    port: Optional[str] = None
    safe_z: float = 100.0
    draw_z: float = 20.0
    drawing_speed: int = 500
    movement_speed: int = 1000
    
@dataclass
class GameConfig:
    """Game logic configuration"""
    difficulty: int = 5
    max_detection_wait_time: float = 10.0
    max_retry_count: int = 3
    move_cooldown_seconds: float = 2.0
```

## Utility Functions

### Game Logic (`app/main/game_logic.py`)

Core game logic functions.

```python
def create_board() -> List[List[str]]:
    """Create empty 3x3 game board"""

def is_valid_move(board: List[List[str]], row: int, col: int) -> bool:
    """Check if move is valid
    
    Args:
        board: Current board state
        row: Move row (0-2)
        col: Move column (0-2)
        
    Returns:
        True if move is valid
    """

def make_move(board: List[List[str]], row: int, col: int, player: str) -> List[List[str]]:
    """Make move on board
    
    Args:
        board: Current board state
        row: Move row (0-2)
        col: Move column (0-2)
        player: Player symbol ('X' or 'O')
        
    Returns:
        Updated board state
    """

def check_winner(board: List[List[str]]) -> Optional[str]:
    """Check for game winner
    
    Args:
        board: Current board state
        
    Returns:
        'X', 'O', 'tie', or None
    """

def is_board_full(board: List[List[str]]) -> bool:
    """Check if board is full"""

def get_empty_cells(board: List[List[str]]) -> List[Tuple[int, int]]:
    """Get list of empty cell coordinates"""

def get_winning_line(board: List[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """Get winning line coordinates if game is won"""
```

### Game Utils (`app/main/game_utils.py`)

General utility functions.

```python
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """

def validate_board_state(board: List[List[str]]) -> bool:
    """Validate board state is legal
    
    Args:
        board: Board state to validate
        
    Returns:
        True if valid
    """

def calculate_move_confidence(detection_history: List[Dict]) -> float:
    """Calculate confidence in detected move
    
    Args:
        detection_history: History of detections
        
    Returns:
        Confidence score (0.0-1.0)
    """
```

### Path Utils (`app/main/path_utils.py`)

Path and project setup utilities.

```python
def setup_project_path() -> None:
    """Setup project path for imports"""

def get_project_root() -> str:
    """Get project root directory path"""

def get_weights_path() -> str:
    """Get path to model weights directory"""

def get_calibration_path() -> str:
    """Get path to calibration data directory"""
```

## Error Handling

### ErrorHandler (`app/main/error_handler.py`)

Centralized error handling.

```python
class ErrorHandler:
    """Centralized error handling and logging"""
    
    @staticmethod
    def log_error(logger: logging.Logger, operation: str, error: Exception, 
                  critical: bool = False) -> None:
        """Log error with context
        
        Args:
            logger: Logger instance
            operation: Operation that failed
            error: Exception that occurred
            critical: Whether error is critical
        """

    @staticmethod
    def handle_camera_error(error: Exception) -> bool:
        """Handle camera-related errors
        
    Args:
        error: Camera error
        
    Returns:
        True if recovery possible
    """

    @staticmethod
    def handle_robot_error(error: Exception) -> bool:
        """Handle robot-related errors
        
        Args:
            error: Robot error
            
        Returns:
            True if recovery possible
        """
```

## Constants

### Detector Constants (`app/core/detector_constants.py`)

Computer vision detection constants.

```python
# Detection thresholds
BBOX_CONF_THRESHOLD = 0.45
POSE_CONF_THRESHOLD = 0.45
KEYPOINT_VISIBLE_THRESHOLD = 0.3

# Model paths
DEFAULT_DETECT_MODEL_PATH = "weights/best_detection.pt"
DEFAULT_POSE_MODEL_PATH = "weights/best_pose.pt"

# Homography parameters
MIN_POINTS_FOR_HOMOGRAPHY = 4
RANSAC_REPROJ_THRESHOLD = 5.0

# Grid validation
GRID_DIST_STD_DEV_THRESHOLD = 0.3
GRID_ANGLE_TOLERANCE_DEG = 15.0

# Debug colors (BGR format)
COLOR_GRID_POINTS = (0, 255, 0)      # Green
COLOR_SYMBOLS = (255, 0, 0)          # Blue
COLOR_CELL_CENTERS = (0, 0, 255)     # Red
```

### Main Constants (`app/main/constants.py`)

Application-wide constants.

```python
# Camera settings
DEFAULT_CAMERA_INDEX = 0

# Robot movement
DEFAULT_SAFE_Z = 100.0
DEFAULT_DRAW_Z = 20.0
DRAWING_SPEED = 500
MAX_SPEED = 1000

# Game settings
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

# UI update intervals
STATUS_UPDATE_INTERVAL = 100  # milliseconds
FPS_UPDATE_INTERVAL = 1000    # milliseconds
```

## Signal Reference

### GameController Signals

```python
status_changed = pyqtSignal(str)          # Status message updates
ai_move_ready = pyqtSignal(int, int, str) # AI move: (row, col, symbol)
game_ended = pyqtSignal(str)              # Game end: winner
board_updated = pyqtSignal(object)        # Board state updates
```

### CameraController Signals

```python
frame_ready = pyqtSignal(object)          # New camera frame
game_state_updated = pyqtSignal(object)   # Detected game state
fps_updated = pyqtSignal(float)           # FPS updates
grid_warning = pyqtSignal(str)            # Grid visibility warnings
grid_incomplete = pyqtSignal(bool)        # Grid completeness status
```

### ArmMovementController Signals

```python
arm_connected = pyqtSignal(bool)          # Arm connection status
arm_move_completed = pyqtSignal(bool)     # Move completion status
arm_status_changed = pyqtSignal(str)      # Arm status updates
```

## Usage Examples

### Basic Game Initialization

```python
from app.core.config import AppConfig
from app.main.game_controller import GameController
from app.main.camera_controller import CameraController
from app.main.arm_movement_controller import ArmMovementController

# Create configuration
config = AppConfig()

# Initialize controllers
game_controller = GameController(config)
camera_controller = CameraController(main_window, camera_index=0)
arm_controller = ArmMovementController(main_window, config)

# Connect signals
camera_controller.game_state_updated.connect(
    game_controller.handle_detected_game_state
)
game_controller.ai_move_ready.connect(
    arm_controller.draw_ai_symbol
)

# Start components
camera_controller.start()
game_controller.start_new_game()
```

### Manual Move Processing

```python
# Handle user click on board
def on_board_click(self, row: int, col: int):
    if self.game_controller.handle_user_move(row, col):
        print(f"Move made at ({row}, {col})")
    else:
        print("Invalid move")
```

### Detection Threshold Adjustment

```python
# Adjust detection sensitivity
camera_controller.set_detection_threshold(0.6)  # Higher confidence required
current_threshold = camera_controller.get_detection_threshold()
print(f"Detection threshold: {current_threshold}")
```

### Robot Movement Control

```python
# Draw X at position (1, 1)
success = arm_controller.draw_ai_symbol(1, 1, 'X')
if success:
    print("Symbol drawn successfully")

# Check arm status
status = arm_controller.get_arm_status()
print(f"Arm connected: {status['connected']}")
print(f"Safe Z height: {status['safe_z']}")
```

---

This API reference provides comprehensive documentation for all public interfaces in the Robotic TicTacToe application.
