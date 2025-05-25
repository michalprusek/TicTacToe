# Testing Documentation

Comprehensive guide to testing the Robotic TicTacToe application.

## Testing Strategy

### Framework: Pure pytest
- **No unittest dependencies** - all tests use pure pytest syntax
- **Hardware mocking** - tests run without physical devices
- **Signal testing** - PyQt5 signals properly tested
- **80%+ coverage** achieved for critical modules

## Test Structure

```
tests/
├── conftest.py                          # Test configuration and fixtures
├── test_*_pytest.py                     # Pure pytest test modules
│
├── Core Logic Tests
├── test_config_pytest.py                # Configuration management
├── test_utils_pytest.py                 # Utility functions  
├── test_strategy_pytest.py              # AI strategy algorithms
├── test_game_state_pytest.py            # Game state management
├── test_game_logic_pytest.py            # Core game logic
│
├── Detection Pipeline Tests
├── test_game_detector_pytest.py         # Main detection pipeline
├── test_grid_detector_pytest.py         # Grid detection algorithms
├── test_detector_constants.py           # Detection constants validation
│
├── Hardware Controller Tests
├── test_arm_movement_controller_pytest.py # Robot arm coordination
├── test_camera_controller_pytest.py       # Camera management
│
└── Utility Tests
    ├── test_path_utils_pytest.py        # Path utilities
    └── test_constants_pytest.py         # Application constants
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_game_logic_pytest.py -v

# Run tests matching pattern
pytest tests/ -k "detector" -v
```

### Coverage Testing

```bash
# Basic coverage report
pytest --cov=app --cov-report=term-missing

# HTML coverage report
pytest --cov=app --cov-report=html

# Coverage for specific modules
pytest --cov=app.main.game_detector --cov-report=term-missing
```

### Component-Specific Testing

```bash
# Detection pipeline tests
pytest tests/ -k "detector" -v

# Game logic tests  
pytest tests/ -k "game_logic" -v

# AI strategy tests
pytest tests/ -k "strategy" -v

# Configuration tests
pytest tests/ -k "config" -v

# Hardware controller tests
pytest tests/ -k "controller" -v
```

## Test Configuration

### PyQt5 Mocking (`conftest.py`)

```python
import sys
from unittest.mock import Mock, MagicMock
import pytest

# Mock PyQt5 to avoid GUI dependencies
sys.modules['PyQt5'] = Mock()
sys.modules['PyQt5.QtCore'] = Mock()
sys.modules['PyQt5.QtWidgets'] = Mock()
sys.modules['PyQt5.QtGui'] = Mock()

# Configure Qt mock objects
qt_core_mock = sys.modules['PyQt5.QtCore']
qt_core_mock.QObject = Mock
qt_core_mock.pyqtSignal = Mock
qt_core_mock.QThread = Mock
qt_core_mock.QTimer = Mock
qt_core_mock.Qt = Mock()
qt_core_mock.Qt.QueuedConnection = Mock()

@pytest.fixture
def qt_app():
    """Mock Qt application for testing"""
    return Mock()
```

### Hardware Mocking

```python
@pytest.fixture
def mock_camera():
    """Mock camera for testing without hardware"""
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    camera.set.return_value = True
    camera.get.return_value = 30  # FPS
    return camera

@pytest.fixture  
def mock_robot_arm():
    """Mock robot arm for testing without hardware"""
    arm = Mock()
    arm.connected = True
    arm.go_to_position.return_value = True
    arm.draw_x.return_value = True
    arm.draw_o.return_value = True
    return arm
```

## Test Categories

### 1. Core Logic Tests

#### Configuration Testing (`test_config_pytest.py`)
```python
class TestAppConfig:
    """Test application configuration"""
    
    def test_default_config_creation(self):
        """Test creation of default configuration"""
        config = AppConfig()
        assert config.detector.camera_index == 0
        assert config.arm.safe_z == 100.0
        assert config.game.difficulty == 5

    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = AppConfig()
        config.game.difficulty = 15  # Invalid: max is 10
        
        with pytest.raises(ValueError):
            validate_config(config)

    @pytest.mark.parametrize("difficulty,expected_prob", [
        (1, 0.05), (5, 0.25), (10, 0.95)
    ])
    def test_difficulty_scaling(self, difficulty, expected_prob):
        """Test AI difficulty scaling"""
        config = AppConfig()
        config.game.difficulty = difficulty
        prob = calculate_minimax_probability(config.game.difficulty)
        assert abs(prob - expected_prob) < 0.1
```

#### Game Logic Testing (`test_game_logic_pytest.py`)
```python
class TestGameLogic:
    """Test core game logic functions"""
    
    def test_create_board(self):
        """Test board creation"""
        board = create_board()
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)
        assert all(cell == EMPTY for row in board for cell in row)

    def test_valid_moves(self):
        """Test move validation"""
        board = create_board()
        assert is_valid_move(board, 0, 0) is True
        assert is_valid_move(board, 3, 3) is False  # Out of bounds
        
        board[1][1] = 'X'
        assert is_valid_move(board, 1, 1) is False  # Occupied

    def test_winner_detection(self):
        """Test winner detection"""
        board = [
            ['X', 'X', 'X'],
            [' ', 'O', ' '],
            ['O', ' ', 'O']
        ]
        assert check_winner(board) == 'X'

    def test_winning_line_detection(self):
        """Test winning line coordinates"""
        board = [
            ['X', 'X', 'X'],
            [' ', 'O', ' '],
            ['O', ' ', 'O']
        ]
        winning_line = get_winning_line(board)
        assert winning_line == [(0, 0), (0, 1), (0, 2)]
```

#### AI Strategy Testing (`test_strategy_pytest.py`)
```python
class TestMinimaxStrategy:
    """Test AI strategy implementation"""
    
    def test_winning_move_detection(self):
        """Test AI finds winning moves"""
        strategy = MinimaxStrategy()
        board = [
            ['X', 'X', ' '],
            ['O', 'O', ' '],
            [' ', ' ', ' ']
        ]
        move = strategy.get_best_move(board, 'X')
        assert move == (0, 2)  # Winning move

    def test_blocking_move_detection(self):
        """Test AI blocks opponent wins"""
        strategy = MinimaxStrategy()
        board = [
            ['O', 'O', ' '],
            ['X', ' ', ' '],
            [' ', ' ', ' ']
        ]
        move = strategy.get_best_move(board, 'X')
        assert move == (0, 2)  # Block opponent win

    @pytest.mark.parametrize("difficulty,iterations", [
        (1, 100), (5, 100), (10, 100)
    ])
    def test_strategy_selection_distribution(self, difficulty, iterations):
        """Test strategy selection follows expected distribution"""
        selector = BernoulliStrategySelector(difficulty)
        board = create_board()
        
        minimax_count = 0
        for _ in range(iterations):
            strategy = selector.select_strategy(board)
            if strategy == 'minimax':
                minimax_count += 1
        
        expected_ratio = calculate_minimax_probability(difficulty)
        actual_ratio = minimax_count / iterations
        assert abs(actual_ratio - expected_ratio) < 0.1
```

### 2. Detection Pipeline Tests

#### Game Detector Testing (`test_game_detector_pytest.py`)
```python
class TestGameDetector:
    """Test main detection pipeline"""
    
    @pytest.fixture
    def mock_yolo_models(self):
        """Mock YOLO models for testing"""
        detect_model = Mock()
        pose_model = Mock()
        detect_model.to.return_value = detect_model
        pose_model.to.return_value = pose_model
        return detect_model, pose_model

    def test_frame_processing_pipeline(self, game_detector):
        """Test complete frame processing"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_time = time.time()
        
        # Mock detector responses
        game_detector.symbol_detector.detect_symbols.return_value = (frame, [])
        game_detector.grid_detector.detect_grid.return_value = (frame, [])
        game_detector.grid_detector.sort_grid_points.return_value = []
        game_detector.grid_detector.is_valid_grid.return_value = False
        
        result_frame, result_state = game_detector.process_frame(frame, frame_time)
        
        assert result_frame is not None
        assert result_state is not None

    def test_device_selection(self, mock_config):
        """Test proper device selection for models"""
        with patch('torch.cuda.is_available', return_value=True):
            detector = GameDetector(config=mock_config)
            assert detector.device == 'cuda'
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            detector = GameDetector(config=mock_config)
            assert detector.device == 'mps'
```

#### Grid Detector Testing (`test_grid_detector_pytest.py`)
```python
class TestGridDetector:
    """Test grid detection algorithms"""
    
    def test_grid_point_sorting(self, grid_detector):
        """Test grid point sorting algorithm"""
        # Unsorted grid points
        keypoints = np.array([
            [300, 300], [100, 100], [400, 400], [200, 200],
            [400, 300], [100, 200], [300, 400], [200, 100],
            [400, 200], [100, 300], [300, 200], [200, 300],
            [400, 100], [100, 400], [300, 100], [200, 400]
        ], dtype=np.float32)
        
        sorted_keypoints = grid_detector.sort_grid_points(keypoints)
        
        # Should be sorted row by row, left to right
        expected_order = np.array([
            [100, 100], [200, 100], [300, 100], [400, 100],  # Row 1
            [100, 200], [200, 200], [300, 200], [400, 200],  # Row 2
            [100, 300], [200, 300], [300, 300], [400, 300],  # Row 3
            [100, 400], [200, 400], [300, 400], [400, 400]   # Row 4
        ], dtype=np.float32)
        
        assert np.array_equal(sorted_keypoints, expected_order)

    def test_homography_computation(self, grid_detector):
        """Test homography matrix computation"""
        keypoints = np.array([
            [100, 100], [200, 100], [300, 100], [400, 100],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0], [0, 0]
        ], dtype=np.float32)
        
        with patch('cv2.findHomography') as mock_find_homography:
            mock_homography = np.eye(3, dtype=np.float32)
            mock_find_homography.return_value = (mock_homography, None)
            
            homography = grid_detector.compute_homography(keypoints)
            
            assert homography is not None
            assert np.array_equal(homography, mock_homography)

    def test_grid_status_tracking(self, grid_detector):
        """Test grid status change detection"""
        current_time = time.time()
        
        # First valid detection
        changed = grid_detector.update_grid_status(True, current_time)
        assert changed is True
        
        # Continued valid detection
        changed = grid_detector.update_grid_status(True, current_time + 1.0)
        assert changed is False
        
        # Invalid detection after timeout
        changed = grid_detector.update_grid_status(False, current_time + 3.0)
        assert changed is True
```

### 3. Hardware Controller Tests

#### Robot Controller Testing (`test_arm_movement_controller_pytest.py`)
```python
class TestArmMovementController:
    """Test robot arm control"""
    
    def test_coordinate_transformation(self, arm_controller):
        """Test camera-to-robot coordinate transformation"""
        # Mock game state and calibration
        game_state_obj = Mock()
        game_state_obj.get_cell_center_uv.return_value = (500, 400)
        
        arm_controller.main_window.camera_controller._get_detection_data = Mock(
            return_value=(None, game_state_obj)
        )
        
        calibration_data = {
            "perspective_transform_matrix_xy_to_uv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
        arm_controller.main_window.camera_controller.get_calibration_data = Mock(
            return_value=calibration_data
        )
        
        result = arm_controller._get_cell_coordinates_from_yolo(1, 1)
        
        assert result is not None
        assert len(result) == 2

    def test_symbol_drawing(self, arm_controller):
        """Test symbol drawing operations"""
        arm_controller._get_cell_coordinates_from_yolo = Mock(return_value=(100, 50))
        arm_controller.arm_controller.draw_o = Mock(return_value=True)
        
        result = arm_controller.draw_ai_symbol(1, 1, 'O')
        
        assert result is True
        arm_controller.arm_controller.draw_o.assert_called_once_with(
            center_x=100, center_y=50, radius=7.5, speed=500
        )

    def test_safety_validation(self, arm_controller):
        """Test movement safety checks"""
        arm_controller.arm_thread.connected = False
        
        with pytest.raises(RuntimeError, match="robotic arm is not available"):
            arm_controller.draw_ai_symbol(1, 1, 'X')
```

#### Camera Controller Testing (`test_camera_controller_pytest.py`)
```python
class TestCameraController:
    """Test camera management"""
    
    def test_camera_startup_shutdown(self, camera_controller):
        """Test camera lifecycle management"""
        camera_controller.start()
        camera_controller.camera_thread.start.assert_called_once()
        
        camera_controller.stop()
        camera_controller.camera_thread.stop.assert_called_once()
        camera_controller.camera_thread.wait.assert_called_once()

    def test_grid_warning_handling(self, camera_controller):
        """Test grid visibility warning system"""
        game_state = Mock()
        game_state.is_physical_grid_valid.return_value = False
        game_state._grid_points = [Mock()] * 10  # Incomplete grid
        
        camera_controller._handle_grid_warnings(game_state)
        
        assert camera_controller.grid_warning_active is True

    @patch('cv2.VideoCapture')
    def test_direct_camera_setup(self, mock_cv2_capture, camera_controller):
        """Test direct camera configuration"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cv2_capture.return_value = mock_cap
        
        result = camera_controller.setup_camera_direct(camera_index=0)
        
        assert result is True
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## Mocking Strategies

### PyQt5 Signal Testing

```python
def test_signal_emissions(self, game_controller):
    """Test that signals are properly emitted"""
    with patch.object(game_controller, 'status_changed') as mock_signal:
        game_controller.emit_status("Test status")
        mock_signal.emit.assert_called_once_with("Test status")
```

### Hardware Abstraction

```python
@pytest.fixture
def mock_hardware_setup():
    """Complete hardware mocking setup"""
    with patch('cv2.VideoCapture') as mock_camera, \
         patch('pyuarm.SwiftAPI') as mock_robot, \
         patch('torch.cuda.is_available', return_value=False):
        
        # Configure camera mock
        mock_camera_instance = Mock()
        mock_camera_instance.isOpened.return_value = True
        mock_camera.return_value = mock_camera_instance
        
        # Configure robot mock
        mock_robot_instance = Mock()
        mock_robot_instance.connected = True
        mock_robot.return_value = mock_robot_instance
        
        yield {
            'camera': mock_camera_instance,
            'robot': mock_robot_instance
        }
```

### Model Mocking

```python
@pytest.fixture
def mock_yolo_detection():
    """Mock YOLO model detection results"""
    with patch('ultralytics.YOLO') as mock_yolo:
        mock_model = Mock()
        
        # Mock detection results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = Mock()
        mock_result.boxes.conf = Mock()
        mock_result.boxes.cls = Mock()
        
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        yield mock_model
```

## Performance Testing

### Load Testing

```python
@pytest.mark.performance
def test_detection_performance():
    """Test detection pipeline performance"""
    detector = GameDetector(config=test_config)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    for _ in range(100):
        detector.process_frame(frame, time.time())
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.5  # Should process in under 500ms
```

### Memory Testing

```python
@pytest.mark.memory
def test_memory_usage():
    """Test memory usage stays within bounds"""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    
    # Run detection loop
    for _ in range(1000):
        # Simulate frame processing
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        process_frame(frame)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

## Coverage Analysis

### Current Coverage Status

```
app/core/detector_constants.py: 100%
app/main/grid_detector.py: 98%
app/core/config.py: 96%
app/core/utils.py: 96% 
app/main/game_utils.py: 95%
app/main/game_logic.py: 84%
app/core/game_state.py: 85%
app/core/strategy.py: 88%
```

### Coverage Goals

- **Critical modules**: 95%+ coverage required
- **Utility modules**: 90%+ coverage required  
- **GUI modules**: Excluded from coverage (mocked)
- **Hardware interfaces**: Mocked, logic tested

### Uncovered Code Analysis

```bash
# Generate detailed coverage report
pytest --cov=app --cov-report=html --cov-fail-under=80

# View uncovered lines
pytest --cov=app --cov-report=term-missing
```

## Continuous Integration

### Test Pipeline

```yaml
# GitHub Actions example
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml --cov-fail-under=80
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Quality Gates

- **Coverage threshold**: 80% minimum
- **Test success rate**: 100% required
- **Performance tests**: Must pass timing requirements
- **Linting**: pylint score > 8.0

## Test Data Management

### Fixture Data

```python
@pytest.fixture
def sample_board_states():
    """Sample board states for testing"""
    return {
        'empty': create_board(),
        'x_wins': [['X', 'X', 'X'], [' ', 'O', ' '], ['O', ' ', 'O']],
        'o_wins': [['O', 'X', 'X'], ['O', 'X', ' '], ['O', ' ', 'X']],
        'tie': [['X', 'O', 'X'], ['O', 'X', 'O'], ['O', 'X', 'O']]
    }

@pytest.fixture
def sample_detection_data():
    """Sample computer vision detection data"""
    return {
        'grid_points': np.array([[100, 100], [200, 100], [300, 100]], dtype=np.float32),
        'symbols': [
            {'class': 'X', 'confidence': 0.95, 'bbox': [150, 150, 200, 200]},
            {'class': 'O', 'confidence': 0.87, 'bbox': [250, 250, 300, 300]}
        ]
    }
```

### Test Image Generation

```python
def generate_test_frame(width=640, height=480, add_noise=True):
    """Generate synthetic test frames"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if add_noise:
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
    
    return frame
```

## Debugging Tests

### Test Debugging

```bash
# Run single test with debugging
pytest tests/test_game_logic_pytest.py::TestGameLogic::test_winner_detection -v -s

# Drop into debugger on failure
pytest --pdb tests/test_strategy_pytest.py

# Run only failed tests from last run
pytest --lf
```

### Mock Debugging

```python
def test_with_mock_debugging():
    """Example of debugging mocked calls"""
    with patch('app.main.game_controller.GameController.start_new_game') as mock_start:
        controller = GameController()
        controller.start_new_game()
        
        # Debug mock calls
        print(f"Mock called: {mock_start.called}")
        print(f"Call count: {mock_start.call_count}")
        print(f"Call args: {mock_start.call_args}")
```

---

This testing framework ensures reliable, maintainable code with comprehensive coverage while enabling development without physical hardware dependencies.