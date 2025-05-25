# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run the main application:
```bash
python -m app.main.main_pyqt
```

Alternative entry point:
```bash
python run.py
```

With command line options:
```bash
python -m app.main.main_pyqt --camera 1 --debug --difficulty 7
```

Calibrate robot arm:
```bash
python -m app.calibration.calibration
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Lint code:
```bash
pylint app/
# Or use the wrapper script:
./pylint.sh
# For score only:
./pylint.sh --score=y
```

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=app --cov-report=term-missing
```

Run tests with HTML coverage report:
```bash
pytest --cov=app --cov-report=html
```

Run specific test modules:
```bash
pytest tests/test_game_logic_pytest.py -v
pytest tests/test_strategy_fixed.py -v
pytest tests/test_config_pytest.py -v
pytest tests/test_utils_pytest.py -v
pytest tests/test_game_detector_simple.py -v
pytest tests/test_grid_detector_pytest.py -v
pytest tests/test_arm_controller_simple.py -v
pytest tests/test_camera_controller_simple.py -v
```

Run tests for specific components:
```bash
pytest tests/ -k "detector" -v         # Detection-related tests
pytest tests/ -k "game_logic" -v       # Game logic tests
pytest tests/ -k "strategy" -v         # AI strategy tests
pytest tests/ -k "config" -v           # Configuration tests
```

Run tests with verbose output and coverage:
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

Debug scripts for computer vision development:
```bash
python debug_coordinates.py          # Test coordinate transformation
python debug_grid_mapping.py         # Debug grid point detection
python debug_resolution_mismatch.py  # Fix camera resolution issues
python debug_transformation_log.py   # Log homography calculations
python test_resolution_scaling.py    # Test camera scaling
```

Utility scripts:
```bash
python scripts/utils/symbol_tester.py     # Test symbol detection
python scripts/utils/webcam_recorder.py   # Record camera feed
python scripts/utils/save_frames_on_key.py # Save frames for training
python scripts/utils/annotate_frames.py   # Annotate detection data
```

## Architecture

This is a robotic TicTacToe application that combines computer vision, AI strategy, and robotic arm control for playing physical TicTacToe games with a uArm Swift Pro robot.

### Signal-Based Communication Architecture

The application uses PyQt5 signals for decoupled communication between components:
- `GameController.status_changed` → `StatusManager.update_status()` for UI updates
- `CameraController.game_state_updated` → `GameController.handle_detected_game_state()` for vision pipeline
- `CameraController.grid_incomplete` → status notifications for grid visibility

### State Management Pattern

**Authoritative Board State**: `GameController.authoritative_board` is the single source of truth, updated only from camera detection. GUI board widget reflects this state but never modifies it directly.

**Turn Coordination**: Game turns are managed through state flags:
- `arm_move_in_progress`: Robot is physically moving
- `waiting_for_detection`: Waiting for camera to confirm move
- `current_turn`: Whose turn it is (human vs AI)

### Computer Vision Pipeline

**Two-Stage Detection**:
1. **Grid Detection**: 16 keypoints forming 4x4 grid using pose estimation model
2. **Symbol Detection**: X/O symbols within grid cells using object detection model

**Coordinate Transformation Flow**:
```
Camera pixels → Homography → Normalized grid space → Cell mapping
```

**Robust Grid Sorting**: Handles rotation and perspective by sorting detected points into canonical 4x4 grid order before homography calculation.

### Threading and Real-Time Processing

- **Detection Thread** (`app/core/detection_thread.py`): Runs vision pipeline at ~2 FPS
- **Arm Thread** (`app/core/arm_thread.py`): Handles robot commands asynchronously
- **Main Thread**: PyQt5 GUI with 100ms update timer

**Critical Race Condition Handling**: Detection results are queued and processed in main thread to avoid GUI updates from worker threads.

### Error Recovery and Retry Logic

**Detection Timeout Handling**: If robot move isn't detected within `max_detection_wait_time`, system retries up to `max_retry_count` times before falling back to human turn.

**Grid Visibility Monitoring**: System pauses game when grid isn't fully visible, showing warning notification until all 16 grid points are detected.

### Configuration System

`AppConfig` dataclass with nested configs:
- `DetectorConfig`: Camera settings, model paths, confidence thresholds
- `ArmConfig`: Movement speeds, heights, calibration matrices
- `GameConfig`: Difficulty, timing, UI preferences

Hand-eye calibration stored in `app/calibration/hand_eye_calibration.json` as 4x4 transformation matrix.

### AI Strategy Architecture

`BernoulliStrategySelector` uses difficulty parameter (1-10) to blend random moves with minimax strategy. Higher difficulty = more optimal play.

**Strategy Selection**: Each move decides between random and minimax based on Bernoulli probability derived from difficulty setting.

### Testing Architecture

**Framework**: Pure pytest (no unittest dependencies)

**Coverage**: 80%+ comprehensive test coverage achieved for critical modules

**Test Structure**:
- `tests/conftest.py`: PyQt5 mocking and test configuration  
- `tests/test_*_pytest.py`: Pure pytest test modules
- Mock hardware dependencies (camera, robotic arm) for testing without physical devices

**Key Test Modules**:
- **Core Logic**: `test_game_logic_pytest.py`, `test_strategy_pytest.py`, `test_config_pytest.py`
- **Detection Pipeline**: `test_game_detector_pytest.py`, `test_grid_detector_pytest.py`, `test_detector_constants.py`
- **Hardware Controllers**: `test_arm_movement_controller_pytest.py`, `test_camera_controller_pytest.py`
- **Utilities**: `test_utils_pytest.py`, `test_path_utils_pytest.py`, `test_game_state_pytest.py`

**Mocking Strategy**:
- PyQt5 GUI components mocked to avoid display dependencies
- Hardware interfaces (uArm, camera) mocked for testing without physical devices
- YOLO models mocked for vision pipeline testing
- Threading components safely mocked for deterministic testing

**Test Features**:
- Parametrized tests for comprehensive scenario coverage
- Edge case testing including error conditions and hardware failures
- Fixture-based test organization for code reuse
- Mock signal emissions for PyQt5 signal testing

**Coverage Highlights**:
- `app/core/detector_constants.py`: 100%
- `app/main/grid_detector.py`: 98%
- `app/core/config.py`: 96%
- `app/core/utils.py`: 96%
- `app/main/game_utils.py`: 95%

**Testing Notes**:
- DŮLEŽITÉ: Používej pouze pytest! Nepoužívej unittest!
- Všechny testy jsou v pytest formátu bez unittest závislostí
- PyQt komponenty se netestují - jsou mockované
- Hardware závislosti jsou abstrahovány přes mocking
