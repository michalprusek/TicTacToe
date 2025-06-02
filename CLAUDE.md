# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Running the Application
```bash
# Main application with PyQt5 GUI
python -m app.main.main_pyqt

# Alternative launcher
python run.py

# With options
python -m app.main.main_pyqt --camera 1 --debug --difficulty 7

# Robot calibration (requires uArm)
python -m app.calibration.calibration
```

### Testing and Quality Checks
```bash
# Run all tests with coverage
pytest

# Coverage with detailed report
pytest --cov=app --cov-report=term-missing
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_game_logic.py

# Run single test
pytest tests/test_game_logic.py::TestGameLogic::test_check_winner

# Linting
pylint app

# Complete quality check
pytest && pylint app
```

## Architecture Overview

### Signal-Based Communication Architecture
The application uses PyQt5 signals for thread-safe communication between components:

1. **Detection Pipeline**: `CameraController` → `DetectionThread` → `GameController`
   - Camera captures frames at 30 FPS
   - Detection processes at ~2 FPS using YOLO models
   - Game state updates via `game_state_updated` signal

2. **Control Flow**: `GameController` → `ArmMovementController` → `ArmThread`
   - AI moves trigger `ai_move_ready` signal
   - Robot movements execute asynchronously
   - Completion confirmed via `arm_move_completed` signal

3. **UI Updates**: All components → `MainGUI`/`BoardWidget`
   - Status updates via `status_changed` signal
   - Board state synchronized through `authoritative_board`

### Core Components

#### Detection System (`app/core/detection_thread.py`)
- Two-stage YOLO detection: grid pose + symbol detection
- Homography transformation for coordinate mapping
- Confidence thresholds: 0.75 for both models

#### Game State Management (`app/core/game_state.py`)
- Single source of truth for board state
- Coordinate transformation: camera → grid → robot
- Symbol-to-cell mapping with validation

#### Robot Control (`app/core/arm_thread.py`)
- Asynchronous command execution
- Safety validation and collision avoidance
- Command queuing with status feedback

### Critical Implementation Details

1. **Thread Safety**: All cross-thread communication uses Qt signals
2. **State Authority**: Only `GameController.authoritative_board` is trusted
3. **Turn Coordination**: Flags prevent race conditions during moves
4. **Error Recovery**: Retry logic with graceful degradation

### Configuration System
Hierarchical configuration in `app/core/config.py`:
- `AppConfig` → `GameDetectorConfig`, `ArmControllerConfig`, `GameConfig`
- Command-line arguments override defaults
- Calibration data stored separately

## Development Notes

### Model Files
- Grid detection: `weights/best_pose.pt` (YOLO pose estimation)
- Symbol detection: `weights/best_detection.pt` (YOLO object detection)

### Calibration Data
- Camera-robot calibration: `app/calibration/hand_eye_calibration.json`
- Run calibration tool before first use with robot

### Performance Targets
- Detection: ~2 FPS (configurable in config)
- UI updates: 100ms timer cycle
- Robot movements: Speed 500-1000 units

### Common Issues
- Camera index: Use `--camera N` to select different camera
- Robot connection: Ensure uArm Swift Pro is connected via USB
- Detection failures: Check lighting and grid visibility