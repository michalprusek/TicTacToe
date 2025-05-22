# Robot TicTacToe

An application for playing Tic Tac Toe with a uArm Swift Pro robotic arm. The system uses computer vision to detect the game board and symbols, artificial intelligence for game strategy, and a robotic arm for physically placing symbols on the game board.

## Features

- Game board and symbol detection using computer vision
- Control of the uArm Swift Pro robotic arm
- Adjustable AI opponent difficulty
- Graphical user interface built with PyQt5
- Support for debugging and detection visualization

## Requirements

- Python 3.6+
- uArm Swift Pro with firmware 4.0+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/robot-tictactoe.git
   cd robot-tictactoe
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install uArm Python SDK:
   ```
   cd uArm-Python-SDK
   python setup.py install
   cd ..
   ```

## Running the Application

Launch the main application with GUI:

```
python main_pyqt.py
```

### Command Line Parameters

- `--camera INDEX` - Camera index to use (default: 0)
- `--debug` - Enable debug mode with additional logging and visualization
- `--difficulty LEVEL` - Initial difficulty level (0-10, default: 5)

Example:
```
python main_pyqt.py --camera 1 --debug --difficulty 7
```

## Testing

Run tests:

```
python run_tests.py
```

Or with code coverage:

```
python run_tests.py --coverage
```

## Project Structure

- `app/` - Main application package
  - `core/` - Core components and configuration
- `pyqt_gui.py` - PyQt5 GUI implementation
- `game_detector.py` - Game board and symbol detection
- `arm_controller.py` - Robotic arm control
- `game_logic.py` - Game logic and AI
- `tests/` - Tests

## Robotic Arm Calibration

Before first use, you need to calibrate the robotic arm:

1. Make sure the arm is connected and powered on
2. Run the calibration tool:
   ```
   python calibration.py
   ```
3. Follow the on-screen instructions

## License

MIT
