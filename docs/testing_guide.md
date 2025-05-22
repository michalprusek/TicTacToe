# TicTacToe Testing Guide

This guide explains the unified testing approach implemented for the TicTacToe project, providing examples and best practices for writing maintainable, DRY tests.

> **Important**: For testing PyQt components safely without segmentation faults, please refer to the [PyQt Testing Guide](pyqt_testing_guide.md).

## Table of Contents

- [Introduction](#introduction)
- [Testing Architecture](#testing-architecture)
- [Common Fixtures and Utilities](#common-fixtures-and-utilities)
- [Utility Classes](#utility-classes)
- [Writing New Tests](#writing-new-tests)
- [Mocking Hardware Dependencies](#mocking-hardware-dependencies)
- [Running Tests](#running-tests)
- [Best Practices](#best-practices)

## Introduction

The TicTacToe application has a large test suite covering various aspects of the system. To reduce code duplication and improve maintainability, we've implemented a unified testing approach that centralizes common mocking code, test fixtures, and utility functions.

## Testing Architecture

The testing architecture follows these principles:

1. **Centralized Mocking**: Common mock objects are defined once and reused
2. **Unified Test Helpers**: Helper functions for common testing patterns
3. **Standard Test Setup**: Consistent test setup and teardown procedures
4. **DRY Test Cases**: Test code reuse to avoid duplication

## Common Fixtures and Utilities

### conftest_common.py

This module provides common test fixtures and utility classes shared across all test files:

1. **Common Fixtures**:
   - `qt_app`: QApplication instance for PyQt tests
   - `mock_tic_tac_toe_app`: Standard MockTicTacToeApp instance 
   - `patched_tic_tac_toe_app`: Patched TicTacToeApp with mocked attributes
   - Other fixtures for common test components (frames, detectors, etc.)

2. **Utility Classes**:
   - `PyQtGuiTestCaseBase`: Base class with methods for test setup
   - `AssertionUtils`: Common assertion functions
   - `GameEndCheckTestUtils`: Game end checking test utilities
   - `DrawingTestUtils`: Drawing functionality test utilities
   - `CameraTestUtils`: Camera-related test utilities
   - `UIComponentTestUtils`: UI component test utilities

Example:

```python
from tests.conftest_common import (
    PyQtGuiTestCaseBase, 
    mock_tic_tac_toe_app, 
    AssertionUtils
)

class TestMyComponent:
    def test_status_message(self, mock_tic_tac_toe_app):
        # Test implementation
        mock_tic_tac_toe_app.your_method()
        
        # Use assertion utilities
        AssertionUtils.assert_status_message_once(
            mock_tic_tac_toe_app, 
            "Expected message"
        )
```

### Unified Test Files

We have several unified test files that serve as examples of using the unified test approach:

1. **test_pyqt_gui_unified_events.py** - Consolidates tests for event handling
2. **test_pyqt_gui_unified_game_end.py** - Consolidates tests for game end checking

## Utility Classes

### PyQtGuiTestCaseBase

Base class for PyQt GUI test cases, providing common methods for setup:

```python
def test_with_test_case_base():
    # Create app and patches
    app, patches = PyQtGuiTestCaseBase.create_test_app()
    
    # Test logic here
    
    # Clean up patches
    for p in patches:
        p.stop()
```

### AssertionUtils

Common assertion methods for testing:

```python
def test_with_assertion_utils(mock_tic_tac_toe_app):
    # Test your function
    mock_tic_tac_toe_app.your_function()
    
    # Assertions
    AssertionUtils.assert_status_message_once(mock_tic_tac_toe_app, "Expected message")
    AssertionUtils.assert_arm_not_connected_message(mock_tic_tac_toe_app)
    AssertionUtils.assert_drawing_failed_message(mock_tic_tac_toe_app)
```

### GameEndCheckTestUtils

Utility methods for testing game end check functionality:

```python
def test_game_end_scenarios(mock_tic_tac_toe_app):
    # Test with no winner
    GameEndCheckTestUtils.test_check_game_end_no_winner(mock_tic_tac_toe_app)

    # Test with human player winning
    GameEndCheckTestUtils.test_check_game_end_human_wins(mock_tic_tac_toe_app)

    # Test with AI player winning
    GameEndCheckTestUtils.test_check_game_end_ai_wins(mock_tic_tac_toe_app)

    # Test with a tie
    GameEndCheckTestUtils.test_check_game_end_tie(mock_tic_tac_toe_app)
```

### DrawingTestUtils

Utility methods for testing drawing functionality:

```python
def test_drawing_functionality(mock_tic_tac_toe_app):
    # Prepare drawing X test
    DrawingTestUtils.prepare_draw_x_test(mock_tic_tac_toe_app)
    DrawingTestUtils.prepare_arm_thread(mock_tic_tac_toe_app)
    
    # Call the method to test
    mock_tic_tac_toe_app.draw_ai_symbol(1, 1)
    
    # Verify X was drawn correctly
    DrawingTestUtils.verify_draw_x(mock_tic_tac_toe_app, 200, 0, 30)
    DrawingTestUtils.verify_move_to_neutral(mock_tic_tac_toe_app)
```

### CameraTestUtils

Utility methods for testing camera functionality:

```python
def test_camera_functionality(mock_tic_tac_toe_app):
    # Prepare camera thread
    CameraTestUtils.prepare_mock_camera_thread(mock_tic_tac_toe_app)
    
    # Create a sample frame
    frame = CameraTestUtils.create_sample_frame()
    
    # Process the frame
    mock_tic_tac_toe_app.camera_thread.frame_ready.emit(frame)
    
    # Verify frame was processed
    CameraTestUtils.verify_frame_processed(mock_tic_tac_toe_app.camera_thread, frame)
```

### UIComponentTestUtils

Utility methods for testing UI components:

```python
def test_ui_components(mock_tic_tac_toe_app):
    # Prepare UI components
    UIComponentTestUtils.prepare_ui_components(mock_tic_tac_toe_app)
    
    # Set up game state
    UIComponentTestUtils.prepare_game_state(
        mock_tic_tac_toe_app,
        human=game_logic.PLAYER_X,
        ai=game_logic.PLAYER_O,
        current_turn=game_logic.PLAYER_X
    )
    
    # Call the method to test
    mock_tic_tac_toe_app.your_method()
    
    # Now use other utilities for assertions
    AssertionUtils.assert_status_message_once(mock_tic_tac_toe_app, "Expected message")
```

## Writing New Tests

### Basic Test Structure

```python
import pytest
from unittest.mock import MagicMock
from tests.conftest_common import PyQtGuiTestCaseBase, AssertionUtils

class TestYourComponent:
    def setup_method(self):
        # Create app and patches
        self.app, self.patches = PyQtGuiTestCaseBase.create_test_app()
        self.tic_tac_toe_app = self.app
        
        # Add any additional mocks specific to your test
        self.tic_tac_toe_app.some_component = MagicMock()
        
    def teardown_method(self):
        # Clean up patches
        for p in self.patches:
            p.stop()
            
    def test_your_functionality(self):
        # Test implementation
        self.tic_tac_toe_app.your_method()
        
        # Assertions using utility class
        AssertionUtils.assert_status_message_once(
            self.tic_tac_toe_app, 
            "Expected message"
        )
```

### Using Pytest Fixtures

```python
import pytest
from tests.conftest_common import mock_tic_tac_toe_app, AssertionUtils

def test_with_fixtures(mock_tic_tac_toe_app):
    # Test implementation
    mock_tic_tac_toe_app.your_method()
    
    # Assertions using utility class
    AssertionUtils.assert_status_message_once(
        mock_tic_tac_toe_app, 
        "Expected message"
    )
```

## Mocking Hardware Dependencies

### Robotic Arm Mocking

```python
from tests.conftest_common import DrawingTestUtils

def test_arm_functionality(mock_tic_tac_toe_app):
    # Set up arm controller mock
    DrawingTestUtils.prepare_arm_controller(mock_tic_tac_toe_app)
    
    # Test arm-related functionality
    result = mock_tic_tac_toe_app.move_to_neutral_position()
    
    # Verify the result
    assert result is True
    mock_tic_tac_toe_app.arm_controller.go_to_position.assert_called_once()
    
    # Or use assertion utilities
    AssertionUtils.assert_neutral_position_success(mock_tic_tac_toe_app)
```

### Camera Mocking

```python
from tests.conftest_common import CameraTestUtils

def test_camera_functionality(mock_tic_tac_toe_app):
    # Set up camera mock
    CameraTestUtils.prepare_mock_camera_thread(mock_tic_tac_toe_app)
    
    # Create a detector thread mock
    mock_detection_thread = CameraTestUtils.prepare_mock_detection_thread()
    mock_tic_tac_toe_app.camera_thread.detector = mock_detection_thread
    
    # Test camera-related functionality
    result = mock_tic_tac_toe_app.get_cell_coordinates_from_yolo(1, 1)
    
    # Verify the result
    assert result is not None
```

## Running Tests

### Running All Tests

```bash
python run_tests.py
```

### Running with Code Coverage

```bash
python run_tests.py --coverage
```

### Running Specific Tests

```bash
python run_tests.py --tests tests/unit/test_pyqt_gui_unified_events.py
```

### Running Tests with Timeout

```bash
python run_tests_with_timeout.py --test-file tests/unit/test_pyqt_gui_unified_events.py --timeout 30
```

## Best Practices

1. **Use the Common Fixtures and Utilities**:
   - Use fixtures from `conftest_common.py` like `qt_app` and `mock_tic_tac_toe_app`
   - Use utility classes like `AssertionUtils` and `DrawingTestUtils`

2. **Follow the DRY Principle**:
   - Don't duplicate test setup or mock creation
   - Use helper modules for common testing patterns

3. **One Test, One Responsibility**:
   - Each test should focus on testing one specific behavior
   - Split complex behaviors into multiple tests

4. **Properly Mock Dependencies**:
   - Always mock hardware (arm, camera)
   - Mock time-consuming operations
   - Use appropriate mocks for PyQt components

5. **Clean up Resources**:
   - Always stop patches in teardown methods
   - Clean up any created resources

6. **Name Tests Clearly**:
   - Use descriptive test names that indicate what is being tested
   - Follow the pattern `test_method_condition_expectation`

7. **Document Test Expectations**:
   - Use clear docstrings to explain what the test verifies
   - Include preconditions and expected outcomes

8. **Use Pytest Style**:
   - Prefer assert statements over unittest assertions
   - Use pytest fixtures where appropriate

9. **Keep Tests Independent**:
   - Tests should not depend on the execution of other tests
   - Each test should set up its own test environment

10. **Handle Edge Cases**:
    - Test error conditions and edge cases
    - Test invalid inputs and exceptional conditions

## Extending the Testing Framework

When adding new utilities to the unified helper modules:

1. Place utility functions in the appropriate class in `conftest_common.py`
2. Follow existing patterns for argument naming and structure
3. Add detailed docstrings explaining the purpose and usage
4. Create unit tests for the new utilities
5. Update this documentation to reflect the changes

---

By following this guide, you'll be able to maintain the testing codebase effectively and continue the efforts to reduce code duplication and improve testing quality.