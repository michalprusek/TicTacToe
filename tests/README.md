# TicTacToe Testing Framework

This document provides guidance on the testing framework used in the TicTacToe project, including the unified approach to mocking, test structure, and best practices.

## Unified Testing Approach

To eliminate code duplication and improve maintainability, the project uses a unified testing approach with shared helper modules:

### Key Components

1. **`pyqt_gui_unified_helper.py`**
   - Contains the centralized `MockTicTacToeApp` class for consistent mocking
   - Provides utility functions for testing common events and behaviors
   - Implements `PyQtGuiTestCase` for standardized test setup and teardown
   - Contains test utilities for common behaviors (`EventHandlingTestUtils`, `GameEndCheckTestUtils`)

2. **`test_pyqt_gui_unified_events.py`**
   - Consolidates duplicate event handling tests from across the codebase
   - Uses the shared helper components for consistent testing

3. **`test_pyqt_gui_unified_game_end.py`**
   - Consolidates duplicate game end check tests from across the codebase
   - Demonstrates using the `GameEndCheckTestUtils` for testing game end scenarios

## Using the Unified Testing Framework

### Creating a New Test Class

```python
import unittest
from tests.unit.pyqt_gui_unified_helper import PyQtGuiTestCase

class TestMyComponent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create app instance and patches
        self.app, self.patches = PyQtGuiTestCase.create_test_app()
        self.tic_tac_toe_app = self.app
        
        # Add additional mocks specific to your test
        self.tic_tac_toe_app.some_component = MagicMock()
        
    def tearDown(self):
        """Tear down test environment."""
        # Stop all patches
        for p in self.patches:
            p.stop()
            
    def test_my_feature(self):
        # Test implementation
        pass
```

### Using Test Utilities

#### Event Handling Test Utilities

The `EventHandlingTestUtils` class provides standardized tests for common events:

```python
from tests.unit.pyqt_gui_unified_helper import EventHandlingTestUtils

def test_debug_button(self):
    # Test enabling debug mode
    EventHandlingTestUtils.test_debug_button_enable(self.tic_tac_toe_app)

    # Test disabling debug mode
    EventHandlingTestUtils.test_debug_button_disable(self.tic_tac_toe_app)
```

#### Game End Check Test Utilities

The `GameEndCheckTestUtils` class provides standardized tests for game end scenarios:

```python
from tests.unit.pyqt_gui_unified_helper import GameEndCheckTestUtils

def test_game_end_scenarios(self):
    # Test with no winner
    GameEndCheckTestUtils.test_check_game_end_no_winner(self.tic_tac_toe_app)

    # Test with human player winning
    GameEndCheckTestUtils.test_check_game_end_human_wins(self.tic_tac_toe_app)

    # Test with AI player winning
    GameEndCheckTestUtils.test_check_game_end_ai_wins(self.tic_tac_toe_app)

    # Test with a tie
    GameEndCheckTestUtils.test_check_game_end_tie(self.tic_tac_toe_app)
```

## Mocking Hardware Dependencies

### Robotic Arm

The unified `MockTicTacToeApp` provides standardized mocking for the robotic arm:

```python
# Set up arm mocks if needed
self.tic_tac_toe_app.arm_controller = MagicMock()
self.tic_tac_toe_app.arm_controller.connected = True

# Test arm-related functionality
result = self.tic_tac_toe_app.move_to_neutral_position()
self.assertTrue(result)
self.tic_tac_toe_app.arm_controller.go_to_position.assert_called_once()
```

### Camera

The unified `MockTicTacToeApp` provides standardized mocking for the camera:

```python
# Set up camera mocks if needed
self.tic_tac_toe_app.camera_thread = MagicMock()
self.tic_tac_toe_app.camera_thread.detector = MagicMock()
self.tic_tac_toe_app.camera_thread.detector.game_state = MagicMock()

# Mock camera detection
self.tic_tac_toe_app.camera_thread.detector.game_state.is_valid.return_value = True
self.tic_tac_toe_app.camera_thread.detector.game_state.get_cell_center_uv.return_value = (320, 240)
```

## Best Practices

1. **Avoid duplicate tests**: Use the unified test utilities whenever possible
2. **Follow the DRY principle**: Don't repeat test setup code; use the shared helpers
3. **Test one thing per test method**: Each test should focus on a single functionality
4. **Use meaningful test names**: Name tests to clearly indicate what they're testing
5. **Keep tests independent**: Tests should not depend on the order of execution
6. **Keep tests fast**: Avoid unnecessary operations that slow down test execution
7. **Mock external dependencies**: Always mock hardware, network, and other external dependencies
8. **Use pytest style**: Avoid using unittest-specific assertions and fixtures

## Running Tests

```bash
# Run all tests
python run_tests.py

# Run tests with code coverage
python run_tests.py --coverage

# Run specific test files
python run_tests.py --tests tests/unit/test_pyqt_gui_unified_events.py

# Run tests with timeout
python run_tests_with_timeout.py --test-file tests/unit/test_pyqt_gui_unified_events.py --timeout 30
```

## Adding New Test Utilities

When adding new test utilities to the unified helper:

1. Place utility functions in the appropriate class (`EventHandlingTestUtils`, `GameEndCheckTestUtils`, `PyQtGuiTestCase`, etc.)
2. Follow existing patterns for argument naming and function structure
3. Add detailed docstrings explaining the purpose and usage of the utility
4. Create unit tests for the new utility functions
5. Update the documentation in `docs/testing_guide.md` and this README

## Legacy Tests

The project still contains legacy tests that haven't been migrated to the unified approach. When modifying these tests:

1. Consider migrating them to use the unified testing framework
2. If migrating is not feasible, ensure changes maintain compatibility
3. Document any special considerations for the legacy tests

---

This framework is designed to help maintain a clean, efficient, and DRY test codebase. By following these guidelines, we can ensure that our tests remain maintainable and valuable.