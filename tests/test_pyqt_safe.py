"""
Safe PyQt testing example using pytest-qt.
This demonstrates how to test PyQt applications without segmentation faults.
"""
import pytest
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel

from unittest.mock import MagicMock
from app.main import game_logic


class SimplePyQtWindow(QMainWindow):
    """
    A simple PyQt window for testing purposes.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple PyQt Test Window")
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Add components
        self.label = QLabel("Initial Text")
        self.button = QPushButton("Click Me")
        self.debug_button = QPushButton("Debug Mode")
        
        # Add to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.debug_button)
        
        # Connect signals
        self.button.clicked.connect(self.handle_button_click)
        self.debug_button.clicked.connect(self.handle_debug_toggle)
        
        # State
        self.debug_mode = False
        self.click_count = 0
        
    def handle_button_click(self):
        """Handle button click event."""
        self.click_count += 1
        self.label.setText(f"Button clicked {self.click_count} times")
        
    def handle_debug_toggle(self):
        """Toggle debug mode."""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.debug_button.setText("Disable Debug")
        else:
            self.debug_button.setText("Enable Debug")


class TestPyQtSafely:
    """Test class for safe PyQt testing."""
    
    def test_button_click(self, qtbot):
        """Test button click functionality."""
        # Create window
        window = SimplePyQtWindow()
        
        # Add it to qtbot
        qtbot.addWidget(window)
        
        # Verify initial state
        assert window.label.text() == "Initial Text"
        assert window.click_count == 0
        
        # Click the button using qtbot
        qtbot.mouseClick(window.button, Qt.LeftButton)
        
        # Verify state changed
        assert window.label.text() == "Button clicked 1 times"
        assert window.click_count == 1
        
    def test_debug_mode_toggle(self, qtbot):
        """Test debug mode toggle functionality."""
        # Create window
        window = SimplePyQtWindow()
        
        # Add it to qtbot
        qtbot.addWidget(window)
        
        # Verify initial state
        assert not window.debug_mode
        assert window.debug_button.text() == "Debug Mode"
        
        # Click the debug button using qtbot
        qtbot.mouseClick(window.debug_button, Qt.LeftButton)
        
        # Verify debug mode is enabled
        assert window.debug_mode
        assert window.debug_button.text() == "Disable Debug"
        
        # Click again to disable
        qtbot.mouseClick(window.debug_button, Qt.LeftButton)
        
        # Verify debug mode is disabled again
        assert not window.debug_mode
        assert window.debug_button.text() == "Enable Debug"


class MockTicTacToeAppSafe:
    """
    Safe mock implementation of TicTacToeApp for testing.
    This implementation doesn't inherit from QMainWindow to avoid segfaults.
    """
    
    def __init__(self, debug_mode=False, camera_index=0, difficulty=0.5):
        """Initialize the safe mock app."""
        self.debug_mode = debug_mode
        self.camera_index = camera_index
        self.difficulty = difficulty
        self.human_player = None
        self.ai_player = None
        self.current_turn = None
        self.game_over = False
        self.winner = None
        
        # Mock components
        self.board_widget = MagicMock()
        self.board_widget.board = game_logic.create_board()
        self.status_label = MagicMock()
        self.debug_button = MagicMock()
        self.strategy_selector = MagicMock()
        
    def handle_debug_button_click(self):
        """Mock implementation of handle_debug_button_click."""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.debug_button.setText("Vypnout debug")
        else:
            self.debug_button.setText("Zapnout debug")
            
    def handle_difficulty_changed(self, value):
        """Mock implementation of handle_difficulty_changed."""
        self.difficulty = value / 100.0
        self.strategy_selector.set_difficulty(self.difficulty)


class TestEventHandlingSafely:
    """Test event handling safely without PyQt initialization issues."""
    
    def test_debug_button_click(self):
        """Test debug button functionality."""
        app = MockTicTacToeAppSafe(debug_mode=False)
        
        # Call the method
        app.handle_debug_button_click()
        
        # Verify state changes
        assert app.debug_mode is True
        app.debug_button.setText.assert_called_once_with("Vypnout debug")
        
    def test_difficulty_change(self):
        """Test difficulty change functionality."""
        app = MockTicTacToeAppSafe(difficulty=0.5)
        
        # Call the method
        app.handle_difficulty_changed(75)
        
        # Verify state changes
        assert app.difficulty == 0.75
        app.strategy_selector.set_difficulty.assert_called_once()


def test_with_qapp_directly(qtbot):
    """Test using the qtbot fixture directly."""
    # Create a simple button
    button = QPushButton("Test Button")
    button.setObjectName("testButton")
    button.clicked.connect(lambda: button.setText("Button Clicked"))
    
    # Add it to qtbot
    qtbot.addWidget(button)
    
    # Initial state
    assert button.text() == "Test Button"
    
    # Click the button
    qtbot.mouseClick(button, Qt.LeftButton)
    
    # Check button text changed
    assert button.text() == "Button Clicked"


def test_qtbot_wait_signal(qtbot):
    """Test waiting for signals with qtbot."""
    # Create a button that will emit a signal
    button = QPushButton("Wait Signal Test")
    
    # Add it to qtbot
    qtbot.addWidget(button)
    
    # Use a timer to click the button after a delay
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: button.click())
    
    # Setup signal waiting and start timer
    with qtbot.waitSignal(button.clicked, timeout=1000) as blocker:
        timer.start(200)  # Click after 200ms
    
    # If we get here, the signal was received
    assert blocker.signal_triggered, "Signal was not triggered!"