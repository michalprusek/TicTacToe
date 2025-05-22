"""
Test script to simulate detection of incomplete grid in TicTacToe application.
This script demonstrates the warning system when a grid is not fully detected.
"""
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication

from app.main.pyqt_gui import TicTacToeApp
from app.core.game_state import GameState


class MockGameState(GameState):
    """Mock GameState class that simulates incomplete grid detection"""
    
    def __init__(self, visible_points=8):
        """Initialize with specified number of visible points"""
        super().__init__()
        self._is_valid_grid = True
        self._board_state = [['', '', ''], ['', '', ''], ['', '', '']]
        
        # Create grid points where some might be all zeros (not detected)
        self._grid_points = np.ones((16, 2))
        # Set the visible point count (zeros are considered not visible)
        for i in range(visible_points, 16):
            self._grid_points[i] = np.zeros(2)


class MockDetectionThread:
    """Mock DetectionThread for testing incomplete grid detection"""
    
    def __init__(self, visible_points=8):
        """Initialize with specified number of visible points"""
        self.visible_points = visible_points
        self.latest_game_state = MockGameState(visible_points)
        
        # Create a blank image with text for testing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"MOCK CAMERA - {visible_points}/16 points visible", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.latest_result = (frame, self.latest_game_state)
    
    def get_latest_result(self):
        """Return a mock detection result"""
        return self.latest_result
        
    def get_performance_metrics(self):
        """Return mock metrics"""
        return {'avg_fps': 30.0, 'last_inference_time': 0.033}


class MockCameraThread:
    """Mock CameraThread for testing"""
    
    def __init__(self, visible_points=8):
        """Initialize with specified number of visible points"""
        self.last_board_state = [['', '', ''], ['', '', ''], ['', '', '']]
        self.detection_thread = MockDetectionThread(visible_points)
        
    def set_frame(self, frame):
        """Mock set_frame method"""
        pass
        
    def get_latest_result(self):
        """Forward to detection thread"""
        return self.detection_thread.get_latest_result()
        
    def get_performance_metrics(self):
        """Forward to detection thread"""
        return self.detection_thread.get_performance_metrics()


def test_incomplete_grid(visible_points=8):
    """Create a test app with incomplete grid detection (8 of 16 points)"""
    app = QApplication(sys.argv)
    window = TicTacToeApp()
    
    # Replace camera thread with our mock
    window.camera_thread = MockCameraThread(visible_points)
    
    # Trigger an update to show the warning
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    window.update_camera_view(blank_frame)
    
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Test with 8 out of 16 points visible (warning should appear)
    print("Testing with 8/16 points visible (should show warning)")
    sys.exit(test_incomplete_grid(8))