# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Shared constants for the TicTacToe application.
Consolidates all constants from multiple files to avoid duplication.
"""

# Camera Constants
DEFAULT_CAMERA_INDEX = 0

# Game Constants
DEFAULT_DIFFICULTY = 10

# Arm Movement Constants
DEFAULT_SPEED = 100000
MAX_SPEED_FACTOR = 2
DEFAULT_SAFE_Z = 15.0
DEFAULT_DRAW_Z = 5.0
DRAWING_SPEED = 50000
MAX_SPEED = 100000

# GUI Constants
BUTTON_FONT_SIZE = 10
STATUS_FONT_SIZE = 9
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600

# Detection Constants
DETECTION_FPS = 2
MIN_CONFIDENCE_THRESHOLD = 0.5

# Threading Constants
CAMERA_THREAD_INTERVAL = 500  # ms
ARM_THREAD_INTERVAL = 100    # ms

# File Paths
WEIGHTS_DIR = "weights"
DETECTION_MODEL = "best_detection.pt"
POSE_MODEL = "best_pose.pt"
CALIBRATION_FILE = "hand_eye_calibration.json"
