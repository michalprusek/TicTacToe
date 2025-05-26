# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""Constants for the Game Detector."""
import numpy as np

from app.core.game_state import GRID_POINTS_COUNT

# --- Detection Thresholds --- #
BBOX_CONF_THRESHOLD = 0.45  # Slightly lower threshold for better detection
POSE_CONF_THRESHOLD = 0.45  # Slightly lower threshold for better detection
KEYPOINT_VISIBLE_THRESHOLD = 0.3  # Lower threshold to detect more keypoints

# --- Homography and RANSAC --- #
MIN_POINTS_FOR_HOMOGRAPHY = 4  # Reduced from 6 to be more robust
RANSAC_REPROJ_THRESHOLD = 10.0  # Increased from 5.0 to be more tolerant

# --- Grid Validation --- #
GRID_DIST_STD_DEV_THRESHOLD = 300.0  # Increased from 100.0 to be more tolerant
GRID_ANGLE_TOLERANCE_DEG = 30.0  # Increased from 20.0 to be more tolerant

# --- Grid Detection Retry --- #
MAX_GRID_DETECTION_RETRIES = 3  # Number of retries for grid detection

# --- Default Model Paths --- #
DEFAULT_DETECT_MODEL_PATH = "weights/best_detection.pt"
DEFAULT_POSE_MODEL_PATH = "weights/best_pose.pt"

# --- Debug Drawing Colors --- #
DEBUG_UV_KPT_COLOR = (0, 255, 255)  # Yellow for UV kpts
DEBUG_BBOX_COLOR = (255, 255, 255)  # White for minAreaRect
DEBUG_BBOX_THICKNESS = 1
DEBUG_FPS_COLOR = (0, 255, 0)  # Green for FPS

# --- Message Colors (used in drawing utils potentially) --- #
WARNING_BG_COLOR = (25, 25, 150)  # Dark red background
ERROR_BG_COLOR = (0, 0, 100)      # Dark blue background

# --- Ideal Grid Coordinates --- #
# Ideal grid coordinates (normalized 0-3 for a 4x4 point grid)
IDEAL_GRID_NORM = np.array(
    [(i % 4, i // 4) for i in range(GRID_POINTS_COUNT)], dtype=np.float32
)
