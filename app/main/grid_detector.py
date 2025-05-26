# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Grid detector module for the TicTacToe application.
"""
# pylint: disable=no-member,broad-exception-caught
import logging
from typing import Optional
from typing import Tuple

import cv2  # pylint: disable=import-error
import numpy as np

from app.core.detector_constants import GRID_DIST_STD_DEV_THRESHOLD
from app.core.detector_constants import IDEAL_GRID_NORM
from app.core.detector_constants import MIN_POINTS_FOR_HOMOGRAPHY
from app.core.detector_constants import RANSAC_REPROJ_THRESHOLD

# Number of grid points (4x4 grid has 16 intersection points)
GRID_POINTS_COUNT = 16


class GridDetector:  # pylint: disable=too-many-instance-attributes
    """Detects and processes the Tic Tac Toe grid."""

    def __init__(self, pose_model, config=None, logger=None):
        """Initialize the grid detector.

        Args:
            pose_model: The YOLO pose model for grid detection
            config: Configuration object
            logger: Logger instance
        """
        self.pose_model = pose_model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Grid detection parameters
        self.pose_conf_threshold = getattr(config, 'pose_conf_threshold', 0.5)
        self.grid_detection_retries = 0
        self.last_valid_grid_time = None
        self.max_grid_detection_retries = getattr(config, 'max_grid_detection_retries', 3)
        self.grid_lost_threshold_seconds = getattr(config, 'grid_lost_threshold_seconds', 2.0)

    def detect_grid(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detects the Tic Tac Toe grid in the frame.

        Args:
            frame: The input frame

        Returns:
            Tuple of (processed_frame, keypoints)
            where keypoints is a numpy array of shape (16, 2) containing
            the grid intersection points
        """
        # Use the pose model to find grid keypoints
        pose_results = self.pose_model.predict(
            frame, conf=self.pose_conf_threshold, verbose=False
        )

        # Process the raw keypoints - inicializace prázdným polem
        keypoints = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)

        # Check if we have any results
        if not pose_results or len(pose_results) == 0:
            self.logger.debug("No pose results found")
            return frame, keypoints

        # Get the first result (assuming only one grid in the frame)
        result = pose_results[0]

        # Extract keypoints from the result
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            # YOLO v8 pose format
            kpts = result.keypoints.xy.cpu().numpy()
            if len(kpts) > 0:
                kpts = kpts[0]  # First detection
                # If we have fewer than 16 points, pad with zeros
                if len(kpts) < GRID_POINTS_COUNT:
                    self.logger.debug(
                        "Incomplete grid detection: %s/%s points",
                        len(kpts), GRID_POINTS_COUNT
                    )
                    # Pad with zeros to ensure we have 16 points
                    pad_count = GRID_POINTS_COUNT - len(kpts)
                    kpts = np.vstack([kpts, np.zeros((pad_count, 2))])
                # If we have more than 16 points, truncate
                elif len(kpts) > GRID_POINTS_COUNT:
                    self.logger.debug(
                        "Too many grid points detected: %s (using first %s)",
                        len(kpts), GRID_POINTS_COUNT
                    )
                    kpts = kpts[:GRID_POINTS_COUNT]

                # Update keypoints with the detected points
                keypoints = kpts
        else:
            self.logger.debug("No keypoints found in pose result")

        return frame, keypoints

    def sort_grid_points(
            self, keypoints: np.ndarray) -> np.ndarray:  # pylint: disable=too-many-locals
        """Sorts the grid points into a consistent order.

        Args:
            keypoints: Array of shape (16, 2) containing the grid points

        Returns:
            Sorted array of shape (16, 2)
        """
        # Filter out points with zero coordinates (not detected)
        valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]

        if len(valid_points) < 4:
            self.logger.debug(
                "Not enough valid points to sort grid: %s", len(valid_points))
            return keypoints

        # Sort points for 4x4 grid layout (row by row, left to right)
        # This is CRITICAL for correct cell center calculation!

        self.logger.debug(
            "🔧 Sorting {len(valid_points)} grid points into 4x4 layout")

        # Sort by Y coordinate first (top to bottom)
        y_sorted_indices = np.argsort(valid_points[:, 1])
        y_sorted_points = valid_points[y_sorted_indices]

        # Adaptive row grouping - handle partial grids gracefully
        if len(valid_points) >= 12:  # At least 3 rows
            # Try to group into 4 rows
            points_per_row = len(valid_points) // 4
            remainder = len(valid_points) % 4
            sorted_valid_points = []

            current_idx = 0
            for row in range(4):
                # Calculate points in this row (distribute remainder)
                row_size = points_per_row + (1 if row < remainder else 0)
                if current_idx + row_size <= len(y_sorted_points):
                    row_points = y_sorted_points[current_idx:current_idx + row_size]
                    # Sort this row by X coordinate (left to right)
                    x_sorted_indices = np.argsort(row_points[:, 0])
                    row_sorted = row_points[x_sorted_indices]
                    sorted_valid_points.extend(row_sorted)
                    current_idx += row_size

                    self.logger.debug(
                        "  Row {row}: {row_size} points sorted by X")

            sorted_valid_points = np.array(sorted_valid_points)
        else:
            # Fallback for very incomplete grids - just sort by Y then X
            # Secondary sort by X coordinate
            combined_sort = np.lexsort((valid_points[:, 0], valid_points[:, 1]))
            sorted_valid_points = valid_points[combined_sort]
            self.logger.debug("  Fallback: lexicographic sort (Y,X)")

        # Create a new array for the sorted points
        sorted_keypoints = np.zeros_like(keypoints)
        sorted_keypoints[:len(valid_points)] = sorted_valid_points

        return sorted_keypoints

    def is_valid_grid(self, keypoints: np.ndarray) -> bool:
        """Checks if the detected grid is valid.

        Args:
            keypoints: Array of shape (16, 2) containing the grid points

        Returns:
            True if the grid is valid, False otherwise
        """
        # Count non-zero points
        valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]
        valid_count = len(valid_points)

        # Check if we have enough points
        if valid_count < MIN_POINTS_FOR_HOMOGRAPHY:
            self.logger.debug("Not enough valid grid points: %s/%s",
                              valid_count, MIN_POINTS_FOR_HOMOGRAPHY)
            return False

        # Check if points form a
        # reasonable grid (distances between adjacent points)
        # This is a simplified check - in
        # a real implementation, you would do more validation

        # Calculate distances between all pairs of points
        distances = []
        for i in range(valid_count):
            for j in range(i + 1, valid_count):
                dist = np.linalg.norm(valid_points[i] - valid_points[j])
                distances.append(dist)

        # Check if distances have reasonable standard deviation
        if len(distances) > 0:
            std_dev = np.std(distances)
            mean_dist = np.mean(distances)
            if std_dev / mean_dist > GRID_DIST_STD_DEV_THRESHOLD:
                self.logger.debug(
                    "Grid point distances too variable: std/mean = %.2f",
                    std_dev / mean_dist)
                return False

        return True

    def compute_homography(
            self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Computes the homography matrix from ideal grid to image coordinates.

        Args:
            keypoints: Array of shape (16, 2) containing the grid points

        Returns:
            Homography matrix or None if computation fails
        """
        # Filter out points with zero coordinates (not detected)
        valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]

        if len(valid_points) < MIN_POINTS_FOR_HOMOGRAPHY:
            self.logger.debug("Not enough valid points for homography: %s/%s",
                              len(valid_points), MIN_POINTS_FOR_HOMOGRAPHY)
            return None

        try:
            # Create ideal grid points (normalized coordinates)
            ideal_points = IDEAL_GRID_NORM.copy()

            # Compute homography from ideal grid to image coordinates
            homography_matrix, _ = cv2.findHomography(
                ideal_points[:len(valid_points)],
                valid_points,
                cv2.RANSAC,
                RANSAC_REPROJ_THRESHOLD
            )

            return homography_matrix
        except Exception as e:
            self.logger.error("Error computing homography: %s", e)
            return None

    def update_grid_status(self, is_valid: bool, current_time: float) -> bool:
        """Updates the grid status based on validity and timing.

        Args:
            is_valid: Whether the current grid detection is valid
            current_time: Current timestamp

        Returns:
            True if grid status changed significantly, False otherwise
        """
        grid_status_changed = False

        if is_valid:
            # Grid is valid now
            if self.last_valid_grid_time is None:
                # First valid detection
                self.logger.info("Grid detected for the first time")
                grid_status_changed = True

            # Update last valid time
            self.last_valid_grid_time = current_time
            # Reset retry counter
            self.grid_detection_retries = 0
        else:
            # Grid is invalid now
            if self.last_valid_grid_time is not None:
                # We had a valid grid before
                time_since_valid = current_time - self.last_valid_grid_time

                if time_since_valid > self.grid_lost_threshold_seconds:
                    # Grid has been lost for too long
                    self.logger.info(
                        "Grid lost after %.2f seconds", time_since_valid)
                    self.last_valid_grid_time = None
                    grid_status_changed = True
                else:
                    # Increment retry counter
                    self.grid_detection_retries += 1

                    if self.grid_detection_retries > self.max_grid_detection_retries:
                        # Too many retries, consider grid lost
                        self.logger.info("Grid lost after %s retries",
                                         self.grid_detection_retries)
                        self.last_valid_grid_time = None
                        grid_status_changed = True

        return grid_status_changed
