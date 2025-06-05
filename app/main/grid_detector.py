# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Grid detector module for the TicTacToe application.
"""
# pylint: disable=no-member,broad-exception-caught
import logging
from typing import Optional, Deque
from typing import Tuple
from collections import deque

import cv2  # pylint: disable=import-error
import numpy as np

from app.core.detector_constants import GRID_DIST_STD_DEV_THRESHOLD
from app.core.detector_constants import IDEAL_GRID_NORM
from app.core.detector_constants import MIN_POINTS_FOR_HOMOGRAPHY
from app.core.detector_constants import RANSAC_REPROJ_THRESHOLD

# Number of grid points (4x4 grid has 16 intersection points)
GRID_POINTS_COUNT = 16

# Temporal cache constants
CACHE_SIZE = 10  # Store last 75 frames for averaging
MAX_MISSING_FRAMES = 10  # Max consecutive frames a point can be missing before fallback to interpolation
CACHE_INVALIDATION_THRESHOLD = 50.0  # Pixel distance threshold for cache invalidation
MIN_CACHE_SIZE_FOR_AVERAGING = 5  # Minimum cache entries needed for reliable averaging


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
        self.pose_conf_threshold = getattr(config, 'pose_conf_threshold', 0.45)
        self.grid_detection_retries = 0
        self.last_valid_grid_time = None
        self.max_grid_detection_retries = getattr(config, 'max_grid_detection_retries', 3)
        self.grid_lost_threshold_seconds = getattr(config, 'grid_lost_threshold_seconds', 2.0)

        # Temporal cache system for grid point stability
        self.grid_point_cache = [deque(maxlen=CACHE_SIZE) for _ in range(GRID_POINTS_COUNT)]
        self.missing_frame_count = np.zeros(GRID_POINTS_COUNT, dtype=int)
        self.last_detected_points_mask = np.zeros(GRID_POINTS_COUNT, dtype=bool)
        self.last_cached_points_mask = np.zeros(GRID_POINTS_COUNT, dtype=bool)
        self.last_interpolated_points_mask = np.zeros(GRID_POINTS_COUNT, dtype=bool)
        self.cache_initialized = False

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

    def _update_temporal_cache(self, grid_points: np.ndarray) -> np.ndarray:
        """Update temporal cache with current grid points and return stabilized points.

        Args:
            grid_points: Array of shape (16, 2) containing current grid points

        Returns:
            Stabilized grid points using cache, detected points, and interpolation
        """
        # Initialize result array
        stabilized_points = np.zeros_like(grid_points)

        # Reset status masks
        self.last_detected_points_mask.fill(False)
        self.last_cached_points_mask.fill(False)
        self.last_interpolated_points_mask.fill(False)

        # Check for cache invalidation (significant grid movement)
        if self.cache_initialized and self._should_invalidate_cache(grid_points):
            self._invalidate_cache()
            self.logger.debug("Cache invalidated due to significant grid movement")

        for i in range(GRID_POINTS_COUNT):
            current_point = grid_points[i]

            # Check if current point is detected (non-zero)
            if np.sum(np.abs(current_point)) > 0:
                # Point is detected - add to cache and use it
                self.grid_point_cache[i].append(current_point.copy())
                self.missing_frame_count[i] = 0
                stabilized_points[i] = current_point
                self.last_detected_points_mask[i] = True

            else:
                # Point is missing - try cache first, then interpolation
                self.missing_frame_count[i] += 1

                if (len(self.grid_point_cache[i]) >= MIN_CACHE_SIZE_FOR_AVERAGING and
                    self.missing_frame_count[i] <= MAX_MISSING_FRAMES):
                    # Use cached average
                    cached_point = self._get_cached_average(i)
                    stabilized_points[i] = cached_point
                    self.last_cached_points_mask[i] = True
                    self.logger.debug(f"Using cached point for position {i}")

                else:
                    # Cache not available or point missing too long - mark for interpolation
                    stabilized_points[i] = np.zeros(2)
                    self.last_interpolated_points_mask[i] = True

        # Mark cache as initialized
        if not self.cache_initialized:
            self.cache_initialized = True
            self.logger.debug("Temporal cache initialized")

        return stabilized_points

    def _should_invalidate_cache(self, current_grid: np.ndarray) -> bool:
        """Check if cache should be invalidated due to significant grid movement."""
        if not self.cache_initialized:
            return False

        # Get average positions from cache for comparison
        detected_points = []
        cached_averages = []

        for i in range(GRID_POINTS_COUNT):
            current_point = current_grid[i]
            if (np.sum(np.abs(current_point)) > 0 and
                len(self.grid_point_cache[i]) >= MIN_CACHE_SIZE_FOR_AVERAGING):
                detected_points.append(current_point)
                cached_averages.append(self._get_cached_average(i))

        if len(detected_points) < 4:  # Need at least 4 points for comparison
            return False

        # Calculate average distance between detected and cached points
        detected_points = np.array(detected_points)
        cached_averages = np.array(cached_averages)
        distances = np.linalg.norm(detected_points - cached_averages, axis=1)
        avg_distance = np.mean(distances)

        return avg_distance > CACHE_INVALIDATION_THRESHOLD

    def _invalidate_cache(self):
        """Clear the temporal cache."""
        for i in range(GRID_POINTS_COUNT):
            self.grid_point_cache[i].clear()
        self.missing_frame_count.fill(0)
        self.cache_initialized = False

    def _get_cached_average(self, point_index: int) -> np.ndarray:
        """Get the averaged position for a grid point from cache."""
        if len(self.grid_point_cache[point_index]) == 0:
            return np.zeros(2)

        # Convert deque to numpy array and compute average
        cached_points = np.array(list(self.grid_point_cache[point_index]))
        return np.mean(cached_points, axis=0)

    def _apply_final_interpolation(self, grid_points: np.ndarray) -> np.ndarray:
        """Apply final interpolation to any remaining missing points after cache processing.

        Args:
            grid_points: Array of shape (16, 2) containing grid points with some potentially missing

        Returns:
            Complete grid with all missing points interpolated
        """
        # Create a copy to avoid modifying the original
        final_grid = grid_points.copy()

        # Find missing points (marked for interpolation)
        missing_indices = []
        for i in range(GRID_POINTS_COUNT):
            if self.last_interpolated_points_mask[i] or np.sum(np.abs(final_grid[i])) == 0:
                missing_indices.append(i)

        if not missing_indices:
            return final_grid  # No missing points

        # Create point map from available points for interpolation
        point_map = {}
        for i in range(GRID_POINTS_COUNT):
            if i not in missing_indices:
                row, col = i // 4, i % 4
                point_map[(col, row)] = final_grid[i]

        # Interpolate missing points
        for i in missing_indices:
            row, col = i // 4, i % 4
            interpolated = self._interpolate_point(row, col, point_map)
            if interpolated is not None:
                final_grid[i] = interpolated
                # Update the point map for subsequent interpolations
                point_map[(col, row)] = interpolated
                self.logger.debug(f"Final interpolation for grid position ({row},{col})")
            else:
                # Last resort: use simple grid-based interpolation
                final_grid[i] = self._simple_grid_interpolation(row, col, final_grid)
                self.logger.debug(f"Simple grid interpolation for position ({row},{col})")

        return final_grid

    def _simple_grid_interpolation(self, row: int, col: int, grid: np.ndarray) -> np.ndarray:
        """Simple grid-based interpolation as last resort."""
        # Find bounding box of non-zero points
        non_zero_points = grid[np.sum(np.abs(grid), axis=1) > 0]
        if len(non_zero_points) == 0:
            return np.array([100.0, 100.0])  # Default fallback

        min_x, min_y = np.min(non_zero_points, axis=0)
        max_x, max_y = np.max(non_zero_points, axis=0)

        # Interpolate based on grid position
        x = min_x + (max_x - min_x) * col / 3.0
        y = min_y + (max_y - min_y) * row / 3.0

        return np.array([x, y])

    def sort_grid_points(
            self, keypoints: np.ndarray) -> np.ndarray:  # pylint: disable=too-many-locals
        """Sorts the grid points into a consistent order and interpolates missing points.

        Args:
            keypoints: Array of shape (16, 2) containing the grid points

        Returns:
            Sorted array of shape (16, 2) with stabilized points using cache and interpolation
        """
        # Filter out points with zero coordinates (not detected)
        valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]

        if len(valid_points) < 4:
            self.logger.debug(
                "Not enough valid points to sort grid: %s", len(valid_points))
            return keypoints

        self.logger.debug(
            "Sorting %s grid points into 4x4 layout with temporal cache and interpolation",
            len(valid_points))

        # Try to use robust sorting first
        try:
            sorted_points = self._robust_sort_and_interpolate(valid_points)
            if sorted_points is not None:
                # Apply temporal cache stabilization
                stabilized_points = self._update_temporal_cache(sorted_points)

                # Apply final interpolation for any remaining missing points
                final_points = self._apply_final_interpolation(stabilized_points)
                return final_points
        except Exception as e:
            self.logger.debug("Robust sorting failed: %s, using fallback", e)

        # Fallback to basic sorting
        basic_sorted = self._basic_sort_grid_points(valid_points, keypoints)

        # Apply temporal cache even to basic sorted points
        stabilized_points = self._update_temporal_cache(basic_sorted)
        final_points = self._apply_final_interpolation(stabilized_points)
        return final_points

    def _robust_sort_and_interpolate(self, valid_points: np.ndarray) -> Optional[np.ndarray]:
        """Robustly sort and interpolate grid points using geometric analysis."""
        if len(valid_points) < 6:  # Need at least 6 points for good interpolation
            return None

        # Find corner points using convex hull
        try:
            hull = cv2.convexHull(valid_points.astype(np.float32))
            hull_points = hull.reshape(-1, 2)
            
            # Find 4 extreme corner points
            corners = self._find_corner_points(hull_points)
            if corners is None or len(corners) != 4:
                return None
            
            # Compute homography from corners to ideal grid
            ideal_corners = np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=np.float32)
            homography, _ = cv2.findHomography(corners, ideal_corners, cv2.RANSAC)
            
            if homography is None:
                return None
            
            # Transform all valid points to ideal grid space
            transformed_points = cv2.perspectiveTransform(
                valid_points.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            
            # Map points to grid positions
            grid_positions = self._map_to_grid_positions(transformed_points)
            
            # Create full 4x4 grid with interpolation
            full_grid = self._interpolate_missing_points(valid_points, grid_positions)
            
            return full_grid
            
        except Exception as e:
            self.logger.debug("Error in robust sort and interpolate: %s", e)
            return None

    def _find_corner_points(self, hull_points: np.ndarray) -> Optional[np.ndarray]:
        """Find the 4 corner points from convex hull."""
        if len(hull_points) < 4:
            return None
        
        # Find extremes
        top_left = hull_points[np.argmin(hull_points[:, 0] + hull_points[:, 1])]
        top_right = hull_points[np.argmax(hull_points[:, 0] - hull_points[:, 1])]
        bottom_right = hull_points[np.argmax(hull_points[:, 0] + hull_points[:, 1])]
        bottom_left = hull_points[np.argmin(hull_points[:, 0] - hull_points[:, 1])]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _map_to_grid_positions(self, transformed_points: np.ndarray) -> np.ndarray:
        """Map transformed points to discrete grid positions (0-3, 0-3)."""
        grid_positions = np.round(transformed_points).astype(int)
        grid_positions = np.clip(grid_positions, 0, 3)
        return grid_positions

    def _interpolate_missing_points(self, valid_points: np.ndarray, 
                                  grid_positions: np.ndarray) -> np.ndarray:
        """Interpolate missing grid points using bilinear interpolation."""
        # Create mapping from grid position to actual point
        point_map = {}
        for point, grid_pos in zip(valid_points, grid_positions):
            key = tuple(grid_pos)
            if key not in point_map:
                point_map[key] = point
        
        # Create full 16-point grid
        full_grid = np.zeros((16, 2), dtype=np.float32)
        
        for i in range(16):
            row, col = i // 4, i % 4
            grid_key = (col, row)  # Note: x,y vs col,row
            
            if grid_key in point_map:
                # Use detected point
                full_grid[i] = point_map[grid_key]
            else:
                # Interpolate missing point
                interpolated = self._interpolate_point(row, col, point_map)
                if interpolated is not None:
                    full_grid[i] = interpolated
                    self.logger.debug("Interpolated point at grid position (%d,%d)", row, col)
        
        return full_grid

    def _interpolate_point(self, row: int, col: int, point_map: dict) -> Optional[np.ndarray]:
        """Interpolate a single missing point using nearby detected points."""
        # Try different interpolation strategies
        
        # Strategy 1: Use horizontal/vertical neighbors
        interpolated = self._interpolate_linear(row, col, point_map)
        if interpolated is not None:
            return interpolated
        
        # Strategy 2: Use diagonal neighbors
        interpolated = self._interpolate_diagonal(row, col, point_map)
        if interpolated is not None:
            return interpolated
        
        # Strategy 3: Use corner-based extrapolation
        interpolated = self._interpolate_from_corners(row, col, point_map)
        if interpolated is not None:
            return interpolated
        
        return None

    def _interpolate_linear(self, row: int, col: int, point_map: dict) -> Optional[np.ndarray]:
        """Linear interpolation using horizontal or vertical neighbors."""
        # Try horizontal interpolation
        left_key = (col - 1, row)
        right_key = (col + 1, row)
        if left_key in point_map and right_key in point_map:
            return (point_map[left_key] + point_map[right_key]) / 2
        
        # Try vertical interpolation
        up_key = (col, row - 1)
        down_key = (col, row + 1)
        if up_key in point_map and down_key in point_map:
            return (point_map[up_key] + point_map[down_key]) / 2
        
        # Try extrapolation from one side
        if left_key in point_map and (col - 2, row) in point_map:
            # Extrapolate from two left points
            p1 = point_map[(col - 2, row)]
            p2 = point_map[left_key]
            return p2 + (p2 - p1)
        
        if right_key in point_map and (col + 2, row) in point_map:
            # Extrapolate from two right points
            p1 = point_map[(col + 2, row)]
            p2 = point_map[right_key]
            return p2 + (p2 - p1)
        
        return None

    def _interpolate_diagonal(self, row: int, col: int, point_map: dict) -> Optional[np.ndarray]:
        """Interpolation using diagonal neighbors."""
        corners = [
            (col - 1, row - 1), (col + 1, row - 1),
            (col + 1, row + 1), (col - 1, row + 1)
        ]
        
        available_corners = [point_map[key] for key in corners if key in point_map]
        
        if len(available_corners) >= 2:
            return np.mean(available_corners, axis=0)
        
        return None

    def _interpolate_from_corners(self, row: int, col: int, point_map: dict) -> Optional[np.ndarray]:
        """Bilinear interpolation using the four corner points of the grid."""
        corners = [(0, 0), (3, 0), (3, 3), (0, 3)]
        corner_points = []
        
        for corner_key in corners:
            if corner_key in point_map:
                corner_points.append(point_map[corner_key])
        
        if len(corner_points) >= 3:
            # Use available corners to estimate grid structure and interpolate
            return self._bilinear_from_corners(row, col, corner_points, corners, point_map)
        
        return None

    def _bilinear_from_corners(self, row: int, col: int, corner_points: list, 
                              corner_keys: list, point_map: dict) -> Optional[np.ndarray]:
        """Perform bilinear interpolation from available corner points."""
        try:
            # Simple approach: use grid ratios
            row_ratio = row / 3.0
            col_ratio = col / 3.0
            
            # If we have all 4 corners, do proper bilinear interpolation
            if len(corner_points) == 4 and all(key in point_map for key in corner_keys):
                tl = point_map[(0, 0)]
                tr = point_map[(3, 0)]
                br = point_map[(3, 3)]
                bl = point_map[(0, 3)]
                
                # Bilinear interpolation
                top = tl + col_ratio * (tr - tl)
                bottom = bl + col_ratio * (br - bl)
                return top + row_ratio * (bottom - top)
            
            # Otherwise use mean of available corners with distance weighting
            if len(corner_points) >= 2:
                return np.mean(corner_points, axis=0)
        
        except Exception:
            pass
        
        return None

    def _basic_sort_grid_points(self, valid_points: np.ndarray, 
                               original_keypoints: np.ndarray) -> np.ndarray:
        """Basic sorting fallback method with simple interpolation."""
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

            sorted_valid_points = np.array(sorted_valid_points)
        else:
            # Fallback for very incomplete grids - just sort by Y then X
            combined_sort = np.lexsort((valid_points[:, 0], valid_points[:, 1]))
            sorted_valid_points = valid_points[combined_sort]

        # Create a new array for the sorted points
        sorted_keypoints = np.zeros_like(original_keypoints)
        sorted_keypoints[:len(valid_points)] = sorted_valid_points
        
        # Try simple interpolation for missing points if we have enough data
        if len(valid_points) >= 6:
            sorted_keypoints = self._simple_interpolate_missing(sorted_keypoints, valid_points)

        return sorted_keypoints
        
    def _simple_interpolate_missing(self, grid_points: np.ndarray, 
                                   valid_points: np.ndarray) -> np.ndarray:
        """Simple interpolation for missing grid points."""
        try:
            # Find bounding box of valid points
            min_x, min_y = np.min(valid_points, axis=0)
            max_x, max_y = np.max(valid_points, axis=0)
            
            # Create regular grid in bounding box
            for i in range(16):
                if np.sum(np.abs(grid_points[i])) == 0:  # Missing point
                    row, col = i // 4, i % 4
                    
                    # Interpolate position based on grid structure
                    x = min_x + (max_x - min_x) * col / 3.0
                    y = min_y + (max_y - min_y) * row / 3.0
                    
                    grid_points[i] = [x, y]
                    self.logger.debug("Simple interpolation for grid position (%d,%d): [%.1f, %.1f]", 
                                      row, col, x, y)
        
        except Exception as e:
            self.logger.debug("Error in simple interpolation: %s", e)
        
        return grid_points

    def is_valid_grid(self, keypoints: np.ndarray) -> bool:
        """Checks if the detected grid is valid, considering interpolated points.

        Args:
            keypoints: Array of shape (16, 2) containing the grid points

        Returns:
            True if the grid is valid, False otherwise
        """
        # Count non-zero points (both detected and interpolated)
        valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]
        valid_count = len(valid_points)

        # With interpolation, we should now have more points
        # Lower the minimum requirement since interpolation helps
        min_required = max(4, MIN_POINTS_FOR_HOMOGRAPHY - 2)
        
        if valid_count < min_required:
            self.logger.debug("Not enough valid grid points: %s/%s",
                              valid_count, min_required)
            return False

        # For grids with interpolated points, be more lenient with validation
        if valid_count >= 12:  # Most points available - use strict validation
            return self._validate_grid_geometry(valid_points)
        elif valid_count >= 8:  # Some interpolation - moderate validation
            return self._validate_grid_geometry_lenient(valid_points)
        else:  # Heavy interpolation - minimal validation
            return self._validate_basic_grid_structure(valid_points)

    def _validate_grid_geometry(self, valid_points: np.ndarray) -> bool:
        """Strict grid geometry validation for complete grids."""
        # Calculate distances between all pairs of points
        distances = []
        for i in range(len(valid_points)):
            for j in range(i + 1, len(valid_points)):
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

    def _validate_grid_geometry_lenient(self, valid_points: np.ndarray) -> bool:
        """Lenient validation for partially interpolated grids."""
        if len(valid_points) < 4:
            return False
        
        # Check basic grid structure - points should form roughly rectangular pattern
        try:
            # Find bounding box
            min_x, min_y = np.min(valid_points, axis=0)
            max_x, max_y = np.max(valid_points, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Grid should have reasonable aspect ratio
            if width <= 0 or height <= 0:
                return False
            
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > 3.0:  # Too elongated
                self.logger.debug("Grid aspect ratio too extreme: %.2f", aspect_ratio)
                return False
                
            # Points should be reasonably distributed
            area = width * height
            if area < 1000:  # Too small
                self.logger.debug("Grid area too small: %.1f", area)
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug("Error in lenient grid validation: %s", e)
            return False

    def _validate_basic_grid_structure(self, valid_points: np.ndarray) -> bool:
        """Basic validation for heavily interpolated grids."""
        if len(valid_points) < 4:
            return False
        
        try:
            # Just check that points form a reasonable bounding box
            min_x, min_y = np.min(valid_points, axis=0)
            max_x, max_y = np.max(valid_points, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Very basic checks
            if width <= 10 or height <= 10:  # Too small
                return False
            
            if width > 2000 or height > 2000:  # Too large
                return False
                
            return True
            
        except Exception:
            return False

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
