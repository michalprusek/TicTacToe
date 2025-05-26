"""
Grid utility functions for the TicTacToe application.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import cv2


def robust_sort_grid_points(
    all_16_points: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Robustní metoda usporiadania 16 Grid Pointov a výpočtu Homografie.

    Args:
        all_16_points: numpy array tvaru (16, 2) alebo zoznam túplov (x,y)
        logger: Optional logger for debug messages

    Returns:
        Tuple of (sorted_grid_points_src, h_final) or (None, None) if failed
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate input
    if all_16_points is None or len(all_16_points) != 16:
        logger.warning(
            "Invalid input: expected 16 points, got %s",
            len(all_16_points) if all_16_points is not None else 'None'
        )
        return None, None

    try:
        # Convert to numpy array
        points_np = np.array(all_16_points, dtype=np.float32)

        # Step 1: Find 4 corner points and get preliminary homography
        corners_src, h_prelim = _find_corners_and_preliminary_homography(points_np, logger)
        if corners_src is None or h_prelim is None:
            return None, None

        # Step 2: Transform and sort all points
        sorted_points = _transform_and_sort_points(points_np, h_prelim, logger)
        if sorted_points is None:
            return None, None

        # Step 3: Compute final homography
        h_final = _compute_final_homography(sorted_points, logger)
        if h_final is None:
            return None, None

        logger.debug("Successfully computed robust grid sorting and homography")
        return sorted_points, h_final

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Error in robust_sort_grid_points: %s", e)
        return None, None


def _find_corners_and_preliminary_homography(
    points_np: np.ndarray, logger: logging.Logger
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find 4 corner points and compute preliminary homography."""
    logger.debug("Finding 4 corner points from 16 grid points")

    # Heuristics using coordinate sums and differences
    sum_xy = points_np[:, 0] + points_np[:, 1]
    diff_yx = points_np[:, 1] - points_np[:, 0]  # y - x

    tl_idx = np.argmin(sum_xy)  # Top-left: minimizes x+y
    br_idx = np.argmax(sum_xy)  # Bottom-right: maximizes x+y
    tr_idx = np.argmin(diff_yx)  # Top-right: minimizes y-x
    bl_idx = np.argmax(diff_yx)  # Bottom-left: maximizes y-x

    # Ensure these indices are unique
    corner_indices = {tl_idx, br_idx, tr_idx, bl_idx}

    if len(corner_indices) < 4:
        # Use fallback method
        corners_src = _fallback_corner_detection(points_np, logger)
    else:
        corners_src = np.array([
            points_np[tl_idx], points_np[tr_idx],
            points_np[br_idx], points_np[bl_idx]
        ], dtype=np.float32)
        logger.debug("Successfully found 4 unique corner points using sum/diff heuristics")

    if corners_src is None:
        return None, None

    # Define target points for preliminary homography
    target_size_prelim = 300.0
    corners_dst_prelim = np.array([
        [0, 0], [target_size_prelim, 0],
        [target_size_prelim, target_size_prelim], [0, target_size_prelim]
    ], dtype=np.float32)

    # Compute preliminary homography
    h_prelim, _ = cv2.findHomography(  # pylint: disable=no-member
        corners_src, corners_dst_prelim, cv2.RANSAC  # pylint: disable=no-member
    )
    if h_prelim is None:
        logger.warning("Failed to compute preliminary homography")
        return None, None

    return corners_src, h_prelim


def _fallback_corner_detection(points_np: np.ndarray, logger: logging.Logger) -> Optional[np.ndarray]:  # pylint: disable=line-too-long
    """Fallback corner detection using cv2.minAreaRect."""
    logger.debug("Corner detection failed, using fallback method with cv2.minAreaRect")

    # Find minimal bounding rectangle of all 16 points
    rect = cv2.minAreaRect(points_np)  # pylint: disable=no-member
    box_points = cv2.boxPoints(rect)  # pylint: disable=no-member

    # Sort box_points in order TL, TR, BR, BL
    box_points_sorted_y = sorted(box_points, key=lambda p: p[1])
    # First two are top, last two are bottom
    top_points = sorted(box_points_sorted_y[:2], key=lambda p: p[0])
    bottom_points = sorted(box_points_sorted_y[2:], key=lambda p: p[0])

    # Final order TL, TR, BR, BL
    corners_src = np.array([
        top_points[0], top_points[1],
        bottom_points[1], bottom_points[0]
    ], dtype=np.float32)

    logger.debug("Used fallback minAreaRect method for corner detection")
    return corners_src


def _transform_and_sort_points(  # pylint: disable=line-too-long
    points_np: np.ndarray, h_prelim: np.ndarray, logger: logging.Logger
) -> Optional[np.ndarray]:
    """Transform all 16 points and sort them based on grid positions."""
    # Transform all 16 original Grid Points using h_prelim
    transformed_16_points = cv2.perspectiveTransform(  # pylint: disable=no-member
        points_np.reshape(-1, 1, 2), h_prelim
    ).reshape(-1, 2)

    logger.debug("Sorting original grid points based on transformed positions")

    # Create point pairs with indices
    point_pairs = []
    for i, point in enumerate(points_np):
        point_pairs.append({
            'original': point,
            'transformed': transformed_16_points[i],
            'index': i
        })

    # Sort based on grid indices
    cell_spacing_prelim = 300.0 / 3.0  # target_size_prelim / 3.0

    def get_grid_indices(transformed_coords):
        col_idx = int(round(transformed_coords[0] / cell_spacing_prelim))
        row_idx = int(round(transformed_coords[1] / cell_spacing_prelim))
        # Clamp to valid indices 0-3
        col_idx = max(0, min(3, col_idx))
        row_idx = max(0, min(3, row_idx))
        return row_idx, col_idx

    point_pairs.sort(key=lambda pair: get_grid_indices(pair['transformed']))

    # Extract sorted original points
    return np.array([pair['original'] for pair in point_pairs], dtype=np.float32)


def _compute_final_homography(sorted_points: np.ndarray, logger: logging.Logger) -> Optional[np.ndarray]:  # pylint: disable=line-too-long
    """Compute final homography using ideal target points."""
    # Define ideal target points for final homography
    cell_size_final = 100
    ideal_points_dst_final = []

    for r in range(4):  # 4 rows of points
        for c in range(4):  # 4 columns of points
            ideal_points_dst_final.append([c * cell_size_final, r * cell_size_final])

    ideal_points_dst_final = np.array(ideal_points_dst_final, dtype=np.float32)

    # Compute final homography
    h_final, _ = cv2.findHomography(  # pylint: disable=no-member
        sorted_points, ideal_points_dst_final
    )
    if h_final is None:
        logger.warning("Failed to compute final homography")
        return None

    return h_final
