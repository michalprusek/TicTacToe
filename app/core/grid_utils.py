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

        # Step 1: Find 4 corner points
        logger.debug("Finding 4 corner points from 16 grid points")

        # Heuristics using coordinate sums and differences
        sum_xy = points_np[:, 0] + points_np[:, 1]
        diff_yx = points_np[:, 1] - points_np[:, 0]  # y - x

        tl_idx = np.argmin(sum_xy)  # Top-left: minimizes x+y
        br_idx = np.argmax(sum_xy)  # Bottom-right: maximizes x+y
        tr_idx = np.argmin(diff_yx)  # Top-right: minimizes y-x
        bl_idx = np.argmax(diff_yx)  # Bottom-left: maximizes y-x        # Ensure these indices are unique
        corner_indices = {tl_idx, br_idx, tr_idx, bl_idx}

        if len(corner_indices) < 4:
            logger.debug(
                "Corner detection failed, using fallback method with "
                "cv2.minAreaRect"
            )
            # FALLBACK METHOD: Use cv2.minAreaRect

            # Find minimal bounding rectangle of all 16 points
            rect = cv2.minAreaRect(points_np)
            box_points = cv2.boxPoints(rect)  # Returns 4 corners

            # Sort box_points in order TL, TR, BR, BL
            # 1. Sort by y-coordinate
            box_points_sorted_y = sorted(box_points, key=lambda p: p[1])
            # 2. First two are top, last two are bottom
            # Sort top by x
            top_points = sorted(box_points_sorted_y[:2], key=lambda p: p[0])
            # Sort bottom by x
            bottom_points = sorted(box_points_sorted_y[2:], key=lambda p: p[0])

            # Final order TL, TR, BR, BL
            corners_src_ordered_array = np.array([
                top_points[0], top_points[1],
                bottom_points[1], bottom_points[0]
            ], dtype=np.float32)
            logger.debug(
                "Used fallback minAreaRect method for corner detection"
            )
        else:
            corners_src_ordered_array = np.array([
                points_np[tl_idx],
                points_np[tr_idx],
                points_np[br_idx],
                points_np[bl_idx]
            ], dtype=np.float32)
            logger.debug(
                "Successfully found 4 unique corner points using sum/diff "
                "heuristics"
            )

        # Step 2: Define target points for these 4 outer corners
        # (preliminary normalization)
        target_size_prelim = 300.0
        corners_dst_prelim = np.array([
            [0, 0],                            # TL
            [target_size_prelim, 0],           # TR
            [target_size_prelim, target_size_prelim], # BR
            [0, target_size_prelim]            # BL
        ], dtype=np.float32)

        # Step 3: Compute preliminary homography (h_prelim)
        h_prelim, _ = cv2.findHomography(
            corners_src_ordered_array, corners_dst_prelim, cv2.RANSAC
        )
        if h_prelim is None:
            logger.warning("Failed to compute preliminary homography")
            return None, None        # Step 4: Transform all 16 original Grid Points using h_prelim
        # points_np must be shape (N,1,2) for perspectiveTransform
        transformed_16_points = cv2.perspectiveTransform(
            points_np.reshape(-1, 1, 2), h_prelim
        )
        # Back to shape (16, 2)
        transformed_16_points = transformed_16_points.reshape(-1, 2)

        # Step 5: Sort original Grid Points based on their transformed positions
        logger.debug(
            "Sorting original grid points based on transformed positions"
        )

        point_pairs = []  # (original_point_coords, transformed_point_coords)
        for i in range(len(points_np)):
            point_pairs.append({
                'original': points_np[i],
                'transformed': transformed_16_points[i],
                'index': i
            })

        cell_spacing_prelim = target_size_prelim / 3.0

        def get_grid_indices(transformed_pt_coords):
            # Round to nearest grid index (0, 1, 2, 3)
            col_idx = int(round(
                transformed_pt_coords[0] / cell_spacing_prelim
            ))
            row_idx = int(round(
                transformed_pt_coords[1] / cell_spacing_prelim
            ))
            # Clamp to valid indices 0-3
            col_idx = max(0, min(3, col_idx))
            row_idx = max(0, min(3, row_idx))
            return row_idx, col_idx

        # Sort pairs
        point_pairs.sort(
            key=lambda pair: get_grid_indices(pair['transformed'])
        )

        # Extract sorted original points
        sorted_grid_points_src = np.array(
            [pair['original'] for pair in point_pairs], dtype=np.float32
        )

        # Step 6: Define ideal target points for final homography
        ideal_points_dst_final = []
        # Size of ONE cell in final normalized space
        cell_size_final = 100
        for r in range(4):  # 4 rows of points
            for c in range(4):  # 4 columns of points
                ideal_points_dst_final.append(
                    [c * cell_size_final, r * cell_size_final]
                )
        ideal_points_dst_final = np.array(
            ideal_points_dst_final, dtype=np.float32
        )        # Step 7: Compute final homography (h_final)
        h_final, _ = cv2.findHomography(
            sorted_grid_points_src, ideal_points_dst_final
        )
        if h_final is None:
            logger.warning("Failed to compute final homography")
            return None, None

        logger.debug(
            "Successfully computed robust grid sorting and homography"
        )
        return sorted_grid_points_src, h_final

    except Exception as e:
        logger.error("Error in robust_sort_grid_points: %s", e)
        return None, None
