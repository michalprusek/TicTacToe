"""
Visualization manager module for the TicTacToe application.
"""
# pylint: disable=no-member,broad-exception-caught,unexpected-keyword-arg
import logging
from typing import List, Dict, Optional

import numpy as np
import cv2  # pylint: disable=import-error

from app.core.detector_constants import (
    DEBUG_UV_KPT_COLOR,
    DEBUG_FPS_COLOR
)
from app.main.frame_utils import FrameConverter
from app.main import drawing_utils


class VisualizationManager:
    """Manages visualization of detection results."""

    def __init__(self, config=None, logger=None):
        """Initialize the visualization manager.

        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Visualization settings
        self.show_detections = getattr(config, 'show_detections', True)
        self.show_grid = getattr(config, 'show_grid', True)
        self.show_debug_info = getattr(config, 'show_debug_info', True)

        # Debug window settings
        self.debug_window_scale_factor = getattr(config, 'debug_window_scale_factor', 0.5)

    def draw_detection_results(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        self,
        frame: np.ndarray,
        fps: float,
        pose_kpts_uv: Optional[np.ndarray],
        ordered_kpts_uv: Optional[np.ndarray],  # pylint: disable=unused-argument
        cell_polygons: Optional[List[np.ndarray]],
        detected_symbols: List[Dict],
        homography: Optional[np.ndarray],  # pylint: disable=unused-argument
        game_state=None
    ) -> np.ndarray:
        """Draw detection results onto the frame.

        Args:
            frame: Input frame to draw on
            fps: Current frames per second
            pose_kpts_uv: Raw keypoints from YOLO (16 points)
            ordered_kpts_uv: Validated & sorted keypoints (16 points)
            cell_polygons: Derived 9 cell polygons
            detected_symbols: List of symbol dicts {'label', 'box', 'confidence', 'class_id'}
            homography: Homography matrix (optional)
            game_state: Current game state (optional)

        Returns:
            Frame with detection results drawn on it
        """
        # Create a copy of the frame to draw on
        result_frame = frame.copy()

        # --- 1. Draw FPS counter
        if self.show_debug_info:
            try:
                cv2.putText(
                    result_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    DEBUG_FPS_COLOR,
                    2
                )
            except Exception as e:
                self.logger.debug("Error drawing FPS: %s", e)

        # --- 2. Draw raw keypoints from pose model
        if self.show_grid and pose_kpts_uv is not None:
            try:
                # Filter out points with zero coordinates
                valid_kpts = pose_kpts_uv[np.sum(np.abs(pose_kpts_uv), axis=1) > 0]
                frame_h, frame_w = result_frame.shape[:2]

                # Draw each keypoint
                for i, (x, y) in enumerate(valid_kpts):
                    # Validate coordinates
                    x_int = int(np.clip(x, 0, frame_w - 1))
                    y_int = int(np.clip(y, 0, frame_h - 1))

                    if 0 <= x_int < frame_w and 0 <= y_int < frame_h:
                        cv2.circle(
                            result_frame,
                            (x_int, y_int),
                            5,
                            DEBUG_UV_KPT_COLOR,
                            -1
                        )

                        # Optionally draw keypoint index
                        if self.show_debug_info:
                            text_x = int(np.clip(x + 5, 0, frame_w - 20))
                            text_y = int(np.clip(y - 5, 10, frame_h - 5))
                            cv2.putText(
                                result_frame,
                                str(i),
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                DEBUG_UV_KPT_COLOR,
                                1
                            )
            except Exception as e:
                self.logger.debug("Error drawing keypoints: %s", e)

        # --- 3. Draw cell polygons
        if self.show_grid and cell_polygons:
            try:
                frame_h, frame_w = result_frame.shape[:2]
                for i, polygon in enumerate(cell_polygons):
                    if polygon is not None and len(polygon) > 0:
                        # Validate and clamp polygon coordinates
                        valid_polygon = np.array(polygon, dtype=np.float32)
                        valid_polygon[:, 0] = np.clip(valid_polygon[:, 0], 0, frame_w - 1)
                        valid_polygon[:, 1] = np.clip(valid_polygon[:, 1], 0, frame_h - 1)

                        # Draw cell polygon
                        cv2.polylines(
                            result_frame,
                            [valid_polygon.astype(np.int32)],
                            True,
                            (0, 255, 0),
                            2
                        )

                        # Optionally draw cell index
                        if self.show_debug_info:
                            # Calculate cell center
                            center = np.mean(valid_polygon, axis=0).astype(int)
                            center_x = int(np.clip(center[0], 10, frame_w - 10))
                            center_y = int(np.clip(center[1], 10, frame_h - 10))
                            cv2.putText(
                                result_frame,
                                str(i),
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )
            except Exception as e:
                self.logger.debug("Error drawing cell polygons: %s", e)

        # --- 4. Draw detected symbols
        if self.show_detections and detected_symbols:
            self.logger.debug("Drawing %s detected symbols", len(detected_symbols))
            for det_info in detected_symbols:
                try:
                    # Check if det_info is a dictionary with expected keys
                    required_keys = ['label', 'confidence', 'box', 'class_id']
                    if isinstance(det_info, dict) and all(k in det_info for k in required_keys):
                        label = det_info['label']
                        conf = det_info['confidence']
                        box = det_info['box']
                        class_id = det_info['class_id']

                        # Draw bounding box and label
                        drawing_utils.draw_symbol_box(
                            result_frame, box, conf, class_id, label
                        )
                    # Handle raw detection format (x1,y1,x2,y2,conf,class_id)
                    elif isinstance(det_info, (list, tuple)) and len(det_info) >= 6:
                        x1, y1, x2, y2, conf, class_id = det_info[:6]
                        box = [x1, y1, x2, y2]
                        label = "X" if class_id == 0 else "O"  # class_id 0 = X, class_id 1 = O

                        # Filter by confidence threshold
                        if conf >= getattr(self.config, 'bbox_conf_threshold', 0.5):
                            drawing_utils.draw_symbol_box(
                                result_frame, box, conf, class_id, label
                            )
                except Exception as e:
                    self.logger.error("Error drawing symbol: %s", e)

        # --- 5. Draw game state information if available
        if game_state and self.show_debug_info:
            try:
                # Draw game status
                status_text = f"Game Status: {getattr(game_state, 'status', 'Unknown')}"
                cv2.putText(
                    result_frame,
                    status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Draw winner if game is over
                winner = getattr(game_state, 'winner', None)
                if winner:
                    winner_text = f"Winner: {winner}"
                    cv2.putText(
                        result_frame,
                        winner_text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
            except Exception as e:
                self.logger.debug("Error drawing game state info: %s", e)

        return result_frame

    def draw_debug_info(self, frame: np.ndarray, fps: float, game_state=None) -> None:
        """Draws debug information in a separate window.

        Args:
            frame: Input frame
            fps: Current frames per second
            game_state: Current game state (optional)
        """
        if not self.show_debug_info:
            return

        # Scale down the frame for the debug window
        try:
            debug_frame = FrameConverter.resize_frame(frame, self.debug_window_scale_factor)

            # Prepare text lines for debug info
            texts_to_draw = [f"FPS: {fps:.2f}"]

            if game_state:
                # Add game state information
                winner_text = getattr(game_state, 'winner', 'None')
                texts_to_draw.extend([
                    f"Game Status: {getattr(game_state, 'status', 'Unknown')}",
                    f"Winner: {winner_text}",
                    f"Grid Visible: {getattr(game_state, 'is_grid_visible', False)}",
                    f"Grid Stable: {getattr(game_state, 'is_grid_stable', False)}"
                ])

                # Add grid points and cells information
                grid_points_count = 0
                if hasattr(game_state, 'grid_points') and game_state.grid_points is not None:
                    grid_points_count = len(game_state.grid_points)

                cell_polygons_count = 0
                if hasattr(game_state, 'cell_polygons') and game_state.cell_polygons is not None:
                    cell_polygons_count = len(game_state.cell_polygons)

                texts_to_draw.extend([
                    f"Grid Points: {grid_points_count}/16",
                    f"Cells: {cell_polygons_count}/9"
                ])

            # Position for the text (top-left corner)
            text_x = 10
            text_y = 20

            # Draw text lines
            drawing_utils.draw_text_lines(
                debug_frame,
                texts_to_draw,
                text_x,
                text_y,
                y_offset=20
            )

            # Display the debug window
            cv2.imshow('Debug', debug_frame)
        except Exception as e:
            self.logger.debug("Error in draw_debug_info: %s", e)
