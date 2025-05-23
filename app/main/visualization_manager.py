"""
Visualization manager module for the TicTacToe application.
"""
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

from app.core.detector_constants import (
    DEBUG_UV_KPT_COLOR,
    DEBUG_BBOX_COLOR,
    DEBUG_BBOX_THICKNESS,
    DEBUG_FPS_COLOR
)
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

    def draw_detection_results(
        self,
        frame: np.ndarray,
        fps: float,
        pose_kpts_uv: Optional[np.ndarray],
        ordered_kpts_uv: Optional[np.ndarray],
        cell_polygons: Optional[List[np.ndarray]],
        detected_symbols: List[Dict],
        homography: Optional[np.ndarray],
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
            cv2.putText(
                result_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                DEBUG_FPS_COLOR,
                2
            )
            
        # --- 2. Draw raw keypoints from pose model
        if self.show_grid and pose_kpts_uv is not None:
            # Filter out points with zero coordinates
            valid_kpts = pose_kpts_uv[np.sum(np.abs(pose_kpts_uv), axis=1) > 0]
            
            # Draw each keypoint
            for i, (x, y) in enumerate(valid_kpts):
                cv2.circle(
                    result_frame,
                    (int(x), int(y)),
                    5,
                    DEBUG_UV_KPT_COLOR,
                    -1
                )
                
                # Optionally draw keypoint index
                if self.show_debug_info:
                    cv2.putText(
                        result_frame,
                        str(i),
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        DEBUG_UV_KPT_COLOR,
                        1
                    )
        
        # --- 3. Draw cell polygons
        if self.show_grid and cell_polygons:
            for i, polygon in enumerate(cell_polygons):
                # Draw cell polygon
                cv2.polylines(
                    result_frame,
                    [polygon.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )
                
                # Optionally draw cell index
                if self.show_debug_info:
                    # Calculate cell center
                    center = np.mean(polygon, axis=0).astype(int)
                    cv2.putText(
                        result_frame,
                        str(i),
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        
        # --- 4. Draw detected symbols
        if self.show_detections and detected_symbols:
            self.logger.debug("Drawing %s detected symbols", len(detected_symbols))
            for det_info in detected_symbols:
                try:
                    # Check if det_info is a dictionary with expected keys
                    if isinstance(det_info, dict) and all(k in det_info for k in ['label', 'confidence', 'box', 'class_id']):
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
                        label = "X" if class_id == 0 else "O"
                        
                        # Filter by confidence threshold
                        if conf >= getattr(self.config, 'bbox_conf_threshold', 0.5):
                            drawing_utils.draw_symbol_box(
                                result_frame, box, conf, class_id, label
                            )
                except Exception as e:
                    self.logger.error("Error drawing symbol: %s", e)
        
        # --- 5. Draw game state information if available
        if game_state and self.show_debug_info:
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
        scale_factor = self.debug_window_scale_factor
        debug_frame_height = int(frame.shape[0] * scale_factor)
        debug_frame_width = int(frame.shape[1] * scale_factor)
        debug_frame = cv2.resize(frame, (debug_frame_width, debug_frame_height))
        
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
            y_offset=20,
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1
        )
        
        # Display the debug window
        cv2.imshow('Debug', debug_frame)
