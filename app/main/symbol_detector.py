"""
Symbol detector module for the TicTacToe application.
"""
import logging
from typing import Tuple, List, Dict, Optional

import numpy as np
import cv2


class SymbolDetector:
    """Detects X and O symbols in the Tic Tac Toe grid."""

    def __init__(self, detect_model, config=None, logger=None):
        """Initialize the symbol detector.

        Args:
            detect_model: The YOLO detection model for symbol detection
            config: Configuration object
            logger: Logger instance
        """
        self.detect_model = detect_model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Symbol detection parameters
        self.bbox_conf_threshold = getattr(config, 'bbox_conf_threshold', 0.5)

        # Class ID to label mapping
        # CRITICAL FIX: YOLO model has inverted labels - it detects X as O and O as X
        # Original mapping: {0: 'X', 1: 'O'}
        # Corrected mapping to fix label inversion:
        self.class_id_to_label = getattr(config, 'class_id_to_label', {0: 'O', 1: 'X'})  # Swapped!
        self.class_id_to_player = getattr(config, 'class_id_to_player', {0: 1, 1: 2})  # Swapped! O=1, X=2

    def detect_symbols(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detects X and O symbols in the frame.

        Args:
            frame: The input frame

        Returns:
            Tuple of (processed_frame, symbols)
            where symbols is a list of dictionaries with keys:
            - label: 'X' or 'O'
            - confidence: detection confidence
            - box: [x1, y1, x2, y2] coordinates
            - class_id: class ID (0 for X, 1 for O)
        """
        # Use the detection model to find symbols
        detect_results = self.detect_model.predict(
            frame, conf=self.bbox_conf_threshold, verbose=False
        )

        # Process the results
        symbols = []

        # Check if we have any results
        if not detect_results or len(detect_results) == 0:
            self.logger.debug("No symbol detections found")
            return frame, symbols

        # Get the first result (assuming only one set of detections)
        result = detect_results[0]

        # Extract bounding boxes, confidences, and class IDs
        if hasattr(result, 'boxes') and result.boxes is not None:
            # YOLO v8 format
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Create symbol dictionaries
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                # Get label from class ID
                label = self.class_id_to_label.get(cls_id, f"Unknown-{cls_id}")

                # Create symbol dictionary
                symbol = {
                    'label': label,
                    'confidence': float(conf),
                    'box': box.tolist(),
                    'class_id': int(cls_id)
                }

                symbols.append(symbol)

                self.logger.debug("Detected %s with confidence %.2f at %s",
                                 label, conf, box.tolist())

        return frame, symbols

    def get_nearest_cell(self, x: float, y: float, cell_polygons: List[np.ndarray]) -> Optional[Tuple[int, int]]:
        """Finds the nearest cell to the given point.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            cell_polygons: List of cell polygons

        Returns:
            Tuple of (row, col) for the nearest cell, or None if no cell is found
        """
        if not cell_polygons:
            self.logger.warning("No cell polygons provided to get_nearest_cell")
            return None

        # Check if point is inside any cell
        point = np.array([x, y])

        # First check if the point is inside any cell
        for cell_idx, polygon in enumerate(cell_polygons):
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                # Point is inside this cell
                row = cell_idx // 3
                col = cell_idx % 3
                self.logger.debug("Point (%.1f, %.1f) is inside cell (%s, %s)",
                                 x, y, row, col)
                return row, col

        # If point is not inside any cell, find the nearest cell
        min_distance_sq = float('inf')
        nearest_cell_coords = None

        for cell_idx, polygon in enumerate(cell_polygons):
            # Calculate distance to cell center
            cell_center = np.mean(polygon, axis=0)
            dist_sq = (x - cell_center[0])**2 + (y - cell_center[1])**2

            if dist_sq < min_distance_sq:
                min_distance_sq = dist_sq
                row = cell_idx // 3
                col = cell_idx % 3
                nearest_cell_coords = (row, col)

        if nearest_cell_coords:
            self.logger.debug("Nearest cell to point (%.1f, %.1f) is (%s, %s) with distance %.1f",
                             x, y, nearest_cell_coords[0], nearest_cell_coords[1],
                             np.sqrt(min_distance_sq))
        else:
            self.logger.warning("Could not determine nearest cell for point (%.1f, %.1f)",
                               x, y)

        return nearest_cell_coords

    def assign_symbols_to_cells(self,
                               symbols: List[Dict],
                               cell_polygons: List[np.ndarray]) -> List[Tuple[int, int, str]]:
        """Assigns detected symbols to grid cells.

        Args:
            symbols: List of detected symbols
            cell_polygons: List of cell polygons

        Returns:
            List of (row, col, label) tuples for assigned symbols
        """
        if not symbols or not cell_polygons:
            return []

        assigned_symbols = []

        for symbol in symbols:
            # Get center point of the bounding box
            box = symbol['box']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # Find the nearest cell
            cell_coords = self.get_nearest_cell(center_x, center_y, cell_polygons)

            if cell_coords:
                row, col = cell_coords
                label = symbol['label']
                assigned_symbols.append((row, col, label))
                self.logger.debug("Assigned %s to cell (%s, %s)", label, row, col)

        return assigned_symbols
