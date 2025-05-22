"""Game detector module for Tic Tac Toe."""
import time
import logging
import math
from typing import Optional, Tuple, List, Dict

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from app.core.game_state import (
    GameState, GRID_POINTS_COUNT, EMPTY, TIE,
    PLAYER_X, PLAYER_O
)
from app.core.config import GameDetectorConfig
from app.core.constants import (
    ERROR_GRID_INCOMPLETE_PAUSE, MESSAGE_TEXT_COLOR, MESSAGE_BG_COLOR,
    GRID_PARTIALLY_VISIBLE_ERROR
)
from app.core.detector_constants import (
    BBOX_CONF_THRESHOLD, POSE_CONF_THRESHOLD, KEYPOINT_VISIBLE_THRESHOLD,
    MIN_POINTS_FOR_HOMOGRAPHY, RANSAC_REPROJ_THRESHOLD,
    GRID_DIST_STD_DEV_THRESHOLD, GRID_ANGLE_TOLERANCE_DEG,
    MAX_GRID_DETECTION_RETRIES, DEFAULT_DETECT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH,
    DEBUG_UV_KPT_COLOR, DEBUG_BBOX_COLOR, DEBUG_BBOX_THICKNESS, DEBUG_FPS_COLOR,
    IDEAL_GRID_NORM
)
from app.core.utils import FPSCalculator
from app.main import drawing_utils

# GUI Colors (for test block)
GUI_SIZE = 300
GUI_CELL_SIZE = GUI_SIZE // 3
GUI_GRID_COLOR = (255, 255, 255)
GUI_O_COLOR = (0, 255, 0)
GUI_X_COLOR = (0, 0, 255)
GUI_LINE_THICKNESS = 2
GUI_SYMBOL_THICKNESS = 4


class GameDetector:
    """Detects Tic Tac Toe grid and symbols using YOLO models."""

    # pylint: disable=too-many-positional-arguments, redefined-outer-name
    def __init__(
        self,
        config: GameDetectorConfig,
        camera_index: int = 0,
        o_val: int = 2,  # Default fallback value for O
        x_val: int = 1,  # Default fallback value for X
        detect_model_path: str = DEFAULT_DETECT_MODEL_PATH,
        pose_model_path: str = DEFAULT_POSE_MODEL_PATH,
        disable_autofocus: bool = True,
        device: Optional[str] = None,
        log_level=logging.INFO
    ):
        """Initializes the detector, loads models, and sets up the camera."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.camera_index = camera_index
        self.detect_model_path = detect_model_path
        self.pose_model_path = pose_model_path
        self.disable_autofocus = disable_autofocus
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.detect_model: Optional[YOLO] = None
        self.pose_model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width: int = 0
        self.frame_height: int = 0

        # Store game logic constants
        self.o_val = o_val
        self.x_val = x_val

        # Mapping from class index (from detection model) to player value (from
        # game logic)
        self.class_id_to_player: Dict[int, int] = {
            0: o_val,  # Assuming class 0 is 'O'
            1: x_val  # Assuming class 1 is 'X'
        }

        # --- Game State Instance --- #
        # Initialize GameState
        self.game_state = GameState()

        if self.config:
            self.max_grid_detection_retries = getattr(
                self.config, 
                'max_grid_detection_retries', 
                3
            )
            self.grid_lost_threshold_seconds = getattr(
                self.config, 
                'grid_lost_threshold_seconds', 
                2.0
            )
            # If GameState needed to be configured with values from self.config,
            # it would happen here using setter methods on self.game_state.
            # For example:
            # self.game_state.set_some_value(self.config.some_value)
        else:
            self.max_grid_detection_retries = 3  # Default
            self.grid_lost_threshold_seconds = 2.0  # Default
            self.logger.warning(
                "GameDetectorConfig not provided during __init__. "
                "GameState initialized, but some functionalities dependent "
                "on config (e.g., ideal grid points for drawing) may be "
                "limited or use defaults."
            )

        # Display settings for debug window
        self.show_detections = True  # Zobrazit detekce symbolů
        self.show_grid = True  # Zobrazit grid mřížku
        
        # Configurable thresholds (can be changed via debug window)
        self.bbox_conf_threshold = BBOX_CONF_THRESHOLD
        self.pose_conf_threshold = POSE_CONF_THRESHOLD

        # Log the current class_id_to_player mapping for clarity
        self.logger.info(
            "Symbol mapping configuration: X=%s, O=%s", self.x_val, self.o_val
        )
        self.logger.info(
            "Class ID to player mapping: %s", self.class_id_to_player
        )

        # Grid detection parameters
        self.grid_detection_retries = 0
        self.last_valid_grid_time: Optional[float] = None

        # For FPS calculation and display
        self.fps_calculator = FPSCalculator(buffer_size=10)
        self.last_log_time = time.time()
        self.log_interval = 5  # Log performance every 5 seconds

        # Cell polygons derived by GameDetector, primarily for drawing if
        # GameState doesn't provide them
        self._detector_derived_cell_polygons: Optional[List[np.ndarray]] = (
            None
        )

        self._load_models()
        self._setup_camera()

    def _load_models(self):
        """Loads the YOLO detection and pose estimation models."""
        # pylint: disable=logging-fstring-interpolation
        self.logger.info("Loading models (device: %s)...", self.device)
        try:
            self.detect_model = YOLO(self.detect_model_path)
            self.pose_model = YOLO(self.pose_model_path)
            self.detect_model.to(self.device)
            self.pose_model.to(self.device)
            self.logger.info("YOLO models loaded successfully.")
        except FileNotFoundError:
            self.logger.error(
                "!!!! Error: Model(s) not found: %s or %s !!!!",
                self.detect_model_path,
                self.pose_model_path
            )
            raise
        except Exception as e:
            self.logger.error("Error loading models: %s", e)
            raise

    def _setup_camera(self):
        """Initializes the camera capture."""
        self.logger.info("Initializing camera (index %s)...",
                         self.camera_index)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            self.logger.error(
                "Failed to open camera index %s.",
                self.camera_index
            )
            raise ConnectionError(
                "Camera %s not found or busy." % self.camera_index
            )

        if self.disable_autofocus:
            self.logger.info("Attempting to disable autofocus...")
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            time.sleep(2)  # Allow time for setting to apply
            autofocus_status = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
            self.logger.info("  Autofocus status: %s", autofocus_status)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = self.cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(
            "Cam %s: %sx%s @ %sFPS.",  # Shortened string
            self.camera_index,
            self.frame_width,
            self.frame_height,
            fps_cam
        )

    def _calculate_grid_homography(
        self, kpts_data_raw: np.ndarray
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """Calculates homography: ideal grid to detected points.
        Returns: Tuple[H_ideal_to_uv, valid_predicted_pts_uv]
        Uses RANSAC to handle cases where some points might be missing or
        occluded, as long as MIN_POINTS_FOR_HOMOGRAPHY are visible and
        reliable.
        Matches visible points to their corresponding ideal grid points using
        their original indices.
        """
        if kpts_data_raw is None or kpts_data_raw.shape[0] == 0:
            self.logger.debug("No keypoint data for homography.")
            return None, None

        valid_predicted_pts_uv = kpts_data_raw.astype(np.float32)
        valid_indices = np.arange(
            min(len(kpts_data_raw), len(IDEAL_GRID_NORM))
        )
        num_valid = len(valid_indices)

        self.logger.debug("Použití všech %s keypoints pro homografii.", num_valid)

        if num_valid < MIN_POINTS_FOR_HOMOGRAPHY:
            self.logger.warning(
                "Not enough valid kpts (%s < %s) for homography.",
                num_valid, MIN_POINTS_FOR_HOMOGRAPHY
            )
            return None, valid_predicted_pts_uv

        if valid_indices.max() >= len(IDEAL_GRID_NORM):
            self.logger.error(
                "Error: Valid keypoint index %s out of bounds for "
                "IDEAL_GRID_NORM (size %s).",
                valid_indices.max(), len(IDEAL_GRID_NORM)
            )
            return None, valid_predicted_pts_uv
        valid_ideal_pts = IDEAL_GRID_NORM[valid_indices]

        try:
            H_ideal_to_uv, ransac_mask = cv2.findHomography(
                valid_ideal_pts, valid_predicted_pts_uv, method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
            )

            if H_ideal_to_uv is None:
                self.logger.warning(
                    "cv2.findHomography failed (returned None)."
                )
                return None, valid_predicted_pts_uv

            num_inliers = np.sum(ransac_mask) if ransac_mask is not None else 0
            self.logger.debug(
                "Homography RANSAC found %s/%s inliers.",
                num_inliers, num_valid)

            if num_inliers < MIN_POINTS_FOR_HOMOGRAPHY:
                self.logger.warning(
                    "Homography found but has low inliers (%s < %s). "
                    "May be unreliable.",
                    num_inliers, MIN_POINTS_FOR_HOMOGRAPHY
                )

        except Exception as e:
            error_type = type(e).__name__
            is_cv_error = (error_type == 'error' and
                           hasattr(e, '__module__') and
                           e.__module__ == 'cv2')
            if is_cv_error:
                self.logger.error("OpenCV error in findHomography: %s", e)
            else:
                self.logger.exception("Error in findHomography: %s", e)
            return None, valid_predicted_pts_uv

        return H_ideal_to_uv, valid_predicted_pts_uv

    def _validate_grid_consistency(self, ordered_kpts_uv: np.ndarray) -> bool:
        if ordered_kpts_uv is None or \
           ordered_kpts_uv.shape != (GRID_POINTS_COUNT, 2):
            return False

        distances_h, distances_v, angles = [], [], []
        try:
            for r in range(4):
                for c in range(3):
                    idx1, idx2 = r * 4 + c, r * 4 + c + 1
                    distances_h.append(
                        np.linalg.norm(
                            ordered_kpts_uv[idx1] -
                            ordered_kpts_uv[idx2]))
            for c in range(4):
                for r in range(3):
                    idx1, idx2 = r * 4 + c, (r + 1) * 4 + c
                    distances_v.append(
                        np.linalg.norm(
                            ordered_kpts_uv[idx1] -
                            ordered_kpts_uv[idx2]))
            for r in range(1, 3):
                for c in range(1, 3):
                    p_left = ordered_kpts_uv[r * 4 + c - 1]
                    p_right = ordered_kpts_uv[r * 4 + c + 1]
                    p_up = ordered_kpts_uv[(r - 1) * 4 + c]
                    p_down = ordered_kpts_uv[(r + 1) * 4 + c]
                    vec_h, vec_v = p_right - p_left, p_down - p_up
                    norm_h, norm_v = np.linalg.norm(
                        vec_h), np.linalg.norm(vec_v)
                    if norm_h > 1e-6 and norm_v > 1e-6:
                        dot_prod = np.clip(
                            np.dot(vec_h / norm_h, vec_v / norm_v),
                            -1.0, 1.0
                        )
                        angles.append(math.degrees(math.acos(dot_prod)))
        except IndexError:
            self.logger.error("IndexError during grid consistency validation.")
            return False
        except Exception as e:
            self.logger.error("Error during grid consistency validation: %s", e)
            return False

        if not distances_h or not distances_v:
            self.logger.warning(
                "Could not calculate grid distances for validation.")
            return False
        std_dev_dist = np.std(np.array(distances_h + distances_v))
        if std_dev_dist > GRID_DIST_STD_DEV_THRESHOLD:
            self.logger.warning("Grid distance std dev too high: %.1f", std_dev_dist)
            return False
        self.logger.debug("Grid distance std dev OK: %.1f", std_dev_dist)

        if not angles:
            self.logger.warning(
                "Could not calculate grid angles for validation.")
            return False
        max_deviation = np.max(np.abs(np.array(angles) - 90.0))
        if max_deviation > GRID_ANGLE_TOLERANCE_DEG:
            self.logger.warning("Max angle deviation too high: %.1f", max_deviation)
            return False
        self.logger.debug("Grid angle dev OK: %.1f", max_deviation)

        self.logger.debug("Grid consistency validation passed.")
        return True

    def _sort_grid_points(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Sorts the 16 grid keypoints primarily by y-coordinate, then by x-coordinate."""
        if keypoints is None:
            self.logger.warning("_sort_grid_points: Received None for keypoints.")
            return None

        if keypoints.shape != (GRID_POINTS_COUNT, 2):
            self.logger.warning(
                "_sort_grid_points: Invalid shape %s for keypoints. Expected (%s, 2).",
                keypoints.shape, GRID_POINTS_COUNT
            )
            return None

        if np.all(keypoints == 0):
            self.logger.debug("Keypoints are all zeros, returning as is without sorting.")
            return keypoints

        points_list = [tuple(p) for p in keypoints]
        points_list.sort(key=lambda p: (p[1], p[0]))
        
        sorted_arr = np.array(points_list, dtype=np.float32)
        self.logger.debug("Sorted grid points: %s", sorted_arr.tolist())
        return sorted_arr

    def _derive_cell_polygons(
        self, sorted_16_grid_points: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """Derives 9 cell polygons from 16 sorted grid points."""
        if (sorted_16_grid_points is None or
                sorted_16_grid_points.shape != (GRID_POINTS_COUNT, 2)):
            self.logger.warning(
                "_derive_cell_polygons: Invalid input sorted_16_grid_points."
            )
            return None

        cell_polygons = []  
        for r_cell in range(3):
            for c_cell in range(3):
                p_tl_idx = r_cell * 4 + c_cell
                p_tr_idx = r_cell * 4 + (c_cell + 1)
                p_bl_idx = (r_cell + 1) * 4 + c_cell
                p_br_idx = (r_cell + 1) * 4 + (c_cell + 1)

                try:
                    cell_poly = np.array([
                        sorted_16_grid_points[p_tl_idx],
                        sorted_16_grid_points[p_tr_idx],
                        sorted_16_grid_points[p_br_idx],
                        sorted_16_grid_points[p_bl_idx]
                    ], dtype=np.float32)
                    cell_polygons.append(cell_poly)
                except IndexError:
                    self.logger.error(
                        "_derive_cell_polygons: IndexError for cell (%s,%s). "
                        "Indices: TL=%s, TR=%s, BL=%s, BR=%s",  # Reverted to original format string
                        r_cell, c_cell,
                        # p_tl_idx, etc., are indices for sorted_16_grid_points
                        p_tl_idx, p_tr_idx, p_bl_idx, p_br_idx
                    )
                    return None
        
        if len(cell_polygons) == 9:
            self.logger.debug(
                "_derive_cell_polygons: Successfully derived %s cell polygons.",
                len(cell_polygons)
            )
            return cell_polygons
        else:
            self.logger.error(
                "_derive_cell_polygons: Failed to derive all 9 cell polygons (got %s).",
                len(cell_polygons)
            )
            return None

    def _get_nearest_cell(
        self, point: List[float], sorted_grid_points: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        if sorted_grid_points is None:
            self.logger.warning("_get_nearest_cell: Received None for sorted_grid_points.")
            return None
        if sorted_grid_points.shape != (GRID_POINTS_COUNT, 2):
            self.logger.warning("_get_nearest_cell: Invalid shape %s for sorted_grid_points.",
                                sorted_grid_points.shape)
            return None
        if np.all(sorted_grid_points == 0):
            self.logger.debug("_get_nearest_cell: sorted_grid_points are all zeros, " 
                              "cannot determine nearest cell.")
            return None
        x, y = point
        min_distance_sq = float('inf')
        nearest_cell_coords = None
        self.logger.debug("_get_nearest_cell: Calculating centers for point (%s,%s)",
                          x, y)
        for r_cell in range(3):
            for c_cell in range(3):
                try:
                    tl_idx = r_cell * 4 + c_cell
                    tr_idx = r_cell * 4 + (c_cell + 1)
                    bl_idx = (r_cell + 1) * 4 + c_cell
                    br_idx = (r_cell + 1) * 4 + (c_cell + 1)
                    if not (0 <= tl_idx < GRID_POINTS_COUNT and \
                            0 <= tr_idx < GRID_POINTS_COUNT and \
                            0 <= bl_idx < GRID_POINTS_COUNT and \
                            0 <= br_idx < GRID_POINTS_COUNT):
                        self.logger.error(
                            "_get_nearest_cell: Index out of bounds. "
                            "tl:%s, tr:%s, bl:%s, br:%s, max_idx:%s",
                            tl_idx, tr_idx, bl_idx, br_idx, GRID_POINTS_COUNT -1
                        )
                        continue
                    p_tl = sorted_grid_points[tl_idx]
                    p_tr = sorted_grid_points[tr_idx]
                    p_bl = sorted_grid_points[bl_idx]
                    p_br = sorted_grid_points[br_idx]
                    center_x = (p_tl[0] + p_tr[0] + p_bl[0] + p_br[0]) / 4.0
                    center_y = (p_tl[1] + p_tr[1] + p_bl[1] + p_br[1]) / 4.0
                    self.logger.debug("  Cell (%s,%s) center: (%s,%s)",
                                      r_cell, c_cell, center_x, center_y)
                    distance_sq = (x - center_x)**2 + (y - center_y)**2
                    if distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        nearest_cell_coords = (r_cell, c_cell)
                except IndexError:
                    self.logger.error("  _get_nearest_cell: IndexError accessing sorted_grid_points for cell (%s,%s).",
                                      r_cell, c_cell)
                    continue
        if nearest_cell_coords:
            self.logger.debug("_get_nearest_cell: Point (%s,%s) assigned to cell %s (sq_dist: %s)",
                              x, y, nearest_cell_coords, min_distance_sq)
        else:
            self.logger.warning("_get_nearest_cell: Could not determine nearest cell for point (%s,%s).",
                                x, y)
        return nearest_cell_coords

    def _detect_symbols(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Detects X and O symbols in the frame.

        Args:
            frame: The input frame

        Returns:
            Tuple of (processed_frame, symbols)
            where symbols is a list of dictionaries with keys:
            - label: 'X' or 'O'
            - confidence: detection confidence
            - box: [x1, y1, x2, y2] coordinates
        """
        # Use the detection model to find symbols
        detect_results = self.detect_model.predict(
            frame, conf=self.bbox_conf_threshold, verbose=False
        )

        # Process the raw detections into a more usable format
        symbols = []
        if detect_results and isinstance(detect_results, list):
            for result in detect_results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes_data = result.boxes.data.cpu().numpy()
                    for box_data in boxes_data:
                        try:
                            if len(box_data) >= 6:
                                # Unpack box data
                                x1, y1, x2, y2 = box_data[0:4]
                                score = box_data[4]
                                class_id_float = box_data[5]
                                class_id = int(class_id_float)
                                x1, y1, x2, y2 = map(
                                    int, [x1, y1, x2, y2]
                                )

                                # Calculate center_uv
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                center_uv = (center_x, center_y)

                                # Map class_id to player symbol
                                player_val = self.class_id_to_player.get(
                                    class_id
                                )
                                # No default, let it be None if not found

                                player_symbol_str = None
                                if player_val == self.x_val:
                                    player_symbol_str = PLAYER_X
                                elif player_val == self.o_val:
                                    player_symbol_str = PLAYER_O

                                if player_symbol_str is None:
                                    self.logger.warning(
                                        "Unknown player_val %s for class_id %s. Skipping symbol.",
                                        player_val, class_id
                                    )
                                    continue  # Skip this symbol

                                # Log detection for debugging
                                self.logger.debug(
                                    "Detected %s (class_id=%s, player_val=%s) at (%s,%s,%s,%s) with confidence %s",
                                    player_symbol_str, class_id, player_val, x1, y1, x2, y2, score
                                )

                                symbols.append({
                                    'label': player_symbol_str,  # For display purposes
                                    'confidence': float(score),
                                    'box': [x1, y1, x2, y2],
                                    'class_id': class_id,
                                    'player': player_symbol_str,  # This is what GameState expects
                                    'center_uv': center_uv
                                })
                        except Exception as e:
                            self.logger.warning(
                                "Error processing box %s: %s",
                                box_data, e
                            )

        return frame, symbols

    def _detect_grid(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        if pose_results and isinstance(pose_results, list):
            for result in pose_results:
                if (
                    hasattr(result, 'keypoints')
                    and result.keypoints is not None
                ):
                    kpts_data = result.keypoints.data.cpu().numpy()
                    if kpts_data is not None and kpts_data.shape[0] > 0:
                        if kpts_data.shape[0] >= 1:
                            instance_kpts = kpts_data[0]
                            if instance_kpts.shape[0] >= GRID_POINTS_COUNT:
                                keypoints = instance_kpts[:GRID_POINTS_COUNT, :2]
                                self.logger.debug(
                                    "Úspěšně extrahováno %s bodů mřížky z YOLO Pose: %s",
                                    GRID_POINTS_COUNT, keypoints.tolist()
                                )
                            else:
                                self.logger.warning(
                                    "Nedostatek keypoints v prvé instanci: %s < %s",
                                    instance_kpts.shape[0], GRID_POINTS_COUNT
                                )
                        else:
                            self.logger.warning(
                                "Nenalezena žádná instance s keypoints v datech z YOLO Pose."
                            )
                    else:
                        self.logger.warning(
                            "Neplatný formát kpts_data (None nebo prázdné pole) z YOLO Pose."
                        )
                else:
                    self.logger.debug(
                        "Výsledek z YOLO Pose neobsahuje atribut 'keypoints' nebo je None."
                    )
        else:
            self.logger.debug(
                "Nebyly detekovány žádné výsledky z YOLO Pose, nebo formát není list."
            )
        return frame, keypoints

    def _is_valid_grid(self, keypoints: np.ndarray) -> bool:
        if keypoints is None or keypoints.shape != (GRID_POINTS_COUNT, 2):
            self.logger.debug(
                "_is_valid_grid: Invalid shape or None. Shape: %s",
                keypoints.shape if keypoints is not None else 'None'
            )
            return False

        # Spočítáme všechny body, které jsou [0, 0] nebo téměř nula
        zero_count = 0
        if keypoints is not None and keypoints.shape[0] == GRID_POINTS_COUNT:
            # Spočítáme všechny body, které jsou [0, 0] nebo téměř nula
            for i in range(GRID_POINTS_COUNT):
                if np.all(np.abs(keypoints[i]) < 0.1):  # Bod je nulový nebo
                    # téměř nulový
                    zero_count += 1

            # Uložíme počet neviditelných bodů pro použití v GUI
            self.game_state.missing_grid_points_count = zero_count
            if zero_count > 0:
                self.logger.warning(
                    "Grid has %s missing points (zero values)",
                    zero_count
                )

            # Pokud chybí příliš mnoho bodů, mřížka není platná
            if zero_count > 4:  # Pokud chybí více než 4 body, nemůžeme ji
                # použít
                self.logger.error(
                    "Too many missing points (%s), grid is invalid",
                    zero_count
                )
                self.game_state.grid_fully_visible = False
                self.game_state.error_message = GRID_PARTIALLY_VISIBLE_ERROR
                return False

            # Pokud je poslední bod [0,0] nebo blízký nule, pokusíme se ho
            # doplnit
            if np.all(np.abs(keypoints[-1]) < 0.1):  # Poslední bod je nulový
                # nebo téměř nulový
                self.logger.warning(
                    "Last grid point is missing (zero). Attempting to reconstruct it."
                )

                # Pokud chybí pouze jeden až dva body, můžeme je rekonstruovat
                if zero_count <= 2:
                    # Nejjednodušší rekonstrukce - odhadneme poslední bod z
                    # jeho sousedů
                    # Poslední (pravý dolní) můžeme odhadnout z bodů vlevo a
                    # nad ním
                    # Zejména bodů 14 (bod vlevo) a 11 (bod nad ním)
                    has_left = not np.all(np.abs(keypoints[14]) < 0.1)
                    has_top = not np.all(np.abs(keypoints[11]) < 0.1)

                    if has_left and has_top:
                        # Máme body vlevo i nad => můžeme doplnit bod
                        # diagonálně
                        # Vypočet: bod 15 = bod 14 + (bod 11 - bod 10)
                        # kde bod 10 je vrchní levý sousední (diagonálně
                        # vzhledem k 15)
                        if not np.all(np.abs(keypoints[10]) < 0.1):
                            diagonal_vector = keypoints[11] - keypoints[10]  # V
                            # ektor z 10 do 11
                            keypoints[-1] = keypoints[14] + diagonal_vector
                            self.logger.info(
                                "Reconstructed missing point %s (diagonal): %s",
                                GRID_POINTS_COUNT-1, keypoints[-1]
                            )
                            zero_count -= 1  # Snížíme počet chybějících bodů
                        else:
                            # Jednodušší metoda - extrapolace z bodů 13 a 14
                            # (poslední řada)
                            if not np.all(np.abs(keypoints[13]) < 0.1):
                                right_vector = keypoints[14] - keypoints[13]  # V
                                # ektor zprava doleva (negat.)
                                keypoints[-1] = keypoints[14] + right_vector
                                self.logger.info(
                                    "Reconstructed missing point %s (horizontal): %s",
                                    GRID_POINTS_COUNT-1, keypoints[-1]
                                )
                                zero_count -= 1  # Snížíme počet chybějících
                                # bodů

            # Pokud po rekonstrukci stále chybí nějaký bod, nastavíme příznak
            # pro varování
            if zero_count > 0:
                self.game_state.grid_fully_visible = False
                self.game_state.error_message = GRID_PARTIALLY_VISIBLE_ERROR
            else:
                self.game_state.grid_fully_visible = True
                # If the grid is now fully visible, clear the specific error if it was set
                if self.game_state.error_message == GRID_PARTIALLY_VISIBLE_ERROR:
                    self.game_state.error_message = None

        # Standardní kontrola konzistence
        is_consistent = self._validate_grid_consistency(keypoints)
        self.logger.debug("_is_valid_grid: Grid consistency check returned: %s", is_consistent)
        return is_consistent

    def _draw_centered_text_message(self, frame: np.ndarray, message_lines: list[str],
                                font_scale: float = 1.2, thickness: int = 2,
                                text_color: tuple[int, int, int] = MESSAGE_TEXT_COLOR,
                                bg_color: tuple[int, int, int] = MESSAGE_BG_COLOR,
                                bg_alpha: float = 0.85) -> None:                   # More opaque
        """Draws a multi-line text message centered on the frame using drawing_utils."""
        drawing_utils.draw_centered_text_message(
            frame,
            message_lines,
            font_scale=font_scale,
            thickness=thickness,
            text_color=text_color,
            bg_color=bg_color,
            bg_alpha=bg_alpha
        )

    def _draw_detection_results(
        self, frame: np.ndarray,
        fps: float,                        # Current frames per second
        pose_kpts_uv: Optional[np.ndarray],  # Raw keypoints from YOLO (16 points)
        ordered_kpts_uv: Optional[np.ndarray],  # Validated & sorted keypoints (16 points)
        # Derived 9 cell polygons (can be from GameState or GameDetector)
        current_derived_cell_polygons: Optional[List[np.ndarray]],
        detected_symbols: List[Dict],      # List of symbol dicts {'label', 'box', 'confidence', 'class_id'}
        H_ideal_to_uv: Optional[np.ndarray]  # Homography matrix (optional)
    ) -> np.ndarray:
        """Draw detection results onto the frame."""
        # Check if game is paused due to incomplete grid
        if self.game_state and self.game_state.is_game_paused_due_to_incomplete_grid():
            error_message_key = self.game_state.get_error_message()
            if error_message_key == ERROR_GRID_INCOMPLETE_PAUSE:
                message_lines = [
                    "⚠️ CHYBA DETEKCE ⚠️",
                    "Nebyla detekována herní mřížka",
                    "Umístěte celou herní plochu",
                    "do záběru kamery"
                ]
                self._draw_centered_text_message(
                    frame,
                    message_lines,
                    text_color=MESSAGE_TEXT_COLOR,
                    bg_color=MESSAGE_BG_COLOR
                )
            # Potentially other error messages could be handled here too
            return frame  # Return early, no other drawing needed
        
        # --- 1. Draw raw grid points from pose estimation (pose_kpts_uv)
        if self.show_grid and pose_kpts_uv is not None:
            self.logger.debug("Drawing RAW POSE grid points.")
            for i, pt_data in enumerate(pose_kpts_uv):
                # Assuming pt_data could be (x, y) or (x, y, conf, class_id_of_point_type)
                # For grid points, we usually just have (x,y) from pose model.
                # If pose_kpts_uv structure is more complex, adjust here.
                if len(pt_data) >= 2:
                    u, v = int(pt_data[0]), int(pt_data[1])
                    # Check if point is within frame boundaries
                    if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                        cv2.circle(frame, (u, v), 3, (255, 100, 100), -1) # Light Blue for raw pose points
                        # cv2.putText(frame, str(i), (u + 4, v - 4),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,100,100), 1)
                else:
                    self.logger.warning("Malformed point data in pose_kpts_uv: %s", pt_data)

        # --- 2. Draw processed (validated & ordered) grid points (ordered_kpts_uv)
        if self.show_grid and ordered_kpts_uv is not None and \
           ordered_kpts_uv.shape == (GRID_POINTS_COUNT, 2) and \
           not np.all(ordered_kpts_uv == 0):
            self.logger.debug("Drawing VALIDATED & ORDERED grid points.")
            for i, (x, y) in enumerate(ordered_kpts_uv):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green for ordered/valid
                cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)

        # --- 3. Draw derived cell polygons (current_derived_cell_polygons)
        if self.show_grid and current_derived_cell_polygons is not None:
            self.logger.debug("Drawing %s derived_cell_polygons.",
                              len(current_derived_cell_polygons))
            for i, polygon in enumerate(current_derived_cell_polygons):
                if polygon is not None and polygon.shape[0] == 4 and polygon.shape[1] == 2:
                    # OpenCV's polylines function expects points as int32
                    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2) # Magenta polygons
                    
                    # Optional: Label the cells with their index or board state
                    if self.game_state:
                        r, c = i // 3, i % 3
                        cell_state = self.game_state.board[r][c]
                        label = "%s" % (cell_state if cell_state != EMPTY else i)
                        # Calculate centroid for label placement
                        M = cv2.moments(pts)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(frame, label, (cX - 7, cY + 7),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                else:
                    self.logger.warning("Skipping drawing invalid polygon at index %s: %s", i, polygon)

        # --- 4. Draw detected symbols (bounding boxes with labels and confidence)
        if self.show_detections and detected_symbols:
            self.logger.debug("Drawing %s detected symbols.",
                              len(detected_symbols))
            for det_info in detected_symbols:
                try:
                    # Pokud je det_info typu list a obsahuje 6 prvků (x1,y1,x2,y2,conf,class_id),
                    # použijeme přímo tyto hodnoty
                    if isinstance(det_info, (list, tuple)) and len(det_info) >= 6:
                        x1, y1, x2, y2, conf_val, class_id_val = det_info[:6]
                        box = [x1, y1, x2, y2]
                        conf = conf_val
                        class_id = class_id_val
                        label = "X" if class_id_val == 0 else "O"
                        
                        # Filter by confidence threshold
                        if conf >= self.bbox_conf_threshold:
                            drawing_utils.draw_symbol_box(frame, box, conf, class_id, label)
                        continue
                    
                    # Pro slovníkové objekty
                    box = det_info.get('box') # [x1, y1, x2, y2]
                    conf = det_info.get('confidence')
                    class_id = det_info.get('class_id')
                    label = det_info.get('label')
                    
                    if not all([box, conf is not None]):
                        self.logger.error("Malformed symbol dict: %s. Skipping draw.", det_info)
                        continue
                    
                    # Filter by confidence threshold
                    if conf < self.bbox_conf_threshold:
                        continue
                    
                    # If class_id is not provided, try to infer from label
                    if class_id is None and label:
                        class_id = 0 if label.upper() == "X" else 1
                    
                    # If label is not provided, try to infer from class_id
                    if label is None and class_id is not None:
                        label = "X" if class_id == 0 else "O"
                    
                    # Log detailed bbox information for debugging
                    x1, y1, x2, y2 = map(int, box)
                    self.logger.debug("Symbol: %s, Class ID: %s, Confidence: %s, Bbox: [(%s,%s), (%s,%s)]",
                                      label, class_id, conf, x1, y1, x2, y2)
                    
                    drawing_utils.draw_symbol_box(frame, box, conf, class_id, label)

                except (ValueError, TypeError, IndexError, AttributeError) as e:
                    self.logger.error("Error drawing symbol %s: %s", det_info, e)
                    continue  # Skip to next detection
        
        # --- 5. Draw FPS
        fps_text = "FPS: %s" % fps
        # Get frame dimensions for FPS text placement
        # (frame_h, frame_w) = frame.shape[:2]
        cv2.putText(frame, fps_text, (10, 30), # Adjusted y position for clarity
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- 6. Display Game State information (optional, can be toggled)
        if self.game_state and self.config.show_game_state_on_frame:
            board_str_list = ["".join(row) for row in self.game_state.board]
            text_y_offset = 60 # Starting Y position for game state text
            last_line_idx = 0 # Default for positioning winner/draw text
            for i, line_str in enumerate(board_str_list):
                cv2.putText(frame, line_str, (10, text_y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                last_line_idx = i # Keep track of the last index used
            
            game_winner_status = self.game_state.get_winner() # Returns 'X', 'O', 'Draw', or None

            # Position winner/draw text below the board state display
            # Use last_line_idx to ensure it's correctly placed even if board_str_list was unexpectedly empty
            status_text_y = text_y_offset + (last_line_idx + 2) * 20
            if len(board_str_list) == 0: # If board_str_list was empty, adjust base position
                status_text_y = text_y_offset + 2 * 20

            if game_winner_status and game_winner_status != TIE:  # Check against TIE constant
                winner_text = "Winner: %s" % game_winner_status
                cv2.putText(frame, winner_text, (10, status_text_y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 120), 2)
            elif game_winner_status == TIE:  # Check against TIE constant
                draw_text = "Draw!"
                cv2.putText(frame, draw_text, (10, status_text_y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 100, 0), 2)

        # --- 7. Draw ideal grid projected by homography (if H is available)
        if self.show_grid and H_ideal_to_uv is not None:
            self.logger.debug("Drawing IDEAL grid projected by homography.")
            ideal_kps_for_drawing = None

            if self.config and hasattr(self.config, 'ideal_grid_keypoints_4x4'):
                ideal_kps_for_drawing = self.config.ideal_grid_keypoints_4x4
            
            if ideal_kps_for_drawing is None:
                self.logger.warning(
                    "_draw_detection_results: Ideal keypoints not available from config. Using generic points for homography drawing."
                )
                # Last resort generic points (e.g., a 4x4 grid in a 0-3 range)
                ideal_kps_for_drawing = np.array([
                    [x, y] for y in range(4) for x in range(4)], 
                    dtype=np.float32
                )
            
            # Define connections for the 3x3 cell grid (lines between the 4x4 points)
            # These are indices into the 16 ideal_grid_kps
            grid_lines = [
                # Horizontal lines
                (0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7),
                (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15),
                # Vertical lines
                (0, 4), (4, 8), (8, 12), (1, 5), (5, 9), (9, 13),
                (2, 6), (6, 10), (10, 14), (3, 7), (7, 11), (11, 15)
            ]

            if ideal_kps_for_drawing.shape == (self.config.grid_points_count, 2):
                projected_kps = cv2.perspectiveTransform(
                    ideal_kps_for_drawing.reshape(-1, 1, 2), H_ideal_to_uv
                )
                if projected_kps is not None:
                    projected_kps = projected_kps.reshape(-1, 2).astype(np.int32)
                    for p1_idx, p2_idx in grid_lines:
                        if (0 <= p1_idx < len(projected_kps) and 
                            0 <= p2_idx < len(projected_kps)):
                            pt1 = tuple(projected_kps[p1_idx])
                            pt2 = tuple(projected_kps[p2_idx])
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 1) # Cyan
                        else:
                            self.logger.warning(
                                "Homography line drawing: index out of bounds. p1_idx=%s, p2_idx=%s, len=%s",
                                p1_idx, p2_idx, len(projected_kps)
                            )
                else:
                    self.logger.warning(
                        "cv2.perspectiveTransform returned None for homography points."
                    )
            else:
                self.logger.warning(
                    "Could not draw homography-projected grid: ideal_kps shape %s mismatch or not %s points.",
                    ideal_kps_for_drawing.shape, self.config.grid_points_count
                )

        return frame

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def process_frame(self,
                      frame: np.ndarray,
                      frame_time: float) -> Tuple[np.ndarray,
                                                  Optional[GameState]]:
        """Processes a single frame: detects grid, symbols, updates state."""

        # Start FPS counter for processing time
        start_time_processing = time.perf_counter()

        # --- 1. Detect Symbols (X/O) --- #
        frame_for_detect = frame.copy()
        _, detected_symbols_data = self._detect_symbols(frame_for_detect)
        processed_symbols_list = []
        if detected_symbols_data:
            for det_info in detected_symbols_data:
                if isinstance(det_info, dict) and all(
                    k in det_info for k in ['label', 'confidence', 'box']):
                    processed_symbols_list.append(det_info)
                else:
                    try:
                        if len(det_info) >= 6: # Assuming (x1,y1,x2,y2,conf,class_id)
                            label = self.config.class_id_to_label.get(int(det_info[5]), "UNK")
                            processed_symbols_list.append({
                                'box': list(map(float, det_info[0:4])),
                                'confidence': float(det_info[4]),
                                'class_id': int(det_info[5]),
                                'label': label
                            })
                        else:
                            self.logger.warning(
                                "Skipping malformed raw symbol data: %s",
                                det_info
                            )
                    except (TypeError, ValueError, IndexError, AttributeError) as e:
                        self.logger.warning(
                            "Error processing raw symbol data %s: %s",
                            det_info, e
                        )
                    # continue  # Skip to next detection
        
        # --- 2. Detect Grid --- #
        frame_for_pose = frame.copy()
        _, raw_kpts = self._detect_grid(frame_for_pose)

        # --- 3. Sort and Validate Grid Keypoints --- #
        sorted_kpts = self._sort_grid_points(raw_kpts)
        grid_is_valid_and_ordered = self._is_valid_grid(sorted_kpts)

        final_kpts_for_processing: Optional[np.ndarray] = None
        current_H_ideal_to_uv: Optional[np.ndarray] = None  # Keep for backwards compatibility
        grid_status_changed = False

        current_time = time.time()

        if grid_is_valid_and_ordered and sorted_kpts is not None:
            self.logger.debug("Sorted grid points are valid and consistent.")
            final_kpts_for_processing = sorted_kpts
            
            # We still calculate homography for backwards compatibility
            # But the new GameState implementation doesn't rely on it
            current_H_ideal_to_uv, _ = self._calculate_grid_homography(
                final_kpts_for_processing)
            if current_H_ideal_to_uv is None:
                self.logger.debug(
                    "Homography calculation failed, but will use direct grid points mapping instead.")
            
            if self.last_valid_grid_time is None: # Grid was just found
                grid_status_changed = True 
            self.last_valid_grid_time = current_time
            self.grid_detection_retries = 0 # Reset retries on valid grid
            
            # Důležité: když je mřížka vidět kompletně, vyčistíme příznaky problémů s mřížkou
            # v game_state, pokud existují
            if hasattr(self, 'game_state') and self.game_state and not self.game_state.grid_fully_visible:
                if hasattr(self.game_state, 'grid_issue_type'):
                    self.logger.info("Grid is fully visible now. Clearing grid_issue_type attribute.")
                    delattr(self.game_state, 'grid_issue_type')
                if hasattr(self.game_state, 'grid_issue_message'):
                    delattr(self.game_state, 'grid_issue_message')

        else: # Grid is not valid or not found
            # Kontrolujeme důvod nevalidní mřížky
            if hasattr(self, 'game_state') and self.game_state and not self.game_state.grid_fully_visible:
                # Mřížka byla vidět, ale některé body chybí (je mimo záběr)
                missing_count = getattr(self.game_state, 'missing_grid_points_count', 0)
                self.logger.warning(
                    "Grid has %s missing points (zero values)",
                    missing_count
                )

                # Nastavit příznak pro GUI, aby zobrazilo varování
                if hasattr(self.game_state, 'grid_issue_type'):
                    self.logger.info("Grid is partially visible. Setting grid_issue_type attribute.")
                    setattr(self.game_state, 'grid_issue_type', 'incomplete_visibility')
                    setattr(self.game_state, 'grid_issue_message', 
                            "Hrací plocha není celá v záběru kamery! Chybí %s bodů. Narovnejte hrací plochu a umístěte ji do středu záběru." % missing_count)
            else:
                # Standardní hhlášení o neviditelné mřížce
                self.logger.debug(
                    "PROCESS_FRAME: Grid is not valid/found. final_kpts_for_processing=None."
                )
                
                # Nastavit příznak pro GUI, aby zobrazilo varování že není vidět žádná mřížka
                if hasattr(self.game_state, 'grid_issue_type'):
                    self.logger.info("Grid is not visible. Setting grid_issue_type attribute.")
                    setattr(self.game_state, 'grid_issue_type', 'grid_not_found')
                    setattr(self.game_state, 'grid_issue_message', 
                            "Hrací plocha není v záběru kamery! Umístěte ji do středu záběru.")
                    
            if self.last_valid_grid_time is not None: # Grid was just lost
                if (current_time - self.last_valid_grid_time) > self.grid_lost_threshold_seconds:
                    self.logger.warning(
                        "Grid lost for %s seconds. Resetting GameState and retrying grid detection.",
                        self.grid_lost_threshold_seconds
                    )
                    if self.game_state:
                        try:
                            # Použijeme try-except, protože reset_game() může být nová metoda
                            self.game_state.reset_game() # Full reset if grid lost too long
                        except AttributeError as e:
                            self.logger.warning("Could not reset game state: %s. Using fallback.", e)
                            # Fallback - pokud reset_game() neexistuje, vytvoříme nový game state objekt
                            from app.core.game_state import GameState
                            self.game_state = GameState()
                    self.last_valid_grid_time = None # Mark as lost
                    grid_status_changed = True
                    # final_kpts_for_processing and current_H_ideal_to_uv remain None
                else:
                    # Grid recently lost, keep using last known good state for a bit if configured
                    # For now, we don't persist old keypoints/homography if detector can't see it.
                    pass 
            # If grid was never valid or already marked as lost, continue with None for kpts/H

        # --- 4. Update Game State --- #
        polygons_from_gs = self._update_game_state(
            frame.copy(),
            final_kpts_for_processing,
            current_H_ideal_to_uv,
            processed_symbols_list,
            self.class_id_to_player,
            frame_time,
            grid_status_changed # Pass grid status change flag
        )
        # Store polygons from GameState if available, otherwise use GameDetector's derived ones
        self._detector_derived_cell_polygons = polygons_from_gs 

        # Calculate FPS for display using the FPSCalculator instance
        self.fps_calculator.tick()
        fps = self.fps_calculator.get_fps()
        
        # Log performance periodically
        if current_time - self.last_log_time > self.log_interval:
            end_time_processing = time.perf_counter()
            processing_time_ms = (end_time_processing - start_time_processing) * 1000
            self.logger.info("Performance: Current FPS: %s, Processing Time: %s ms",
                             fps, processing_time_ms)
            self.last_log_time = current_time

        # --- 5. Draw Detection Results --- #
        frame_to_draw_on = frame.copy()
        annotated_frame = self._draw_detection_results(
            frame_to_draw_on,
            fps,                                  # Pass current FPS
            raw_kpts,                             # Raw keypoints from pose model for drawing
            final_kpts_for_processing,            # Validated & sorted keypoints
            self._detector_derived_cell_polygons, # Polygons (from GS or GD)
            processed_symbols_list,               # Detected symbols with their info
            current_H_ideal_to_uv                 # Homography matrix
        )

        return annotated_frame, self.game_state

    def _update_game_state(
            self,
            frame: np.ndarray,
            ordered_kpts_uv: Optional[np.ndarray],
            homography: Optional[np.ndarray],
            detected_symbols: List, # Can be list of dicts or list of raw tuples
            class_id_to_player: Dict[int, int],
            timestamp: float,
            grid_status_changed: bool = False # New parameter
            ) -> Optional[List[np.ndarray]]:
        """Updates the game state by calling GameState.update_from_detection.

        Also handles GameState reset if the grid status significantly changes.

        Args:
            frame: The current video frame.
            ordered_kpts_uv: Ordered keypoints in UV coordinates (image space).
            homography: Homography matrix from ideal grid to image (ideal -> UV).
            detected_symbols: List of detected symbols (dicts or raw tuples).
            class_id_to_player: Mapping from class ID to player value.
            timestamp: Current timestamp.
            grid_status_changed: True if grid was just found or lost beyond threshold.
        
        Returns:
            Optional list of derived cell polygons if they were computed by GameState.
        """
        if self.game_state is None:
            self.game_state = GameState() 
            self.logger.info(
                "Initialized new GameState in _update_game_state."
            )
        elif grid_status_changed and ordered_kpts_uv is None: # Grid lost and confirmed lost
            self.logger.warning(
                "Grid lost, resetting game state inside _update_game_state."
            )
            self.game_state.reset_game() # Full reset if grid is lost
        
        # Always reset changed cells for the current detection cycle, 
        # regardless of full game reset.
        self.game_state.reset_changed_cells()

        # Delegate actual update logic to GameState
        self.game_state.update_from_detection(
            frame,
            ordered_kpts_uv,
            homography,
            detected_symbols,
            self.class_id_to_player,  # Použijeme vlastní mapování
            timestamp
        )

        # Retrieve derived polygons from GameState if available for drawing
        # Assumes GameState has get_latest_derived_cell_polygons()
        if hasattr(self.game_state, 'get_latest_derived_cell_polygons'):
            polygons = self.game_state.get_latest_derived_cell_polygons()
            if polygons is not None:
                return polygons
        
        # Fallback: if GameState doesn't provide them but we have keypoints, 
        # GameDetector can derive them for drawing.
        # This is only for drawing, GameState handles its own internal grid representation.
        if ordered_kpts_uv is not None:
            self.logger.debug(
                "_update_game_state: GameState did not provide polygons, "
                "GameDetector deriving for drawing."
            )
            return self._derive_cell_polygons(ordered_kpts_uv)
        
        return None

    # Main loop for running detection and displaying results.
    def run_detection(self):
        """Main loop for running detection and displaying results."""
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("Failed to grab frame from camera.")
                time.sleep(0.5)  # Wait a bit before retrying
                continue

            frame_time = time.time()  # Get timestamp for the current frame
            try:
                processed_frame, current_game_state = self.process_frame(
                    frame, frame_time)

                # Display the processed frame
                cv2.imshow('Tic Tac Toe Detection', processed_frame)

                # Print board state if valid
                if current_game_state and current_game_state.is_valid():
                    # Log the board state periodically or on change (already
                    # logged in GameState)
                    # Logging happens within game_state.update_from_detection
                    pass
                else:
                    # Optionally log when the grid is not detected or invalid
                    self.logger.debug("Waiting for valid grid detection...")

            except Exception as e:
                self.logger.exception("Error during frame processing: %s", e)

            key = cv2.waitKey(1) & 0xFF
            # Check for q, Q, or ESC key (27) to exit
            if key == ord('q') or key == ord('Q') or key == 27:
                self.logger.info("Quit key pressed.")
                break

    def release(self):
        """Releases the camera and destroys windows."""
        self.logger.info("Releasing camera resource...")
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released.")
        else:
            self.logger.info(
                "Camera was not open or capture object did not exist."
            )

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        if hasattr(self, 'cap') and self.cap:
            self.release()

    def _draw_debug_info(self, frame: np.ndarray, fps: float) -> None:
        """Draws debug information like FPS on the frame."""
        if not self.show_debug_info:
            return

        # Scale down the frame for the debug window
        scale_factor = self.config.debug_window_scale_factor
        debug_frame_height = int(frame.shape[0] * scale_factor)
        debug_frame_width = int(frame.shape[1] * scale_factor)
        debug_frame = cv2.resize(frame, (debug_frame_width, debug_frame_height))

        # Prepare text lines for debug info
        winner_text = self.game_state.winner if self.game_state.winner else 'None'
        grid_points_val = (
            len(self.game_state.grid_points)
            if self.game_state.grid_points is not None else 0
        )
        cell_polygons_val = (
            len(self.game_state.cell_polygons)
            if self.game_state.cell_polygons is not None else 0
        )

        texts_to_draw = [
            f"FPS: {fps:.2f}",  # Using f-string for better formatting
            f"Game Status: {self.game_state.status.value}",
            f"Winner: {winner_text}",
            f"Grid Visible: {self.game_state.is_grid_visible}",
            f"Grid Stable: {self.game_state.is_grid_stable}",
            f"Grid Points: {grid_points_val}/{GRID_POINTS_COUNT}",
            f"Cells: {cell_polygons_val}/9"
        ]

        # Position for the text (top-left corner)
        text_x = 10
        text_y = 20

        drawing_utils.draw_text_lines(
            debug_frame,
            texts_to_draw,
            text_x,
            text_y,
            y_offset=20,  # Default is 20, can be omitted if preferred
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1
        )

        # Display the debug window
        cv2.imshow('Debug', debug_frame)


# --- Integrated Test Block --- #
if __name__ == '__main__':
    # Configure logging
    # Use DEBUG for detailed info, INFO for less verbose
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    detector = None
    try:
        # Pass log level to detector
        config = GameDetectorConfig()
        detector = GameDetector(
            config=config, camera_index=0, log_level=log_level)
        detector.run_detection()
    except (ConnectionError, FileNotFoundError, Exception) as e:
        logger.error("Initialization or Runtime Error: %s", e)
    finally:
        if detector and detector.cap:
            detector.release()
        cv2.destroyAllWindows()
        logger.info("Application exited.")
