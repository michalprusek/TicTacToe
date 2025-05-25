# Computer Vision Pipeline Documentation

Detailed documentation of the computer vision system for TicTacToe game detection.

## Overview

The computer vision pipeline uses a two-stage YOLO-based approach to detect the game board and symbols:

1. **Grid Detection**: Pose estimation to find 16 keypoints forming a 4×4 grid
2. **Symbol Detection**: Object detection to identify X and O symbols within grid cells

## Models

### Grid Detection Model (`weights/best_pose.pt`)
- **Architecture**: YOLOv8 Pose Estimation
- **Input**: 640×640 RGB images
- **Output**: 16 keypoints representing grid intersection points
- **Training**: Custom dataset with annotated grid corners and intersections

### Symbol Detection Model (`weights/best_detection.pt`)
- **Architecture**: YOLOv8 Object Detection
- **Input**: 640×640 RGB images  
- **Output**: Bounding boxes and classes for X and O symbols
- **Classes**: 
  - 0: X symbol
  - 1: O symbol

## Detection Pipeline

### Stage 1: Frame Preprocessing

```python
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    # 1. Resize to model input size
    resized = cv2.resize(frame, (640, 640))
    
    # 2. Normalize pixel values
    normalized = resized / 255.0
    
    # 3. Convert BGR to RGB (YOLO expects RGB)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    
    return rgb
```

### Stage 2: Grid Detection

```python
def detect_grid(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Run pose model
    pose_results = self.pose_model.predict(
        frame, 
        conf=self.pose_conf_threshold,
        verbose=False
    )
    
    # Extract keypoints
    if pose_results and len(pose_results) > 0:
        result = pose_results[0]
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()[0]
            return frame, keypoints
    
    return frame, np.zeros((16, 2), dtype=np.float32)
```

### Stage 3: Grid Point Sorting

Critical for correct cell mapping - sorts detected points into canonical 4×4 grid order:

```python
def sort_grid_points(self, keypoints: np.ndarray) -> np.ndarray:
    valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]
    
    if len(valid_points) < 4:
        return keypoints
    
    # Sort by Y coordinate (top to bottom)
    y_sorted = valid_points[np.argsort(valid_points[:, 1])]
    
    # Group into rows (adaptive for partial grids)
    if len(valid_points) >= 12:
        # Full grid: 4 rows
        points_per_row = len(valid_points) // 4
        remainder = len(valid_points) % 4
        sorted_points = []
        
        current_idx = 0
        for row in range(4):
            row_size = points_per_row + (1 if row < remainder else 0)
            row_points = y_sorted[current_idx:current_idx + row_size]
            # Sort row by X coordinate (left to right)
            x_sorted = row_points[np.argsort(row_points[:, 0])]
            sorted_points.extend(x_sorted)
            current_idx += row_size
    else:
        # Partial grid: lexicographic sort
        sorted_points = valid_points[np.lexsort((valid_points[:, 0], valid_points[:, 1]))]
    
    # Update keypoints array
    result = np.zeros_like(keypoints)
    result[:len(sorted_points)] = sorted_points
    return result
```

### Stage 4: Grid Validation

```python
def is_valid_grid(self, keypoints: np.ndarray) -> bool:
    valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]
    
    # Minimum points required
    if len(valid_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return False
    
    # Check distance variance (points should be reasonably spaced)
    distances = []
    for i in range(len(valid_points)):
        for j in range(i+1, len(valid_points)):
            dist = np.linalg.norm(valid_points[i] - valid_points[j])
            distances.append(dist)
    
    if len(distances) > 0:
        std_dev = np.std(distances)
        mean_dist = np.mean(distances)
        if std_dev / mean_dist > GRID_DIST_STD_DEV_THRESHOLD:
            return False
    
    return True
```

### Stage 5: Symbol Detection

```python
def detect_symbols(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    # Run detection model
    detection_results = self.detect_model.predict(
        frame,
        conf=self.detection_conf_threshold,
        verbose=False
    )
    
    symbols = []
    if detection_results and len(detection_results) > 0:
        result = detection_results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extract bounding box and confidence
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                symbol = {
                    'class': 'X' if cls == 0 else 'O',
                    'confidence': float(conf),
                    'bbox': bbox.tolist(),
                    'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                }
                symbols.append(symbol)
    
    return frame, symbols
```

## Coordinate Transformation

### Homography Matrix Computation

Maps from ideal 4×4 grid to detected camera coordinates:

```python
def compute_homography(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
    valid_points = keypoints[np.sum(np.abs(keypoints), axis=1) > 0]
    
    if len(valid_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        return None
    
    # Ideal normalized grid coordinates
    ideal_points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],  # Row 0
        [0, 1], [1, 1], [2, 1], [3, 1],  # Row 1  
        [0, 2], [1, 2], [2, 2], [3, 2],  # Row 2
        [0, 3], [1, 3], [2, 3], [3, 3]   # Row 3
    ], dtype=np.float32)
    
    try:
        H, _ = cv2.findHomography(
            ideal_points[:len(valid_points)],
            valid_points,
            cv2.RANSAC,
            RANSAC_REPROJ_THRESHOLD
        )
        return H
    except Exception as e:
        self.logger.error(f"Homography computation failed: {e}")
        return None
```

### Cell Center Calculation

```python
def get_cell_center_uv(self, row: int, col: int) -> Optional[Tuple[float, float]]:
    if not self.is_valid() or self._grid_points is None:
        return None
    
    # Calculate indices for cell corners in 4×4 grid
    top_left_idx = row * 4 + col
    top_right_idx = row * 4 + (col + 1)
    bottom_left_idx = (row + 1) * 4 + col
    bottom_right_idx = (row + 1) * 4 + (col + 1)
    
    # Verify indices are valid
    max_idx = max(top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx)
    if max_idx >= len(self._grid_points):
        return None
    
    # Get corner points
    corners = [
        self._grid_points[top_left_idx],
        self._grid_points[top_right_idx], 
        self._grid_points[bottom_left_idx],
        self._grid_points[bottom_right_idx]
    ]
    
    # Calculate center as average of corners
    center_u = sum(corner[0] for corner in corners) / 4
    center_v = sum(corner[1] for corner in corners) / 4
    
    return (center_u, center_v)
```

### Camera-to-Robot Coordinate Mapping

```python
def transform_uv_to_robot_coordinates(self, uv_point: Tuple[float, float], calibration_data: Dict) -> Tuple[float, float]:
    # Get transformation matrix from calibration
    xy_to_uv_matrix = np.array(calibration_data["perspective_transform_matrix_xy_to_uv"])
    uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)
    
    # Transform using homogeneous coordinates
    uv_homogeneous = np.array([uv_point[0], uv_point[1], 1.0])
    xy_transformed = np.dot(uv_to_xy_matrix, uv_homogeneous)
    
    # Convert back to 2D coordinates
    if xy_transformed[2] != 0:
        robot_x = xy_transformed[0] / xy_transformed[2]
        robot_y = xy_transformed[1] / xy_transformed[2]
        return (robot_x, robot_y)
    else:
        raise ValueError("Invalid homography transformation (division by zero)")
```

## Symbol-to-Cell Mapping

### Mapping Algorithm

```python
def map_symbols_to_cells(self, symbols: List[Dict], game_state: GameState) -> Dict[Tuple[int, int], str]:
    cell_assignments = {}
    
    for symbol in symbols:
        if symbol['confidence'] < self.symbol_confidence_threshold:
            continue
            
        symbol_center = symbol['center']
        best_cell = None
        min_distance = float('inf')
        
        # Find closest cell center
        for row in range(3):
            for col in range(3):
                cell_center = game_state.get_cell_center_uv(row, col)
                if cell_center is None:
                    continue
                
                distance = np.linalg.norm([
                    symbol_center[0] - cell_center[0],
                    symbol_center[1] - cell_center[1]
                ])
                
                if distance < min_distance and distance < self.max_symbol_distance:
                    min_distance = distance
                    best_cell = (row, col)
        
        if best_cell is not None:
            cell_assignments[best_cell] = symbol['class']
    
    return cell_assignments
```

### Confidence Filtering

```python
def filter_symbols_by_confidence(self, symbols: List[Dict]) -> List[Dict]:
    filtered = []
    
    for symbol in symbols:
        # Base confidence threshold
        if symbol['confidence'] < self.detection_conf_threshold:
            continue
            
        # Additional filtering based on bounding box quality
        bbox = symbol['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 0
        
        # Reject symbols with extreme aspect ratios
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
            
        # Reject very small detections (likely noise)
        if width < 20 or height < 20:
            continue
            
        filtered.append(symbol)
    
    return filtered
```

## Performance Optimization

### Model Optimization

```python
# GPU acceleration when available
if torch.cuda.is_available():
    device = 'cuda'
    self.pose_model.to(device)
    self.detect_model.to(device)
elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon
    self.pose_model.to(device)
    self.detect_model.to(device)
else:
    device = 'cpu'
```

### Frame Processing Optimization

```python
class FrameProcessor:
    def __init__(self):
        self.frame_skip_counter = 0
        self.skip_frames = 2  # Process every 3rd frame
        
    def should_process_frame(self) -> bool:
        self.frame_skip_counter += 1
        if self.frame_skip_counter >= self.skip_frames:
            self.frame_skip_counter = 0
            return True
        return False
```

### Memory Management

```python
def cleanup_resources(self):
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear frame buffers
    self.frame_buffer.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
```

## Calibration and Configuration

### Detection Thresholds

```python
# Grid detection confidence (pose estimation)
POSE_CONF_THRESHOLD = 0.45

# Symbol detection confidence (object detection)
BBOX_CONF_THRESHOLD = 0.45

# Keypoint visibility threshold
KEYPOINT_VISIBLE_THRESHOLD = 0.3

# Maximum distance for symbol-to-cell mapping (pixels)
MAX_SYMBOL_DISTANCE_PIXELS = 50
```

### Grid Validation Parameters

```python
# Minimum points required for homography
MIN_POINTS_FOR_HOMOGRAPHY = 4

# RANSAC reprojection threshold for homography
RANSAC_REPROJ_THRESHOLD = 5.0

# Maximum allowed standard deviation in grid point distances
GRID_DIST_STD_DEV_THRESHOLD = 0.3

# Grid angle tolerance (degrees)
GRID_ANGLE_TOLERANCE_DEG = 15.0
```

### Camera Settings

```python
# Recommended camera configuration
CAMERA_CONFIG = {
    'resolution': (1920, 1080),
    'fps': 30,
    'autofocus': False,  # Critical for consistent detection
    'exposure': 'auto',
    'white_balance': 'auto'
}
```

## Troubleshooting

### Common Detection Issues

#### Grid Not Detected
- **Cause**: Poor lighting, reflections, or occlusion
- **Solution**: Improve lighting, remove reflective surfaces, ensure clear view

#### Inconsistent Grid Detection
- **Cause**: Camera autofocus, movement, or variable lighting
- **Solution**: Disable autofocus, stabilize camera, use consistent lighting

#### Symbol Misclassification
- **Cause**: Low contrast, partial occlusion, or similar-looking marks
- **Solution**: Use high-contrast markers, clean game board, retrain model

#### Coordinate Transformation Errors
- **Cause**: Inaccurate grid detection or calibration drift
- **Solution**: Re-run calibration, verify grid point accuracy

### Debug Visualization

```python
def draw_detection_debug(self, frame: np.ndarray, grid_points: np.ndarray, symbols: List[Dict]) -> np.ndarray:
    debug_frame = frame.copy()
    
    # Draw grid points
    for i, point in enumerate(grid_points):
        if point[0] > 0 and point[1] > 0:  # Valid point
            cv2.circle(debug_frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(debug_frame, str(i), tuple(point.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw symbol detections
    for symbol in symbols:
        bbox = symbol['bbox']
        cv2.rectangle(debug_frame, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (255, 0, 0), 2)
        cv2.putText(debug_frame, 
                   f"{symbol['class']} {symbol['confidence']:.2f}",
                   (int(bbox[0]), int(bbox[1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return debug_frame
```

---

This computer vision pipeline provides robust, real-time game detection suitable for interactive robotic gameplay applications.