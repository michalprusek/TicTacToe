#!/usr/bin/env python3
"""
Debug resolution mismatch between calibration and runtime
"""
import sys
import os
import json
import cv2
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def analyze_resolution_mismatch():
    """Analyze resolution mismatch between calibration and runtime."""
    print("🔍 RESOLUTION MISMATCH ANALYSIS")
    print("=" * 60)
    
    # 1. Analyze calibration resolution
    print("📊 CALIBRATION RESOLUTION ANALYSIS:")
    print("-" * 60)
    
    with open('app/calibration/hand_eye_calibration.json', 'r') as f:
        cal_data = json.load(f)
    
    # Find max UV coordinates in calibration
    max_u = 0
    max_v = 0
    min_u = float('inf')
    min_v = float('inf')
    
    for point in cal_data["calibration_points_raw"]:
        u, v = point["target_uv"]
        max_u = max(max_u, u)
        max_v = max(max_v, v)
        min_u = min(min_u, u)
        min_v = min(min_v, v)
    
    print(f"Calibration UV coordinates range:")
    print(f"  U: {min_u} → {max_u} (range: {max_u - min_u})")
    print(f"  V: {min_v} → {max_v} (range: {max_v - min_v})")
    print(f"  Estimated calibration resolution: ~{max_u + 100}x{max_v + 100}")
    
    # 2. Check current camera resolution
    print(f"\n📷 CURRENT CAMERA RESOLUTION:")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        # Get current resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Current camera resolution: {width}x{height}")
        
        # Capture a frame to see actual size
        ret, frame = cap.read()
        if ret:
            actual_height, actual_width = frame.shape[:2]
            print(f"Actual frame size: {actual_width}x{actual_height}")
        
        cap.release()
    else:
        print("❌ Cannot open camera")
    
    # 3. Check YOLO model input size
    print(f"\n🤖 YOLO MODEL INPUT SIZE:")
    print("-" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Load detection model
        detect_model = YOLO("weights/best_detection.pt")
        pose_model = YOLO("weights/best_pose.pt")
        
        # Check model input size
        print(f"Detection model input size: {detect_model.model.args.get('imgsz', 'Unknown')}")
        print(f"Pose model input size: {pose_model.model.args.get('imgsz', 'Unknown')}")
        
        # Test with a sample frame
        if 'frame' in locals():
            print(f"\nTesting detection with frame {frame.shape}:")
            
            # Run detection
            detect_results = detect_model.predict(frame, verbose=False)
            pose_results = pose_model.predict(frame, verbose=False)
            
            if detect_results and len(detect_results) > 0:
                result = detect_results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    print(f"  Detection boxes coordinates range:")
                    if len(boxes) > 0:
                        print(f"    X: {boxes[:, 0].min():.1f} → {boxes[:, 2].max():.1f}")
                        print(f"    Y: {boxes[:, 1].min():.1f} → {boxes[:, 3].max():.1f}")
            
            if pose_results and len(pose_results) > 0:
                result = pose_results[0]
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints.xy.cpu().numpy()
                    if len(kpts) > 0 and len(kpts[0]) > 0:
                        points = kpts[0]
                        print(f"  Pose keypoints coordinates range:")
                        print(f"    X: {points[:, 0].min():.1f} → {points[:, 0].max():.1f}")
                        print(f"    Y: {points[:, 1].min():.1f} → {points[:, 1].max():.1f}")
                        print(f"    Example points: {points[:3].tolist()}")
        
    except Exception as e:
        print(f"❌ Error loading YOLO models: {e}")
    
    # 4. Calculate scaling factors
    print(f"\n📐 SCALING FACTOR CALCULATION:")
    print("-" * 60)
    
    # Assume calibration was done at 1920x1080 (based on max coordinates ~1400x1000)
    calibration_width = 1920
    calibration_height = 1080
    
    # Current runtime resolution (from camera or YOLO input)
    if 'actual_width' in locals() and 'actual_height' in locals():
        runtime_width = actual_width
        runtime_height = actual_height
        
        scale_x = runtime_width / calibration_width
        scale_y = runtime_height / calibration_height
        
        print(f"Estimated calibration resolution: {calibration_width}x{calibration_height}")
        print(f"Current runtime resolution: {runtime_width}x{runtime_height}")
        print(f"Scaling factors:")
        print(f"  X scale: {scale_x:.4f}")
        print(f"  Y scale: {scale_y:.4f}")
        
        # Test transformation with scaling
        print(f"\n🧪 TRANSFORMATION TEST WITH SCALING:")
        print("-" * 60)
        
        # Load transformation matrix
        xy_to_uv_matrix = np.array(cal_data["perspective_transform_matrix_xy_to_uv"], dtype=np.float32)
        uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)
        
        # Test with a sample UV coordinate from current runtime
        if 'points' in locals() and len(points) > 0:
            test_uv_runtime = points[0]  # First detected point
            
            # Scale to calibration resolution
            test_uv_calibration = [
                test_uv_runtime[0] / scale_x,
                test_uv_runtime[1] / scale_y
            ]
            
            print(f"Runtime UV: ({test_uv_runtime[0]:.1f}, {test_uv_runtime[1]:.1f})")
            print(f"Scaled to calibration: ({test_uv_calibration[0]:.1f}, {test_uv_calibration[1]:.1f})")
            
            # Apply transformation
            uv_homogeneous = np.array([test_uv_calibration[0], test_uv_calibration[1], 1.0], dtype=np.float32).reshape(3, 1)
            xy_transformed_homogeneous = np.dot(uv_to_xy_matrix, uv_homogeneous)
            
            if xy_transformed_homogeneous[2, 0] != 0:
                arm_x = xy_transformed_homogeneous[0, 0] / xy_transformed_homogeneous[2, 0]
                arm_y = xy_transformed_homogeneous[1, 0] / xy_transformed_homogeneous[2, 0]
                print(f"Transformed XY: ({arm_x:.1f}, {arm_y:.1f})")
    
    # 5. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    print("1. Implement coordinate scaling in arm_movement_controller.py")
    print("2. Scale UV coordinates from runtime resolution to calibration resolution")
    print("3. Apply transformation matrix using scaled coordinates")
    print("4. Alternative: Recalibrate using current runtime resolution")
    
    print(f"\n🔧 IMPLEMENTATION STEPS:")
    print("-" * 60)
    print("1. Detect current camera/YOLO resolution")
    print("2. Calculate scaling factors (runtime/calibration)")
    print("3. In _get_cell_coordinates_from_yolo():")
    print("   - Scale UV coordinates: uv_scaled = uv_runtime / scale_factor")
    print("   - Apply transformation matrix to scaled coordinates")
    print("4. Test with known calibration points")

if __name__ == "__main__":
    analyze_resolution_mismatch()
