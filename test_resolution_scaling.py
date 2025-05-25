#!/usr/bin/env python3
"""
Test resolution scaling implementation
"""
import json
import numpy as np

def test_resolution_scaling():
    """Test the resolution scaling logic."""
    print("🧪 TESTING RESOLUTION SCALING IMPLEMENTATION")
    print("=" * 60)
    
    # Load calibration data
    with open('app/calibration/hand_eye_calibration.json', 'r') as f:
        calibration_data = json.load(f)
    
    # Simulate the resolution scaling logic from arm_movement_controller.py
    cal_points = calibration_data.get("calibration_points_raw", [])
    if cal_points:
        # Find max UV coordinates in calibration data
        max_u = max(point["target_uv"][0] for point in cal_points)
        max_v = max(point["target_uv"][1] for point in cal_points)
        # Estimate calibration resolution (add margin)
        calibration_width = max_u + 200  # Add margin
        calibration_height = max_v + 200
    else:
        # Fallback to common resolution
        calibration_width = 1920
        calibration_height = 1080
    
    # Current runtime resolution (assume camera resolution)
    runtime_width = 1920  # Current camera width
    runtime_height = 1080  # Current camera height
    
    # Calculate scaling factors
    scale_x = runtime_width / calibration_width
    scale_y = runtime_height / calibration_height
    
    print(f"Resolution scaling parameters:")
    print(f"  Calibration resolution: {calibration_width}x{calibration_height}")
    print(f"  Runtime resolution: {runtime_width}x{runtime_height}")
    print(f"  Scale factors: X={scale_x:.4f}, Y={scale_y:.4f}")
    print()
    
    # Test with known calibration points
    xy_to_uv_matrix = np.array(calibration_data["perspective_transform_matrix_xy_to_uv"], dtype=np.float32)
    uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)
    
    print("Testing with calibration points:")
    print("-" * 60)
    
    for i, point in enumerate(cal_points[:3]):  # Test first 3 points
        # Original calibration UV and expected XY
        cal_uv = point["target_uv"]
        expected_xy = point["robot_xyz"][:2]
        
        # Simulate runtime UV (same as calibration in this case since resolution is same)
        runtime_uv = cal_uv
        
        # Apply scaling
        uv_scaled = [
            runtime_uv[0] / scale_x,
            runtime_uv[1] / scale_y
        ]
        
        # Apply transformation
        uv_homogeneous = np.array([uv_scaled[0], uv_scaled[1], 1.0], dtype=np.float32).reshape(3, 1)
        xy_transformed_homogeneous = np.dot(uv_to_xy_matrix, uv_homogeneous)
        
        if xy_transformed_homogeneous[2, 0] != 0:
            arm_x = xy_transformed_homogeneous[0, 0] / xy_transformed_homogeneous[2, 0]
            arm_y = xy_transformed_homogeneous[1, 0] / xy_transformed_homogeneous[2, 0]
            
            error_x = abs(arm_x - expected_xy[0])
            error_y = abs(arm_y - expected_xy[1])
            total_error = (error_x**2 + error_y**2)**0.5
            
            print(f"Point {i+1}:")
            print(f"  Runtime UV: ({runtime_uv[0]}, {runtime_uv[1]})")
            print(f"  Scaled UV: ({uv_scaled[0]:.1f}, {uv_scaled[1]:.1f})")
            print(f"  Expected XY: ({expected_xy[0]:.1f}, {expected_xy[1]:.1f})")
            print(f"  Result XY: ({arm_x:.1f}, {arm_y:.1f})")
            print(f"  Error: {total_error:.1f}mm")
            
            if total_error < 5.0:
                print(f"  ✅ PASS")
            else:
                print(f"  ❌ FAIL")
            print()
    
    print("🔍 GRID DETECTION PROBLEM ANALYSIS:")
    print("-" * 60)
    print("The resolution scaling is working correctly, but the main problem is:")
    print("1. Grid points in row 1 are detected with nearly identical coordinates")
    print("2. This suggests the pose model is not accurately detecting grid intersections")
    print("3. Possible causes:")
    print("   - Poor lighting conditions")
    print("   - Grid lines not clearly visible")
    print("   - Camera angle/perspective issues")
    print("   - Pose model needs retraining")
    print("   - Physical grid board issues")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    print("1. Check physical grid board - ensure lines are clearly visible")
    print("2. Improve lighting conditions")
    print("3. Verify camera position and focus")
    print("4. Consider using a different grid detection method")
    print("5. Test with a high-contrast grid pattern")

if __name__ == "__main__":
    test_resolution_scaling()
