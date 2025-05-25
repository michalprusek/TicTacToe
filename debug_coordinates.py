#!/usr/bin/env python3
"""
Debug script pro analýzu transformace souřadnic.
Analyzuje celý pipeline od detekce grid points až po world coordinates.
"""

import sys
import os
import json
import numpy as np

# Přidání path pro import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.main.path_utils import setup_project_path
setup_project_path()

def load_calibration_data():
    """Načte kalibrační data."""
    cal_file = "app/calibration/hand_eye_calibration.json"
    try:
        with open(cal_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Chyba při načítání kalibrace: {e}")
        return None

def analyze_calibration():
    """Analyzuje kalibrační data."""
    print("=== ANALÝZA KALIBRAČNÍCH DAT ===")
    
    cal_data = load_calibration_data()
    if not cal_data:
        return
        
    # Matice transformace
    matrix = cal_data["perspective_transform_matrix_xy_to_uv"]
    print(f"Transformační matice XY→UV:")
    for row in matrix:
        print(f"  {row}")
    
    # Analýza kalibračních bodů
    points = cal_data["calibration_points_raw"]
    print(f"\nKalibrační body ({len(points)} bodů):")
    
    uv_coords = []
    xy_coords = []
    
    for i, point in enumerate(points):
        uv = point["target_uv"]
        xy = point["robot_xyz"][:2]  # jen X,Y
        uv_coords.append(uv)
        xy_coords.append(xy)
        print(f"  {i+1:2d}: UV=({uv[0]:4d}, {uv[1]:4d}) → XY=({xy[0]:6.1f}, {xy[1]:6.1f})")
    
    # Rozsahy
    uv_coords = np.array(uv_coords)
    xy_coords = np.array(xy_coords)
    
    print(f"\nRozsahy:")
    print(f"  UV: X={uv_coords[:,0].min():4.0f}..{uv_coords[:,0].max():4.0f}, Y={uv_coords[:,1].min():4.0f}..{uv_coords[:,1].max():4.0f}")
    print(f"  XY: X={xy_coords[:,0].min():6.1f}..{xy_coords[:,0].max():6.1f}, Y={xy_coords[:,1].min():6.1f}..{xy_coords[:,1].max():6.1f}")

def test_transformation(test_uv_points):
    """Testuje transformaci konkrétních UV bodů."""
    print("\n=== TEST TRANSFORMACE ===")
    
    cal_data = load_calibration_data()
    if not cal_data:
        return
        
    # Matice transformace
    xy_to_uv_matrix = np.array(cal_data["perspective_transform_matrix_xy_to_uv"], dtype=np.float32)
    uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)
    
    print(f"Inverzní matice UV→XY:")
    for row in uv_to_xy_matrix:
        print(f"  {row}")
    
    print(f"\nTest transformace bodů:")
    for i, (u, v) in enumerate(test_uv_points):
        # Homogenní souřadnice
        uv_point = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
        xy_result = np.dot(uv_to_xy_matrix, uv_point)
        
        if xy_result[2, 0] != 0:
            x = xy_result[0, 0] / xy_result[2, 0]
            y = xy_result[1, 0] / xy_result[2, 0]
            print(f"  {i+1}: UV=({u:4.0f}, {v:4.0f}) → XY=({x:6.1f}, {y:6.1f})")
        else:
            print(f"  {i+1}: UV=({u:4.0f}, {v:4.0f}) → CHYBA (dělení nulou)")

def simulate_grid_cells():
    """Simuluje typické pozice center buněk."""
    print("\n=== SIMULACE BUNĚK ===")
    
    # Typické pozice center buněk na 1920x1080 rozlišení
    # Pokud je grid někde uprostřed obrazu
    center_x, center_y = 960, 540
    cell_size = 100  # přibližná velikost buňky v pixelech
    
    test_points = []
    print("Simulované pozice center buněk:")
    for row in range(3):
        for col in range(3):
            # Střed gridu + offset pro každou buňku
            u = center_x + (col - 1) * cell_size
            v = center_y + (row - 1) * cell_size
            test_points.append((u, v))
            print(f"  Buňka ({row},{col}): UV=({u:4.0f}, {v:4.0f})")
    
    return test_points

if __name__ == "__main__":
    print("🔍 DEBUG ANALÝZA TRANSFORMACE SOUŘADNIC")
    print("=" * 50)
    
    # Analýza kalibračních dat
    analyze_calibration()
    
    # Simulace a test buněk
    test_points = simulate_grid_cells()
    test_transformation(test_points)
    
    print("\n✅ Analýza dokončena")
    print("\nPro další debugging spusť aplikaci a sleduj logy z _get_cell_coordinates_from_yolo()")