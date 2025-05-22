#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skript pro analýzu detekovaných bodů mřížky a určení správných souřadnic pro buňku (2, 1).
Tento skript zachytí snímek z kamery, detekuje mřížku pomocí YOLO modelu,
a vypočítá správné souřadnice pro buňku (2, 1) na základě detekovaných bodů.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO

# Přidání cesty k projektu do PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main.arm_controller import ArmController
from app.main import game_logic
from app.core.game_state import GameState, GRID_POINTS_COUNT

# Konstanty
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "app", "calibration", "hand_eye_calibration.json")
POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "weights", "best_pose.pt")
CAM_INDEX = 0
POSE_CONF_THRESHOLD = 0.45
WINDOW_NAME = "Grid Analysis"

# Nastavení loggeru
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("grid_analysis")


def load_calibration(filename: str) -> Optional[Dict]:
    """Načte kalibrační data ze souboru JSON."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        logger.info(f"Kalibrace úspěšně načtena z {filename}")
        return data
    except FileNotFoundError:
        logger.error(f"CHYBA: Kalibrační soubor '{filename}' nenalezen!")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"CHYBA: Neplatný formát JSON v souboru '{filename}': {e}")
        return None
    except Exception as e:
        logger.error(f"CHYBA: Nepodařilo se načíst kalibrační soubor: {e}")
        return None


def save_calibration(filename: str, data: Dict) -> bool:
    """Uloží kalibrační data do souboru JSON."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Kalibrace úspěšně uložena do {filename}")
        return True
    except Exception as e:
        logger.error(f"CHYBA: Nepodařilo se uložit kalibrační soubor: {e}")
        return False


def calculate_uv_to_xy_transform(calibration_data: Dict) -> Optional[np.ndarray]:
    """Vypočítá transformační matici UV -> XY z kalibračních bodů."""
    if not calibration_data or "calibration_points_raw" not in calibration_data:
        logger.error("Chybí data pro výpočet UV->XY transformace.")
        return None

    raw_points = calibration_data["calibration_points_raw"]
    min_points = 4
    if len(raw_points) < min_points:
        logger.error(f"Nedostatek bodů ({len(raw_points)} < {min_points}) v kalibraci pro výpočet UV->XY.")
        return None

    # Extrahujeme UV a XY souřadnice z kalibračních bodů
    points_uv = []
    points_xy = []
    valid_points_count = 0

    for p in raw_points:
        if ('target_uv' in p and len(p['target_uv']) == 2 and
                'robot_xyz' in p and len(p['robot_xyz']) >= 2):
            points_uv.append(p['target_uv'])
            points_xy.append(p['robot_xyz'][:2])  # Potřebujeme jen XY
            valid_points_count += 1

    if valid_points_count < min_points:
        logger.error(f"Nedostatek platných bodů ({valid_points_count} < {min_points}) pro výpočet transformace UV->XY.")
        return None

    # Převedeme na numpy pole
    np_points_uv = np.array(points_uv, dtype=np.float32)
    np_points_xy = np.array(points_xy, dtype=np.float32)

    # Vypočítáme homografii (perspektivní transformaci)
    try:
        transform_matrix, _ = cv2.findHomography(
            np_points_uv, np_points_xy, method=cv2.RANSAC, ransacReprojThreshold=10.0
        )
        logger.info("Transformační matice UV->XY úspěšně vypočtena.")
        return transform_matrix
    except Exception as e:
        logger.error(f"CHYBA při výpočtu transformace UV->XY: {e}")
        return None


def detect_grid_and_analyze_cell(row: int, col: int) -> Optional[Tuple[float, float]]:
    """
    Detekuje mřížku a analyzuje souřadnice buňky.
    
    Args:
        row: Řádek buňky (0-2)
        col: Sloupec buňky (0-2)
        
    Returns:
        Tuple souřadnic (x, y) v milimetrech nebo None při selhání
    """
    # Načtení kalibračních dat
    calibration_data = load_calibration(CALIBRATION_FILE)
    if not calibration_data:
        return None
    
    # Načtení YOLO modelu pro detekci mřížky
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
        logger.info(f"YOLO model úspěšně načten z {POSE_MODEL_PATH}")
    except Exception as e:
        logger.error(f"CHYBA při načítání YOLO modelu: {e}")
        return None
    
    # Inicializace kamery
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        logger.error(f"Nepodařilo se otevřít kameru s indexem {CAM_INDEX}")
        return None
    
    # Nastavení rozlišení kamery
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Vytvoření okna pro zobrazení
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Inicializace GameState pro zpracování mřížky
    game_state = GameState()
    
    # Výpočet transformační matice UV->XY
    transform_uv_to_xy = calculate_uv_to_xy_transform(calibration_data)
    if transform_uv_to_xy is None:
        logger.error("Nepodařilo se vypočítat transformační matici UV->XY")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Zachycení několika snímků pro stabilizaci kamery
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            logger.error("Nepodařilo se zachytit snímek z kamery")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    # Zachycení a analýza snímku
    ret, frame = cap.read()
    if not ret:
        logger.error("Nepodařilo se zachytit snímek z kamery")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Detekce mřížky pomocí YOLO modelu
    pose_results = pose_model(frame, verbose=False, conf=POSE_CONF_THRESHOLD)
    
    # Zpracování výsledků detekce
    if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
        # Extrakce keypoints
        kpts_data = pose_results[0].keypoints.data[0].cpu().numpy()
        
        # Kontrola, zda máme dostatek bodů
        if kpts_data.shape[0] >= GRID_POINTS_COUNT:
            # Extrakce souřadnic bodů
            grid_points = kpts_data[:GRID_POINTS_COUNT, :2]
            
            # Aktualizace GameState s detekovanými body
            game_state._grid_points = grid_points
            game_state._is_valid_grid = True
            
            # Výpočet transformace mřížky
            game_state._compute_grid_transformation()
            
            # Získání středu buňky v UV souřadnicích
            cell_center_uv = game_state.get_cell_center_uv(row, col)
            
            if cell_center_uv:
                logger.info(f"Střed buňky ({row}, {col}) v UV souřadnicích: {cell_center_uv}")
                
                # Transformace UV souřadnic na XY souřadnice
                center_uv_np = np.array([[cell_center_uv]], dtype=np.float32)
                transformed_xy = cv2.perspectiveTransform(center_uv_np, transform_uv_to_xy)
                
                if transformed_xy is not None and transformed_xy.shape == (1, 1, 2):
                    center_xy = tuple(transformed_xy[0, 0])
                    logger.info(f"Střed buňky ({row}, {col}) v XY souřadnicích: {center_xy}")
                    
                    # Vykreslení detekované mřížky a středu buňky
                    display_frame = frame.copy()
                    
                    # Vykreslení mřížky
                    game_state.draw_grid(display_frame)
                    
                    # Vykreslení středu buňky
                    cv2.circle(display_frame, 
                              (int(cell_center_uv[0]), int(cell_center_uv[1])), 
                              10, (0, 0, 255), -1)
                    
                    # Zobrazení souřadnic
                    text = f"Cell ({row}, {col}): UV={cell_center_uv}, XY={center_xy}"
                    cv2.putText(display_frame, text, (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Zobrazení snímku
                    cv2.imshow(WINDOW_NAME, display_frame)
                    cv2.waitKey(0)
                    
                    # Uvolnění zdrojů
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    return center_xy
                else:
                    logger.error("Nepodařilo se transformovat UV souřadnice na XY souřadnice")
            else:
                logger.error(f"Nepodařilo se získat střed buňky ({row}, {col})")
        else:
            logger.error(f"Nedostatek bodů mřížky: {kpts_data.shape[0]} < {GRID_POINTS_COUNT}")
    else:
        logger.error("Nepodařilo se detekovat mřížku")
    
    # Uvolnění zdrojů
    cap.release()
    cv2.destroyAllWindows()
    return None


def update_calibration_for_cell(row: int, col: int, new_xy: Tuple[float, float]) -> bool:
    """
    Aktualizuje kalibrační data pro danou buňku.
    
    Args:
        row: Řádek buňky (0-2)
        col: Sloupec buňky (0-2)
        new_xy: Nové souřadnice (x, y) v milimetrech
        
    Returns:
        bool: True pokud aktualizace proběhla úspěšně, jinak False
    """
    # Načtení kalibračních dat
    calibration_data = load_calibration(CALIBRATION_FILE)
    if not calibration_data:
        return False
    
    # Kontrola, zda máme mapování souřadnic pro buňky
    if "grid_positions" not in calibration_data:
        logger.error("Chybí mapování souřadnic pro buňky v kalibračních datech.")
        return False
    
    # Aktualizace souřadnic pro danou buňku
    cell_key = f"{row}_{col}"
    if cell_key not in calibration_data["grid_positions"]:
        logger.error(f"Buňka {cell_key} není v kalibračních datech.")
        return False
    
    # Získání aktuálních souřadnic
    current_x = calibration_data["grid_positions"][cell_key]["x"]
    current_y = calibration_data["grid_positions"][cell_key]["y"]
    
    # Aktualizace souřadnic
    calibration_data["grid_positions"][cell_key]["x"] = new_xy[0]
    calibration_data["grid_positions"][cell_key]["y"] = new_xy[1]
    
    logger.info(f"Aktualizace souřadnic buňky ({row}, {col}): "
               f"({current_x}, {current_y}) -> ({new_xy[0]}, {new_xy[1]})")
    
    # Uložení aktualizovaných kalibračních dat
    return save_calibration(CALIBRATION_FILE, calibration_data)


def main():
    """Hlavní funkce skriptu."""
    logger.info("Analýza souřadnic buňky (2, 1) na základě detekovaných bodů mřížky")
    
    # Detekce mřížky a analýza buňky (2, 1)
    cell_xy = detect_grid_and_analyze_cell(2, 1)
    
    if cell_xy:
        # Zaokrouhlení souřadnic na celá čísla
        rounded_xy = (round(cell_xy[0]), round(cell_xy[1]))
        logger.info(f"Detekované souřadnice buňky (2, 1): {cell_xy}")
        logger.info(f"Zaokrouhlené souřadnice: {rounded_xy}")
        
        # Aktualizace kalibračních dat
        if update_calibration_for_cell(2, 1, rounded_xy):
            logger.info("Kalibrace úspěšně aktualizována.")
            return 0
        else:
            logger.error("Nepodařilo se aktualizovat kalibraci.")
            return 1
    else:
        logger.error("Nepodařilo se detekovat souřadnice buňky (2, 1).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
