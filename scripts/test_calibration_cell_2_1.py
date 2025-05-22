#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test skript pro ověření kalibrace buňky (2, 1) na herní desce.
Tento skript načte kalibrační data a otestuje kreslení symbolu na pozici (2, 1).
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Optional, Tuple

# Přidání cesty k projektu do PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main.arm_controller import ArmController
from app.main import game_logic

# Konstanty
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "app", "calibration", "hand_eye_calibration.json")
DEFAULT_SYMBOL_SIZE_MM = 40.0
DRAWING_SPEED = 6000  # Rychlost kreslení

# Nastavení loggeru
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_calibration")


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


def test_cell_calibration(row: int, col: int, symbol: str = 'O') -> bool:
    """
    Otestuje kalibraci buňky kreslením symbolu na dané pozici.
    
    Args:
        row: Řádek buňky (0-2)
        col: Sloupec buňky (0-2)
        symbol: Symbol k nakreslení ('X' nebo 'O')
        
    Returns:
        bool: True pokud test proběhl úspěšně, jinak False
    """
    # Načtení kalibračních dat
    calibration_data = load_calibration(CALIBRATION_FILE)
    if not calibration_data:
        return False
    
    # Kontrola, zda máme mapování souřadnic pro buňky
    if "grid_positions" not in calibration_data:
        logger.error("Chybí mapování souřadnic pro buňky v kalibračních datech.")
        return False
    
    # Získání souřadnic pro danou buňku
    cell_key = f"{row}_{col}"
    if cell_key not in calibration_data["grid_positions"]:
        logger.error(f"Buňka {cell_key} není v kalibračních datech.")
        return False
    
    target_x = calibration_data["grid_positions"][cell_key]["x"]
    target_y = calibration_data["grid_positions"][cell_key]["y"]
    
    # Získání Z souřadnic pro kreslení
    safe_z = calibration_data.get("safe_z", 15.0)
    draw_z = calibration_data.get("touch_z", 5.0)
    
    logger.info(f"Testování kalibrace buňky ({row}, {col}) na souřadnicích ({target_x}, {target_y})")
    
    # Inicializace robotické ruky
    controller = ArmController()
    controller.safe_z = safe_z
    controller.draw_z = draw_z
    
    if not controller.connect():
        logger.error("Nepodařilo se připojit k robotické ruce.")
        return False
    
    try:
        # Přesun do bezpečné pozice
        logger.info("Přesun do bezpečné pozice...")
        controller.go_to_position(
            x=calibration_data.get("neutral_position", {}).get("x", 200),
            y=calibration_data.get("neutral_position", {}).get("y", 0),
            z=safe_z,
            wait=True
        )
        time.sleep(1)
        
        # Přesun nad cílovou buňku
        logger.info(f"Přesun nad buňku ({row}, {col})...")
        controller.go_to_position(
            x=target_x,
            y=target_y,
            z=safe_z,
            wait=True
        )
        time.sleep(1)
        
        # Kreslení symbolu
        success = False
        if symbol.upper() == 'X':
            logger.info(f"Kreslení symbolu X na souřadnicích ({target_x}, {target_y})...")
            success = controller.draw_x(
                target_x, target_y, DEFAULT_SYMBOL_SIZE_MM, speed=DRAWING_SPEED
            )
        else:  # 'O'
            logger.info(f"Kreslení symbolu O na souřadnicích ({target_x}, {target_y})...")
            success = controller.draw_o(
                target_x, target_y, DEFAULT_SYMBOL_SIZE_MM / 2, speed=DRAWING_SPEED
            )
        
        if success:
            logger.info(f"Symbol {symbol} úspěšně nakreslen na souřadnicích ({target_x}, {target_y}).")
        else:
            logger.error(f"Nepodařilo se nakreslit symbol {symbol}.")
        
        # Návrat do bezpečné pozice
        logger.info("Návrat do bezpečné pozice...")
        controller.go_to_position(
            x=calibration_data.get("neutral_position", {}).get("x", 200),
            y=calibration_data.get("neutral_position", {}).get("y", 0),
            z=safe_z,
            wait=True
        )
        
        return success
    
    finally:
        # Odpojení robotické ruky
        controller.disconnect()


def main():
    """Hlavní funkce skriptu."""
    parser = argparse.ArgumentParser(description='Test kalibrace buňky (2, 1) na herní desce.')
    parser.add_argument('--symbol', type=str, choices=['X', 'O'], default='O',
                        help='Symbol k nakreslení (X nebo O)')
    args = parser.parse_args()
    
    # Test kalibrace buňky (2, 1)
    success = test_cell_calibration(2, 1, args.symbol)
    
    if success:
        logger.info("Test kalibrace buňky (2, 1) proběhl úspěšně.")
        return 0
    else:
        logger.error("Test kalibrace buňky (2, 1) selhal.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
