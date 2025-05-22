#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test skript pro ověření správného kreslení na buňku (2, 1) v rámci hry.
Tento skript simuluje tah AI na pozici (2, 1) a ověří, že robotická ruka
kreslí symbol na správné pozici.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Optional, Tuple

# Přidání cesty k projektu do PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main.arm_controller import ArmController
from app.main import game_logic
from app.core.arm_thread import ArmThread

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
logger = logging.getLogger("test_game_cell")


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


class MockGameState:
    """Simulace herního stavu pro testování."""
    
    def __init__(self, calibration_data):
        """Inicializace herního stavu."""
        self.board = game_logic.create_board()
        self.calibration_data = calibration_data
        
    def get_cell_center_uv(self, row, col):
        """Simulace získání středu buňky v pixelech."""
        # Pro účely testu vrátíme fiktivní hodnoty
        return (800 + col * 200, 500 + row * 200)


def test_game_cell_drawing(row: int, col: int, symbol: str = 'O') -> bool:
    """
    Otestuje kreslení symbolu na dané buňce v rámci hry.
    
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
    
    expected_x = calibration_data["grid_positions"][cell_key]["x"]
    expected_y = calibration_data["grid_positions"][cell_key]["y"]
    
    # Získání Z souřadnic pro kreslení
    safe_z = calibration_data.get("safe_z", 15.0)
    draw_z = calibration_data.get("touch_z", 5.0)
    
    logger.info(f"Testování kreslení na buňku ({row}, {col})")
    logger.info(f"Očekávané souřadnice: ({expected_x}, {expected_y})")
    
    # Inicializace robotické ruky
    arm_thread = ArmThread()
    arm_thread.start()
    
    try:
        # Připojení k robotické ruce
        if not arm_thread.connect():
            logger.error("Nepodařilo se připojit k robotické ruce.")
            return False
        
        # Vytvoření simulovaného herního stavu
        game_state = MockGameState(calibration_data)
        
        # Přesun do bezpečné pozice
        logger.info("Přesun do bezpečné pozice...")
        arm_thread.move(
            x=calibration_data.get("neutral_position", {}).get("x", 200),
            y=calibration_data.get("neutral_position", {}).get("y", 0),
            z=safe_z
        )
        time.sleep(2)
        
        # Získání souřadnic buňky z kalibračních dat
        target_x = calibration_data["grid_positions"][cell_key]["x"]
        target_y = calibration_data["grid_positions"][cell_key]["y"]
        
        # Kreslení symbolu
        logger.info(f"Kreslení symbolu {symbol} na buňku ({row}, {col})...")
        success = False
        if symbol.upper() == 'X':
            success = arm_thread.draw_x(
                target_x, target_y, DEFAULT_SYMBOL_SIZE_MM, speed=DRAWING_SPEED
            )
        else:  # 'O'
            success = arm_thread.draw_o(
                target_x, target_y, DEFAULT_SYMBOL_SIZE_MM / 2, speed=DRAWING_SPEED
            )
        
        if success:
            logger.info(f"Symbol {symbol} úspěšně nakreslen na souřadnicích ({target_x}, {target_y}).")
            
            # Ověření, že souřadnice odpovídají očekávaným hodnotám
            if target_x == expected_x and target_y == expected_y:
                logger.info("Souřadnice odpovídají očekávaným hodnotám.")
            else:
                logger.error(f"Souřadnice neodpovídají očekávaným hodnotám! "
                            f"Očekáváno: ({expected_x}, {expected_y}), "
                            f"Získáno: ({target_x}, {target_y})")
                success = False
        else:
            logger.error(f"Nepodařilo se nakreslit symbol {symbol}.")
        
        # Návrat do bezpečné pozice
        logger.info("Návrat do bezpečné pozice...")
        arm_thread.move(
            x=calibration_data.get("neutral_position", {}).get("x", 200),
            y=calibration_data.get("neutral_position", {}).get("y", 0),
            z=safe_z
        )
        time.sleep(2)
        
        return success
    
    finally:
        # Odpojení a ukončení vlákna robotické ruky
        arm_thread.disconnect()
        arm_thread.stop()
        arm_thread.join(timeout=5)


def main():
    """Hlavní funkce skriptu."""
    # Test kreslení na buňku (2, 1)
    success = test_game_cell_drawing(2, 1, 'O')
    
    if success:
        logger.info("Test kreslení na buňku (2, 1) proběhl úspěšně.")
        return 0
    else:
        logger.error("Test kreslení na buňku (2, 1) selhal.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
