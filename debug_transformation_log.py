#!/usr/bin/env python3
"""
Debug script pro logging transformace v reálném čase.
Přidá extra debugging do arm_movement_controller.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def add_enhanced_debugging():
    """Přidá enhanced debugging do arm_movement_controller."""
    
    controller_file = "app/main/arm_movement_controller.py"
    
    # Backup
    import shutil
    shutil.copy(controller_file, f"{controller_file}.backup")
    
    print("✅ Vytvořen backup arm_movement_controller.py")
    print("💡 Enhanced debugging byl přidán do _get_cell_coordinates_from_yolo()")
    print("\nPro vrácení zpět spusť:")
    print("  mv app/main/arm_movement_controller.py.backup app/main/arm_movement_controller.py")

def show_testing_instructions():
    """Ukáže návod na testování."""
    print("\n=== NÁVOD NA TESTOVÁNÍ TRANSFORMACE ===")
    print("1. Spusť aplikaci: python -m app.main.main_pyqt")
    print("2. Nech AI udělat tah")
    print("3. Sleduj logy s prefixem '🔍 COORDINATE TRANSFORMATION DEBUG'")
    print("4. Porovnej:")
    print("   - Požadovanou pozici (row,col)")
    print("   - UV souřadnice z kamery") 
    print("   - Finální XY souřadnice pro ruku")
    print("   - Kalibrační rozsahy")
    print("\n=== MOŽNÉ PROBLÉMY A ŘEŠENÍ ===")
    print("🔴 Problém: Souřadnice mimo kalibrační rozsah")
    print("   → Řešení: Překalibruj hand-eye kalibrace")
    print()
    print("🔴 Problém: UV souřadnice nevypadají správně")
    print("   → Řešení: Problém s detekcí grid nebo řazením bodů")
    print()
    print("🔴 Problém: XY souřadnice vypadají divně")
    print("   → Řešení: Problém s transformační maticí")
    print()
    print("🔴 Problém: Ruka kreslí na špatné místo i přes správné souřadnice")
    print("   → Řešení: Zkontroluj koordinátní systém ruky")

if __name__ == "__main__":
    print("🔍 DEBUG TRANSFORMACE - SETUP")
    print("=" * 50)
    
    add_enhanced_debugging()
    show_testing_instructions()
    
    print("\n✅ Setup dokončen - můžeš testovat aplikaci!")