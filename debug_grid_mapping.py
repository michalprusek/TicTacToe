#!/usr/bin/env python3
"""
Debug script pro analýzu mapování grid points → cell centers.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.main.path_utils import setup_project_path
setup_project_path()

def analyze_grid_indexing():
    """Analyzuje jak se indexují grid points a cell centers."""
    print("=== ANALÝZA GRID INDEXOVÁNÍ ===")
    
    # Simulace 4x4 grid points (16 bodů)
    print("Grid points uspořádání (4x4 = 16 bodů):")
    
    # Index mapping
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            print(f"  Grid point ({row},{col}) → index {idx:2d}")
    
    print("\nCell centers počítání (3x3 = 9 buněk):")
    print("Pro každou buňku (r,c) používáme 4 okolní grid points:")
    
    for r_cell in range(3):
        for c_cell in range(3):
            # Indexy grid points podle kódu z game_state.py
            p_tl_idx = r_cell * 4 + c_cell          # top-left
            p_tr_idx = r_cell * 4 + (c_cell + 1)    # top-right  
            p_bl_idx = (r_cell + 1) * 4 + c_cell    # bottom-left
            p_br_idx = (r_cell + 1) * 4 + (c_cell + 1)  # bottom-right
            
            print(f"  Cell ({r_cell},{c_cell}):")
            print(f"    Grid points: TL={p_tl_idx:2d}, TR={p_tr_idx:2d}, BL={p_bl_idx:2d}, BR={p_br_idx:2d}")

def test_coordinate_conversion():
    """Test konverze souřadnic."""
    print("\n=== TEST KONVERZE SOUŘADNIC ===")
    
    # Typické row,col pozice v tic-tac-toe
    positions = [
        (0, 0), (0, 1), (0, 2),  # horní řada
        (1, 0), (1, 1), (1, 2),  # střední řada  
        (2, 0), (2, 1), (2, 2)   # spodní řada
    ]
    
    print("Tic-tac-toe pozice:")
    print("  (0,0) (0,1) (0,2)")
    print("  (1,0) (1,1) (1,2)")  
    print("  (2,0) (2,1) (2,2)")
    
    print("\nPozor na možné problémy:")
    print("  1. Je row=Y a col=X? Nebo opačně?")
    print("  2. Jsou indexy 0-based?")
    print("  3. Je origem vlevo nahoře nebo jinde?")
    print("  4. Jsou grid points správně seřazené po detekci?")

if __name__ == "__main__":
    print("🔍 DEBUG ANALÝZA GRID MAPOVÁNÍ")
    print("=" * 40)
    
    analyze_grid_indexing()
    test_coordinate_conversion()
    
    print("\n💡 DOPORUČENÍ PRO DEBUGGING:")
    print("1. Spusť aplikaci a pozoruj logy z get_cell_center_uv()")
    print("2. Zkus manuálně kliknout na buňky a porovnej UV souřadnice")
    print("3. Zkontroluj, jestli grid points jsou správně seřazené po detekci")
    print("4. Možná je problém v row/col vs x/y koordinátním systému")