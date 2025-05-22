#!/usr/bin/env python3
"""
Script to fix import statements in test files.
"""
import os
import re


def fix_imports_in_file(file_path):
    """Fix import statements in a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Mapping of old imports to new imports
    imports_to_fix = {
        r'import\s+game_logic': 'from app.main import game_logic',
        r'import\s+arm_controller': 'from app.main import arm_controller',
        r'import\s+game_detector': 'from app.main import game_detector',
        r'import\s+pyqt_gui': 'from app.main import pyqt_gui',
        r'import\s+main_pyqt': 'from app.main import main_pyqt',
        r'from\s+arm_controller\s+import': 'from app.main.arm_controller import',
        r'from\s+game_logic\s+import': 'from app.main.game_logic import',
        r'from\s+game_detector\s+import': 'from app.main.game_detector import',
        r'from\s+pyqt_gui\s+import': 'from app.main.pyqt_gui import',
        r'from\s+main_pyqt\s+import': 'from app.main.main_pyqt import',
    }

    fixed_content = content
    for old_pattern, new_import in imports_to_fix.items():
        fixed_content = re.sub(old_pattern, new_import, fixed_content)

    if fixed_content != content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(fixed_content)
        print(f"Fixed imports in {file_path}")
    else:
        print(f"No changes needed in {file_path}")


def fix_imports_in_directory(directory):
    """Fix import statements in all Python files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_imports_in_file(file_path)


if __name__ == "__main__":
    fix_imports_in_directory("/Users/michalprusek/PycharmProjects/TicTacToe/tests")