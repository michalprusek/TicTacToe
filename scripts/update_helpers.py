#!/usr/bin/env python3
"""
Script to update test files to use the common fixture from conftest_common.py.

This script looks for test files that import MockTicTacToeApp from
pyqt_gui_unified_helper.py and updates them to import it from conftest_common.py instead.
"""
import os
import re
import glob

# Directory to search
TEST_DIR = "tests/unit"

# Pattern to search for
OLD_IMPORT = "from tests.unit.pyqt_gui_unified_helper import PyQtGuiTestCase, MockTicTacToeApp"
OLD_IMPORT_MOCKONLY = "from tests.unit.pyqt_gui_unified_helper import MockTicTacToeApp"
NEW_IMPORT = "from tests.conftest_common import MockTicTacToeApp, qt_app"

PYQTGUITESTCASE_IMPORT = "from tests.unit.pyqt_gui_unified_helper import PyQtGuiTestCase"

# Find all Python files in the test directory
python_files = glob.glob(os.path.join(TEST_DIR, "*.py"))

# Keep track of modified files
modified_files = []

for file_path in python_files:
    # Skip the conftest_common.py file itself and pyqt_gui_unified_helper.py
    if file_path.endswith("conftest_common.py") or file_path.endswith("pyqt_gui_unified_helper.py"):
        continue
        
    # Read the file content
    with open(file_path, "r") as f:
        content = f.read()
    
    # Check if the file contains the old import
    if OLD_IMPORT in content:
        # Replace the old import with the new one
        new_content = content.replace(OLD_IMPORT, NEW_IMPORT + "\n" + PYQTGUITESTCASE_IMPORT)
        modified_files.append(file_path)
        
        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(new_content)
            
    elif OLD_IMPORT_MOCKONLY in content:
        # Replace the old import with the new one
        new_content = content.replace(OLD_IMPORT_MOCKONLY, "from tests.conftest_common import MockTicTacToeApp, qt_app")
        modified_files.append(file_path)
        
        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(new_content)

# Print summary
print(f"Updated {len(modified_files)} files:")
for file in modified_files:
    print(f"  - {file}")