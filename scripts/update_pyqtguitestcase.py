#!/usr/bin/env python3
"""
Script to update PyQtGuiTestCase references to PyQtGuiTestCaseBase
from conftest_common.py.
"""
import os
import re
import glob

# Directory to search
TEST_DIR = "tests/unit"

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
    
    # Check if the file contains PyQtGuiTestCase
    if "PyQtGuiTestCase" in content:
        # Update imports
        content = content.replace(
            "from tests.unit.pyqt_gui_unified_helper import PyQtGuiTestCase", 
            "from tests.conftest_common import PyQtGuiTestCaseBase"
        )
        
        # Replace class references
        content = content.replace("PyQtGuiTestCase", "PyQtGuiTestCaseBase")
        
        modified_files.append(file_path)
        
        # Write the updated content back to the file
        with open(file_path, "w") as f:
            f.write(content)

# Print summary
print(f"Updated {len(modified_files)} files:")
for file in modified_files:
    print(f"  - {file}")