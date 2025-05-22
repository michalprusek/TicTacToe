"""
Script to update imports from pyqt_gui_test_helper to pyqt_gui_unified_helper
"""
import os
import re

def find_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def update_imports(file_path):
    """Update imports in the given file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace import statements
    updated_content = re.sub(
        r'from tests\.unit\.pyqt_gui_test_helper import', 
        'from tests.unit.pyqt_gui_unified_helper import', 
        content
    )
    
    # If changes were made, write the updated content back to the file
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        print(f"Updated imports in {file_path}")

if __name__ == "__main__":
    # Find all Python files in the tests directory
    python_files = find_python_files('tests')
    
    # Update imports in each file
    for file_path in python_files:
        update_imports(file_path)
    
    print("Import updates completed.")