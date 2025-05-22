#!/usr/bin/env python3
"""
Script to fix common test file issues:
1. from  import MagicMock, patch -> from unittest.mock import MagicMock, patch
2. Fix line indentation for assert statements
"""
import os
import re
import sys
import glob

def find_test_files(directory=None):
    """Find all test files in the directory."""
    if directory is None:
        directory = os.path.join(os.getcwd(), 'tests')
    
    test_files = []
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                test_files.append(file_path)
    
    return test_files

def fix_empty_imports(content):
    """Fix imports like 'from  import MagicMock, patch'."""
    return re.sub(
        r"from\s+\s*import\s+(MagicMock|patch|call|Mock)([^$\n]*)",
        r"from unittest.mock import \1\2",
        content
    )

def fix_indentation(content):
    """Fix unexpected indentation in assert statements."""
    # Find all lines with extra indentation
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check for lines with extra indentation
        if re.match(r"^(\s+)assert ", line):
            # Get the indentation level
            indent_match = re.match(r"^(\s+)", line)
            if indent_match:
                indent = indent_match.group(1)
                # If indentation is more than 8 spaces (2 levels), reduce it
                if len(indent) > 8:
                    # Reduce to 8 spaces (2 levels of indentation)
                    line = "        " + line.lstrip()
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_syntax_errors(content):
    """Fix common syntax errors."""
    # Fix missing commas in tuples
    content = re.sub(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)([^\)\,])",
        r"(\1, \2, (\3, \4), (\5, \6)),\7",
        content
    )
    
    return content

def fix_file(file_path):
    """Fix common issues in the test file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply fixes
    content = fix_empty_imports(content)
    content = fix_indentation(content)
    content = fix_syntax_errors(content)
    
    # If content was changed, write it back
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path}")
    else:
        print(f"No changes needed for {file_path}")

def main():
    """Main function to run the script."""
    if len(sys.argv) > 1:
        # Fix specific file
        file_path = sys.argv[1]
        fix_file(file_path)
    else:
        # Find and fix all test files
        test_files = find_test_files()
        for file_path in test_files:
            try:
                fix_file(file_path)
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")

if __name__ == "__main__":
    main()