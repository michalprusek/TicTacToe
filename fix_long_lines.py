#!/usr/bin/env python3
# @generated [all] Claude Code 2025-01-01: AI-assisted PEP8 long line fix script
"""
Advanced script to automatically fix PEP8 E501 long line issues.
"""
import os
import re
import sys
from pathlib import Path


def fix_long_lines_in_file(file_path):
    """Fix long lines in a Python file."""
    print(f"Processing {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for i, line in enumerate(lines):
        if len(line.rstrip()) > 79:
            fixed_line = fix_long_line(line, i + 1)
            if fixed_line != line:
                modified = True
                if isinstance(fixed_line, list):
                    new_lines.extend(fixed_line)
                else:
                    new_lines.append(fixed_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  Fixed {file_path}")

    return modified


def fix_long_line(line, line_num):
    """Fix a single long line."""
    stripped = line.rstrip()
    indent = len(line) - len(line.lstrip())

    # Skip if line is not too long
    if len(stripped) <= 79:
        return line

    # Fix docstrings - shorten them
    if '"""' in line and 'return' in line.lower():
        if 'return changed cell if any' in line:
            return line.replace('return changed cell if any', 'return changed cell')
        if 'Process individual symbol' in line:
            return line.replace('Process individual symbol', 'Process symbol')

    # Fix logger calls with long messages
    if '.logger.' in line and '"' in line:
        return fix_logger_message(line, indent)

    # Fix function definitions with long parameter lists
    if 'def ' in line and '(' in line and ')' in line:
        return fix_function_definition(line, indent)

    # Fix long comments with pylint disable
    if '# pylint: disable=' in line:
        return fix_pylint_comment(line, indent)

    # Fix long string literals
    if '"' in line and len(stripped) > 79:
        return fix_string_literal(line, indent)

    # Fix long variable assignments
    if '=' in line and not line.strip().startswith('#'):
        return fix_assignment(line, indent)

    return line


def fix_logger_message(line, indent):
    """Fix long logger messages."""
    # Shorten common long messages
    replacements = {
        'Invalid grid point indices for cell center calculation': 'Invalid grid indices for cell calculation',
        'IndexError computing logic cell centers from _grid_points': 'IndexError computing cell centers',
        'Error computing logic cell centers from _grid_points': 'Error computing cell centers',
        'Computed %d cell centers for game logic from _grid_points.': 'Computed %d cell centers for game logic.',
        'Winner: %s, Line: %s': 'Winner: %s, Line: %s',
    }

    for old, new in replacements.items():
        if old in line:
            return line.replace(old, new)

    return line


def fix_function_definition(line, indent):
    """Fix long function definitions."""
    # For update_from_detection function
    if 'update_from_detection' in line and 'too-many-arguments' in line:
        # Split the pylint comment to next line
        parts = line.split('  # pylint:')
        if len(parts) == 2:
            return [
                parts[0].rstrip() + '\n',
                ' ' * indent + '# pylint:' + parts[1]
            ]

    return line


def fix_pylint_comment(line, indent):
    """Fix long pylint disable comments."""
    if 'too-many-arguments,too-many-branches' in line:
        return line.replace('too-many-arguments,too-many-branches', 'too-many-arguments')

    if 'unused-argument' in line and len(line.rstrip()) > 79:
        # Move pylint comment to previous line
        parts = line.split('  # pylint:')
        if len(parts) >= 2:
            code_part = parts[0].rstrip()
            comment_part = '  # pylint:' + parts[1]
            return [
                ' ' * indent + comment_part,
                code_part + '\n'
            ]

    return line


def fix_string_literal(line, indent):
    """Fix long string literals."""
    # Common string shortenings
    replacements = {
        'perspectiveTransform(  # pylint: disable=no-member': 'perspectiveTransform(',
    }

    for old, new in replacements.items():
        if old in line:
            return line.replace(old, new)

    return line


def fix_assignment(line, indent):
    """Fix long assignment lines."""
    if 'points_count = (len(self._grid_points) if self._grid_points is not None' in line:
        return [
            ' ' * indent + 'points_count = (\n',
            ' ' * (indent + 4) + 'len(self._grid_points) if self._grid_points is not None\n',
            ' ' * (indent + 4) + "else 'None'\n",
            ' ' * indent + ')\n'
        ]

    return line


def main():
    """Main function to process all Python files in app/ directory."""
    app_dir = Path('app')
    if not app_dir.exists():
        print("app/ directory not found")
        return 1

    python_files = list(app_dir.rglob('*.py'))
    print(f"Found {len(python_files)} Python files")

    total_fixed = 0
    for file_path in python_files:
        if fix_long_lines_in_file(file_path):
            total_fixed += 1

    print(f"Fixed {total_fixed} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())
