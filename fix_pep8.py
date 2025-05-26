#!/usr/bin/env python3
# @generated [all] Claude Code 2025-01-01: AI-assisted PEP8 fix script
"""
Script to automatically fix PEP8 line length issues in the app/ directory.
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
            # Try to fix common patterns
            fixed_line = fix_line(line, i + 1)
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


def fix_line(line, line_num):
    """Fix a single long line."""
    original = line
    stripped = line.rstrip()
    indent = len(line) - len(line.lstrip())

    # Skip if line is not too long
    if len(stripped) <= 79:
        return line

    # Fix AI annotation lines
    if '# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes' in line:
        return line.replace('and pylint fixes', '')

    # Fix import lines
    if line.strip().startswith('from ') and ' import ' in line:
        return fix_import_line(line)

    # Fix function definitions
    if 'def ' in line and '(' in line:
        return fix_function_def(line)

    # Fix class definitions
    if 'class ' in line and '(' in line:
        return fix_class_def(line)

    # Fix logger calls
    if '.logger.' in line and '(' in line:
        return fix_logger_call(line)

    # Fix string formatting
    if 'f"' in line or "f'" in line:
        return fix_f_string(line)

    # Fix long comments
    if line.strip().startswith('#') and len(stripped) > 79:
        return fix_comment(line)

    # Fix long string literals
    if '"""' in line or "'''" in line:
        return fix_docstring(line)

    # Fix method calls with many arguments
    if '(' in line and ')' in line and ',' in line:
        return fix_method_call(line)

    return line


def fix_import_line(line):
    """Fix long import lines."""
    if 'from ' in line and ' import ' in line:
        parts = line.split(' import ')
        if len(parts) == 2:
            from_part = parts[0]
            import_part = parts[1].strip()

            if len(line.rstrip()) > 79:
                indent = len(line) - len(line.lstrip())
                return [
                    from_part + ' import (\n',
                    ' ' * (indent + 4) + import_part + '\n',
                    ' ' * indent + ')\n'
                ]
    return line


def fix_function_def(line):
    """Fix long function definition lines."""
    if 'def ' in line and '(' in line and ')' in line:
        # Simple case - just break after opening parenthesis
        indent = len(line) - len(line.lstrip())
        if '(' in line and ')' in line:
            before_paren = line[:line.find('(') + 1]
            after_paren = line[line.find('(') + 1:]

            if len(line.rstrip()) > 79:
                return [
                    before_paren + '\n',
                    ' ' * (indent + 8) + after_paren
                ]
    return line


def fix_class_def(line):
    """Fix long class definition lines."""
    return line  # Keep as is for now


def fix_logger_call(line):
    """Fix long logger call lines."""
    if '.logger.' in line and '(' in line:
        indent = len(line) - len(line.lstrip())
        # Try to break after opening parenthesis
        if '(' in line and len(line.rstrip()) > 79:
            paren_pos = line.find('(')
            before_paren = line[:paren_pos + 1]
            after_paren = line[paren_pos + 1:]

            return [
                before_paren + '\n',
                ' ' * (indent + 4) + after_paren
            ]
    return line


def fix_f_string(line):
    """Fix long f-string lines."""
    return line  # Keep as is for now


def fix_comment(line):
    """Fix long comment lines."""
    if line.strip().startswith('#') and len(line.rstrip()) > 79:
        # Try to break long comments
        words = line.strip().split()
        if len(words) > 3:
            # Break roughly in the middle
            mid = len(words) // 2
            first_part = ' '.join(words[:mid])
            second_part = ' '.join(words[mid:])

            indent = len(line) - len(line.lstrip())
            return [
                ' ' * indent + first_part + '\n',
                ' ' * indent + '# ' + second_part + '\n'
            ]
    return line


def fix_docstring(line):
    """Fix long docstring lines."""
    return line  # Keep as is for now


def fix_method_call(line):
    """Fix long method call lines."""
    if '(' in line and ')' in line and ',' in line and len(line.rstrip()) > 79:
        # Try to break after opening parenthesis
        indent = len(line) - len(line.lstrip())
        paren_pos = line.find('(')
        if paren_pos > 0:
            before_paren = line[:paren_pos + 1]
            after_paren = line[paren_pos + 1:]

            return [
                before_paren + '\n',
                ' ' * (indent + 4) + after_paren
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
