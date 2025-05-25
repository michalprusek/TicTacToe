#!/usr/bin/env python3
"""
Script to fix logging format string errors in game_state.py
"""
import re
import sys

def fix_logging_format_strings(file_path):
    """Fix logging format strings that have E1205 errors."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern 1: Remove colons at the end of logging format strings with %d
    # e.g., "DETECTED %d SYMBOLS:" -> "DETECTED %d SYMBOLS"
    patterns = [
        # Remove trailing colons from format strings with placeholders
        (r'(logger\.\w+\(["\'])([^"\']*%[^"\']*):(["\'])', r'\1\2\3'),
        
        # Fix single quotes around format placeholders
        # e.g., "'%s'" -> "%s"
        (r'(logger\.\w+\(["\'][^"\']*)"([^"\']*)"([^"\']*["\'])', r'\1\2\3'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Additional specific fixes for the colon issue
    content = content.replace('SYMBOLS:",', 'SYMBOLS",')
    content = content.replace('SYMBOLS:"', 'SYMBOLS"')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    file_path = 'app/core/game_state.py'
    fix_logging_format_strings(file_path)
    print(f"Fixed logging format strings in {file_path}")
