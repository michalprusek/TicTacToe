#!/usr/bin/env python3
"""
Script to remove trailing spaces from files.
"""
import os
import sys


def fix_trailing_spaces(file_path):
    """Remove trailing spaces from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    fixed_lines = [line.rstrip() + '\n' for line in lines]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(fixed_lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_trailing_spaces(sys.argv[1])
        print(f"Fixed trailing spaces in {sys.argv[1]}")
    else:
        print("Usage: python fix_trailing_spaces.py <file_path>")