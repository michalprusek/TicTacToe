#!/usr/bin/env python3
import re

def convert_assertions(file_path):
    """Convert pytest assertions to unittest assertions."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Common assertion conversions
    patterns = [
        (r'assert\s+(.+?)\s*==\s*(.+)', r'self.assertEqual(\1, \2)'),
        (r'assert\s+(.+?)\s*!=\s*(.+)', r'self.assertNotEqual(\1, \2)'),
        (r'assert\s+(.+?)\s+is\s+None', r'self.assertIsNone(\1)'),
        (r'assert\s+(.+?)\s+is\s+not\s+None', r'self.assertIsNotNone(\1)'),
        (r'assert\s+(.+?)\s+not\s+in\s+(.+)', r'self.assertNotIn(\1, \2)'),
        (r'assert\s+(.+?)\s+in\s+(.+)', r'self.assertIn(\1, \2)'),
        (r'assert\s+(.+)', r'self.assertTrue(\1)'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    convert_assertions('/Users/michalprusek/PycharmProjects/TicTacToe/tests/test_game_logic_unittest.py')
    print("Converted assertions to unittest format")