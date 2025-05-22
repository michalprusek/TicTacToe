#!/usr/bin/env python3
"""
Script for detecting and analyzing code duplications in a Python project.
Provides functionality to:
1. Detect duplicate code blocks
2. Generate detailed reports
3. Suggest consolidated versions

Usage:
    python code_duplicates_detector.py [--dir DIR] [--min-lines MIN_LINES] [--similarity SIMILARITY] [--consolidate]
    
Options:
    --dir           Directory to analyze (default: current directory)
    --min-lines     Minimum lines for a duplicate block (default: 5)
    --similarity    Minimum similarity threshold (0.0-1.0) (default: 0.8)
    --consolidate   Generate consolidated versions of duplicated code
"""

import os
import re
import ast
import argparse
import hashlib
import difflib
import itertools
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any


class CodeBlock:
    """Represents a block of code from a file."""
    
    def __init__(self, file_path: str, start_line: int, end_line: int, code: str):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.code = code.strip()
        self.hash = self._compute_hash()
        
    def _compute_hash(self) -> str:
        """Compute a hash for this code block."""
        normalized_code = self._normalize_code(self.code)
        return hashlib.md5(normalized_code.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code to ignore whitespace, variable names, etc."""
        # Remove comments and blank lines
        lines = []
        for line in code.split('\n'):
            line = line.split('#')[0].rstrip()
            if line:
                lines.append(line)
        
        # Normalize whitespace (but preserve indentation)
        normalized_lines = []
        for line in lines:
            # Count leading spaces for indentation
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # Normalize the rest of the line: collapse multiple spaces into one
            content = re.sub(r'\s+', ' ', line[indent:].strip())
            normalized_lines.append(f"{indent_str}{content}")
            
        return '\n'.join(normalized_lines)
    
    def similarity_ratio(self, other: 'CodeBlock') -> float:
        """Compute similarity ratio between two code blocks using difflib."""
        return difflib.SequenceMatcher(None, self.code, other.code).ratio()
    
    def __repr__(self) -> str:
        return f"CodeBlock({self.file_path}:{self.start_line}-{self.end_line})"


class DuplicateGroup:
    """Represents a group of similar code blocks."""
    
    def __init__(self):
        self.blocks: List[CodeBlock] = []
        self.similarity_matrix: Dict[Tuple[int, int], float] = {}
        
    def add_block(self, block: CodeBlock) -> None:
        """Add a block to this group and update similarity matrix."""
        idx = len(self.blocks)
        for i, existing_block in enumerate(self.blocks):
            similarity = block.similarity_ratio(existing_block)
            self.similarity_matrix[(i, idx)] = similarity
            self.similarity_matrix[(idx, i)] = similarity
        self.blocks.append(block)
    
    @property
    def avg_similarity(self) -> float:
        """Calculate the average similarity in this group."""
        if not self.similarity_matrix:
            return 1.0
        return sum(self.similarity_matrix.values()) / len(self.similarity_matrix)
    
    @property
    def total_lines(self) -> int:
        """Calculate the total lines of code in all blocks."""
        return sum(block.end_line - block.start_line + 1 for block in self.blocks)
    
    @property
    def block_count(self) -> int:
        """Return the number of blocks in this group."""
        return len(self.blocks)
    
    def __repr__(self) -> str:
        return f"DuplicateGroup({len(self.blocks)} blocks, avg_sim={self.avg_similarity:.2f})"


class CodeDuplicateDetector:
    """Main class for detecting code duplicates."""
    
    def __init__(self, directory: str = '.', min_lines: int = 5, similarity_threshold: float = 0.8):
        self.directory = os.path.abspath(directory)
        self.min_lines = min_lines
        self.similarity_threshold = similarity_threshold
        self.code_blocks: List[CodeBlock] = []
        self.duplicate_groups: List[DuplicateGroup] = []
        
    def extract_code_blocks(self, file_path: str) -> List[CodeBlock]:
        """Extract syntactically meaningful code blocks from a Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Skip files with syntax errors
            return []
        
        blocks = []
        
        # Extract class and function definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = self._find_end_line(content, start_line)
                
                # Extract the code for this block
                code_lines = content.split('\n')[start_line-1:end_line]
                code = '\n'.join(code_lines)
                
                # Only include blocks that meet the minimum line requirement
                if end_line - start_line + 1 >= self.min_lines:
                    blocks.append(CodeBlock(file_path, start_line, end_line, code))
        
        return blocks
    
    def _find_end_line(self, content: str, start_line: int) -> int:
        """Find the end line of a block starting at start_line."""
        lines = content.split('\n')
        
        # Get the indentation of the initial line
        initial_indent = len(lines[start_line-1]) - len(lines[start_line-1].lstrip())
        
        # Find the first line after start_line with the same or less indentation
        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= initial_indent:
                if i > start_line:  # Make sure we've moved past the first line
                    return i - 1
        
        # If we reach the end of the file
        return len(lines) - 1
    
    def analyze_directory(self) -> None:
        """Analyze all Python files in the directory."""
        print(f"Analyzing directory: {self.directory}")
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if "uArm-Python-SDK" not in file_path:  # Skip SDK files
                        print(f"Processing file: {file_path}")
                        self.code_blocks.extend(self.extract_code_blocks(file_path))
        
        print(f"Extracted {len(self.code_blocks)} code blocks")
    
    def detect_duplicates(self) -> None:
        """Detect duplicate code blocks based on similarity."""
        # Group blocks by hash (exact duplicates)
        blocks_by_hash: Dict[str, List[CodeBlock]] = defaultdict(list)
        for block in self.code_blocks:
            blocks_by_hash[block.hash].append(block)
        
        # Create duplicate groups for exact matches
        for hash_val, blocks in blocks_by_hash.items():
            if len(blocks) > 1:
                group = DuplicateGroup()
                for block in blocks:
                    group.add_block(block)
                self.duplicate_groups.append(group)
        
        # Find similar blocks using pairwise comparison
        ungrouped_blocks = [block for block in self.code_blocks 
                            if len(blocks_by_hash[block.hash]) == 1]
        
        # Group by approximate line count to reduce comparisons
        blocks_by_size: Dict[int, List[CodeBlock]] = defaultdict(list)
        for block in ungrouped_blocks:
            size = (block.end_line - block.start_line + 1) // 5  # Group in chunks of 5 lines
            blocks_by_size[size].append(block)
        
        # For each size group, compare blocks
        for size, size_blocks in blocks_by_size.items():
            # Compare with blocks of similar size (same, +1, -1)
            comparison_blocks = []
            for s in [size-1, size, size+1]:
                comparison_blocks.extend(blocks_by_size.get(s, []))
            
            for i, block1 in enumerate(size_blocks):
                similar_blocks = [block1]
                
                for j in range(i+1, len(comparison_blocks)):
                    block2 = comparison_blocks[j]
                    
                    # Skip if blocks are from the same file (unless they're very far apart)
                    if (block1.file_path == block2.file_path and 
                        abs(block1.start_line - block2.start_line) < 50):
                        continue
                    
                    similarity = block1.similarity_ratio(block2)
                    if similarity >= self.similarity_threshold:
                        similar_blocks.append(block2)
                
                if len(similar_blocks) > 1:
                    group = DuplicateGroup()
                    for block in similar_blocks:
                        group.add_block(block)
                    self.duplicate_groups.append(group)
        
        # Sort groups by total lines (most impactful duplications first)
        self.duplicate_groups.sort(key=lambda g: (g.total_lines, g.block_count), reverse=True)
        
        print(f"Found {len(self.duplicate_groups)} duplicate groups")
    
    def print_duplicates_report(self) -> None:
        """Print a detailed report of all duplicate code blocks."""
        if not self.duplicate_groups:
            print("No duplicates found.")
            return
        
        print("\n===== DUPLICATE CODE REPORT =====")
        print(f"Minimum lines per block: {self.min_lines}")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Total duplicate groups: {len(self.duplicate_groups)}")
        
        total_files = len(set(block.file_path for group in self.duplicate_groups 
                              for block in group.blocks))
        print(f"Files with duplicates: {total_files}")
        
        total_duplicate_lines = sum(group.total_lines for group in self.duplicate_groups)
        print(f"Total duplicate lines: {total_duplicate_lines}")
        
        for i, group in enumerate(self.duplicate_groups, 1):
            print(f"\n----- Group {i} -----")
            print(f"Blocks: {group.block_count}")
            print(f"Average similarity: {group.avg_similarity:.2f}")
            print(f"Total lines: {group.total_lines}")
            
            for j, block in enumerate(group.blocks, 1):
                rel_path = os.path.relpath(block.file_path, self.directory)
                print(f"  Block {j}: {rel_path}:{block.start_line}-{block.end_line} ({block.end_line - block.start_line + 1} lines)")
    
    def generate_consolidated_versions(self) -> Dict[int, str]:
        """Generate consolidated versions of duplicate code blocks."""
        consolidated = {}
        
        for i, group in enumerate(self.duplicate_groups, 1):
            if len(group.blocks) < 2:
                continue
            
            # Start with the first block as the base
            base_block = group.blocks[0]
            consolidated_code = base_block.code
            
            # Extract function/class name if applicable
            match = re.match(r'(def|class)\s+(\w+)', base_block.code)
            name = match.group(2) if match else f"consolidated_block_{i}"
            
            # Create a consolidated version with a new name
            if 'def ' in consolidated_code:
                if 'def __init__' in consolidated_code:
                    # This is a class method
                    consolidated[i] = consolidated_code
                else:
                    # This is a standalone function
                    consolidated[i] = consolidated_code.replace(name, f"consolidated_{name}", 1)
            elif 'class ' in consolidated_code:
                # This is a class definition
                consolidated[i] = consolidated_code.replace(name, f"Consolidated{name}", 1)
            else:
                # This is some other code block
                consolidated[i] = f"# Consolidated version of duplicate block {i}\n{consolidated_code}"
        
        return consolidated
    
    def save_consolidated_code(self, output_dir: str = 'consolidated_code') -> None:
        """Save consolidated versions of duplicate code to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        consolidated = self.generate_consolidated_versions()
        
        for group_id, code in consolidated.items():
            filename = f"consolidated_group_{group_id}.py"
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print(f"Saved consolidated code for group {group_id} to {output_path}")
    
    def consolidate_file(self, file_path: str, only_groups: List[int] = None) -> str:
        """Generate a version of a file with duplicate code blocks replaced by imports."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        new_lines = list(lines)  # Copy to modify
        
        # Track imports to add
        imports_to_add = set()
        
        # Process each duplicate group that affects this file
        for group_idx, group in enumerate(self.duplicate_groups, 1):
            if only_groups and group_idx not in only_groups:
                continue
                
            # Find blocks from this file in the group
            file_blocks = [block for block in group.blocks if block.file_path == file_path]
            if not file_blocks:
                continue
            
            # Get the consolidated module and function/class name
            consolidated_module = f"consolidated_code.consolidated_group_{group_idx}"
            
            # Determine the type and name of the first block
            first_block = file_blocks[0]
            first_line = first_block.code.split('\n')[0].strip()
            
            if first_line.startswith('def '):
                match = re.match(r'def\s+(\w+)', first_line)
                name = match.group(1) if match else f"function_{group_idx}"
                consolidated_name = f"consolidated_{name}"
                import_stmt = f"from {consolidated_module} import {consolidated_name} as {name}"
            
            elif first_line.startswith('class '):
                match = re.match(r'class\s+(\w+)', first_line)
                name = match.group(1) if match else f"Class{group_idx}"
                consolidated_name = f"Consolidated{name}"
                import_stmt = f"from {consolidated_module} import {consolidated_name} as {name}"
            
            else:
                # Skip blocks that aren't functions or classes
                continue
            
            imports_to_add.add(import_stmt)
            
            # Replace code blocks with pass statements or remove them
            for block in file_blocks:
                # Mark lines for removal (replace with empty strings first to maintain line numbers)
                for i in range(block.start_line - 1, block.end_line):
                    new_lines[i] = ""
        
        # Add imports at the top, after any existing imports
        import_line = 0
        for i, line in enumerate(lines):
            if re.match(r'import\s+|from\s+\w+\s+import', line):
                import_line = i + 1
        
        # Insert the new imports
        for import_stmt in sorted(imports_to_add):
            new_lines.insert(import_line, import_stmt)
            import_line += 1
        
        # Remove consecutive empty lines
        result_lines = []
        for i, line in enumerate(new_lines):
            if not line and i > 0 and not new_lines[i-1]:
                continue
            result_lines.append(line)
        
        return '\n'.join(result_lines)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Code duplication detector and consolidator.')
    parser.add_argument('--dir', default='.', help='Directory to analyze')
    parser.add_argument('--min-lines', type=int, default=5, help='Minimum lines for a duplicate block')
    parser.add_argument('--similarity', type=float, default=0.8, 
                        help='Minimum similarity threshold (0.0-1.0)')
    parser.add_argument('--consolidate', action='store_true', 
                        help='Generate consolidated versions of duplicated code')
    
    args = parser.parse_args()
    
    detector = CodeDuplicateDetector(
        directory=args.dir,
        min_lines=args.min_lines,
        similarity_threshold=args.similarity
    )
    
    detector.analyze_directory()
    detector.detect_duplicates()
    detector.print_duplicates_report()
    
    if args.consolidate:
        detector.save_consolidated_code()


if __name__ == "__main__":
    main()