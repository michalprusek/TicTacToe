#!/usr/bin/env python3
"""
Script to convert unittest files to pytest format.
Usage: python convert_unittest_to_pytest.py [test_file_path]
If no file path is provided, it will find and convert all test files.
"""
import os
import re
import sys
import ast
from typing import List, Dict, Tuple, Optional, Set


def find_test_files(directory: str) -> List[str]:
    """Find all unittest test files in the directory."""
    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    if "import unittest" in content or "from unittest" in content:
                        test_files.append(file_path)
    return test_files


def contains_unittest(content: str) -> bool:
    """Check if file contains unittest imports or TestCase class."""
    return (
        re.search(r"import\s+unittest", content) is not None
        or re.search(r"from\s+unittest\s+import", content) is not None
        or re.search(r"class\s+\w+\(unittest\.TestCase\)", content) is not None
        or re.search(r"class\s+\w+\(.*TestCase\)", content) is not None
    )


def extract_setup_and_teardown(class_node: ast.ClassDef) -> Tuple[Optional[ast.FunctionDef], Optional[ast.FunctionDef]]:
    """Extract setUp and tearDown methods from TestCase class."""
    setup = None
    teardown = None
    
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "setUp":
                setup = node
            elif node.name == "tearDown":
                teardown = node
    
    return setup, teardown


def fix_mock_patches(source: str) -> str:
    """Fix mock patch paths in the source code."""
    # Fix common module paths
    replacements = [
        (r"with\s+mock\.patch\(['\"]game_logic\.(.*?)['\"]", r"with mock.patch('app.main.game_logic.\1'"),
        (r"with\s+mock\.patch\(['\"]game_detector\.(.*?)['\"]", r"with mock.patch('app.main.game_detector.\1'"),
        (r"with\s+mock\.patch\(['\"]game_state\.(.*?)['\"]", r"with mock.patch('app.core.game_state.\1'"),
        (r"with\s+mock\.patch\(['\"]strategy\.(.*?)['\"]", r"with mock.patch('app.core.strategy.\1'"),
        (r"with\s+mock\.patch\(['\"]arm_controller\.(.*?)['\"]", r"with mock.patch('app.main.arm_controller.\1'"),
        (r"with\s+mock\.patch\(['\"]arm_thread\.(.*?)['\"]", r"with mock.patch('app.core.arm_thread.\1'"),
        (r"with\s+mock\.patch\(['\"]detection_thread\.(.*?)['\"]", r"with mock.patch('app.core.detection_thread.\1'"),
        (r"with\s+mock\.patch\(['\"]pyqt_gui\.(.*?)['\"]", r"with mock.patch('app.main.pyqt_gui.\1'"),
        (r"with\s+mock\.patch\(['\"]main_pyqt\.(.*?)['\"]", r"with mock.patch('app.main.main_pyqt.\1'"),
        (r"with\s+mock\.patch\(['\"]config\.(.*?)['\"]", r"with mock.patch('app.core.config.\1'"),
        # Fix class decorators
        (r"@patch\(['\"]game_logic\.(.*?)['\"]", r"@patch('app.main.game_logic.\1'"),
        (r"@patch\(['\"]game_detector\.(.*?)['\"]", r"@patch('app.main.game_detector.\1'"),
        (r"@patch\(['\"]game_state\.(.*?)['\"]", r"@patch('app.core.game_state.\1'"),
        (r"@patch\(['\"]strategy\.(.*?)['\"]", r"@patch('app.core.strategy.\1'"),
        (r"@patch\(['\"]arm_controller\.(.*?)['\"]", r"@patch('app.main.arm_controller.\1'"),
        (r"@patch\(['\"]arm_thread\.(.*?)['\"]", r"@patch('app.core.arm_thread.\1'"),
        (r"@patch\(['\"]detection_thread\.(.*?)['\"]", r"@patch('app.core.detection_thread.\1'"),
        (r"@patch\(['\"]pyqt_gui\.(.*?)['\"]", r"@patch('app.main.pyqt_gui.\1'"),
        (r"@patch\(['\"]main_pyqt\.(.*?)['\"]", r"@patch('app.main.main_pyqt.\1'"),
        (r"@patch\(['\"]config\.(.*?)['\"]", r"@patch('app.core.config.\1'"),
    ]

    for pattern, replacement in replacements:
        source = re.sub(pattern, replacement, source)

    return source


def convert_assertions(source: str) -> str:
    """Convert unittest assertion methods to pytest assertions."""
    # Convert simple assertions
    patterns = [
        (r"self\.assertEqual\((.*?),\s*(.*?)\)", r"assert \1 == \2"),
        (r"self\.assertEquals\((.*?),\s*(.*?)\)", r"assert \1 == \2"),
        (r"self\.assertNotEqual\((.*?),\s*(.*?)\)", r"assert \1 != \2"),
        (r"self\.assertTrue\((.*?)\)", r"assert \1"),
        (r"self\.assertFalse\((.*?)\)", r"assert not \1"),
        (r"self\.assertIsNone\((.*?)\)", r"assert \1 is None"),
        (r"self\.assertIsNotNone\((.*?)\)", r"assert \1 is not None"),
        (r"self\.assertIn\((.*?),\s*(.*?)\)", r"assert \1 in \2"),
        (r"self\.assertNotIn\((.*?),\s*(.*?)\)", r"assert \1 not in \2"),
        (r"self\.assertIs\((.*?),\s*(.*?)\)", r"assert \1 is \2"),
        (r"self\.assertIsNot\((.*?),\s*(.*?)\)", r"assert \1 is not \2"),
        (r"self\.assertIsInstance\((.*?),\s*(.*?)\)", r"assert isinstance(\1, \2)"),
        (r"self\.assertNotIsInstance\((.*?),\s*(.*?)\)", r"assert not isinstance(\1, \2)"),
        (r"self\.assertGreater\((.*?),\s*(.*?)\)", r"assert \1 > \2"),
        (r"self\.assertGreaterEqual\((.*?),\s*(.*?)\)", r"assert \1 >= \2"),
        (r"self\.assertLess\((.*?),\s*(.*?)\)", r"assert \1 < \2"),
        (r"self\.assertLessEqual\((.*?),\s*(.*?)\)", r"assert \1 <= \2"),
        (r"self\.assertAlmostEqual\((.*?),\s*(.*?)\)", r"assert \1 == pytest.approx(\2)"),
        (r"self\.assertNotAlmostEqual\((.*?),\s*(.*?)\)", r"assert \1 != pytest.approx(\2)"),
        (r"self\.assertRaises\((.*?),\s*(.*?)\)", r"with pytest.raises(\1):\n        \2"),
    ]

    # Apply each pattern
    for pattern, replacement in patterns:
        # Using a complex pattern matching to avoid replacing inside strings or comments
        lines = source.split('\n')
        for i, line in enumerate(lines):
            # Apply pattern on this line
            match = re.search(pattern, line)
            if match:
                # Ensure indentation is preserved
                indentation = re.match(r'^\s*', line).group(0)
                replacement_with_indent = indentation + replacement
                for j in range(1, match.lastindex + 1 if match.lastindex else 1):
                    replacement_with_indent = replacement_with_indent.replace(f"\\{j}", match.group(j))
                lines[i] = re.sub(pattern, replacement_with_indent, line)
        source = '\n'.join(lines)

    return source


def convert_imports(source: str) -> str:
    """Convert unittest imports to pytest imports."""
    # Replace unittest imports with pytest
    updated_source = re.sub(
        r"import unittest\n", 
        "import pytest\n", 
        source
    )
    updated_source = re.sub(
        r"from unittest import TestCase\n", 
        "import pytest\n", 
        updated_source
    )
    
    # Handle unittest.mock to retain mock functionality
    if re.search(r"from unittest import mock", updated_source):
        updated_source = re.sub(
            r"from unittest import mock", 
            "from unittest import mock", 
            updated_source
        )
    
    # Handle the case where TestCase is imported directly
    if "TestCase" in updated_source and not "from unittest import TestCase" in updated_source:
        updated_source = re.sub(
            r"from unittest import .*?(TestCase).*?\n", 
            lambda m: "import pytest\n" + (m.group(0) if "mock" in m.group(0) else ""),
            updated_source
        )
        
    return updated_source


def identify_fixture_candidates(class_node: ast.ClassDef) -> Set[str]:
    """Identify instance variables that could be converted to fixtures."""
    fixture_candidates = set()
    setup_node = None
    
    # Find setUp method
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "setUp":
            setup_node = node
            break
            
    if not setup_node:
        return fixture_candidates
        
    # Find self.xxx assignments in setUp
    for node in ast.walk(setup_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == "self":
                        fixture_candidates.add(target.attr)
                        
    return fixture_candidates


def create_fixture_from_setup(setup_node: ast.FunctionDef, fixture_name: str) -> str:
    """Create a pytest fixture from a setUp method."""
    # Get the body of the setUp method excluding the function declaration
    setup_source = ast.unparse(setup_node)
    
    # Extract the body of the function
    body_lines = setup_source.split('\n')[1:]  # Skip the function declaration
    body = '\n'.join(body_lines)
    
    # Replace self.x with local variables
    body = re.sub(r"self\.(\w+)", r"\1", body)
    
    # Create a fixture
    fixture = f"""
@pytest.fixture
def {fixture_name}():
    \"\"\"Provide {fixture_name} for tests.\"\"\"
{body}
    return {fixture_name}
"""
    return fixture


def convert_class_tests_to_functions(
    content: str, class_node: ast.ClassDef, fixture_candidates: Set[str]
) -> Tuple[str, List[str]]:
    """Convert test methods in a class to individual pytest functions."""
    test_functions = []
    fixtures_used = []
    
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Extract function source
            function_source = ast.unparse(node)
            
            # Create a new pytest function 
            param_list = []
            for fixture in fixture_candidates:
                if re.search(f"self\\.{fixture}", function_source):
                    param_list.append(fixture)
                    if fixture not in fixtures_used:
                        fixtures_used.append(fixture)
            
            param_str = ", ".join(param_list)
            function_name = node.name
            
            # Convert function body
            function_body = function_source.split("\n", 1)[1] if "\n" in function_source else ""
            
            # Replace self.x with x (for fixture variables)
            for fixture in fixture_candidates:
                function_body = re.sub(f"self\\.{fixture}", fixture, function_body)
                
            # Replace other self.x references (likely test class helper methods)
            function_body = re.sub(r"self\.(\w+)\(", r"\1(", function_body)
            
            new_function = f"""
def {function_name}({param_str}):
    \"\"\"{node.body[0].value.s if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str) else ""}\"\"\"
{function_body}
"""
            test_functions.append(new_function)
            
    # Remove the class from the content
    class_definition = ast.unparse(class_node)
    content = content.replace(class_definition, "")
    
    return content, fixtures_used


def create_fixtures_from_class(class_node: ast.ClassDef) -> List[str]:
    """Create fixtures from class instance methods and variables."""
    fixtures = []
    fixture_candidates = identify_fixture_candidates(class_node)
    
    # Find setUp method
    setup_node = None
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "setUp":
            setup_node = node
            break
            
    if setup_node and fixture_candidates:
        # Process the setUp method for each fixture candidate
        for fixture in fixture_candidates:
            fixtures.append(create_fixture_for_variable(setup_node, fixture))
            
    return fixtures


def create_fixture_for_variable(setup_node: ast.FunctionDef, var_name: str) -> str:
    """Create a pytest fixture for a single variable from setUp."""
    # Extract the relevant parts from setUp that affect this variable
    setup_src = ast.unparse(setup_node)
    
    # Find and extract assignments to self.var_name
    assignments = []
    for line in setup_src.split('\n'):
        if f"self.{var_name}" in line and "=" in line:
            # Replace self.var_name with var_name
            modified_line = line.replace(f"self.{var_name}", var_name)
            assignments.append(modified_line)
        elif f"self.{var_name}" in line:
            # This line references var_name but doesn't assign to it
            modified_line = line.replace(f"self.{var_name}", var_name)
            assignments.append(modified_line)
    
    # Create the fixture
    fixture_code = f"""
@pytest.fixture
def {var_name}():
    \"\"\"{var_name} fixture for tests.\"\"\"
"""
    
    # Add the assignments
    for assignment in assignments:
        # Extract indentation
        indent = len(assignment) - len(assignment.lstrip())
        # Add proper indentation (4 spaces)
        fixture_code += f"    {assignment[indent:]}\n"
        
    # Add return statement
    fixture_code += f"    return {var_name}\n"
    
    return fixture_code


def convert_file(file_path: str, dry_run: bool = False) -> str:
    """Convert a unittest file to pytest format."""
    with open(file_path, "r") as f:
        content = f.read()

    if not contains_unittest(content):
        print(f"Skipping {file_path} - does not contain unittest code")
        return content

    # Parse the file
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Skipping {file_path} - syntax error when parsing")
        return content
    except IndentationError:
        print(f"Skipping {file_path} - indentation error when parsing")
        return content
    
    # Find TestCase classes
    test_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (isinstance(base, ast.Name) and base.id == 'TestCase') or \
                   (isinstance(base, ast.Attribute) and base.attr == 'TestCase'):
                    test_classes.append(node)
                    
    if not test_classes:
        print(f"No TestCase classes found in {file_path}")
        return content
        
    # Process imports
    new_content = convert_imports(content)
    
    # Add additional pytest import if needed for specific assertions
    if "self.assertAlmostEqual" in new_content or "self.assertNotAlmostEqual" in new_content:
        new_content = new_content.replace("import pytest", "import pytest")
    
    # Process each TestCase class
    all_fixtures = []
    for class_node in test_classes:
        # Identify fixture candidates
        fixture_candidates = identify_fixture_candidates(class_node)
        
        # Convert class tests to functions
        new_content, fixtures_used = convert_class_tests_to_functions(
            new_content, class_node, fixture_candidates
        )
        
        # Create fixtures from setUp
        setup_node, _ = extract_setup_and_teardown(class_node)
        if setup_node and fixtures_used:
            for fixture in fixtures_used:
                fixture_code = create_fixture_for_variable(setup_node, fixture)
                all_fixtures.append(fixture_code)
    
    # Add fixtures to content
    if all_fixtures:
        new_content = add_fixtures_to_content(new_content, all_fixtures)
    
    # Convert assertions
    new_content = convert_assertions(new_content)

    # Fix mock patch paths
    new_content = fix_mock_patches(new_content)

    # Remove unittest.main() if present
    new_content = re.sub(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*unittest\.main\(\)", "", new_content)

    # Clean up any lingering unittest references
    new_content = re.sub(r"unittest\.\w+", "", new_content)
    
    # Add blank line at the end of the file if not already there
    if not new_content.endswith("\n"):
        new_content += "\n"
        
    # Write the modified content back to the file
    if not dry_run:
        with open(file_path, "w") as f:
            f.write(new_content)
        print(f"Converted {file_path}")
    else:
        print(f"Would convert {file_path} (dry run)")
        
    return new_content


def add_fixtures_to_content(content: str, fixtures: List[str]) -> str:
    """Add fixtures to content after imports."""
    # Find the position after imports
    lines = content.split("\n")
    insert_pos = 0
    
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_pos = i + 1
        elif line.strip() and not line.startswith("#") and insert_pos > 0:
            break
    
    # Insert fixtures after imports
    fixtures_text = "\n" + "\n".join(fixtures)
    lines.insert(insert_pos, fixtures_text)
    
    return "\n".join(lines)


def main():
    """Main function to run the conversion."""
    dry_run = "--dry-run" in sys.argv

    # Remove --dry-run from arguments to avoid treating it as a file path
    args = [arg for arg in sys.argv[1:] if arg != "--dry-run"]

    if args:
        file_path = args[0]
        convert_file(file_path, dry_run)
    else:
        # Find all test files
        base_dir = os.getcwd()
        test_files = find_test_files(base_dir)

        if not test_files:
            print(f"No unittest files found in {base_dir}")
            return

        print(f"Found {len(test_files)} unittest files:")
        for file in test_files:
            print(f"  {file}")

        for file in test_files:
            convert_file(file, dry_run)


if __name__ == "__main__":
    main()