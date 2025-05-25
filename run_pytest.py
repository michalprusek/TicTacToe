#!/usr/bin/env python3
"""Pure pytest runner."""

import subprocess
import argparse


def main():
    """Main pytest runner function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    pytest_tests = [
        'tests/test_detector_constants.py',
        'tests/test_game_logic_pytest.py',
        'tests/test_frame_utils_pytest.py',
        'tests/test_main_constants_pytest.py',
        'tests/test_config_helper_pytest.py',
        'tests/test_style_manager_pytest.py',
        'tests/test_utils_pure_pytest.py',
        'tests/test_simple_pytest.py',
        'tests/test_config_pytest.py'
    ]
    
    cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        cmd.append('-v')
    
    if args.coverage:
        cmd.extend(['--cov=app', '--cov-report=term-missing'])
    
    cmd.extend(pytest_tests)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())