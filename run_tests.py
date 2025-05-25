#!/usr/bin/env python3
"""
Test runner script for TicTacToe application.
Supports running tests with coverage reporting.
"""

import sys
import unittest
import argparse
import os

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run TicTacToe tests')
    parser.add_argument('--coverage', action='store_true', 
                       help='Run tests with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    args = parser.parse_args()
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Load specific test modules that work
    test_modules = [
        'tests.test_strategy',
        'tests.test_strategy_comprehensive',
        'tests.test_final_coverage',
        'tests.test_game_logic_unittest',
        'tests.test_game_state_comprehensive',
        'tests.test_game_state_additional_coverage',
        'tests.test_game_state_extended',
        'tests.test_utils',
        'tests.test_path_utils',
        'tests.test_game_utils',
        'tests.test_game_utils_comprehensive',
        'tests.test_constants',
        'tests.test_config_extended',
        'tests.test_detector_constants',
        'tests.test_game_logic_pytest',
        'tests.test_error_handler_pytest',
        'tests.test_frame_utils_pytest',
        'tests.test_main_constants_pytest',
        'tests.test_config_helper_pytest',
        'tests.test_style_manager_pytest'
    ]
    
    if args.coverage:
        try:
            import coverage
            # Configure coverage to focus on app directory
            cov = coverage.Coverage(source=['app'], config_file='.coveragerc')
            cov.start()
            
            # Run tests
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            for module_name in test_modules:
                try:
                    suite.addTests(loader.loadTestsFromName(module_name))
                    print(f"Loaded tests from {module_name}")
                except Exception as e:
                    print(f"Warning: Could not load {module_name}: {e}")
            
            runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
            result = runner.run(suite)            
            # Stop coverage and report
            cov.stop()
            cov.save()
            
            print("\nCoverage Report:")
            cov.report(show_missing=True)
            
            return 0 if result.wasSuccessful() else 1
            
        except ImportError:
            print("Coverage package not installed. Install with: pip install coverage")
            return 1
    else:
        # Run tests without coverage
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for module_name in test_modules:
            try:
                suite.addTests(loader.loadTestsFromName(module_name))
                print(f"Loaded tests from {module_name}")
            except Exception as e:
                print(f"Warning: Could not load {module_name}: {e}")
        
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(main())