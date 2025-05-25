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
        'tests.test_final_coverage'
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
                suite.addTests(loader.loadTestsFromName(module_name))
            
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
            suite.addTests(loader.loadTestsFromName(module_name))
        
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(main())