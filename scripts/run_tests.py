#!/usr/bin/env python
"""
Script to run all tests and measure code coverage.
"""
import sys
import subprocess
import argparse
import time


def run_tests(args):
    """Run tests with coverage."""
    # Create command
    cmd = ["python", "-m", "pytest"]

    # Add coverage options
    if args.coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Timeout is now specified in pytest.ini

    # Add specific tests
    if args.tests:
        cmd.extend(args.tests)
    else:
        cmd.append("tests/")

    # Run command with time tracking
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    print(f"Tests completed in {elapsed_time:.2f} seconds.")

    # Return exit code
    return result.returncode


def run_pylint(args):
    """Run pylint on the codebase."""
    # Create command
    cmd = ["pylint"]

    # Add modules to check
    if args.modules:
        cmd.extend(args.modules)
    else:
        cmd.extend(["app"])

    # Run command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Return exit code
    return result.returncode


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run tests and measure code coverage."
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Measure code coverage"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--pylint",
        action="store_true",
        help="Run pylint"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        help="Specific tests to run"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific modules to check with pylint"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Test timeout in seconds (default: 180)"
    )

    args = parser.parse_args()

    # Start time tracking for the whole process
    total_start_time = time.time()

    # Run tests
    test_result = run_tests(args)

    # Run pylint
    pylint_result = 0
    if args.pylint:
        pylint_result = run_pylint(args)

    # Calculate and print total time
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    # Return exit code
    return test_result or pylint_result


if __name__ == "__main__":
    sys.exit(main())
