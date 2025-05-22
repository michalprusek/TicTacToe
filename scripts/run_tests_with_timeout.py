#!/usr/bin/env python
"""
Script to run tests with a timeout.
"""
import os
import sys
import subprocess
import time
import signal
import argparse


def run_command_with_timeout(cmd, timeout_sec):
    """Run a command with a timeout."""
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
        end_time = time.time()
        return {
            'stdout': stdout.decode('utf-8'),
            'stderr': stderr.decode('utf-8'),
            'returncode': process.returncode,
            'timeout': False,
            'execution_time': end_time - start_time
        }
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        process.kill()
        try:
            process.wait(timeout=5)  # Give it 5 seconds to die
        except subprocess.TimeoutExpired:
            # If it's still not dead, try harder
            os.kill(process.pid, signal.SIGKILL)
        
        end_time = time.time()
        return {
            'stdout': '',
            'stderr': 'Command timed out after {} seconds'.format(timeout_sec),
            'returncode': -1,
            'timeout': True,
            'execution_time': end_time - start_time
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run tests with a timeout')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout in seconds')
    parser.add_argument('--test-file', type=str, help='Test file to run')
    parser.add_argument('--test-class', type=str, help='Test class to run')
    parser.add_argument('--test-method', type=str, help='Test method to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Build the command
    cmd = 'conda activate TicTacToe && python -m pytest'
    
    if args.test_file:
        cmd += ' ' + args.test_file
        if args.test_class:
            cmd += '::' + args.test_class
            if args.test_method:
                cmd += '::' + args.test_method
    
    if args.verbose:
        cmd += ' -v'
    
    print(f'Running command: {cmd}')
    print(f'Timeout: {args.timeout} seconds')
    
    # Run the command with a timeout
    result = run_command_with_timeout(cmd, args.timeout)
    
    # Print the result
    if result['timeout']:
        print(f'Command timed out after {args.timeout} seconds')
    else:
        print(f'Command completed in {result["execution_time"]:.2f} seconds')
        print(f'Return code: {result["returncode"]}')
        
    print('Stdout:')
    print(result['stdout'])
    
    print('Stderr:')
    print(result['stderr'])
    
    # Return the return code
    return result['returncode'] if not result['timeout'] else 1


if __name__ == '__main__':
    sys.exit(main())
