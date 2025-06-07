#!/usr/bin/env python
"""
Test runner script for Ember ML library.

This script runs the tests for the Ember ML library using pytest.
It provides options for running specific test modules, test classes,
or individual test methods.

Usage:
    python run_tests.py                # Run all tests
    python run_tests.py tensor         # Run all tensor tests
    python run_tests.py math           # Run all math tests
    python run_tests.py backend        # Run all backend tests
    python run_tests.py tensor.TestTensorCreation  # Run specific test class
    python run_tests.py tensor.TestTensorCreation.test_zeros  # Run specific test method
    python run_tests.py --report       # Run all tests and generate HTML report
    python run_tests.py --cov          # Run all tests with coverage
"""

import sys
import os
import pytest
import argparse

def main():
    """Run the tests."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Ember ML tests')
    parser.add_argument('pattern', nargs='?', default='test_*.py', 
                        help='Test pattern to run (default: all tests)')
    parser.add_argument('--report', action='store_true', 
                        help='Generate HTML report')
    parser.add_argument('--cov', action='store_true', 
                        help='Run with coverage')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Verbose output')
    
    # Parse arguments
    args = parser.parse_known_args()[0]
    
    # Get the test pattern
    pattern = args.pattern
    
    # Map shorthand names to test modules
    if pattern == 'tensor':
        pattern = 'test_ops_tensor.py'
    elif pattern == 'math':
        pattern = 'test_ops_math.py'
    elif pattern == 'backend':
        pattern = 'test_backend.py'
    elif pattern.startswith('tensor.'):
        # Convert tensor.TestClass to test_ops_tensor.py::TestClass
        pattern = 'test_ops_tensor.py::' + pattern[7:]
    elif pattern.startswith('math.'):
        # Convert math.TestClass to test_ops_math.py::TestClass
        pattern = 'test_ops_math.py::' + pattern[5:]
    elif pattern.startswith('backend.'):
        # Convert backend.TestClass to test_backend.py::TestClass
        pattern = 'test_backend.py::' + pattern[8:]
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Build pytest arguments
    pytest_args = ['-xvs' if args.verbose else '-xs', pattern]
    
    # Add HTML report if requested
    if args.report:
        pytest_args.append('--html=report.html')
        pytest_args.append('--self-contained-html')
    
    # Add coverage if requested
    if args.cov:
        pytest_args.append('--cov=ember_ml')
        pytest_args.append('--cov-report=term')
        pytest_args.append('--cov-report=html')
    
    # Run the tests
    return pytest.main(pytest_args)

if __name__ == '__main__':
    sys.exit(main())