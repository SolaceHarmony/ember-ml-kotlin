#!/usr/bin/env python
"""
Test backend consistency across different backends.

This script checks if operations are consistently implemented across all backends
in the new folder structure (ember_ml/backend/numpy, ember_ml/backend/torch, ember_ml/backend/mlx).
"""

import os
import ast
import argparse
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any

# Paths to backend implementation directories
BACKEND_DIRS = {
    "numpy": "../ember_ml/backend/numpy",
    "torch": "../ember_ml/backend/torch",
    "mlx": "../ember_ml/backend/mlx"
}

# Operation types and their corresponding files
OPERATION_TYPES = {
    "tensor": "tensor_ops.py",
    "math": "math_ops.py",
    "random": "random_ops.py",
    "comparison": "comparison_ops.py",
    "dtype": "dtype_ops.py",
    "device": "device_ops.py",
    "io": "io_ops.py"
}

def extract_operations_from_file(file_path: str) -> Set[str]:
    """Extract operation names from a backend implementation file."""
    operations = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all function definitions at the module level
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if not node.name.startswith('_') or node.name.startswith('__') and node.name.endswith('__'):
                    operations.add(node.name)
                    
            # Also look for class methods
            elif isinstance(node, ast.ClassDef):
                # Get the class name
                class_name = node.name
                
                # Find all methods in the class
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        # Skip private methods and __init__
                        if (not subnode.name.startswith('_') or
                            (subnode.name.startswith('__') and subnode.name.endswith('__') and
                             subnode.name != '__init__')):
                            # Add the method name without the class prefix
                            operations.add(subnode.name)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return operations

def extract_operations_by_type() -> Dict[str, Dict[str, Set[str]]]:
    """Extract operations by type for each backend."""
    operations_by_backend_and_type = {}
    
    for backend, backend_dir in BACKEND_DIRS.items():
        operations_by_type = {}
        
        for op_type, op_file in OPERATION_TYPES.items():
            file_path = os.path.join(backend_dir, op_file)
            print(f"Checking file: {file_path}, exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                operations = extract_operations_from_file(file_path)
                operations_by_type[op_type] = operations
                print(f"Found {len(operations)} operations in {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        operations_by_backend_and_type[backend] = operations_by_type
    
    return operations_by_backend_and_type

def check_consistency() -> Dict[str, Dict[str, List[str]]]:
    """Check consistency of operations across backends."""
    operations_by_backend_and_type = extract_operations_by_type()
    
    # For each operation type, find operations that are missing in some backends
    inconsistencies = {}
    
    for op_type in OPERATION_TYPES.keys():
        # Get all operations of this type across all backends
        all_operations = set()
        for backend, ops_by_type in operations_by_backend_and_type.items():
            if op_type in ops_by_type:
                all_operations.update(ops_by_type[op_type])
        
        # Check which operations are missing in each backend
        missing_by_backend = {}
        for backend, ops_by_type in operations_by_backend_and_type.items():
            if op_type in ops_by_type:
                backend_ops = ops_by_type[op_type]
                missing = all_operations - backend_ops
                if missing:
                    missing_by_backend[backend] = sorted(list(missing))
        
        if missing_by_backend:
            inconsistencies[op_type] = missing_by_backend
    
    return inconsistencies

def generate_report(inconsistencies: Dict[str, Dict[str, List[str]]], output_dir: str = ".") -> None:
    """Generate a report of inconsistencies."""
    # Create a DataFrame for the report
    rows = []
    
    for op_type, missing_by_backend in inconsistencies.items():
        for backend, missing_ops in missing_by_backend.items():
            for op in missing_ops:
                rows.append({
                    "Operation Type": op_type,
                    "Backend": backend,
                    "Missing Operation": op
                })
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "backend_inconsistencies.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved inconsistency report to {csv_path}")
        
        # Print summary
        print(f"\nFound {len(rows)} inconsistencies across {len(inconsistencies)} operation types")
        for op_type, missing_by_backend in inconsistencies.items():
            print(f"\n{op_type} operations:")
            for backend, missing_ops in missing_by_backend.items():
                print(f"  {backend} is missing {len(missing_ops)} operations: {', '.join(missing_ops[:5])}" + 
                      (f" and {len(missing_ops) - 5} more" if len(missing_ops) > 5 else ""))
    else:
        print("No inconsistencies found. All operations are implemented across all backends.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test backend consistency across different backends.")
    parser.add_argument("--output-dir", "-o", default=".", help="Directory to save output files")
    parser.add_argument("--debug", "-d", action="store_true", help="Print debug information")
    args = parser.parse_args()
    
    # Extract operations by type
    operations_by_backend_and_type = extract_operations_by_type()
    
    # Print debug information if requested
    if args.debug:
        print("\nOperations by backend and type:")
        for backend, ops_by_type in operations_by_backend_and_type.items():
            print(f"\n{backend} backend:")
            for op_type, ops in ops_by_type.items():
                print(f"  {op_type} operations: {sorted(list(ops))}")
    
    # Check consistency
    inconsistencies = check_consistency()
    
    # Generate report
    generate_report(inconsistencies, args.output_dir)

if __name__ == "__main__":
    main()
