#!/usr/bin/env python
"""
Analyze operations across different backends to identify gaps and inconsistencies.

This script examines the implementation files for each backend (NumPy, PyTorch, MLX)
and generates a report of operations that are missing in one or more backends.
It also helps with migrating operations from ops/torch, ops/mlx, and ops/numpy
to the backend directory, as per the architectural rules.
"""

import os
import re
import ast
import argparse
import shutil
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd

# Paths to backend implementation files and folders
BACKEND_PATHS = {
    "numpy": ["ember_ml/backend/numpy"],
    "torch": ["ember_ml/backend/torch"],
    "mlx": ["ember_ml/backend/mlx"]
}

# Paths to ops implementation directories (legacy, may not exist in new structure)
# These are no longer used in the new structure, but kept for backward compatibility
OPS_PATHS = {
    "numpy": "ember_ml/backend/numpy",
    "torch": "ember_ml/backend/torch",
    "mlx": "ember_ml/backend/mlx"
}

# Paths to frontend abstraction files
FRONTEND_PATHS = {
    "tensor": "ember_ml/ops/interfaces/tensor_ops.py",
    "math": "ember_ml/ops/interfaces/math_ops.py",
    "random": "ember_ml/ops/interfaces/random_ops.py",
    "comparison": "ember_ml/ops/interfaces/comparison_ops.py",
    "dtype": "ember_ml/ops/interfaces/dtype_ops.py",
    "device": "ember_ml/ops/interfaces/device_ops.py",
    "feature": "ember_ml/ops/interfaces/feature_ops.py",
    "io": "ember_ml/ops/interfaces/io_ops.py"
}

def extract_operations_from_file(file_path: str) -> Set[str]:
    """Extract operation names from a backend implementation file."""
    operations = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if not node.name.startswith('_') or node.name.startswith('__') and node.name.endswith('__'):
                    operations.add(node.name)
                    
            # Also look for class methods
            elif isinstance(node, ast.ClassDef):
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.FunctionDef):
                        # Skip private methods
                        if not subnode.name.startswith('_') or subnode.name.startswith('__') and subnode.name.endswith('__'):
                            operations.add(f"{node.name}.{subnode.name}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return operations

def extract_operations_from_directory(directory: str) -> Dict[str, Set[str]]:
    """Extract operation names from all Python files in a directory."""
    operations_by_file = {}
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    operations = extract_operations_from_file(file_path)
                    if operations:
                        operations_by_file[file_path] = operations
    
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")
    
    return operations_by_file

def analyze_backends(backend_paths: Optional[Dict[str, List[str]]] = None,
                    ops_paths: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Set[str]]:
    """Analyze all backends and return operations by backend and the union of all operations."""
    if backend_paths is None:
        backend_paths = BACKEND_PATHS
    
    if ops_paths is None:
        ops_paths = OPS_PATHS
    
    operations_by_backend = {}
    all_operations = set()
    
    # Extract operations from backend directories
    for backend, paths in backend_paths.items():
        operations = set()
        
        # Check if the directory exists
        if isinstance(paths, list) and len(paths) > 0 and os.path.exists(paths[0]):
            dir_operations = set()
            for file_path, file_operations in extract_operations_from_directory(paths[0]).items():
                dir_operations.update(file_operations)
            operations.update(dir_operations)
        
        if operations:
            operations_by_backend[backend] = operations
            all_operations.update(operations)
    
    # Extract operations from ops directories
    ops_operations_by_backend = {}
    for backend, path in ops_paths.items():
        if os.path.exists(path):
            ops_operations = set()
            for file_path, operations in extract_operations_from_directory(path).items():
                ops_operations.update(operations)
            ops_operations_by_backend[backend] = ops_operations
            all_operations.update(ops_operations)
    
    return operations_by_backend, ops_operations_by_backend, all_operations

def generate_operation_matrix(operations_by_backend: Dict[str, Set[str]], all_operations: Set[str]) -> pd.DataFrame:
    """Generate a matrix showing which operations are available in which backends."""
    data = []
    
    for operation in sorted(all_operations):
        row = {"Operation": operation}
        
        for backend in operations_by_backend.keys():
            row[backend] = operation in operations_by_backend[backend]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def identify_missing_operations(operations_by_backend: Dict[str, Set[str]], all_operations: Set[str]) -> Dict[str, Set[str]]:
    """Identify operations that are missing in each backend."""
    missing_operations = {}
    
    for backend, operations in operations_by_backend.items():
        missing = all_operations - operations
        missing_operations[backend] = missing
    
    return missing_operations

def identify_common_operations(operations_by_backend: Dict[str, Set[str]]) -> Set[str]:
    """Identify operations that are common to all backends."""
    if not operations_by_backend:
        return set()
    
    # Start with all operations from the first backend
    common_operations = set(next(iter(operations_by_backend.values())))
    
    # Intersect with operations from all other backends
    for operations in operations_by_backend.values():
        common_operations &= operations
    
    return common_operations

def identify_operations_to_migrate(backend_operations: Dict[str, Set[str]], 
                                  ops_operations: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Identify operations that need to be migrated from ops to backend."""
    operations_to_migrate = {}
    
    for backend in ops_operations:
        if backend in backend_operations:
            # Operations in ops but not in backend need to be migrated
            to_migrate = ops_operations[backend] - backend_operations[backend]
            operations_to_migrate[backend] = to_migrate
    
    return operations_to_migrate

def extract_function_code(file_path: str, function_name: str) -> str:
    """Extract the code for a specific function from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the source code for the function
                start_line = node.lineno
                end_line = node.end_lineno
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                function_code = ''.join(lines[start_line-1:end_line])
                return function_code
            
            # Check for class methods
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.FunctionDef) and f"{class_name}.{subnode.name}" == function_name:
                        # Get the source code for the method
                        start_line = subnode.lineno
                        end_line = subnode.end_lineno
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        method_code = ''.join(lines[start_line-1:end_line])
                        return method_code
    
    except Exception as e:
        print(f"Error extracting function {function_name} from {file_path}: {e}")
    
    return ""

def find_function_in_directory(directory: str, function_name: str) -> str:
    """Find a function in a directory and return its file path."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                operations = extract_operations_from_file(file_path)
                if function_name in operations:
                    return file_path
    
    return ""

def generate_migration_plan(operations_to_migrate: Dict[str, Set[str]],
                           ops_paths: Dict[str, str],
                           backend_paths: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    """Generate a plan for migrating operations from ops to backend."""
    migration_plan = {}
    
    for backend, operations in operations_to_migrate.items():
        backend_plan = []
        
        for operation in operations:
            # Find the file containing the operation
            ops_dir = ops_paths[backend]
            source_file = find_function_in_directory(ops_dir, operation)
            
            if source_file:
                # Extract the function code
                function_code = extract_function_code(source_file, operation)
                
                if function_code:
                    # Determine the operation type (math, tensor, random, etc.)
                    operation_type = "other"
                    if "math" in source_file:
                        operation_type = "math"
                    elif "tensor" in source_file:
                        operation_type = "tensor"
                    elif "random" in source_file:
                        operation_type = "random"
                    elif "comparison" in source_file:
                        operation_type = "comparison"
                    elif "dtype" in source_file:
                        operation_type = "dtype"
                    elif "device" in source_file:
                        operation_type = "device"
                    elif "solver" in source_file:
                        operation_type = "solver"
                    
                    # Determine target file based on operation type
                    target_file = ""
                    if isinstance(backend_paths[backend], list) and len(backend_paths[backend]) > 0:
                        # Use the directory structure
                        target_dir = backend_paths[backend][0]
                        if operation_type == "math":
                            target_file = os.path.join(target_dir, "math_ops.py")
                        elif operation_type == "tensor":
                            target_file = os.path.join(target_dir, "tensor_ops.py")
                        elif operation_type == "random":
                            target_file = os.path.join(target_dir, "random_ops.py")
                        elif operation_type == "comparison":
                            target_file = os.path.join(target_dir, "comparison_ops.py")
                        elif operation_type == "dtype":
                            target_file = os.path.join(target_dir, "dtype_ops.py")
                        elif operation_type == "device":
                            target_file = os.path.join(target_dir, "device_ops.py")

                        else:
                            target_file = os.path.join(target_dir, "other_ops.py")
                    
                    # Add to migration plan
                    backend_plan.append({
                        "operation": operation,
                        "source_file": source_file,
                        "target_file": target_file,
                        "function_code": function_code,
                        "operation_type": operation_type
                    })
        
        migration_plan[backend] = backend_plan
    
    return migration_plan

def generate_report(operations_by_backend: Dict[str, Set[str]], 
                   ops_operations_by_backend: Dict[str, Set[str]],
                   all_operations: Set[str], 
                   output_dir: str = ".", 
                   verbose: bool = False) -> Dict[str, Any]:
    """Generate a report of operations across backends."""
    # Generate operation matrix
    matrix = generate_operation_matrix(operations_by_backend, all_operations)
    
    # Identify missing operations
    missing_operations = identify_missing_operations(operations_by_backend, all_operations)
    
    # Identify common operations
    common_operations = identify_common_operations(operations_by_backend)
    
    # Identify operations to migrate
    operations_to_migrate = identify_operations_to_migrate(operations_by_backend, ops_operations_by_backend)
    
    # Generate migration plan
    migration_plan = generate_migration_plan(operations_to_migrate, OPS_PATHS, BACKEND_PATHS)
    
    # Create summary
    summary = {
        "total_operations": len(all_operations),
        "common_operations": len(common_operations),
        "operations_by_backend": {backend: len(operations) for backend, operations in operations_by_backend.items()},
        "missing_operations": {backend: len(missing) for backend, missing in missing_operations.items()},
        "operations_to_migrate": {backend: len(ops) for backend, ops in operations_to_migrate.items()},
        "missing_details": missing_operations,
        "migration_plan": migration_plan,
        "matrix": matrix
    }
    
    # Print summary if verbose
    if verbose:
        print(f"Total unique operations: {len(all_operations)}")
        print(f"Operations common to all backends: {len(common_operations)}")
        
        for backend, operations in operations_by_backend.items():
            print(f"Operations in {backend} backend: {len(operations)}")
            print(f"Operations in {backend} ops: {len(ops_operations_by_backend.get(backend, set()))}")
            print(f"Operations to migrate from {backend} ops to backend: {len(operations_to_migrate.get(backend, set()))}")
            print(f"Operations missing in {backend}: {len(missing_operations[backend])}")
        
        # Print operations to migrate for each backend
        print("\nOperations to Migrate:")
        for backend, operations in operations_to_migrate.items():
            if operations:
                print(f"\n{backend} operations to migrate ({len(operations)}):")
                for op in sorted(operations):
                    print(f"  - {op}")
    
    # Save matrix to CSV
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "backend_operations_matrix.csv")
    matrix.to_csv(matrix_path, index=False)
    if verbose:
        print(f"\nOperation matrix saved to {matrix_path}")
    
    # Save missing operations to CSV
    missing_data = []
    for backend, missing in missing_operations.items():
        for op in missing:
            available_in = [b for b, ops in operations_by_backend.items() if op in ops]
            missing_data.append({
                "Backend": backend,
                "Missing Operation": op,
                "Available In": ", ".join(available_in)
            })
    
    missing_df = pd.DataFrame(missing_data)
    missing_path = os.path.join(output_dir, "missing_operations.csv")
    missing_df.to_csv(missing_path, index=False)
    if verbose:
        print(f"Missing operations saved to {missing_path}")
    
    # Save migration plan to CSV
    migration_data = []
    for backend, plan in migration_plan.items():
        for item in plan:
            migration_data.append({
                "Backend": backend,
                "Operation": item["operation"],
                "Source File": item["source_file"],
                "Target File": item["target_file"],
                "Operation Type": item["operation_type"]
            })
    
    migration_df = pd.DataFrame(migration_data)
    migration_path = os.path.join(output_dir, "migration_plan.csv")
    migration_df.to_csv(migration_path, index=False)
    if verbose:
        print(f"Migration plan saved to {migration_path}")
    
    return summary

def check_backend_consistency(file_path: str, operations_by_backend: Dict[str, Set[str]],
                               all_operations: Set[str]) -> Dict[str, Any]:
    """Check if a file has operations that are inconsistent across backends."""
    # Extract operations from the file
    file_operations = extract_operations_from_file(file_path)
    
    # Check if each operation is available in all backends
    inconsistent_operations = []
    for op in file_operations:
        # Check if this is a class method (contains a dot)
        if '.' in op:
            # Extract the class name and method name
            class_name, method_name = op.split('.', 1)
            
            # For class methods, check if the method exists in any class in each backend
            available_in = []
            for backend, ops in operations_by_backend.items():
                # Check if any operation in this backend ends with the method name
                method_exists = any(o.endswith('.' + method_name) for o in ops)
                if method_exists:
                    available_in.append(backend)
            
            if len(available_in) < len(operations_by_backend):
                missing_in = [backend for backend in operations_by_backend.keys() if backend not in available_in]
                inconsistent_operations.append({
                    "operation": op,  # Keep the full operation name for reference
                    "available_in": available_in,
                    "missing_in": missing_in
                })
        else:
            # For non-class methods, check exact match
            available_in = [backend for backend, ops in operations_by_backend.items() if op in ops]
            if len(available_in) < len(operations_by_backend):
                missing_in = [backend for backend in operations_by_backend.keys() if backend not in available_in]
                inconsistent_operations.append({
                    "operation": op,
                    "available_in": available_in,
                    "missing_in": missing_in
                })
    
    return {
        "file": file_path,
        "has_inconsistent_operations": len(inconsistent_operations) > 0,
        "inconsistent_operations": inconsistent_operations
    }

def migrate_operation(operation: str, source_file: str, target_file: str, function_code: str,
                     dry_run: bool = False) -> bool:
    """Migrate an operation from ops to backend."""
    try:
        # Extract the function name (without class prefix)
        function_name = operation.split('.')[-1]
        
        # Check if target file exists
        if not os.path.exists(os.path.dirname(target_file)):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            print(f"Created directory {os.path.dirname(target_file)}")
            
        # Create the file if it doesn't exist
        if not os.path.exists(target_file):
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\n{os.path.basename(target_file)} for {backend} backend.\n"""\n\n')
            print(f"Created file {target_file}")
        
        # Check if the operation already exists in the target file
        with open(target_file, 'r', encoding='utf-8') as f:
            target_content = f.read()
        
        # Check if the function already exists in the target file
        function_pattern = rf'def\s+{function_name}\s*\('
        if re.search(function_pattern, target_content):
            print(f"Function {function_name} already exists in {target_file}")
            return False
        
        if dry_run:
            print(f"Would migrate {operation} from {source_file} to {target_file}")
            print(f"Function code:\n{function_code}")
            return True
        
        # Append the function to the target file
        with open(target_file, 'a', encoding='utf-8') as f:
            f.write("\n\n")
            f.write(function_code)
        
        print(f"Migrated {operation} from {source_file} to {target_file}")
        return True
    
    except Exception as e:
        print(f"Error migrating {operation} from {source_file} to {target_file}: {e}")
        return False

def migrate_operations(backend: str, operations: List[Dict[str, str]], dry_run: bool = False) -> int:
    """Migrate a list of operations from ops to backend."""
    success_count = 0
    
    for op_info in operations:
        operation = op_info["operation"]
        source_file = op_info["source_file"]
        target_file = op_info["target_file"]
        function_code = op_info["function_code"]
        
        if migrate_operation(operation, source_file, target_file, function_code, dry_run):
            success_count += 1
    
    return success_count

def update_frontend_abstraction(operation: str, operation_type: str, dry_run: bool = False) -> bool:
    """Update a frontend abstraction to dispatch to the backend implementation."""
    try:
        # Determine the frontend file based on operation type
        frontend_file = FRONTEND_PATHS.get(operation_type)
        if not frontend_file or not os.path.exists(frontend_file):
            print(f"Frontend file for {operation_type} not found")
            return False
        
        # Extract the function name (without class prefix)
        function_name = operation.split('.')[-1]
        
        # Check if the operation already exists in the frontend file
        with open(frontend_file, 'r', encoding='utf-8') as f:
            frontend_content = f.read()
        
        # Check if the function already exists in the frontend file
        function_pattern = rf'def\s+{function_name}\s*\('
        if not re.search(function_pattern, frontend_content):
            print(f"Function {function_name} not found in {frontend_file}")
            return False
        
        # Check if the function already dispatches to the backend
        dispatch_pattern = rf'get_backend\(\)\.{function_name}'
        if re.search(dispatch_pattern, frontend_content):
            print(f"Function {function_name} already dispatches to backend in {frontend_file}")
            return True
        
        if dry_run:
            print(f"Would update {function_name} in {frontend_file} to dispatch to backend")
            return True
        
        # This is a simplified approach - in a real implementation, you would need to
        # parse the AST, modify the function body, and rewrite the file
        print(f"Updating {function_name} in {frontend_file} to dispatch to backend would require AST manipulation")
        print(f"This is not implemented in this script")
        
        return False
    
    except Exception as e:
        print(f"Error updating frontend abstraction for {operation}: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze operations across different backends and help with migration.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")
    parser.add_argument("--output-dir", "-o", default=".", help="Directory to save output files")
    parser.add_argument("--check-file", help="Check a specific file for backend consistency")
    parser.add_argument("--migrate", action="store_true", help="Migrate operations from ops to backend")
    parser.add_argument("--backend", choices=["numpy", "torch", "mlx", "all"], default="all", 
                        help="Backend to migrate (default: all)")
    parser.add_argument("--operation", help="Specific operation to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    args = parser.parse_args()
    
    # Analyze backends
    backend_operations, ops_operations, all_operations = analyze_backends()
    
    if args.check_file:
        # Check a specific file for backend consistency
        result = check_backend_consistency(args.check_file, backend_operations, all_operations)
        if args.verbose:
            print(f"Checking {args.check_file} for backend consistency:")
            if result["has_inconsistent_operations"]:
                print(f"Found {len(result['inconsistent_operations'])} inconsistent operations:")
                for op in result["inconsistent_operations"]:
                    print(f"  - {op['operation']} (available in: {', '.join(op['available_in'])}, missing in: {', '.join(op['missing_in'])})")
            else:
                print("All operations are consistent across backends.")
    elif args.migrate:
        # Generate migration plan
        operations_to_migrate = identify_operations_to_migrate(backend_operations, ops_operations)
        migration_plan = generate_migration_plan(operations_to_migrate, OPS_PATHS, BACKEND_PATHS)
        
        # Determine which backends to migrate
        backends_to_migrate = list(migration_plan.keys()) if args.backend == "all" else [args.backend]
        
        # Migrate operations
        for backend in backends_to_migrate:
            if backend in migration_plan:
                operations = migration_plan[backend]
                
                # Filter by specific operation if provided
                if args.operation:
                    operations = [op for op in operations if op["operation"] == args.operation]
                
                if operations:
                    print(f"Migrating {len(operations)} operations for {backend} backend")
                    success_count = migrate_operations(backend, operations, args.dry_run)
                    print(f"Successfully migrated {success_count}/{len(operations)} operations")
                else:
                    print(f"No operations to migrate for {backend} backend")
            else:
                print(f"No migration plan for {backend} backend")
    else:
        # Generate a full report
        generate_report(backend_operations, ops_operations, all_operations, args.output_dir, args.verbose)

if __name__ == "__main__":
    main()