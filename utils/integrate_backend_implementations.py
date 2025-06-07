#!/usr/bin/env python
"""
Integrate backend implementations into the backend files.

This script integrates the implementations from the ember_ml/backend/implementations
directory into the appropriate backend files.
"""

import os
import re
import argparse
import importlib.util
from typing import Dict, List, Set, Tuple

# Paths to backend files
BACKEND_FILES = {
    "numpy": "ember_ml/backend/numpy_backend.py",
    "torch": "ember_ml/backend/torch_backend.py",
    "mlx": "ember_ml/backend/mlx_backend.py"
}

# Path to ops module
OPS_MODULE_PATH = "ember_ml/ops/__init__.py"

# Path to implementations directory
IMPLEMENTATIONS_DIR = "ember_ml/backend/implementations"

def load_module_from_file(file_path: str, module_name: str = None):
    """Load a module from a file path."""
    if module_name is None:
        module_name = os.path.basename(file_path).split('.')[0]
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def get_function_names_from_module(module) -> Dict[str, str]:
    """Get function names from a module."""
    function_names = {}
    
    for name in dir(module):
        if name.startswith('__'):
            continue
        
        attr = getattr(module, name)
        if callable(attr):
            # Extract the backend name from the function name (e.g., numpy_power -> numpy)
            parts = name.split('_', 1)
            if len(parts) == 2 and parts[0] in BACKEND_FILES:
                backend = parts[0]
                operation = parts[1]
                function_names[name] = operation
    
    return function_names

def check_if_function_exists(file_path: str, function_name: str) -> bool:
    """Check if a function exists in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for function definition
    pattern = r'def\s+' + re.escape(function_name) + r'\s*\('
    return bool(re.search(pattern, content))

def add_function_to_file(file_path: str, function_name: str, function_code: str) -> bool:
    """Add a function to a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the function already exists
    if check_if_function_exists(file_path, function_name):
        print(f"Function {function_name} already exists in {file_path}")
        return False
    
    # Add the function to the end of the file
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write(function_code)
    
    print(f"Added function {function_name} to {file_path}")
    return True

def get_function_code(module, function_name: str) -> str:
    """Get the code for a function from a module."""
    import inspect
    
    function = getattr(module, function_name)
    return inspect.getsource(function)

def update_ops_module(operations: Set[str]) -> bool:
    """Update the ops module to expose the new operations."""
    with open(OPS_MODULE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the operations are already exposed
    exposed_operations = set()
    for operation in operations:
        pattern = r'def\s+' + re.escape(operation) + r'\s*\('
        if re.search(pattern, content):
            exposed_operations.add(operation)
    
    # Add the missing operations
    missing_operations = operations - exposed_operations
    if not missing_operations:
        print("All operations are already exposed in the ops module")
        return False
    
    # Add the operations to the ops module
    with open(OPS_MODULE_PATH, 'a', encoding='utf-8') as f:
        for operation in sorted(missing_operations):
            f.write(f'\n\ndef {operation}(*args, **kwargs):\n')
            f.write(f'    """Wrapper for backend-specific {operation} implementation."""\n')
            f.write(f'    from ember_ml.backend import get_backend_module\n')
            f.write(f'    return get_backend_module().{operation}(*args, **kwargs)\n')
    
    print(f"Added {len(missing_operations)} operations to the ops module")
    return True

def integrate_implementations() -> bool:
    """Integrate implementations into the backend files."""
    # Get all implementation files
    implementation_files = []
    for file_name in os.listdir(IMPLEMENTATIONS_DIR):
        if file_name.endswith('.py'):
            implementation_files.append(os.path.join(IMPLEMENTATIONS_DIR, file_name))
    
    if not implementation_files:
        print("No implementation files found")
        return False
    
    # Process each implementation file
    all_operations = set()
    for file_path in implementation_files:
        module = load_module_from_file(file_path)
        function_names = get_function_names_from_module(module)
        
        for function_name, operation in function_names.items():
            all_operations.add(operation)
            
            # Extract the backend name from the function name
            backend = function_name.split('_', 1)[0]
            backend_file = BACKEND_FILES.get(backend)
            
            if backend_file:
                function_code = get_function_code(module, function_name)
                
                # Rename the function to remove the backend prefix
                function_code = function_code.replace(f'def {function_name}', f'def {operation}')
                
                # Add the function to the backend file
                add_function_to_file(backend_file, operation, function_code)
    
    # Update the ops module
    update_ops_module(all_operations)
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integrate backend implementations.")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run mode - no files will be modified")
    else:
        integrate_implementations()

if __name__ == "__main__":
    main()