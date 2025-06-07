#!/usr/bin/env python
"""
Update EmberLint to improve backend consistency checking.

This script enhances the EmberLint tool with the following improvements:
1. Check for functions in monolithic backend files that haven't been moved to the folder structure
2. Check ops/__init__.py for proper exposure of functions and interfaces
3. Check for direct backend imports in ops files (e.g., "import backend as K")
4. Improved backend consistency checking across the new folder structure
"""

import os
import re
import ast
import sys
import importlib
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path

# Paths to backend implementation files and folders
BACKEND_PATHS = {
    "numpy": ["ember_ml/backend/numpy"],
    "torch": ["ember_ml/backend/torch"],
    "mlx": ["ember_ml/backend/mlx"]
}

# Paths to ops interfaces
OPS_INTERFACES_PATH = "ember_ml/ops/interfaces"

def extract_functions_from_file(file_path: str) -> Set[str]:
    """Extract function names from a Python file."""
    functions = set()
    
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
                    functions.add(node.name)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return functions

def extract_functions_from_directory(directory: str) -> Dict[str, Set[str]]:
    """Extract function names from all Python files in a directory."""
    functions_by_file = {}
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    functions = extract_functions_from_file(file_path)
                    if functions:
                        functions_by_file[file_path] = functions
    
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")
    
    return functions_by_file

def check_monolithic_backend_files() -> Dict[str, Dict[str, Any]]:
    """
    Check for functions in monolithic backend files that haven't been moved to the folder structure.
    
    Note: This function is now deprecated as the monolithic backend files no longer exist.
    It now simply returns an empty dictionary.
    
    Returns:
        Empty dictionary (monolithic files no longer exist)
    """
    # The monolithic backend files no longer exist, so we return an empty dictionary
    return {}

def check_ops_init_exposure(ops_init_path: str = "ember_ml/ops/__init__.py") -> Dict[str, Any]:
    """
    Check ops/__init__.py for proper exposure of functions and interfaces.
    
    Args:
        ops_init_path: Path to the ops/__init__.py file
        
    Returns:
        Dict containing:
            - 'exposed_interfaces': Set of exposed interfaces
            - 'exposed_functions': Set of exposed functions
            - 'missing_interfaces': Set of interfaces that should be exposed
            - 'missing_functions': Set of functions that should be exposed
    """
    # Extract interfaces from the interfaces directory
    interfaces_dir = OPS_INTERFACES_PATH
    interfaces = set()
    
    for file in os.listdir(interfaces_dir):
        if file.endswith('_ops.py'):
            interface_name = file[:-3]  # Remove .py extension
            # Convert snake_case to CamelCase and add "Ops" suffix
            interface_class = ''.join(word.capitalize() for word in interface_name.split('_'))
            interfaces.add(interface_class)
    
    # Extract exposed interfaces and functions from ops/__init__.py
    exposed_interfaces = set()
    exposed_functions = set()
    
    try:
        with open(ops_init_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'interfaces' in node.module:
                    for name in node.names:
                        exposed_interfaces.add(name.name)
            
            # Find all lambda functions (which are used to expose backend functions)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Lambda):
                            exposed_functions.add(target.id)
    
    except Exception as e:
        print(f"Error processing {ops_init_path}: {e}")
    
    # Find missing interfaces and functions
    missing_interfaces = interfaces - exposed_interfaces
    
    # For functions, we would need to know all functions that should be exposed
    # This is more complex and would require analyzing all backend implementations
    # For now, we'll just return the exposed functions
    
    return {
        'exposed_interfaces': exposed_interfaces,
        'exposed_functions': exposed_functions,
        'missing_interfaces': missing_interfaces,
        'missing_functions': set()  # Placeholder
    }

def check_direct_backend_imports(ops_dir: str = "ember_ml/ops") -> Dict[str, List[str]]:
    """
    Check for direct backend imports in ops files (e.g., "import backend as K").
    
    Args:
        ops_dir: Path to the ops directory
        
    Returns:
        Dict with file paths as keys and lists of direct backend import lines as values
    """
    direct_imports = {}
    
    # Regular expressions to match different forms of direct backend imports
    patterns = [
        r'from\s+ember_ml\s+import\s+backend\s+as\s+\w+',
        r'import\s+ember_ml\.backend\s+as\s+\w+',
        r'from\s+ember_ml\.backend\s+import\s+\w+'
    ]
    
    # Walk through the ops directory
    for root, _, files in os.walk(ops_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Skip the interfaces directory
                if 'interfaces' in file_path:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for direct backend imports
                    direct_import_lines = []
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        direct_import_lines.extend(matches)
                    
                    if direct_import_lines:
                        direct_imports[file_path] = direct_import_lines
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return direct_imports

def check_backend_consistency() -> Dict[str, Any]:
    """
    Check that functions in one backend are in the other two and vice versa.
    
    Returns:
        Dict containing:
            - 'functions_by_backend': Dict with backend names as keys and sets of functions as values
            - 'all_functions': Set of all functions across all backends
            - 'inconsistent_functions': Dict with function names as keys and dicts containing:
                - 'available_in': List of backends where the function is available
                - 'missing_in': List of backends where the function is missing
    """
    # Extract functions from all backends
    functions_by_backend = {}
    
    for backend, paths in BACKEND_PATHS.items():
        backend_functions = set()
        
        # Check folder structure
        folder_path = paths[0]
        if os.path.exists(folder_path):
            folder_functions_by_file = extract_functions_from_directory(folder_path)
            for file_functions in folder_functions_by_file.values():
                backend_functions.update(file_functions)
        
        functions_by_backend[backend] = backend_functions
    
    # Get the union of all functions
    all_functions = set()
    for functions in functions_by_backend.values():
        all_functions.update(functions)
    
    # Check for inconsistencies
    inconsistent_functions = {}
    
    for function in all_functions:
        available_in = []
        missing_in = []
        
        for backend, functions in functions_by_backend.items():
            if function in functions:
                available_in.append(backend)
            else:
                missing_in.append(backend)
        
        if missing_in:
            inconsistent_functions[function] = {
                'available_in': available_in,
                'missing_in': missing_in
            }
    
    return {
        'functions_by_backend': functions_by_backend,
        'all_functions': all_functions,
        'inconsistent_functions': inconsistent_functions
    }

def check_interface_implementation(interface_path: str, backend_paths: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Check that all interfaces have corresponding implementations in all backends.
    
    Args:
        interface_path: Path to the interfaces directory
        backend_paths: Dict with backend names as keys and lists of paths as values
        
    Returns:
        Dict containing:
            - 'interfaces': Dict with interface names as keys and sets of methods as values
            - 'implementations_by_backend': Dict with backend names as keys and dicts of implementations as values
            - 'missing_implementations': Dict with interface names as keys and dicts containing:
                - 'methods': Set of methods in the interface
                - 'missing_by_backend': Dict with backend names as keys and sets of missing methods as values
    """
    # Extract interfaces and their methods
    interfaces = {}
    
    for file in os.listdir(interface_path):
        if file.endswith('_ops.py'):
            interface_file = os.path.join(interface_path, file)
            
            try:
                with open(interface_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content)
                
                # Find all class definitions
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and 'Ops' in node.name:
                        interface_name = node.name
                        methods = set()
                        
                        # Find all method definitions
                        for subnode in ast.walk(node):
                            if isinstance(subnode, ast.FunctionDef):
                                # Skip private methods
                                if not subnode.name.startswith('_') or subnode.name.startswith('__') and subnode.name.endswith('__'):
                                    methods.add(subnode.name)
                        
                        interfaces[interface_name] = methods
            
            except Exception as e:
                print(f"Error processing {interface_file}: {e}")
    
    # Extract implementations from all backends
    implementations_by_backend = {}
    
    for backend, paths in backend_paths.items():
        implementations = {}
        
        # Check folder structure
        folder_path = paths[0]
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('_ops.py'):
                    implementation_file = os.path.join(folder_path, file)
                    
                    try:
                        with open(implementation_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Parse the file
                        tree = ast.parse(content)
                        
                        # Find all class definitions
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and 'Ops' in node.name:
                                implementation_name = node.name
                                methods = set()
                                
                                # Find all method definitions
                                for subnode in ast.walk(node):
                                    if isinstance(subnode, ast.FunctionDef):
                                        # Skip private methods
                                        if not subnode.name.startswith('_') or subnode.name.startswith('__') and subnode.name.endswith('__'):
                                            methods.add(subnode.name)
                                
                                implementations[implementation_name] = methods
                    
                    except Exception as e:
                        print(f"Error processing {implementation_file}: {e}")
        
        implementations_by_backend[backend] = implementations
    
    # Check for missing implementations
    missing_implementations = {}
    
    for interface_name, methods in interfaces.items():
        missing_by_backend = {}
        
        for backend, implementations in implementations_by_backend.items():
            # Find the corresponding implementation class
            implementation_name = None
            for name in implementations.keys():
                if interface_name.replace('Ops', '') in name:
                    implementation_name = name
                    break
            
            if implementation_name:
                # Check for missing methods
                implementation_methods = implementations[implementation_name]
                missing_methods = methods - implementation_methods
                
                if missing_methods:
                    missing_by_backend[backend] = missing_methods
            else:
                # The entire interface is missing
                missing_by_backend[backend] = methods
        
        if missing_by_backend:
            missing_implementations[interface_name] = {
                'methods': methods,
                'missing_by_backend': missing_by_backend
            }
    
    return {
        'interfaces': interfaces,
        'implementations_by_backend': implementations_by_backend,
        'missing_implementations': missing_implementations
    }

def main():
    """Main function."""
    # Monolithic backend files no longer exist, so we skip this check
    # print("Checking monolithic backend files...")
    # monolithic_results = check_monolithic_backend_files()
    
    # for backend, result in monolithic_results.items():
    #     print(f"\n{backend} backend:")
    #     print(f"  Functions in monolithic file: {len(result['monolithic_functions'])}")
    #     print(f"  Functions in folder structure: {len(result['folder_functions'])}")
    #     print(f"  Unmigrated functions: {len(result['unmigrated_functions'])}")
    
    # Check ops/__init__.py for proper exposure
    print("\nChecking ops/__init__.py for proper exposure...")
    ops_init_results = check_ops_init_exposure()
    
    print(f"  Exposed interfaces: {len(ops_init_results['exposed_interfaces'])}")
    print(f"  Exposed functions: {len(ops_init_results['exposed_functions'])}")
    
    if ops_init_results['missing_interfaces']:
        print(f"  Missing interfaces: {', '.join(sorted(ops_init_results['missing_interfaces']))}")
    
    # Check for direct backend imports
    print("\nChecking for direct backend imports...")
    direct_imports = check_direct_backend_imports()
    
    if direct_imports:
        print(f"  Found {len(direct_imports)} files with direct backend imports:")
        for file_path, imports in direct_imports.items():
            print(f"  {file_path}:")
            for import_line in imports:
                print(f"    {import_line}")
    else:
        print("  No direct backend imports found.")
    
    # Check backend consistency
    print("\nChecking backend consistency...")
    consistency_results = check_backend_consistency()
    
    print(f"  Total unique functions: {len(consistency_results['all_functions'])}")
    for backend, functions in consistency_results['functions_by_backend'].items():
        print(f"  Functions in {backend} backend: {len(functions)}")
    
    inconsistent_count = len(consistency_results['inconsistent_functions'])
    print(f"  Inconsistent functions: {inconsistent_count}")
    
    if inconsistent_count > 0:
        print("\nTop 10 inconsistent functions:")
        for i, (function, details) in enumerate(sorted(consistency_results['inconsistent_functions'].items())):
            if i >= 10:
                break
            print(f"  {function} (available in: {', '.join(details['available_in'])}, missing in: {', '.join(details['missing_in'])})")
    
    # Check interface implementation
    print("\nChecking interface implementation...")
    interface_results = check_interface_implementation(OPS_INTERFACES_PATH, BACKEND_PATHS)
    
    print(f"  Total interfaces: {len(interface_results['interfaces'])}")
    for backend, implementations in interface_results['implementations_by_backend'].items():
        print(f"  Implementations in {backend} backend: {len(implementations)}")
    
    missing_count = len(interface_results['missing_implementations'])
    print(f"  Interfaces with missing implementations: {missing_count}")
    
    if missing_count > 0:
        print("\nInterfaces with missing implementations:")
        for interface, details in interface_results['missing_implementations'].items():
            print(f"  {interface}:")
            for backend, missing_methods in details['missing_by_backend'].items():
                print(f"    {backend} backend missing {len(missing_methods)} methods")
                if len(missing_methods) <= 5:
                    print(f"      Missing methods: {', '.join(sorted(missing_methods))}")

if __name__ == "__main__":
    main()