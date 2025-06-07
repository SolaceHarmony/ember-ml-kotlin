#!/usr/bin/env python
"""
Detect NumPy usage, precision-reducing casts, and tensor conversions in Python files.

This script scans Python files to detect:
1. If NumPy is imported or used
2. Casts that reduce precision (e.g., float() casts)
3. Unnecessary conversions of tensors to NumPy

It can be used to ensure that backend-agnostic code remains pure and efficient.
"""

import os
import re
import ast
import argparse
from typing import List, Dict, Tuple, Set, Optional, Any

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_numpy_import(file_path: str) -> Tuple[bool, List[str]]:
    """Check if NumPy is imported in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for numpy imports
    numpy_imports = []
    
    # Regular expression patterns for different import styles
    patterns = [
        r'import\s+numpy\s+as\s+(\w+)',  # import numpy as np
        r'from\s+numpy\s+import\s+(.*)',  # from numpy import ...
        r'import\s+numpy\b',  # import numpy
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            if pattern == r'import\s+numpy\s+as\s+(\w+)':
                # For "import numpy as np" style, capture the alias
                numpy_imports.extend(matches)
            elif pattern == r'from\s+numpy\s+import\s+(.*)':
                # For "from numpy import ..." style, capture the imported names
                for match in matches:
                    imports = [name.strip() for name in match.split(',')]
                    numpy_imports.extend(imports)
            else:
                # For "import numpy" style, add "numpy" to the list
                numpy_imports.append("numpy")
    
    return bool(numpy_imports), numpy_imports

def check_numpy_usage(file_path: str, numpy_aliases: List[str]) -> Tuple[bool, List[str]]:
    """Check if NumPy is used in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for numpy usage
    numpy_usages = []
    
    for alias in numpy_aliases:
        # Pattern to match usage of the numpy alias
        pattern = r'\b' + re.escape(alias) + r'\.\w+'
        matches = re.findall(pattern, content)
        if matches:
            numpy_usages.extend(matches)
    
    return bool(numpy_usages), numpy_usages

class PrecisionReducingVisitor(ast.NodeVisitor):
    """AST visitor to find precision-reducing casts, tensor conversions, and Python operators."""
    
    def __init__(self):
        self.precision_reducing_casts = []
        self.tensor_conversions = []
        self.python_operators = []
        self.numpy_imports = set()
        self.numpy_aliases = set()
        self.current_function = None
        self.current_line = 0
    
    def visit_Import(self, node):
        """Visit import statements."""
        for name in node.names:
            if name.name == 'numpy':
                self.numpy_imports.add(f"import {name.name}")
                self.numpy_aliases.add(name.asname or name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module == 'numpy':
            for name in node.names:
                self.numpy_imports.add(f"from numpy import {name.name}")
                self.numpy_aliases.add(name.asname or name.name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_BinOp(self, node):
        """Visit binary operations to detect Python operators."""
        # Map AST operator types to their string representations
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.MatMult: '@',
        }
        
        # Check if this is a Python operator we want to detect
        op_type = type(node.op)
        if op_type in op_map:
            location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
            self.python_operators.append({
                'type': op_map[op_type],
                'location': location,
                'line': node.lineno
            })
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls to detect precision-reducing casts and tensor conversions."""
        # Check for float(), int() casts
        if isinstance(node.func, ast.Name) and node.func.id in ('float', 'int'):
            location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
            self.precision_reducing_casts.append({
                'type': node.func.id,
                'location': location,
                'line': node.lineno
            })
        
        # Check for tensor to numpy conversions
        if isinstance(node.func, ast.Attribute):
            # Check for tensor.numpy(), tensor.cpu().numpy(), etc.
            if node.func.attr == 'numpy':
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.tensor_conversions.append({
                    'type': 'tensor.numpy()',
                    'location': location,
                    'line': node.lineno
                })
            
            # Check for tensor.convert_to_tensor(tensor), tensor.asarray(tensor), etc.
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.numpy_aliases:
                if node.func.attr in ('array', 'asarray'):
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    self.tensor_conversions.append({
                        'type': f"{node.func.value.id}.{node.func.attr}",
                        'location': location,
                        'line': node.lineno
                    })
        
        self.generic_visit(node)

def check_ast_for_issues(file_path: str) -> Tuple[bool, List[str], List[str], List[Dict], List[Dict], List[Dict]]:
    """Use AST to check for NumPy imports, usage, precision-reducing casts, tensor conversions, and Python operators."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, [], [], [], [], []
    
    # Check for precision-reducing casts, tensor conversions, and Python operators
    visitor = PrecisionReducingVisitor()
    visitor.visit(tree)
    
    # Check for numpy usage
    numpy_usages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in visitor.numpy_aliases:
                numpy_usages.append(f"{node.value.id}.{node.attr}")
    
    return (
        bool(visitor.numpy_imports),
        list(visitor.numpy_imports),
        numpy_usages,
        visitor.precision_reducing_casts,
        visitor.tensor_conversions,
        visitor.python_operators
    )

def analyze_file(file_path: str) -> Dict:
    """Analyze a file for NumPy imports, usage, precision-reducing casts, tensor conversions, and Python operators."""
    # Check for NumPy imports using regex
    has_numpy_import, numpy_imports = check_numpy_import(file_path)
    
    # If NumPy is imported, check for usage
    has_numpy_usage = False
    numpy_usages = []
    if has_numpy_import:
        has_numpy_usage, numpy_usages = check_numpy_usage(file_path, numpy_imports)
    
    # Use AST for more accurate detection
    ast_has_numpy, ast_numpy_imports, ast_numpy_usages, precision_casts, tensor_conversions, python_operators = check_ast_for_issues(file_path)
    
    # Combine results
    has_numpy = has_numpy_import or ast_has_numpy
    all_imports = list(set(numpy_imports + ast_numpy_imports))
    all_usages = list(set(numpy_usages + ast_numpy_usages))
    
    return {
        "file": file_path,
        "has_numpy": has_numpy,
        "imports": all_imports,
        "usages": all_usages,
        "precision_casts": precision_casts,
        "tensor_conversions": tensor_conversions,
        "python_operators": python_operators
    }

def analyze_directory(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[Dict]:
    """Analyze all Python files in a directory for NumPy usage."""
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Find all Python files
    python_files = find_python_files(directory)
    
    # Filter out excluded directories
    filtered_files = []
    for file_path in python_files:
        exclude = False
        for exclude_dir in exclude_dirs:
            if exclude_dir in file_path:
                exclude = True
                break
        if not exclude:
            filtered_files.append(file_path)
    
    # Analyze each file
    results = []
    for file_path in filtered_files:
        result = analyze_file(file_path)
        results.append(result)
    
    return results

def print_results(results: List[Dict], verbose: bool = False, show_numpy: bool = True,
                 show_precision: bool = True, show_conversion: bool = True, show_operators: bool = True):
    """Print the analysis results."""
    numpy_files = [result for result in results if result["has_numpy"]]
    files_with_precision_casts = [result for result in results if result["precision_casts"]]
    files_with_tensor_conversions = [result for result in results if result["tensor_conversions"]]
    files_with_python_operators = [result for result in results if result["python_operators"]]
    
    print(f"Total files analyzed: {len(results)}")
    if len(results) > 0:
        if show_numpy:
            print(f"Files with NumPy: {len(numpy_files)} ({len(numpy_files)/len(results)*100:.2f}%)")
        if show_precision:
            print(f"Files with precision-reducing casts: {len(files_with_precision_casts)} ({len(files_with_precision_casts)/len(results)*100:.2f}%)")
        if show_conversion:
            print(f"Files with tensor conversions: {len(files_with_tensor_conversions)} ({len(files_with_tensor_conversions)/len(results)*100:.2f}%)")
        if show_operators:
            print(f"Files with Python operators: {len(files_with_python_operators)} ({len(files_with_python_operators)/len(results)*100:.2f}%)")
    else:
        if show_numpy:
            print("Files with NumPy: 0 (0.00%)")
        if show_precision:
            print("Files with precision-reducing casts: 0 (0.00%)")
        if show_conversion:
            print("Files with tensor conversions: 0 (0.00%)")
        if show_operators:
            print("Files with Python operators: 0 (0.00%)")
    
    if verbose:
        if show_numpy and numpy_files:
            print("\nFiles with NumPy:")
            for result in numpy_files:
                print(f"\n{result['file']}:")
                print(f"  Imports: {', '.join(result['imports'])}")
                print(f"  Usages: {', '.join(result['usages'])}")
        
        if show_precision and files_with_precision_casts:
            print("\nFiles with precision-reducing casts:")
            for result in files_with_precision_casts:
                print(f"\n{result['file']}:")
                for cast in result["precision_casts"]:
                    print(f"  {cast['type']} at {cast['location']}")
        
        if show_conversion and files_with_tensor_conversions:
            print("\nFiles with tensor conversions:")
            for result in files_with_tensor_conversions:
                print(f"\n{result['file']}:")
                for conv in result["tensor_conversions"]:
                    print(f"  {conv['type']} at {conv['location']}")
        
        if show_operators and files_with_python_operators:
            print("\nFiles with Python operators:")
            for result in files_with_python_operators:
                print(f"\n{result['file']}:")
                for op in result["python_operators"]:
                    print(f"  {op['type']} operator at {op['location']}")
    
    # Print summary by directory
    print("\nSummary by directory:")
    dir_summary = {}
    for result in results:
        dir_path = os.path.dirname(result["file"])
        if dir_path not in dir_summary:
            dir_summary[dir_path] = {
                "total": 0,
                "numpy": 0,
                "precision_casts": 0,
                "tensor_conversions": 0,
                "python_operators": 0
            }
        dir_summary[dir_path]["total"] += 1
        if result["has_numpy"]:
            dir_summary[dir_path]["numpy"] += 1
        if result["precision_casts"]:
            dir_summary[dir_path]["precision_casts"] += 1
        if result["tensor_conversions"]:
            dir_summary[dir_path]["tensor_conversions"] += 1
        if result["python_operators"]:
            dir_summary[dir_path]["python_operators"] += 1
    
    for dir_path, stats in sorted(dir_summary.items()):
        # Check if there are any issues to show for this directory
        has_issues = (
            (show_numpy and stats["numpy"] > 0) or
            (show_precision and stats["precision_casts"] > 0) or
            (show_conversion and stats["tensor_conversions"] > 0) or
            (show_operators and stats["python_operators"] > 0)
        )
        
        if has_issues:
            print(f"{dir_path}:")
            if show_numpy and stats["numpy"] > 0:
                print(f"  NumPy: {stats['numpy']}/{stats['total']} files ({stats['numpy']/stats['total']*100:.2f}%)")
            if show_precision and stats["precision_casts"] > 0:
                print(f"  Precision casts: {stats['precision_casts']}/{stats['total']} files ({stats['precision_casts']/stats['total']*100:.2f}%)")
            if show_conversion and stats["tensor_conversions"] > 0:
                print(f"  Tensor conversions: {stats['tensor_conversions']}/{stats['total']} files ({stats['tensor_conversions']/stats['total']*100:.2f}%)")
            if show_operators and stats["python_operators"] > 0:
                print(f"  Python operators: {stats['python_operators']}/{stats['total']} files ({stats['python_operators']/stats['total']*100:.2f}%)")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Detect NumPy usage, precision-reducing casts, tensor conversions, and Python operators in Python files.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--exclude", nargs="+", help="Directories to exclude", default=[])
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")
    parser.add_argument("--numpy-only", action="store_true", help="Only check for NumPy usage")
    parser.add_argument("--precision-only", action="store_true", help="Only check for precision-reducing casts")
    parser.add_argument("--conversion-only", action="store_true", help="Only check for tensor conversions")
    parser.add_argument("--operators-only", action="store_true", help="Only check for Python operators (+, -, *, /, etc.)")
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Analyze a single file
        result = analyze_file(args.path)
        results = [result]
    else:
        # Analyze a directory
        results = analyze_directory(args.path, args.exclude)
    
    # Determine what to display based on flags
    show_numpy = not (args.precision_only or args.conversion_only or args.operators_only)
    show_precision = not (args.numpy_only or args.conversion_only or args.operators_only)
    show_conversion = not (args.numpy_only or args.precision_only or args.operators_only)
    show_operators = not (args.numpy_only or args.precision_only or args.conversion_only)
    
    if args.numpy_only:
        show_numpy = True
    elif args.precision_only:
        show_precision = True
    elif args.conversion_only:
        show_conversion = True
    elif args.operators_only:
        show_operators = True
    
    # If no specific flag is set, show everything
    if not (args.numpy_only or args.precision_only or args.conversion_only or args.operators_only):
        show_numpy = show_precision = show_conversion = show_operators = True
    
    print_results(results, args.verbose, show_numpy, show_precision, show_conversion, show_operators)
    
    # Return a boolean indicating if any issues were found
    numpy_files = [result for result in results if result["has_numpy"]]
    precision_files = [result for result in results if result["precision_casts"]]
    conversion_files = [result for result in results if result["tensor_conversions"]]
    operator_files = [result for result in results if result["python_operators"]]
    
    # Determine what to check based on flags
    if args.numpy_only:
        return bool(numpy_files)
    elif args.precision_only:
        return bool(precision_files)
    elif args.conversion_only:
        return bool(conversion_files)
    elif args.operators_only:
        return bool(operator_files)
    else:
        # Check for any issues
        return bool(numpy_files or precision_files or conversion_files or operator_files)

if __name__ == "__main__":
    main()