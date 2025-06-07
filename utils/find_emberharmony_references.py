#!/usr/bin/env python3
"""
Find all occurrences of 'ember_ml' in the codebase and make a list of files to update.

This script searches for 'ember_ml' in all files in the codebase and outputs a list
of files that need to be updated to use 'ember_ml' instead.
"""

import os
import re
import argparse
from typing import List, Dict, Tuple, Set
import json

def find_references(directory: str, ignore_dirs: List[str] = None) -> Dict[str, List[Tuple[int, str]]]:
    """
    Find all occurrences of 'ember_ml' in the given directory.
    
    Args:
        directory: Directory to search in
        ignore_dirs: List of directories to ignore
        
    Returns:
        Dictionary mapping file paths to lists of (line_number, line_content) tuples
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.env']
    
    references = {}
    pattern = re.compile(r'ember_ml', re.IGNORECASE)
    
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip binary files and certain file types
            if os.path.splitext(file)[1] in ['.pyc', '.pyo', '.so', '.o', '.a', '.lib', '.dll', '.exe']:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                    file_references = []
                    for i, line in enumerate(lines):
                        if pattern.search(line):
                            file_references.append((i + 1, line.strip()))
                    
                    if file_references:
                        references[file_path] = file_references
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return references

def categorize_files(references: Dict[str, List[Tuple[int, str]]]) -> Dict[str, List[str]]:
    """
    Categorize files based on the type of reference.
    
    Args:
        references: Dictionary mapping file paths to lists of (line_number, line_content) tuples
        
    Returns:
        Dictionary mapping categories to lists of file paths
    """
    categories = {
        'code': [],      # Python code files with import statements or function calls
        'docstring': [], # Python files with docstrings containing ember_ml
        'comment': [],   # Python files with comments containing ember_ml
        'markdown': [],  # Markdown files
        'other': []      # Other file types
    }
    
    for file_path, refs in references.items():
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.py':
            has_import = any('import' in line and 'ember_ml' in line for _, line in refs)
            has_from = any('from' in line and 'ember_ml' in line for _, line in refs)
            has_docstring = any('"""' in line and 'ember_ml' in line for _, line in refs)
            has_comment = any('#' in line and 'ember_ml' in line for _, line in refs)
            
            if has_import or has_from:
                categories['code'].append(file_path)
            elif has_docstring:
                categories['docstring'].append(file_path)
            elif has_comment:
                categories['comment'].append(file_path)
            else:
                categories['code'].append(file_path)
        elif file_ext in ['.md', '.markdown']:
            categories['markdown'].append(file_path)
        else:
            categories['other'].append(file_path)
    
    return categories

def generate_report(references: Dict[str, List[Tuple[int, str]]], categories: Dict[str, List[str]], output_file: str = None):
    """
    Generate a report of files that need to be updated.
    
    Args:
        references: Dictionary mapping file paths to lists of (line_number, line_content) tuples
        categories: Dictionary mapping categories to lists of file paths
        output_file: Path to output file (if None, print to stdout)
    """
    report = {
        'summary': {
            'total_files': len(references),
            'total_references': sum(len(refs) for refs in references.values()),
            'categories': {category: len(files) for category, files in categories.items()}
        },
        'categories': categories,
        'files': {file_path: [{'line': line_num, 'content': content} for line_num, content in refs] 
                 for file_path, refs in references.items()}
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    else:
        print("\n=== ember_ml REFERENCES REPORT ===\n")
        print(f"Total files with references: {report['summary']['total_files']}")
        print(f"Total references: {report['summary']['total_references']}")
        print("\nCategories:")
        for category, count in report['summary']['categories'].items():
            if count > 0:
                print(f"  {category}: {count} files")
        
        print("\nFiles to update:")
        for category, files in categories.items():
            if files:
                print(f"\n{category.upper()} FILES:")
                for file_path in sorted(files):
                    ref_count = len(references[file_path])
                    print(f"  {file_path} ({ref_count} references)")
        
        print("\nFor detailed references, use the --output option to save to a JSON file.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find all occurrences of 'ember_ml' in the codebase")
    parser.add_argument("--directory", "-d", default=".", help="Directory to search in")
    parser.add_argument("--ignore", "-i", nargs="+", help="Directories to ignore", 
                        default=['.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.env'])
    parser.add_argument("--output", "-o", help="Output file for detailed report (JSON format)")
    args = parser.parse_args()
    
    references = find_references(args.directory, args.ignore)
    categories = categorize_files(references)
    generate_report(references, categories, args.output)

if __name__ == "__main__":
    main()