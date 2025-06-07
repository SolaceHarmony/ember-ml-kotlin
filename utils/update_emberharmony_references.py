#!/usr/bin/env python3
"""
Update all occurrences of 'ember_ml' to 'ember_ml' in the codebase.

This script searches for 'ember_ml' in all files in the codebase and replaces
it with 'ember_ml', with appropriate capitalization.
"""

import os
import re
import argparse
from typing import List, Dict, Tuple, Set
import json

def update_references(directory: str, ignore_dirs: List[str] = None, dry_run: bool = False) -> Dict[str, int]:
    """
    Update all occurrences of 'ember_ml' to 'ember_ml' in the given directory.
    
    Args:
        directory: Directory to search in
        ignore_dirs: List of directories to ignore
        dry_run: If True, don't actually modify files
        
    Returns:
        Dictionary mapping file paths to number of replacements made
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.env']
    
    replacements = {}
    
    # Patterns to match different capitalizations
    patterns = [
        (re.compile(r'ember_ml', re.IGNORECASE), lambda m: 'ember_ml'),
        (re.compile(r'ember_ml'), lambda m: 'Ember ML'),
        (re.compile(r'ember_ml'), lambda m: 'EMBER_ML'),
        (re.compile(r'ember_ml'), lambda m: 'Ember ML'),
        (re.compile(r'ember_ml'), lambda m: 'ember_ml'),
    ]
    
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
                    content = f.read()
                
                # Count replacements for each pattern
                total_replacements = 0
                new_content = content
                
                for pattern, replacement_func in patterns:
                    # Use a function to handle different capitalizations
                    new_content, count = pattern.subn(replacement_func, new_content)
                    total_replacements += count
                
                if total_replacements > 0:
                    replacements[file_path] = total_replacements
                    
                    if not dry_run:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Updated {file_path}: {total_replacements} replacements")
                    else:
                        print(f"Would update {file_path}: {total_replacements} replacements")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return replacements

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update all occurrences of 'ember_ml' to 'ember_ml'")
    parser.add_argument("--directory", "-d", default=".", help="Directory to search in")
    parser.add_argument("--ignore", "-i", nargs="+", help="Directories to ignore", 
                        default=['.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.env'])
    parser.add_argument("--dry-run", "-n", action="store_true", help="Don't actually modify files")
    parser.add_argument("--output", "-o", help="Output file for report (JSON format)")
    args = parser.parse_args()
    
    replacements = update_references(args.directory, args.ignore, args.dry_run)
    
    # Generate report
    total_files = len(replacements)
    total_replacements = sum(replacements.values())
    
    print(f"\nSummary:")
    print(f"  Files updated: {total_files}")
    print(f"  Total replacements: {total_replacements}")
    
    if args.output:
        report = {
            'summary': {
                'total_files': total_files,
                'total_replacements': total_replacements
            },
            'files': replacements
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")

if __name__ == "__main__":
    main()