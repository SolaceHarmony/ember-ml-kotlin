# Utilities

This directory contains utility scripts and helper functions for the Ember ML framework.

## Files

- `update_imports.py`: Script for updating import statements across the codebase
- `generic_feature_extraction.py`: Utilities for extracting features from various data sources

## Usage

These utility scripts can be used to help with development and maintenance tasks:

```bash
# Update import statements across the codebase
python utils/update_imports.py

# Extract features from a dataset
python utils/generic_feature_extraction.py --input data.csv --output features.csv
```

## Adding New Utilities

When adding new utility scripts to this directory, please follow these guidelines:

1. Use clear, descriptive names for files and functions
2. Include docstrings and comments to explain what the utility does
3. Add the new utility to this README.md file
4. Write tests for the utility in the tests directory