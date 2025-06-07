# Compatibility Tests

This directory contains compatibility test scripts for Ember ML.

## Files

- **check_matplotlib_compatibility.py**: Tests compatibility between Ember ML tensors and matplotlib plotting
- **check_torch_backend.py**: Tests compatibility between PyTorch backend and matplotlib plotting
- **check_torch_backend_cpu.py**: Tests compatibility between PyTorch backend with CPU conversion and matplotlib plotting
- **test_ember_backend.py**: Tests basic operations with the Ember backend

## Usage

These scripts can be run directly to test compatibility:

```bash
# Test matplotlib compatibility
python tests/compatibility/check_matplotlib_compatibility.py

# Test PyTorch backend compatibility
python tests/compatibility/check_torch_backend.py

# Test PyTorch backend with CPU conversion
python tests/compatibility/check_torch_backend_cpu.py

# Test Ember backend operations
python tests/compatibility/test_ember_backend.py
```

## Purpose

These tests help ensure that Ember ML tensors work correctly with external libraries like matplotlib and that backend-specific operations function as expected.