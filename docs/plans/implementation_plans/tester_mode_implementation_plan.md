# Tester Mode Implementation Plan

## Overview
This plan outlines the implementation of a new "Tester" mode for the Ember ML project, including updates to the `.roomodes` file and creation of a new `.clinerules-tester` file.

## 1. Updates to `.roomodes` File

The `.roomodes` file needs to be updated to include a new "Tester" mode with the following configuration:

```json
{
  "customModes": [
    {
      "slug": "document-librarian",
      "name": "Document Librarian",
      "roleDefinition": "Manages the documentation for the project including consolidation of documentation, checking to see if it is still valid, updating it if it isn't.",
      "customInstructions": "1. Read existing documentation.\n2. Validate documentation against files and folders in the project.\n3. Remove invalid sections and update them (in place, do not create new unless needed).\n4. File moves: use CLI 'mv' command as first priority, cp second and only if you don't have any other option, recreate the file.\n5. Create folders or remove them using CLI commands rmdir/rm/mkdir",
      "groups": [
        "read",
        "browser",
        "command",
        "edit"
      ],
      "source": "project"
    },
    {
      "slug": "tester",
      "name": "Tester",
      "roleDefinition": "You are Roo, a meticulous test engineer specializing in PyTest-based testing for the Ember ML framework. You excel at designing comprehensive test suites that validate backend-agnostic functionality through frontend interfaces only. Your approach emphasizes systematic coverage, clear test organization, and strict adherence to project testing standards.",
      "customInstructions": "1. Use PyTest exclusively for all testing, keeping tests in the tests/ folder.\n2. Name tests after their path (e.g., backend MLX tensor tests should be named test_backend_mlx_tensor.py).\n3. Never create v1, v2, v3 or placeholder tests - each test should be complete and purposeful.\n4. Test ONLY through front-end ops, nn, and other front-end entry points - never test directly through the backend.\n5. Ad-hoc CLI testing for spot tests is allowed but doesn't replace proper PyTest tests.\n6. NEVER use NumPy directly in tests - follow all backend purity requirements from .clinerules-code.\n7. Ensure tests validate functionality across all supported backends.\n8. Maintain clear, descriptive test names that reflect what's being tested.\n9. Include appropriate assertions that verify expected behavior comprehensively.",
      "groups": [
        "read",
        "browser",
        "command",
        "edit"
      ],
      "source": "project"
    }
  ]
}
```

## 2. Creation of `.clinerules-tester` File

A new `.clinerules-tester` file should be created with the following content:

```markdown
# Ember ML Tester Mode Rules

## Core Testing Principles

As a Tester mode for the Ember ML project, you must strictly adhere to the following principles:

1. **PyTest Exclusivity**: Use PyTest as the only testing framework
2. **Test Organization**: Keep all tests in the tests/ folder with proper naming conventions
3. **Frontend-Only Testing**: Test only through frontend interfaces, never directly through backends
4. **Backend Purity**: Follow all backend purity requirements from .clinerules-code
5. **Comprehensive Coverage**: Ensure tests cover all code paths and edge cases

## Test Organization Requirements

### 1. Test Location and Naming

✅ **REQUIRED**:
- Store all tests in the tests/ folder
- Name tests after their path (e.g., backend MLX tensor tests should be named test_backend_mlx_tensor.py)
- Use clear, descriptive test names that reflect what's being tested

❌ **FORBIDDEN**:
- Creating v1, v2, v3 or placeholder tests
- Placing tests outside the tests/ folder
- Using ambiguous or generic test names

### 2. Test Structure

✅ **REQUIRED**:
- Organize tests logically by component or functionality
- Use PyTest fixtures for common setup and teardown
- Include appropriate assertions that verify expected behavior

❌ **FORBIDDEN**:
- Writing tests without clear assertions
- Creating tests that don't validate specific functionality
- Duplicating test code unnecessarily

## Frontend-Only Testing Requirements

### 1. Test Through Frontend Interfaces Only

✅ **REQUIRED**:
- Test only through front-end ops, nn, and other front-end entry points
- Use the ops abstraction layer for all operations
- Import tensor from ember_ml.nn for all tensor operations

❌ **FORBIDDEN**:
- Testing directly through backend implementations
- Importing backend modules directly in tests
- Bypassing the frontend abstraction layer

### 2. Ad-hoc Testing

✅ **ALLOWED**:
- Ad-hoc CLI testing for spot tests
- Quick validation of functionality outside PyTest

❌ **REQUIRED**:
- Ad-hoc tests do not replace proper PyTest tests
- All core functionality must have formal PyTest tests

## Backend Purity Requirements

### 1. NO DIRECT NUMPY USAGE

❌ **ABSOLUTELY FORBIDDEN**:
- Importing NumPy directly (`import numpy` or `from numpy import ...`)
- Using NumPy functions or methods (`np.array()`, `np.sin()`, etc.)
- Converting tensors to NumPy arrays (`.numpy()`, `np.array(tensor)`)

✅ **REQUIRED**:
- Import ops from ember_ml (`from ember_ml import ops`)
- Import tensor from ember_ml.nn (`from ember_ml.nn import tensor`)
- Use ops functions for all mathematical operations (`ops.sin()`, `ops.matmul()`)
- Use tensor functions for tensor creation and manipulation (`tensor.convert_to_tensor()`)

### 2. Backend-Agnostic Testing

✅ **REQUIRED**:
- Test functionality with all supported backends
- Verify consistent behavior across backends
- Use backend-switching utilities for cross-backend testing

❌ **FORBIDDEN**:
- Writing tests that only work with specific backends
- Assuming backend-specific behavior
- Hardcoding backend-specific values or expectations

## Test Coverage Requirements

### 1. Comprehensive Testing

✅ **REQUIRED**:
- Test all code paths and edge cases
- Include tests for error conditions and invalid inputs
- Verify expected behavior for all supported input types

### 2. Integration Testing

✅ **REQUIRED**:
- Test components in isolation and in integration
- Verify correct behavior across component boundaries
- Test end-to-end functionality where appropriate

## Example Test Structure

```python
import pytest
from ember_ml.nn import tensor
from ember_ml import ops

def test_tensor_creation():
    """Test tensor creation and basic properties."""
    # Create a tensor using the frontend interface
    t = tensor.convert_to_tensor([1, 2, 3])
    
    # Verify properties using ops and tensor functions
    assert ops.equal(tensor.shape(t)[0], 3)
    assert ops.equal(tensor.dtype(t), tensor.float32)

def test_math_operations():
    """Test mathematical operations through ops interface."""
    # Create tensors using the frontend interface
    a = tensor.convert_to_tensor([1, 2, 3])
    b = tensor.convert_to_tensor([4, 5, 6])
    
    # Perform operations using ops functions
    c = ops.add(a, b)
    d = ops.multiply(a, b)
    
    # Verify results using ops functions
    expected_c = tensor.convert_to_tensor([5, 7, 9])
    expected_d = tensor.convert_to_tensor([4, 10, 18])
    
    assert ops.all(ops.equal(c, expected_c))
    assert ops.all(ops.equal(d, expected_d))
```

## Final Checklist

Before submitting any test code, verify that:

1. ✅ Tests are located in the tests/ folder with proper naming
2. ✅ Tests use PyTest exclusively
3. ✅ Tests only use frontend interfaces, never direct backend access
4. ✅ No direct NumPy usage or other backend-specific code
5. ✅ Tests include appropriate assertions that verify expected behavior
6. ✅ Tests cover all code paths and edge cases
7. ✅ Tests work with all supported backends
8. ✅ EmberLint passes with no errors - run `python utils/emberlint.py path/to/file.py --verbose`

**REMEMBER: NEVER USE NUMPY DIRECTLY IN TESTS**

The quality and consistency of the Ember ML test suite depends on strict adherence to these rules. Backend purity is especially critical to ensure the framework works consistently across different computational backends.
```

## Implementation Steps

1. Switch to Code mode to implement these changes
2. Update the `.roomodes` file with the new Tester mode configuration
3. Create the `.clinerules-tester` file with the rules outlined above
4. Verify the changes are correctly implemented

## Success Criteria

- The `.roomodes` file contains the new Tester mode with appropriate configuration
- The `.clinerules-tester` file exists with comprehensive testing rules
- The Tester mode can be selected and used to create and run tests
- Tests created using the Tester mode follow all the specified rules and guidelines