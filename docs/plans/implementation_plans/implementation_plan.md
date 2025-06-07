# Comprehensive Implementation Plan for Tensor Operations Refactoring

This document provides a comprehensive implementation plan for refactoring tensor operations across all backends and ensuring frontend compatibility. It combines the master implementation guide, frontend implementation guide, and backend-specific implementation plans.

## Overview

The goal of this refactoring is to make tensor operations more flexible by allowing them to be called both as standalone functions and as methods within tensor classes. The refactoring will:

1. Extract backend operations into standalone functions
2. Maintain method interfaces in backend classes as thin wrappers
3. Ensure a clean API by exposing tensor operations ONLY through `ember_ml.nn.tensor`
4. Remove any tensor-related imports from `ember_ml.ops`

## Implementation Phases

### Phase 1: MLX Backend Refactoring

#### 1.1 Directory Structure Setup

Create the following directory structure:

```
ember_ml/backend/mlx/tensor/
  ├── __init__.py           # Export MLXTensor and all operations
  ├── tensor.py             # Contains the MLXTensor class with method interfaces
  ├── ops/                  # Directory for operation modules
  │   ├── __init__.py       # Export all operations
  │   ├── casting.py        # Contains cast() and related functions
  │   ├── creation.py       # Contains zeros(), ones(), etc.
  │   ├── manipulation.py   # Contains reshape(), transpose(), etc.
  │   ├── random.py         # Contains random_normal(), random_uniform(), etc.
  │   ├── indexing.py       # Contains slice(), gather(), etc.
  │   └── utility.py        # Contains utility functions
```

#### 1.2 Operation Categorization

Organize the tensor operations into the following categories:

- **Casting Operations**: cast()
- **Creation Operations**: zeros(), ones(), eye(), arange(), linspace(), full(), zeros_like(), ones_like(), full_like()
- **Manipulation Operations**: reshape(), transpose(), concatenate(), stack(), split(), expand_dims(), squeeze(), tile(), pad()
- **Indexing Operations**: slice(), slice_update(), gather(), tensor_scatter_nd_update()
- **Utility Operations**: convert_to_tensor(), to_numpy(), item(), shape(), dtype(), copy(), var(), sort(), argsort(), maximum()
- **Random Operations**: random_normal(), random_uniform(), random_binomial(), random_gamma(), random_exponential(), random_poisson(), random_categorical(), random_permutation(), shuffle(), set_seed(), get_seed()

#### 1.3 Implementation Steps

1. Create the directory structure
2. Implement operations by category
3. Update the MLXTensor class to use the standalone functions
4. Update the main __init__.py file to export both the MLXTensor class and all operations
5. Add tests to verify the implementation

### Phase 2: NumPy Backend Refactoring

Follow the same approach as the MLX backend refactoring, but for the NumPy backend:

1. Create the directory structure
2. Implement operations by category
3. Update the NumPyTensor class to use the standalone functions
4. Update the main __init__.py file to export both the NumPyTensor class and all operations
5. Add tests to verify the implementation

### Phase 3: PyTorch Backend Refactoring

Follow the same approach as the MLX backend refactoring, but for the PyTorch backend:

1. Create the directory structure
2. Implement operations by category
3. Update the TorchTensor class to use the standalone functions
4. Update the main __init__.py file to export both the TorchTensor class and all operations
5. Add tests to verify the implementation

### Phase 4: Frontend Compatibility and API Cleanup

#### 4.1 Frontend Compatibility

The frontend tensor implementation in `ember_ml.nn.tensor` already has a structure that supports both function and method calling patterns. Our refactored backend will maintain compatibility with this structure.

##### Current Frontend Structure

The frontend defines standalone functions in `ember_ml/nn/tensor/common/__init__.py` using lambda functions that delegate to the backend:

```python
# Define tensor operations using lambda functions
zeros = lambda *args, **kwargs: _get_tensor_ops().zeros(*args, **kwargs)
ones = lambda *args, **kwargs: _get_tensor_ops().ones(*args, **kwargs)
# ...
cast = lambda *args, **kwargs: _get_tensor_ops().cast(*args, **kwargs)
```

The `EmberTensor` class provides methods that call these standalone functions:

```python
def cast(self, x: Any, dtype: Union[DType, str, Callable[[], Any]]) -> 'EmberTensor':
    # ...
    tensor = cast(x, processed_dtype)
    return EmberTensor(tensor, dtype=processed_dtype, device=self.device, requires_grad=self._requires_grad)
```

##### Verifying Frontend Compatibility

After refactoring the backend, verify that the frontend still works correctly:

1. Test the frontend standalone functions
2. Test the frontend methods
3. Test with different backends

#### 4.2 API Cleanup

**NO BACKWARDS COMPATIBILITY ALLOWED**

Remove all tensor-related imports from `ember_ml.ops/__init__.py` to ensure a clean API.

##### Identify Tensor-Related Imports

In `ember_ml/ops/__init__.py`, identify all tensor-related imports:

```python
# Import data types and tensor operations from nn.tensor
from ember_ml.nn.tensor import (
    # Data types
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16,
    get_dtype, to_dtype_str, from_dtype_str,
    
    # Tensor operations
    EmberTensor, zeros, ones, eye, arange, linspace,
    zeros_like, ones_like, full, full_like,
    reshape, transpose, concatenate, stack, split,
    expand_dims, squeeze, tile, gather, tensor_scatter_nd_update,
    slice, slice_update, convert_to_tensor, cast, copy, var, pad,
    sort, argsort, to_numpy, item, shape, dtype,
    random_uniform, random_normal
)
```

##### Remove Tensor-Related Imports

Remove the tensor-related imports from `ember_ml/ops/__init__.py` and update the `__all__` list accordingly.

##### Update Documentation

Add a comment to `ember_ml/ops/__init__.py` to clearly state that tensor operations are only available through `ember_ml.nn.tensor`.

##### Fix Any References

Search for any code that imports tensor operations from `ember_ml.ops` and update it to import from `ember_ml.nn.tensor` instead.

## Implementation Order

For each backend, implement the operations in the following order:

1. **Casting Operations**: cast()
2. **Creation Operations**: zeros(), ones(), eye(), arange(), linspace(), full(), zeros_like(), ones_like(), full_like()
3. **Manipulation Operations**: reshape(), transpose(), concatenate(), stack(), split(), expand_dims(), squeeze(), tile(), pad()
4. **Indexing Operations**: slice(), slice_update(), gather(), tensor_scatter_nd_update()
5. **Utility Operations**: convert_to_tensor(), to_numpy(), item(), shape(), dtype(), copy(), var(), sort(), argsort(), maximum()
6. **Random Operations**: random_normal(), random_uniform(), random_binomial(), random_gamma(), random_exponential(), random_poisson(), random_categorical(), random_permutation(), shuffle(), set_seed(), get_seed()

## Testing Strategy

For each component of the refactoring:

1. **Unit Tests**: Create unit tests for each standalone function and method
2. **Integration Tests**: Create integration tests that use the frontend with each backend
3. **Regression Tests**: Run existing tests to ensure the refactoring doesn't break existing functionality
4. **Performance Tests**: Benchmark critical operations to ensure there's no significant performance degradation

## Key Principles to Follow

1. **Function-First Approach**: Implement each operation as a standalone function first, then add the method wrapper.
2. **Consistent API**: Maintain the same parameter order and names across all implementations.
3. **Proper Documentation**: Document all functions and methods thoroughly.
4. **Clean API**: Ensure tensor operations are only exposed through `ember_ml.nn.tensor`.
5. **NO BACKWARDS COMPATIBILITY**: This is a pre-production release with no customers, so there's no need to maintain backward compatibility.

## Implementation Checklist

### MLX Backend
- [ ] Create directory structure
- [ ] Implement casting operations
- [ ] Implement creation operations
- [ ] Implement manipulation operations
- [ ] Implement indexing operations
- [ ] Implement utility operations
- [ ] Implement random operations
- [ ] Add tests

### NumPy Backend
- [ ] Create directory structure
- [ ] Implement casting operations
- [ ] Implement creation operations
- [ ] Implement manipulation operations
- [ ] Implement indexing operations
- [ ] Implement utility operations
- [ ] Implement random operations
- [ ] Add tests

### PyTorch Backend
- [ ] Create directory structure
- [ ] Implement casting operations
- [ ] Implement creation operations
- [ ] Implement manipulation operations
- [ ] Implement indexing operations
- [ ] Implement utility operations
- [ ] Implement random operations
- [ ] Add tests

### Frontend and API Cleanup
- [ ] Verify frontend compatibility
- [ ] Remove tensor-related imports from `ember_ml.ops/__init__.py`
- [ ] Update documentation
- [ ] Add tests
- [ ] Perform final cleanup

## Documentation

After implementing the changes, update the documentation to reflect the new architecture:

1. Update the README.md file to clearly state that tensor operations are only available through `ember_ml.nn.tensor`.
2. Update any tutorials or examples that use tensor operations to import from `ember_ml.nn.tensor`.
3. Add a note to the documentation about the refactoring and the new architecture.

## Final Cleanup

After all the changes have been implemented and tested, perform a final cleanup:

1. Remove any deprecated code or comments.
2. Ensure consistent coding style across all files.
3. Run linters and formatters to ensure code quality.
4. Run all tests to ensure everything works correctly.

## Implementation Steps Summary

1. Refactor the backend tensor operations to be standalone functions.
2. Verify frontend compatibility with the refactored backend.
3. Remove tensor-related imports from `ember_ml.ops/__init__.py`.
4. Update documentation to reflect the new architecture.
5. Test to ensure everything works correctly.
6. Perform final cleanup.