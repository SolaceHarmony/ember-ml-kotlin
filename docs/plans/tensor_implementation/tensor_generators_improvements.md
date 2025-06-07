# Tensor and DType Generators Improvements

## Overview

This document outlines planned improvements to tensor and dtype generators in the Ember ML framework to enhance user experience and maintain backend purity.

## Background

During refactoring of tests to maintain backend purity, we identified that the current tensor creation and comparison utilities could be improved to provide a more intuitive and user-friendly experience. This would reduce the temptation for developers to fall back on backend-specific code (like NumPy) and make it easier to maintain backend purity.

## Proposed Improvements

### EmberTensor Operators

1. **Operator Overloading**: Implement operator overloading for EmberTensor to enable intuitive arithmetic and comparison operations:
   ```python
   # Arithmetic operations
   c = EmberTensor([1, 2]) + EmberTensor([2, 3])  # [3, 5]
   c = EmberTensor([5, 6]) - EmberTensor([2, 3])  # [3, 3]
   c = EmberTensor([1, 2]) * EmberTensor([2, 3])  # [2, 6]
   c = EmberTensor([4, 6]) / EmberTensor([2, 3])  # [2, 2]
   
   # Comparison operations
   result = EmberTensor([1, 1]) == EmberTensor([1, 1])  # [True, True]
   result = EmberTensor([1, 2]) != EmberTensor([1, 1])  # [False, True]
   result = EmberTensor([1, 2]) > EmberTensor([2, 1])   # [False, True]
   result = EmberTensor([1, 2]) < EmberTensor([2, 1])   # [True, False]
   
   # Scalar operations
   c = EmberTensor([1, 2]) + 2  # [3, 4]
   c = 3 * EmberTensor([1, 2])  # [3, 6]
   ```

   On the backend, these operations would translate to the appropriate backend functions:
   ```python
   # For MLX backend
   mx.add(mx.array([1, 2]), mx.array([2, 3]))
   
   # For PyTorch backend
   torch.add(torch.tensor([1, 2]), torch.tensor([2, 3]))
   
   # For NumPy backend
   np.add(np.array([1, 2]), np.array([2, 3]))
   ```

### EmberTensor Generators

1. **Static Methods**: Implement static methods on the EmberTensor class for common operations to avoid needing temporary tensor objects:
   ```python
   # Current approach
   tensor_obj = EmberTensor([0])
   a = tensor_obj.zeros((2, 3))
   
   # Proposed approach
   a = EmberTensor.zeros((2, 3))
   ```

2. **Comparison Utilities**: Add intuitive array comparison functions directly in the tensor module:
   ```python
   # Proposed utilities
   from ember_ml.nn.tensor import array_equal, allclose
   
   assert array_equal(tensor1, tensor2)
   assert allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8)
   ```

3. **Testing Helpers**: Implement helper functions for common testing patterns:
   ```python
   # Proposed testing helpers
   from ember_ml.nn.tensor.testing import assert_tensor_equal, assert_tensor_allclose
   
   assert_tensor_equal(tensor1, tensor2)
   assert_tensor_allclose(tensor1, tensor2)
   ```

4. **Convenience Constructors**: Add more convenience constructors for common tensor types:
   ```python
   # Proposed constructors
   identity_matrix = EmberTensor.identity(3)
   diagonal_matrix = EmberTensor.diag([1, 2, 3])
   random_tensor = EmberTensor.random_normal((2, 3))
   ```

### DType Generators

1. **DType Creation**: Simplify DType creation with more intuitive constructors:
   ```python
   # Current approach
   dtype = get_dtype('float32')
   
   # Proposed approach
   dtype = DType.float32()
   ```

2. **DType Conversion**: Add utilities for converting between different dtype representations:
   ```python
   # Proposed utilities
   from ember_ml.nn.tensor import convert_dtype
   
   numpy_dtype = convert_dtype('float32', target='numpy')
   torch_dtype = convert_dtype('float32', target='torch')
   ```

3. **DType Compatibility**: Add utilities for checking dtype compatibility:
   ```python
   # Proposed utilities
   from ember_ml.nn.tensor import is_compatible_dtype, get_common_dtype
   
   if is_compatible_dtype(dtype1, dtype2):
       # Do something
   
   common_dtype = get_common_dtype([dtype1, dtype2, dtype3])
   ```

4. **DType Registry**: Implement a registry for custom dtypes:
   ```python
   # Proposed registry
   from ember_ml.nn.tensor import register_dtype
   
   register_dtype('custom_float', backend_mappings={
       'numpy': np.float32,
       'torch': torch.float32,
       'mlx': mlx.core.float32
   })
   ```

## Implementation Plan

1. Create new modules:
   - `ember_ml.nn.tensor.testing` for testing utilities
   - `ember_ml.nn.tensor.dtypes.registry` for dtype registry

2. Update existing modules:
   - `ember_ml.nn.tensor.common.ember_tensor` to add static methods
   - `ember_ml.nn.tensor.common.dtypes` to add dtype utilities

3. Update documentation:
   - Add examples for all new utilities
   - Create tutorials for common tensor operations

4. Add tests:
   - Test all new utilities
   - Ensure backward compatibility

## Benefits

These improvements will:

1. Make the framework more user-friendly
2. Reduce the temptation to fall back on backend-specific code
3. Make it easier to maintain backend purity
4. Improve testing capabilities
5. Enhance the overall developer experience

## Timeline

- Initial implementation: Q2 2025
- Documentation and testing: Q2 2025
- Release: Q3 2025