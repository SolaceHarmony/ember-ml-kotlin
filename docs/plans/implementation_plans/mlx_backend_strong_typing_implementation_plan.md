# MLX Backend Strong Typing Implementation Plan

This document outlines the specific implementation steps to standardize strong typing across the MLX backend files in the Ember ML project.

## 1. Define Standard Type Aliases

First, we'll create a set of standard type aliases that can be used across all MLX backend files. These will be defined in a central location and imported where needed.

### Implementation Steps:

1. Create a new file `ember_ml/backend/mlx/typing.py` with the following content:

```python
"""Type definitions for MLX backend."""

from typing import Union, Optional, Sequence, Any, List, Tuple, Dict
import numpy as np
import mlx.core as mx

from ember_ml.backend.mlx.tensor import MLXTensor

# Standard type aliases
TensorLike = Optional[Union[int, float, bool, list, tuple, np.ndarray, mx.array, MLXTensor]]
Shape = Union[int, Sequence[int]]
DType = Optional[Union[str, Any]]  # Any covers mx.dtype objects
Device = Optional[str]
```

2. Update the imports in all MLX backend files to include these type aliases.

## 2. Update Core Utility Functions

The core utility functions in `tensor/ops/utility.py` need to be updated to ensure consistent type handling.

### Implementation Steps:

1. Update the `_convert_input` function to ensure it properly handles all input types and explicitly rejects incompatible types:

```python
def _convert_input(x: Any) -> mx.array:
    """
    Convert input to MLX array.
    
    Handles various input types:
    - MLX arrays (returned as-is)
    - NumPy arrays (converted to MLX arrays)
    - EmberTensor/MLXTensor objects (extract underlying data)
    - Python scalars (int, float, bool)
    - Python sequences (list, tuple)
    
    Args:
        x: Input data to convert
        
    Returns:
        MLX array
        
    Raises:
        ValueError: If the input cannot be converted to an MLX array
    """
    # Already an MLX array - check by type and module
    if isinstance(x, mx.array) or (hasattr(x, '__class__') and
                                  hasattr(x.__class__, '__module__') and
                                  x.__class__.__module__ == 'mlx.core' and
                                  x.__class__.__name__ == 'array'):
        return x
        
    # Handle EmberTensor and MLXTensor objects
    if hasattr(x, '__class__') and hasattr(x.__class__, '__name__') and x.__class__.__name__ in ['EmberTensor', 'MLXTensor']:
        # For EmberTensor, extract the underlying data and convert to numpy first
        if hasattr(x, 'to_numpy'):
            return mx.array(x.to_numpy())
        # If it has a _tensor attribute, use that
        elif hasattr(x, '_tensor'):
            return _convert_input(x._tensor)
            
    # Check for NumPy arrays by type name rather than direct import
    if hasattr(x, '__class__') and x.__class__.__module__ == 'numpy' and x.__class__.__name__ == 'ndarray':
        return mx.array(x)
        
    # Handle Python scalars and sequences
    if isinstance(x, (int, float, bool, list, tuple)):
        try:
            return mx.array(x)
        except Exception as e:
            raise ValueError(f"Cannot convert {type(x)} to MLX array: {e}")
    
    # Check for PyTorch tensors and reject them explicitly
    if hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'torch':
        raise ValueError(f"Cannot convert {type(x)} to MLX array. Use tensor.to_numpy() first.")
            
    # Try to convert to MLX array as a last resort
    try:
        return mx.array(x)
    except Exception as e:
        raise ValueError(f"Cannot convert {type(x)} to MLX array: {e}")
```

2. Update the `convert_to_tensor` function to use the new type aliases:

```python
def convert_to_tensor(data: TensorLike, dtype: DType = None, device: Device = None) -> mx.array:
    """
    Convert input to MLX array.
    
    Args:
        data: Input data
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array
    """
    tensor = _convert_input(data)
    if dtype is not None:
        mlx_dtype = _validate_dtype(Tensor._dtype_cls, dtype)
        if mlx_dtype is not None:
            tensor = tensor.astype(mlx_dtype)
    return tensor
```

## 3. Update Operation Files

For each operation file (math_ops.py, solver_ops.py, comparison_ops.py, etc.), we need to update the function signatures to use the standard type aliases.

### Implementation Steps:

1. Update the imports to include the standard type aliases:

```python
from ember_ml.backend.mlx.typing import TensorLike, Shape, DType, Device
```

2. Update function signatures to use these type aliases:

```python
def add(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Add two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise sum
    """
    return mx.add(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))
```

3. Ensure all functions have return type annotations.

## 4. Update Tensor Operation Files

For each tensor operation file (tensor/ops/creation.py, tensor/ops/manipulation.py, etc.), we need to update the function signatures to use the standard type aliases.

### Implementation Steps:

1. Update the imports to include the standard type aliases:

```python
from ember_ml.backend.mlx.typing import TensorLike, Shape, DType, Device
```

2. Update function signatures to use these type aliases:

```python
def zeros(shape: Shape, dtype: DType = None, device: Device = None) -> mx.array:
    """
    Create an MLX array of zeros.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros
    """
    mlx_dtype = _validate_dtype(Tensor._dtype_cls, dtype)
    return mx.zeros(shape, dtype=mlx_dtype)
```

3. Update functions that currently lack type annotations:

```python
def zeros_like(tensor: TensorLike, dtype: DType = None, device: Device = None) -> mx.array:
    """
    Create an MLX array of zeros with the same shape as the input.
    
    Args:
        tensor: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros with the same shape as tensor
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        dtype = MLXDType().from_dtype_str(dtype)
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    # MLX zeros_like doesn't accept dtype parameter
    if dtype is None:
        return mx.zeros_like(tensor_array)
    else:
        # Create zeros with the same shape but specified dtype
        return mx.zeros(tensor_array.shape, dtype=dtype)
```

## 5. Update Comparison Operations

The comparison_ops.py file needs to be updated to use the standard type aliases.

### Implementation Steps:

1. Update the imports to include the standard type aliases:

```python
from ember_ml.backend.mlx.typing import TensorLike, Shape, DType, Device
```

2. Update function signatures to use these type aliases:

```python
def equal(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if two MLX arrays are equal element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements are equal
    """
    return mx.equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))
```

## 6. Update Tests

Update the test_mlx_backend_strong_typing.py file to test the standardized strong typing implementation.

### Implementation Steps:

1. Add tests for the new type aliases:

```python
def test_tensor_like_type_conversion():
    """Test conversion of various TensorLike inputs."""
    # Set the backend to MLX
    set_backend('mlx')
    ops.set_ops('mlx')
    
    # Test with Python scalar
    x = tensor.convert_to_tensor(5)
    assert isinstance(x, mx.array)
    
    # Test with Python list
    x = tensor.convert_to_tensor([1, 2, 3])
    assert isinstance(x, mx.array)
    
    # Test with NumPy array
    x = tensor.convert_to_tensor(np.array([1, 2, 3]))
    assert isinstance(x, mx.array)
    
    # Test with MLX array
    x = tensor.convert_to_tensor(mx.array([1, 2, 3]))
    assert isinstance(x, mx.array)
    
    # Test with EmberTensor
    x_ember = tensor.ones((3,))
    x = tensor.convert_to_tensor(x_ember)
    assert isinstance(x, mx.array)
    
    # Test with PyTorch tensor - should raise ValueError
    x_torch = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        x = tensor.convert_to_tensor(x_torch)
```

2. Add tests for the standardized error messages:

```python
def test_error_messages():
    """Test that error messages are clear and informative."""
    # Set the backend to MLX
    set_backend('mlx')
    ops.set_ops('mlx')
    
    # Test with PyTorch tensor - should raise ValueError with specific message
    x_torch = torch.tensor([1, 2, 3])
    try:
        x = tensor.convert_to_tensor(x_torch)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot convert" in str(e)
        assert "Use tensor.to_numpy() first" in str(e)
```

## 7. File-by-File Implementation Order

Here's the recommended order for implementing these changes:

1. Create the typing.py file with standard type aliases
2. Update tensor/ops/utility.py
3. Update math_ops.py
4. Update solver_ops.py
5. Update comparison_ops.py
6. Update tensor/ops/creation.py
7. Update tensor/ops/indexing.py
8. Update tensor/ops/manipulation.py
9. Update tensor/ops/random.py
10. Update tensor/ops/casting.py
11. Update remaining operation files
12. Update tests

## 8. Testing Strategy

After implementing these changes, we should run the following tests:

1. Run the existing test_mlx_backend_strong_typing.py to ensure basic functionality still works
2. Run the new tests for the standardized strong typing implementation
3. Run all other tests to ensure we haven't broken anything

## 9. Documentation Updates

Finally, we should update the documentation to reflect the standardized strong typing implementation:

1. Update architecture_summary.md to include details about the standardized strong typing implementation
2. Add examples of proper type usage in the MLX backend
3. Document the allowed conversion paths between different tensor types