# MLX Backend Implementation Guide

This guide provides detailed instructions for implementing tensor operations in the MLX backend. For a comprehensive understanding of the tensor operations architecture, please refer to the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document and the [Tensor Implementation Guide](tensor_impl_guide.md).

## 1. Directory Structure

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

## 2. Implementation Steps

### 2.1. Create the ops directory

```bash
mkdir -p ember_ml/backend/mlx/tensor/ops
```

### 2.2. Create the ops/__init__.py file

Create the file `ember_ml/backend/mlx/tensor/ops/__init__.py` with the following content:

```python
"""
MLX tensor operations.

This module provides standalone functions for tensor operations using the MLX backend.
These functions can be called directly or through the MLXTensor class methods.
"""

# Import operations from modules
from ember_ml.backend.mlx.tensor.ops.casting import cast
# Import other operations as they are implemented

# Export all operations
__all__ = [
    'cast',
    # Add other operations as they are implemented
]
```

### 2.3. Implement the cast() operation

Create the file `ember_ml/backend/mlx/tensor/ops/casting.py` with the following content:

```python
"""MLX tensor casting operations."""

import mlx.core as mx
from typing import Any, Optional

from ember_ml.backend.mlx.tensor.dtype import MLXDType, DType

def _validate_dtype(dtype_cls: MLXDType, dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to MLX format.
    
    Args:
        dtype_cls: MLXDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated MLX dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return dtype_cls.from_dtype_str(str(dtype.name))
        
    # If it's already an MLX dtype, return as is
    if isinstance(dtype, type(mx.float32)):
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def cast(tensor_obj, dtype):
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor_obj: MLXTensor instance
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Get the tensor array from the tensor object
    tensor_array = tensor_obj.convert_to_tensor(tensor_obj)
    
    # Validate the dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # If mlx_dtype is None, return the tensor as is
    if mlx_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype
    return tensor_array.astype(mlx_dtype)
```

### 2.4. Update the MLXTensor class

Update the file `ember_ml/backend/mlx/tensor/tensor.py` to use the standalone cast() function:

```python
def cast(self, dtype):
    """
    Cast a tensor to a new data type.
    
    Args:
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    from ember_ml.backend.mlx.tensor.ops.casting import cast as cast_func
    return cast_func(self, dtype)
```

### 2.5. Update the main __init__.py

Update the file `ember_ml/backend/mlx/tensor/__init__.py` to export both the MLXTensor class and the standalone functions:

```python
"""MLX tensor module."""

from ember_ml.backend.mlx.tensor.tensor import MLXTensor
from ember_ml.backend.mlx.tensor.ops import cast
# Import other operations as they are implemented

__all__ = [
    'MLXTensor',
    'cast',
    # Add other operations as they are implemented
]
```

## 3. Testing

After implementing the changes, test the refactored code to ensure it works correctly:

1. Test the standalone function:
```python
from ember_ml.backend.mlx.tensor.ops import cast
result = cast(tensor_obj, dtype)
```

2. Test the method:
```python
result = tensor_obj.cast(dtype)
```

## 4. Next Steps

After implementing the cast() operation, continue with the other operations following the same pattern:

1. Create the appropriate module file (e.g., creation.py, manipulation.py)
2. Implement the standalone functions
3. Update the MLXTensor class to use the standalone functions
4. Update the __init__.py files to export the functions

Follow the categorization outlined in the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document:

- **Creation Operations**: zeros(), ones(), eye(), arange(), linspace(), full(), zeros_like(), ones_like(), full_like()
- **Manipulation Operations**: reshape(), transpose(), concatenate(), stack(), split(), expand_dims(), squeeze(), tile(), pad()
- **Random Operations**: random_normal(), random_uniform(), random_binomial(), random_gamma(), random_exponential(), random_poisson(), random_categorical(), random_permutation(), shuffle(), set_seed(), get_seed()
- **Indexing Operations**: slice(), slice_update(), gather(), tensor_scatter_nd_update()
- **Utility Operations**: convert_to_tensor(), to_numpy(), item(), shape(), dtype(), copy(), var(), sort(), argsort(), maximum()

For more detailed information on implementing tensor operations, please refer to the [Tensor Implementation Guide](tensor_impl_guide.md).