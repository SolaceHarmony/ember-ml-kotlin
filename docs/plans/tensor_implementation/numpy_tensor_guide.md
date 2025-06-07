# NumPy Backend Implementation Guide

This guide provides detailed instructions for implementing tensor operations in the NumPy backend. For a comprehensive understanding of the tensor operations architecture, please refer to the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document and the [Tensor Implementation Guide](tensor_impl_guide.md).

## 1. Directory Structure

Create the following directory structure:

```
ember_ml/backend/numpy/tensor/
  ├── __init__.py           # Export NumpyTensor and all operations
  ├── tensor.py             # Contains the NumpyTensor class with method interfaces
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
mkdir -p ember_ml/backend/numpy/tensor/ops
```

### 2.2. Create the ops/__init__.py file

Create the file `ember_ml/backend/numpy/tensor/ops/__init__.py` with the following content:

```python
"""
NumPy tensor operations.

This module provides standalone functions for tensor operations using the NumPy backend.
These functions can be called directly or through the NumpyTensor class methods.
"""

# Import operations from modules
from ember_ml.backend.numpy.tensor.ops.casting import cast
# Import other operations as they are implemented

# Export all operations
__all__ = [
    'cast',
    # Add other operations as they are implemented
]
```

### 2.3. Implement the cast() operation

Create the file `ember_ml/backend/numpy/tensor/ops/casting.py` with the following content:

```python
"""NumPy tensor casting operations."""

import numpy as np
from typing import Any, Optional

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

def cast(tensor_obj, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        tensor_obj: NumpyTensor instance
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    if not isinstance(tensor_obj._tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor_obj._tensor)
    else:
        tensor = tensor_obj._tensor
    
    numpy_dtype = NumpyDType().from_dtype_str(dtype)
    return tensor.astype(numpy_dtype)
```

### 2.4. Update the NumpyTensor class

Update the file `ember_ml/backend/numpy/tensor/tensor.py` to use the standalone cast() function:

```python
def cast(self, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    from ember_ml.backend.numpy.tensor.ops.casting import cast as cast_func
    return cast_func(self, dtype)
```

### 2.5. Update the main __init__.py

Update the file `ember_ml/backend/numpy/tensor/__init__.py` to export both the NumpyTensor class and the standalone functions:

```python
"""NumPy tensor module."""

from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
from ember_ml.backend.numpy.tensor.ops import cast
# Import other operations as they are implemented

__all__ = [
    'NumpyTensor',
    'cast',
    # Add other operations as they are implemented
]
```

## 3. Testing

After implementing the changes, test the refactored code to ensure it works correctly:

1. Test the standalone function:
```python
from ember_ml.backend.numpy.tensor.ops import cast
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
3. Update the NumpyTensor class to use the standalone functions
4. Update the __init__.py files to export the functions

Follow the categorization outlined in the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document:

- **Creation Operations**: zeros(), ones(), eye(), arange(), linspace(), full(), zeros_like(), ones_like(), full_like()
- **Manipulation Operations**: reshape(), transpose(), concatenate(), stack(), split(), expand_dims(), squeeze(), tile(), pad()
- **Random Operations**: random_normal(), random_uniform(), random_binomial(), random_gamma(), random_exponential(), random_poisson(), random_categorical(), random_permutation(), shuffle(), set_seed(), get_seed()
- **Indexing Operations**: slice(), slice_update(), gather(), tensor_scatter_nd_update()
- **Utility Operations**: convert_to_tensor(), to_numpy(), item(), shape(), dtype(), copy(), var(), sort(), argsort(), maximum()

## 5. Example: Creation Operations

Here's an example of how to implement the zeros() operation:

Create the file `ember_ml/backend/numpy/tensor/ops/creation.py` with the following content:

```python
"""NumPy tensor creation operations."""

import numpy as np
from typing import Any, Optional, Union, Sequence

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

def zeros(tensor_obj, shape: Union[int, Sequence[int]], dtype: Optional[Any] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor of zeros.
    
    Args:
        tensor_obj: NumpyTensor instance
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device (ignored for NumPy backend)
        
    Returns:
        Tensor of zeros
    """
    numpy_dtype = None
    if dtype is not None:
        numpy_dtype = NumpyDType().from_dtype_str(dtype)
    
    return np.zeros(shape, dtype=numpy_dtype)
```

Then update the NumpyTensor class:

```python
def zeros(self, shape, dtype=None, device=None):
    """
    Create a tensor of zeros.
    
    Args:
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device (ignored for NumPy backend)
        
    Returns:
        Tensor of zeros
    """
    from ember_ml.backend.numpy.tensor.ops.creation import zeros as zeros_func
    return zeros_func(self, shape, dtype, device)
```

And update the ops/__init__.py file:

```python
from ember_ml.backend.numpy.tensor.ops.casting import cast
from ember_ml.backend.numpy.tensor.ops.creation import zeros

__all__ = [
    'cast',
    'zeros',
]
```

Finally, update the main __init__.py file:

```python
from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
from ember_ml.backend.numpy.tensor.ops import cast, zeros

__all__ = [
    'NumpyTensor',
    'cast',
    'zeros',
]
```

For more detailed information on implementing tensor operations, please refer to the [Tensor Implementation Guide](tensor_impl_guide.md).