# PyTorch Backend Implementation Guide

This guide provides detailed instructions for implementing tensor operations in the PyTorch backend. For a comprehensive understanding of the tensor operations architecture, please refer to the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document and the [Tensor Implementation Guide](tensor_impl_guide.md).

## 1. Directory Structure

Create the following directory structure:

```
ember_ml/backend/torch/tensor/
  ├── __init__.py           # Export TorchTensor and all operations
  ├── tensor.py             # Contains the TorchTensor class with method interfaces
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
mkdir -p ember_ml/backend/torch/tensor/ops
```

### 2.2. Create the ops/__init__.py file

Create the file `ember_ml/backend/torch/tensor/ops/__init__.py` with the following content:

```python
"""
PyTorch tensor operations.

This module provides standalone functions for tensor operations using the PyTorch backend.
These functions can be called directly or through the TorchTensor class methods.
"""

# Import operations from modules
from ember_ml.backend.torch.tensor.ops.casting import cast
# Import other operations as they are implemented

# Export all operations
__all__ = [
    'cast',
    # Add other operations as they are implemented
]
```

### 2.3. Implement the cast() operation

Create the file `ember_ml/backend/torch/tensor/ops/casting.py` with the following content:

```python
"""PyTorch tensor casting operations."""

import torch
from typing import Any

from ember_ml.backend.torch.tensor.dtype import TorchDType

def cast(tensor_obj, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        tensor_obj: TorchTensor instance
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    if not isinstance(tensor_obj._tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor_obj._tensor)
    else:
        tensor = tensor_obj._tensor
    
    torch_dtype = TorchDType().from_dtype_str(dtype)
    return tensor.to(torch_dtype)
```

### 2.4. Update the TorchTensor class

Update the file `ember_ml/backend/torch/tensor/tensor.py` to use the standalone cast() function:

```python
def cast(self, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    from ember_ml.backend.torch.tensor.ops.casting import cast as cast_func
    return cast_func(self, dtype)
```

### 2.5. Update the main __init__.py

Update the file `ember_ml/backend/torch/tensor/__init__.py` to export both the TorchTensor class and the standalone functions:

```python
"""PyTorch tensor module."""

from ember_ml.backend.torch.tensor.tensor import TorchTensor
from ember_ml.backend.torch.tensor.ops import cast
# Import other operations as they are implemented

__all__ = [
    'TorchTensor',
    'cast',
    # Add other operations as they are implemented
]
```

## 3. Testing

After implementing the changes, test the refactored code to ensure it works correctly:

1. Test the standalone function:
```python
from ember_ml.backend.torch.tensor.ops import cast
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
3. Update the TorchTensor class to use the standalone functions
4. Update the __init__.py files to export the functions

Follow the categorization outlined in the [Tensor Operations Architecture](docs/api/tensor_architecture.md) document:

- **Creation Operations**: zeros(), ones(), eye(), arange(), linspace(), full(), zeros_like(), ones_like(), full_like()
- **Manipulation Operations**: reshape(), transpose(), concatenate(), stack(), split(), expand_dims(), squeeze(), tile(), pad()
- **Random Operations**: random_normal(), random_uniform(), random_binomial(), random_gamma(), random_exponential(), random_poisson(), random_categorical(), random_permutation(), shuffle(), set_seed(), get_seed()
- **Indexing Operations**: slice(), slice_update(), gather(), tensor_scatter_nd_update()
- **Utility Operations**: convert_to_tensor(), to_numpy(), item(), shape(), dtype(), copy(), var(), sort(), argsort(), maximum()

## 5. Example: Creation Operations

Here's an example of how to implement the zeros() operation:

Create the file `ember_ml/backend/torch/tensor/ops/creation.py` with the following content:

```python
"""PyTorch tensor creation operations."""

import torch
from typing import Any, Optional, Union, Sequence

from ember_ml.backend.torch.tensor.dtype import TorchDType

def zeros(tensor_obj, shape: Union[int, Sequence[int]], dtype: Optional[Any] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros.
    
    Args:
        tensor_obj: TorchTensor instance
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    if isinstance(shape, int):
        shape = (shape,)
    
    return torch.zeros(shape, dtype=torch_dtype, device=device)
```

Then update the TorchTensor class:

```python
def zeros(self, shape, dtype=None, device=None):
    """
    Create a tensor of zeros.
    
    Args:
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros
    """
    from ember_ml.backend.torch.tensor.ops.creation import zeros as zeros_func
    return zeros_func(self, shape, dtype, device)
```

And update the ops/__init__.py file:

```python
from ember_ml.backend.torch.tensor.ops.casting import cast
from ember_ml.backend.torch.tensor.ops.creation import zeros

__all__ = [
    'cast',
    'zeros',
]
```

Finally, update the main __init__.py file:

```python
from ember_ml.backend.torch.tensor.tensor import TorchTensor
from ember_ml.backend.torch.tensor.ops import cast, zeros

__all__ = [
    'TorchTensor',
    'cast',
    'zeros',
]
```

## 6. Special Considerations for PyTorch

### 6.1. Device Handling

PyTorch tensors can be placed on different devices (CPU, CUDA, etc.). Make sure to handle the device parameter correctly in all operations:

```python
def convert_to_tensor(tensor_obj, data, dtype=None, device=None):
    """
    Convert data to a PyTorch tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        data: The data to convert
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor
    """
    # If it's already a PyTorch tensor, return it
    if isinstance(data, torch.Tensor):
        if dtype is not None:
            # Convert dtype if needed
            torch_dtype = TorchDType().from_dtype_str(dtype)
            data = data.to(torch_dtype)
        if device is not None:
            data = data.to(device)
        return data
    
    # Convert to PyTorch tensor
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Handle EmberTensor objects
    if isinstance(data, object) and getattr(data.__class__, '__name__', '') == 'EmberTensor':
        # For EmberTensor, extract the underlying PyTorch tensor
        data = getattr(data, '_tensor')
    
    # Handle array-like objects
    try:
        tensor = torch.tensor(data, dtype=torch_dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    except:
        raise ValueError(f"Cannot convert {type(data)} to PyTorch tensor")
```

### 6.2. Gradient Tracking

PyTorch tensors can track gradients for automatic differentiation. Make sure to handle the requires_grad parameter correctly:

```python
def convert_to_tensor(tensor_obj, data, dtype=None, device=None, requires_grad=False):
    """
    Convert data to a PyTorch tensor.
    
    Args:
        tensor_obj: TorchTensor instance
        data: The data to convert
        dtype: Optional data type
        device: Optional device to place the tensor on
        requires_grad: Whether the tensor requires gradients
        
    Returns:
        PyTorch tensor
    """
    # Implementation...
    
    tensor = torch.tensor(data, dtype=torch_dtype)
    if device is not None:
        tensor = tensor.to(device)
    tensor.requires_grad = requires_grad
    return tensor
```

For more detailed information on implementing tensor operations, please refer to the [Tensor Implementation Guide](tensor_impl_guide.md).