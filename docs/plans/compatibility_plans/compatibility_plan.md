# Comprehensive Compatibility Plan

This document provides a comprehensive plan for ensuring compatibility across different components of the Ember ML framework during the tensor operations refactoring. It covers frontend compatibility, wiring module fixes, and dynamic dtype properties implementation.

## 1. Frontend Compatibility

### 1.1 Current Frontend Architecture

The frontend tensor implementation currently consists of:

1. **Standalone Functions**: Defined in `ember_ml/nn/tensor/common/__init__.py` as lambda functions that delegate to the current backend's tensor operations through an instance.

```python
# Define tensor operations using lambda functions
zeros = lambda *args, **kwargs: _get_tensor_ops().zeros(*args, **kwargs)
ones = lambda *args, **kwargs: _get_tensor_ops().ones(*args, **kwargs)
# ...
cast = lambda *args, **kwargs: _get_tensor_ops().cast(*args, **kwargs)
```

2. **EmberTensor Class**: Implements the `TensorInterface` and provides methods that call these standalone functions.

3. **Backend Integration**: Uses `_get_tensor_ops()` to get an instance of the tensor operations class for the current backend.

### 1.2 Improved Frontend Architecture

With our refactored backend structure, we can simplify the frontend implementation:

1. **Direct Function Imports**: Instead of using lambda functions that delegate to an instance, we can directly import the standalone functions from the backend.

```python
# Import backend functions directly
from ember_ml.backend import get_backend

def _get_backend_module():
    """Get the current backend module."""
    try:
        return get_backend_module()
    except (ImportError, ModuleNotFoundError):
        # If backend-specific implementation not found, use common implementation
        return importlib.import_module('ember_ml.backend.numpy')

def _import_backend_function(function_name):
    """Import a function from the current backend."""
    backend = get_backend()
    backend_module = _get_backend_module()
    
    # Import the function from the backend's tensor.ops module
    try:
        # Try to import from the new structure
        module = importlib.import_module(f'ember_ml.backend.{backend}.tensor.ops')
        return getattr(module, function_name)
    except (ImportError, AttributeError):
        # Fallback to the old structure
        tensor_ops = getattr(backend_module, 'Tensor')()
        return getattr(tensor_ops, function_name)

# Define tensor operations by importing them directly
zeros = _import_backend_function('zeros')
ones = _import_backend_function('ones')
# ...
cast = _import_backend_function('cast')
```

2. **EmberTensor Class**: The `EmberTensor` class can remain largely unchanged, as it already calls the standalone functions.

3. **Backend Integration**: We no longer need to instantiate a tensor operations class for each backend. Instead, we can directly import the functions we need.

### 1.3 Implementation Strategy

#### Phase 1: Compatibility Layer

1. Implement the refactored backend structure with standalone functions.
2. Create a compatibility layer in the frontend that can work with both the old and new backend structures.
3. Test the frontend with this compatibility layer.

#### Phase 2: Frontend Refactoring

1. Update the frontend to directly import the standalone functions from the backend.
2. Maintain backward compatibility by falling back to the old structure if needed.
3. Test the refactored frontend with each backend.

#### Phase 3: Complete Migration

1. Once all backends are refactored and the frontend is updated, we can remove the compatibility layer.
2. Simplify the frontend code to only use the new approach.
3. Run comprehensive tests to verify everything works correctly.

### 1.4 Benefits of the New Approach

1. **Simplicity**: The frontend code becomes simpler and more direct.
2. **Performance**: We eliminate the overhead of going through an instance for function calls.
3. **Flexibility**: It's easier to add new operations or modify existing ones.
4. **Consistency**: The frontend and backend use the same function-based approach.

## 2. Wiring Module Fixes

### 2.1 Current Issues

The current implementation of the wiring module has several issues related to the tensor refactoring:

1. It tries to use EmberTensor methods as static methods, which is not supported
2. It imports EmberTensor from the wrong location (`ember_ml.ops.tensor` instead of `ember_ml.nn.tensor`)
3. It uses a `data` property on EmberTensor, which doesn't exist in the new implementation
4. It specifies dtypes as strings (e.g., `'int32'`) instead of using dtype objects (e.g., `int32`)

### 2.2 Proposed Solution

We need to update the wiring module to use the new tensor API correctly. This includes:

1. Updating imports to use the correct module paths
2. Using module-level functions instead of static methods
3. Removing references to the non-existent `data` property
4. Using dtype objects directly instead of strings

### 2.3 Implementation Plan

#### 2.3.1 Fix ember_ml/nn/wirings/wiring.py

Update the imports and method calls:

```python
# Before
from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor

class Wiring:
    # ...
    def __init__(self, units, output_dim=None, input_dim=None, sparsity_level=0.5, seed=None):
        # ...
        # Initialize adjacency matrices
        self.adjacency_matrix = EmberTensor.zeros([units, units], dtype='int32')
        # ...
```

```python
# After
from ember_ml.nn.tensor import zeros, convert_to_tensor, int32

class Wiring:
    # ...
    def __init__(self, units, output_dim=None, input_dim=None, sparsity_level=0.5, seed=None):
        # ...
        # Initialize adjacency matrices
        self.adjacency_matrix = zeros([units, units], dtype=int32)
        # ...
```

#### 2.3.2 Fix ember_ml/nn/wirings/random_wiring.py

Update the imports and method calls:

```python
# Before
from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor
from ember_ml.ops.dtypes import EmberDtype
from ember_ml.nn.wirings.wiring import Wiring

class RandomWiring(Wiring):
    # ...
    def build(self):
        # ...
        # Create random masks
        input_mask = ops.cast(
            ops.random_uniform((self.input_dim,)) >= self.sparsity_level,
            ops.int32
        )
        recurrent_mask = EmberTensor.cast(
            EmberTensor.random_uniform((self.units, self.units)) >= self.sparsity_level,
            dtype='int32'
        )
        # ...
```

```python
# After
from ember_ml.nn.tensor import zeros, random_uniform, cast, array, int32
from ember_ml.nn.wirings.wiring import Wiring

class RandomWiring(Wiring):
    # ...
    def build(self, input_dim=None):
        # ...
        # Create random masks
        input_mask = cast(
            random_uniform((self.input_dim,)) >= self.sparsity_level,
            dtype=int32
        )
        recurrent_mask = cast(
            random_uniform((self.units, self.units)) >= self.sparsity_level,
            dtype=int32
        )
        # ...
```

#### 2.3.3 Fix Other Wiring Implementations

Check and fix other wiring implementations that might have similar issues, such as:

- ember_ml/nn/wirings/ncp_wiring.py
- ember_ml/nn/wirings/full_wiring.py
- Any other wiring implementations

### 2.4 Using DType Objects

With the new tensor API, we can use dtype objects directly instead of strings. This is more consistent with the PyTorch and MLX APIs and provides better type checking.

```python
# Old way (using strings)
tensor = zeros([3, 4], dtype='int32')

# New way (using dtype objects)
from ember_ml.nn.tensor import int32
tensor = zeros([3, 4], dtype=int32)
```

The dtype objects are imported directly from `ember_ml.nn.tensor`:

```python
from ember_ml.nn.tensor import (
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16
)
```

These dtype objects are callable, so they can be used to convert values to a specific dtype:

```python
# Convert a tensor to float32
tensor = cast(tensor, dtype=float32)

# Create a tensor with a specific dtype
tensor = array([1, 2, 3], dtype=float32)
```

## 3. Dynamic DType Properties

### 3.1 Current Implementation

Currently, dtypes are defined statically in `ember_ml.nn.tensor.common.dtypes`:

```python
float32 = lambda: dtypes.float32
float64 = lambda: dtypes.float64
int32 = lambda: dtypes.int32
# ... other dtypes
```

Each backend (NumPy, PyTorch, MLX) has its own DType class (NumpyDType, TorchDType, MLXDType) that provides properties for accessing backend-specific dtypes.

### 3.2 Implementation Plan

#### 3.2.1 Add `get_available_dtypes()` Method to Backend DType Classes

First, we need to add a method to each backend's DType class to get the available dtypes.

Example for NumpyDType (ember_ml/backend/numpy/tensor/dtype.py):

```python
def get_available_dtypes(self):
    """
    Get all available data types for NumPy.
    
    Returns:
        List of data type names
    """
    return [
        'float32', 'float64', 'int32', 'int64', 'bool_',
        'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16'
    ]
```

Example for TorchDType (ember_ml/backend/torch/tensor/dtype.py):

```python
def get_available_dtypes(self):
    """
    Get all available data types for PyTorch.
    
    Returns:
        List of data type names
    """
    return [
        'float32', 'float64', 'int32', 'int64', 'bool_',
        'int8', 'int16', 'uint8', 'float16', 'bfloat16'  # PyTorch has bfloat16
    ]
```

Example for MLXDType (ember_ml/backend/mlx/tensor/dtype.py):

```python
def get_available_dtypes(self):
    """
    Get all available data types for MLX.
    
    Returns:
        List of data type names
    """
    return [
        'float32', 'float16', 'int32', 'int64', 'bool_',
        'uint8', 'uint16', 'uint32'  # MLX has a different set of dtypes
    ]
```

#### 3.2.2 Add `get_available_dtypes()` Function to Common Module

Next, we need to add a function to the common module to get the available dtypes from the current backend.

Update ember_ml/nn/tensor/common/__init__.py:

```python
def get_available_dtypes():
    """
    Get all available data types from the current backend.
    
    Returns:
        List of data type names
    """
    ops = _get_tensor_ops()
    if hasattr(ops, 'get_available_dtypes'):
        return ops.get_available_dtypes()
    
    # Fallback: Try to get a list of standard dtypes
    standard_dtypes = [
        'float32', 'float64', 'int32', 'int64', 'bool_',
        'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16'
    ]
    available_dtypes = []
    for dtype_name in standard_dtypes:
        try:
            # Check if the dtype is available
            if hasattr(ops, dtype_name):
                available_dtypes.append(dtype_name)
        except (AttributeError, ImportError):
            pass
    return available_dtypes
```

#### 3.2.3 Update DTypes Module to Dynamically Create Properties

Now, we need to update the dtypes module to dynamically create properties based on the available dtypes.

Update ember_ml/nn/tensor/common/dtypes.py:

```python
"""
Data types for ember_ml.nn.tensor.

This module provides a backend-agnostic data type system that can be used
across different backends.
"""

from typing import Any, Dict, List

from ember_ml.nn.tensor.common import get_available_dtypes

# Create a class that implements the DTypeInterface
class EmberDType:
    """
    A backend-agnostic data type.
    
    This class represents data types that can be used across different backends.
    """
    
    def __init__(self, name: str):
        """
        Initialize an EmberDType.
        
        Args:
            name: The name of the data type
        """
        self.name = name
    
    def __repr__(self) -> str:
        """Return a string representation of the data type."""
        return self.name
    
    def __str__(self) -> str:
        """Return a string representation of the data type."""
        return self.name
    
    def __eq__(self, other: Any) -> bool:
        """Check if two data types are equal."""
        if isinstance(other, EmberDType):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False
    
    def __call__(self) -> Any:
        """Return the specific data type."""
        # This is a convenience method to make the EmberDType callable
        # It allows us to use EmberDType instances as functions
        # For example: float32() instead of float32
        from ember_ml.nn.tensor.common import _get_tensor_ops
        return getattr(_get_tensor_ops(), self.name)

# Define functions for data type operations
def get_dtype(name):
    """Get a data type by name."""
    from ember_ml.nn.tensor.common import _get_tensor_ops
    return _get_tensor_ops().get_dtype(name)

def to_dtype_str(dtype):
    """Convert a data type to a string representation."""
    from ember_ml.nn.tensor.common import _get_tensor_ops
    return _get_tensor_ops().to_dtype_str(dtype)

def from_dtype_str(dtype_str):
    """Convert a string representation to a data type."""
    from ember_ml.nn.tensor.common import _get_tensor_ops
    return _get_tensor_ops().from_dtype_str(dtype_str)

# Dynamically create properties for each data type
_dtype_properties: Dict[str, EmberDType] = {}
for dtype_name in get_available_dtypes():
    _dtype_properties[dtype_name] = EmberDType(dtype_name)

# Export the dynamically created properties
globals().update(_dtype_properties)

# Define a list of all available data types and functions
__all__ = [
    'EmberDType',
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
    *_dtype_properties.keys()
]
```

#### 3.2.4 Update Tensor Module to Import and Re-export Properties

Finally, we need to update the tensor module to import and re-export the dynamically created properties.

Update ember_ml/nn/tensor/__init__.py:

```python
"""
Tensor module for ember_ml.

This module provides a backend-agnostic tensor implementation that works with
any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.
"""

# Import interfaces
from ember_ml.nn.tensor.interfaces import TensorInterface  # noqa
from ember_ml.nn.tensor.interfaces.dtype import DTypeInterface  # noqa

# Import EmberTensor class
from ember_ml.nn.tensor.common import EmberTensor  # noqa

# Import module-level functions
from ember_ml.nn.tensor.common import (  # noqa
    zeros, ones, eye, arange, linspace,
    reshape, transpose, concatenate, stack, split,
    expand_dims, squeeze, tile, gather, tensor_scatter_nd_update,
    slice, slice_update, convert_to_tensor, cast, copy, var, pad,
    sort, argsort, to_numpy, item, shape, dtype,
    # ... other operations
)

# Import data types
from ember_ml.nn.tensor.common.dtypes import (  # noqa
    EmberDType, get_dtype, to_dtype_str, from_dtype_str
)

# Import all available dtypes
from ember_ml.nn.tensor.common.dtypes import __all__ as _dtype_all
_dtype_names = [name for name in _dtype_all if name not in ['EmberDType', 'get_dtype', 'to_dtype_str', 'from_dtype_str']]
for _dtype_name in _dtype_names:
    globals()[_dtype_name] = getattr(ember_ml.nn.tensor.common.dtypes, _dtype_name)

# Export all classes and functions
__all__ = [
    # Interfaces
    'TensorInterface',
    'DTypeInterface',
    
    # Implementations
    'EmberTensor',
    'EmberDType',
    
    # Module-level functions
    'zeros', 'ones', 'eye', 'arange', 'linspace',
    'reshape', 'transpose', 'concatenate', 'stack', 'split',
    'expand_dims', 'squeeze', 'tile', 'gather', 'tensor_scatter_nd_update',
    'slice', 'slice_update', 'convert_to_tensor', 'cast', 'copy', 'var', 'pad',
    'sort', 'argsort', 'to_numpy', 'item', 'shape', 'dtype',
    # ... other operations
    
    # Data types
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
    *_dtype_names,
]
```

### 3.3 Benefits of This Approach

1. **Backend Purity**: The frontend remains completely agnostic of backend implementation details
2. **Automatic Support for New DTypes**: When new data types become available in backends, they will be automatically exposed
3. **Environment-Specific Optimization**: Only expose data types that are actually supported in the current environment
4. **Reduced Maintenance**: No need to manually update the list of data types when new ones are added
5. **Consistency**: Ensures that the `__all__` list and the actual exported properties are always in sync

## 4. Testing Strategy

### 4.1 Frontend Compatibility Testing

1. **Unit Tests**: Create unit tests for each frontend function and method, verifying that they work correctly with both the old and new backend structures.
2. **Integration Tests**: Create integration tests that use the frontend with each backend, verifying end-to-end functionality.
3. **Regression Tests**: Run existing tests to ensure that the refactoring doesn't break existing functionality.
4. **Performance Tests**: Benchmark critical operations to compare the performance of the old and new approaches.

### 4.2 Wiring Module Testing

1. **Unit Tests**: Add unit tests for the wiring module to ensure it works correctly with the new tensor API
2. **Integration Tests**: Add integration tests to verify that the wiring module works correctly with different backends
3. **Edge Cases**: Test edge cases such as switching backends at runtime to ensure the wiring module works correctly

### 4.3 Dynamic DType Properties Testing

1. **Unit Tests**: Add unit tests for the `get_available_dtypes()` function to ensure it returns the expected dtypes for each backend
2. **Integration Tests**: Add integration tests to verify that the dynamically created properties work correctly with different backends
3. **Edge Cases**: Test edge cases such as switching backends at runtime to ensure the dtypes are updated correctly

## 5. Implementation Steps Summary

1. Implement the frontend compatibility layer
2. Fix the wiring module to use the new tensor API
3. Implement dynamic dtype properties
4. Add tests to verify all implementations
5. Update documentation to explain the changes