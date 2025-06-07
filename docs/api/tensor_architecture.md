# Tensor Operations Architecture

This document describes the architecture and design principles of tensor operations in the Ember ML framework.

## Overview

The Ember ML framework provides a flexible and backend-agnostic approach to tensor operations. The architecture follows a function-first design pattern, where each tensor operation is implemented as a standalone function that can be called directly or through a method on a tensor class.

## Architecture

### Data Flow and Component Roles

The tensor architecture consists of three main components:

1. **EmberTensor**: The frontend tensor class that users interact with
   - Stores actual tensor data in the `_tensor` instance variable
   - Delegates operations to the backend implementation
   - Provides a consistent API regardless of the backend

2. **Backend Tensor Classes** (MLXTensor, NumPyTensor, TorchTensor):
   - Service classes that provide tensor operations
   - Do NOT store tensor data themselves
   - Delegate to specialized operation modules in the ops directory

3. **Operation Functions**:
   - Standalone functions that implement tensor operations
   - Take a tensor object (backend tensor class instance) as the first argument
   - Perform the actual operation using the backend-specific implementation

The data flow is as follows:
1. User creates an EmberTensor or calls an operation
2. EmberTensor delegates to a standalone function
3. The standalone function calls the appropriate backend implementation
4. The backend implementation performs the operation using the backend-specific library
5. The result is returned to the user

### Backend Implementation

Each backend (MLX, NumPy, PyTorch) implements tensor operations following this structure:

```
ember_ml/backend/{backend_name}/tensor/
  ├── __init__.py           # Exports the tensor class and operations
  ├── tensor.py             # Contains the tensor class with method interfaces
  ├── dtype.py              # Contains the data type class
  ├── ops/                  # Directory for operation modules
  │   ├── __init__.py       # Exports all operations
  │   ├── casting.py        # Contains cast() and related functions
  │   ├── creation.py       # Contains zeros(), ones(), etc.
  │   ├── manipulation.py   # Contains reshape(), transpose(), etc.
  │   ├── indexing.py       # Contains slice(), gather(), etc.
  │   ├── utility.py        # Contains convert_to_tensor(), to_numpy(), etc.
  │   └── random.py         # Contains random_normal(), random_uniform(), etc.
```

### Frontend Implementation

The frontend provides a unified interface through `ember_ml.nn.tensor`:

- Standalone functions in `ember_ml.nn.tensor` delegate to the current backend's tensor operations
- The `EmberTensor` class provides methods that call these standalone functions
- Users can call operations either as standalone functions (e.g., `cast(tensor, dtype)`) or as methods on `EmberTensor` instances (e.g., `tensor.cast(dtype)`)

## Type System and Backend Selection

### Type System

The Ember ML framework uses a comprehensive type system to ensure type safety and backend compatibility:

1. **EmberDType**: A backend-agnostic data type class
   - Represents data types across different backends
   - Handles conversion between backend-specific types
   - Provides consistent type names regardless of backend

2. **Backend-Specific DTypes**: Each backend has its own data type implementation
   - MLXDType for MLX
   - NumPyDType for NumPy
   - TorchDType for PyTorch

3. **Type Conversion**:
   - The `convert_to_tensor` function handles type conversion
   - The `cast` function changes the data type of a tensor
   - Type validation ensures compatibility across backends

### Backend Selection

Backend selection is managed through the `ember_ml.backend` module:

1. **Selection Mechanism**:
   - `set_backend(name)`: Sets the active backend
   - `get_backend()`: Gets the current backend
   - `get_device()`: Gets the current device

2. **Backend Switching**:
   - Backend can be switched at runtime
   - Tensors are automatically converted to the new backend
   - Operations use the currently selected backend

3. **Implementation Requirements**:
   - All backends must implement the same API
   - Type conversion must be handled properly
   - Device management must be consistent

For proper typing to work correctly:

1. All function signatures must use consistent parameter names
2. Type annotations must be used consistently
3. Backend-specific types must be properly converted
4. The EmberDType system must handle all backend-specific types

## Design Principles

### 1. Function-First Approach

Each tensor operation is implemented as a standalone function first, with the class methods being thin wrappers that call these functions.

```python
# In ember_ml/backend/{backend_name}/tensor/ops/casting.py
def cast(tensor_obj, data, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        tensor_obj: The tensor object (instance of the backend's tensor class)
        data: The tensor to cast
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    # Implementation
    return result
```

### 2. Method as Passthrough

The class methods are thin wrappers that call the standalone functions:

```python
# In ember_ml/backend/{backend_name}/tensor/tensor.py
def cast(self, data, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        data: The tensor to cast
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    from ember_ml.backend.{backend_name}.tensor.ops.casting import cast as cast_func
    return cast_func(self, data, dtype)
```

### 3. Self as First Argument

When called as a method, the class instance (`self`) is passed as the first argument to the function. This allows the function to access the tensor class's methods and properties.

### 4. Consistent API

The function and method interfaces maintain the same parameter order and names, ensuring a consistent API across the framework.

### 5. Backend Purity

Each backend implementation is kept pure, with no direct imports of backend-specific libraries in the frontend code. This ensures that the framework remains backend-agnostic and can easily switch between backends.

### 6. Proper Abstraction

The framework uses proper abstraction layers to hide backend-specific details from the user. This includes:

- Using tensor/* signatures instead of direct backend-specific dtype references
- Implementing missing operations across all backends to ensure consistency
- Providing a unified API through the frontend

## Operation Categories

Tensor operations are organized into the following categories:

### Casting Operations

Operations for converting tensors to different data types:

- `cast(tensor, dtype)`: Cast a tensor to a different data type

### Creation Operations

Operations for creating new tensors:

- `zeros(shape, dtype=None, device=None)`: Create a tensor of zeros
- `ones(shape, dtype=None, device=None)`: Create a tensor of ones
- `eye(n, m=None, dtype=None, device=None)`: Create an identity matrix
- `zeros_like(tensor, dtype=None, device=None)`: Create a tensor of zeros with the same shape as the input
- `ones_like(tensor, dtype=None, device=None)`: Create a tensor of ones with the same shape as the input
- `full(shape, fill_value, dtype=None, device=None)`: Create a tensor filled with a scalar value
- `full_like(tensor, fill_value, dtype=None, device=None)`: Create a tensor filled with a scalar value with the same shape as the input
- `arange(start, stop=None, step=1, dtype=None, device=None)`: Create a tensor with evenly spaced values within a given interval
- `linspace(start, stop, num, dtype=None, device=None)`: Create a tensor with evenly spaced values within a given interval

### Manipulation Operations

Operations for manipulating tensor shapes and dimensions:

- `reshape(tensor, shape)`: Reshape a tensor
- `transpose(tensor, axes=None)`: Transpose a tensor
- `concatenate(tensors, axis=0)`: Concatenate tensors along a specified axis
- `stack(tensors, axis=0)`: Stack tensors along a new axis
- `split(tensor, num_or_size_splits, axis=0)`: Split a tensor into sub-tensors
- `expand_dims(tensor, axis)`: Insert a new axis into a tensor's shape
- `squeeze(tensor, axis=None)`: Remove single-dimensional entries from a tensor's shape
- `tile(tensor, reps)`: Construct a tensor by tiling a given tensor
- `pad(tensor, paddings, constant_values=0)`: Pad a tensor with a constant value

### Indexing Operations

Operations for indexing and slicing tensors:

- `slice_tensor(tensor, starts, sizes)`: Extract a slice from a tensor
- `slice_update(tensor, slices, updates)`: Update a tensor at specific indices
- `gather(tensor, indices, axis=0)`: Gather slices from a tensor along an axis
- `tensor_scatter_nd_update(tensor, indices, updates)`: Updates values of a tensor at specified indices

### Utility Operations

Utility operations for working with tensors:

- `convert_to_tensor(data, dtype=None, device=None)`: Convert data to a tensor
- `to_numpy(tensor)`: Convert a tensor to a NumPy array
- `item(tensor)`: Get the value of a scalar tensor
- `shape(tensor)`: Get the shape of a tensor
- `dtype(tensor)`: Get the data type of a tensor
- `copy(tensor)`: Create a copy of a tensor
- `var(tensor, axis=None, keepdims=False)`: Compute the variance of a tensor along specified axes
- `sort(tensor, axis=-1, descending=False)`: Sort a tensor along a specified axis
- `argsort(tensor, axis=-1, descending=False)`: Return the indices that would sort a tensor along a specified axis
- `maximum(x, y)`: Element-wise maximum of two tensors

### Random Operations

Operations for generating random tensors:

- `random_normal(shape, mean=0.0, stddev=1.0, dtype=None, device=None)`: Create a tensor with random values from a normal distribution
- `random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, device=None)`: Create a tensor with random values from a uniform distribution
- `random_binomial(shape, p=0.5, dtype=None, device=None)`: Create a tensor with random values from a binomial distribution
- `random_gamma(shape, alpha=1.0, beta=1.0, dtype=None, device=None)`: Generate random values from a gamma distribution
- `random_exponential(shape, scale=1.0, dtype=None, device=None)`: Generate random values from an exponential distribution
- `random_poisson(shape, lam=1.0, dtype=None, device=None)`: Generate random values from a Poisson distribution
- `random_categorical(logits, num_samples, dtype=None, device=None)`: Draw samples from a categorical distribution
- `random_permutation(x, dtype=None, device=None)`: Generate a random permutation
- `shuffle(x)`: Randomly shuffle a tensor along the first dimension
- `set_seed(seed)`: Set the random seed for reproducibility
- `get_seed()`: Get the current random seed

## Benefits

1. **Flexibility**: Operations can be called as functions or methods
2. **Consistency**: Unified implementation for each operation
3. **Maintainability**: Easier to add new operations or modify existing ones
4. **Testability**: Functions can be tested independently
5. **Discoverability**: Better IDE support for discovering available operations
6. **Backend Agnosticism**: The framework can easily switch between backends without affecting the user code