# Tensor Operations Implementation Guide

This guide provides a comprehensive overview of tensor operations implementation in the Ember ML framework, including the core architecture and backend-specific considerations.

## Directory Structure

Each backend should follow this directory structure:

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

## Implementation Approach

The implementation follows a function-first architecture:

1. **Standalone Functions**: Each operation is implemented as a standalone function first
2. **Method Wrappers**: The tensor class provides methods that are thin wrappers around these functions
3. **Consistent API**: The API is consistent across all backends
4. **Backend Purity**: Backend-specific details are hidden from the user

## Implementation Steps

### 1. Function Implementation

Each operation should be implemented as a standalone function in the appropriate module:

```python
def operation_name(tensor_obj, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        tensor_obj: The tensor object (instance of the backend's tensor class)
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    # Implementation
    return result
```

### 2. Method Implementation

Each method in the tensor class should be a thin wrapper that calls the standalone function:

```python
def operation_name(self, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    from ember_ml.backend.{backend_name}.tensor.ops.{module} import operation_name as op_func
    return op_func(self, *args, **kwargs)
```

### 3. Export Operations

The `__init__.py` files should be set up to allow both function and method calling patterns.

## Operation Categories

### Casting Operations
- `cast(tensor_obj, tensor, dtype)`: Cast a tensor to a different data type

### Creation Operations
- `zeros(tensor_obj, shape, dtype=None, device=None)`: Create a tensor of zeros
- `ones(tensor_obj, shape, dtype=None, device=None)`: Create a tensor of ones
- `eye(tensor_obj, n, m=None, dtype=None, device=None)`: Create an identity matrix
- `zeros_like(tensor_obj, tensor, dtype=None, device=None)`: Create a tensor of zeros with the same shape as the input
- `ones_like(tensor_obj, tensor, dtype=None, device=None)`: Create a tensor of ones with the same shape as the input
- `full(tensor_obj, shape, fill_value, dtype=None, device=None)`: Create a tensor filled with a scalar value
- `full_like(tensor_obj, tensor, fill_value, dtype=None, device=None)`: Create a tensor filled with a scalar value with the same shape as the input
- `arange(tensor_obj, start, stop=None, step=1, dtype=None, device=None)`: Create a tensor with evenly spaced values within a given interval
- `linspace(tensor_obj, start, stop, num, dtype=None, device=None)`: Create a tensor with evenly spaced values within a given interval

### Manipulation Operations
- `reshape(tensor_obj, tensor, shape)`: Reshape a tensor
- `transpose(tensor_obj, tensor, axes=None)`: Transpose a tensor
- `concatenate(tensor_obj, tensors, axis=0)`: Concatenate tensors along a specified axis
- `stack(tensor_obj, tensors, axis=0)`: Stack tensors along a new axis
- `split(tensor_obj, tensor, num_or_size_splits, axis=0)`: Split a tensor into sub-tensors
- `expand_dims(tensor_obj, tensor, axis)`: Insert a new axis into a tensor's shape
- `squeeze(tensor_obj, tensor, axis=None)`: Remove single-dimensional entries from a tensor's shape
- `tile(tensor_obj, tensor, reps)`: Construct a tensor by tiling a given tensor
- `pad(tensor_obj, tensor, paddings, constant_values=0)`: Pad a tensor with a constant value

### Indexing Operations
- `slice_tensor(tensor_obj, tensor, starts, sizes)`: Extract a slice from a tensor
- `slice_update(tensor_obj, tensor, slices, updates)`: Update a tensor at specific indices
- `gather(tensor_obj, tensor, indices, axis=0)`: Gather slices from a tensor along an axis
- `tensor_scatter_nd_update(tensor_obj, tensor, indices, updates)`: Updates values of a tensor at specified indices

### Utility Operations
- `convert_to_tensor(tensor_obj, data, dtype=None, device=None)`: Convert data to a tensor
- `to_numpy(tensor_obj, tensor)`: Convert a tensor to a NumPy array
- `item(tensor_obj, tensor)`: Get the value of a scalar tensor
- `shape(tensor_obj, tensor)`: Get the shape of a tensor
- `dtype(tensor_obj, tensor)`: Get the data type of a tensor
- `copy(tensor_obj, tensor)`: Create a copy of a tensor
- `var(tensor_obj, tensor, axis=None, keepdims=False)`: Compute the variance of a tensor along specified axes
- `sort(tensor_obj, tensor, axis=-1, descending=False)`: Sort a tensor along a specified axis
- `argsort(tensor_obj, tensor, axis=-1, descending=False)`: Return the indices that would sort a tensor along a specified axis
- `maximum(tensor_obj, x, y)`: Element-wise maximum of two tensors

### Random Operations
- `random_normal(tensor_obj, shape, mean=0.0, stddev=1.0, dtype=None, device=None)`: Create a tensor with random values from a normal distribution
- `random_uniform(tensor_obj, shape, minval=0.0, maxval=1.0, dtype=None, device=None)`: Create a tensor with random values from a uniform distribution
- `random_binomial(tensor_obj, shape, p=0.5, dtype=None, device=None)`: Create a tensor with random values from a binomial distribution
- `random_gamma(tensor_obj, shape, alpha=1.0, beta=1.0, dtype=None, device=None)`: Generate random values from a gamma distribution
- `random_exponential(tensor_obj, shape, scale=1.0, dtype=None, device=None)`: Generate random values from an exponential distribution
- `random_poisson(tensor_obj, shape, lam=1.0, dtype=None, device=None)`: Generate random values from a Poisson distribution
- `random_categorical(tensor_obj, logits, num_samples, dtype=None, device=None)`: Draw samples from a categorical distribution
- `random_permutation(tensor_obj, x, dtype=None, device=None)`: Generate a random permutation
- `shuffle(tensor_obj, x)`: Randomly shuffle a tensor along the first dimension
- `set_seed(tensor_obj, seed)`: Set the random seed for reproducibility
- `get_seed(tensor_obj)`: Get the current random seed

## Backend-Specific Considerations

### MLX Backend

The MLX backend has some specific considerations:

1. **Data Types**: MLX supports a different set of data types than other backends
2. **Device Handling**: MLX has specific device handling for Apple Silicon
3. **Function Signatures**: Some MLX functions have different signatures than their NumPy/PyTorch counterparts

### NumPy Backend

The NumPy backend has some specific considerations:

1. **No GPU Support**: NumPy doesn't have native GPU support
2. **Array vs Tensor**: NumPy uses arrays instead of tensors
3. **Function Names**: Some NumPy functions have different names than their PyTorch/MLX counterparts

### PyTorch Backend

The PyTorch backend has some specific considerations:

1. **Autograd**: PyTorch has built-in autograd support
2. **Device Handling**: PyTorch has specific device handling for CUDA
3. **In-place Operations**: PyTorch has in-place operations that modify tensors

## Backend Consistency

To ensure consistency across backends, all operations should be implemented for all backends. If an operation is not natively supported by a backend, it should be implemented using other operations.

## Backend Purity

To maintain backend purity, follow these guidelines:

1. **No Direct Imports**: Don't import backend-specific libraries in the frontend code
2. **Use Tensor/* Signatures**: Use tensor/* signatures instead of direct backend-specific dtype references
3. **Consistent API**: Maintain a consistent API across all backends
4. **Proper Abstraction**: Hide backend-specific details from the user

## Testing

Each operation should be tested to ensure it works correctly. Tests should cover:

1. **Function Calling**: Test calling the operation as a standalone function
2. **Method Calling**: Test calling the operation as a method on a tensor class
3. **Edge Cases**: Test edge cases such as empty tensors, scalar tensors, etc.
4. **Backend Consistency**: Test that the operation works consistently across all backends

## Documentation

Each operation should be well-documented with:

1. **Function Documentation**: Document the function's purpose, parameters, and return value
2. **Method Documentation**: Document the method's purpose, parameters, and return value
3. **Examples**: Provide examples of how to use the operation
4. **Edge Cases**: Document any edge cases or limitations