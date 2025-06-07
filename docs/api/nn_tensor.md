# Tensor Module (nn.tensor)

The `ember_ml.nn.tensor` module provides a backend-agnostic tensor implementation that works with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer. This module is the foundation for tensor operations in Ember ML.

## Overview

The tensor module is designed to provide a consistent API for tensor operations across different backends. It consists of the following components:

- `EmberTensor`: A backend-agnostic tensor class that delegates operations to the current backend
- `EmberDType`: A backend-agnostic data type class that represents data types across different backends
- Common tensor operations: Creation, manipulation, and conversion functions

## Architecture

The tensor module follows the backend abstraction architecture of Ember ML:

1. **Frontend Abstractions**: The `ember_ml.nn.tensor` module provides abstract interfaces and common implementations
2. **Backend Implementations**: The actual implementations reside in the backend directory, with specific implementations for each supported backend (NumPy, PyTorch, MLX)
3. **Dispatch Mechanism**: The frontend abstractions dispatch calls to the appropriate backend implementation based on the currently selected backend

For a detailed explanation of the tensor operations architecture, see the [Tensor Operations Architecture](tensor_architecture.md) document.

### Backend Folder Structure

Each backend has a `tensor` subfolder that contains the backend-specific tensor and data type implementations:

```
ember_ml/backend/
├── numpy/
│   ├── tensor/
│   │   ├── tensor.py  # NumPy tensor implementation
│   │   ├── dtype.py   # NumPy data type implementation
│   │   ├── ops/       # NumPy tensor operations
│   │   │   ├── casting.py
│   │   │   ├── creation.py
│   │   │   ├── manipulation.py
│   │   │   ├── indexing.py
│   │   │   ├── utility.py
│   │   │   └── random.py
├── torch/
│   ├── tensor/
│   │   ├── tensor.py  # PyTorch tensor implementation
│   │   ├── dtype.py   # PyTorch data type implementation
│   │   ├── ops/       # PyTorch tensor operations
│   │   │   ├── casting.py
│   │   │   ├── creation.py
│   │   │   ├── manipulation.py
│   │   │   ├── indexing.py
│   │   │   ├── utility.py
│   │   │   └── random.py
├── mlx/
│   ├── tensor/
│   │   ├── tensor.py  # MLX tensor implementation
│   │   ├── dtype.py   # MLX data type implementation
│   │   ├── ops/       # MLX tensor operations
│   │   │   ├── casting.py
│   │   │   ├── creation.py
│   │   │   ├── manipulation.py
│   │   │   ├── indexing.py
│   │   │   ├── utility.py
│   │   │   └── random.py
```

## Function-First Design

The tensor operations in Ember ML follow a function-first design pattern, where each operation is implemented as a standalone function that can be called directly or through a method on a tensor class.

For example, the `cast()` operation can be called in two ways:

```python
# As a standalone function
from ember_ml.nn.tensor import cast
result = cast(tensor, dtype)

# As a method on EmberTensor
result = tensor.cast(dtype)
```

This design provides flexibility and consistency across the framework. For more details, see the [Tensor Operations Architecture](tensor_architecture.md) document.

## Importing

```python
from ember_ml.nn import tensor
```

## Core Classes

### EmberTensor

`EmberTensor` is the primary tensor class in Ember ML, providing a consistent API across different backends.

```python
# Create a tensor
x = tensor.EmberTensor([1, 2, 3])

# Access properties
print(x.shape)  # (3,)
print(x.dtype)  # int64
print(x.device)  # 'cpu'
```

### EmberDType

`EmberDType` represents data types across different backends.

```python
# Access data types
from ember_ml.nn.tensor import float32, int32, bool_

# Create a tensor with a specific data type
x = tensor.EmberTensor([1, 2, 3], dtype=float32)
```

## Tensor Creation

| Function | Description |
|----------|-------------|
| `tensor.array(data, dtype=None, device=None, requires_grad=False)` | Create a tensor from data |
| `tensor.convert_to_tensor(data, dtype=None, device=None, requires_grad=False)` | Convert data to a tensor |
| `tensor.zeros(shape, dtype=None, device=None)` | Create a tensor of zeros |
| `tensor.ones(shape, dtype=None, device=None)` | Create a tensor of ones |
| `tensor.eye(n, m=None, dtype=None, device=None)` | Create an identity matrix |
| `tensor.arange(start, stop=None, step=1, dtype=None, device=None)` | Create a tensor with evenly spaced values |
| `tensor.linspace(start, stop, num, dtype=None, device=None)` | Create a tensor with linearly spaced values |
| `tensor.zeros_like(x, dtype=None, device=None)` | Create a tensor of zeros with the same shape as x |
| `tensor.ones_like(x, dtype=None, device=None)` | Create a tensor of ones with the same shape as x |
| `tensor.full(shape, fill_value, dtype=None, device=None)` | Create a tensor filled with a scalar value |
| `tensor.full_like(x, fill_value, dtype=None, device=None)` | Create a tensor filled with a scalar value with the same shape as x |

## Tensor Manipulation

| Function | Description |
|----------|-------------|
| `tensor.reshape(x, shape)` | Reshape a tensor to a new shape |
| `tensor.transpose(x, axes=None)` | Permute the dimensions of a tensor |
| `tensor.concatenate(tensors, axis=0)` | Concatenate tensors along an axis |
| `tensor.stack(tensors, axis=0)` | Stack tensors along a new axis |
| `tensor.split(x, num_or_size_splits, axis=0)` | Split a tensor into sub-tensors |
| `tensor.expand_dims(x, axis)` | Expand the shape of a tensor |
| `tensor.squeeze(x, axis=None)` | Remove dimensions of size 1 |
| `tensor.tile(x, reps)` | Construct a tensor by tiling a given tensor |
| `tensor.gather(params, indices, axis=0)` | Gather slices from params according to indices |
| `tensor.scatter(indices, updates, shape)` | Scatter updates into a new tensor according to indices |
| `tensor.tensor_scatter_nd_update(tensor, indices, updates)` | Update tensor values by scattering updates |
| `tensor.slice(x, begin, size)` | Extract a slice from a tensor |
| `tensor.slice_update(x, begin, updates)` | Update a slice of a tensor |
| `tensor.pad(x, paddings, mode='constant', constant_values=0)` | Pad a tensor |

## Indexing and Slicing

Ember ML provides two ways to index and slice tensors:

### Bracket Notation (Recommended)

The recommended way to index and slice tensors is using bracket notation, which is more intuitive and consistent with Python's standard indexing syntax:

```python
from ember_ml.nn.tensor import EmberTensor

# Create a tensor
a = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Get a single element
element = a[0, 1]  # 2

# Get a row
row = a[0]  # [1, 2, 3]

# Get a column
column = a[:, 1]  # [2, 5]

# Get a slice
slice_a = a[0:1, 1:3]  # [[2, 3]]

# Set values
a[0, 1] = 10  # a is now [[1, 10, 3], [4, 5, 6]]

# Set a row
a[0, :] = [7, 8, 9]  # a is now [[7, 8, 9], [4, 5, 6]]

# Set a column
a[:, 1] = [10, 11]  # a is now [[7, 10, 9], [4, 11, 6]]
```

### Index Update API (Alternative)

Alternatively, you can use the `index_update` function with the `index` object, but note that this approach requires using the `value` keyword argument:

```python
from ember_ml.nn.tensor import EmberTensor, index, index_update

# Create a tensor
a = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Update a single element
a = index_update(a, index[0, 1], value=10)  # a is now [[1, 10, 3], [4, 5, 6]]

# Update a row
a = index_update(a, index[0, :], value=[7, 8, 9])  # a is now [[7, 8, 9], [4, 5, 6]]

# Update a column
a = index_update(a, index[:, 1], value=[10, 11])  # a is now [[7, 10, 9], [4, 11, 6]]
```

**Important Note**: When using `index_update`, you must use the `value` keyword argument. Passing the value as a positional argument will result in an error.

## Random Operations

| Function | Description |
|----------|-------------|
| `tensor.random_uniform(shape, minval=0, maxval=1, dtype=None, device=None, seed=None)` | Generate random values from a uniform distribution |
| `tensor.random_normal(shape, mean=0, stddev=1, dtype=None, device=None, seed=None)` | Generate random values from a normal distribution |
| `tensor.random_bernoulli(shape, p=0.5, dtype=None, device=None, seed=None)` | Generate random values from a Bernoulli distribution |
| `tensor.random_gamma(shape, alpha, beta=1.0, dtype=None, device=None, seed=None)` | Generate random values from a gamma distribution |
| `tensor.random_exponential(shape, scale=1.0, dtype=None, device=None, seed=None)` | Generate random values from an exponential distribution |
| `tensor.random_poisson(shape, lam, dtype=None, device=None, seed=None)` | Generate random values from a Poisson distribution |
| `tensor.random_categorical(logits, num_samples, dtype=None, device=None, seed=None)` | Draw samples from a categorical distribution |
| `tensor.random_permutation(x, seed=None)` | Randomly permute a sequence |
| `tensor.shuffle(x, axis=0, seed=None)` | Randomly shuffle a tensor along an axis |
| `tensor.set_seed(seed)` | Set the random seed |
| `tensor.get_seed()` | Get the current random seed |

## Data Types

The following data types are available in the tensor module:

| Data Type | Description |
|-----------|-------------|
| `tensor.float32` | 32-bit floating-point |
| `tensor.float64` | 64-bit floating-point |
| `tensor.int32` | 32-bit signed integer |
| `tensor.int64` | 64-bit signed integer |
| `tensor.bool_` | Boolean |
| `tensor.int8` | 8-bit signed integer |
| `tensor.int16` | 16-bit signed integer |
| `tensor.uint8` | 8-bit unsigned integer |
| `tensor.uint16` | 16-bit unsigned integer |
| `tensor.uint32` | 32-bit unsigned integer |
| `tensor.uint64` | 64-bit unsigned integer |
| `tensor.float16` | 16-bit floating-point |

## Data Type Operations

| Function | Description |
|----------|-------------|
| `tensor.get_dtype(x)` | Get the data type of a tensor |
| `tensor.to_dtype_str(dtype)` | Convert a data type to a string |
| `tensor.from_dtype_str(dtype_str)` | Convert a string to a data type |

## Examples

### Creating Tensors

```python
from ember_ml.nn import tensor

# Create tensors
x = tensor.convert_to_tensor([1, 2, 3])
y = tensor.zeros((3, 3))
z = tensor.ones((2, 2))
a = tensor.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
b = tensor.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
```

### Manipulating Tensors

```python
from ember_ml.nn import tensor

# Create a tensor
x = tensor.convert_to_tensor([[1, 2], [3, 4]])

# Reshape
y = tensor.reshape(x, (4,))  # [1, 2, 3, 4]

# Transpose
z = tensor.transpose(x)  # [[1, 3], [2, 4]]

# Concatenate
a = tensor.concatenate([x, x], axis=0)  # [[1, 2], [3, 4], [1, 2], [3, 4]]
b = tensor.concatenate([x, x], axis=1)  # [[1, 2, 1, 2], [3, 4, 3, 4]]

# Stack
c = tensor.stack([x, x], axis=0)  # [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
```

### Random Operations

```python
from ember_ml.nn import tensor

# Set seed for reproducibility
tensor.set_seed(42)

# Generate random tensors
a = tensor.random_uniform((3, 3))
b = tensor.random_normal((3, 3), mean=0, stddev=1)
c = tensor.random_bernoulli((3, 3), p=0.7)

# Shuffle a tensor
x = tensor.convert_to_tensor([1, 2, 3, 4, 5])
shuffled = tensor.shuffle(x)
```

### Working with Different Backends

```python
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

# Create a tensor with NumPy backend
set_backend('numpy')
x_numpy = tensor.convert_to_tensor([1, 2, 3])

# Create a tensor with PyTorch backend
set_backend('torch')
x_torch = tensor.convert_to_tensor([1, 2, 3])

# Create a tensor with MLX backend
set_backend('mlx')
x_mlx = tensor.convert_to_tensor([1, 2, 3])
```

## Backend Purity

The tensor module maintains backend purity by ensuring that all tensor operations go through the backend abstraction layer via the `ops` module. This means that you can use the same code with different backends without having to change your code.

**Important Note on Return Types:** Functions within the `ops` module (e.g., `ops.add`, `ops.matmul`) operate on and return **native backend tensors** (like `mlx.core.array`, `torch.Tensor`, or `numpy.ndarray`), not `EmberTensor` instances. The `EmberTensor` class serves as a wrapper primarily for user interaction and parameter management. When combining results from `ops` functions with `EmberTensor` or `Parameter` objects, ensure all inputs to subsequent `ops` calls are handled correctly by the backend's conversion utilities (which typically accept native tensors, `EmberTensor`, and `Parameter` objects). Avoid mixing types with direct Python operators (`+`, `*`).

For example, the following code will work with any backend:

```python
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

# Function that works with any backend
def process_tensor(x):
    y = tensor.reshape(x, (-1,))
    z = tensor.random_normal(tensor.shape(y))
    return tensor.concatenate([y, z], axis=0)

# Use with NumPy backend
set_backend('numpy')
x_numpy = tensor.convert_to_tensor([1, 2, 3, 4])
result_numpy = process_tensor(x_numpy)

# Use with PyTorch backend
set_backend('torch')
x_torch = tensor.convert_to_tensor([1, 2, 3, 4])
result_torch = process_tensor(x_torch)
```

## Device Support

The tensor module supports different devices depending on the backend:

- **NumPy**: CPU only
- **PyTorch**: CPU, CUDA (NVIDIA GPUs), MPS (Apple Silicon)
- **MLX**: CPU, Metal (Apple Silicon)

```python
from ember_ml.nn import tensor

# Create a tensor on CPU
x_cpu = tensor.convert_to_tensor([1, 2, 3], device='cpu')

# Create a tensor on CUDA (if available)
x_cuda = tensor.convert_to_tensor([1, 2, 3], device='cuda')

# Create a tensor on MPS (if available)
x_mps = tensor.convert_to_tensor([1, 2, 3], device='mps')
```

## Backend Selection

The tensor module uses the current backend selected by the `ember_ml.backend` module. You can change the backend using the `set_backend` function:

```python
from ember_ml.ops import set_backend

# Set the backend to PyTorch
set_backend('torch')

# Set the backend to NumPy
set_backend('numpy')

# Set the backend to MLX
set_backend('mlx')
```

## Implementation Details

The tensor module is implemented using a layered architecture:

1. **Frontend Abstractions**: The `tensor` module provides abstract interfaces and common implementations
2. **Backend Implementations**: The actual implementations reside in the backend directory, with specific implementations for each supported backend
3. **Dispatch Mechanism**: The frontend abstractions dispatch calls to the appropriate backend implementation based on the currently selected backend

This architecture allows Ember ML to provide a consistent API across different backends while still leveraging the unique capabilities of each backend.

## Implementation Guide

For developers who want to implement new tensor operations or modify existing ones, see the [Tensor Implementation Guide](../tensor_impl_guide.md) document.