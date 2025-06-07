# Tensor Module

The `ember_ml.nn.tensor` module provides a backend-agnostic tensor implementation that works with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.

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

## EmberTensor

The `EmberTensor` class is a backend-agnostic tensor implementation that delegates operations to the current backend. It provides a consistent API for tensor operations across different backends.

### Creating Tensors

```python
from ember_ml.nn.tensor import EmberTensor

# Create a tensor from a list
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Create a tensor with a specific data type
from ember_ml.nn.tensor import float32
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]], dtype=float32)

# Create a tensor on a specific device
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]], device='cuda')

# Create a tensor that requires gradients
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
```

### Tensor Properties

```python
# Get the shape of a tensor
shape = tensor.shape  # (2, 3)

# Get the data type of a tensor
dtype = tensor.dtype  # int64

# Get the device of a tensor
device = tensor.device  # 'cpu' or 'cuda' or 'mps'

# Check if a tensor requires gradients
requires_grad = tensor.requires_grad  # False
```

### Tensor Operations

```python
# Create a tensor of zeros
zeros_tensor = tensor.zeros((2, 3))

# Create a tensor of ones
ones_tensor = tensor.ones((2, 3))

# Reshape a tensor
reshaped_tensor = tensor.reshape(tensor, (3, 2))

# Transpose a tensor
transposed_tensor = tensor.transpose(tensor)

# Concatenate tensors
concat_tensor = tensor.concatenate([tensor, tensor], axis=0)

# Stack tensors
stacked_tensor = tensor.stack([tensor, tensor], axis=0)

# Split a tensor
split_tensors = tensor.split(tensor, 3, axis=1)
```

## EmberDType

The `EmberDType` class is a backend-agnostic data type class that represents data types across different backends.

### Available Data Types

```python
from ember_ml.nn.tensor import float32, float64, int32, int64, bool_

# Create a tensor with a specific data type
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]], dtype=float32)

# Cast a tensor to a different data type
float_tensor = tensor.cast(tensor, float32)
```

## Device Support

The tensor module supports different devices depending on the backend:

- **NumPy**: CPU only
- **PyTorch**: CPU, CUDA (NVIDIA GPUs), MPS (Apple Silicon)
- **MLX**: CPU, Metal (Apple Silicon)

```python
# Create a tensor on a specific device
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]], device='cuda')  # NVIDIA GPU
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]], device='mps')   # Apple Silicon GPU
```

## Backend Selection

The tensor module uses the current backend selected by the `ember_ml.backend` module. You can change the backend using the `set_backend` function:

```python
from ember_ml.backend import set_backend

# Set the backend to PyTorch
set_backend('torch')

# Set the backend to NumPy
set_backend('numpy')

# Set the backend to MLX
set_backend('mlx')
```

## Backend Purity

The tensor module maintains backend purity by ensuring that all tensor operations go through the backend abstraction layer. This means that you can use the same code with different backends without having to change your code.

For example, the following code will work with any backend:

```python
from ember_ml.nn.tensor import EmberTensor

# Create a tensor
tensor = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Reshape the tensor
reshaped = tensor.reshape(tensor, (3, 2))

# Print the result
print(reshaped)
```

This code will work with NumPy, PyTorch, or MLX backends without any changes.

## Implementation Guide

For developers who want to implement new tensor operations or modify existing ones, see the [Tensor Implementation Guide](../tensor_impl_guide.md) document.