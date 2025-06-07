# ember_ml Backend

This directory contains the backend implementations for ember_ml, providing a unified interface for tensor operations across different backends (NumPy, PyTorch, MLX).

## Overview

ember_ml is designed to be backend-agnostic, meaning it can work with different tensor computation libraries without any code changes. This is achieved through the use of the `backend` module, which provides a unified interface for tensor operations across different backends.

## Available Backends

### Base Backend

The `base.py` file defines the `Backend` abstract base class, which specifies the interface that all backend implementations must adhere to. This ensures that all backends provide the same functionality, making it possible to switch between backends seamlessly.

### NumPy Backend

The `numpy_backend.py` file implements the NumPy backend, which uses NumPy for tensor operations. This is the default backend and is always available.

```python
import ember_ml as eh

# Set the backend to NumPy
eh.set_backend('numpy')

# Create a tensor
x = eh.random_normal((3, 4))
```

### PyTorch Backend

The `torch_backend.py` file implements the PyTorch backend, which uses PyTorch for tensor operations. This backend is available if PyTorch is installed.

```python
import ember_ml as eh

# Set the backend to PyTorch
eh.set_backend('torch')

# Create a tensor
x = eh.random_normal((3, 4))
```

### Optimized PyTorch Backend

The `torch_backend_optimized.py` file implements an optimized version of the PyTorch backend, which uses PyTorch for tensor operations with additional optimizations. This backend is available if PyTorch is installed.

```python
import ember_ml as eh

# Set the backend to optimized PyTorch
eh.set_backend('torch_optimized')

# Create a tensor
x = eh.random_normal((3, 4))
```

### MLX Backend

The `mlx_backend.py` file implements the MLX backend, which uses MLX for tensor operations. This backend is available if MLX is installed.

```python
import ember_ml as eh

# Set the backend to MLX
eh.set_backend('mlx')

# Create a tensor
x = eh.random_normal((3, 4))
```

## Backend Configuration

The backend configuration is stored in two files:

- `.backend`: Contains the name of the current backend
- `.device`: Contains the name of the current device (e.g., 'cpu', 'cuda')

These files are used to persist the backend configuration across sessions.

## Backend Selection

ember_ml provides several ways to select the backend:

### Manual Selection

You can manually select the backend using the `set_backend` function:

```python
import ember_ml as eh

# Set the backend to PyTorch
eh.set_backend('torch')
```

### Automatic Selection

ember_ml can automatically select the best available backend based on the available libraries and hardware:

```python
import ember_ml as eh

# Automatically select the best available backend
eh.auto_select_backend()
```

### Environment Variable

You can set the `ember_ml_BACKEND` environment variable to specify the backend:

```bash
export ember_ml_BACKEND=torch
```

### Configuration File

You can create a configuration file at `~/.ember_ml/config` with the following content:

```
backend = torch
device = cuda
```

## Backend Agnosticism

ember_ml achieves backend agnosticism through the use of the `ops` module, which provides a unified interface for tensor operations across different backends. The `ops` module delegates the actual operations to the current backend, making it possible to write code that works with any backend.

```python
from ember_ml import ops

# Create a tensor
x = tensor.random_normal((3, 4))

# Perform operations
y = ops.matmul(x, ops.transpose(x))
z = ops.relu(y)
```

## Implementation Details

### Backend Interface

The `Backend` abstract base class defines the interface that all backend implementations must adhere to. This includes methods for tensor creation, manipulation, and mathematical operations.

```python
class Backend(ABC):
    @abstractmethod
    def tensor(self, data, dtype=None):
        """Create a tensor from data."""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create a tensor of zeros."""
        pass
    
    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create a tensor of ones."""
        pass
    
    # ... and many more methods
```

### Backend Registration

Backends are registered using the `register_backend` function, which associates a backend name with a backend class:

```python
from ember_ml.backend.base import register_backend
from ember_ml.backend.numpy_backend import NumPyBackend

register_backend('numpy', NumPyBackend)
```

### Backend Selection

The `set_backend` function selects the backend by name and initializes it:

```python
from ember_ml.backend.base import set_backend

set_backend('numpy')
```

### Backend Auto-Selection

The `auto_select_backend` function selects the best available backend based on the available libraries and hardware:

```python
from ember_ml.backend.base import auto_select_backend

auto_select_backend()
```

## Usage Examples

### Basic Usage

```python
import ember_ml as eh
from ember_ml import ops

# Set the backend
eh.set_backend('torch')

# Create tensors
x = tensor.random_normal((3, 4))
y = tensor.random_normal((4, 5))

# Perform operations
z = ops.matmul(x, y)
w = ops.relu(z)

print(f"x shape: {tensor.shape(x)}")
print(f"y shape: {tensor.shape(y)}")
print(f"z shape: {tensor.shape(z)}")
print(f"w shape: {tensor.shape(w)}")
```

### Backend Auto-Selection

```python
import ember_ml as eh
from ember_ml import ops

# Automatically select the best available backend
eh.auto_select_backend()

# Print the selected backend
print(f"Selected backend: {eh.get_backend()}")

# Create tensors
x = tensor.random_normal((3, 4))
y = tensor.random_normal((4, 5))

# Perform operations
z = ops.matmul(x, y)
w = ops.relu(z)

print(f"x shape: {tensor.shape(x)}")
print(f"y shape: {tensor.shape(y)}")
print(f"z shape: {tensor.shape(z)}")
print(f"w shape: {tensor.shape(w)}")
```

### Backend-Specific Code

If you need to use backend-specific features, you can access the backend directly:

```python
import ember_ml as eh
from ember_ml import ops

# Set the backend
eh.set_backend('torch')

# Get the backend
backend = eh.get_backend()

# Create a tensor
x = tensor.random_normal((3, 4))

# Convert to a backend-specific tensor
if backend.name == 'torch':
    import torch
    x_torch = torch.tensor(tensor.to_numpy(x))
    # Use PyTorch-specific features
    x_torch = x_torch.cuda()
    # Convert back to ember_ml tensor
    x = tensor.convert_to_tensor(x_torch.cpu().numpy())
```

## Relationship with Other Modules

The backend module is used by the `ops` module to provide a unified interface for tensor operations across different backends. The `ops` module delegates the actual operations to the current backend, making it possible to write code that works with any backend.

For more information on the `ops` module, see the [Operations documentation](../ops/README.md).