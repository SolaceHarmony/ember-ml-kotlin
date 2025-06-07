# Getting Started with Ember ML

This tutorial will guide you through the process of installing Ember ML and running your first operations.

## Installation

You can install Ember ML using pip:

```bash
pip install ember-ml
```

## Basic Usage

### Importing Ember ML

```python
import ember_ml as eh
from ember_ml import ops
```

### Creating Tensors

Ember ML provides a unified API for tensor operations across different backends:

```python
# Create a tensor
x = ops.ones((3, 3))
print(x)

# Create a tensor with specific values
y = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y)
```

### Basic Operations

```python
# Addition
z = ops.add(x, y)
print("Addition:")
print(z)

# Multiplication
z = ops.multiply(x, y)
print("Multiplication:")
print(z)

# Matrix multiplication
z = ops.matmul(x, y)
print("Matrix multiplication:")
print(z)
```

### Backend Selection

Ember ML automatically selects the best backend based on your hardware:

```python
from ember_ml.backend import get_backend

# Get the current backend
backend = get_backend()
print(f"Using backend: {backend}")

# Manually set a backend
from ember_ml.ops import set_backend

# Set to NumPy backend
set_backend('numpy')
print(f"Now using backend: {get_backend()}")

# Set to PyTorch backend (if available)
try:
    set_backend('torch')
    print(f"Now using backend: {get_backend()}")
except ValueError:
    print("PyTorch backend not available")

# Set to MLX backend (if available, Apple Silicon only)
try:
    set_backend('mlx')
    print(f"Now using backend: {get_backend()}")
except ValueError:
    print("MLX backend not available")
```

## Next Steps

Now that you've installed Ember ML and run some basic operations, you can:

1. Explore the [API Reference](../api/index.md) to learn about all available functions
2. Check out the [Examples](../examples/index.md) for more complex use cases
3. Learn about [Feature Extraction](feature_extraction_basics.md) for working with datasets

Happy coding with Ember ML!