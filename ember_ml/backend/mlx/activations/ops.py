# ember_ml/backend/mlx/activations/ops.py
import mlx.core as mx
import mlx.nn as mx_nn
from typing import Any

from ember_ml.backend.mlx.types import TensorLike

# Epsilon for numerical stability (if needed)
EPSILON = 1e-7

# Define top-level functions
def relu(x: TensorLike) -> mx.array:
    """Apply Rectified Linear Unit activation."""
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x)
    # MLX uses mx.maximum
    return mx.maximum(x_tensor, 0)

def sigmoid(x: TensorLike) -> mx.array:
    """Apply Sigmoid activation."""
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x)
    return mx.sigmoid(x_tensor)

def tanh(x: TensorLike) -> mx.array:
    """Apply Hyperbolic Tangent activation."""
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x)
    return mx.tanh(x_tensor)

def softmax(x: TensorLike, axis: int = -1) -> mx.array:
    """Apply Softmax activation."""
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x)
    return mx.softmax(x_tensor, axis=axis)

def softplus(x: TensorLike) -> mx.array:
    """Apply Softplus activation."""
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x)
    # Use the mlx.nn functional version
    return mx_nn.softplus(x_tensor)

# No class wrapper needed - the functions are directly exposed