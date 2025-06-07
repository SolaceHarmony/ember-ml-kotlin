# ember_ml/backend/numpy/activations/ops.py
import numpy as np
from typing import Any

from ember_ml.backend.numpy.types import TensorLike
# Import NumpyTensor for lazy loading inside functions
# Removed ActivationOps import as class wrapper is being removed
# from ember_ml.nn.modules.activations.ops.activation_ops import ActivationOps

# Epsilon for numerical stability
EPSILON = 1e-7

# Define top-level functions
def relu(x: TensorLike) -> np.ndarray:
    """Apply Rectified Linear Unit activation."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    Tensor = NumpyTensor # Lazy load
    x_arr = Tensor().convert_to_tensor(data=x)
    return np.maximum(0, x_arr)

def sigmoid(x: TensorLike) -> np.ndarray:
    """Apply Sigmoid activation."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    Tensor = NumpyTensor # Lazy load
    # Convert input to numpy array
    x_arr = Tensor().convert_to_tensor(data=x)
    # Stable sigmoid: 1 / (1 + exp(-x))
    # Clipping input to avoid overflow in exp
    x_safe = np.clip(x_arr, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x_safe))

def tanh(x: TensorLike) -> np.ndarray:
    """Apply Hyperbolic Tangent activation."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    Tensor = NumpyTensor # Lazy load
    # Convert input to numpy array
    x_arr = Tensor().convert_to_tensor(data=x)
    return np.tanh(x_arr)

def softmax(x: TensorLike, axis: int = -1) -> np.ndarray:
    """Apply Softmax activation."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    Tensor = NumpyTensor # Lazy load
    # Convert input to numpy array
    x_arr = Tensor().convert_to_tensor(data=x)
    # Stable softmax: subtract max before exp
    x_max = np.max(x_arr, axis=axis, keepdims=True)
    exp_x = np.exp(x_arr - x_max)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    # Add epsilon to denominator for stability if sum is zero
    return exp_x / (sum_exp_x + EPSILON)

def softplus(x: TensorLike) -> np.ndarray:
    """Apply Softplus activation."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    Tensor = NumpyTensor # Lazy load
    # Convert input to numpy array
    x_arr = Tensor().convert_to_tensor(data=x)
    # log(1 + exp(x)) - clip for stability
    x_safe = np.clip(x_arr, -88.0, 88.0)
    return np.log(1.0 + np.exp(x_safe))

# Removed NumpyActivationOps class wrapper