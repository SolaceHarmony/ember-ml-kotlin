# ember_ml/backend/torch/activations/ops.py
import torch
from typing import Any

from ember_ml.backend.torch.types import TensorLike
# Removed ActivationOps import as class wrapper is being removed
# from ember_ml.nn.modules.activations.ops.activation_ops import ActivationOps

# Epsilon for numerical stability (if needed, e.g., for manual softmax)
EPSILON = 1e-7

# Define top-level functions
def relu(x: TensorLike) -> torch.Tensor:
    """Apply Rectified Linear Unit activation."""
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(data=x)
    return torch.relu(x_tensor)

def sigmoid(x: TensorLike) -> torch.Tensor:
    """Apply Sigmoid activation."""
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(data=x)
    # Use torch.sigmoid for stability
    return torch.sigmoid(x_tensor)

def tanh(x: TensorLike) -> torch.Tensor:
    """Apply Hyperbolic Tangent activation."""
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(data=x)
    return torch.tanh(x_tensor)

def softmax(x: TensorLike, axis: int = -1) -> torch.Tensor:
    """Apply Softmax activation."""
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(data=x)
    return torch.softmax(x_tensor, dim=axis)

def softplus(x: TensorLike) -> torch.Tensor:
    """Apply Softplus activation."""
    from ember_ml.backend.torch.tensor import TorchTensor
    import torch.nn.functional as F  # Import torch functional
    x_tensor = TorchTensor().convert_to_tensor(data=x)
    return F.softplus(x_tensor)

# Removed TorchActivationOps class wrapper