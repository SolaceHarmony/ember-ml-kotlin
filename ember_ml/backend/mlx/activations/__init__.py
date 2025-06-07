"""MLX backend activation operations for ember_ml."""

# Import activation functions from ops.py
from ember_ml.backend.mlx.activations.ops import (
    relu,
    sigmoid,
    tanh,
    softmax,
    softplus
)

__all__ = [
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus"
]