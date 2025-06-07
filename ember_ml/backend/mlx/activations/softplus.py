# ember_ml/backend/mlx/activations/ops/softplus.py
import mlx.core as mx
import mlx.nn as mx_nn
from typing import Optional
from ember_ml.backend.mlx.types import TensorLike, DType

def softplus(x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Apply Softplus activation.
    
    Args:
        x: Input tensor
        dtype: Optional output data type. If None, uses default_float for floating point data
              and default_int for integer data.
        device: Optional device to place the output tensor on (ignored for MLX backend)
        
    Returns:
        Output tensor with Softplus activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x, dtype=dtype)
    # Use the mlx.nn functional version
    return mx_nn.softplus(x_tensor)