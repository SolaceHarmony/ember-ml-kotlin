# ember_ml/backend/mlx/activations/ops/relu.py
import mlx.core as mx
from typing import Optional
from ember_ml.backend.mlx.types import TensorLike, DType

def relu(x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Apply Rectified Linear Unit activation.
    
    Args:
        x: Input tensor
        dtype: Optional output data type. If None, uses default_float for floating point data
              and default_int for integer data.
        device: Optional device to place the output tensor on (ignored for MLX backend)
        
    Returns:
        Output tensor with ReLU activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x, dtype=dtype)
    
    # Get zero tensor with same shape and dtype as input
    zero = mx.zeros_like(x_tensor)
    
    # MLX uses mx.maximum
    return mx.maximum(x_tensor, zero)