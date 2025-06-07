# ember_ml/backend/mlx/activations/ops/softmax.py
import mlx.core as mx
from typing import Optional
from ember_ml.backend.mlx.types import TensorLike, DType

def softmax(x: TensorLike, axis: int = -1, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Apply Softmax activation.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute softmax, default is -1 (last dimension)
        dtype: Optional output data type. If None, uses default_float for floating point data
              and default_int for integer data.
        device: Optional device to place the output tensor on (ignored for MLX backend)
        
    Returns:
        Output tensor with Softmax activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.mlx.tensor import MLXTensor
    x_tensor = MLXTensor().convert_to_tensor(data=x, dtype=dtype)
    return mx.softmax(x_tensor, axis=axis)