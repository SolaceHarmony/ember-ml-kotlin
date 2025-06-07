# ember_ml/backend/torch/activations/softmax.py
import torch
from typing import Optional
from ember_ml.backend.torch.types import TensorLike, DType

def softmax(x: TensorLike, axis: int = -1, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Apply Softmax activation.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute softmax, default is -1 (last dimension)
        dtype: Optional output data type. If None, uses default_float for floating point data
               and default_int for integer data.
        device: Optional device to place the output tensor on.
        
    Returns:
        Output tensor with Softmax activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(data=x, dtype=dtype, device=device)
    return torch.softmax(x_tensor, dim=axis) # PyTorch uses 'dim' instead of 'axis'