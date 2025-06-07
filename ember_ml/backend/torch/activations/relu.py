# ember_ml/backend/torch/activations/relu.py
import torch
from typing import Optional
from ember_ml.backend.torch.types import TensorLike, DType

def relu(x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Apply Rectified Linear Unit activation.
    
    Args:
        x: Input tensor
        dtype: Optional output data type. If None, uses default_float for floating point data
               and default_int for integer data.
        device: Optional device to place the output tensor on.
        
    Returns:
        Output tensor with ReLU activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(data=x, dtype=dtype, device=device)
    
    # Get zero tensor with same shape and dtype as input
    zero = torch.zeros_like(x_tensor)
    
    # PyTorch uses torch.maximum
    return torch.maximum(x_tensor, zero)