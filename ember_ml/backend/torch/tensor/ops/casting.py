"""PyTorch tensor casting operations."""

import torch

# from typing import Optional # Removed unused import
from ember_ml.backend.torch.types import DType, TensorLike


# _validate_dtype helper function removed. Logic moved to TorchDType.validate_dtype

def cast(tensor: TensorLike, dtype: DType) -> torch.Tensor:
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Import Torch specifics lazily
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    from ember_ml.backend.torch.tensor.dtype import TorchDType

    # 1. Validate the target dtype using the class method
    torch_dtype = TorchDType().validate_dtype(dtype)

    # 2. Convert the input tensor to a base PyTorch tensor
    tensor_obj = TorchTensor()
    tensor_array = tensor_obj.convert_to_tensor(tensor) # Assuming this doesn't cast
    
    # If mlx_dtype is None, return the tensor as is
    if torch_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype using tensor_array, not tensor
    # tensor_array is the PyTorch tensor that has the .to() method
    return tensor_array.to(torch_dtype)