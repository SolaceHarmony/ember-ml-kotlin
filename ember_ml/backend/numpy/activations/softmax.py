# ember_ml/backend/numpy/activations/softmax.py
import numpy as np
from typing import Optional
from ember_ml.backend.numpy.types import TensorLike, DType

def softmax(x: TensorLike, axis: int = -1, dtype: Optional[DType] = None) -> np.ndarray:
    """
    Apply Softmax activation.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute softmax, default is -1 (last dimension)
        dtype: Optional output data type. If None, uses default_float for floating point data
               and default_int for integer data.
        
    Returns:
        Output tensor with Softmax activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.numpy.tensor import NumpyTensor
    x_tensor = NumpyTensor().convert_to_tensor(data=x, dtype=dtype)
    
    # NumPy implementation of softmax
    e_x = np.exp(x_tensor - np.max(x_tensor, axis=axis, keepdims=True)) # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)