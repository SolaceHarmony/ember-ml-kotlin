# ember_ml/backend/numpy/activations/relu.py
import numpy as np
from typing import Optional
from ember_ml.backend.numpy.types import TensorLike, DType

def relu(x: TensorLike, dtype: Optional[DType] = None) -> np.ndarray:
    """
    Apply Rectified Linear Unit activation.
    
    Args:
        x: Input tensor
        dtype: Optional output data type. If None, uses default_float for floating point data
               and default_int for integer data.
        
    Returns:
        Output tensor with ReLU activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.numpy.tensor import NumpyTensor
    x_tensor = NumpyTensor().convert_to_tensor(data=x, dtype=dtype)
    
    # Get zero tensor with same shape and dtype as input
    zero = np.zeros_like(x_tensor)
    
    # NumPy uses np.maximum
    return np.maximum(x_tensor, zero)