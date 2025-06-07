# ember_ml/backend/numpy/activations/tanh.py
import numpy as np
from typing import Optional
from ember_ml.backend.numpy.types import TensorLike, DType

def tanh(x: TensorLike, dtype: Optional[DType] = None) -> np.ndarray:
    """
    Apply Hyperbolic Tangent activation.
    
    Args:
        x: Input tensor
        dtype: Optional output data type. If None, uses default_float for floating point data
               and default_int for integer data.
        
    Returns:
        Output tensor with Tanh activation applied
    """
    # Lazy load the tensor class
    from ember_ml.backend.numpy.tensor import NumpyTensor
    x_tensor = NumpyTensor().convert_to_tensor(data=x, dtype=dtype)
    return np.tanh(x_tensor)