"""NumPy tensor casting operations."""

import numpy as np
from typing import Any, Optional

from ember_ml.backend.numpy.types import DType, TensorLike

def _validate_dtype(dtype: DType) -> Optional[Any]:
    """
    Validate and convert dtype to NumPy format.
    
    Args:
        dtype_cls: NumpyDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated NumPy dtype or None
    """
    if dtype is None:
        return None
    
    # EmberDType handling
    if (hasattr(dtype, '__class__') and
        hasattr(dtype.__class__, '__name__') and
        dtype.__class__.__name__ == 'EmberDType'):
        from ember_ml.nn.tensor.common.dtypes import EmberDType
        if isinstance(dtype, EmberDType):
            dtype_from_ember = dtype._backend_dtype
            if dtype_from_ember is not None:
                return dtype_from_ember
            
    # EmberTensor string handling
    if isinstance(dtype, str):
        from ember_ml.backend.numpy.tensor.dtype import NumpyDType # Corrected import path
        return NumpyDType().from_dtype_str(dtype=dtype)

    # If it's already a NumPy dtype, return as is
    if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64,
                                              np.bool_, np.int8, np.int16, np.uint8,
                                              np.uint16, np.uint32, np.uint64, np.float16]:
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def cast(tensor: TensorLike, dtype: DType) -> np.ndarray:
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Import helper locally to avoid potential cycles if this module grows
    from ember_ml.backend.numpy.tensor.ops.utility import _convert_input
    # Ensure input is a NumPy array first
    tensor_array = _convert_input(tensor)
    
    # Validate the dtype
    numpy_dtype = _validate_dtype(dtype) # _validate_dtype doesn't need NumpyDType() instance
    
    # If numpy_dtype is None, return the tensor as is
    if numpy_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype
    return tensor_array.astype(numpy_dtype)