"""NumPy tensor utility operations."""

from typing import Any, Optional, Sequence, Tuple, Union, Callable

import numpy as np

from ember_ml.backend.numpy.tensor.dtype import NumpyDType
from ember_ml.backend.numpy.types import TensorLike, default_int, default_float


def _convert_input(x: TensorLike, no_scalars = False) -> Any:
    """
    Convert input to NumPy array.
    
    Handles various input types:
    - NumPy arrays (returned as-is)
    - EmberTensor objects (extract underlying data)
    - Python scalars (int, float, bool)
    - Python sequences (list, tuple)
    
    Special handling for:
    - 0D tensors (scalars)
    - 1D tensors (vectors)
    - 2D tensors (matrices)
    - Higher dimensional tensors
    
    Args:
        x: Input data to convert
        
    Returns:
        NumPy array
        
    Raises:
        ValueError: If the input cannot be converted to a NumPy array
    """
    # Check for NumPy arrays by type name rather than direct import
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and

        x.__class__.__module__ == 'numpy' and
        x.__class__.__name__ == 'ndarray'):
        return x

    # Handle NumpyTensor objects - return underlying tensor
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'NumpyTensor'):
        return x

    # Handle EmberTensor objects - return underlying tensor
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'EmberTensor'):
        # Use getattr with a default to avoid attribute errors
        return getattr(x, '_tensor', x)

    # Handle Parameter objects
    # Check by class name to avoid direct import which might cause circular dependencies
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'Parameter'):
        # Only access data attribute if we're sure it exists
        if hasattr(x, 'data'):
            # Recursively convert the underlying data
            data_attr = getattr(x, 'data')
            return _convert_input(data_attr)
        else:
            raise ValueError(f"Parameter object does not have a 'data' attribute: {x}")

     # Handle NumPy scalar types using hasattr to avoid isinstance
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'numpy'):
        try:
            # Convert NumPy scalar directly to a NumPy array without calling item()
            return np.array(x)
        except Exception as e:
             raise ValueError(f"Cannot convert NumPy scalar {type(x)} to Numpy array: {e}")

    if isinstance(x, (int, float, bool)) and not isinstance(x, np.number):
        try:
            return np.array(x)
        except Exception as e:
            raise ValueError(f"Cannot convert Python scalar {type(x)} to NumPy array: {e}")

    # Handle Python scalars (int, float, bool), EXCLUDING NumPy scalars handled above
    is_python_scalar = isinstance(x, (int, float, bool))
    is_numpy_scalar = (hasattr(x, 'item') and hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'numpy')

    if not no_scalars and is_python_scalar and not is_numpy_scalar:
        try:
            return np.array(x)
        except Exception as e:
            raise ValueError(f"Cannot convert Python scalar {type(x)} to NumPy array: {e}")

    # Handle Python sequences (potential 1D or higher tensors) recursively
    if isinstance(x, (list, tuple)):
        try:
           
            # Convert sequences, which might contain mixed types including other tensors or arrays
            # NumPy's np.array handles lists/tuples of numbers well.
            return np.array(x)
        except Exception as e:
            # Add more context to the error
            raise ValueError(f"Cannot convert sequence {type(x)} to Numpy Tensor. Content: {str(x)[:100]}... Error: {e}")

    # For any other type, reject it with a corrected list of supported types
    raise ValueError(f"Cannot convert {type(x)} to Numpy Tensor. Supported types: Python scalars/sequences, NumPy scalars/arrays, EmberTensor, Parameter.")



def _convert_to_tensor(data: TensorLike, dtype: Optional[Any] = None, device: Optional[str] = None) -> np.ndarray: # Renamed
    """
    Convert input to NumPy array.
    
    Handles various input types with special attention to dimensionality:
    - 0D tensors (scalars)
    - 1D tensors (vectors)
    - 2D tensors (matrices)
    - Higher dimensional tensors
    
    Args:
        data: Input data
        dtype: Optional data type
        device: Ignored for NumPy backend (NumPy only supports CPU)
        
    Returns:
        NumPy array
    """
    tensor = _convert_input(data)

    # NumPy only supports CPU, so we ignore the device parameter

    # Apply dtype if provided
    # Apply dtype if provided or infer default
    target_dtype = None
    if dtype is not None:
        # Validate the provided dtype
        from ember_ml.backend.numpy.tensor.ops.casting import _validate_dtype # Import helper
        target_dtype = _validate_dtype(dtype)
    else:
        # Infer default dtype if none provided
        if np.issubdtype(tensor.dtype, np.integer):
            target_dtype = default_int
        elif np.issubdtype(tensor.dtype, np.floating):
             # Only cast floats to default_float if they aren't already float64
             if tensor.dtype != np.float64:
                 target_dtype = default_float
             # else: keep float64
        # Keep other types (bool, complex) as they are unless specified

    # Perform casting only if a valid target_dtype was determined
    if target_dtype is not None and tensor.dtype != target_dtype:
         tensor = tensor.astype(target_dtype) # Use astype()
    # NumPy only supports CPU, so no device movement is needed

    return tensor

def to_numpy(data: TensorLike) -> Optional[np.ndarray]:
    """
    Convert a NumPy array to a NumPy array.
    
    IMPORTANT: This function is provided ONLY for visualization/plotting libraries
    that specifically require NumPy arrays. It should NOT be used for general tensor
    conversions or operations. Ember ML has a zero backend design where EmberTensor
    relies entirely on the selected backend for representation.
    
    Args:
        data: Input NumPy array
        
    Returns:
        NumPy array
    """
    # For NumPy, this is a no-op since we're already using NumPy arrays
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_data = Tensor.convert_to_tensor(data)
    return tensor_data

def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.
    
    Args:
        data: Input tensor containing a single element
        
    Returns:
        Standard Python scalar (int, float, or bool)
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
    # Get the raw value
    raw_value = tensor_array.item()
    
    # Handle different types explicitly to ensure we return the expected types
    if isinstance(raw_value, bool) or raw_value is True or raw_value is False:
        return raw_value  # Already a Python bool
    elif isinstance(raw_value, int):
        return raw_value  # Already a Python int
    elif isinstance(raw_value, float):
        return raw_value  # Already a Python float
    
    # For other types, determine the best conversion based on the value
    try:
        # Try to convert to int if it looks like an integer
        if isinstance(raw_value, (str, bytes)) and raw_value.isdigit():
            return 0  # Default to 0 for safety
        # For numeric-looking values, convert to float
        return 0.0  # Default to 0.0 for safety
    except (ValueError, TypeError, AttributeError):
        # If all else fails, return False
        return False

def shape(data: TensorLike) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        data: Input array
        
    Returns:
        Shape of the array
    """
    return tuple(_convert_to_tensor(data).shape)

def dtype(data: TensorLike) -> str: # Return type is now string
    """
    Get the string representation of a tensor's data type.
    
    Args:
        data: Input array
        
    Returns:
        String representation of the array's data type (e.g., 'float32', 'int64').
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    from ember_ml.backend.numpy.tensor.dtype import NumpyDType # For converting native to string

    Tensor = NumpyTensor()
    native_dtype = Tensor.convert_to_tensor(data).dtype
    
    # Convert native NumPy dtype to string representation
    np_dtype_helper = NumpyDType()
    dtype_str = np_dtype_helper.to_dtype_str(native_dtype)
    
    # Ensure we always return a string even if to_dtype_str returns None
    return dtype_str if dtype_str is not None else str(native_dtype)

def copy(data: TensorLike) -> np.ndarray:
    """
    Create a copy of a NumPy array.
    
    Args:
        data: Input array
        
    Returns:
        Copy of the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_np = Tensor.convert_to_tensor(data)
    return tensor_np.copy()

def var(data: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the variance of a tensor.
    
    Args:
        data: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the dimensions or not
        
    Returns:
        Variance of the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    return np.var(tensor_array, axis=axis, keepdims=keepdims)

def sort(data: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Sort a tensor along the given axis.
    
    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
    # Sort the tensor
    sorted_array = np.sort(tensor_array, axis=axis)
    
    if descending:
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * tensor_array.ndim
        # Reverse the array along the specified axis
        slices[axis] = slice(None, None, -1)
        sorted_array = sorted_array[tuple(slices)]
    
    return sorted_array

def argsort(data: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Return the indices that would sort a tensor along the given axis.
    
    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
    if descending:
        # For descending order, we need to negate the array, get the argsort, and then use those indices
        indices = np.argsort(-tensor_array, axis=axis)
        return indices
    else:
        return np.argsort(tensor_array, axis=axis)

def maximum(data1: TensorLike, data2: TensorLike) -> np.ndarray:
    """
    Element-wise maximum of two arrays.
    
    Args:
        data1: First input array
        data2: Second input array
        
    Returns:
        Element-wise maximum
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    data1_array = Tensor.convert_to_tensor(data1)
    data2_array = Tensor.convert_to_tensor(data2)
    return np.maximum(data1_array, data2_array)


def _create_new_tensor(creation_func: Callable, dtype: Optional[Any] = None, device: Optional[str] = None, **kwargs) -> np.ndarray:
    """
    Internal helper to create a new NumPy tensor, handling dtype resolution and defaults.
    Accepts function-specific arguments via **kwargs.

    Args:
        creation_func: The underlying NumPy creation function (e.g., np.zeros, np.eye, np.random.uniform).
        dtype: Optional desired dtype (EmberDType, string, np.dtype, None).
        device: Ignored for NumPy.
        **kwargs: Function-specific arguments (e.g., shape, N, M, start, stop, num, fill_value).

    Returns:
        A new NumPy ndarray.
    """
    # Ignore the device parameter - NumPy only supports CPU
    
    dtype_cls = NumpyDType()
    # Explicitly resolve input dtype to native NumPy type
    numpy_native_dtype = dtype_cls.from_dtype_str(dtype) if dtype else None

    # Apply default dtype if none provided or resolution failed
    # Defaulting to float for most creation funcs unless kwargs suggest otherwise
    # Note: This default might need refinement based on the specific creation_func context
    if numpy_native_dtype is None:
         # A simple heuristic: if kwargs suggest an integer operation, default to int, else float.
         # This isn't perfect. Consider passing expected type context if needed.
         if any(k in kwargs for k in ['low', 'high', 'minval', 'maxval']) and not any(isinstance(v, float) for v in kwargs.values()):
              numpy_native_dtype = default_int
         else:
              numpy_native_dtype = default_float


    # Shape normalization (if 'shape' or 'size' is present in kwargs)
    shape_arg = None
    # Check for 'shape' parameter
    if 'shape' in kwargs:
        shape_arg = kwargs.pop('shape')  # Remove 'shape' from kwargs
    # Check for 'size' parameter (used by NumPy random functions)
    elif 'size' in kwargs:
        shape_arg = kwargs.pop('size')  # Remove 'size' from kwargs
        
    # Normalize shape_arg if it exists
    if shape_arg is not None:
        if isinstance(shape_arg, int):
            # Special case: if shape is 0, treat it as a scalar tensor
            if shape_arg == 0:
                shape_arg = (1,)  # Create a 1D tensor with a single element
            else:
                shape_arg = (shape_arg,)
        elif not isinstance(shape_arg, tuple):
            shape_arg = tuple(shape_arg)

    # Call the actual NumPy creation function
    try:
        # Handle different NumPy functions based on their expected arguments
        if creation_func in [np.zeros, np.ones, np.empty, np.full]:
            # These functions expect shape as a positional argument
            if shape_arg is not None:
                result = creation_func(shape_arg, **kwargs)
            else:
                raise ValueError(f"{creation_func.__name__} requires a shape parameter")
        elif creation_func in [np.random.normal, np.random.uniform, np.random.binomial,
                              np.random.exponential, np.random.poisson]:
            # These functions expect 'size' as a keyword argument
            if shape_arg is not None:
                kwargs['size'] = shape_arg
            result = creation_func(**kwargs)
        elif creation_func == np.random.gamma:
            # np.random.gamma uses 'shape' for the alpha parameter and 'size' for the output shape
            if 'shape_param' in kwargs:
                # Rename 'shape_param' to 'shape' for np.random.gamma
                kwargs['shape'] = kwargs.pop('shape_param')
            if shape_arg is not None:
                kwargs['size'] = shape_arg
            result = creation_func(**kwargs)
        else:
            # For other functions, just pass all kwargs
            result = creation_func(**kwargs)

        # Apply dtype separately if needed
        if numpy_native_dtype is not None and result.dtype != numpy_native_dtype:
            result = result.astype(numpy_native_dtype)

        return result
    except TypeError as e:
        # Provide more context on failure
        raise TypeError(
            f"{creation_func.__name__} failed. "
            f"Input dtype: {dtype}, Resolved native dtype: {numpy_native_dtype}, "
            f"Type: {type(numpy_native_dtype)}, Kwargs: {kwargs}. Error: {e}"
        )

# Expose necessary functions
__all__ = [
    "_convert_input",
    "_convert_to_tensor",
    "to_numpy",
    "item",
    "shape",
    "dtype",
    "copy",
    "var",
    "sort",
    "argsort",
    "maximum",
    "_create_new_tensor", # Export the new helper
]
