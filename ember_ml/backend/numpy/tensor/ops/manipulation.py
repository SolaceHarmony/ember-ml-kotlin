"""NumPy tensor manipulation operations."""

import numpy as np
from typing import Optional, Union, List

from ember_ml.backend.numpy.types import TensorLike, ShapeLike, Shape

def reshape(tensor: TensorLike, shape: ShapeLike) -> np.ndarray:
    """
    Reshape a NumPy array to a new shape.
    
    Args:
        tensor: Input array
        shape: New shape
        
    Returns:
        Reshaped NumPy array
    """
    # Ensure shape is a sequence
    if isinstance(shape, int):
        shape = (shape,)
    
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    return np.reshape(Tensor.convert_to_tensor(tensor), shape)

def transpose(tensor: TensorLike, axes: Optional[Shape] = None) -> np.ndarray:
    """
    Permute the dimensions of a NumPy array.
    
    Args:
        tensor: Input array
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed NumPy array
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        ndim = len(tensor_array.shape)
        if ndim <= 1:
            return tensor_array
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
    
    return np.transpose(tensor_array, axes)

def concatenate(tensors: List[TensorLike], axis: int = 0) -> np.ndarray:
    """
    Concatenate NumPy arrays along a specified axis.
    
    Args:
        tensors: Sequence of arrays
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated NumPy array
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    return np.concatenate([Tensor.convert_to_tensor(arr) for arr in tensors], axis=axis)

def vstack(tensors: List[TensorLike]) -> np.ndarray:
    """
    Stack arrays vertically (row wise).
    
    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape (N,) have been reshaped to (1,N). Rebuilds arrays divided by vsplit.
    
    Args:
        tensors: Sequence of arrays
        
    Returns:
        Stacked NumPy array
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    # Convert to NumPy arrays
    numpy_tensors = []
    for t in tensors:
        tensor = Tensor.convert_to_tensor(t)
        # If 1D tensor, reshape to (1, N)
        if len(tensor.shape) == 1:
            tensor = np.reshape(tensor, (1, -1))
        numpy_tensors.append(tensor)
    
    # Concatenate along the first axis
    return np.concatenate(numpy_tensors, axis=0)

def hstack(tensors: List[TensorLike]) -> np.ndarray:
    """
    Stack arrays horizontally (column wise).
    
    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided by hsplit.
    
    Args:
        tensors: Sequence of arrays
        
    Returns:
        Stacked NumPy array
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    # Convert to NumPy arrays
    numpy_tensors = [Tensor.convert_to_tensor(t) for t in tensors]
    
    # Check if tensors are 1D
    if all(len(t.shape) == 1 for t in numpy_tensors):
        # For 1D tensors, concatenate along axis 0
        return np.concatenate(numpy_tensors, axis=0)
    else:
        # For nD tensors, concatenate along axis 1
        return np.concatenate(numpy_tensors, axis=1)

def stack(tensors: List[TensorLike], axis: int = 0) -> np.ndarray:
    """
    Stack NumPy arrays along a new axis.
    
    Args:
        tensors: Sequence of arrays
        axis: Axis along which to stack
        
    Returns:
        Stacked NumPy array
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    return np.stack([Tensor.convert_to_tensor(arr) for arr in tensors], axis=axis)

def split(tensor: TensorLike, num_or_size_splits: Union[int, List[int]], axis: int = 0) -> List[np.ndarray]:
    """
    Split a NumPy array into sub-arrays.
    
    Args:
        tensor: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split
        
    Returns:
        List of sub-arrays
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    if isinstance(num_or_size_splits, int):
        # Use array_split for integer splits
        result = np.array_split(tensor_array, num_or_size_splits, axis=axis)
    else:
        # Use split for explicit section sizes
        result = np.split(tensor_array, num_or_size_splits, axis=axis)
    
    # Convert to list if it's not already a list
    if isinstance(result, list):
        return result
    else:
        # If it's a single array, return a list with that array
        return [result]

def expand_dims(tensor: TensorLike, axis: ShapeLike) -> np.ndarray:
    """
    Insert new axes into a NumPy array's shape.
    
    Args:
        tensor: Input array
        axis: Position(s) where new axes should be inserted
        
    Returns:
        NumPy array with expanded dimensions
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis):
            tensor_array = np.expand_dims(tensor_array, ax)
        return tensor_array
    
    return np.expand_dims(tensor_array, axis)

def squeeze(tensor: TensorLike, axis: Optional[Union[int, List[int]]] = None) -> np.ndarray:
    """
    Remove single-dimensional entries from a NumPy array's shape.
    
    Args:
        tensor: Input array
        axis: Position(s) where dimensions should be removed
        
    Returns:
        NumPy array with squeezed dimensions
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    return np.squeeze(tensor_array, axis=axis)

def tile(tensor: TensorLike, reps: List[int]) -> np.ndarray:
    """
    Construct a NumPy array by tiling a given array.
    
    Args:
        tensor: Input array
        reps: Number of repetitions for each dimension
        
    Returns:
        Tiled NumPy array
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    return np.tile(tensor_array, reps)

def pad(tensor: TensorLike, paddings: List[List[int]], constant_values: int = 0) -> np.ndarray:
    """
    Pad a tensor with a constant value.
    
    Args:
        tensor: Input tensor
        paddings: List of lists of integers specifying the padding for each dimension
                 Each inner list should contain two integers: [pad_before, pad_after]
        constant_values: Value to pad with
        
    Returns:
        Padded tensor
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Convert paddings to the format expected by np.pad
    # NumPy expects ((pad_before_dim1, pad_after_dim1), (pad_before_dim2, pad_after_dim2), ...)
    pad_width = tuple(tuple(p) for p in paddings)
    
    # Pad the tensor
    return np.pad(tensor_array, pad_width, mode='constant', constant_values=constant_values)