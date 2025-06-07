"""MLX tensor manipulation operations."""

import mlx.core as mx
from typing import Optional,Union, List

from ember_ml.backend.mlx.types import TensorLike,ShapeLike,Shape

def reshape(tensor: TensorLike, shape: ShapeLike) -> mx.array:
    """
    Reshape an MLX array to a new shape.
    
    Args:
        tensor: Input array
        shape: New shape
        
    Returns:
        Reshaped MLX array
    """
    # Ensure shape is a sequence

    if isinstance(shape, int):
        shape = (shape,)
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    return mx.reshape(Tensor.convert_to_tensor(tensor), shape)

def transpose(tensor: TensorLike, axes: Optional[Shape]=None):
    """
    Permute the dimensions of an MLX array.
    
    Args:
        tensor: Input array
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed MLX array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    tensor_array = Tensor.convert_to_tensor(tensor)

    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        ndim = len(tensor_array.shape)
        if ndim <= 1:
            return tensor_array
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]

    return mx.transpose(tensor_array, axes)

def concatenate(tensors: list[TensorLike], axis: Optional[int]=0):
    """
    Concatenate MLX arrays along a specified axis.

    Args:
        tensors: Sequence of arrays
        axis: Axis along which to concatenate

    Returns:
        Concatenated MLX array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    return mx.concatenate([Tensor.convert_to_tensor(arr) for arr in tensors], axis=axis)

def vstack(tensors: List[TensorLike]) -> mx.array:
    """
    Stack arrays vertically (row wise).
    
    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape (N,) have been reshaped to (1,N). Rebuilds arrays divided by vsplit.
    
    Args:
        tensors: Sequence of arrays
        
    Returns:
        Stacked MLX array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    
    # Convert to MLX arrays
    mlx_tensors = []
    for t in tensors:
        tensor = Tensor.convert_to_tensor(t)
        # If 1D tensor, reshape to (1, N)
        if len(tensor.shape) == 1:
            tensor = mx.reshape(tensor, (1, -1))
        mlx_tensors.append(tensor)
    
    # Concatenate along the first axis
    return mx.concatenate(mlx_tensors, axis=0)

def hstack(tensors: List[TensorLike]) -> mx.array:
    """
    Stack arrays horizontally (column wise).
    
    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided by hsplit.
    
    Args:
        tensors: Sequence of arrays
        
    Returns:
        Stacked MLX array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    
    # Convert to MLX arrays
    mlx_tensors = [Tensor.convert_to_tensor(t) for t in tensors]
    
    # Check if tensors are 1D
    if all(len(t.shape) == 1 for t in mlx_tensors):
        # For 1D tensors, concatenate along axis 0
        return mx.concatenate(mlx_tensors, axis=0)
    else:
        # For nD tensors, concatenate along axis 1
        return mx.concatenate(mlx_tensors, axis=1)

def stack(tensors : list[TensorLike], axis: Optional[int]=0):
    """
    Stack MLX arrays along a new axis.

    Args:
        tensors: Sequence of arrays
        axis: Axis along which to stack

    Returns:
        Stacked MLX array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    return mx.stack([Tensor.convert_to_tensor(arr) for arr in tensors], axis=axis)

def split(tensor: TensorLike, num_or_size_splits: Union[int, list[int]], axis=0) -> list[mx.array]:
    """
    Split an MLX array into sub-arrays.

    Args:
        tensor: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split

    Returns:
        List of sub-arrays
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Handle different types of splits
    if isinstance(num_or_size_splits, int):
        # Split into equal parts
        return split_tensor(tensor_array, num_or_size_splits, axis)
    else:
        # Split at specified indices
        return split_at_indices(tensor_array, num_or_size_splits, axis)

def split_tensor(tensor: mx.array, num_splits: int, axis=0) -> list[mx.array]:
    """
    Split a tensor into num_splits equal parts along the given axis.
    
    Args:
        tensor: Input tensor
        num_splits: Number of equal parts to split into
        axis: Axis along which to split
        
    Returns:
        List of sub-tensors
    """
    # Get the size of the dimension to split
    dim_size = tensor.shape[axis]
    
    # Calculate the size of each split
    split_size = dim_size // num_splits
    
    # Handle case where the split doesn't divide evenly
    remainder = dim_size % num_splits
    
    # Create a list to store the results
    result = []
    
    # Calculate the start indices for each split
    start_idx = 0
    for i in range(num_splits):
        # Adjust the size of this split if there's a remainder
        this_split_size = split_size + (1 if i < remainder else 0)
        
        # Create slices for each dimension
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start_idx, start_idx + this_split_size)
        
        # Extract the slice and add to results
        result.append(tensor[tuple(slices)])
        
        # Update the start index for the next split
        start_idx += this_split_size
    
    return result

def split_at_indices(tensor: mx.array, indices: list[int], axis=0) -> list[mx.array]:
    """
    Split a tensor at the specified indices along the given axis.
    
    Args:
        tensor: Input tensor
        indices: List of indices where to split the tensor
        axis: Axis along which to split
        
    Returns:
        List of sub-tensors
    """
    # Get the size of the dimension to split
    dim_size = tensor.shape[axis]
    
    # Ensure indices are sorted
    indices = sorted(indices)
    
    # Create a list to store the results
    result = []
    
    # Calculate the start indices for each split
    start_idx = 0
    for idx in indices:
        if idx <= start_idx:
            # Skip invalid indices
            continue
            
        # Create slices for each dimension
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start_idx, idx)
        
        # Extract the slice and add to results
        result.append(tensor[tuple(slices)])
        
        # Update the start index for the next split
        start_idx = idx
    
    # Add the final piece if needed
    if start_idx < dim_size:
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start_idx, dim_size)
        result.append(tensor[tuple(slices)])
    
    return result

def expand_dims(tensor : TensorLike, axis: ShapeLike) -> mx.array:
    """
    Insert new axes into an MLX array's shape.

    Args:
        tensor: Input array
        axis: Position(s) where new axes should be inserted

    Returns:
        MLX array with expanded dimensions
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    tensor_array = Tensor.convert_to_tensor(tensor)

    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis):
            tensor_array = mx.expand_dims(tensor_array, ax)
        return tensor_array

    return mx.expand_dims(tensor_array, axis)

def squeeze(tensor: TensorLike, axis : Union[None,ShapeLike]=None):
    """
    Remove single-dimensional entries from an MLX array's shape.

    Args:
        tensor: Input array
        axis: Position(s) where dimensions should be removed

    Returns:
        MLX array with squeezed dimensions
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    tensor_array = Tensor.convert_to_tensor(tensor)

    if axis is None:
        return mx.squeeze(tensor_array)

    return mx.squeeze(tensor_array, axis)

def tile(tensor : TensorLike, reps : ShapeLike) -> mx.array:
    """
    Construct an MLX array by tiling a given array.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input array
        reps: Number of repetitions for each dimension
        
    Returns:
        Tiled MLX array
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    tensor_array = Tensor.convert_to_tensor(tensor)
    return mx.tile(tensor_array, reps)

def pad(tensor: TensorLike, paddings, mode='constant', constant_values=0):
    """
    Pad a tensor with a constant value.

    Args:
        tensor: Input tensor
        paddings: Sequence of sequences of integers specifying the padding for each dimension
                Each inner sequence should contain two integers: [pad_before, pad_after]
        mode: Padding mode. MLX supports 'constant' and 'edge'. Default is 'constant'.
        constant_values: Value to pad with when mode is 'constant'. Default is 0.

    Returns:
        Padded tensor
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()

    tensor_array = Tensor.convert_to_tensor(tensor)

    # Convert paddings to the format expected by mx.pad
    # MLX expects a tuple of (pad_before, pad_after) for each dimension
    pad_width = tuple(tuple(p) for p in paddings)

    # MLX supports 'constant' and 'edge' modes
    valid_mode = 'constant'
    if mode in ['constant', 'edge']:
        valid_mode = mode

    # Pad the tensor using MLX's pad function with the correct argument order
    return mx.pad(tensor_array, pad_width, valid_mode, constant_values)