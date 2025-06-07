"""
Backend-agnostic implementation of tensor feature operations.

This module provides tensor feature operations using the ops abstraction layer,
making it compatible with all backends (NumPy, PyTorch, MLX).
"""

from typing import Any, Optional, Union, Sequence, List, Tuple, cast
import numpy as np # Import numpy for coordinate construction

from ember_ml import ops
from ember_ml.nn import tensor

# --- Standalone Functions ---

def one_hot(
    indices: Any,
    num_classes: int,
    *,
    axis: int = -1,
    dtype: Any = None
) -> Any:
    """
    Create a one-hot tensor using tensor_scatter_nd_update.

    Args:
        indices: A tensor of indices (non-negative integers).
        num_classes: The number of classes (depth of the one-hot dimension).
        axis: The axis to place the one-hot dimension. Defaults to -1.
        dtype: The data type of the output tensor. Defaults to tensor.float32.

    Returns:
        A tensor with one-hot encoding.
    """
    # Convert indices to tensor and ensure integer type
    indices_tensor = tensor.cast(tensor.convert_to_tensor(indices), tensor.int32)
    input_shape = tensor.shape(indices_tensor)
    rank = len(input_shape)

    # Handle negative axis
    if axis < 0:
        axis = rank + axis + 1

    # Determine output shape
    output_shape = list(input_shape)
    output_shape.insert(axis, num_classes)

    # Create the coordinate tensor for scattering
    # This involves combining the original coordinates with the one-hot indices
    coords = []
    # Create meshgrid for original dimensions' coordinates
    original_coords_ranges = [tensor.arange(s) for s in input_shape]
    meshgrid_coords = tensor.meshgrid(*original_coords_ranges, indexing='ij')

    # Flatten meshgrid coordinates and original indices
    flat_meshgrid_coords = [tensor.reshape(coord, (-1,)) for coord in meshgrid_coords]
    flat_indices = tensor.reshape(indices_tensor, (-1,))

    # Construct the final coordinates array
    final_coords_list = []
    coord_idx = 0
    for d in range(rank + 1):
        if d == axis:
            final_coords_list.append(flat_indices)
        else:
            final_coords_list.append(flat_meshgrid_coords[coord_idx])
            coord_idx += 1

    # Stack coordinates along the last axis -> shape (N, rank+1)
    # Use numpy stack temporarily if tensor.stack has issues or different signature
    try:
         final_coords_tensor = tensor.stack(final_coords_list, axis=-1)
    except Exception:
         # Fallback or alternative stacking method if tensor.stack fails/differs
         np_coords = [tensor.to_numpy(c) for c in final_coords_list]
         final_coords_tensor = tensor.convert_to_tensor(tensor.stack(np_coords, axis=-1))


    # Create the updates tensor (all ones)
    num_updates = tensor.shape(flat_indices)[0]
    if dtype is None:
        dtype = tensor.float32 # Default dtype
    updates = tensor.ones((num_updates,), dtype=dtype)

    # Create the base tensor of zeros
    zeros_tensor = tensor.zeros(tuple(output_shape), dtype=dtype)

    # Perform the scatter operation
    # Ensure tensor_scatter_nd_update exists and handles this correctly
    if hasattr(tensor, 'tensor_scatter_nd_update'):
        one_hot_tensor = tensor.tensor_scatter_nd_update(zeros_tensor, final_coords_tensor, updates)
    else:
        # Fallback or error if the required scatter function isn't available
        raise NotImplementedError("tensor.tensor_scatter_nd_update is required but not found/aliased.")

    return one_hot_tensor

# scatter function removed as it belongs in nn.tensor

__all__ = [
    'one_hot' # Removed trailing comma
    # scatter removed
]
