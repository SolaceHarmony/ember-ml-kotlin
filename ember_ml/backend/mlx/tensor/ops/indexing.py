"""MLX tensor indexing operations."""

import mlx.core as mx

from typing import Union, Optional, Literal, Any, List
from ember_ml.backend.mlx.types import (
    TensorLike, Shape, ShapeLike
)
from ember_ml.backend.mlx.tensor.ops.utility import to_numpy

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Shape) -> mx.array:
    """
    Extract a slice from a tensor.
    
    Args:
        tensor: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Handle scalar starts/sizes by converting to lists
    if isinstance(starts, (int, float)):
        starts = [int(starts)]
    if isinstance(sizes, (int, float)):
        sizes = [int(sizes)]
    
    # Ensure starts and sizes are the same length
    if len(starts) != len(sizes):
        raise ValueError(f"starts and sizes must have the same length, got {len(starts)} and {len(sizes)}")
    
    # Handle -1 in sizes by calculating actual sizes
    tensor_shape = tensor_array.shape
    sizes_list = []
    for i, size in enumerate(sizes):
        if size == -1:
            # Calculate remaining size for this dimension
            remaining_size = tensor_shape[i] - starts[i]
            sizes_list.append(remaining_size)
        else:
            sizes_list.append(size)
    
    # Create a list of slice objects for each dimension
    slices = []
    for start, size in zip(starts, sizes_list):
        slices.append(slice(start, start + size))
    
    # Apply the slices to the tensor
    # For dimensions not specified, use the full range
    full_slices = [slice(None)] * tensor_array.ndim
    for i, s in enumerate(slices):
        if i < tensor_array.ndim:
            full_slices[i] = s
    
    # Extract the slice using standard indexing
    return tensor_array[tuple(full_slices)]

def gather(tensor: TensorLike, indices: TensorLike, axis: int = 0) -> mx.array:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        tensor: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Convert axis to positive index if negative
    if axis < 0:
        ndim = len(tensor_array.shape)
        axis = ndim + axis
        
    # Ensure indices are valid
    max_index = tensor_array.shape[axis]
    indices_int = indices_array.astype(mx.int32)
    
    # Create mask for valid indices
    valid_mask = mx.logical_and(
        mx.greater_equal(indices_int, mx.array(0)),
        mx.less(indices_int, mx.array(max_index))
    )
    
    # Replace invalid indices with 0 (will be masked out)
    safe_indices = mx.where(valid_mask, indices_int, mx.array(0))
    
    # Perform gather operation
    result = mx.take(tensor_array, safe_indices, axis=axis)
    
    # Zero out results from invalid indices if any were found
    if not mx.all(valid_mask).item():  # We need this one .item() for control flow
        # Create a broadcast-compatible mask
        mask_shape = [1] * len(result.shape)
        mask_shape[axis] = valid_mask.shape[0]
        reshaped_mask = mx.reshape(valid_mask, mask_shape)
        zeros = mx.zeros_like(result)
        result = mx.where(reshaped_mask, result, zeros)
        
    return result

def tensor_scatter_nd_update(tensor: TensorLike, indices: TensorLike, updates: TensorLike) -> mx.array:
    """
    Update tensor elements at given indices.
    
    Args:
        tensor: Input tensor to update
        indices: N-dimensional indices where to update values
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = mx.array(tensor_array)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(mx.int32)
    
    # Handle different index dimensions
    if indices_int.ndim == 1:
        # For 1D indices, reshape to 2D with one index per row
        indices_reshaped = mx.reshape(indices_int, (-1, 1))
    else:
        indices_reshaped = indices_int
    
    # Process each index
    for i in range(indices_reshaped.shape[0]):
        # Get current index
        current_idx = indices_reshaped[i]
        # Get axes for slice_update (list of consecutive integers)
        axes = list(range(len(current_idx)))
        
        # Update the tensor using slice_update
        # Convert index to array for slice_update
        idx_array = mx.array(current_idx)
        result = mx.slice_update(result, updates_array[i], idx_array, axes)
    
    return result

def slice_update(tensor: TensorLike, slices: Any, updates: Optional[TensorLike] = None) -> mx.array:
    """
    Update a tensor at specific indices or return a slice.
    
    Args:
        tensor: Input tensor to update
        slices: Indices or slices where to update values
        updates: Values to insert at the specified indices. If None, return slice
        
    Returns:
        Updated tensor or slice of tensor if updates is None
    """
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)

    # If updates is None, return slice using MLX's array indexing
    if updates is None:
        return tensor_array[slices]

    # Convert updates to MLX array
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Handle different types of slice specifications
    if isinstance(slices, (int, float)):
        # Single index - convert to MLX array
        indices = mx.array([int(slices)], dtype=mx.int32)
        axes = [0]
    elif isinstance(slices, (list, tuple)):
        # Multiple indices - convert to MLX array
        indices = mx.array(slices, dtype=mx.int32)
        axes = list(range(len(slices)))
    elif isinstance(slices, mx.array):
        # MLX array of indices
        indices = slices.astype(mx.int32)
        axes = list(range(indices.ndim))
    elif hasattr(slices, 'start') and hasattr(slices, 'stop'):
        # Handle Python slice objects by attributes rather than type
        start = 0 if slices.start is None else slices.start
        stop = tensor_array.shape[0] if slices.stop is None else slices.stop
        step = 1 if slices.step is None else slices.step
        indices = mx.arange(start, stop, step, dtype=mx.int32)
        axes = [0]
    else:
        raise TypeError(f"Unsupported slice type: {type(slices)}")
    
    return mx.slice_update(tensor_array, updates_array, indices, axes)

def scatter(indices: TensorLike, updates: TensorLike, shape: Union[ShapeLike, int, mx.array],
           aggr: Literal["add", "max", "min", "mean", "softmax"] = "add") -> mx.array:
    """
    Scatter values into a new tensor.
    
    Args:
        indices: N-D indices where to scatter the updates
        updates: Values to scatter at the specified indices
        shape: Shape of the output tensor or dimension size for a specific axis
        aggr: Aggregation method to use for duplicate indices
        
    Returns:
        Tensor with scattered values
    """
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    indices_array = Tensor.convert_to_tensor(indices)
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(mx.int32)
    
    # Determine output shape
    if isinstance(shape, (list, tuple)):
        output_shape = tuple(shape)
    elif isinstance(shape, int):
        output_shape = (shape,) * indices_array.shape[-1]
    elif isinstance(shape, mx.array):
        # Convert MLX array shape to Python tuple safely
        # For single values, handle scalar array
        if shape.ndim == 0:
            # Convert scalar MLX array to Python int
            shape_val = int(shape[0])
            output_shape = (shape_val,)
        else:
            # For arrays, convert to list of ints
            shape_list = [int(x) for x in shape]
            output_shape = tuple(shape_list)
    else:
        raise TypeError(f"Unsupported type for shape: {type(shape)}")
    
    # Create a zeros tensor of the appropriate shape
    result = mx.zeros(output_shape, dtype=updates_array.dtype)
    
    # Handle the case where indices is a tuple of arrays (for multi-dimensional indexing)
    if isinstance(indices, tuple) and all(isinstance(idx, (mx.array, list, tuple)) for idx in indices):
        # This is the case used in the eye function: (indices, indices)
        # Ensure all index arrays have the same length
        index_lengths = [len(idx) if hasattr(idx, '__len__') else 1 for idx in indices]
        if len(set(index_lengths)) > 1:
            raise ValueError("All index arrays must have the same length")
        
        num_indices = index_lengths[0]
        num_dims = len(indices)
        
        # Check if dimensions match
        if num_dims != len(output_shape):
            raise ValueError(f"Number of index dimensions ({num_dims}) must match output shape dimensions ({len(output_shape)})")
        
        # Process each index
        for i in range(num_indices):
            # Extract the i-th index from each dimension
            current_indices = []
            for dim in range(num_dims):
                idx = indices[dim]
                if isinstance(idx, (list, tuple)):
                    current_indices.append(int(idx[i]))
                elif isinstance(idx, mx.array):
                    current_indices.append(int(idx[i].item()))
                else:
                    raise TypeError(f"Unsupported index type: {type(idx)}")
            
            # Get the update value
            if updates_array.ndim == 0:
                # Scalar update
                update_value = updates_array
            elif updates_array.ndim == 1 and updates_array.shape[0] == num_indices:
                # Vector of updates
                update_value = updates_array[i]
            else:
                raise ValueError(f"Updates shape {updates_array.shape} doesn't match number of indices {num_indices}")
            
            # Update the result tensor
            if aggr == "add":
                # Get current value
                current_value = result[tuple(current_indices)]
                # Add update
                new_value = current_value + update_value
                # Update result
                indices_mx = mx.array(current_indices, dtype=mx.int32)
                axes = list(range(len(current_indices)))
                result = mx.slice_update(result, new_value, indices_mx, axes)
            elif aggr == "max":
                current_value = result[tuple(current_indices)]
                new_value = mx.maximum(current_value, update_value)
                indices_mx = mx.array(current_indices, dtype=mx.int32)
                axes = list(range(len(current_indices)))
                result = mx.slice_update(result, new_value, indices_mx, axes)
            elif aggr == "min":
                current_value = result[tuple(current_indices)]
                new_value = mx.minimum(current_value, update_value)
                indices_mx = mx.array(current_indices, dtype=mx.int32)
                axes = list(range(len(current_indices)))
                result = mx.slice_update(result, new_value, indices_mx, axes)
    else:
        # Handle standard case where indices is a single array
        # Determine the number of updates to process
        num_updates = updates_array.size if updates_array.ndim == 0 else updates_array.shape[0]
        
        if indices_array.ndim == 2:
            # Handle multi-dimensional indices
            for i in range(num_updates):
                # Convert indices safely using tolist() and ensure list format
                current_indices = indices_int[i].tolist()
                if not isinstance(current_indices, list):
                    current_indices = [current_indices]
    
                # Prepare the update value (ensure it's an MLX array)
                update_value = mx.array(updates_array.flatten()[i]) # Use flatten() to get element
    
                # Use slice_update to update the element at the multi-dimensional index
                # slice_update expects start_indices and axes
                # For a single element update at a multi-dimensional index,
                # start_indices are the indices themselves, and axes are the dimensions being indexed.
                axes = list(range(len(current_indices)))
                start_indices = mx.array(current_indices, dtype=mx.int32)
    
                if aggr == "add":
                    # Read the current value at the index, add the update, and write back
                    current_value = result[tuple(current_indices)]
                    new_value = mx.add(current_value, update_value)
                    result = mx.slice_update(result, new_value, start_indices, axes)
                elif aggr == "max":
                     current_value = result[tuple(current_indices)]
                     new_value = mx.maximum(current_value, update_value)
                     result = mx.slice_update(result, new_value, start_indices, axes)
                elif aggr == "min":
                     current_value = result[tuple(current_indices)]
                     new_value = mx.minimum(current_value, update_value)
                     result = mx.slice_update(result, new_value, start_indices, axes)
        else:
            # Handle 1D indices
            for i in range(num_updates):
                # Convert index safely using tolist() which handles scalar arrays correctly
                current_idx = indices_int[i].tolist()
                idx_array = mx.array([current_idx], dtype=mx.int32)
    
                # Prepare the update value (ensure it's an MLX array)
                update_value = mx.array(updates_array.flatten()[i]) # Use flatten() to get element
    
                if aggr == "add":
                    current = result[current_idx]
                    result = mx.slice_update(result, mx.add(current, update_value), idx_array, [0])
                elif aggr == "max":
                    current = result[current_idx]
                    result = mx.slice_update(result, mx.maximum(current, update_value), idx_array, [0])
                elif aggr == "min":
                    current = result[current_idx]
                    result = mx.slice_update(result, mx.minimum(current, update_value), idx_array, [0])
    
    return result

def scatter_op(src: mx.array, index: mx.array, dim_size: int,
                axis: int, op: Literal["add", "max", "min", "softmax"]) -> mx.array:
    """
    Helper function for scatter operations.
    
    Args:
        src: Source tensor containing values to scatter
        index: Index tensor specifying where to scatter values
        dim_size: Size of the dimension along the scatter axis
        axis: Dimension along which to scatter
        op: Operation to perform when there are multiple values for an index
            ("add", "max", "min", "softmax")
            
    Returns:
        Tensor with scattered values aggregated according to op
    """
    # Handle empty case
    if dim_size == 0 or index.size == 0:
        # Return empty array with appropriate shape
        output_shape = list(src.shape)
        if axis < 0:
            axis = len(output_shape) + axis
        output_shape[axis] = 0
        return mx.zeros(output_shape, dtype=src.dtype)
    
    # Get shape of output tensor
    output_shape = list(src.shape)
    if axis < 0:
        axis = len(output_shape) + axis
    output_shape[axis] = dim_size
    
    # Initialize output tensor based on operation
    if op == "add":
        out = mx.zeros(output_shape, dtype=src.dtype)
    elif op in ["max", "softmax"]:
        out = mx.full(output_shape, float('-inf'), dtype=src.dtype)
    elif op == "min":
        out = mx.full(output_shape, float('inf'), dtype=src.dtype)
    else:
        raise ValueError(f"Unknown operation: {op}")
    
    # Convert indices to integers and ensure 1D
    indices_1d = mx.reshape(index.astype(mx.int32), (-1,))
    
    # Create a mask for valid indices
    valid_mask = mx.logical_and(
        mx.greater_equal(indices_1d, mx.array(0)),
        mx.less(indices_1d, mx.array(dim_size))
    )
    
    # Only process valid indices
    valid_indices = mx.where(valid_mask, indices_1d, mx.array(0))
    
    # Process each valid index
    for i in range(indices_1d.size):
        if not valid_mask[i].item():  # We need this one .item() call for control flow
            continue
            
        idx = valid_indices[i]
        val_i = src[i]
        
        # Create arrays for slice_update
        idx_array = mx.array([idx], dtype=mx.int32)
        axes = mx.array([axis], dtype=mx.int32)
        
        if op == "add":
            current = mx.take(out, idx_array, axis=axis)
            update_val = mx.add(current, val_i)
        elif op == "max":
            current = mx.take(out, idx_array, axis=axis)
            update_val = mx.maximum(current, val_i)
        elif op == "min":
            current = mx.take(out, idx_array, axis=axis)
            update_val = mx.minimum(current, val_i)
        else:
            raise ValueError(f"Operation {op} not implemented")
            
        out = mx.slice_update(out, update_val, idx_array, [axis])
    
    return out

def scatter_add(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values using addition for duplicate indices.
    
    Args:
        src: Source tensor containing values to scatter
        index: Index tensor specifying where to scatter values
        dim_size: Size of the output tensor along the scatter dimension
        axis: Dimension along which to scatter (default: 0)
        
    Returns:
        Tensor where values at duplicate indices are added together
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    # Enforce Python int type for dim_size
    return scatter_op(src_array, index_array, int(dim_size), axis, "add")

def scatter_max(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values using maximum for duplicate indices.
    
    Args:
        src: Source tensor containing values to scatter
        index: Index tensor specifying where to scatter values
        dim_size: Size of the output tensor along the scatter dimension
        axis: Dimension along which to scatter (default: 0)
        
    Returns:
        Tensor where values at duplicate indices are replaced with their maximum
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    # Enforce Python int type for dim_size
    return scatter_op(src_array, index_array, int(dim_size), axis, "max")

def scatter_min(src: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """
    Scatter values using minimum for duplicate indices.
    
    Args:
        src: Source tensor containing values to scatter
        index: Index tensor specifying where to scatter values
        dim_size: Size of the output tensor along the scatter dimension
        axis: Dimension along which to scatter (default: 0)
        
    Returns:
        Tensor where values at duplicate indices are replaced with their minimum
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    src_array = Tensor.convert_to_tensor(src)
    index_array = Tensor.convert_to_tensor(index)
    # Enforce Python int type for dim_size
    return scatter_op(src_array, index_array, int(dim_size), axis, "min")

def scatter_mean(values: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values and compute mean."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    
    # First compute sum
    sum_result = scatter_op(values_array, index_array, dim_size, axis, "add")
    
    # Then compute count using ones
    ones = mx.ones_like(values_array)
    count = scatter_op(ones, index_array, dim_size, axis, "add")
    
    # Create mask for zero counts
    zero_mask = mx.equal(count, mx.zeros_like(count))
    safe_count = mx.where(zero_mask, mx.ones_like(count), count)
    
    # Compute mean with safe division
    mean_result = mx.divide(sum_result, safe_count)
    
    # Zero out any results where count was zero
    return mx.where(zero_mask, mx.zeros_like(mean_result), mean_result)

def scatter_softmax(values: TensorLike, index: TensorLike, dim_size: int, axis: int = 0) -> mx.array:
    """Scatter values and compute softmax."""
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    
    # First compute max for numerical stability
    max_vals = scatter_op(values_array, index_array, dim_size, axis, "max")
    
    # Create mask for valid indices
    valid_mask = mx.logical_and(
        mx.greater_equal(index_array, mx.array(0)),
        mx.less(index_array, mx.array(dim_size))
    )
    
    # Compute exp(x - max) with proper broadcasting
    max_broadcast = mx.take(max_vals, index_array, axis=axis)
    exp_vals = mx.exp(mx.subtract(values_array, max_broadcast))
    
    # Sum exp values
    sum_exp = scatter_op(exp_vals, index_array, dim_size, axis, "add")
    
    # Broadcast sum_exp to match exp_vals shape
    sum_exp_broadcast = mx.take(sum_exp, index_array, axis=axis)
    
    # Compute softmax with safe division
    safe_sum = mx.where(
        mx.equal(sum_exp_broadcast, mx.array(0.0)),
        mx.ones_like(sum_exp_broadcast),
        sum_exp_broadcast
    )
    softmax_result = mx.divide(exp_vals, safe_sum)
    
    # Zero out invalid indices
    return mx.where(valid_mask, softmax_result, mx.zeros_like(softmax_result))
# No compatibility wrapper for slice_tensor to avoid conflicts with built-in slice

def nonzero(tensor: TensorLike) -> mx.array:
    """
    Returns the indices of the elements that are non-zero.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor containing the indices of the non-zero elements
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Create a boolean mask for non-zero elements
    mask = mx.not_equal(tensor_array, mx.array(0))
    
    # MLX doesn't support boolean indexing directly, so we'll use NumPy as a fallback
    # Convert to NumPy, use np.nonzero, then convert back to MLX
    
    # Convert to NumPy array
    import numpy as np
    np_array = to_numpy(tensor_array)
    
    # Use NumPy's nonzero function
    np_indices = np.nonzero(np_array)
    
    # Stack the indices to get the expected format
    if len(np_indices) == 1:
        # For 1D tensors, reshape to Nx1
        result_np = np.reshape(np_indices[0], (-1, 1))
    else:
        # For multi-dimensional tensors, stack the indices
        result_np = np.stack(np_indices, axis=1)
    
    # Convert back to MLX array
    if result_np.size == 0:
        # Handle empty case
        return mx.zeros((0, len(tensor_array.shape)), dtype=mx.int32)
    return mx.array(result_np, dtype=mx.int32)

def meshgrid(*arrays: TensorLike, indexing: Literal['xy', 'ij'] = 'xy') -> List[mx.array]:
    """
    Generate multidimensional coordinate grids from 1-D coordinate arrays.

    Args:
        *arrays: 1-D arrays representing the coordinates of a grid.
        indexing: Cartesian ('xy', default) or matrix ('ij') indexing.

    Returns:
        List of MLX arrays representing the coordinate grids.
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    mlx_arrays = [Tensor.convert_to_tensor(arr) for arr in arrays]
    # MLX meshgrid defaults to 'ij', opposite of NumPy/Torch default 'xy'
    # We match the NumPy/Torch default ('xy') for our common API
    mlx_indexing = 'ij' if indexing == 'xy' else 'xy'
    result = mx.meshgrid(*mlx_arrays, indexing=mlx_indexing)
    # Convert MLX array to list of arrays if needed
    return list(result) if isinstance(result, (list, tuple)) else [result]

def index_update(tensor: TensorLike, *indices, value: TensorLike) -> mx.array:
    """
    Update the tensor at the specified indices with the given value.
    
    Args:
        tensor: The tensor to update
        *indices: The indices to update (can be integers, slices, or arrays)
        value: The value to set at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    value_array = Tensor.convert_to_tensor(value)
    
    # Create a copy of the tensor
    result = mx.array(tensor_array)
    
    # Handle different indexing patterns
    if len(indices) == 1:
        # Single index
        idx = indices[0]
        # Use MLX's at/set for updating
        result = result.at[idx].add(value_array)
    elif len(indices) == 2:
        # Two indices (common case for 2D tensors)
        i, j = indices
        result = result.at[i, j].add(value_array)
    elif len(indices) == 3:
        # Three indices
        i, j, k = indices
        result = result.at[i, j, k].add(value_array)
    else:
        # General case
        idx_tuple = tuple(indices)
        result = result.at[idx_tuple].add(value_array)
    
    return result
