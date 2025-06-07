"""NumPy tensor indexing operations."""



import numpy as np

from typing import Any, List, Literal, Optional, Sequence, Union

from ember_ml.backend.numpy.types import (
    TensorLike, Shape
)

def slice_tensor(tensor: TensorLike, starts: Shape, sizes: Shape) -> np.ndarray:
    """
    Extract a slice from a tensor.
    
    Args:
        data: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)

    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        # Convert to tensor to avoid precision-reducing casts
        start_tensor = np.array(start, dtype=np.int32)
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            # Calculate the remaining size for this dimension
            remaining_size = tensor_array.shape[i] - start_tensor.item()
            # Create a slice from start to start + remaining_size
            slice_obj = slice(start_tensor.item(), None)
            slice_objects.append(slice_obj)
        else:
            # Convert size to tensor to avoid precision-reducing casts
            size_tensor = np.array(size, dtype=np.int32)
            end_tensor = np.add(start_tensor, size_tensor)
            # Use Python's built-in slice function, not our slice_tensor function
            slice_obj = slice(start_tensor.item(), end_tensor.item())
            slice_objects.append(slice_obj)
    
    # Extract the slice
    return tensor_array[tuple(slice_objects)]


def gather(tensor: TensorLike, indices: TensorLike, axis: int = 0) -> np.ndarray:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        data: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(np.int32)
    
    # Use take operation for gathering
    return np.take(tensor_array, indices_int, axis=axis)

def tensor_scatter_nd_update(tensor: TensorLike, indices: TensorLike, updates: TensorLike) -> np.ndarray:
    """
    Update tensor elements at given indices.
    
    Args:
        tensor: Input tensor to update
        indices: N-dimensional indices where to update values
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    indices_array = Tensor.convert_to_tensor(indices)
    updates_array = Tensor.convert_to_tensor(updates)
    
    # Create a copy of the tensor
    result = tensor_array.copy()
    
    # Convert indices to integer lists for safe indexing
    if indices_array.ndim == 1:
        indices_list = [indices_array.tolist()]
    else:
        # Handle multi-dimensional indices
        if len(indices_array.shape) > 1:
            # Convert each index to a list
            indices_list = [tuple(idx.tolist()) for idx in indices_array]
        else:
            indices_list = [indices_array.tolist()]
    
    # Update the tensor using our slice_update function
    for i, idx in enumerate(indices_list):
        result = slice_update(result, idx, updates_array[i])
    
    return result

def slice_update(tensor: TensorLike, slices: TensorLike, updates: Optional[TensorLike] = None) -> np.ndarray:
    """Update a tensor at specific indices."""
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)

    # Handle different types for slices
    # If slices is already a type that NumPy indexing understands, use it directly
    if isinstance(slices, (slice, int, np.integer, tuple)) or (isinstance(slices, np.ndarray) and slices.dtype == bool):
        indices_to_use = slices
    else:
        # Attempt conversion for list/array-like integer indices
        try:
            slices_array = Tensor.convert_to_tensor(slices)
            # Ensure integer or boolean type for indexing
            if not np.issubdtype(slices_array.dtype, np.integer) and slices_array.dtype != bool:
                 raise TypeError(f"Index array must be integer or boolean type, got {slices_array.dtype}")
            indices_to_use = slices_array
        except (ValueError, TypeError) as e:
            raise TypeError(f"Unsupported slice/index type for NumPy backend: {type(slices)} - {e}")

    if updates is None:
        # Perform slicing
        return tensor_array[indices_to_use]

    # Perform update
    updates_array = Tensor.convert_to_tensor(updates)
    result = tensor_array.copy()
    result[indices_to_use] = updates_array
    return result

def scatter_nd(self: np.ndarray, dim: int, index: np.ndarray, src: Union[np.ndarray, float, int]) -> np.ndarray:
    """
    Writes all values from the tensor src into self at the indices specified in the index tensor.

    Args:
        dim: The axis along which to index
        index: The indices of elements to scatter
        src: The source element(s) to scatter

    Returns:
        self with values from src scattered
    """
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    index_array = Tensor.convert_to_tensor(index)
    
    # Validations
    if index_array.dtype != np.dtype('int32') and index_array.dtype != np.dtype('int64'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index_array.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Handle negative dimension
        dim = self.ndim + dim
    
    # Check cross-section shapes
    idx_xsection_shape = index_array.shape[:dim] + index_array.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(f"Except for dimension {dim}, all dimensions of index and output should be the same size")
    
    # Check index ranges
    if (index_array >= self.shape[dim]).any() or (index_array < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    # Helper function to create slices
    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return tuple(slc)

    # Create advanced indexing
    # Convert multidimensional indices to linear indices for numpy advanced indexing
    indices = np.indices(idx_xsection_shape)
    indices = [indices[i].reshape(-1) for i in range(indices.shape[0])]
    
    # Add the actual indices to use for scattering
    idx_list = []
    for i in range(index_array.shape[dim]):
        # Get indices at this position along dim
        idx_at_pos = index_array[make_slice(index_array, dim, i)].reshape(-1)
        # Combine with other dimensional indices
        dim_indices = indices.copy()
        dim_indices.insert(dim, idx_at_pos)
        idx_list.append(tuple(dim_indices))
    
    # Combine all indices
    idx = list(zip(*[np.concatenate([idx_list[j][i] for j in range(len(idx_list))])
                    for i in range(len(idx_list[0]))]))
    
    # Handle scalar or array source
    if np.isscalar(src):
        self[tuple(idx)] = src
    else:
        src_array = Tensor.convert_to_tensor(src)
        if index_array.shape[dim] > src_array.shape[dim]:
            raise IndexError(f"Dimension {dim} of index cannot be bigger than that of src")
        
        src_xsection_shape = src_array.shape[:dim] + src_array.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError(f"Except for dimension {dim}, all dimensions of index and src should be the same size")
        
        # Create source indices
        src_idx = []
        for i in range(len(indices)):
            src_idx.append(np.concatenate([np.full_like(indices[0], j) for j in range(index_array.shape[dim])]))
        src_idx.insert(dim, np.repeat(np.arange(index_array.shape[dim]), np.prod(idx_xsection_shape)))
        
        self[tuple(idx)] = src_array[tuple(src_idx)]
    
    return self

def scatter(data: TensorLike, indices: TensorLike, dim_size: Optional[Union[int, np.ndarray]] = None,
           aggr: Literal["add", "max", "min", "mean", "softmax"] = "add", axis: int = 0) -> np.ndarray:
    """
    Scatter values into a new tensor.
    
    Args:
        data: Input tensor with values to scatter
        indices: Indices where to scatter the values
        dim_size: Size of output dimension (if None, computed from indices)
        aggr: Aggregation method to use ("add", "max", "min", "mean", "softmax")
        axis: Axis along which to scatter
        
    Returns:
        New tensor with scattered values
    """
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    data_array = Tensor.convert_to_tensor(data)
    indices_array = Tensor.convert_to_tensor(indices)
    
    # Ensure indices are integers
    indices_int = indices_array.astype(np.int32)
    
    # Handle dim_size to determine output shape
    if dim_size is None:
        # If dim_size is not provided, infer it from the max index
        computed_dim_size_for_axis = int(np.max(indices_int) + 1)
        output_shape = list(data_array.shape)
        output_shape[axis] = computed_dim_size_for_axis
    elif isinstance(dim_size, int):
        # If dim_size is an integer, use it for the specified axis
        output_shape = list(data_array.shape)
        output_shape[axis] = dim_size
    elif isinstance(dim_size, (list, tuple, np.ndarray)):
        # If dim_size is a sequence (like a shape tuple), use it directly as the output shape
        # This handles the case where the test provides the full output shape
        output_shape = tuple(dim_size)
        # Basic validation
        if len(output_shape) != data_array.ndim:
             raise ValueError(f"Provided shape dim_size {output_shape} has different rank ({len(output_shape)}) than data ({data_array.ndim})")
    else:
        raise TypeError(f"Unsupported type for dim_size: {type(dim_size)}")
    
    # Initialize output tensor based on operation
    if aggr == "add" or aggr == "mean" or aggr == "softmax":
        output = np.zeros(output_shape, dtype=data_array.dtype)
    elif aggr == "max":
        output = np.full(output_shape, -np.inf, dtype=data_array.dtype)
    elif aggr == "min":
        output = np.full(output_shape, np.inf, dtype=data_array.dtype)
    else:
        raise ValueError(f"Unknown operation: {aggr}")
    
    # For "mean" aggregation, we need to track counts
    if aggr == "mean":
        # Initialize count tensor
        count = np.zeros(output_shape, dtype=np.int32)
        
        # Use direct indexing for scattering with aggregation
        if indices_int.ndim == 1:
            # Create slices for advanced indexing
            for i, idx in enumerate(indices_int):
                # Select source value
                if axis == 0:
                    src_value = data_array[i]
                else:
                    idx_tuple = tuple(slice(None) if j != axis else i for j in range(data_array.ndim))
                    src_value = data_array[idx_tuple]
                
                # Create index for scatter operation
                scatter_idx = [slice(None)] * output.ndim
                scatter_idx[axis] = idx
                scatter_idx = tuple(scatter_idx)
                
                # Add value and increment count
                output[scatter_idx] += src_value
                count[scatter_idx] += 1
            
            # Calculate mean by dividing by count
            # Avoid division by zero
            count = np.where(count == 0, np.ones_like(count), count)
            output = np.divide(output, count)
        
        return output
    
    # For other aggregation methods
    if indices_int.ndim == 1:
        # Create slices for advanced indexing
        for i, idx in enumerate(indices_int):
            # Select source value
            if axis == 0:
                src_value = data_array[i]
            else:
                idx_tuple = tuple(slice(None) if j != axis else i for j in range(data_array.ndim))
                src_value = data_array[idx_tuple]
            
            # Create index for scatter operation
            scatter_idx = [slice(None)] * output.ndim
            scatter_idx[axis] = idx
            scatter_idx = tuple(scatter_idx)
            
            # Apply operation based on aggregation method
            if aggr == "add":
                output[scatter_idx] += src_value
            elif aggr == "max":
                output[scatter_idx] = np.maximum(output[scatter_idx], src_value)
            elif aggr == "min":
                output[scatter_idx] = np.minimum(output[scatter_idx], src_value)
            elif aggr == "softmax":
                output[scatter_idx] += src_value
    
    return output

# Helper functions for scatter operations
def scatter_add(src: TensorLike, index: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> np.ndarray:
    """
    Scatter values using addition.
    
    Args:
        src: Input tensor with values to scatter
        index: Indices where to scatter the values
        dim_size: Size of output dimension (if None, computed from indices)
        axis: Axis along which to scatter
        
    Returns:
        New tensor with values scattered using addition
    """
    return scatter(src, index, dim_size, "add", axis)

def scatter_max(src: TensorLike, index: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> np.ndarray:
    """
    Scatter values using maximum.
    
    Args:
        src: Input tensor with values to scatter
        index: Indices where to scatter the values
        dim_size: Size of output dimension (if None, computed from indices)
        axis: Axis along which to scatter
        
    Returns:
        New tensor with values scattered using maximum
    """
    return scatter(src, index, dim_size, "max", axis)

def scatter_min(src: TensorLike, index: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> np.ndarray:
    """
    Scatter values using minimum.
    
    Args:
        src: Input tensor with values to scatter
        index: Indices where to scatter the values
        dim_size: Size of output dimension (if None, computed from indices)
        axis: Axis along which to scatter
        
    Returns:
        New tensor with values scattered using minimum
    """
    return scatter(src, index, dim_size, "min", axis)

def scatter_mean(values: TensorLike, index: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> np.ndarray:
    """
    Scatter values and compute mean.
    
    Args:
        values: Input tensor with values to scatter
        index: Indices where to scatter the values
        dim_size: Size of output dimension (if None, computed from indices)
        axis: Axis along which to scatter
        
    Returns:
        New tensor with values scattered using mean aggregation
    """
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    values_array = Tensor.convert_to_tensor(values)
    indices_array = Tensor.convert_to_tensor(index)
    
    # Handle dim_size
    if dim_size is None:
        computed_dim_size = int(np.max(indices_array) + 1)
    else:
        computed_dim_size = int(dim_size)
    
    # Create output shape
    output_shape = list(values_array.shape)
    output_shape[axis] = computed_dim_size
    
    # Initialize arrays for sum and count
    sum_result = np.zeros(output_shape, dtype=values_array.dtype)
    count = np.zeros(output_shape, dtype=np.int32)
    
    # Directly compute sum and count for each index
    if indices_array.ndim == 1:
        for i, idx in enumerate(indices_array):
            # Select source value
            if axis == 0:
                src_value = values_array[i]
            else:
                idx_tuple = tuple(slice(None) if j != axis else i for j in range(values_array.ndim))
                src_value = values_array[idx_tuple]
            
            # Create index for scatter operation
            scatter_idx = [slice(None)] * sum_result.ndim
            scatter_idx[axis] = idx
            scatter_idx = tuple(scatter_idx)
            
            # Update sum and count
            sum_result[scatter_idx] += src_value
            count[scatter_idx] += 1
    
    # Avoid division by zero
    count = np.where(count == 0, np.ones_like(count), count)
    
    # Compute mean
    return np.divide(sum_result, count)

def scatter_softmax(values: TensorLike, index: TensorLike, dim_size: Optional[int] = None, axis: int = 0) -> np.ndarray:
    """
    Scatter values and compute softmax.
    
    Args:
        values: Input tensor with values to scatter
        index: Indices where to scatter the values
        dim_size: Size of output dimension (if None, computed from indices)
        axis: Axis along which to scatter
        
    Returns:
        New tensor with values scattered using softmax aggregation
    """
    # First compute max for numerical stability
    max_vals = scatter(values, index, dim_size, "max", axis)
    
    # Compute exp(x - max)
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    values_array = Tensor.convert_to_tensor(values)
    index_array = Tensor.convert_to_tensor(index)
    
    # Gather the max values for each index
    gathered_max = np.zeros_like(values_array)
    for i, idx in enumerate(index_array):
        if axis == 0:
            gathered_max[i] = max_vals[idx]
        else:
            src_idx = tuple(slice(None) if j != axis else i for j in range(values_array.ndim))
            dst_idx = tuple(slice(None) if j != axis else idx for j in range(values_array.ndim))
            gathered_max[src_idx] = max_vals[dst_idx]
    
    # Compute exp(x - max) for numerical stability
    exp_vals = np.exp(values_array - gathered_max)
    
    # Sum exp values
    sum_exp = scatter(exp_vals, index, dim_size, "add", axis)
    
    # Gather sum_exp values
    gathered_sum = np.zeros_like(exp_vals)
    for i, idx in enumerate(index_array):
        if axis == 0:
            gathered_sum[i] = sum_exp[idx]
        else:
            src_idx = tuple(slice(None) if j != axis else i for j in range(exp_vals.ndim))
            dst_idx = tuple(slice(None) if j != axis else idx for j in range(exp_vals.ndim))
            gathered_sum[src_idx] = sum_exp[dst_idx]
    
    # Compute softmax
    return np.divide(exp_vals, gathered_sum)

def nonzero(tensor: TensorLike) -> np.ndarray:
    """
    Returns the indices of the elements that are non-zero.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor containing the indices of the non-zero elements
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Get indices of non-zero elements
    return np.array(np.nonzero(tensor_array)).T

# No longer need the alias since we're using the built-in slice directly

def index_update(tensor: TensorLike, *indices, value: TensorLike) -> np.ndarray:
    """
    Update the tensor at the specified indices with the given value.
    
    Args:
        tensor: The tensor to update
        *indices: The indices to update (can be integers, slices, or arrays)
        value: The value to set at the specified indices
        
    Returns:
        Updated tensor
    """
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(tensor)
    value_array = Tensor.convert_to_tensor(value)
    
    # Create a copy of the tensor to avoid in-place modification
    result = tensor_array.copy()
    
    # Handle different indexing patterns
    if len(indices) == 1:
        # Single index
        idx = indices[0]
        result[idx] = value_array
    elif len(indices) == 2:
        # Two indices (common case for 2D tensors)
        i, j = indices
        result[i, j] = value_array
    elif len(indices) == 3:
        # Three indices
        i, j, k = indices
        result[i, j, k] = value_array
    else:
        # General case
        idx_tuple = tuple(indices)
        result[idx_tuple] = value_array
    
    return result