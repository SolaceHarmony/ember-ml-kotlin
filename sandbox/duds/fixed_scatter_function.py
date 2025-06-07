"""
Fixed implementation of the scatter function for the MLX backend.
"""

import mlx.core as mx
from typing import Union, Literal, Tuple, List, Any

# Type aliases for clarity
TensorLike = Any
ShapeLike = Union[List[int], Tuple[int, ...]]

def fixed_scatter(indices: TensorLike, updates: TensorLike, shape: Union[ShapeLike, int, mx.array],
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
    indices_array = mx.array(indices)
    updates_array = mx.array(updates)
    
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
            shape_val = int(shape.item())
            output_shape = (shape_val,)
        else:
            # For arrays, convert to list of ints
            shape_list = [int(x) for x in shape.tolist()]
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
        if indices_array.ndim == 2:
            # Handle multi-dimensional indices
            for i in range(indices_array.shape[0]):
                # Get current indices
                current_indices = indices_int[i].tolist()
                
                # Get update value
                update_value = updates_array[i] if updates_array.ndim > 0 else updates_array
                
                # Update result
                if aggr == "add":
                    current_value = result[tuple(current_indices)]
                    new_value = current_value + update_value
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
            # Handle 1D indices
            for i in range(indices_array.size):
                # Get current index
                current_idx = int(indices_int[i].item())
                
                # Get update value
                update_value = updates_array[i] if updates_array.ndim > 0 else updates_array
                
                # Update result
                if aggr == "add":
                    current_value = result[current_idx]
                    new_value = current_value + update_value
                    result = mx.slice_update(result, new_value, mx.array([current_idx], dtype=mx.int32), [0])
                elif aggr == "max":
                    current_value = result[current_idx]
                    new_value = mx.maximum(current_value, update_value)
                    result = mx.slice_update(result, new_value, mx.array([current_idx], dtype=mx.int32), [0])
                elif aggr == "min":
                    current_value = result[current_idx]
                    new_value = mx.minimum(current_value, update_value)
                    result = mx.slice_update(result, new_value, mx.array([current_idx], dtype=mx.int32), [0])
    
    return result

def test_fixed_scatter():
    """Test the fixed scatter function."""
    print("Testing fixed_scatter function")
    
    # Test case 1: Create a 3x3 identity matrix
    n = 3
    indices = (mx.arange(n), mx.arange(n))
    updates = mx.ones(n)
    shape = (n, n)
    
    print(f"\nTest case 1: Creating a {n}x{n} identity matrix")
    result = fixed_scatter(indices, updates, shape)
    print(f"Shape: {result.shape}")
    print(f"Content:\n{result}")
    
    # Test case 2: Create a 3x5 rectangular matrix with ones on the diagonal
    n, m = 3, 5
    indices = (mx.arange(n), mx.arange(n))
    updates = mx.ones(n)
    shape = (n, m)
    
    print(f"\nTest case 2: Creating a {n}x{m} matrix with ones on the diagonal")
    result = fixed_scatter(indices, updates, shape)
    print(f"Shape: {result.shape}")
    print(f"Content:\n{result}")
    
    # Test case 3: Create a 5x3 rectangular matrix with ones on the diagonal
    n, m = 5, 3
    indices = (mx.arange(m), mx.arange(m))
    updates = mx.ones(m)
    shape = (n, m)
    
    print(f"\nTest case 3: Creating a {n}x{m} matrix with ones on the diagonal")
    result = fixed_scatter(indices, updates, shape)
    print(f"Shape: {result.shape}")
    print(f"Content:\n{result}")

if __name__ == "__main__":
    test_fixed_scatter()