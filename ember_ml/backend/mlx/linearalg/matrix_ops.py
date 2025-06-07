"""
MLX matrix linear algebra operations for ember_ml.

This module provides MLX operations.
"""

import mlx.core as mx
from typing import Union, Tuple, Optional, Literal

# Import from tensor_ops
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.linearalg.svd_ops import svd # Corrected path
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import OrdLike

dtype_obj = MLXDType()

def norm(x: TensorLike, 
         ord: OrdLike = None, 
         axis: Optional[Union[int, Tuple[int, ...]]] = None, 
         keepdim: bool = False) -> mx.array:
    """
    Compute the matrix or vector norm.
    
    Args:
        x: Input matrix or vector
        ord: Order of the norm
        axis: Axis along which to compute the norm
        keepdim: Whether to keep the reduced dimensions
    
    Returns:
        Norm of the matrix or vector
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    TensorInstance = MLXTensor()
    x_array = TensorInstance.convert_to_tensor(x)
    
    # Default values
    if ord is None:
        if axis is None:
            # Default to Frobenius norm for matrices, L2 norm for vectors
            if x_array.ndim > 1:
                ord = 'fro'
            else:
                ord = 2
        else:
            # Default to L2 norm along the specified axis
            ord = 2
    
    # Vector norm
    if axis is not None or x_array.ndim == 1:
        if axis is None:
            axis = 0
        
        if ord == 'inf':
            # L-infinity norm (maximum absolute value)
            if isinstance(axis, int):
                result = mx.max(mx.abs(x_array), axis=axis)
            else:
                raise ValueError("For L-infinity norm, axis must be an integer")
        elif ord == 1:
            # L1 norm (sum of absolute values)
            result = mx.sum(mx.abs(x_array), axis=axis)
        elif ord == 2:
            # L2 norm (Euclidean norm)
            result = mx.sqrt(mx.sum(mx.square(x_array), axis=axis))
        else:
            # General Lp norm
            if isinstance(ord, (int, float)):
                result = mx.power(
                    mx.sum(mx.power(mx.abs(x_array), ord), axis=axis),
                    mx.divide(mx.array(1.0), mx.array(ord))
                )
            else:
                # Handle case where ord is a string
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Matrix norm
    else:
        if ord == 'fro':
            # Frobenius norm
            result = mx.sqrt(mx.sum(mx.square(x_array)))
        elif ord == 'nuc':
            # Nuclear norm (sum of singular values)
            s_values = svd(x_array, compute_uv=False)
            if isinstance(s_values, tuple):
                # Handle case where svd returns a tuple
                result = mx.sum(s_values[0])
            else:
                # Handle case where svd returns an array
                result = mx.sum(s_values)
        elif ord == 1:
            # Maximum absolute column sum
            result = mx.max(mx.sum(mx.abs(x_array), axis=0))
        elif ord == 'inf':
            # Maximum absolute row sum
            result = mx.max(mx.sum(mx.abs(x_array), axis=1))
        elif ord == -1:
            # Minimum absolute column sum
            result = mx.min(mx.sum(mx.abs(x_array), axis=0))
        elif ord == '-inf':
            # Minimum absolute row sum
            result = mx.min(mx.sum(mx.abs(x_array), axis=1))
        else:
            # For other matrix norms, use the singular values
            s_values = svd(x_array, compute_uv=False)
            if isinstance(s_values, tuple):
                # Handle case where svd returns a tuple
                s_array = s_values[0]
            else:
                # Handle case where svd returns an array
                s_array = s_values
                
            if ord == 2:
                # Spectral norm (maximum singular value)
                result = s_array[0]
            elif ord == -2:
                # Minimum singular value
                result = s_array[-1]
            else:
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Keep dimensions if requested
    if keepdim and axis is not None:
        # Reshape to keep dimensions
        if isinstance(axis, tuple):
            shape = list(x_array.shape)
            for ax in sorted(axis, reverse=True):
                shape[ax] = 1
            result = mx.reshape(result, tuple(shape))
        else:
            shape = list(x_array.shape)
            shape[axis] = 1
            result = mx.reshape(result, tuple(shape))
    
    return result

def det(a: TensorLike) -> mx.array:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Determinant of the matrix
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    TensorInstance = MLXTensor()
    a_array = TensorInstance.convert_to_tensor(a)
    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    
    # Special cases for small matrices
    if n == 1:
        return a_array[0, 0]
    elif n == 2:
        term1 = mx.multiply(a_array[0, 0], a_array[1, 1])
        term2 = mx.multiply(a_array[0, 1], a_array[1, 0])
        return mx.subtract(term1, term2)
    
    # For larger matrices, use LU decomposition
    # This is a simplified implementation and may not be numerically stable
    # For a more robust implementation, consider using a dedicated algorithm
    
    # Make a copy of the matrix
    a_copy = mx.array(a_array)
    
    # Initialize determinant
    det_value = mx.array(1.0, dtype=a_array.dtype)
    
    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = a_copy[i, i]
        
        # Update determinant
        det_value = mx.multiply(det_value, pivot)
        
        # If pivot is zero, determinant is zero
        if mx.abs(pivot) < 1e-10:
            return mx.array(0.0, dtype=a_array.dtype)
        
        # Eliminate below
        for j in range(i + 1, n):
            factor = mx.divide(a_copy[j, i], pivot)
            
            # Update row
            a_copy = a_copy.at[j, i:].set(mx.subtract(a_copy[j, i:], mx.multiply(factor, a_copy[i, i:])))
    
    return det_value

def diag(x: TensorLike, k: int = 0) -> mx.array:
    """
    Extract a diagonal or construct a diagonal matrix.
    
    Args:
        x: Input array. If x is 2-D, return the k-th diagonal.
           If x is 1-D, return a 2-D array with x on the k-th diagonal.
        k: Diagonal offset. Use k>0 for diagonals above the main diagonal,
           and k<0 for diagonals below the main diagonal.
           
    Returns:
        The extracted diagonal or constructed diagonal matrix.
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    TensorInstance = MLXTensor()
    x_array = TensorInstance.convert_to_tensor(x)
    
    # Check if input is 1-D or 2-D
    if x_array.ndim == 1:
        # Construct a diagonal matrix
        n = x_array.shape[0]
        
        # Calculate the size of the output matrix
        m = n + abs(k)
            
        # Create a zero matrix
        from ember_ml.backend.mlx.tensor.ops.creation import zeros
        result = zeros((m, m), dtype=x_array.dtype)
       
        # Fill the diagonal
        from ember_ml.backend.mlx.tensor.ops.indexing import scatter

        for i in range(n):
            if k >= 0:
                # Diagonal above main
                result = scatter(mx.array([i, i + k]), x_array[i], (m, m))
            else:
                # Diagonal below main
                result = scatter(mx.array([i - k, i]), x_array[i], (m, m))
                
        return result
    
    elif x_array.ndim == 2:
        # Extract a diagonal
        rows, cols = x_array.shape
        
        # Calculate the length of the diagonal
        if k >= 0:
            diag_len = min(rows, cols - k)
        else:
            diag_len = min(rows + k, cols)
            
        if diag_len <= 0:
            # Empty diagonal
            return mx.array([], dtype=x_array.dtype)
            
        # Create an array to hold the diagonal
        result = mx.zeros((diag_len,), dtype=x_array.dtype)
        
        # Extract the diagonal
        for i in range(diag_len):
            if k >= 0:
                from ember_ml.backend.mlx.tensor.ops.indexing import scatter
                result = scatter(result, mx.array([i]), x_array[i, i + k])
            else:
                result = scatter(result, mx.array([i]), x_array[i - k, i])
                
        return result
    
    else:
        raise ValueError("Input must be 1-D or 2-D")

def diagonal(x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> mx.array:
    """
    Return specified diagonals of an array.
    
    Args:
        x: Input array
        offset: Offset of the diagonal from the main diagonal
        axis1: First axis of the 2-D sub-arrays from which the diagonals should be taken
        axis2: Second axis of the 2-D sub-arrays from which the diagonals should be taken
        
    Returns:
        Array of diagonals. For a 3D input array, the output contains the diagonal 
        from each 2D slice, maintaining the structure of the non-diagonal dimensions.
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    TensorInstance = MLXTensor()
    x_array = TensorInstance.convert_to_tensor(x)
    
    # Initial validations
    if x_array.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 must be different")
        
    # Normalize negative axes
    ndim = x_array.ndim
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim
        
    # Validate axes
    if not (0 <= axis1 < ndim and 0 <= axis2 < ndim):
        raise ValueError("axis1 and axis2 must be within dimensions")
    
    # Get shape and calculate diagonal length
    shape = x_array.shape
    diag_len = min(shape[axis1], shape[axis2] - offset if offset >= 0 else shape[axis1] + offset)
    
    # Create result shape preserving the order of non-diagonal axes
    non_diag_axes = [i for i in range(ndim) if i != axis1 and i != axis2]
    result_shape = [diag_len] + [shape[i] for i in non_diag_axes]
    
    # Initialize result tensor with correct shape
    result = mx.zeros(tuple(result_shape), dtype=x_array.dtype)
    
    # Calculate source indices for each diagonal element
    for i in range(diag_len):
        # Get the indices for the i-th diagonal element
        if offset >= 0:
            idx_axis1 = i
            idx_axis2 = i + offset
        else:
            idx_axis1 = i - offset
            idx_axis2 = i
        
        # Create slices for extracting and assigning values
        src_slice = []
        dst_slice = [i]
        dst_idx = 1  # Start after the diagonal dimension
        
        for d in range(ndim):
            if d == axis1:
                src_slice.append(idx_axis1)
            elif d == axis2:
                src_slice.append(idx_axis2)
            else:
                # For non-diagonal dimensions, use full slices
                src_slice.append(slice(None))
                dst_slice.append(slice(None))
                dst_idx += 1
                
        # Extract the diagonal element and set it in the result
        src_value = x_array[tuple(src_slice)]
        from ember_ml.backend.mlx.tensor.ops.indexing import scatter

        result = scatter(result, mx.array([i] + [slice(None)] * (len(dst_slice) - 1)), src_value)
    
    return result
