"""
PyTorch matrix operations for ember_ml.

This module provides PyTorch implementations of matrix operations.
"""

import torch
from typing import Union, Tuple, Optional, Literal

# Import from tensor_ops
from ember_ml.backend.torch.types import TensorLike, OrdLike
from ember_ml.backend.torch.linearalg.decomp_ops import svd # Corrected import path
from ember_ml.backend.torch.tensor import TorchDType

dtype_obj = TorchDType()

def norm(x: TensorLike, 
         ord: OrdLike = None, 
         axis: Optional[Union[int, Tuple[int, ...]]] = None, 
         keepdim: bool = False) -> torch.Tensor:    
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
    # Convert input to torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()
    x_array = TensorInstance.convert_to_tensor(x)
    
    # Handle default ord values without using invalid torch.string type
    if ord is None:
        if x_array.ndim > 1 and axis is None:
            ord = 'fro'  # Default matrix norm
        else:
            ord = 2     # Default vector norm
    
    # Vector norm handling
    is_vector = axis is not None or x_array.ndim == 1
    if is_vector:
        # Default axis handling with torch
        if axis is None:
            axis = 0
            
        # Compute vector norm based on ord
        if ord == 'inf':
            if isinstance(axis, int):
                result = torch.max(torch.abs(x_array), dim=axis)[0]
            else:
                raise ValueError("For L-infinity norm, axis must be an integer")
        elif ord == 1:
            result = torch.sum(torch.abs(x_array), dim=axis)
        elif ord == 2:
            result = torch.sqrt(torch.sum(torch.square(x_array), dim=axis))
        else:
            if isinstance(ord, (int, float)):
                ord_tensor = torch.tensor(ord, dtype=x_array.dtype, device=x_array.device)
                result = torch.pow(
                    torch.sum(torch.abs(x_array).pow(ord_tensor), dim=axis),
                    torch.reciprocal(ord_tensor)
                )
            else:
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Matrix norm
    else:
        if ord == 'fro':
            result = torch.sqrt(torch.sum(torch.square(x_array)))
        elif ord == 'nuc':
            s_values = svd(x_array, compute_uv=False)
            result = torch.sum(s_values[0] if isinstance(s_values, tuple) else s_values)
        elif ord == 1:
            result = torch.max(torch.sum(torch.abs(x_array), dim=0))
        elif ord == 'inf':
            result = torch.max(torch.sum(torch.abs(x_array), dim=1))
        elif ord == -1:
            result = torch.min(torch.sum(torch.abs(x_array), dim=0))
        elif ord == '-inf':
            result = torch.min(torch.sum(torch.abs(x_array), dim=1))
        else:
            s_values = svd(x_array, compute_uv=False)
            s_array = s_values[0] if isinstance(s_values, tuple) else s_values
            
            if ord == 2:
                result = s_array[0]
            elif ord == -2:
                result = s_array[-1]
            else:
                raise ValueError(f"Invalid norm order: {ord}")
    
    # Handle keepdim with pure torch operations
    if keepdim and axis is not None:
        if isinstance(axis, tuple):
            shape = list(x_array.shape)
            for ax in sorted(axis, reverse=True):
                shape[ax] = 1
            result = result.view(*shape)
        else:
            shape = list(x_array.shape)
            shape[axis] = 1
            result = result.view(*shape)
    
    return result

def det(a: TensorLike) -> torch.Tensor:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Determinant of the matrix
    """
    # Convert input to torch.Tensor array
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()
    a_array = TensorInstance.convert_to_tensor(a)

    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    
    # Special cases for small matrices
    if torch.equal(torch.tensor(n), torch.tensor(1)):
        return a_array[0, 0]
    elif torch.equal(torch.tensor(n), torch.tensor(2)):
        term1 = torch.multiply(a_array[0, 0], a_array[1, 1])
        term2 = torch.multiply(a_array[0, 1], a_array[1, 0])
        return torch.subtract(term1, term2)
    
    # For larger matrices, use LU decomposition
    # This is a simplified implementation and may not be numerically stable
    # For a more robust implementation, consider using a dedicated algorithm
    
    # Make a copy of the matrix
    a_copy = torch.tensor(a_array)
    
    # Initialize determinant
    det_value = torch.tensor(1.0, dtype=a_array.dtype)
    
    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = a_copy[i, i]
        
        # Update determinant
        det_value = torch.multiply(det_value, pivot)
        
        # If pivot is zero, determinant is zero
        if torch.less(torch.abs(pivot), torch.tensor(1e-10)):
            return torch.tensor(0.0, dtype=a_array.dtype)
        
        # Eliminate below
        # Use direct integer calculation
        i_plus_1_int = i + 1
        for j in range(i_plus_1_int, n):
            factor = torch.divide(a_copy[j, i], pivot)
            
            # Calculate the new row
            new_row = torch.subtract(a_copy[j, i:], torch.multiply(factor, a_copy[i, i:]))
            
            # Update a_copy using direct indexing
            for k in range(i, n):
                a_copy[j, k] = new_row[k - i]
    
    return det_value

def diag(x: TensorLike, k: int = 0) -> torch.Tensor:
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
    # Convert input to torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()
    x_array = TensorInstance.convert_to_tensor(x)
    
    # Check if input is 1-D or 2-D
    if x_array.ndim == 1:
        # Construct a diagonal matrix
        n = x_array.shape[0]
        
        # Calculate the size of the output matrix
        m = torch.where(torch.greater_equal(torch.tensor(k), torch.tensor(0)),
                     torch.add(torch.tensor(n), k),
                     torch.subtract(torch.tensor(n), -k))
            
        # Ensure we use a compatible dtype (not int64)
        dtype = x_array.dtype
        if dtype == torch.int64:
            dtype = torch.int32
            
        # Create a zero matrix with proper dtype
        result = torch.zeros([int(m), int(m)], dtype=dtype)
        
        # Import the scatter function from indexing
        from ember_ml.backend.torch.tensor.ops.indexing import scatter_add
        
        # Fill the diagonal
        # Use torch.greater_equal for comparison
        is_non_negative = torch.greater_equal(torch.tensor(k), torch.tensor(0))
        
        for i in range(n):
            # Create a copy of the result
            result_copy = result.clone()
            
            # Use torch.where to conditionally select the indices
            row = torch.where(is_non_negative,
                          torch.tensor(i),
                          torch.subtract(torch.tensor(i), torch.tensor(k)))
            col = torch.where(is_non_negative,
                          torch.add(torch.tensor(i), torch.tensor(k)),
                          torch.tensor(i))
            
            # Update the element directly
            result_copy[int(row.item()), int(col.item())] += x_array[i].item()            
            result = result_copy
                
        # No need for transposition here - the diag function has no concept of axes
        # This is internal to the diagonal function implementation
        return result
    
    elif x_array.ndim == 2:
        # Extract a diagonal
        rows, cols = x_array.shape
        
        # Calculate the length of the diagonal
        # Use torch.greater_equal, torch.subtract, torch.add, and torch.minimum for operations
        is_non_negative = torch.greater_equal(torch.tensor(k), torch.tensor(0))
        diag_len_if_positive = torch.minimum(torch.tensor(rows), torch.subtract(torch.tensor(cols), torch.tensor(k)))
        diag_len_if_negative = torch.minimum(torch.add(torch.tensor(rows), torch.tensor(k)), torch.tensor(cols))
        diag_len = torch.where(is_non_negative, diag_len_if_positive, diag_len_if_negative).item()
            
        # Use torch.less_equal for comparison
        if torch.less_equal(torch.tensor(diag_len), torch.tensor(0)):
            # Empty diagonal
            return torch.tensor([], dtype=x_array.dtype)
            
        # Ensure we use a compatible dtype (not int64)
        dtype = x_array.dtype
        if dtype == torch.int64:
            dtype = torch.int32
            
        # Create an array to hold the diagonal
        result = torch.zeros([int(diag_len)], dtype=dtype)
        
        # Extract the diagonal
        # Use torch.greater_equal for comparison
        # Create scalar tensors using item() method to avoid warnings
        k_tensor = torch.tensor(k, dtype=torch.int32)
        zero_tensor = torch.tensor(0, dtype=torch.int32)
        is_non_negative = torch.greater_equal(k_tensor, zero_tensor)
        
        for i in range(int(diag_len)):
            # Create a copy of the result using clone().detach() instead of torch.tensor()
            result_copy = result.clone().detach()
            
            # Create scalar tensor once for this iteration
            i_tensor = torch.tensor(i, dtype=torch.int32)
            
            # Use torch.where to conditionally select the indices
            row = torch.where(is_non_negative,
                          i_tensor,
                          torch.subtract(i_tensor, k_tensor))
            col = torch.where(is_non_negative,
                          torch.add(i_tensor, k_tensor),
                          i_tensor)
            
            # Update the element directly
            # Create index tensor properly to avoid warnings
            idx_tensor = torch.tensor([i], dtype=torch.int64)
            result_copy.index_copy_(0, idx_tensor, x_array[i, i].unsqueeze(0))
            result = result_copy
                
        return result
    
    else:
        raise ValueError("Input must be 1-D or 2-D")

def diagonal(x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
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
    # Convert input to Torch array
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()
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
    non_diag_axes = [shape[i] for i in range(ndim) if i not in (axis1, axis2)]
    result_shape = [int(diag_len)] + non_diag_axes
    
    # Initialize result tensor with correct shape
    result = torch.zeros(tuple(result_shape), dtype=x_array.dtype)
    
    # Calculate source indices for each diagonal element and assign to result
    for i in range(diag_len):
        # Build the index tuple for extraction
        src_idx = []
        for ax in range(x_array.ndim):
            if ax == axis1:
                if offset >= 0:
                    src_idx.append(i)
                else:
                    src_idx.append(i - offset)
            elif ax == axis2:
                if offset >= 0:
                    src_idx.append(i + offset)
                else:
                    src_idx.append(i)
            else:
                src_idx.append(slice(None))
        
        # Extract the diagonal slice
        value = x_array[tuple(src_idx)]
        
        # Create result slice with proper indexing for the first dimension
        result_slice = tuple([i] + [slice(None)] * (len(result_shape) - 1))
        result[result_slice] = value
    
    return result