"""
Fixed implementation of the eye() function for the MLX backend.
"""

import mlx.core as mx
from typing import Optional, Union

def fixed_eye(n: int, m: Optional[int] = None, dtype=None, device=None) -> mx.array:
    """Create a 2D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n: int, number of rows.
        m: int, optional, number of columns. If None, defaults to n.
        dtype: data type of the returned tensor.
        device: device on which to place the created tensor.

    Returns:
        A 2D tensor with ones on the diagonal and zeros elsewhere.
    """
    if m is None:
        m = n
    
    # Validate dtype
    if dtype is None:
        mlx_dtype = mx.float32
    else:
        # Simple mapping for common dtypes
        dtype_str = str(dtype)
        if 'float32' in dtype_str:
            mlx_dtype = mx.float32
        elif 'float64' in dtype_str:
            mlx_dtype = mx.float32  # MLX doesn't support float64, use float32
        elif 'int32' in dtype_str:
            mlx_dtype = mx.int32
        elif 'int64' in dtype_str:
            mlx_dtype = mx.int32  # Use int32 for int64
        elif 'bool' in dtype_str:
            mlx_dtype = mx.bool_
        else:
            mlx_dtype = mx.float32  # Default to float32
    
    # Create a zeros tensor with shape (n, m)
    result = mx.zeros((n, m), dtype=mlx_dtype)
    
    # Set diagonal elements to 1 using slice_update
    min_dim = min(n, m)
    for i in range(min_dim):
        # Create indices for this diagonal element
        indices = mx.array([i, i], dtype=mx.int32)
        # Create value to set (1)
        value = mx.array(1, dtype=mlx_dtype)
        # Update the element
        result = mx.slice_update(result, value, indices, [0, 1])
    
    return result

def test_fixed_eye():
    """Test the fixed eye function."""
    print("Testing fixed_eye function")
    
    # Test case 1: Square matrix
    n = 3
    print(f"\nTest case 1: fixed_eye({n})")
    eye_matrix = fixed_eye(n)
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content:\n{eye_matrix}")
    
    # Test case 2: Rectangular matrix
    n, m = 3, 5
    print(f"\nTest case 2: fixed_eye({n}, {m})")
    eye_matrix = fixed_eye(n, m)
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content:\n{eye_matrix}")
    
    # Test case 3: With specific dtype
    n = 4
    print(f"\nTest case 3: fixed_eye({n}, dtype='float32')")
    eye_matrix = fixed_eye(n, dtype='float32')
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content:\n{eye_matrix}")
    
    # Test case 4: Test with the same dimensions as in the failing test
    n = 10
    print(f"\nTest case 4: fixed_eye({n}, dtype='float32')")
    eye_matrix = fixed_eye(n, dtype='float32')
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content (first few elements):\n{eye_matrix}")

if __name__ == "__main__":
    test_fixed_eye()