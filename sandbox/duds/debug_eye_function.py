"""
Debug script for the eye() function in MLX backend.

This script tests the eye() function and prints the shape and content
of the resulting matrix to help diagnose the issue.
"""

from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml.ops import linearalg
import sys

# Set the backend to MLX
set_backend("mlx")

def test_eye_function():
    """Test the eye() function with different parameters."""
    print("Testing eye() function in MLX backend")
    
    # Test case 1: Square matrix
    n = 3
    print(f"\nTest case 1: eye({n})")
    eye_matrix = tensor.eye(n)
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content:\n{eye_matrix}")
    
    # Test case 2: Rectangular matrix
    n, m = 3, 5
    print(f"\nTest case 2: eye({n}, {m})")
    eye_matrix = tensor.eye(n, m)
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content:\n{eye_matrix}")
    
    # Test case 3: With specific dtype
    n = 4
    print(f"\nTest case 3: eye({n}, dtype=tensor.float32)")
    eye_matrix = tensor.eye(n, dtype=tensor.float32)
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content:\n{eye_matrix}")
    print(f"Type: {type(eye_matrix)}")
    
    # Test case 4: Test with the same dimensions as in the failing test
    n = 10
    print(f"\nTest case 4: eye({n}, dtype=tensor.float32)")
    eye_matrix = tensor.eye(n, dtype=tensor.float32)
    print(f"Shape: {eye_matrix.shape}")
    print(f"Content (first few elements):\n{eye_matrix}")

def inspect_eye_implementation():
    """Print the source code of the eye() function."""
    import inspect
    from ember_ml.backend.mlx.tensor.ops.creation import eye
    
    print("\nInspecting eye() implementation:")
    print(inspect.getsource(eye))

if __name__ == "__main__":
    test_eye_function()
    inspect_eye_implementation()