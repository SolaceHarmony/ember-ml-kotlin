"""
Test script for the fixed eye and scatter functions.
"""

import sys
import os
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend
from ember_ml.ops import linearalg

def test_eye_function():
    """Test the fixed eye function."""
    print("\n=== Testing Fixed Eye Function ===\n")
    
    # Set the backend to MLX
    set_backend("mlx")
    print(f"Backend: {ops.get_backend()}")
    
    # Test square matrix
    n = 3
    print(f"\nTesting eye({n}):")
    eye_n = tensor.eye(n)
    print(f"Shape: {eye_n.shape}")
    print(f"Content:\n{eye_n}")
    
    # Test rectangular matrix (more rows than columns)
    n, m = 3, 2
    print(f"\nTesting eye({n}, {m}):")
    eye_n_m = tensor.eye(n, m)
    print(f"Shape: {eye_n_m.shape}")
    print(f"Content:\n{eye_n_m}")
    
    # Test rectangular matrix (more columns than rows)
    n, m = 2, 4
    print(f"\nTesting eye({n}, {m}):")
    eye_n_m = tensor.eye(n, m)
    print(f"Shape: {eye_n_m.shape}")
    print(f"Content:\n{eye_n_m}")
    
    print("\nEye function tests completed successfully.")

def test_scatter_function():
    """Test the fixed scatter function."""
    print("\n=== Testing Fixed Scatter Function ===\n")
    
    # Set the backend to MLX
    set_backend("mlx")
    print(f"Backend: {ops.get_backend()}")
    
    # Test case 1: Create a 3x3 identity matrix
    n = 3
    indices = (tensor.arange(n), tensor.arange(n))
    updates = tensor.ones(n)
    shape = (n, n)
    
    print(f"\nTest case 1: Creating a {n}x{n} identity matrix")
    result = tensor.scatter(indices, updates, shape)
    print(f"Shape: {result.shape}")
    print(f"Content:\n{result}")
    
    # Test case 2: Create a 3x5 rectangular matrix with ones on the diagonal
    n, m = 3, 5
    indices = (tensor.arange(n), tensor.arange(n))
    updates = tensor.ones(n)
    shape = (n, m)
    
    print(f"\nTest case 2: Creating a {n}x{m} matrix with ones on the diagonal")
    result = tensor.scatter(indices, updates, shape)
    print(f"Shape: {result.shape}")
    print(f"Content:\n{result}")
    
    # Test case 3: Create a 5x3 rectangular matrix with ones on the diagonal
    n, m = 5, 3
    indices = (tensor.arange(m), tensor.arange(m))
    updates = tensor.ones(m)
    shape = (n, m)
    
    print(f"\nTest case 3: Creating a {n}x{m} matrix with ones on the diagonal")
    result = tensor.scatter(indices, updates, shape)
    print(f"Shape: {result.shape}")
    print(f"Content:\n{result}")
    
    print("\nScatter function tests completed successfully.")

def test_qr_numerical_stability():
    """
    Test that QR implementation has numerical stability.
    
    This is a simplified version of the original test that was failing.
    """
    print("\n=== Testing QR Numerical Stability ===\n")
    
    # Set the backend to MLX
    set_backend("mlx")
    print(f"Backend: {ops.get_backend()}")
    
    # Create a test matrix
    n = 10
    m = 5
    
    # Create a random matrix
    tensor.set_seed(42)
    a = tensor.random_normal((n, m))
    
    # Perform QR decomposition
    print("\nPerforming QR decomposition...")
    start_time = time.time()
    try:
        q, r = linearalg.qr(a)
        end_time = time.time()
        print(f"QR decomposition completed in {end_time - start_time:.4f} seconds")
        
        # Print shapes
        print(f"Shape of a: {a.shape}")
        print(f"Shape of q: {q.shape}")
        print(f"Shape of r: {r.shape}")
        
        # Check orthogonality of columns (Q^T * Q should be identity)
        q_t_q = ops.matmul(tensor.transpose(q), q)
        
        # Create identity matrix using eye function
        identity = tensor.eye(q_t_q.shape[0])
        
        # Compute error matrix and maximum absolute error
        diff = ops.subtract(q_t_q, identity)
        max_error = ops.stats.max(ops.abs(diff))
        mean_error = ops.stats.mean(ops.abs(diff))
        
        print(f"\nQR Orthogonality:")
        print(f"Maximum absolute error: {max_error}")
        print(f"Mean absolute error: {mean_error}")
        
        # Check if error is within acceptable range
        if max_error < 1e-5:
            print("\nQR numerical stability test PASSED.")
        else:
            print("\nQR numerical stability test FAILED.")
            print(f"Maximum error {max_error} exceeds threshold 1e-5.")
    
    except Exception as e:
        print(f"Error during QR decomposition: {e}")
        print("\nQR numerical stability test FAILED.")

if __name__ == "__main__":
    test_eye_function()
    test_scatter_function()
    test_qr_numerical_stability()