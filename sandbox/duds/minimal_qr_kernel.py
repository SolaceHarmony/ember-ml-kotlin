"""
Minimal QR kernel that just initializes Q to identity and R to the input matrix.
"""

import sys
import os
import time
import mlx.core as mx

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Minimal QR kernel
_MINIMAL_QR_SRC = r"""
    // Thread ID and dimensions
    uint tid = thread_position_in_grid.x;
    uint total_threads = grid_size.x * threads_per_threadgroup.x;
    
    // Get matrix dimensions
    const uint m = shape[0];
    const uint n = shape[1];
    
    // Set debug values
    if (tid == 0) {
        dbg[0] = 1.0f;  // Kernel executed
        dbg[1] = float(m);  // Number of rows
        dbg[2] = float(n);  // Number of columns
        dbg[3] = float(total_threads);  // Total number of threads
        dbg[15] = 1.0f;  // Success flag
    }
    
    // Initialize Q to identity matrix
    for (uint idx = tid; idx < m * m; idx += total_threads) {
        uint row = idx / m;
        uint col = idx % m;
        Q_out[idx] = (row == col) ? 1.0f : 0.0f;
    }
    
    // Copy A to R
    for (uint idx = tid; idx < m * n; idx += total_threads) {
        R_out[idx] = A[idx];
    }
"""

# Compile the kernel
_MINIMAL_QR_KERNEL = mx.fast.metal_kernel(
    name="minimal_qr_kernel",
    source=_MINIMAL_QR_SRC,
    input_names=["A", "shape"],
    output_names=["Q_out", "R_out", "dbg"],
    ensure_row_contiguous=True
)

def minimal_qr(A):
    """Minimal QR decomposition that just initializes Q to identity and R to A."""
    # Convert input to MLX array
    A = mx.array(A, dtype=mx.float32)
    
    # Get dimensions
    m, n = A.shape
    
    # Prepare inputs
    shape = mx.array([m, n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    print(f"Matrix dimensions: {m}x{n}")
    
    # Prepare outputs
    output_shapes = [(m, m), (m, n), (16,)]
    output_dtypes = [mx.float32, mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling minimal QR kernel...")
    start_time = time.time()
    outputs = _MINIMAL_QR_KERNEL(
        inputs=[A, shape],
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup
    )
    end_time = time.time()
    print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
    
    # Get outputs
    Q, R, dbg = outputs
    
    return Q, R, dbg

def test_minimal_qr():
    """Test the minimal QR decomposition."""
    print("\n=== Minimal QR Test ===\n")
    
    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Perform minimal QR decomposition
    print("\nPerforming minimal QR decomposition...")
    q, r, dbg = minimal_qr(a)
    
    # Print shapes
    print(f"Q shape: {q.shape}")
    print(f"R shape: {r.shape}")
    print(f"Debug info shape: {dbg.shape}")
    
    # Print debug info
    print("\nDebug info:")
    print(dbg)
    
    # Check if debug info contains non-zero values
    dbg_nonzero = mx.any(mx.abs(dbg) > 0).item()
    print(f"Debug info contains non-zero values: {dbg_nonzero}")
    
    if dbg_nonzero:
        for i in range(dbg.shape[0]):
            if abs(dbg[i]) > 0:
                print(f"  dbg[{i}] = {dbg[i]}")
    
    # Print Q and R matrices
    print("\nQ matrix:")
    print(q)
    
    print("\nR matrix:")
    print(r)
    
    # Check if Q is identity
    identity = mx.eye(q.shape[0])
    q_is_identity = mx.all(mx.abs(q - identity) < 1e-6).item()
    print(f"Q is identity: {q_is_identity}")
    
    # Check if R is equal to A
    r_equals_a = mx.all(mx.abs(r - a) < 1e-6).item()
    print(f"R equals A: {r_equals_a}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if dbg_nonzero else 'FAILURE'}")
    print(f"Q is identity: {'SUCCESS' if q_is_identity else 'FAILURE'}")
    print(f"R equals A: {'SUCCESS' if r_equals_a else 'FAILURE'}")

if __name__ == "__main__":
    test_minimal_qr()