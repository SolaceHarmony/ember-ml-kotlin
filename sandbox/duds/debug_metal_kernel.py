"""
Debug script with a simplified Metal kernel to verify kernel execution.
"""

import sys
import os
import time
import mlx.core as mx

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple Metal kernel that just copies input to output and sets debug values
_DEBUG_KERNEL_SRC = r"""
    // Thread ID and dimensions
    uint tid = tid;
    uint3 tpg = tpg;
    uint3 gpg = gpg;
    
    // Get matrix dimensions
    const uint m = shape[0];
    const uint n = shape[1];
    const uint total_threads = gpg.x * tpg.x;
    
    // Set debug values
    if (tid == 0) {
        dbg[0] = 1.0f;  // Kernel executed
        dbg[1] = float(m);  // Number of rows
        dbg[2] = float(n);  // Number of columns
        dbg[3] = float(total_threads);  // Total number of threads
        dbg[4] = float(tid);  // Thread ID
        dbg[5] = float(tpg.x);  // Threadgroup size x
        dbg[6] = float(gpg.x);  // Grid size x
    }
    
    // Copy input to output
    for (uint idx = tid; idx < m * n; idx += total_threads) {
        output[idx] = input[idx];
    }
"""

# Compile the kernel
_DEBUG_KERNEL = mx.fast.metal_kernel(
    name="debug_kernel",
    source=_DEBUG_KERNEL_SRC,
    input_names=["input", "shape"],
    output_names=["output", "dbg"],
    ensure_row_contiguous=True
)

def debug_metal_kernel():
    """Test a simplified Metal kernel to verify kernel execution."""
    print("\n=== Debug Metal Kernel Test ===\n")
    
    # Create a test matrix
    m, n = 3, 2
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Prepare inputs
    shape = mx.array([m, n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs
    output_shapes = [(m * n,), (16,)]
    output_dtypes = [mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling Metal kernel...")
    start_time = time.time()
    outputs = _DEBUG_KERNEL(
        inputs=[a.flatten(), shape],
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup
    )
    end_time = time.time()
    print(f"Metal kernel execution completed in {end_time - start_time:.4f} seconds")
    
    # Get outputs
    output_flat, dbg = outputs
    output = output_flat.reshape(m, n)
    
    # Print outputs
    print(f"Output matrix shape: {output.shape}")
    print(f"Output matrix:\n{output}")
    print(f"Debug info:\n{dbg}")
    
    # Check if debug info contains non-zero values
    dbg_nonzero = mx.any(mx.abs(dbg) > 0).item()
    print(f"Debug info contains non-zero values: {dbg_nonzero}")
    
    if dbg_nonzero:
        for i in range(dbg.shape[0]):
            if abs(dbg[i]) > 0:
                print(f"  dbg[{i}] = {dbg[i]}")
    
    # Check if output matches input
    match = mx.all(mx.abs(output - a) < 1e-6).item()
    print(f"Output matches input: {match}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if dbg_nonzero else 'FAILURE'}")
    print(f"Input/output match: {'SUCCESS' if match else 'FAILURE'}")

if __name__ == "__main__":
    debug_metal_kernel()