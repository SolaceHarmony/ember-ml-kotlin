"""
Fixed QR decomposition Metal kernel for MLX, addressing shared memory issues.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

# Fixed Metal kernel source code
_FIXED_QR_PROTOTYPE_SRC = r"""
    // Get thread ID and local thread ID
    uint tid = thread_position_in_grid.x;
    uint local_tid = thread_position_in_threadgroup.x;
    uint num_threads_in_group = threads_per_threadgroup.x;

    // Get matrix dimensions from uint32 buffer
    const uint m = A_shape[0];
    const uint n = A_shape[1];
    const uint min_dim = (m < n ? m : n);
    const uint grid_sz = grid_size.x * threads_per_threadgroup.x;

    // Calculate total elements for initialization
    const uint q_elements = m * m;
    const uint r_elements = m * n;
    const uint total_init_elements = q_elements + r_elements;
    
    // Global memory for reduction across threadgroups
    device uint global_limbs[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    device uint error_flag = 0u;
    device float shared_vtv = 0.0f;
    device float shared_inv_vtv = 0.0f;

    // Set debug values - only thread 0 should set these
    if (tid == 0) {
        dbg[0] = 1.0f;  // Kernel executed flag
        dbg[1] = float(m);  // Number of rows
        dbg[2] = float(n);  // Number of columns
        dbg[3] = float(min_dim); // min(m, n)
        dbg[4] = float(thread_position_in_grid.x);  // Thread ID
        dbg[5] = float(threads_per_threadgroup.x);  // Threads per threadgroup
        dbg[6] = float(grid_sz);  // Grid size (total threads)
        dbg[15] = 1.0f;  // Success flag
    }

    // Initialize Q_out to identity matrix and R_out to A
    // Use thread_position_in_grid.x for linear indexing
    if (tid < total_init_elements) {
        // Initialize Q_out elements
        if (tid < q_elements) {
            uint row = tid / m; // Assuming row-major
            uint col = tid % m;
            Q_out[tid] = (row == col) ? 1.0f : 0.0f;
        }
        // Initialize R_out elements by copying from A
        else {
            uint r_idx = tid - q_elements;
            R_out[r_idx] = A[r_idx]; // Direct linear copy
        }
    }

    // Synchronize after initialization
    threadgroup_barrier(mem_flags::mem_device);

    // --- First Householder Reflection (k=0) ---
    uint k = 0;

    // Find the maximum absolute value in the first column of R (R[0...m-1, 0])
    float thread_max = 0.0f;
    for (uint i = k + tid; i < m; i += grid_sz) {
        thread_max = fmax(thread_max, fabs(R_out[i * n + k]));
    }

    // Reduce max within threadgroup
    threadgroup float tg_max[32]; // Assuming max 32 threads per threadgroup
    
    // Initialize to zero
    if (local_tid < 32) {
        tg_max[local_tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread writes its max to shared memory
    if (local_tid < 32) {
        tg_max[local_tid] = thread_max;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    if (local_tid == 0) {
        float group_max = 0.0f;
        for (uint i = 0; i < min(32u, num_threads_in_group); ++i) {
            group_max = fmax(group_max, tg_max[i]);
        }
        
        // Atomic max to find global maximum
        atomic_fetch_max_explicit((device atomic_float*)&shared_vtv, group_max, memory_order_relaxed, memory_scope_device);
    }
    
    // Wait for all threadgroups to finish
    threadgroup_barrier(mem_flags::mem_device);
    
    // Get the global maximum
    float cmax = shared_vtv;
    
    // Reset shared_vtv for next use
    if (tid == 0) {
        shared_vtv = 0.0f;
        dbg[7] = cmax; // Store cmax in debug buffer
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Scale the first column of R (R[0...m-1, 0])
    float scale = (cmax > 1e-10f ? 1.0f / cmax : 1.0f);
    if (tid < m) { // Threads 0 to m-1 process the first column
        R_out[tid * n + k] *= scale; // Accessing R_out[tid, 0]
    }

    // Synchronize after scaling
    threadgroup_barrier(mem_flags::mem_device);

    // Build Householder vector v
    // Parallel reduction for norm calculation
    float partial_sum = 0.0f;
    if (tid < m) {
        float v_i = R_out[tid * n + k];
        partial_sum = v_i * v_i;
    }
    
    // Reduce within threadgroup
    threadgroup float tg_sum[32]; // Assuming max 32 threads per threadgroup
    
    // Initialize to zero
    if (local_tid < 32) {
        tg_sum[local_tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread writes its sum to shared memory
    if (local_tid < 32) {
        tg_sum[local_tid] = partial_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    if (local_tid == 0) {
        float group_sum = 0.0f;
        for (uint i = 0; i < min(32u, num_threads_in_group); ++i) {
            group_sum += tg_sum[i];
        }
        
        // Atomic add to find global sum
        atomic_fetch_add_explicit((device atomic_float*)&shared_vtv, group_sum, memory_order_relaxed, memory_scope_device);
    }
    
    // Wait for all threadgroups to finish
    threadgroup_barrier(mem_flags::mem_device);
    
    // Get the global sum and calculate norm
    float norm = 0.0f;
    if (tid == 0) {
        norm = sqrt(shared_vtv);
        dbg[8] = norm;
        
        // Update v[0]
        float r00 = R_out[0];
        float sign_r00 = (r00 >= 0.0f ? 1.0f : -1.0f);
        R_out[0] = r00 + sign_r00 * norm; // Update R[0, 0] which is v[0]
        dbg[9] = R_out[0]; // Store updated R[0, 0] in debug buffer
        
        // Calculate vtv and inv_vtv
        float vtv = R_out[0] * R_out[0];
        for (uint i = 1; i < m; ++i) {
            vtv += R_out[i*n + k] * R_out[i*n + k];
        }
        
        float inv_vtv = (vtv > 1e-10f ? 1.0f / vtv : 0.0f);
        
        // Store for other threads
        shared_vtv = vtv;
        shared_inv_vtv = inv_vtv;
    }
    
    // Wait for thread 0 to finish calculations
    threadgroup_barrier(mem_flags::mem_device);
    
    // Get shared values
    float vtv = shared_vtv;
    float inv_vtv = shared_inv_vtv;
    
    // Reflect R (simplified - only update the first row for prototype)
    if (tid == 0) {
        // In a real kernel, this would involve reflecting columns k to n-1 of R
        // For prototype, just set a debug value
        dbg[10] = 42.0f; // Placeholder debug value
    }

    // Synchronize after reflecting R
    threadgroup_barrier(mem_flags::mem_device);

    // Reflect Q (simplified - only update the first column for prototype)
    if (tid == 0) {
        // In a real kernel, this would involve reflecting columns 0 to m-1 of Q
        // For prototype, just set a debug value
        dbg[11] = 84.0f; // Placeholder debug value
    }

    // Synchronize after reflecting Q
    threadgroup_barrier(mem_flags::mem_device);

    // Un-scale the first column of R (R[0...m-1, 0])
    if (tid < m) { // Threads 0 to m-1 process the first column
        R_out[tid * n + k] /= scale; // Accessing R_out[tid, 0]
    }

    // Final synchronization
    threadgroup_barrier(mem_flags::mem_device);

    // Set final success flag (redundant with dbg[15] but good for clarity)
    if (tid == 0) {
        dbg[12] = 1.0f;
    }
"""

# Compile the kernel
_FIXED_QR_PROTOTYPE_KERNEL = mx.fast.metal_kernel(
    name="fixed_qr_prototype_kernel",
    input_names=["A"],
    output_names=["Q_out", "R_out", "dbg"],
    source=_FIXED_QR_PROTOTYPE_SRC,
    ensure_row_contiguous=True # Ensure inputs are row-contiguous
)

def fixed_qr_prototype(A):
    """
    Fixed QR decomposition function using the Metal kernel.
    """
    # Convert input to MLX array and ensure contiguity
    A = mx.array(A, dtype=mx.float32).reshape(*A.shape)

    # Get dimensions
    m, n = A.shape

    # Calculate total elements for initialization
    q_elements = m * m
    r_elements = m * n
    total_init_elements = q_elements + r_elements

    # Configure kernel dispatch
    # Use our improved thread allocation strategy
    if max(m, n) <= 8:
        # For tiny matrices, use minimal threads
        threadgroup_size = 32
        total_threads_to_launch = 32
    elif max(m, n) <= 32:
        # For small matrices, use one thread per element
        threadgroup_size = 32
        total_threads_to_launch = min(total_init_elements, 256)
    else:
        # For medium and large matrices, use more threads
        threadgroup_size = 32
        # Use at most 1024 threads for any matrix
        total_threads_to_launch = min(1024, total_init_elements)

    # Ensure total_threads_to_launch is a multiple of threadgroup_size
    total_threads_to_launch = (total_threads_to_launch + threadgroup_size - 1) // threadgroup_size * threadgroup_size

    grid = (total_threads_to_launch, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)

    print(f"\nInput matrix: {m}x{n}")
    print(f"Total elements for initialization: {total_init_elements}")
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    print(f"Total threads launched: {grid[0]}")

    # Prepare outputs
    output_shapes = [(m, m), (m, n), (16,)] # Q, R, dbg
    output_dtypes = [mx.float32, mx.float32, mx.float32]


    # Call the kernel
    print("Calling fixed QR prototype kernel...")
    start_time = time.time()
    outputs = _FIXED_QR_PROTOTYPE_KERNEL(
        inputs=[A],
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup,
        verbose=True  # Print generated code for debugging
    )
    end_time = time.time()
    print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")

    # Get outputs
    Q, R, dbg = outputs

    return Q, R, dbg

def test_fixed_qr_prototype():
    """Test the fixed QR prototype kernel."""
    print("\n=== Fixed QR Prototype Kernel Test ===\n")

    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")

    # Perform QR prototype decomposition
    print("\nPerforming fixed QR prototype decomposition...")
    q, r, dbg = fixed_qr_prototype(a)

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

    # For this prototype, we only check initialization and basic debug values
    # A real test would verify orthogonality and reconstruction

    # Check if kernel executed flag is set
    kernel_executed = dbg[0].item() == 1.0
    print(f"Kernel executed: {'SUCCESS' if kernel_executed else 'FAILURE'}")

    # Validate QR decomposition properties
    # 1. Check Q is orthogonal (Q^T Q = I)
    qtq = mx.matmul(q.T, q)
    identity = mx.eye(q.shape[0])
    q_orthogonal = mx.allclose(qtq, identity, atol=1e-4).item()
    
    # 2. Check R is upper triangular (only first column for prototype)
    r_upper = mx.all(mx.abs(r[1:, 0]) < 1e-5).item()  # Zeros below diagonal in first column
    
    # 3. Check A = QR reconstruction
    qr_product = mx.matmul(q, r)
    reconstruction_valid = mx.allclose(qr_product, a, atol=1e-4).item()
    
    print(f"Q orthogonal: {'SUCCESS' if q_orthogonal else 'FAILURE'}")
    print(f"R upper triangular (first column): {'SUCCESS' if r_upper else 'FAILURE'}")
    print(f"A = QR reconstruction: {'SUCCESS' if reconstruction_valid else 'FAILURE'}")

    # Check if debug values were set
    debug_values_set = dbg[0].item() == 1.0 and dbg[15].item() == 1.0
    print(f"Debug values set: {'SUCCESS' if debug_values_set else 'FAILURE'}")

    # Check if cmax, norm, updated R[0,0], reflected R/Q debug values were set (if applicable)
    # These checks depend on the simplified logic in the prototype kernel
    cmax_set = dbg[7].item() > 0.0 # Check if cmax was calculated and stored
    norm_set = dbg[8].item() > 0.0 # Check if norm was calculated and stored
    r00_updated = dbg[9].item() != a[0,0].item() # Check if R[0,0] was updated
    reflected_r_set = dbg[10].item() == 42.0 # Check placeholder debug value
    reflected_q_set = dbg[11].item() == 84.0 # Check placeholder debug value
    final_flag_set = dbg[12].item() == 1.0 # Check final flag
    
    print(f"cmax calculated: {'SUCCESS' if cmax_set else 'FAILURE'}")
    print(f"norm calculated: {'SUCCESS' if norm_set else 'FAILURE'}")
    print(f"R[0,0] updated: {'SUCCESS' if r00_updated else 'FAILURE'}")
    print(f"Reflected R debug set: {'SUCCESS' if reflected_r_set else 'FAILURE'}")
    print(f"Reflected Q debug set: {'SUCCESS' if reflected_q_set else 'FAILURE'}")
    print(f"Final flag set: {'SUCCESS' if final_flag_set else 'FAILURE'}")


if __name__ == "__main__":
    test_fixed_qr_prototype()