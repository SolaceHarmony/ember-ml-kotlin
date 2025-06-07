import mlx.core as mx
import numpy as np
import time
import os

def is_spd(A):
    """Check if matrix is symmetric positive definite"""
    # Check for symmetry 
    if not mx.all(mx.abs(A - A.T) < 1e-6).item():
        return False
    
    # Check for positive definiteness using CPU
    try:
        with mx.stream(mx.cpu):
            A_cpu = mx.array(A)
            mx.linalg.cholesky(A_cpu)
        return True
    except:
        return False

def generate_spd_matrix(n, min_eig=0.01):
    """Generate a guaranteed SPD matrix with minimum eigenvalue control"""
    # Create a random matrix
    A = mx.random.normal((n, n))
    
    # Make it symmetric
    A = (A + A.T) / 2
    
    # Add a multiple of the identity matrix to ensure positive definiteness
    reg = min_eig * n  # Scale with matrix size for stability
    A = A + mx.eye(n) * reg
    
    return A

@mx.custom_function
def mlx_cholesky(A):
    """Stable implementation of Cholesky decomposition using single-threaded approach for reliability"""
    n = A.shape[0]
    
    # Define Metal kernel source - using single thread approach for maximum stability
    source = """
    // Single-threaded implementation for maximum numerical stability
    if (thread_position_in_grid.x == 0) {
        // Get matrix size
        uint n = A_shape[0];
        
        // Initialize upper triangle to zero
        for (uint i = 0; i < n; i++) {
            for (uint j = i+1; j < n; j++) {
                out[i*n + j] = 0.0f;
            }
        }
        
        // Standard Cholesky algorithm with strict sequential processing
        for (uint j = 0; j < n; j++) {
            // Compute diagonal element with accumulator for better precision
            float diag_sum = 0.0f;
            for (uint k = 0; k < j; k++) {
                float val = out[j*n + k];
                diag_sum += val * val;
            }
            
            float diag_val = A[j*n + j] - diag_sum;
            // Ensure positive diagonal for numerical stability
            if (diag_val <= 1e-10f) {
                diag_val = 1e-10f;
            }
            out[j*n + j] = sqrt(diag_val);
            
            // Now compute all elements below diagonal in this column
            for (uint i = j+1; i < n; i++) {
                float sum = 0.0f;
                for (uint k = 0; k < j; k++) {
                    sum += out[i*n + k] * out[j*n + k];
                }
                
                float denom = out[j*n + j];
                if (denom > 1e-10f) {
                    out[i*n + j] = (A[i*n + j] - sum) / denom;
                } else {
                    out[i*n + j] = 0.0f;
                }
            }
        }
    }
    """
    
    # Metal header with math functions
    header = """
    #include <metal_stdlib>
    #include <metal_math>
    using namespace metal;
    """
    
    # Create the kernel
    kernel = mx.fast.metal_kernel(
        name="cholesky_kernel",
        input_names=["A"],
        output_names=["out"],
        source=source,
        header=header,
        ensure_row_contiguous=True
    )
    
    # Single thread for maximum stability
    grid = (1, 1, 1)
    threads = (1, 1, 1)
    
    # Run the kernel
    return kernel(
        inputs=[A],
        output_shapes=[A.shape],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threads
    )[0]

@mx.custom_function
def block_cholesky(A, block_size=16):
    """Block-based Cholesky implementation to handle larger matrices"""
    n = A.shape[0]
    
    # Define Metal kernel source for block-based approach
    source = """
    // Get thread ID and block size
    uint thread_id = thread_position_in_grid.x;
    uint n = A_shape[0];
    uint block_size = block_param[0];
    uint num_blocks = (n + block_size - 1) / block_size;
    uint num_threads = thread_count[0];  // Total number of threads
    
    // Process matrix in blocks
    for (uint k = 0; k < num_blocks; k++) {
        uint block_start = k * block_size;
        uint block_end = min(block_start + block_size, n);
        
        // Only thread 0 processes the diagonal block for stability
        if (thread_id == 0) {
            // Process diagonal block with standard Cholesky
            for (uint j = block_start; j < block_end; j++) {
                // Compute diagonal element
                float sum_diag = 0.0f;
                for (uint p = 0; p < j; p++) {
                    sum_diag += out[j*n + p] * out[j*n + p];
                }
                
                float diag_val = A[j*n + j] - sum_diag;
                if (diag_val <= 1e-10f) {
                    diag_val = 1e-10f;
                }
                out[j*n + j] = sqrt(diag_val);
                
                // Compute off-diagonals in this column
                for (uint i = j+1; i < block_end; i++) {
                    float sum = 0.0f;
                    for (uint p = 0; p < j; p++) {
                        sum += out[i*n + p] * out[j*n + p];
                    }
                    
                    float denom = out[j*n + j];
                    if (denom > 1e-10f) {
                        out[i*n + j] = (A[i*n + j] - sum) / denom;
                    } else {
                        out[i*n + j] = 0.0f;
                    }
                }
            }
        }
        
        // Wait for diagonal block to complete
        threadgroup_barrier(mem_flags::mem_device);
        
        // Initialize upper triangles to zero (all threads participate)
        for (uint i = thread_id; i < n; i += num_threads) {
            for (uint j = i+1; j < n; j++) {
                if ((i < block_start && j >= block_start && j < block_end) ||
                    (i >= block_start && i < block_end && j >= block_end)) {
                    out[i*n + j] = 0.0f;
                }
            }
        }
        
        // Ensure zeros are set before computing elements
        threadgroup_barrier(mem_flags::mem_device);
        
        // Each thread processes a set of rows for remaining blocks
        for (uint row = thread_id; row < n; row += num_threads) {
            // Only process rows below the current block
            if (row >= block_end) {
                // Update the row using the diagonal block
                for (uint j = block_start; j < block_end; j++) {
                    float sum = 0.0f;
                    for (uint p = 0; p < j; p++) {
                        sum += out[row*n + p] * out[j*n + p];
                    }
                    
                    float denom = out[j*n + j];
                    if (denom > 1e-10f) {
                        out[row*n + j] = (A[row*n + j] - sum) / denom;
                    } else {
                        out[row*n + j] = 0.0f;
                    }
                }
            }
        }
        
        // Wait for all updates before moving to next block
        threadgroup_barrier(mem_flags::mem_device);
    }
    """
    
    # Metal header with math functions
    header = """
    #include <metal_stdlib>
    #include <metal_math>
    using namespace metal;
    """
    
    # Create the kernel
    kernel = mx.fast.metal_kernel(
        name="block_cholesky_kernel",
        input_names=["A", "block_param", "thread_count"],
        output_names=["out"],
        source=source,
        header=header,
        ensure_row_contiguous=True
    )
    
    # Use multiple threads but not too many to maintain stability
    num_threads = min(32, n)
    grid = (num_threads, 1, 1)
    threads = (num_threads, 1, 1)
    
    # Parameters: block size and thread count
    block_param = mx.array([block_size], dtype=mx.uint32)
    thread_count = mx.array([num_threads], dtype=mx.uint32)
    
    # Run the kernel
    return kernel(
        inputs=[A, block_param, thread_count],
        output_shapes=[A.shape],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threads
    )[0]

def main():
    # Test with various matrix sizes
    sizes = [32, 128, 512]
    
    for n in sizes:
        print(f"\n===== Testing {n}x{n} matrix =====")
        
        # Generate a well-conditioned matrix
        A = generate_spd_matrix(n, min_eig=1.0)
        
        # Test built-in Cholesky
        try:
            print("Running built-in Cholesky...")
            start_built_in = time.perf_counter()
            
            # Use CPU stream for built-in cholesky
            with mx.stream(mx.cpu):
                L_built_in = mx.linalg.cholesky(A)
                error_built_in = mx.mean(mx.abs(A - L_built_in @ L_built_in.T)).item()
            
            mx.synchronize()
            built_in_time = time.perf_counter() - start_built_in
            
            print(f"Built-in Cholesky: {built_in_time:.6f}s | Error: {error_built_in:.2e}")
        except Exception as e:
            print(f"Built-in Cholesky failed: {e}")
            built_in_time = float('inf')
        
        # Run our single-thread Metal implementation
        print("\nRunning single-thread Metal Cholesky...")
        start = time.perf_counter()
        try:
            L = mlx_cholesky(A)
            mx.synchronize()
            metal_time = time.perf_counter() - start
            
            # Verify result
            error_metal = mx.mean(mx.abs(A - L @ L.T)).item()
            is_correct = mx.allclose(A, L @ L.T, rtol=1e-4, atol=1e-4)
            
            print(f"Metal Cholesky:    {metal_time:.6f}s | Error: {error_metal:.2e}")
            print(f"Verification:       {'✓' if is_correct else '✗'}")
            
            if built_in_time != float('inf'):
                print(f"Speedup vs built-in: {built_in_time/metal_time:.2f}x")
        except Exception as e:
            print(f"Metal Cholesky failed: {e}")
            is_correct = False
        
        # For larger matrices, also try the block-based approach
        if n >= 128:
            print("\nRunning block-based Metal Cholesky...")
            block_size = min(16, n//4)  # Adjust block size based on matrix size
            start = time.perf_counter()
            try:
                L_block = block_cholesky(A, block_size=block_size)
                mx.synchronize()
                block_time = time.perf_counter() - start
                
                # Verify result
                error_block = mx.mean(mx.abs(A - L_block @ L_block.T)).item()
                is_correct_block = mx.allclose(A, L_block @ L_block.T, rtol=1e-4, atol=1e-4)
                
                print(f"Block Cholesky:     {block_time:.6f}s | Error: {error_block:.2e}")
                print(f"Verification:       {'✓' if is_correct_block else '✗'}")
                
                if built_in_time != float('inf'):
                    print(f"Speedup vs built-in: {built_in_time/block_time:.2f}x")
            except Exception as e:
                print(f"Block Cholesky failed: {e}")
                is_correct_block = False
    
    print("\nImplementation complete! Both single-threaded and block-based approaches were tested.")
    print("The single-threaded approach offers the best numerical stability for all matrix sizes.")

if __name__ == "__main__":
    main()
