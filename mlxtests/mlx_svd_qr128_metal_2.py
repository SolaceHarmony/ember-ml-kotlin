"""
Standalone SVD implementation using a Metal kernel for QR decomposition with 128-bit precision.
This version integrates the QR-128 Metal kernel with the SVD implementation.
"""
from typing import Union, Tuple, Optional
import mlx.core as mx

# Import the QR-128 Metal kernel implementation
from mlxtests.mlx_qr_128_metal_kernel import qr_128_metal, MAX_MATRIX_DIM

# Define Metal kernel source string for power iteration (from mlx_svd_standalone.py)
_power_iter_tensor_kernel_source = """
#define NUM_LIMBS 8
#define LIMB_TILE_SIZE 32
#define EPSILON 1e-10f

// Single-threaded power iteration with improved precision using 16-bit limbs
if (thread_position_in_grid.x == 0) {
    uint n = shapeParams[0];
    uint k = shapeParams[1];
    uint num_iterations = iterParams[0];
    float tolerance = tolParams[0];

    // Initialize Q_out with Q_init
    for (uint row = 0; row < n; row++) {
        for (uint col = 0; col < k; col++) {
            Q_out[row * k + col] = Q_init[row * k + col];
        }
    }

    // Power iteration with improved precision
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Process each column
        for (uint col = 0; col < k; col++) {
            // Matrix multiplication: Z = A * Q[:, col]
            float Z[1024];  // Use float for Z to avoid precision issues
            
            // Initialize Z to zero
            for (uint row = 0; row < n; row++) {
                Z[row] = 0.0f;
            }
            
            // Compute Z = A * Q[:, col] in tiles for better memory management
            for (uint row = 0; row < n; row++) {
                for (uint tile_start = 0; tile_start < n; tile_start += LIMB_TILE_SIZE) {
                    uint tile_end = min(tile_start + LIMB_TILE_SIZE, n);
                    for (uint i = tile_start; i < tile_end; i++) {
                        Z[row] += A[row * n + i] * Q_out[i * k + col];
                    }
                }
            }
            
            // Orthogonalize Z against previous columns
            for (uint j = 0; j < col; j++) {
                // Compute dot product: proj = Q[:, j]' * Z
                float proj = 0.0f;
                
                for (uint row = 0; row < n; row++) {
                    proj += Q_out[row * k + j] * Z[row];
                }
                
                // Subtract projection: Z = Z - proj * Q[:, j]
                for (uint row = 0; row < n; row++) {
                    Z[row] -= proj * Q_out[row * k + j];
                }
            }
            
            // Compute norm squared: norm_sq = Z' * Z
            float norm_sq = 0.0f;
            
            for (uint row = 0; row < n; row++) {
                norm_sq += Z[row] * Z[row];
            }
            
            // Compute norm = sqrt(norm_sq)
            float norm = sqrt(norm_sq);
            
            // Normalize Z and store in Q[:, col]
            if (norm > tolerance) {
                float inv_norm = 1.0f / norm;
                
                for (uint row = 0; row < n; row++) {
                    Q_out[row * k + col] = Z[row] * inv_norm;
                }
            } else {
                // If norm is too small, set to zero
                for (uint row = 0; row < n; row++) {
                    Q_out[row * k + col] = 0.0f;
                }
            }
        }
    }
    
    // Final Gram-Schmidt orthogonalization for numerical stability
    for (uint col = 0; col < k; col++) {
        // Orthogonalize against previous columns
        for (uint j = 0; j < col; j++) {
            float dot = 0.0f;
            for (uint row = 0; row < n; row++) {
                dot += Q_out[row * k + j] * Q_out[row * k + col];
            }
            
            for (uint row = 0; row < n; row++) {
                Q_out[row * k + col] -= dot * Q_out[row * k + j];
            }
        }
        
        // Renormalize
        float norm_sq = 0.0f;
        for (uint row = 0; row < n; row++) {
            norm_sq += Q_out[row * k + col] * Q_out[row * k + col];
        }
        
        float norm = sqrt(norm_sq);
        if (norm > EPSILON) {
            float inv_norm = 1.0f / norm;
            for (uint row = 0; row < n; row++) {
                Q_out[row * k + col] *= inv_norm;
            }
        } else {
            // Handle numerically zero vectors
            for (uint row = 0; row < n; row++) {
                Q_out[row * k + col] = 0.0f;
            }
        }
    }
}
"""

# Compile the kernel at module level
_power_iter_tensor_kernel_compiled = mx.fast.metal_kernel(
    name="power_iter_tensor_kernel",
    source=_power_iter_tensor_kernel_source,
    input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
    output_names=["Q_out"],
    ensure_row_contiguous=True
)

def svd_with_metal_qr(a: mx.array, full_matrices: bool = True, compute_uv: bool = True, 
                    k: int = -1) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute SVD using a custom Metal kernel for QR decomposition and eigendecomposition.
    This implementation uses the QR-128 Metal kernel for improved numerical stability.

    Args:
        a: Input matrix.
        full_matrices: If True, compute full U and Vh.
        compute_uv: If True, compute U and Vh.
        k: Number of singular values/vectors to compute (-1 for all).

    Returns:
        U, S, Vh if compute_uv is True, otherwise S.
    """
    # Convert input to float32 array
    a_arr = mx.array(a, dtype=mx.float32)
    
    # Get dimensions
    m, n = a_arr.shape
    
    # Check if dimensions exceed the maximum supported by the Metal kernel
    if m > MAX_MATRIX_DIM or n > MAX_MATRIX_DIM:
        raise ValueError(f"Matrix dimensions exceed maximum supported size ({MAX_MATRIX_DIM})")
    
    # Compute rank and handle k parameter
    rank = min(m, n)
    k_val = rank if k == -1 else min(k, rank)
    
    # Determine if m >= n
    if m >= n:
        # Compute A^T A
        ata = mx.matmul(mx.transpose(a_arr), a_arr)
        
        # Initialize Q with random orthonormal vectors using GPU-compatible QR
        q_init = mx.random.normal((n, k_val), dtype=a_arr.dtype)
        q_init, _ = qr_128_metal(q_init)  # Use our Metal kernel QR implementation
        
        # Prepare kernel parameters
        shape_params = mx.array([n, k_val], dtype=mx.uint32)
        iter_params = mx.array([100], dtype=mx.uint32)
        tol_params = mx.array([1e-7], dtype=mx.float32)
        
        # Call the kernel
        grid = (1, 1, 1)
        threads = (1, 1, 1)
        
        v = _power_iter_tensor_kernel_compiled(
            inputs=[ata, q_init, shape_params, iter_params, tol_params],
            output_shapes=[(n, k_val)],
            output_dtypes=[a_arr.dtype],
            grid=grid,
            threadgroup=threads
        )[0]
        
        # Calculate eigenvalues using Rayleigh quotient
        rayleigh = mx.matmul(mx.transpose(v), mx.matmul(ata, v))
        eigenvalues = mx.diag(rayleigh)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = mx.argsort(mx.negative(mx.abs(eigenvalues)))
        eigenvalues = eigenvalues[sort_indices]
        v = v[:, sort_indices]
        
        # Singular values are sqrt of eigenvalues
        # Add a small epsilon to avoid negative values due to numerical errors
        epsilon = mx.array(1e-10)
        eigenvalues_safe = mx.maximum(eigenvalues, epsilon)
        s = mx.sqrt(eigenvalues_safe)
        
        if not compute_uv:
            return s
        
        # Compute U = A V S^-1
        epsilon = mx.multiply(mx.array(1e-10), mx.max(s))
        mask = mx.subtract(mx.abs(s), epsilon)  # Avoid direct comparison
        s_inv = mx.where(mx.greater_equal(mask, 0), mx.reciprocal(s), mx.zeros_like(s))
        u = mx.matmul(a_arr, mx.multiply(v, s_inv.reshape(1, -1)))
        
        # Complete basis if needed
        if full_matrices:
            if m > k_val:
                # Generate random vectors for remaining columns
                remaining_cols = mx.subtract(m, k_val)
                
                # Start with random vectors
                random_basis = mx.random.normal((m, remaining_cols), dtype=u.dtype)
                
                # Orthogonalize against existing vectors
                for i in mx.arange(k_val.item() if isinstance(k_val, mx.array) else k_val).tolist():
                    ui = u[:, i:i+1]
                    proj = mx.matmul(mx.transpose(ui), random_basis)
                    random_basis = mx.subtract(random_basis, mx.matmul(ui, proj))
                
                # Normalize columns
                for i in mx.arange(remaining_cols.item()).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = mx.linalg.norm(col)
                    mask = mx.subtract(mx.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = mx.where(mx.greater_equal(mask, 0), mx.reciprocal(col_norm), mx.array(0.0))
                    
                    # Update column using tensor operations
                    scaled_col = mx.multiply(col, scale)
                    random_basis = mx.array(random_basis)  # Create a copy
                    
                    # Update column i
                    for j in mx.arange(random_basis.shape[0]).tolist():
                        random_basis = random_basis.at[j, i].add(mx.subtract(scaled_col[j, 0], random_basis[j, i]))
                
                u = mx.concatenate([u, random_basis], axis=1)
            
            if n > k_val:
                # Generate random vectors for remaining columns
                remaining_cols = mx.subtract(n, k_val)
                
                # Start with random vectors
                random_basis = mx.random.normal((n, remaining_cols), dtype=v.dtype)
                
                # Orthogonalize against existing vectors
                for i in mx.arange(k_val.item() if isinstance(k_val, mx.array) else k_val).tolist():
                    vi = v[:, i:i+1]
                    proj = mx.matmul(mx.transpose(vi), random_basis)
                    random_basis = mx.subtract(random_basis, mx.matmul(vi, proj))
                
                # Normalize columns
                for i in mx.arange(remaining_cols.item()).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = mx.linalg.norm(col)
                    mask = mx.subtract(mx.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = mx.where(mx.greater_equal(mask, 0), mx.reciprocal(col_norm), mx.array(0.0))
                    
                    # Update column using tensor operations
                    scaled_col = mx.multiply(col, scale)
                    random_basis = mx.array(random_basis)  # Create a copy
                    
                    # Update column i
                    for j in mx.arange(random_basis.shape[0]).tolist():
                        random_basis = random_basis.at[j, i].add(mx.subtract(scaled_col[j, 0], random_basis[j, i]))
                
                v = mx.concatenate([v, random_basis], axis=1)
        
        return u, s, mx.transpose(v)
    
    else:  # m < n
        # Compute A A^T
        aat = mx.matmul(a_arr, mx.transpose(a_arr))
        
        # Initialize Q with random orthonormal vectors using GPU-compatible QR
        q_init = mx.random.normal((m, k_val), dtype=a_arr.dtype)
        q_init, _ = qr_128_metal(q_init)  # Use our Metal kernel QR implementation
        
        # Prepare kernel parameters
        shape_params = mx.array([m, k_val], dtype=mx.uint32)
        iter_params = mx.array([100], dtype=mx.uint32)
        tol_params = mx.array([1e-7], dtype=mx.float32)
        
        # Call the kernel
        grid = (1, 1, 1)
        threads = (1, 1, 1)
        
        u = _power_iter_tensor_kernel_compiled(
            inputs=[aat, q_init, shape_params, iter_params, tol_params],
            output_shapes=[(m, k_val)],
            output_dtypes=[a_arr.dtype],
            grid=grid,
            threadgroup=threads
        )[0]
        
        # Calculate eigenvalues using Rayleigh quotient
        rayleigh = mx.matmul(mx.transpose(u), mx.matmul(aat, u))
        eigenvalues = mx.diag(rayleigh)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = mx.argsort(mx.negative(mx.abs(eigenvalues)))
        eigenvalues = eigenvalues[sort_indices]
        u = u[:, sort_indices]
        
        # Singular values are sqrt of eigenvalues
        # Add a small epsilon to avoid negative values due to numerical errors
        epsilon = mx.array(1e-10)
        eigenvalues_safe = mx.maximum(eigenvalues, epsilon)
        s = mx.sqrt(eigenvalues_safe)
        
        if not compute_uv:
            return s
        
        # Compute V = A^T U S^-1
        epsilon = mx.multiply(mx.array(1e-10), mx.max(s))
        mask = mx.subtract(mx.abs(s), epsilon)  # Avoid direct comparison
        s_inv = mx.where(mx.greater_equal(mask, 0), mx.reciprocal(s), mx.zeros_like(s))
        
        # Count nonzero singular values
        nonzero_mask = mx.greater_equal(mask, 0)
        nonzero_count = int(mx.sum(nonzero_mask).item())  # Need item() for slicing
        
        # Use the mask to select columns of u and s_inv
        u_nonzero = u[:, :nonzero_count]
        s_inv_nonzero = s_inv[:nonzero_count]
        
        v = mx.matmul(mx.transpose(a_arr), mx.multiply(u_nonzero, s_inv_nonzero.reshape(1, -1)))
        
        # Complete basis if needed
        if full_matrices:
            if n > nonzero_count:
                # Generate random vectors for remaining columns
                remaining_cols = mx.subtract(n, nonzero_count)
                
                # Start with random vectors
                random_basis = mx.random.normal((n, remaining_cols), dtype=v.dtype)
                
                # Orthogonalize against existing vectors
                for i in mx.arange(nonzero_count if isinstance(nonzero_count, (int, float)) else nonzero_count.item()).tolist():
                    vi = v[:, i:i+1]
                    proj = mx.matmul(mx.transpose(vi), random_basis)
                    random_basis = mx.subtract(random_basis, mx.matmul(vi, proj))
                
                # Normalize columns
                for i in mx.arange(remaining_cols.item()).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = mx.linalg.norm(col)
                    mask = mx.subtract(mx.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = mx.where(mx.greater_equal(mask, 0), mx.reciprocal(col_norm), mx.array(0.0))
                    
                    # Update column using tensor operations
                    scaled_col = mx.multiply(col, scale)
                    random_basis = mx.array(random_basis)  # Create a copy
                    
                    # Update column i
                    for j in mx.arange(random_basis.shape[0]).tolist():
                        random_basis = random_basis.at[j, i].add(mx.subtract(scaled_col[j, 0], random_basis[j, i]))
                
                v = mx.concatenate([v, random_basis], axis=1)
            
            if m > nonzero_count:
                # Generate random vectors for remaining columns
                remaining_cols = mx.subtract(m, nonzero_count)
                
                # Start with random vectors
                random_basis = mx.random.normal((m, remaining_cols), dtype=u.dtype)
                
                # Orthogonalize against existing vectors
                for i in mx.arange(nonzero_count if isinstance(nonzero_count, (int, float)) else nonzero_count.item()).tolist():
                    ui = u[:, i:i+1]
                    proj = mx.matmul(mx.transpose(ui), random_basis)
                    random_basis = mx.subtract(random_basis, mx.matmul(ui, proj))
                
                # Normalize columns
                for i in mx.arange(remaining_cols.item()).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = mx.linalg.norm(col)
                    mask = mx.subtract(mx.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = mx.where(mx.greater_equal(mask, 0), mx.reciprocal(col_norm), mx.array(0.0))
                    
                    # Update column using tensor operations
                    scaled_col = mx.multiply(col, scale)
                    random_basis = mx.array(random_basis)  # Create a copy
                    
                    # Update column i
                    for j in mx.arange(random_basis.shape[0]).tolist():
                        random_basis = random_basis.at[j, i].add(mx.subtract(scaled_col[j, 0], random_basis[j, i]))
                
                u = mx.concatenate([u, random_basis], axis=1)
        
        return u, s, mx.transpose(v)


if __name__ == "__main__":
    print("Testing SVD with Metal Kernel QR-128...")
    
    A = mx.array([[1.0, 2.0], [3.0, 4.0]])
    print("Input Matrix A:\n", A)

    try:
        # Use the integrated function
        U, S, Vh = svd_with_metal_qr(A, full_matrices=True, compute_uv=True)
        print("\nResult (SVD with Metal Kernel QR-128):")
        print("U:\n", U)
        print("S:\n", S)
        print("Vh:\n", Vh)

        # Reconstruction Check
        k_rank = S.shape[0]
        U_rec = U[:, :k_rank]
        Vh_rec = Vh[:k_rank, :]
        A_rec = mx.matmul(mx.multiply(U_rec, S.reshape(1, -1)), Vh_rec)
        print("\nReconstruction Check:")
        print("Original A:\n", A)
        print("Reconstructed A:\n", A_rec)

        # Calculate reconstruction error
        reconstruction_error = mx.sqrt(mx.mean(mx.square(mx.subtract(A, A_rec))))
        max_error = mx.max(mx.abs(mx.subtract(A, A_rec)))
        print("Reconstruction Error (RMSE):", reconstruction_error)
        print("Max Absolute Error:", max_error)

        # Test with different matrix shapes
        print("\nTesting with different matrix shapes...")
        
        # Test with tall-skinny matrix
        A_tall = mx.random.normal((60, 30))
        U_tall, S_tall, Vh_tall = svd_with_metal_qr(A_tall)
        A_tall_rec = mx.matmul(mx.multiply(U_tall[:, :30], S_tall.reshape(1, -1)), Vh_tall[:30])
        error_tall = mx.sqrt(mx.mean(mx.square(mx.subtract(A_tall, A_tall_rec))))
        print(f"Tall matrix (60x30) - Reconstruction error: {error_tall}")
        
        # Test with wide matrix
        A_wide = mx.random.normal((20, 60))
        U_wide, S_wide, Vh_wide = svd_with_metal_qr(A_wide)
        A_wide_rec = mx.matmul(mx.multiply(U_wide[:, :20], S_wide.reshape(1, -1)), Vh_wide[:20])
        error_wide = mx.sqrt(mx.mean(mx.square(mx.subtract(A_wide, A_wide_rec))))
        print(f"Wide matrix (20x60) - Reconstruction error: {error_wide}")
        
        # Test with square matrix
        A_square = mx.random.normal((60, 60))
        U_square, S_square, Vh_square = svd_with_metal_qr(A_square)
        A_square_rec = mx.matmul(mx.multiply(U_square, S_square.reshape(1, -1)), Vh_square)
        error_square = mx.sqrt(mx.mean(mx.square(mx.subtract(A_square, A_square_rec))))
        print(f"Square matrix (60x60) - Reconstruction error: {error_square}")

    except Exception as e:
        print(f"\nError during SVD with Metal Kernel QR-128 test: {e}")
        import traceback
        traceback.print_exc()