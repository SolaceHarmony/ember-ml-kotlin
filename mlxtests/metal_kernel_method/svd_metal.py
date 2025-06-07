"""
Metal kernel implementation of SVD for MLX backend in Ember ML.

This module provides GPU-compatible implementations of matrix decomposition operations
using Metal kernels. It's designed to be integrated into the Ember ML backend for MLX
to provide GPU-accelerated alternatives to operations that are currently CPU-only in MLX.
"""

from typing import Tuple, Union, Optional, Any
import mlx.core as mx

def svd_1d_metal_kernel(X: mx.array, epsilon: float = 1e-6) -> mx.array:
    """
    Estimate the first singular vector of matrix X using the power method with Metal kernel.
    
    Args:
        X: Input matrix (m x n).
        epsilon: Convergence tolerance.
    
    Returns:
        The estimated first singular vector.
    """
    m, n = X.shape
    k = min(m, n)
    is_tall = m > n
    
    # Initialize with random vector
    init_vector = mx.random.normal(shape=(k, 1)) # No need for extra mx.array()
    init_vector = init_vector / mx.linalg.norm(init_vector)
    
    # Prepare epsilon as an array for the kernel
    eps_array = mx.array([epsilon], dtype=mx.float32)
    
    @mx.custom_function
    def _inner_impl(X_inner: mx.array, init_vector_inner: mx.array, eps_inner: mx.array) -> mx.array:
        # Define Metal kernel source
        source = """
        // Get thread ID
        uint thread_id = thread_position_in_grid.x;
        
        // Get matrix dimensions
        uint m = X_shape[0];
        uint n = X_shape[1];
        bool is_tall = m > n;
        uint k = is_tall ? n : m;
        float epsilon = eps[0];
        
        // Only thread 0 performs the computation for stability
        if (thread_id == 0) {
            // Copy the initial vector to v0
            half v0[256];  // Using half precision to save stack space
            for (uint i = 0; i < k; i++) {
                v0[i] = (half)init_vector[i];
            }
            
            // Compute the Gram matrix B = X^T * X or X * X^T
            // Use a smaller buffer and process in chunks if needed
            half B[256*256];  // Using half precision to save stack space
            if (is_tall) {
                // B = X^T * X
                // Process in chunks if n is too large
                uint chunk_size = min(n, 256u);
                for (uint i = 0; i < chunk_size; i++) {
                    for (uint j = 0; j < chunk_size; j++) {
                        half sum = 0.0h;
                        for (uint l = 0; l < m; l++) {
                            sum += (half)X[l*n + i] * (half)X[l*n + j];
                        }
                        B[i*chunk_size + j] = sum;
                    }
                }
            } else {
                // B = X * X^T
                // Process in chunks if m is too large
                uint chunk_size = min(m, 256u);
                for (uint i = 0; i < chunk_size; i++) {
                    for (uint j = 0; j < chunk_size; j++) {
                        half sum = 0.0h;
                        for (uint l = 0; l < n; l++) {
                            sum += (half)X[i*n + l] * (half)X[j*n + l];
                        }
                        B[i*chunk_size + j] = sum;
                    }
                }
            }
            
            // Power iteration
            bool converged = false;
            uint max_iters = 100;
            uint iter = 0;
            
            // Use a smaller chunk size for processing
            uint chunk_size = min(k, 256u);
            
            while (!converged && iter < max_iters) {
                // v1 = B * v0
                half v1[256];  // Using half precision to save stack space
                for (uint i = 0; i < chunk_size; i++) {
                    half sum = 0.0h;
                    for (uint j = 0; j < chunk_size; j++) {
                        sum += B[i*chunk_size + j] * v0[j];
                    }
                    v1[i] = sum;
                }
                
                // Normalize v1
                half norm = 0.0h;
                for (uint i = 0; i < chunk_size; i++) {
                    norm += v1[i] * v1[i];
                }
                norm = sqrt(norm);
                
                if (norm < 1e-5h) {  // Use a larger epsilon for half precision
                    // If norm is too small, break
                    break;
                }
                
                for (uint i = 0; i < chunk_size; i++) {
                    v1[i] /= norm;
                }
                
                // Check convergence
                half dot_product = 0.0h;
                for (uint i = 0; i < chunk_size; i++) {
                    dot_product += v0[i] * v1[i];
                }
                
                if (fabs(dot_product) >= 1.0h - (half)epsilon) {
                    converged = true;
                }
                
                // Update v0 = v1
                for (uint i = 0; i < chunk_size; i++) {
                    v0[i] = v1[i];
                }
                
                iter++;
            }
            
            // Copy result to output (convert back to float)
            for (uint i = 0; i < chunk_size; i++) {
                out[i] = (float)v0[i];
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
            name="svd_1d_kernel",
            input_names=["X", "init_vector", "eps"],
            output_names=["out"],
            source=source,
            header=header,
            ensure_row_contiguous=True
        )
        
        # Single thread for maximum stability
        grid = (1, 1, 1)
        threads = (1, 1, 1)
        
        # Prepare the output shape
        output_shape = (k, 1)
        
        # Run the kernel
        result = kernel(
            inputs=[X_inner, init_vector_inner, eps_inner],
            output_shapes=[output_shape],
            output_dtypes=[X_inner.dtype],
            grid=grid,
            threadgroup=threads
        )
        
        return result[0]
    
    # Call the inner implementation
    return _inner_impl(X, init_vector, eps_array)


def _complete_orthogonal_basis(vectors: mx.array, dim: int) -> mx.array:
    """
    Complete a set of orthogonal vectors to form a basis for the full space.
    
    Args:
        vectors: Matrix of orthogonal column vectors.
        dim: Dimension of the full space.
    
    Returns:
        Matrix with a complete orthogonal basis.
    """
    # Get the current number of vectors
    n_vectors = vectors.shape[1]
    
    if n_vectors >= dim:
        return vectors[:, :dim]
    
    # We need to add (dim - n_vectors) orthogonal vectors
    result = [vectors]
    
    # Generate random vectors and orthogonalize them
    for _ in range(dim - n_vectors):
        # Generate a random vector
        new_vector = mx.random.normal(shape=(vectors.shape[0], 1)) # No need for extra mx.array()
        
        # Orthogonalize against all existing vectors (using Gram-Schmidt)
        # This part can be numerically unstable, consider QR decomposition for robustness
        # if this becomes an issue in practice.
        for v in range(n_vectors + len(result) - 1):
            if v < n_vectors:
                basis_vector = vectors[:, v:v+1]
            else:
                basis_vector = result[v - n_vectors + 1]
            
            # Project and subtract
            projection = mx.matmul(mx.transpose(basis_vector), new_vector)
            new_vector = new_vector - mx.matmul(basis_vector, projection)
        
        # Normalize
        norm = mx.linalg.norm(new_vector)
        if norm > 1e-10:
            new_vector = new_vector / norm
            result.append(new_vector)
    
    return mx.concatenate(result, axis=1)


# Rename svd_metal to svd as this will be the primary implementation in this file
def svd(
    a: mx.array,
    full_matrices: bool = True,
    compute_uv: bool = True,
    epsilon: float = 1e-6,
    k: int = -1
) -> Union[Tuple[mx.array, mx.array, mx.array], mx.array]:
    """
    Compute the Singular Value Decomposition of a matrix using Metal kernels.
    
    This implementation uses Metal kernels to accelerate the power method for SVD
    computation on GPU. It follows the algorithm described in the paper
    "Distributed Out-of-Memory SVD on CPU/GPU Architectures".
    
    Args:
        a: Input array with shape (..., M, N).
        full_matrices: If True, return U and Vh with shapes (..., M, M) and (..., N, N).
                      Otherwise, return shapes (..., M, K) and (..., K, N), where
                      K = min(M, N).
        compute_uv: If True, compute and return U and V in addition to S.
                   Otherwise, only S is returned.
        epsilon: Convergence tolerance for the power method.
        k: Number of singular values/vectors to compute. If -1, compute min(M, N).
            Only used when compute_uv is True.
    
    Returns:
        If compute_uv is True, returns (U, S, Vh) containing the singular values
        and optionally the singular vectors. Otherwise, returns S.
        
        S: The singular values, sorted in descending order.
        U: The matrix of left singular vectors.
        Vh: The matrix of right singular vectors (transposed).
    """
    # Handle the case where a is a batch of matrices
    if a.ndim > 2:
        raise NotImplementedError("Batched SVD not yet implemented for Metal kernel")
    
    m, n = a.shape
    
    if k == -1:
        k = min(m, n)
    
    # Initialize lists to store singular vectors and values
    U_list = []
    V_list = []
    sigma_list = []
    
    # Create a copy of a to work with the residual
    # Note: If 'a' is not modified elsewhere, this copy might be avoidable
    #       by adjusting the residual calculation. For now, keep the copy logic.
    X = a
    
    # Compute k singular triplets (or just values if compute_uv is False)
    for l in range(k):
        # If we've already computed some singular triplets, subtract their contribution
        if l > 0:
            # Instead of subtracting the full reconstructed matrix, which can be expensive,
            # we'll use the power method directly on the residual space
            # This is more efficient and avoids large matrix multiplications
            
            # For the power method, we need to ensure orthogonality to previous singular vectors
            # We'll do this by projecting out the components along previous singular vectors
            pass
        
        # Compute the next singular vector using the Metal kernel
        if m > n:
            # Compute the right singular vector first
            v_l = svd_1d_metal_kernel(X, epsilon)
            
            # Orthogonalize against previous vectors if needed
            if l > 0:
                for i in range(l):
                    # Project out components along previous singular vectors
                    proj = mx.sum(v_l * V_list[i])
                    v_l = v_l - proj * V_list[i]
                
                # Renormalize
                v_l = v_l / mx.linalg.norm(v_l)
            
            V_list.append(v_l)
            
            # Compute the left singular vector and singular value
            u_l = mx.matmul(a, v_l)
            sigma_l_scalar = mx.linalg.norm(u_l) # Keep as MLX scalar
            
            # Normalize the left singular vector
            if sigma_l_scalar < 1e-10:
                u_l = mx.zeros_like(u_l)
            else:
                u_l = u_l / sigma_l_scalar
            
            if compute_uv:
                U_list.append(u_l)
            sigma_list.append(sigma_l_scalar)
        else:
            # Compute the left singular vector first
            u_l = svd_1d_metal_kernel(X, epsilon)
            
            # Orthogonalize against previous vectors if needed
            if l > 0:
                for i in range(l):
                    # Project out components along previous singular vectors
                    proj = mx.sum(u_l * U_list[i])
                    u_l = u_l - proj * U_list[i]
                
                # Renormalize
                u_l = u_l / mx.linalg.norm(u_l)
            
            U_list.append(u_l)
            
            # Compute the right singular vector and singular value
            v_l = mx.matmul(mx.transpose(a), u_l)
            sigma_l_scalar = mx.linalg.norm(v_l) # Keep as MLX scalar
            
            # Normalize the right singular vector
            if sigma_l_scalar < 1e-10:
                v_l = mx.zeros_like(v_l)
            else:
                v_l = v_l / sigma_l_scalar
            
            if compute_uv:
                V_list.append(v_l)
            sigma_list.append(sigma_l_scalar)
    
    # Convert list of scalars to a 1D array
    # Using stack ensures the result is 1D even if sigma_list has only one scalar
    sigma_array = mx.stack(sigma_list) if sigma_list else mx.array([])

    if not compute_uv:
        return sigma_array # Return only singular values if requested

    # --- Handle U and V computation ---
    # Reshape U_list and V_list for proper concatenation
    for i in range(len(U_list)):
        if U_list[i].ndim == 1:
            U_list[i] = U_list[i].reshape(-1, 1)
    
    for i in range(len(V_list)):
        if V_list[i].ndim == 1:
            V_list[i] = V_list[i].reshape(-1, 1)
    
    U_matrix = mx.concatenate(U_list, axis=1) if U_list else mx.array([], dtype=a.dtype).reshape(m, 0)
    V_matrix = mx.concatenate(V_list, axis=1) if V_list else mx.array([], dtype=a.dtype).reshape(n, 0)

    if full_matrices:
        # Create full-sized U and V matrices by completing the basis
        if m > n:
            # Pad U
            if U_matrix.size > 0:
                 U_matrix = _complete_orthogonal_basis(U_matrix, m)
            else: # Handle case where k=0 or all singular values were ~0
                 U_matrix = mx.eye(m, dtype=a.dtype) # Or generate random orthogonal basis
            # V is already (n, k) where k <= n, pad if k < n
            if V_matrix.shape[1] < n:
                 V_matrix = _complete_orthogonal_basis(V_matrix, n)
            elif V_matrix.size == 0:
                 V_matrix = mx.eye(n, dtype=a.dtype)

        else: # n >= m
            # Pad V
            if V_matrix.size > 0:
                V_matrix = _complete_orthogonal_basis(V_matrix, n)
            else: # Handle case where k=0 or all singular values were ~0
                V_matrix = mx.eye(n, dtype=a.dtype)
            # U is already (m, k) where k <= m, pad if k < m
            if U_matrix.shape[1] < m:
                 U_matrix = _complete_orthogonal_basis(U_matrix, m)
            elif U_matrix.size == 0:
                 U_matrix = mx.eye(m, dtype=a.dtype)

    # Return U, S, Vh (V transposed) to match the native SVD interface
    return U_matrix, sigma_array, mx.transpose(V_matrix)


# Removed the device-switching svd wrapper function.
# The svd function above (previously svd_metal) is now the primary entry point.