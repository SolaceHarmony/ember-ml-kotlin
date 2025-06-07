"""
GPU-compatible SVD implementation for MLX backend in Ember ML.

This module provides an implementation of the SVD algorithm using the power method
as described in the paper "Distributed Out-of-Memory SVD on CPU/GPU Architectures".
It's designed to be integrated into the Ember ML backend for MLX to provide a
GPU-compatible alternative to the native MLX SVD which currently only works on CPU.
"""

from typing import Tuple, Optional, Union, Any
import mlx.core as mx

def svd_gpu_compatible(
    a: mx.array, 
    full_matrices: bool = True, 
    compute_uv: bool = True,
    epsilon: float = 1e-6,
    k: int = -1
) -> Union[Tuple[mx.array, mx.array, mx.array], mx.array]:
    """
    Compute the Singular Value Decomposition of a matrix using the power method.
    
    This implementation is based on the algorithm described in the paper
    "Distributed Out-of-Memory SVD on CPU/GPU Architectures" and provides
    a GPU-compatible alternative to the native MLX SVD which currently
    only works on CPU.
    
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
    # This implementation is intended for GPU compatibility.
    # We will remove the CPU fallback logic.
    
    # Handle the case where a is a batch of matrices
    if a.ndim > 2:
        raise NotImplementedError("Batched SVD not yet implemented for power method")
    
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
            U_l = mx.concatenate(U_list[:l], axis=1)
            V_l = mx.concatenate(V_list[:l], axis=1)
            sigma_l = mx.array(sigma_list[:l])
            
            # Create a diagonal matrix from sigma_l
            Sigma_l_diag = mx.diag(sigma_l)
            
            # Compute the part to subtract: U_l @ Sigma_l_diag @ V_l.transpose()
            reconstructed_part = mx.matmul(mx.matmul(U_l, Sigma_l_diag), mx.transpose(V_l))
            
            # Subtract from the current X to get the residual
            X = X - reconstructed_part
        
        # Compute the next singular vector using the power method
        if m > n:
            # Compute the right singular vector first
            v_l = _svd_1d_power_method(X, epsilon)
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
            u_l = _svd_1d_power_method(X, epsilon)
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


def _svd_1d_power_method(X: mx.array, epsilon: float) -> mx.array:
    """
    Estimate the first singular vector of matrix X using the power method.
    
    Args:
        X: Input matrix (m x n).
        epsilon: Convergence tolerance.
    
    Returns:
        The estimated first singular vector.
    """
    m, n = X.shape
    k = min(m, n)
    
    # Initialize with random vector
    x = mx.random.normal(shape=(k, 1)) # No need for extra mx.array()
    
    # Normalize
    x = x / mx.linalg.norm(x)
    v0 = x
    
    # Compute the appropriate Gram matrix
    if m > n:
        B = mx.matmul(mx.transpose(X), X)
    else:
        B = mx.matmul(X, mx.transpose(X))
    
    # Power iteration
    while True:
        v1 = mx.matmul(B, v0)
        
        # Normalize
        norm_v1 = mx.linalg.norm(v1)
        if norm_v1 < 1e-10:
            break
        v1 = v1 / norm_v1
        
        # Check convergence
        dot_product = mx.squeeze(mx.matmul(mx.transpose(v0), v1))
        if mx.abs(dot_product) >= 1.0 - epsilon:
            return v1
        
        v0 = v1
    
    return v1


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


# For integration into Ember ML backend (GPU-only version)
def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True, k: int = -1) -> Union[Tuple[Any, Any, Any], Any]:
    """
    Compute the Singular Value Decomposition of a matrix using a GPU-compatible method.
    
    This function is designed to be integrated into the Ember ML backend for MLX,
    providing a GPU-compatible SVD implementation using the power method.
    It assumes execution on a GPU or a backend where the power method is preferred.
    
    Args:
        a: Input array with shape (..., M, N). Assumed to be an mx.array.
        full_matrices: If True, return U and Vh with shapes (..., M, M) and (..., N, N).
                      Otherwise, return shapes (..., M, K) and (..., K, N), where
                      K = min(M, N) or the specified k.
        compute_uv: If True, compute and return U and Vh in addition to S.
                   Otherwise, only S is returned.
        k: Number of singular values/vectors to compute. If -1, compute min(M, N).
           Only used when compute_uv is True.
    
    Returns:
        If compute_uv is True, returns (U, S, Vh) containing the singular values
        and optionally the singular vectors. Otherwise, returns S.
        
        S: The singular values, sorted in descending order.
        U: The matrix of left singular vectors.
        Vh: The matrix of right singular vectors (transposed).
    """
    # Directly use the GPU-compatible power method implementation
    # Pass the 'k' parameter along
    return svd_gpu_compatible(a, full_matrices=full_matrices, compute_uv=compute_uv, k=k)


# Removed __main__ block for library integration.
# Testing should be done via dedicated test files (e.g., pytest).