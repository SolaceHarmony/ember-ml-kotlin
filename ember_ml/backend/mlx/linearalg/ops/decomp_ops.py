"""
MLX solver operations for ember_ml.

This module provides MLX implementations of solver operations.
"""
from typing import Union, Tuple, Literal
import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike

dtype_obj = MLXDType()

def cholesky(a: TensorLike) -> mx.array:
    """
    Compute the Cholesky decomposition of a positive definite matrix.
    
    Args:
        a: Input positive definite matrix
    
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    
    Notes:
        This is a simplified implementation of the Cholesky decomposition.
        For large matrices or high precision requirements, consider using
        a more sophisticated algorithm.
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    
    # Initialize the result matrix
    l = mx.zeros((n, n), dtype=a_array.dtype)
    
    # Compute the Cholesky decomposition
    for i in range(n):
        # Use direct integer calculation
        i_plus_1_int = i + 1
        for j in range(i_plus_1_int):
            if mx.equal(i, j):
                # Diagonal element
                s = mx.subtract(a_array[i, i], mx.sum(mx.square(l[i, :j])))
                if mx.less(s, mx.array(0)):
                    raise ValueError("Matrix is not positive definite")
                
                # Create a new array with the updated value
                temp = mx.zeros_like(l)
                temp = temp.at[i, i].add(mx.sqrt(s))
                l = l + temp
            else:
                # Off-diagonal element
                s = mx.subtract(a_array[i, j], mx.sum(mx.multiply(l[i, :j], l[j, :j])))
                
                # Create a new array with the updated value
                temp = mx.zeros_like(l)
                temp = temp.at[i, j].add(mx.divide(s, l[j, j]))
                l = l + temp
    
    return l

def svd(a: TensorLike, 
        full_matrices: bool = True, compute_uv: bool = True) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute the singular value decomposition of a matrix using power iteration.
    
    Args:
        a: Input matrix
        full_matrices: If True, return full U and Vh matrices
        compute_uv: If True, compute U and Vh matrices
    
    Returns:
        If compute_uv is True, returns (U, S, Vh), otherwise returns S
    
    Notes:
        This is a simplified implementation using power iteration.
        For large matrices or high precision requirements, consider using
        a more sophisticated algorithm.
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor, MLXDType
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Get matrix dimensions
    m, n = a_array.shape
    k = min(m, n)
    
    # Compute A^T A for eigendecomposition
    if m >= n:
        # Use A^T A which is smaller
        ata = mx.matmul(mx.transpose(a_array), a_array)
        # Compute eigendecomposition of A^T A
        eigenvalues, eigenvectors = eig(ata)
        # Singular values are square roots of eigenvalues
        s = mx.sqrt(mx.abs(eigenvalues[:k]))
        # Sort singular values in descending order
        idx = mx.argsort(-s)
        s = s[idx]
        
        if compute_uv:
            # V comes directly from eigenvectors
            v = eigenvectors[:, idx]
            # Compute U from A*V/S
            u = mx.zeros((m, k), dtype=a_array.dtype)
            for i in range(k):
                if mx.greater(s[i], mx.array(1e-10)):  # Avoid division by very small values
                    u_col = mx.divide(mx.matmul(a_array, v[:, i]), s[i])
                    
                    # Update u column by column using direct indexing
                    for j in range(m):
                        u[j, i] = u_col[j]
                else:
                    # For very small singular values, use a different approach
                    u_col = mx.zeros((m,), dtype=a_array.dtype)
                    index = mx.remainder(mx.array(i), mx.array(m)).item()
                    u_col[index] = 1.0
                    
                    # Update u column by column using direct indexing
                    for j in range(m):
                        u[j, i] = u_col[j]
            
            # If full_matrices is True, pad U and V
            if full_matrices:
                if m > k:
                    # Pad U with orthogonal vectors
                    u_pad = mx.zeros((m, mx.subtract(mx.array(m), mx.array(k))), dtype=a_array.dtype)
                    # Simple orthogonalization (not fully robust)
                    # Use direct integer calculation
                    m_minus_k_int = m - k
                    for i in range(m_minus_k_int):
                        u_pad_col = mx.zeros((m,), dtype=a_array.dtype)
                        
                        # Calculate index
                        index = mx.add(mx.array(k), mx.array(i)).item()
                        
                        # Update u_pad_col using direct indexing
                        u_pad_col[index] = 1.0
                        u = mx.concatenate([u, u_pad_col.reshape(m, 1)], axis=1)
            
            # Return U, S, V^H
            return u, s, mx.transpose(v)
        else:
            return s
    else:
        # Use A A^T which is smaller
        aat = mx.matmul(a_array, mx.transpose(a_array))
        # Compute eigendecomposition of A A^T
        eigenvalues, eigenvectors = eig(aat)
        # Singular values are square roots of eigenvalues
        s = mx.sqrt(mx.abs(eigenvalues[:k]))
        # Sort singular values in descending order
        idx = mx.argsort(-s)
        s = s[idx]
        
        if compute_uv:
            # U comes directly from eigenvectors
            u = eigenvectors[:, idx]
            # Compute V from A^T*U/S
            v = mx.zeros((n, k), dtype=a_array.dtype)
            for i in range(k):
                if mx.greater(s[i], mx.array(1e-10)):  # Avoid division by very small values
                    v_col = mx.divide(mx.matmul(mx.transpose(a_array), u[:, i]), s[i])
                    
                    # Update v column by column using direct indexing
                    for j in range(n):
                        v[j, i] = v_col[j]
                else:
                    # For very small singular values, use a different approach
                    v_col = mx.zeros((n,), dtype=a_array.dtype)
                    index = mx.remainder(mx.array(i), mx.array(n)).item()
                    v_col[index] = 1.0
                    
                    # Update v column by column using direct indexing
                    for j in range(n):
                        v[j, i] = v_col[j]
            
            # If full_matrices is True, pad U and V
            if full_matrices:
                if n > k:
                    # Pad V with orthogonal vectors
                    v_pad = mx.zeros((n, mx.subtract(mx.array(n), mx.array(k))), dtype=a_array.dtype)
                    # Simple orthogonalization (not fully robust)
                    # Use direct integer calculation
                    n_minus_k_int = n - k
                    for i in range(n_minus_k_int):
                        v_pad_col = mx.zeros((n,), dtype=a_array.dtype)
                        
                        # Calculate index
                        index = mx.add(mx.array(k), mx.array(i)).item()
                        
                        # Update v_pad_col using direct indexing
                        v_pad_col[index] = 1.0
                        v = mx.concatenate([v, v_pad_col.reshape(n, 1)], axis=1)
            
            # Return U, S, V^H
            return u, s, mx.transpose(v)
        else:
            return s

def eig(a: TensorLike) -> Tuple[mx.array, mx.array]:
    """
    Compute the eigenvalues and eigenvectors of a square matrix using power iteration.
    
    Args:
        a: Input square matrix
    
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    
    Notes:
        This is a simplified implementation using power iteration.
        For large matrices or high precision requirements, consider using
        a more sophisticated algorithm.
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    
    # Initialize eigenvalues and eigenvectors
    eigenvalues = mx.zeros((n,), dtype=a_array.dtype)
    eigenvectors = mx.zeros((n, n), dtype=a_array.dtype)
    
    # Make a copy of the matrix that we can modify
    a_copy = mx.array(a_array)
    
    # Use power iteration to find eigenvalues and eigenvectors
    for i in range(n):
        # Initialize random vector
        v = mx.random.normal((n,), dtype=a_array.dtype)
        v = mx.divide(v, mx.sqrt(mx.sum(mx.square(v))))
        
        # Power iteration
        for _ in range(100):  # Maximum iterations
            v_new = mx.matmul(a_copy, v)
            v_new_norm = mx.sqrt(mx.sum(mx.square(v_new)))
            
            # Check for convergence
            if mx.less(v_new_norm, mx.array(1e-10)):
                break
                
            v = mx.divide(v_new, v_new_norm)
        
        # Compute Rayleigh quotient to get eigenvalue
        eigenvalue = mx.sum(mx.multiply(v, mx.matmul(a_copy, v)))
        
        # Store eigenvalue and eigenvector using MLX array operations
        # Create a new array with the updated value at index i
        eigenvalues = mx.array([eigenvalue if idx == i else eigenvalues[idx] for idx in range(n)])
        
        # Update eigenvectors using direct indexing
        for j in range(n):
            eigenvectors[j, i] = v[j]
        
        # Deflate the matrix to find the next eigenvalue
        # This is a simplified deflation and may not be numerically stable
        a_copy = mx.subtract(a_copy, mx.multiply(eigenvalue, mx.outer(v, v)))
    
    return eigenvalues, eigenvectors

def eigvals(a: TensorLike) -> mx.array:
    """
    Compute the eigenvalues of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Eigenvalues of the matrix
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_obj = Tensor.convert_to_tensor(a)
    eigenvalues, _ = eig(tensor_obj)
    return eigenvalues

def _add_double_single(a_high, a_low, b_high, b_low):
    """Helper for double-single precision arithmetic."""
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

class HPC16x8:
    """
    High-Precision Computing class for MLX.
    This is a simplified version of the HPC16x8 class from the orthogonal.py file.
    It provides support for non-square matrix QR decomposition.
    """
    
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def from_array(cls, array):
        """Convert a regular MLX array to HPC format."""
        return cls(array)
        
    def to_float32(self):
        """Convert back to regular MLX float32 array."""
        return self.data
        
    def qr(self):
        """
        Perform QR decomposition on the HPC matrix.
        This is a simplified implementation that delegates to the standard QR.
        """
        # For simplicity, we'll use the standard QR decomposition
        # In a real implementation, this would use a more numerically stable approach
        q, r = _standard_qr(self.data)
        return HPC16x8(q), HPC16x8(r)

def _standard_qr(a_array):
    """Standard QR decomposition using Gram-Schmidt orthogonalization."""
    m, n = a_array.shape
    k = min(m, n)
    q = mx.zeros((m, k), dtype=a_array.dtype)
    r = mx.zeros((k, n), dtype=a_array.dtype)
    
    # Modified Gram-Schmidt orthogonalization
    for j in range(k):
        # Get the j-th column of A
        v = a_array[:, j]
        
        # Orthogonalize against previous columns of Q
        for i in range(j):
            # Calculate r[i, j]
            r_ij = mx.sum(mx.multiply(q[:, i], v))
            
            # Create a new array with the updated value
            temp = mx.zeros_like(r)
            temp = temp.at[i, j].add(r_ij)
            r = r + temp
            
            # Update v
            v = mx.subtract(v, mx.multiply(r[i, j], q[:, i]))
        
        # Compute the norm of the orthogonalized vector
        r_jj = mx.sqrt(mx.sum(mx.square(v)))
        
        # Handle the case where the vector is close to zero
        if mx.less(r_jj, mx.array(1e-10)):
            # Update q using direct indexing
            for i in range(m):
                q[i, j] = 0.0
        else:
            # Update q using direct indexing
            v_normalized = mx.divide(v, r_jj)
            for i in range(m):
                q[i, j] = v_normalized[i]
        
        # Update R
        r[j, j] = r_jj
        
        # Compute the remaining elements of the j-th row of R
        j_plus_1_int = j + 1
        for k_idx in range(j_plus_1_int, n):
            r_jk = mx.sum(mx.multiply(q[:, j], a_array[:, k_idx]))
            
            # Create a new array with the updated value
            temp = mx.zeros_like(r)
            temp = temp.at[j, k_idx].add(r_jk)
            r = r + temp
    
    return q, r

def _custom_qr(matrix_high, matrix_low=None):
    """
    MLX-specific QR decomposition with increased numerical stability.
    
    Args:
        matrix_high: High-precision part of the matrix
        matrix_low: Low-precision part of the matrix (optional)
        
    Returns:
        Tuple of (Q, R) matrices
    """
    rows, cols = matrix_high.shape
    
    # Use HPC implementation for non-square matrices
    if rows != cols:
        matrix_hpc = HPC16x8.from_array(matrix_high)  # Convert to HPC format
        q_hpc, r_hpc = matrix_hpc.qr()  # HPC QR decomposition
        return q_hpc.to_float32(), r_hpc.to_float32()
    
    # For square matrices, use a more numerically stable approach
    # If no low-precision part is provided, create one
    if matrix_low is None:
        matrix_low = mx.zeros_like(matrix_high)
    
    # Square matrix case - use existing implementation
    q_high = mx.zeros((rows, cols), dtype=mx.float32)
    r_high = mx.zeros((cols, cols), dtype=mx.float32)
    r_low  = mx.zeros((cols, cols), dtype=mx.float32)

    for i in range(cols):
        v_high, v_low = matrix_high[:, i], matrix_low[:, i]

        for j in range(i):
            r_high[j, i] = mx.matmul(q_high[:, j].reshape(1, -1), v_high.reshape(-1, 1)).item()
            r_low[j, i]  = (
                mx.matmul(q_high[:, j].reshape(1, -1), v_low.reshape(-1, 1))
                + mx.matmul(mx.zeros_like(q_high[:, j]).reshape(1, -1), v_high.reshape(-1, 1))
                + mx.matmul(mx.zeros_like(q_high[:, j]).reshape(1, -1), v_low.reshape(-1, 1))
            ).item()

            proj_high = mx.matmul(q_high[:, j].reshape(-1, 1), mx.array(r_high[j, i]).reshape(1, 1))
            proj_low  = mx.matmul(q_high[:, j].reshape(-1, 1), mx.array(r_low[j, i]).reshape(1, 1))

            v_high, v_low = _add_double_single(v_high, v_low, -proj_high[:, 0], -proj_low[:, 0])
            matrix_high[:, i], matrix_low[:, i] = v_high, v_low

        norm_high = mx.linalg.norm(v_high)
        if norm_high < 1e-10:
            # If the column norm is too small, we'll use the standard QR decomposition
            return _standard_qr(matrix_high)

        q_high[:, i] = (v_high / norm_high).astype(mx.float32)

    return q_high, r_high

def qr(a: TensorLike,
       mode: Literal['reduced','complete','r','raw'] = 'reduced') -> Tuple[mx.array, mx.array]:
    """
    Compute the QR decomposition of a matrix using a numerically stable approach.
    
    Args:
        a: Input matrix
        mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
    
    Returns:
        Tuple of (Q, R) matrices
    
    Notes:
        This implementation uses a numerically stable approach for QR decomposition.
        For non-square matrices, it uses a specialized HPC implementation.
        For square matrices, it uses a double-single precision approach for increased stability.
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Get matrix dimensions
    m, n = a_array.shape
    
    # Use the custom QR decomposition
    q, r = _custom_qr(a_array)
    
    # Handle different modes
    if mode == 'complete' and m > n:
        # Pad Q with orthogonal vectors to make it square
        q_pad = mx.zeros((m, m - n), dtype=a_array.dtype)
        
        # Simple orthogonalization (not fully robust)
        for i in range(m - n):
            q_pad_col = mx.zeros((m,), dtype=a_array.dtype)
            q_pad_col[n + i] = 1.0
            
            # Orthogonalize against existing columns of Q
            for j in range(n):
                dot_product = mx.sum(mx.multiply(q[:, j], q_pad_col))
                q_pad_col = mx.subtract(q_pad_col, mx.multiply(dot_product, q[:, j]))
            
            # Normalize
            norm = mx.sqrt(mx.sum(mx.square(q_pad_col)))
            if mx.greater(norm, mx.array(1e-10)):
                q_pad_col = mx.divide(q_pad_col, norm)
            
            # Update q_pad
            for j in range(m):
                q_pad[j, i] = q_pad_col[j]
        
        # Concatenate Q and q_pad
        q = mx.concatenate([q, q_pad], axis=1)
        
        # Pad R with zeros
        r_pad = mx.zeros((m - n, n), dtype=a_array.dtype)
        r = mx.concatenate([r, r_pad], axis=0)
    
    if mode == 'r':
        return r, r
    elif mode == 'raw':
        # Not implemented in this simplified version
        raise ValueError("Mode 'raw' is not implemented in this simplified version")
    else:
        return q, r
        return q, r