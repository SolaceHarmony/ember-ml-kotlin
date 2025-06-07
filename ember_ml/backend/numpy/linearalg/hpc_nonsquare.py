"""NumPy implementation for non-square orthogonal matrices."""
import numpy as np
from typing import Tuple, Optional

def orthogonalize_nonsquare(a: np.ndarray, max_block_size: int = 32) -> np.ndarray:
    """
    Create orthogonal basis for non-square matrix using NumPy.
    
    Args:
        a: Input matrix to orthogonalize
        max_block_size: Maximum block size for tiled computation
        
    Returns:
        Orthogonalized matrix
    """
    # NumPy implementation of the modified Gram-Schmidt process
    m, n = a.shape
    result = np.copy(a)
    
    # Process in blocks for better cache efficiency
    block_size = min(max_block_size, n)
    
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        
        for col in range(block_start, block_end):
            # Get current column
            curr_col = result[:, col]
            
            # First normalize the current column
            norm = np.linalg.norm(curr_col)
            if norm > 1e-10:
                curr_col = curr_col / norm
                result[:, col] = curr_col
            
            # Then orthogonalize against previous columns
            for prev_col in range(block_start, col):
                prev_vec = result[:, prev_col]
                
                # Compute dot product with extended precision
                # This is a simplified version of the extended precision calculation
                dot = np.dot(curr_col, prev_vec)
                
                # Subtract projection
                curr_col = curr_col - dot * prev_vec
                
                # Renormalize after orthogonalization
                norm = np.linalg.norm(curr_col)
                if norm > 1e-10:
                    curr_col = curr_col / norm
                
                result[:, col] = curr_col
    
    return result

def complete_orthogonal_basis_metal(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Complete an orthogonal basis for the column space of a matrix using NumPy.
    
    Args:
        a: Input matrix whose columns form a partial orthogonal basis
        eps: Tolerance for numerical zero
        
    Returns:
        Matrix with completed orthogonal basis
    """
    m, n = a.shape
    result = np.array(a)
    
    # Number of additional vectors needed
    n_additional = m - n
    
    if n_additional <= 0:
        return result
    
    # Generate random vectors for remaining columns
    additional = np.random.normal(size=(m, n_additional))
    
    # Orthogonalize new vectors against existing ones and each other
    for i in range(n_additional):
        v = additional[:, i]
        
        # Orthogonalize against all previous vectors
        for j in range(n + i):
            # Get existing orthogonal vector
            if j < n:
                q = result[:, j]
            else:
                q = additional[:, j - n]
            
            # Compute projection and subtract
            proj = np.sum(np.multiply(v, q))
            v = np.subtract(v, np.multiply(proj, q))
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm > eps:
            v = np.divide(v, norm)
        else:
            # If vector is too small, generate a new random vector
            v = np.random.normal(size=(m,))
            # Recursively orthogonalize it
            temp_basis = np.concatenate([result, additional[:, :i]], axis=1)
            completed = complete_orthogonal_basis_metal(temp_basis, eps=eps)
            v = completed[:, -1]
        
        # Update array
        additional[:, i] = v
    
    # Combine original and additional vectors
    return np.concatenate([result, additional], axis=1)