"""
PyTorch decomposition operations for ember_ml.

This module provides PyTorch implementations of matrix decomposition operations.
"""

from typing import Optional, Union, List, Tuple, Any, Literal
import torch
from ember_ml.backend.torch.tensor import TorchDType
from ember_ml.backend.torch.types import TensorLike

def cholesky(a: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the Cholesky decomposition of a matrix.
    
    Args:
        a: Input positive-definite matrix
        
    Returns:
        Lower triangular matrix L such that a = L * L.T
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    tensor = Tensor.convert_to_tensor(a)
    return torch.linalg.cholesky(tensor)


def svd(a: TensorLike, full_matrices: bool = True, compute_uv: bool = True) -> Union[Any, Tuple[Any, Any, Any]]: # Changed torch.Tensor to Any
    """
    Compute the singular value decomposition of a matrix.
    
    Args:
        a: Input matrix
        full_matrices: If True, return full U and V matrices
        compute_uv: If True, compute U and V matrices
        
    Returns:
        If compute_uv is True, returns (U, S, V), otherwise returns S
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    tensor = Tensor.convert_to_tensor(a)
    
    if compute_uv:
        # PyTorch's svd returns (U, S, V) where V is already transposed
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=full_matrices)
        return U, S, Vh
    else:
        # Just return singular values
        return torch.linalg.svdvals(tensor)


def eig(a: TensorLike) -> Tuple[Any, Any]: # Changed torch.Tensor to Any
    """
    Compute the eigenvalues and eigenvectors of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()

    tensor = TensorInstance.convert_to_tensor(a)
    
    # PyTorch's eig is deprecated, use eigvals instead
    # But eigvals only returns eigenvalues, so we use torch.linalg.eig
    eigenvalues, eigenvectors = torch.linalg.eig(tensor)
    
    return eigenvalues, eigenvectors


def eigvals(a: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the eigenvalues of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Eigenvalues of the matrix
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()

    tensor = TensorInstance.convert_to_tensor(a)
    return torch.linalg.eigvals(tensor)


def qr(a: TensorLike, mode: Literal['reduced', 'complete', 'r', 'raw'] = 'reduced') -> Any: # Changed Tuple[torch.Tensor, torch.Tensor] to Any, as R can be returned alone
    """
    Compute the QR decomposition of a matrix.
    
    Args:
        a: Input matrix
        mode: Mode of decomposition
            'reduced': Return Q, R with shapes (M, K), (K, N) where K = min(M, N)
            'complete': Return Q, R with shapes (M, M), (M, N)
            'r': Return only R with shape (K, N) where K = min(M, N)
            'raw': Return only R with shape (M, N)
        
    Returns:
        Tuple of (Q, R) matrices
    """
     # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    TensorInstance = TorchTensor()

    tensor = TensorInstance.convert_to_tensor(a)
    
    # PyTorch's qr doesn't support 'r' or 'raw' modes directly
    # We'll implement them manually
    if mode in ['reduced', 'complete']:
        # Use PyTorch's built-in QR decomposition
        Q, R = torch.linalg.qr(tensor, mode=mode)
        return Q, R
    elif mode == 'r':
        # Return only R from reduced QR
        _, R = torch.linalg.qr(tensor, mode='reduced')
        return R
    elif mode == 'raw':
        # Return only R from complete QR
        _, R = torch.linalg.qr(tensor, mode='complete')
        return R
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected one of ['reduced', 'complete', 'r', 'raw']")


def eigh(matrix: TensorLike) -> Tuple[Any, Any]: # Changed tuple[torch.Tensor, torch.Tensor] to Tuple[Any, Any]
    """Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix.
    
    Args:
        matrix: Square matrix of shape (..., M, M) that is Hermitian/symmetric
    
    Returns:
        Tuple of:
            - eigenvalues (..., M) in ascending order
            - eigenvectors (..., M, M) where v[..., :, i] is eigenvector i
    """
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    tensor = TorchTensor()
    
    matrix = tensor.convert_to_tensor(matrix)
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    return eigenvals, eigenvecs