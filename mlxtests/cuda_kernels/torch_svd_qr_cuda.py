"""
PyTorch + CUDA implementation of SVD using QR decomposition.
This is a translation of the MLX Metal implementation to PyTorch with CUDA support.
"""
from typing import Union, Tuple, Optional
import torch

def svd_with_cuda_qr(a: torch.Tensor, full_matrices: bool = True, compute_uv: bool = True, 
                    k: int = -1) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute SVD using PyTorch's operations with CUDA support.
    This implementation is based on the power iteration method.

    Args:
        a: Input matrix.
        full_matrices: If True, compute full U and Vh.
        compute_uv: If True, compute U and Vh.
        k: Number of singular values/vectors to compute (-1 for all).

    Returns:
        U, S, Vh if compute_uv is True, otherwise S.
    """
    # Convert input to float32 tensor and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_tensor = a.to(device=device, dtype=torch.float32) if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32, device=device)
    
    # Get dimensions
    m, n = a_tensor.shape
    
    # Compute rank and handle k parameter
    rank = min(m, n)
    k_val = rank if k == -1 else min(k, rank)
    
    # Determine if m >= n
    if m >= n:
        # Compute A^T A
        ata = torch.matmul(a_tensor.T, a_tensor)
        
        # Initialize Q with random orthonormal vectors using QR
        q_init = torch.randn((n, k_val), dtype=a_tensor.dtype, device=device)
        q_init, _ = torch.linalg.qr(q_init)  # Use PyTorch's built-in QR
        
        # Power iteration
        v = q_init.clone()
        max_iterations = 200 if n > 30 else 100  # More iterations for larger matrices
        for _ in range(max_iterations):
            # Matrix multiplication: v = A^T A v
            v_new = torch.matmul(ata, v)
            
            # Orthogonalize columns
            for i in range(v_new.shape[1]):
                # Orthogonalize against previous columns
                for j in range(i):
                    v_new[:, i] = torch.sub(v_new[:, i], 
                                           torch.mul(
                                               torch.sum(torch.mul(v[:, j], v_new[:, i])),
                                               v[:, j]
                                           ))
                
                # Normalize
                norm = torch.linalg.norm(v_new[:, i])
                if norm > 1e-7:  # Tolerance
                    v_new[:, i] = torch.div(v_new[:, i], norm)
                else:
                    v_new[:, i] = torch.zeros_like(v_new[:, i])
            
            # Check convergence
            diff = torch.linalg.norm(torch.sub(v, v_new))
            v = v_new.clone()
            if diff < 1e-7:
                break
        
        # Calculate eigenvalues using Rayleigh quotient
        rayleigh = torch.matmul(v.T, torch.matmul(ata, v))
        eigenvalues = torch.diag(rayleigh)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = torch.argsort(torch.neg(torch.abs(eigenvalues)))
        eigenvalues = eigenvalues[sort_indices]
        v = v[:, sort_indices]
        
        # Singular values are sqrt of eigenvalues
        # Add a small epsilon to avoid negative values due to numerical errors
        epsilon = torch.tensor(1e-10, device=device)
        eigenvalues_safe = torch.maximum(eigenvalues, epsilon)
        s = torch.sqrt(eigenvalues_safe)
        
        if not compute_uv:
            return s
        
        # Compute U = A V S^-1
        epsilon = torch.mul(torch.tensor(1e-10, device=device), torch.max(s))
        mask = torch.sub(torch.abs(s), epsilon)  # Avoid direct comparison
        s_inv = torch.where(torch.ge(mask, 0), torch.reciprocal(s), torch.zeros_like(s))
        u = torch.matmul(a_tensor, torch.mul(v, s_inv.reshape(1, -1)))
        
        # Complete basis if needed
        if full_matrices:
            if m > k_val:
                # Generate random vectors for remaining columns
                remaining_cols = m - k_val
                
                # Start with random vectors
                random_basis = torch.randn((m, remaining_cols), dtype=u.dtype, device=device)
                
                # Orthogonalize against existing vectors
                for i in torch.arange(k_val).tolist():
                    ui = u[:, i:i+1]
                    proj = torch.matmul(ui.T, random_basis)
                    random_basis = torch.sub(random_basis, torch.matmul(ui, proj))
                
                # Normalize columns
                for i in torch.arange(remaining_cols).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = torch.linalg.norm(col)
                    mask = torch.sub(torch.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = torch.where(torch.ge(mask, 0), torch.reciprocal(col_norm), torch.tensor(0.0, device=device))
                    
                    # Update column using tensor operations
                    scaled_col = torch.mul(col, scale)
                    random_basis[:, i] = scaled_col.squeeze()
                
                u = torch.cat([u, random_basis], dim=1)
            
            if n > k_val:
                # Generate random vectors for remaining columns
                remaining_cols = n - k_val
                
                # Start with random vectors
                random_basis = torch.randn((n, remaining_cols), dtype=v.dtype, device=device)
                
                # Orthogonalize against existing vectors
                for i in torch.arange(k_val).tolist():
                    vi = v[:, i:i+1]
                    proj = torch.matmul(vi.T, random_basis)
                    random_basis = torch.sub(random_basis, torch.matmul(vi, proj))
                
                # Normalize columns
                for i in torch.arange(remaining_cols).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = torch.linalg.norm(col)
                    mask = torch.sub(torch.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = torch.where(torch.ge(mask, 0), torch.reciprocal(col_norm), torch.tensor(0.0, device=device))
                    
                    # Update column using tensor operations
                    scaled_col = torch.mul(col, scale)
                    random_basis[:, i] = scaled_col.squeeze()
                
                v = torch.cat([v, random_basis], dim=1)
        
        return u, s, v.T
    
    else:  # m < n
        # Compute A A^T
        aat = torch.matmul(a_tensor, a_tensor.T)
        
        # Initialize Q with random orthonormal vectors using QR
        q_init = torch.randn((m, k_val), dtype=a_tensor.dtype, device=device)
        q_init, _ = torch.linalg.qr(q_init)  # Use PyTorch's built-in QR
        
        # Power iteration
        u = q_init.clone()
        max_iterations = 200 if m > 30 else 100  # More iterations for larger matrices
        for _ in range(max_iterations):
            # Matrix multiplication: u = A A^T u
            u_new = torch.matmul(aat, u)
            
            # Orthogonalize columns
            for i in range(u_new.shape[1]):
                # Orthogonalize against previous columns
                for j in range(i):
                    u_new[:, i] = torch.sub(u_new[:, i], 
                                           torch.mul(
                                               torch.sum(torch.mul(u[:, j], u_new[:, i])),
                                               u[:, j]
                                           ))
                
                # Normalize
                norm = torch.linalg.norm(u_new[:, i])
                if norm > 1e-7:  # Tolerance
                    u_new[:, i] = torch.div(u_new[:, i], norm)
                else:
                    u_new[:, i] = torch.zeros_like(u_new[:, i])
            
            # Check convergence
            diff = torch.linalg.norm(torch.sub(u, u_new))
            u = u_new.clone()
            if diff < 1e-7:
                break
        
        # Calculate eigenvalues using Rayleigh quotient
        rayleigh = torch.matmul(u.T, torch.matmul(aat, u))
        eigenvalues = torch.diag(rayleigh)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = torch.argsort(torch.neg(torch.abs(eigenvalues)))
        eigenvalues = eigenvalues[sort_indices]
        u = u[:, sort_indices]
        
        # Singular values are sqrt of eigenvalues
        # Add a small epsilon to avoid negative values due to numerical errors
        epsilon = torch.tensor(1e-10, device=device)
        eigenvalues_safe = torch.maximum(eigenvalues, epsilon)
        s = torch.sqrt(eigenvalues_safe)
        
        if not compute_uv:
            return s
        
        # Compute V = A^T U S^-1
        epsilon = torch.mul(torch.tensor(1e-10, device=device), torch.max(s))
        mask = torch.sub(torch.abs(s), epsilon)  # Avoid direct comparison
        s_inv = torch.where(torch.ge(mask, 0), torch.reciprocal(s), torch.zeros_like(s))
        
        # Count nonzero singular values
        nonzero_mask = torch.ge(mask, 0)
        nonzero_count = int(torch.sum(nonzero_mask).item())  # Need item() for slicing
        
        # Use the mask to select columns of u and s_inv
        u_nonzero = u[:, :nonzero_count]
        s_inv_nonzero = s_inv[:nonzero_count]
        
        v = torch.matmul(a_tensor.T, torch.mul(u_nonzero, s_inv_nonzero.reshape(1, -1)))
        
        # Complete basis if needed
        if full_matrices:
            if n > nonzero_count:
                # Generate random vectors for remaining columns
                remaining_cols = n - nonzero_count
                
                # Start with random vectors
                random_basis = torch.randn((n, remaining_cols), dtype=v.dtype, device=device)
                
                # Orthogonalize against existing vectors
                for i in torch.arange(nonzero_count).tolist():
                    vi = v[:, i:i+1]
                    proj = torch.matmul(vi.T, random_basis)
                    random_basis = torch.sub(random_basis, torch.matmul(vi, proj))
                
                # Normalize columns
                for i in torch.arange(remaining_cols).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = torch.linalg.norm(col)
                    mask = torch.sub(torch.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = torch.where(torch.ge(mask, 0), torch.reciprocal(col_norm), torch.tensor(0.0, device=device))
                    
                    # Update column using tensor operations
                    scaled_col = torch.mul(col, scale)
                    random_basis[:, i] = scaled_col.squeeze()
                
                v = torch.cat([v, random_basis], dim=1)
            
            if m > nonzero_count:
                # Generate random vectors for remaining columns
                remaining_cols = m - nonzero_count
                
                # Start with random vectors
                random_basis = torch.randn((m, remaining_cols), dtype=u.dtype, device=device)
                
                # Orthogonalize against existing vectors
                for i in torch.arange(nonzero_count).tolist():
                    ui = u[:, i:i+1]
                    proj = torch.matmul(ui.T, random_basis)
                    random_basis = torch.sub(random_basis, torch.matmul(ui, proj))
                
                # Normalize columns
                for i in torch.arange(remaining_cols).tolist():
                    col = random_basis[:, i:i+1]
                    col_norm = torch.linalg.norm(col)
                    mask = torch.sub(torch.abs(col_norm), epsilon)  # Avoid direct comparison
                    scale = torch.where(torch.ge(mask, 0), torch.reciprocal(col_norm), torch.tensor(0.0, device=device))
                    
                    # Update column using tensor operations
                    scaled_col = torch.mul(col, scale)
                    random_basis[:, i] = scaled_col.squeeze()
                
                u = torch.cat([u, random_basis], dim=1)
        
        return u, s, v.T


if __name__ == "__main__":
    print("Testing SVD with PyTorch CUDA...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    print("Input Matrix A:\n", A)

    try:
        # Use the integrated function
        U, S, Vh = svd_with_cuda_qr(A, full_matrices=True, compute_uv=True)
        print("\nResult (SVD with PyTorch CUDA):")
        print("U:\n", U)
        print("S:\n", S)
        print("Vh:\n", Vh)

        # Reconstruction Check
        k_rank = S.shape[0]
        U_rec = U[:, :k_rank]
        Vh_rec = Vh[:k_rank, :]
        A_rec = torch.matmul(torch.mul(U_rec, S.reshape(1, -1)), Vh_rec)
        print("\nReconstruction Check:")
        print("Original A:\n", A)
        print("Reconstructed A:\n", A_rec)

        # Calculate reconstruction error
        reconstruction_error = torch.sqrt(torch.mean(torch.square(torch.sub(A, A_rec))))
        max_error = torch.max(torch.abs(torch.sub(A, A_rec)))
        print("Reconstruction Error (RMSE):", reconstruction_error.item())
        print("Max Absolute Error:", max_error.item())

        # Test with different matrix shapes
        print("\nTesting with different matrix shapes...")
        
        # Test with tall-skinny matrix
        A_tall = torch.randn((60, 30), device=device)
        U_tall, S_tall, Vh_tall = svd_with_cuda_qr(A_tall)
        A_tall_rec = torch.matmul(torch.mul(U_tall[:, :30], S_tall.reshape(1, -1)), Vh_tall[:30])
        error_tall = torch.sqrt(torch.mean(torch.square(torch.sub(A_tall, A_tall_rec))))
        print(f"Tall matrix (60x30) - Reconstruction error: {error_tall.item()}")
        
        # Test with wide matrix
        A_wide = torch.randn((20, 60), device=device)
        U_wide, S_wide, Vh_wide = svd_with_cuda_qr(A_wide)
        A_wide_rec = torch.matmul(torch.mul(U_wide[:, :20], S_wide.reshape(1, -1)), Vh_wide[:20])
        error_wide = torch.sqrt(torch.mean(torch.square(torch.sub(A_wide, A_wide_rec))))
        print(f"Wide matrix (20x60) - Reconstruction error: {error_wide.item()}")
        
        # Test with square matrix
        A_square = torch.randn((60, 60), device=device)
        U_square, S_square, Vh_square = svd_with_cuda_qr(A_square)
        A_square_rec = torch.matmul(torch.mul(U_square, S_square.reshape(1, -1)), Vh_square)
        error_square = torch.sqrt(torch.mean(torch.square(torch.sub(A_square, A_square_rec))))
        print(f"Square matrix (60x60) - Reconstruction error: {error_square.item()}")
        
        # Compare with PyTorch's built-in SVD
        print("\nComparing with PyTorch's built-in SVD...")
        U_torch, S_torch, Vh_torch = torch.linalg.svd(A)
        A_torch_rec = torch.matmul(torch.mul(U_torch[:, :k_rank], S_torch[:k_rank].reshape(1, -1)), Vh_torch[:k_rank])
        error_torch = torch.sqrt(torch.mean(torch.square(torch.sub(A, A_torch_rec))))
        print(f"PyTorch SVD - Reconstruction error: {error_torch.item()}")

    except Exception as e:
        print(f"\nError during SVD with PyTorch CUDA test: {e}")
        import traceback
        traceback.print_exc()