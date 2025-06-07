import mlx.core as mx
from typing import Tuple

def svd_1d(X: mx.array, epsilon: float) -> mx.array:
    """
    Algorithm 2: SVD 1D(X, epsilon)
    Estimates the first singular vector of matrix X using the power method.

    Args:
        X: Input matrix (m x n).
        epsilon: Convergence tolerance.

    Returns:
        The estimated first singular vector (size min(m, n)).
    """
    m, n = X.shape
    k = min(m, n)

    # 3: x ≈ N(0, 1) where x, 0, 1 ∈ Rk+ . Sample x from a multivariate normal
    # distribution of mean 0 and variance 1
    # Note: MLX random normal generates with mean 0 and variance 1 by default
    x = mx.random.normal(shape=(k, 1))

    # 4: x = x / ||x|| . Normalize x
    x = x / mx.linalg.norm(x)

    # 5: set v0 = x
    v0 = x

    # 6: if m > n then
    if m > n:
        # 7: B = XT @ X . @ is matrix multiplication operator
        B = X.transpose() @ X
    # 8: else
    else:
        # 9: B = X @ XT
        B = X @ X.transpose()

    # 10: while true do
    while True:
        # 11: v1 = B @ v0
        v1 = B @ v0

        # 12: v1 = v1 / ||v1|| . ||.|| represents l2 norm
        norm_v1 = mx.linalg.norm(v1)
        # Avoid division by zero if v1 is a zero vector
        if norm_v1 < 1e-10:
             # If v1 is zero, break or handle appropriately.
             # For power method, this might indicate issues or convergence to zero vector.
             # For now, let's break.
             break
        v1 = v1 / norm_v1

        # 13: if |v0 @ v1| >= 1 - epsilon then . |.| is modulus operator
        # Note: Using mx.abs and dot product for |v0 @ v1|
        dot_product = (v0.transpose() @ v1).squeeze()
        if mx.abs(dot_product) >= 1.0 - epsilon:
            # 14: return v1
            return v1
        # 15: v0 = v1
        v0 = v1

    # Return the last computed v1 if loop breaks without convergence (e.g., zero vector)
    return v1


def svd_truncated(A: mx.array, epsilon: float, k: int = -1) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Algorithm 1: SVD(A, epsilon, k = -1) - Truncated SVD
    Computes the truncated Singular Value Decomposition of matrix A.

    Args:
        A: Input matrix (m x n).
        epsilon: Convergence tolerance for SVD 1D.
        k: Number of singular values/vectors to compute. If -1, compute min(m, n).

    Returns:
        A tuple containing:
        - U: Left singular vectors (m x k).
        - Sigma: Singular values (k,).
        - V: Right singular vectors (n x k).
    """
    m, n = A.shape

    # 2: if k == -1 then
    if k == -1:
        # 3: k = min(m, n)
        k = min(m, n)

    # 4: U, V, sigma = [], [], [] . Initialize as empty arrays
    # Initialize U, V, sigma as MLX arrays
    U_list = []
    V_list = []
    sigma_list = []

    # Create a copy of A to work with the residual
    X = mx.array(A)

    # 5: for l in [1, k] do
    for l in range(k):
        # 6: X = A (This line is effectively handled by the residual update in line 8)

        # 7: if l > 1 then
        if l > 0:
            # 8: X = X - U[: l] diag(sigma[: l]) V[: l]T
            # Reconstruct the part of A already decomposed
            U_l = mx.concatenate(U_list[:l], axis=1)
            V_l = mx.concatenate(V_list[:l], axis=1)
            sigma_l = mx.array(sigma_list[:l])

            # Create a diagonal matrix from sigma_l
            Sigma_l_diag = mx.diag(sigma_l)

            # Compute the part to subtract: U_l @ Sigma_l_diag @ V_l.transpose()
            # Note: MLX matmul handles this directly
            reconstructed_part = U_l @ Sigma_l_diag @ V_l.transpose()

            # Subtract from the original matrix A to get the residual
            # The paper's Algorithm 1 line 8 subtracts from X, which is initially A.
            # Subsequent iterations subtract from the residual.
            # We need to subtract from the original A to get the correct residual for the next iteration.
            # Let's re-read Algorithm 1 carefully. Line 6 says X = A. Line 8 says X = X - ...
            # This implies X is the residual. So we should subtract from the current X.
            X = X - reconstructed_part


        # 9: if m > n then
        if m > n:
            # 10: V(l) = SVD 1D(X, epsilon)
            v_l = svd_1d(X, epsilon)
            V_list.append(v_l)

            # 11: U(l) = A @ V(l) . @ stands for matrix multiplication operation
            # Note: The paper uses A here, not X (the residual). This is crucial.
            u_l = A @ v_l

            # 12: sigma(l) = ||U(l)||
            sigma_l = mx.linalg.norm(u_l).item() # Use .item() to get a scalar

            # 13: U(l) = U(l) / sigma(l)
            # Avoid division by zero if sigma_l is zero
            if sigma_l < 1e-10:
                 # If sigma is zero, the singular vector is zero.
                 u_l = mx.zeros_like(u_l)
            else:
                u_l = u_l / sigma_l
            U_list.append(u_l)
            sigma_list.append(sigma_l)

        # 14: else (m <= n)
        else:
            # 15: U(l) = SVD 1D(X, epsilon)
            u_l = svd_1d(X, epsilon)
            U_list.append(u_l)

            # 16: V(l) = AT @ U(l)
            # Note: The paper uses AT here, not XT. This is crucial.
            v_l = A.transpose() @ u_l

            # 17: sigma(l) = ||V(l)||
            sigma_l = mx.linalg.norm(v_l).item() # Use .item() to get a scalar

            # 18: V(l) = V(l) / sigma(l)
            # Avoid division by zero if sigma_l is zero
            if sigma_l < 1e-10:
                 # If sigma is zero, the singular vector is zero.
                 v_l = mx.zeros_like(v_l)
            else:
                v_l = v_l / sigma_l
            V_list.append(v_l)
            sigma_list.append(sigma_l)

    # Ensure: U ∈ Rm×k, σ ∈ Rk, V ∈ Rn×k
    # Ensure: UUT = Im×m, V V T = In×n where I is Identity matrix and diag(Σ) = σ where σ = {σ1, σ2, ...σk} and Σ ∈ Rk×k

    # Concatenate the lists of vectors into matrices
    U_matrix = mx.concatenate(U_list, axis=1) if U_list else mx.array([])
    V_matrix = mx.concatenate(V_list, axis=1) if V_list else mx.array([])
    sigma_array = mx.array(sigma_list)

    return U_matrix, sigma_array, V_matrix

# Basic Test Cases
if __name__ == "__main__":
    print("Running basic SVD tests...")

    # Test Case 1: Simple 2x2 matrix
    A1 = mx.array([[1.0, 2.0], [3.0, 4.0]])
    epsilon1 = 1e-6
    k1 = 2
    U1, sigma1, V1 = svd_truncated(A1, epsilon1, k1)
    print("\nTest Case 1:")
    print("Input Matrix A:")
    print(A1)
    print("U:")
    print(U1)
    print("Sigma:")
    print(sigma1)
    print("V:")
    print(V1)
    # Verification: A ≈ U @ diag(sigma) @ V.transpose()
    Sigma1_diag = mx.diag(sigma1)
    A1_reconstructed = U1 @ Sigma1_diag @ V1.transpose()
    print("Reconstructed A:")
    print(A1_reconstructed)
    print("Is A1_reconstructed close to A1?", mx.allclose(A1_reconstructed, A1))

    # Test Case 2: Non-square matrix (3x2)
    A2 = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    epsilon2 = 1e-6
    k2 = 2
    U2, sigma2, V2 = svd_truncated(A2, epsilon2, k2)
    print("\nTest Case 2:")
    print("Input Matrix A:")
    print(A2)
    print("U:")
    print(U2)
    print("Sigma:")
    print(sigma2)
    print("V:")
    print(V2)
    # Verification: A ≈ U @ diag(sigma) @ V.transpose()
    Sigma2_diag = mx.diag(sigma2)
    A2_reconstructed = U2 @ Sigma2_diag @ V2.transpose()
    print("Reconstructed A:")
    print(A2_reconstructed)
    print("Is A2_reconstructed close to A2?", mx.allclose(A2_reconstructed, A2))

    # Test Case 3: Non-square matrix (2x3)
    A3 = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    epsilon3 = 1e-6
    k3 = 2
    U3, sigma3, V3 = svd_truncated(A3, epsilon3, k3)
    print("\nTest Case 3:")
    print("Input Matrix A:")
    print(A3)
    print("U:")
    print(U3)
    print("Sigma:")
    print(sigma3)
    print("V:")
    print(V3)
    # Verification: A ≈ U @ diag(sigma) @ V.transpose()
    Sigma3_diag = mx.diag(sigma3)
    A3_reconstructed = U3 @ Sigma3_diag @ V3.transpose()
    print("Reconstructed A:")
    print(A3_reconstructed)
    print("Is A3_reconstructed close to A3?", mx.allclose(A3_reconstructed, A3))

    # Test Case 4: Larger matrix with k < min(m, n)
    A4 = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    epsilon4 = 1e-6
    k4 = 2
    U4, sigma4, V4 = svd_truncated(A4, epsilon4, k4)
    print("\nTest Case 4:")
    print("Input Matrix A:")
    print(A4)
    print("U (truncated):")
    print(U4)
    print("Sigma (truncated):")
    print(sigma4)
    print("V (truncated):")
    print(V4)
    # Verification: A ≈ U @ diag(sigma) @ V.transpose() (truncated)
    Sigma4_diag = mx.diag(sigma4)
    A4_reconstructed = U4 @ Sigma4_diag @ V4.transpose()
    print("Reconstructed A (truncated):")
    print(A4_reconstructed)
    # Note: For truncated SVD, the reconstructed matrix will be an approximation
    # We can compare the singular values and vectors to a known implementation if available,
    # or check if the reconstruction is "close enough" for the dominant components.
    # For now, let's just print the reconstruction.
    # A more rigorous test would compare with mx.linalg.svd results for the top k components.

    # Test Case 5: Test with k = -1 (compute full SVD)
    A5 = mx.array([[1.0, 2.0], [3.0, 4.0]])
    epsilon5 = 1e-6
    k5 = -1
    U5, sigma5, V5 = svd_truncated(A5, epsilon5, k5)
    print("\nTest Case 5 (k=-1):")
    print("Input Matrix A:")
    print(A5)
    print("U:")
    print(U5)
    print("Sigma:")
    print(sigma5)
    print("V:")
    print(V5)
    # Verification: A ≈ U @ diag(sigma) @ V.transpose()
    Sigma5_diag = mx.diag(sigma5)
    A5_reconstructed = U5 @ Sigma5_diag @ V5.transpose()
    print("Reconstructed A:")
    print(A5_reconstructed)
    print("Is A5_reconstructed close to A5?", mx.allclose(A5_reconstructed, A5))

    # Compare with MLX native SVD for a simple case
    print("\nComparing with MLX native SVD for A1:")
    U1_mlx, sigma1_mlx, V1_mlx = mx.linalg.svd(A1,stream=mx.cpu)
    print("MLX Native U:")
    print(U1_mlx)
    print("MLX Native Sigma:")
    print(sigma1_mlx)
    print("MLX Native Vh (transpose of V):")
    print(V1_mlx) # MLX returns Vh (V transpose)

    # Note: Singular vectors are unique up to sign. We need to check if the columns
    # of our U and V are the same as MLX's up to a sign flip.
    # Also, MLX returns Vh (V transpose), so we compare our V with MLX's Vh.transpose()
    print("\nComparing our results with MLX native SVD (up to sign and order):")
    # Simple check for singular values (should be positive and in decreasing order)
    print("Are our singular values close to MLX's?", mx.allclose(sigma1, sigma1_mlx))
    # More rigorous comparison of vectors would involve checking column by column
    # and allowing for sign flips and potentially different ordering (though power method
    # usually finds the largest first). This is more complex and might be added later.