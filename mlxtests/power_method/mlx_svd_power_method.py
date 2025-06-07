"""
Implementation of SVD using the power method with pure MLX.

This module implements the truncated SVD algorithm described in the paper
"Distributed Out-of-Memory SVD on CPU/GPU Architectures" using direct MLX
calls to avoid any dependency on the Ember ML abstraction layer.
"""

from typing import Tuple
import mlx.core as mx

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
    # Note: Using mx.array with random values for initialization
    x = mx.array(mx.random.normal(shape=(k, 1)))

    # 4: x = x / ||x|| . Normalize x
    x = x / mx.linalg.norm(x)

    # 5: set v0 = x
    v0 = x

    # 6: if m > n then
    if m > n:
        # 7: B = XT @ X . @ is matrix multiplication operator
        B = mx.matmul(mx.transpose(X), X)
    # 8: else
    else:
        # 9: B = X @ XT
        B = mx.matmul(X, mx.transpose(X))

    # 10: while true do
    while True:
        # 11: v1 = B @ v0
        v1 = mx.matmul(B, v0)

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
        dot_product = mx.squeeze(mx.matmul(mx.transpose(v0), v1))
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
    # Initialize U, V, sigma as lists to collect vectors
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
            reconstructed_part = mx.matmul(mx.matmul(U_l, Sigma_l_diag), mx.transpose(V_l))

            # Subtract from the current X to get the residual
            X = X - reconstructed_part

        # 9: if m > n then
        if m > n:
            # 10: V(l) = SVD 1D(X, epsilon)
            v_l = svd_1d(X, epsilon)
            V_list.append(v_l)

            # 11: U(l) = A @ V(l) . @ stands for matrix multiplication operation
            # Note: The paper uses A here, not X (the residual). This is crucial.
            u_l = mx.matmul(A, v_l)

            # 12: sigma(l) = ||U(l)||
            sigma_l = float(mx.linalg.norm(u_l).item())  # Convert to Python float

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
            v_l = mx.matmul(mx.transpose(A), u_l)

            # 17: sigma(l) = ||V(l)||
            sigma_l = float(mx.linalg.norm(v_l).item())  # Convert to Python float

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
    print("Running basic SVD tests with pure MLX...")

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
    A1_reconstructed = mx.matmul(mx.matmul(U1, Sigma1_diag), mx.transpose(V1))
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
    A2_reconstructed = mx.matmul(mx.matmul(U2, Sigma2_diag), mx.transpose(V2))
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
    A3_reconstructed = mx.matmul(mx.matmul(U3, Sigma3_diag), mx.transpose(V3))
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
    A4_reconstructed = mx.matmul(mx.matmul(U4, Sigma4_diag), mx.transpose(V4))
    print("Reconstructed A (truncated):")
    print(A4_reconstructed)
    # Note: For truncated SVD, the reconstructed matrix will be an approximation

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
    A5_reconstructed = mx.matmul(mx.matmul(U5, Sigma5_diag), mx.transpose(V5))
    print("Reconstructed A:")
    print(A5_reconstructed)
    print("Is A5_reconstructed close to A5?", mx.allclose(A5_reconstructed, A5))

    # Compare with MLX native SVD for a simple case
    print("\nComparing with MLX native SVD for A1:")
    # Use CPU stream for SVD since it's not supported on GPU
    with mx.stream(mx.cpu):
        U1_native, sigma1_native, VT1_native = mx.linalg.svd(A1)
        print("MLX Native U:")
        print(U1_native)
        print("MLX Native Sigma:")
        print(sigma1_native)
        print("MLX Native VT (transpose of V):")
        print(VT1_native)  # MLX returns VT (V transpose)

        # Note: Singular vectors are unique up to sign. We need to check if the columns
        # of our U and V are the same as the native implementation's up to a sign flip.
        print("\nComparing our results with MLX native SVD (up to sign and order):")
        # Simple check for singular values (should be positive and in decreasing order)
        print("Are our singular values close to MLX's?", mx.allclose(sigma1, sigma1_native))