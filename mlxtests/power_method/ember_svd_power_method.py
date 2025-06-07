"""
Implementation of SVD using the power method with Ember ML ops abstraction.

This module implements the truncated SVD algorithm described in the paper
"Distributed Out-of-Memory SVD on CPU/GPU Architectures" using Ember ML's
ops abstraction layer instead of direct MLX calls.
"""

from typing import Tuple
from ember_ml import ops
from ember_ml.nn import tensor

def svd_1d(X: tensor.EmberTensor, epsilon: float) -> tensor.EmberTensor:
    """
    Algorithm 2: SVD 1D(X, epsilon)
    Estimates the first singular vector of matrix X using the power method.

    Args:
        X: Input matrix (m x n).
        epsilon: Convergence tolerance.

    Returns:
        The estimated first singular vector (size min(m, n)).
    """
    m, n = tensor.shape(X)
    k = min(m, n)

    # 3: x ≈ N(0, 1) where x, 0, 1 ∈ Rk+ . Sample x from a multivariate normal
    # distribution of mean 0 and variance 1
    # Note: Using tensor.random_normal for backend-agnostic random generation
    x = tensor.random_normal((k, 1))

    # 4: x = x / ||x|| . Normalize x
    x = ops.divide(x, ops.norm(x))

    # 5: set v0 = x
    v0 = x

    # 6: if m > n then
    if m > n:
        # 7: B = XT @ X . @ is matrix multiplication operator
        B = ops.matmul(ops.transpose(X), X)
    # 8: else
    else:
        # 9: B = X @ XT
        B = ops.matmul(X, ops.transpose(X))

    # 10: while true do
    while True:
        # 11: v1 = B @ v0
        v1 = ops.matmul(B, v0)

        # 12: v1 = v1 / ||v1|| . ||.|| represents l2 norm
        norm_v1 = ops.norm(v1)
        # Avoid division by zero if v1 is a zero vector
        if ops.less(norm_v1, tensor.convert_to_tensor(1e-10)):
             # If v1 is zero, break or handle appropriately.
             # For power method, this might indicate issues or convergence to zero vector.
             # For now, let's break.
             break
        v1 = ops.divide(v1, norm_v1)

        # 13: if |v0 @ v1| >= 1 - epsilon then . |.| is modulus operator
        # Note: Using ops.abs and dot product for |v0 @ v1|
        dot_product = ops.squeeze(ops.matmul(ops.transpose(v0), v1))
        if ops.greater_equal(ops.abs(dot_product), ops.subtract(tensor.convert_to_tensor(1.0), tensor.convert_to_tensor(epsilon))):
            # 14: return v1
            return v1
        # 15: v0 = v1
        v0 = v1

    # Return the last computed v1 if loop breaks without convergence (e.g., zero vector)
    return v1


def svd_truncated(A: tensor.EmberTensor, epsilon: float, k: int = -1) -> Tuple[tensor.EmberTensor, tensor.EmberTensor, tensor.EmberTensor]:
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
    m, n = tensor.shape(A)

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
    X = tensor.copy(A)

    # 5: for l in [1, k] do
    for l in range(k):
        # 6: X = A (This line is effectively handled by the residual update in line 8)

        # 7: if l > 1 then
        if l > 0:
            # 8: X = X - U[: l] diag(sigma[: l]) V[: l]T
            # Reconstruct the part of A already decomposed
            U_l = ops.concatenate(U_list[:l], axis=1)
            V_l = ops.concatenate(V_list[:l], axis=1)
            sigma_l = tensor.convert_to_tensor(sigma_list[:l])

            # Create a diagonal matrix from sigma_l
            Sigma_l_diag = ops.diag(sigma_l)

            # Compute the part to subtract: U_l @ Sigma_l_diag @ V_l.transpose()
            reconstructed_part = ops.matmul(ops.matmul(U_l, Sigma_l_diag), ops.transpose(V_l))

            # Subtract from the current X to get the residual
            X = ops.subtract(X, reconstructed_part)

        # 9: if m > n then
        if m > n:
            # 10: V(l) = SVD 1D(X, epsilon)
            v_l = svd_1d(X, epsilon)
            V_list.append(v_l)

            # 11: U(l) = A @ V(l) . @ stands for matrix multiplication operation
            # Note: The paper uses A here, not X (the residual). This is crucial.
            u_l = ops.matmul(A, v_l)

            # 12: sigma(l) = ||U(l)||
            sigma_l = ops.norm(u_l).item() # Use .item() to get a scalar

            # 13: U(l) = U(l) / sigma(l)
            # Avoid division by zero if sigma_l is zero
            if sigma_l < 1e-10:
                 # If sigma is zero, the singular vector is zero.
                 u_l = tensor.zeros_like(u_l)
            else:
                u_l = ops.divide(u_l, tensor.convert_to_tensor(sigma_l))
            U_list.append(u_l)
            sigma_list.append(sigma_l)

        # 14: else (m <= n)
        else:
            # 15: U(l) = SVD 1D(X, epsilon)
            u_l = svd_1d(X, epsilon)
            U_list.append(u_l)

            # 16: V(l) = AT @ U(l)
            # Note: The paper uses AT here, not XT. This is crucial.
            v_l = ops.matmul(ops.transpose(A), u_l)

            # 17: sigma(l) = ||V(l)||
            sigma_l = ops.norm(v_l).item() # Use .item() to get a scalar

            # 18: V(l) = V(l) / sigma(l)
            # Avoid division by zero if sigma_l is zero
            if sigma_l < 1e-10:
                 # If sigma is zero, the singular vector is zero.
                 v_l = tensor.zeros_like(v_l)
            else:
                v_l = ops.divide(v_l, tensor.convert_to_tensor(sigma_l))
            V_list.append(v_l)
            sigma_list.append(sigma_l)

    # Ensure: U ∈ Rm×k, σ ∈ Rk, V ∈ Rn×k
    # Ensure: UUT = Im×m, V V T = In×n where I is Identity matrix and diag(Σ) = σ where σ = {σ1, σ2, ...σk} and Σ ∈ Rk×k

    # Concatenate the lists of vectors into matrices
    U_matrix = ops.concatenate(U_list, axis=1) if U_list else tensor.convert_to_tensor([])
    V_matrix = ops.concatenate(V_list, axis=1) if V_list else tensor.convert_to_tensor([])
    sigma_array = tensor.convert_to_tensor(sigma_list)

    return U_matrix, sigma_array, V_matrix

# Basic Test Cases
if __name__ == "__main__":
    print("Running basic SVD tests with Ember ML ops...")

    # Test Case 1: Simple 2x2 matrix
    A1 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
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
    Sigma1_diag = ops.diag(sigma1)
    A1_reconstructed = ops.matmul(ops.matmul(U1, Sigma1_diag), ops.transpose(V1))
    print("Reconstructed A:")
    print(A1_reconstructed)
    print("Is A1_reconstructed close to A1?", ops.allclose(A1_reconstructed, A1))

    # Test Case 2: Non-square matrix (3x2)
    A2 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
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
    Sigma2_diag = ops.diag(sigma2)
    A2_reconstructed = ops.matmul(ops.matmul(U2, Sigma2_diag), ops.transpose(V2))
    print("Reconstructed A:")
    print(A2_reconstructed)
    print("Is A2_reconstructed close to A2?", ops.allclose(A2_reconstructed, A2))

    # Test Case 3: Non-square matrix (2x3)
    A3 = tensor.convert_to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
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
    Sigma3_diag = ops.diag(sigma3)
    A3_reconstructed = ops.matmul(ops.matmul(U3, Sigma3_diag), ops.transpose(V3))
    print("Reconstructed A:")
    print(A3_reconstructed)
    print("Is A3_reconstructed close to A3?", ops.allclose(A3_reconstructed, A3))

    # Test Case 4: Larger matrix with k < min(m, n)
    A4 = tensor.convert_to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
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
    Sigma4_diag = ops.diag(sigma4)
    A4_reconstructed = ops.matmul(ops.matmul(U4, Sigma4_diag), ops.transpose(V4))
    print("Reconstructed A (truncated):")
    print(A4_reconstructed)
    # Note: For truncated SVD, the reconstructed matrix will be an approximation

    # Test Case 5: Test with k = -1 (compute full SVD)
    A5 = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
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
    Sigma5_diag = ops.diag(sigma5)
    A5_reconstructed = ops.matmul(ops.matmul(U5, Sigma5_diag), ops.transpose(V5))
    print("Reconstructed A:")
    print(A5_reconstructed)
    print("Is A5_reconstructed close to A5?", ops.allclose(A5_reconstructed, A5))

    # Compare with native SVD for a simple case
    print("\nComparing with native SVD for A1:")
    U1_native, sigma1_native, V1_native = ops.svd(A1)
    print("Native SVD U:")
    print(U1_native)
    print("Native SVD Sigma:")
    print(sigma1_native)
    print("Native SVD V (or Vh):")
    print(V1_native)

    # Note: Singular vectors are unique up to sign. We need to check if the columns
    # of our U and V are the same as the native implementation's up to a sign flip.
    print("\nComparing our results with native SVD (up to sign and order):")
    # Simple check for singular values (should be positive and in decreasing order)
    print("Are our singular values close to native SVD's?", ops.allclose(sigma1, sigma1_native))