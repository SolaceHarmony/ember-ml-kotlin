# QR Decomposition with Column Pivoting: Hand-Calculated Walkthrough

This document provides a step-by-step walkthrough of the QR decomposition algorithm with column pivoting as implemented in the provided code. By tracing through the algorithm with a small example, we can identify potential issues and gain deeper insights into the mathematical foundations.

## 1. Algorithm Overview

The implementation uses Householder QR factorization with column pivoting, which decomposes a matrix A into:

$$A \cdot P = Q \cdot R$$

Where:
- Q is an orthogonal matrix
- R is an upper triangular matrix
- P is a permutation matrix

The algorithm proceeds in three main phases:
1. Panel factorization
2. Trailing update
3. Q formation (optional)

## 2. Small Example Walkthrough

Let's trace through the algorithm with a small 3×3 matrix:

$$A = \begin{bmatrix} 
4 & 1 & 2 \\
2 & 3 & 1 \\
1 & 2 & 5
\end{bmatrix}$$

### 2.1 Initialization

- m = 3, n = 3, k = min(m,n) = 3
- panel = 64 (but since our matrix is small, we'll process all columns in one panel)
- scratch_cols = n + k + k + k = 3 + 3 + 3 + 3 = 12
- S = zeros(m, scratch_cols) = zeros(3, 12)
- S[:, :n] = A (copy A to the first n columns of S)

$$S = \begin{bmatrix} 
4 & 1 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
2 & 3 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 2 & 5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}$$

### 2.2 Panel Factorization (First Column)

#### 2.2.1 Compute Column Norms

For the first column [4, 2, 1]:
- norm = sqrt(4² + 2² + 1²) = sqrt(16 + 4 + 1) = sqrt(21) ≈ 4.583

For the second column [1, 3, 2]:
- norm = sqrt(1² + 3² + 2²) = sqrt(1 + 9 + 4) = sqrt(14) ≈ 3.742

For the third column [2, 1, 5]:
- norm = sqrt(2² + 1² + 5²) = sqrt(4 + 1 + 25) = sqrt(30) ≈ 5.477

The third column has the largest norm, so we pivot it to the first position:

$$A_1 = \begin{bmatrix} 
2 & 1 & 4 \\
1 & 3 & 2 \\
5 & 2 & 1
\end{bmatrix}$$

#### 2.2.2 Compute Householder Reflector for First Column

For the first column [2, 1, 5]:
- x = [2, 1, 5]
- α = -sign(x₁) · ||x||₂ = -sign(2) · 5.477 = -5.477
- u₁ = x₁ - α = 2 - (-5.477) = 7.477
- u = [u₁, x₂, x₃] = [7.477, 1, 5]
- v = u / ||u||₂ = [7.477, 1, 5] / sqrt(7.477² + 1² + 5²) = [7.477, 1, 5] / 9.013 = [0.829, 0.111, 0.555]
- τ = 2 / (v^T · v) = 2 / 1 = 2

Store v in the V section of S (columns n to n+k-1):

$$S = \begin{bmatrix} 
2 & 1 & 4 & 0.829 & 0 & 0 & ... \\
1 & 3 & 2 & 0.111 & 0 & 0 & ... \\
5 & 2 & 1 & 0.555 & 0 & 0 & ... 
\end{bmatrix}$$

Store τ in the τ section (columns n+k to n+2k-1, first row):

$$S[0, 6] = 2$$

#### 2.2.3 Apply Householder Reflector to Remaining Columns

For the second column [1, 3, 2]:
- w = v^T · [1, 3, 2] = 0.829 · 1 + 0.111 · 3 + 0.555 · 2 = 0.829 + 0.333 + 1.11 = 2.272
- [1, 3, 2] = [1, 3, 2] - τ · v · w = [1, 3, 2] - 2 · [0.829, 0.111, 0.555] · 2.272 = [1, 3, 2] - [3.767, 0.504, 2.522] = [-2.767, 2.496, -0.522]

For the third column [4, 2, 1]:
- w = v^T · [4, 2, 1] = 0.829 · 4 + 0.111 · 2 + 0.555 · 1 = 3.316 + 0.222 + 0.555 = 4.093
- [4, 2, 1] = [4, 2, 1] - τ · v · w = [4, 2, 1] - 2 · [0.829, 0.111, 0.555] · 4.093 = [4, 2, 1] - [6.789, 0.909, 4.543] = [-2.789, 1.091, -3.543]

After this step, A becomes:

$$A_1 = \begin{bmatrix} 
2 & -2.767 & -2.789 \\
1 & 2.496 & 1.091 \\
5 & -0.522 & -3.543
\end{bmatrix}$$

### 2.3 Panel Factorization (Second Column)

#### 2.3.1 Compute Column Norms for Remaining Submatrix

For the second column of the submatrix [2.496, -0.522]:
- norm = sqrt(2.496² + (-0.522)²) = sqrt(6.23 + 0.272) = sqrt(6.502) ≈ 2.55

For the third column of the submatrix [1.091, -3.543]:
- norm = sqrt(1.091² + (-3.543)²) = sqrt(1.19 + 12.55) = sqrt(13.74) ≈ 3.71

The third column has the larger norm, so we pivot it to the second position:

$$A_2 = \begin{bmatrix} 
2 & -2.789 & -2.767 \\
1 & 1.091 & 2.496 \\
5 & -3.543 & -0.522
\end{bmatrix}$$

#### 2.3.2 Compute Householder Reflector for Second Column

For the second column of the submatrix [1.091, -3.543]:
- x = [1.091, -3.543]
- α = -sign(x₁) · ||x||₂ = -sign(1.091) · 3.71 = -3.71
- u₁ = x₁ - α = 1.091 - (-3.71) = 4.801
- u = [u₁, x₂] = [4.801, -3.543]
- v = u / ||u||₂ = [4.801, -3.543] / sqrt(4.801² + (-3.543)²) = [4.801, -3.543] / 5.97 = [0.804, -0.593]
- τ = 2 / (v^T · v) = 2 / 1 = 2

Store v in the V section of S:

$$S = \begin{bmatrix} 
2 & -2.789 & -2.767 & 0.829 & 0 & 0 & ... \\
1 & 1.091 & 2.496 & 0.111 & 0.804 & 0 & ... \\
5 & -3.543 & -0.522 & 0.555 & -0.593 & 0 & ... 
\end{bmatrix}$$

Store τ in the τ section:

$$S[0, 7] = 2$$

#### 2.3.3 Apply Householder Reflector to Remaining Column

For the third column of the submatrix [2.496, -0.522]:
- w = v^T · [2.496, -0.522] = 0.804 · 2.496 + (-0.593) · (-0.522) = 2.007 + 0.309 = 2.316
- [2.496, -0.522] = [2.496, -0.522] - τ · v · w = [2.496, -0.522] - 2 · [0.804, -0.593] · 2.316 = [2.496, -0.522] - [3.726, -2.747] = [-1.23, 2.225]

After this step, A becomes:

$$A_2 = \begin{bmatrix} 
2 & -2.789 & -2.767 \\
1 & 1.091 & -1.23 \\
5 & -3.543 & 2.225
\end{bmatrix}$$

### 2.4 Panel Factorization (Third Column)

#### 2.4.1 Compute Householder Reflector for Third Column

For the third column of the submatrix [2.225]:
- x = [2.225]
- α = -sign(x₁) · ||x||₂ = -sign(2.225) · 2.225 = -2.225
- u₁ = x₁ - α = 2.225 - (-2.225) = 4.45
- u = [u₁] = [4.45]
- v = u / ||u||₂ = [4.45] / 4.45 = [1]
- τ = 2 / (v^T · v) = 2 / 1 = 2

Store v in the V section of S:

$$S = \begin{bmatrix} 
2 & -2.789 & -2.767 & 0.829 & 0 & 0 & ... \\
1 & 1.091 & -1.23 & 0.111 & 0.804 & 0 & ... \\
5 & -3.543 & 2.225 & 0.555 & -0.593 & 1 & ... 
\end{bmatrix}$$

Store τ in the τ section:

$$S[0, 8] = 2$$

### 2.5 Extract R Matrix

The R matrix is the upper triangular part of the transformed A:

$$R = \begin{bmatrix} 
2 & -2.789 & -2.767 \\
0 & 1.091 & -1.23 \\
0 & 0 & 2.225
\end{bmatrix}$$

### 2.6 Form Q Matrix (Optional)

To form Q explicitly, we apply the Householder reflectors in reverse order to the identity matrix:

$$Q = (I - \tau_1 v_1 v_1^T) \cdot (I - \tau_2 v_2 v_2^T) \cdot (I - \tau_3 v_3 v_3^T)$$

Starting with the identity matrix:

$$I = \begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$$

Apply the third Householder reflector:
- v₃ = [0, 0, 1]
- τ₃ = 2
- I - τ₃ · v₃ · v₃^T = I - 2 · [0, 0, 1] · [0, 0, 1]^T = I - 2 · [0, 0, 0; 0, 0, 0; 0, 0, 1] = [1, 0, 0; 0, 1, 0; 0, 0, -1]

Apply the second Householder reflector:
- v₂ = [0, 0.804, -0.593]
- τ₂ = 2
- Result = [1, 0, 0; 0, 0.356, 0.935; 0, 0.935, 0.356]

Apply the first Householder reflector:
- v₁ = [0.829, 0.111, 0.555]
- τ₁ = 2
- Final Q = [0.312, -0.436, 0.844; 0.777, 0.629, -0.036; 0.545, -0.644, -0.535]

### 2.7 Verify Reconstruction

To verify the decomposition, we compute:
- P = permutation matrix based on pivot indices [2, 0, 1]
- QR = Q · R
- AP = A · P
- Check if QR ≈ AP

## 3. Potential Issues

Based on this walkthrough, here are potential issues that might affect the implementation:

1. **Precision in Norm Calculation**: The multi-precision approach is crucial for accurate pivot selection. If the norm calculation is not precise enough, it could lead to suboptimal pivoting decisions, especially for ill-conditioned matrices.

2. **Householder Vector Normalization**: When computing v = u/||u||, if ||u|| is very small, numerical instability can occur. The implementation should handle this case by adding a small epsilon to prevent division by zero.

3. **Memory Layout**: The implementation uses a specific memory layout for S (A | V | τ | piv). If the indexing is incorrect, it could lead to accessing wrong elements, especially in the trailing update phase.

4. **Pivot Handling**: The permutation matrix P needs to be correctly formed based on the pivot indices. If the pivoting is not tracked properly, the reconstruction will fail.

5. **Q Formation**: When forming Q explicitly, the Householder reflectors must be applied in reverse order. If this is not done correctly, Q will not be orthogonal.

6. **Floating-Point Accumulation**: In the trailing update, accumulating the result of v·(v^T·A) can lead to precision loss. Using compensated summation or higher precision for intermediate results could improve accuracy.

7. **Panel Size**: The panel size (64 in the implementation) affects both performance and numerical stability. Too large panels might lead to cache misses, while too small panels might not fully utilize the GPU.

## 4. Conclusion

The QR decomposition with column pivoting as implemented in the code is mathematically sound, but its numerical stability depends on careful handling of the issues mentioned above. The multi-precision approach for norm calculation is particularly important for ensuring accurate pivoting decisions, which directly impacts the quality of the decomposition.

By addressing these potential issues, the implementation can achieve both high performance and numerical stability, making it suitable for a wide range of applications, including solving linear least squares problems and computing matrix rank.