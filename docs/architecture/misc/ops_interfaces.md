# Operations Interfaces

This section documents the abstract interfaces for various operations within Ember ML. These interfaces define the contract that backend implementations must adhere to, ensuring backend-agnostic usage in the frontend.

## Core Concepts

Interfaces in Ember ML serve as blueprints for functionality. They specify the methods and properties that concrete implementations (typically within the backend directories) must provide. This allows frontend code to interact with operations through a consistent API without needing to know the specific backend being used.

## Components

### Linear Algebra Operations (`ember_ml.ops.linearalg.linearalg_ops`)

*   **`LinearAlgOps(ABC)`**: The abstract base class for linear algebra operations.
    *   Includes abstract methods for:
        *   `solve(a, b)`: Solve a linear system `Ax = b`.
        *   `inv(a)`: Compute the inverse of a matrix.
        *   `det(a)`: Compute the determinant of a matrix.
        *   `norm(x, ord, axis, keepdims)`: Compute matrix or vector norm.
        *   `qr(a, mode)`: Compute QR decomposition.
        *   `svd(a, full_matrices, compute_uv)`: Compute Singular Value Decomposition.
        *   `cholesky(a)`: Compute Cholesky decomposition.
        *   `lstsq(a, b, rcond)`: Compute least-squares solution.
        *   `eig(a)`: Compute eigenvalues and eigenvectors.
        *   `diag(x, k)`: Extract a diagonal or construct a diagonal matrix.
        *   `diagonal(x, offset, axis1, axis2)`: Return specified diagonals.
        *   `eigvals(a)`: Compute eigenvalues.

### Statistical Operations (`ember_ml.ops.stats.stats_ops`)

*   **`StatsOps(ABC)`**: The abstract base class for statistical operations.
    *   Includes abstract methods for:
        *   `mean(x, axis, keepdims)`: Compute the mean.
        *   `var(x, axis, keepdims, ddof)`: Compute the variance.
        *   `median(x, axis, keepdims)`: Compute the median.
        *   `std(x, axis, keepdims, ddof)`: Compute the standard deviation.
        *   `percentile(x, q, axis, keepdims)`: Compute the q-th percentile.
        *   `max(x, axis, keepdims)`: Compute the maximum value.
        *   `min(x, axis, keepdims)`: Compute the minimum value.
        *   `sum(x, axis, keepdims)`: Compute the sum.
        *   `cumsum(x, axis)`: Compute the cumulative sum.
        *   `argmax(x, axis, keepdims)`: Returns indices of maximum values.
        *   `sort(x, axis, descending)`: Sort a tensor.
        *   `argsort(x, axis, descending)`: Returns indices that would sort a tensor.
        *   `gaussian(input_value, mu, sigma)`: Compute the Gaussian function value.