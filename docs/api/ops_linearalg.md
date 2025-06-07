# Linear Algebra Operations (ops.linearalg)

The `ember_ml.ops.linearalg` module provides a comprehensive set of backend-agnostic linear algebra operations. These operations are essential for performing advanced mathematical computations on tensors.

**Important Note on Input/Output Types:** Functions within the `ops.linearalg` module accept a variety of tensor-like inputs, including native backend tensors (e.g., `mlx.core.array`), `EmberTensor` objects, `Parameter` objects, NumPy arrays, and Python lists/tuples/scalars. The backend implementation automatically handles converting these inputs and unwrapping objects like `Parameter` and `EmberTensor` to access the underlying native tensor data needed for computation. These functions return results as **native backend tensors**, not `EmberTensor` instances.

## Importing

```python
from ember_ml.ops import linearalg
```

## Available Functions

### Matrix Decomposition

| Function | Description |
|----------|-------------|
| `linearalg.qr(a, mode='reduced')` | Compute the QR decomposition of a matrix |
| `linearalg.svd(a, full_matrices=True, compute_uv=True)` | Compute the singular value decomposition of a matrix |
| `linearalg.cholesky(a)` | Compute the Cholesky decomposition of a matrix |
| `linearalg.eig(a)` | Compute the eigenvalues and right eigenvectors of a square matrix |
| `linearalg.eigvals(a)` | Compute the eigenvalues of a square matrix |

### Matrix Operations

| Function | Description |
|----------|-------------|
| `linearalg.solve(a, b)` | Solve a linear matrix equation, or system of linear scalar equations |
| `linearalg.inv(a)` | Compute the inverse of a matrix |
| `linearalg.det(a)` | Compute the determinant of a matrix |
| `linearalg.norm(x, ord=None, axis=None, keepdims=False)` | Compute the matrix or vector norm |
| `linearalg.lstsq(a, b, rcond=None)` | Compute the least-squares solution to a linear matrix equation |
| `linearalg.diag(v, k=0)` | Extract a diagonal or construct a diagonal array |
| `linearalg.diagonal(a, offset=0, axis1=0, axis2=1)` | Return specified diagonals of a tensor |

## Examples

### Matrix Decomposition

```python
from ember_ml import ops
from ember_ml.ops import linearalg
from ember_ml.nn import tensor

# Create a matrix
a = tensor.convert_to_tensor([[1, 2], [3, 4]])

# QR decomposition
q, r = linearalg.qr(a)
print("Q:", q)  # Orthogonal matrix
print("R:", r)  # Upper triangular matrix

# SVD decomposition
u, s, v = linearalg.svd(a)
print("U:", u)  # Left singular vectors
print("S:", s)  # Singular values
print("V:", v)  # Right singular vectors

# Cholesky decomposition
a_positive_definite = tensor.convert_to_tensor([[2, 1], [1, 2]])
l = linearalg.cholesky(a_positive_definite)
print("L:", l)  # Lower triangular matrix
```

### Solving Linear Systems

```python
from ember_ml import ops
from ember_ml.ops import linearalg
from ember_ml.nn import tensor

# Create a coefficient matrix and constants
a = tensor.convert_to_tensor([[3, 1], [1, 2]])
b = tensor.convert_to_tensor([9, 8])

# Solve the system of equations
x = linearalg.solve(a, b)
print("Solution:", x)  # [2, 3]

# Verify the solution
result = ops.matmul(a, x)
print("Verification:", result)  # [9, 8]
```

### Matrix Properties

```python
from ember_ml.ops import linearalg
from ember_ml.nn import tensor

# Create a matrix
a = tensor.convert_to_tensor([[1, 2], [3, 4]])

# Compute determinant
det_a = linearalg.det(a)
print("Determinant:", det_a)  # -2

# Compute inverse
inv_a = linearalg.inv(a)
print("Inverse:", inv_a)  # [[-2, 1], [1.5, -0.5]]

# Compute norm
norm_a = linearalg.norm(a, ord='fro')
print("Frobenius norm:", norm_a)  # 5.477...
```

### Working with Diagonals

```python
from ember_ml.ops import linearalg
from ember_ml.nn import tensor

# Create a vector and form a diagonal matrix
v = tensor.convert_to_tensor([1, 2, 3])
diag_matrix = linearalg.diag(v)
print("Diagonal matrix:\n", diag_matrix)
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]

# Extract diagonal from a matrix
a = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diag_values = linearalg.diagonal(a)
print("Diagonal values:", diag_values)  # [1, 5, 9]

# Extract off-diagonal
off_diag = linearalg.diagonal(a, offset=1)
print("Off-diagonal values:", off_diag)  # [2, 6]
```

## Notes

- All operations are backend-agnostic and work with any backend (NumPy, PyTorch, MLX).
- The operations follow a consistent API across different backends.
- For basic mathematical operations, use the `ember_ml.ops` module.
- For statistical operations, use the `ember_ml.ops.stats` module.
- For tensor creation and manipulation, use the `ember_ml.nn.tensor` module.

## Implementation Details

The linear algebra operations are implemented using the backend abstraction system, which dispatches calls to the appropriate backend implementation based on the currently selected backend. The interfaces are defined in `ember_ml.ops.linearalg.linearalg_ops`, and the implementations are provided in the backend-specific modules.

```python
# Example of backend selection
from ember_ml.ops import set_backend

# Use NumPy backend
set_backend('numpy')
result_numpy = linearalg.det(a)

# Use PyTorch backend
set_backend('torch')
result_torch = linearalg.det(a)

# Use MLX backend
set_backend('mlx')
result_mlx = linearalg.det(a)
```

The results should be consistent across backends, with only minor numerical differences due to backend-specific implementations.