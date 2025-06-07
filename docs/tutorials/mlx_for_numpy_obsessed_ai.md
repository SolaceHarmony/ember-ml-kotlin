# MLX for NumPy Users: A Practical Guide

## Introduction
MLX is Apple's machine learning framework designed for Apple Silicon. While it shares many similarities with NumPy, there are important differences in function signatures, parameter requirements, and behavior that can cause confusion. This guide aims to clarify these differences and provide practical examples.

## Array Creation

| NumPy | MLX | Key Differences |
|-------|-----|----------------|
| `np.array([1, 2, 3])` | `mx.array([1, 2, 3])` | Similar usage |
| `np.zeros((3, 3))` | `mx.zeros((3, 3))` | Similar usage |
| `np.ones((3, 3))` | `mx.ones((3, 3))` | Similar usage |
| `np.eye(3)` | `mx.eye(3)` | Similar usage |
| `np.arange(10)` | `mx.arange(10)` | Similar usage |
| `np.linspace(0, 1, 10)` | `mx.linspace(0, 1, 10)` | Similar usage |

## Indexing and Slicing

| NumPy | MLX | Key Differences |
|-------|-----|----------------|
| `a[0, 1]` | `a[0, 1]` | Similar for basic indexing |
| `a[1:3, 2:4]` | `a[1:3, 2:4]` | Similar for basic slicing |
| `np.take(a, indices, axis=0)` | `mx.take(a, indices, axis=0)` | Similar usage |
| N/A | `mx.slice(a, start_indices, axes, slice_size)` | **MLX-specific!** <br>- `start_indices` must be an MLX array <br>- `axes` must be a Python list/tuple <br>- `slice_size` must be a Python list/tuple |
| N/A | `mx.slice_update(a, update, start_indices, axes)` | **MLX-specific!** <br>- `update` is the new values <br>- `start_indices` must be an MLX array <br>- `axes` must be a Python list/tuple |

### Example: Using `mx.slice` and `mx.slice_update`

```python
import mlx.core as mx

# Create a test array
a = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Extract a slice
start_indices = mx.array([0, 1])  # Start at row 0, column 1
axes = [0, 1]  # Python list, not MLX array!
slice_size = [2, 2]  # Python list, not MLX array!
b = mx.slice(a, start_indices, axes, slice_size)
# b = [[2, 3], [5, 6]]

# Update a slice
update = mx.array([[10, 11], [12, 13]])
c = mx.slice_update(a, update, start_indices, axes)
# c = [[1, 10, 11], [4, 12, 13], [7, 8, 9]]
```

## Mathematical Operations

| NumPy | MLX | Key Differences |
|-------|-----|----------------|
| `a + b` | `mx.add(a, b)` | MLX prefers function calls over operators |
| `a - b` | `mx.subtract(a, b)` | MLX prefers function calls over operators |
| `a * b` | `mx.multiply(a, b)` | MLX prefers function calls over operators |
| `a / b` | `mx.divide(a, b)` | MLX prefers function calls over operators |
| `np.matmul(a, b)` | `mx.matmul(a, b)` | Similar usage |
| `np.dot(a, b)` | `mx.matmul(a, b)` | Use matmul in MLX |
| `np.exp(a)` | `mx.exp(a)` | Similar usage |
| `np.log(a)` | `mx.log(a)` | Similar usage |
| `np.sin(a)` | `mx.sin(a)` | Similar usage |

## Reduction Operations

| NumPy | MLX | Key Differences |
|-------|-----|----------------|
| `np.sum(a, axis=0)` | `mx.sum(a, axis=0)` | Similar usage |
| `np.max(a, axis=0)` | `mx.max(a, axis=0)` | Similar usage |
| `np.min(a, axis=0)` | `mx.min(a, axis=0)` | Similar usage |
| `np.mean(a, axis=0)` | `mx.mean(a, axis=0)` | Similar usage |
| `np.var(a, axis=0)` | `mx.var(a, axis=0)` | Similar usage |
| `np.std(a, axis=0)` | `mx.std(a, axis=0)` | Similar usage |

## Shape Manipulation

| NumPy | MLX | Key Differences |
|-------|-----|----------------|
| `np.reshape(a, (2, 3))` | `mx.reshape(a, (2, 3))` | Similar usage |
| `np.transpose(a)` | `mx.transpose(a)` | Similar usage |
| `np.concatenate([a, b], axis=0)` | `mx.concatenate([a, b], axis=0)` | Similar usage |
| `np.stack([a, b], axis=0)` | `mx.stack([a, b], axis=0)` | Similar usage |
| `np.split(a, 3, axis=0)` | `mx.split(a, 3, axis=0)` | Similar usage |

## Key Differences and Gotchas

1. **Parameter Types**: 
   - MLX functions like `mx.slice` and `mx.slice_update` require specific parameter types
   - Some parameters must be MLX arrays, others must be Python lists/tuples
   - Never use Python type conversions like `int()` or `float()` on MLX arrays

2. **No Direct Python Operators**:
   - Prefer MLX functions (`mx.add`, `mx.subtract`) over Python operators (`+`, `-`)
   - This ensures proper handling of MLX arrays

3. **MLX Arrays vs. Python Types**:
   - Keep values as MLX arrays whenever possible
   - When Python types are required (e.g., for `axes` in `mx.slice`), use Python lists/tuples directly
   - Avoid converting between MLX arrays and Python types

4. **Function Signatures**:
   - Always check the MLX documentation for the exact function signature
   - Pay attention to parameter types and requirements

## Testing MLX Code

Always test MLX operations with simple examples before using them in complex code:

```python
import mlx.core as mx

# Test mx.slice
a = mx.array([[1, 2, 3], [4, 5, 6]])
start = mx.array([0, 1])
axes = [0, 1]  # Python list!
size = [1, 2]  # Python list!
result = mx.slice(a, start, axes, size)
print(result)  # Should be [[2, 3]]

# Test mx.slice_update
update = mx.array([[10, 11]])
result = mx.slice_update(a, update, start, axes)
print(result)  # Should be [[1, 10, 11], [4, 5, 6]]