# Operations (ops) Module

The `ember_ml.ops` module provides the primary, backend-agnostic interface for fundamental operations in Ember ML. Through a dynamic aliasing system, it exposes functions implemented by the currently active backend (NumPy, PyTorch, MLX), ensuring a consistent API regardless of the underlying computation library.

**Important Note on Input Handling:** The `ops` functions accept various tensor-like inputs (native backend tensors, `EmberTensor`, `Parameter`, NumPy arrays, Python lists/scalars). Backend implementations automatically handle input conversion and object unwrapping (e.g., `Parameter`) to access the native tensor data needed for computation.

**Important Note on Return Types:** Functions within the `ops` module return **native backend tensors** (e.g., `mlx.core.array`, `torch.Tensor`, `numpy.ndarray`), not `EmberTensor` instances. `EmberTensor` serves as a user-facing wrapper.

## Importing

```python
from ember_ml import ops
```

## Core Mathematical Operations

| Function | Description |
|----------|-------------|
| `ops.add(x, y)` | Element-wise addition of tensors |
| `ops.subtract(x, y)` | Element-wise subtraction of tensors |
| `ops.multiply(x, y)` | Element-wise multiplication of tensors |
| `ops.divide(x, y)` | Element-wise division of tensors |
| `ops.floor_divide(x, y)` | Element-wise floor division of tensors |
| `ops.mod(x, y)` | Element-wise remainder of division |
| `ops.dot(x, y)` | Dot product of tensors |
| `ops.matmul(x, y)` | Matrix multiplication of tensors |
| `ops.exp(x)` | Element-wise exponential of tensor |
| `ops.log(x)` | Element-wise natural logarithm of tensor |
| `ops.log10(x)` | Element-wise base-10 logarithm of tensor |
| `ops.log2(x)` | Element-wise base-2 logarithm of tensor |
| `ops.pow(x, y)` or `ops.power(x, y)` | Element-wise power function |
| `ops.sqrt(x)` | Element-wise square root of tensor |
| `ops.square(x)` | Element-wise square of tensor |
| `ops.abs(x)` | Element-wise absolute value of tensor |
| `ops.negative(x)` | Element-wise negation of tensor |
| `ops.sign(x)` | Element-wise sign of tensor |
| `ops.clip(x, min_val, max_val)` | Element-wise clipping of tensor values |
| `ops.gradient(f, x, dx=1.0, axis=None, edge_order=1)` | Compute the gradient of a function |

**Constants:**
- `ops.pi`: The mathematical constant pi.

## Trigonometric Functions

| Function | Description |
|----------|-------------|
| `ops.sin(x)` | Element-wise sine of tensor |
| `ops.cos(x)` | Element-wise cosine of tensor |
| `ops.tan(x)` | Element-wise tangent of tensor |
| `ops.sinh(x)` | Element-wise hyperbolic sine of tensor |
| `ops.cosh(x)` | Element-wise hyperbolic cosine of tensor |
| `ops.tanh(x)` | Element-wise hyperbolic tangent of tensor |

## Comparison Operations

| Function | Description |
|----------|-------------|
| `ops.equal(x, y)` | Element-wise equality comparison |
| `ops.not_equal(x, y)` | Element-wise inequality comparison |
| `ops.less(x, y)` | Element-wise less-than comparison |
| `ops.less_equal(x, y)` | Element-wise less-than-or-equal comparison |
| `ops.greater(x, y)` | Element-wise greater-than comparison |
| `ops.greater_equal(x, y)` | Element-wise greater-than-or-equal comparison |
| `ops.logical_and(x, y)` | Element-wise logical AND |
| `ops.logical_or(x, y)` | Element-wise logical OR |
| `ops.logical_not(x)` | Element-wise logical NOT |
| `ops.logical_xor(x, y)` | Element-wise logical XOR |
| `ops.allclose(x, y, rtol=1e-5, atol=1e-8)` | Returns whether all elements are close |
| `ops.isclose(x, y, rtol=1e-5, atol=1e-8)` | Returns whether each element is close |
| `ops.all(x, axis=None, keepdims=False)` | Test whether all elements evaluate to True |
| `ops.any(x, axis=None, keepdims=False)` | Test whether any elements evaluate to True |
| `ops.where(condition, x, y)` | Return elements chosen from x or y depending on condition |
| `ops.isnan(x)` | Test element-wise for NaN |

## Device Operations

| Function | Description |
|----------|-------------|
| `ops.to_device(x, device)` | Move tensor to the specified device |
| `ops.get_device(x)` | Get the device of a tensor |
| `ops.get_available_devices()` | Get a list of available devices |
| `ops.memory_usage(device=None)` | Get memory usage for the specified device |
| `ops.memory_info(device=None)` | Get detailed memory information for the specified device |
| `ops.synchronize(device=None)` | Synchronize computation on the specified device (backend-dependent) |
| `ops.set_default_device(device)` | Set the default device for the current backend |
| `ops.get_default_device()` | Get the default device for the current backend |
| `ops.is_available(device_name)` | Check if a specific device (e.g., 'cuda', 'mps') is available |

## I/O Operations

| Function | Description |
|----------|-------------|
| `ops.save(obj, path)` | Save object to file (backend-specific serialization) |
| `ops.load(path)` | Load object from file (backend-specific serialization) |

## Loss Operations

| Function | Description |
|----------|-------------|
| `ops.mse(y_true, y_pred)` | Mean squared error loss |
| `ops.mean_absolute_error(y_true, y_pred)` | Mean absolute error loss |
| `ops.binary_crossentropy(y_true, y_pred, from_logits=False)` | Binary crossentropy loss |
| `ops.categorical_crossentropy(y_true, y_pred, from_logits=False)` | Categorical crossentropy loss |
| `ops.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)` | Sparse categorical crossentropy loss |
| `ops.huber_loss(y_true, y_pred, delta=1.0)` | Huber loss |
| `ops.log_cosh_loss(y_true, y_pred)` | Logarithm of the hyperbolic cosine loss |

## Vector & FFT Operations

| Function | Description |
|----------|-------------|
| `ops.normalize_vector(x, axis=None)` | Normalize a vector or matrix |
| `ops.compute_energy_stability(x, axis=None)` | Compute energy stability of a vector |
| `ops.compute_interference_strength(x, y)` | Compute interference strength between vectors |
| `ops.compute_phase_coherence(x, y)` | Compute phase coherence between vectors |
| `ops.partial_interference(x, y, mask)` | Compute partial interference between vectors |
| `ops.euclidean_distance(x, y)` | Compute Euclidean distance between vectors |
| `ops.cosine_similarity(x, y)` | Compute cosine similarity between vectors |
| `ops.exponential_decay(x, rate=0.1)` | Apply exponential decay to a vector |
| `ops.fft(x, n=None, axis=-1)` | Compute the one-dimensional discrete Fourier Transform |
| `ops.ifft(x, n=None, axis=-1)` | Compute the one-dimensional inverse discrete Fourier Transform |
| `ops.fft2(x, s=None, axes=(-2,-1))` | Compute the two-dimensional discrete Fourier Transform |
| `ops.ifft2(x, s=None, axes=(-2,-1))` | Compute the two-dimensional inverse discrete Fourier Transform |
| `ops.fftn(x, s=None, axes=None)` | Compute the N-dimensional discrete Fourier Transform |
| `ops.ifftn(x, s=None, axes=None)` | Compute the N-dimensional inverse discrete Fourier Transform |
| `ops.rfft(x, n=None, axis=-1)` | Compute the one-dimensional DFT for real input |
| `ops.irfft(x, n=None, axis=-1)` | Compute the inverse of the RFFT |
| `ops.rfft2(x, s=None, axes=(-2,-1))` | Compute the two-dimensional DFT for real input |
| `ops.irfft2(x, s=None, axes=(-2,-1))` | Compute the inverse of the RFFT2 |
| `ops.rfftn(x, s=None, axes=None)` | Compute the N-dimensional DFT for real input |
| `ops.irfftn(x, s=None, axes=None)` | Compute the inverse of the RFFTN |

## Backend Management

The `ops` module also re-exports key backend management functions:

| Function | Description |
|----------|-------------|
| `ops.get_backend()` | Get the name of the current active backend |
| `ops.set_backend(backend_name)` | Set the active backend (e.g., 'numpy', 'torch', 'mlx') |
| `ops.auto_select_backend()` | Automatically select and set the best backend based on hardware |

## Notes

- All operations are backend-agnostic and work with any backend (NumPy, PyTorch, MLX) set via `ops.set_backend`.
- The operations follow a consistent API across different backends.
- Most operations support broadcasting, similar to NumPy and other array libraries.
- For tensor creation and manipulation (which return `EmberTensor`), use the `ember_ml.nn.tensor` module.
- For statistical operations (e.g., mean, std, var), use the `ember_ml.ops.stats` module.
- For linear algebra operations (e.g., svd, inv, det), use the `ember_ml.ops.linearalg` module.
- For activation functions (functional or Module classes), use the `ember_ml.nn.modules.activations` module.
- For feature extraction operations/factories, use the `ember_ml.nn.features` module.