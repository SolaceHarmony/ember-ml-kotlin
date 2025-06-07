"""NumPy backend initializers."""

import math
import numpy as np

# Orthogonal initializer adapted for NumPy
def orthogonal(shape, gain=1.0):
    """NumPy-specific orthogonal matrix initialization."""
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    rows, cols = shape[0], math.prod(shape[1:])
    
    # Generate a random matrix using NumPy
    # Create a matrix with dimensions max(rows, cols) x min(rows, cols)
    # This is more efficient for QR decomposition
    flat_shape = (max(rows, cols), min(rows, cols))
    a = np.random.normal(0.0, 1.0, flat_shape).astype(np.float32)

    # Perform QR decomposition using NumPy
    q, r = np.linalg.qr(a)

    # Adjust Q shape if necessary
    # If rows < cols, Q will be max(rows, cols) x rows. We need rows x cols.
    # If rows >= cols, Q will be max(rows, cols) x cols. We need rows x cols.
    q = q[:rows, :cols]

    # Apply gain and reshape
    param = np.multiply(gain, q)
    return param.reshape(shape)

# Add other NumPy initializers here as needed (e.g., glorot_uniform, zeros, ones)
# For now, only orthogonal is adapted from the MLX version.

def zeros(shape, dtype=np.float32):
    """NumPy zeros initializer."""
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype=np.float32):
    """NumPy ones initializer."""
    return np.ones(shape, dtype=dtype)

def glorot_uniform(shape, gain=1.0, dtype=np.float32):
    """NumPy Glorot uniform initializer."""
    fan_in, fan_out = _compute_fans(shape)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(dtype)

def _compute_fans(shape):
    """Computes fan-in and fan-out for Glorot initializers."""
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions for fan calculation")
    fan_in = shape[1]
    fan_out = shape[0]
    if len(shape) > 2:
        receptive_field_size = math.prod(shape[2:])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size
    return fan_in, fan_out