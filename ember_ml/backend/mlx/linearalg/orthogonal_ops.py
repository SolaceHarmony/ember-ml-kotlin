"""
MLX implementation of orthogonal matrix operations.

This module provides MLX-specific implementation of orthogonal matrix operations.
"""

import math
import mlx.core as mx
import logging # Import logging
from typing import Tuple, Optional, Any, Union

# Removed HPC16x8 import as it's no longer used here for QR
from ember_ml.backend.mlx.types import TensorLike
# Import the consolidated QR function
from .qr_ops import qr

logger = logging.getLogger(__name__) # Get logger instance

# Removed _add_double_single helper function
# Removed _custom_qr function

def orthogonal(shape: Union[Tuple[int, ...], TensorLike], gain: float = 1.0, dtype: Optional[Any] = None, device: Optional[str] = None) -> mx.array:
    """
    MLX-specific orthogonal matrix initialization using the consolidated QR implementation.

    Args:
        shape: Shape of the tensor to initialize. Must have at least 2 dimensions.
        gain: Multiplicative factor to apply to the orthogonal matrix.
        dtype: Data type of the tensor (optional, not used in MLX implementation).
        device: Device to place the tensor on (optional, not used in MLX implementation).

    Returns:
        A random orthogonal matrix of the specified shape.

    Raises:
        ValueError: If shape has fewer than 2 dimensions.
        RuntimeError: If the underlying QR decomposition fails.
    """
    if isinstance(shape, mx.array):
        # If shape is an MLX array, convert to tuple of Python integers
        shape_tuple = tuple(int(dim.item()) if hasattr(dim, 'item') else int(dim) for dim in shape)
    else:
        # Otherwise, assume it's already a tuple or list
        shape_tuple = tuple(int(dim.item()) if hasattr(dim, 'item') else int(dim) for dim in shape)

    if len(shape_tuple) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    rows, cols = shape_tuple[0], math.prod(shape_tuple[1:])
    size = max(rows, cols) # Create a square matrix for QR

    # Generate a random square matrix
    random_matrix = mx.random.normal(
        shape=(size, size),
        dtype=mx.float32,
        loc=0.0,
        scale=1.0
    )

    # Perform QR decomposition using the consolidated function from qr_ops
    # We need the 'raw' Q matrix (M x M or N x N, whichever is larger)
    try:
        # qr returns Q (M, M), R (M, N) in 'complete' mode
        # qr returns Q (M, M) in 'raw' mode
        q_matrix = qr(random_matrix, mode='raw') # Get the square orthogonal matrix Q
    except Exception as e:
        logger.error(f"QR decomposition failed during orthogonal initialization: {e}")
        raise RuntimeError(f"QR decomposition failed: {e}")

    # Log orthogonality error
    q_t_q = mx.matmul(mx.transpose(q_matrix), q_matrix)
    identity = mx.eye(size)
    error = mx.mean(mx.abs(q_t_q - identity)).item()
    logger.info(f"Orthogonal: shape={shape_tuple}, intermediate_size={size}, error={error:.2e}")

    # Take the relevant part of Q and reshape
    q_final = q_matrix[:rows, :cols]

    # Apply gain and reshape
    logger.info(f"Orthogonal: Returning matrix with shape {shape_tuple}")
    return mx.multiply(gain, q_final.reshape(shape_tuple)) # Use mx.multiply