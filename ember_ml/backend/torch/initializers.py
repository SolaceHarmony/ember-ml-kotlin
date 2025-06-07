"""PyTorch backend initializers."""

import math
import torch
from ember_ml.backend.torch.tensor.ops.random import random_normal # Correct backend import

# Helper for double-single precision arithmetic adapted for PyTorch
def _add_double_single(a_high, a_low, b_high, b_low):
    """Helper for double-single precision arithmetic (PyTorch)."""
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

# Custom QR decomposition adapted for PyTorch (Square Matrix Case)
def _custom_qr_square(matrix_high, matrix_low):
    """PyTorch QR decomposition for square matrices with increased numerical stability."""
    rows, cols = matrix_high.shape
    if rows != cols:
         raise ValueError("_custom_qr_square only supports square matrices.")

    q_high = torch.zeros((rows, cols), dtype=torch.float32, device=matrix_high.device)
    r_high = torch.zeros((cols, cols), dtype=torch.float32, device=matrix_high.device)
    r_low = torch.zeros((cols, cols), dtype=torch.float32, device=matrix_high.device)

    temp_v_high = matrix_high.clone() # Work on a copy
    temp_v_low = matrix_low.clone()   # Work on a copy

    for i in range(cols):
        v_high, v_low = temp_v_high[:, i], temp_v_low[:, i]

        for j in range(i):
            qj_high = q_high[:, j] # Shape (rows,)

            # Calculate r_high[j, i]
            # torch.dot needs 1D tensors
            r_val_high = torch.dot(qj_high, v_high)
            r_high[j, i] = r_val_high

            # Calculate r_low[j, i] (assuming q_low is zero initially)
            r_val_low = torch.dot(qj_high, v_low)
            r_low[j, i] = r_val_low

            # Calculate projection
            # Need to unsqueeze for broadcasting: (rows, 1) * (1,) -> (rows, 1)
            proj_high = qj_high.unsqueeze(1) * r_val_high
            proj_low = qj_high.unsqueeze(1) * r_val_low

            # Subtract projection using double-single arithmetic
            # Ensure shapes match for subtraction (rows,) - (rows, 1)[:, 0]
            v_high, v_low = _add_double_single(v_high, v_low, -proj_high[:, 0], -proj_low[:, 0])

        # Update the temporary matrix columns after projections
        temp_v_high[:, i] = v_high
        temp_v_low[:, i] = v_low

        # Calculate norm
        norm_high = torch.linalg.norm(v_high)
        if norm_high < 1e-10:
             raise ValueError(f"Column norm too small (col={i}). Check initialization.")

        # Normalize and store in Q
        q_col = v_high / norm_high
        q_high[:, i] = q_col

        # Update R diagonal
        r_high[i, i] = norm_high
        # r_low diagonal update might depend on v_low and norm calculation details omitted here

    return q_high, r_high # Returning R low part might be misleading if not fully computed

# Orthogonal initializer adapted for PyTorch
def orthogonal(shape, gain=1.0, device=None):
    """PyTorch orthogonal matrix initialization."""
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    rows, cols = shape[0], math.prod(shape[1:])

    # Generate a random matrix using the backend-specific random_normal
    a = random_normal((rows, cols), dtype=torch.float32, device=device)

    # Perform QR decomposition using PyTorch
    # torch.linalg.qr returns Q, R
    q, _ = torch.linalg.qr(a, mode='reduced' if rows >= cols else 'complete')

    # Ensure Q has the correct shape (rows x cols)
    # 'reduced' gives (rows x min(rows, cols))
    # 'complete' gives (rows x rows)
    # We need to handle both cases to get a (rows x cols) matrix
    if rows < cols:
        # If wide matrix, take the first 'rows' columns of the 'complete' Q
        # then transpose to get cols x rows, then take first cols rows? No.
        # Let's rethink. We need an orthogonal matrix of shape (rows, cols).
        # If rows < cols, generate (cols, rows) and transpose Q.
        a_t = random_normal((cols, rows), dtype=torch.float32, device=device)
        q_t, _ = torch.linalg.qr(a_t, mode='reduced') # q_t is (cols, rows)
        q = q_t.t() # q is (rows, cols)
    else:
        # If tall or square matrix, 'reduced' gives (rows, cols) Q directly
        q = q[:, :cols] # Ensure correct columns if square

    # Apply gain and reshape
    param = torch.multiply(q, gain)
    return param.reshape(shape)


# Standard PyTorch initializers
def zeros(shape, dtype=torch.float32, device=None):
    """PyTorch zeros initializer."""
    return torch.zeros(shape, dtype=dtype, device=device)

def ones(shape, dtype=torch.float32, device=None):
    """PyTorch ones initializer."""
    return torch.ones(shape, dtype=dtype, device=device)

def glorot_uniform(shape, gain=1.0, dtype=torch.float32, device=None):
    """PyTorch Glorot uniform initializer."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    torch.nn.init.xavier_uniform_(tensor, gain=gain)
    return tensor