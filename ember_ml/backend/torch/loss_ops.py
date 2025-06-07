"""PyTorch backend implementation for loss operations."""

from typing import Any, Optional, Union, Sequence
import torch
import torch.nn.functional as F

# from ember_ml.ops.loss_ops import LossOps # REMOVED inheritance
# Removed top-level import: from ember_ml.backend.torch.tensor.tensor import TorchTensor as Tensor
from ember_ml.backend.torch.tensor.tensor import TensorLike
# from ember_ml.backend.torch import math_ops # REMOVED top-level import

# Epsilon for numerical stability
EPSILON = 1e-7

# Helper function (moved to module level)
def _reduce_loss(loss: torch.Tensor,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False,
                 reduction: str = 'mean') -> torch.Tensor:
    """
    Helper to apply reduction to loss tensor.
    
    Args:
        loss: Loss tensor
        axis: Axis or axes along which to reduce
        keepdims: Whether to keep the reduced dimensions
        reduction: Type of reduction ('mean', 'sum', or 'none')
        
    Returns:
        Reduced loss tensor
    """
    if reduction == 'none':
        return loss
        
    if axis is None:
        # Reduce over all elements
        if reduction == 'mean':
            return torch.mean(loss)  # keepdims doesn't apply here
        elif reduction == 'sum':
            return torch.sum(loss)  # keepdims doesn't apply here
    else:
        # Reduce over specified axes
        if isinstance(axis, int):
            dim = axis
        elif isinstance(axis, Sequence):
            dim = tuple(axis)
        else:
            raise ValueError(f"Unsupported axis type for PyTorch reduction: {type(axis)}")
            
        if reduction == 'mean':
            return torch.mean(loss, dim=dim, keepdim=keepdims)
        elif reduction == 'sum':
            return torch.sum(loss, dim=dim, keepdim=keepdims)
    
    raise ValueError(f"Invalid reduction: {reduction}")

# --- Standalone Loss Functions ---

def mse(y_true: TensorLike, y_pred: TensorLike,
                       axis: Optional[Union[int, Sequence[int]]] = None,
                       keepdims: bool = False,
                       reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the mean squared error between predictions and targets.
    
    Args:
        y_true: Target values
        y_pred: Predicted values
        axis: Axis or axes along which to compute the MSE
        keepdims: Whether to keep the reduced dimensions
        reduction: Type of reduction to apply ('mean', 'sum', or 'none')
        
    Returns:
        Mean squared error
    """
    from ember_ml.backend.torch.tensor.ops.utility import _convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true)
    y_pred_t = convert_to_torch_tensor(data=y_pred)
    
    # Calculate squared difference
    squared_diff = torch.square(y_pred_t - y_true_t)
    
    # Use PyTorch's built-in MSE loss for the common case
    if axis is None and not keepdims and reduction == 'mean':
        return F.mse_loss(y_pred_t, y_true_t, reduction='mean')
    elif axis is None and not keepdims and reduction == 'sum':
        return F.mse_loss(y_pred_t, y_true_t, reduction='sum')
    else:
        return _reduce_loss(squared_diff, axis=axis, keepdims=keepdims, reduction=reduction)

def mean_absolute_error(y_true: TensorLike, y_pred: TensorLike,
                         axis: Optional[Union[int, Sequence[int]]] = None,
                         keepdims: bool = False) -> torch.Tensor:
    """PyTorch implementation of mean absolute error."""
    from ember_ml.backend.torch.tensor.ops.utility import convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true)
    y_pred_t = convert_to_torch_tensor(data=y_pred)
    abs_diff = torch.abs(y_pred_t - y_true_t)
    if axis is None and not keepdims:
         return F.l1_loss(y_pred_t, y_true_t, reduction='mean')
    else:
         return _reduce_loss(abs_diff, axis=axis, keepdims=keepdims)

def binary_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                        from_logits: bool = False,
                        axis: Optional[Union[int, Sequence[int]]] = None,
                        keepdims: bool = False) -> torch.Tensor:
    """PyTorch implementation of binary crossentropy."""
    from ember_ml.backend.torch.tensor.ops.utility import convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true)
    y_pred_t = convert_to_torch_tensor(data=y_pred)

    if from_logits:
         loss = F.binary_cross_entropy_with_logits(y_pred_t, y_true_t.to(y_pred_t.dtype), reduction='none')
    else:
         y_pred_t = torch.clamp(y_pred_t, EPSILON, 1.0 - EPSILON)
         loss = F.binary_cross_entropy(y_pred_t, y_true_t.to(y_pred_t.dtype), reduction='none')

    return _reduce_loss(loss, axis=axis, keepdims=keepdims)

def categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                             from_logits: bool = False,
                             axis: Optional[Union[int, Sequence[int]]] = None,
                             keepdims: bool = False) -> torch.Tensor:
    """PyTorch implementation of categorical crossentropy."""
    from ember_ml.backend.torch.tensor.ops.utility import convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true)
    y_pred_t = convert_to_torch_tensor(data=y_pred)

    if from_logits:
        log_probs = F.log_softmax(y_pred_t, dim=-1)
    else:
        y_pred_t = torch.clamp(y_pred_t, EPSILON, 1.0 - EPSILON)
        log_probs = torch.log(y_pred_t)

    cce = -torch.sum(y_true_t * log_probs, dim=-1)
    return _reduce_loss(cce, axis=axis, keepdims=keepdims)

def sparse_categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                                    from_logits: bool = False,
                                    axis: Optional[Union[int, Sequence[int]]] = None,
                                    keepdims: bool = False) -> torch.Tensor:
    """PyTorch implementation of sparse categorical crossentropy."""
    from ember_ml.backend.torch.tensor.ops.utility import convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true).long() # Ensure integer type
    y_pred_t = convert_to_torch_tensor(data=y_pred)

    if not from_logits:
         y_pred_t = torch.clamp(y_pred_t, EPSILON, 1.0 - EPSILON)
         y_pred_t = torch.log(y_pred_t)

    loss = F.cross_entropy(y_pred_t, y_true_t, reduction='none')
    return _reduce_loss(loss, axis=axis, keepdims=keepdims)

def huber_loss(y_true: TensorLike, y_pred: TensorLike, delta: float = 1.0,
               axis: Optional[Union[int, Sequence[int]]] = None,
               keepdims: bool = False) -> torch.Tensor:
    """PyTorch implementation of Huber loss."""
    from ember_ml.backend.torch.tensor.ops.utility import convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true)
    y_pred_t = convert_to_torch_tensor(data=y_pred)
    loss = F.huber_loss(y_pred_t, y_true_t, delta=delta, reduction='none')
    return _reduce_loss(loss, axis=axis, keepdims=keepdims)

def log_cosh_loss(y_true: TensorLike, y_pred: TensorLike,
                  axis: Optional[Union[int, Sequence[int]]] = None,
                  keepdims: bool = False) -> torch.Tensor:
    """PyTorch implementation of log-cosh loss."""
    from ember_ml.backend.torch.tensor.ops.utility import convert_to_torch_tensor # Use functional import
    y_true_t = convert_to_torch_tensor(data=y_true)
    y_pred_t = convert_to_torch_tensor(data=y_pred)
    error = y_pred_t - y_true_t
    logcosh = torch.logaddexp(error, -error) - torch.log(torch.tensor(2.0, dtype=error.dtype, device=error.device))
    return _reduce_loss(logcosh, axis=axis, keepdims=keepdims)

# (Optional) Define __all__ if needed for this module directly
# __all__ = [ 'mse', 'mean_absolute_error', ... ]