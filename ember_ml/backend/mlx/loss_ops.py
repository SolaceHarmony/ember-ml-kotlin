"""
MLX backend loss operations.

This module provides MLX implementations of loss operations.
"""

import mlx.core as mx
from typing import Optional, Union, Sequence

from ember_ml.backend.mlx.types import TensorLike

# Epsilon for numerical stability
EPSILON = 1e-7

def mse(y_true: TensorLike, y_pred: TensorLike,
                       axis: Optional[Union[int, Sequence[int]]] = None,
                       keepdims: bool = False,
                       reduction: str = 'mean') -> mx.array:
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
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    # Calculate squared difference
    squared_diff = mx.square(mx.subtract(y_pred_tensor, y_true_tensor))
    
    # Apply reduction
    if reduction == 'none':
        return squared_diff
    
    # Reduce along specified axes
    if axis is None:
        if reduction == 'mean':
            return mx.mean(squared_diff, keepdims=keepdims)
        elif reduction == 'sum':
            return mx.sum(squared_diff, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = squared_diff
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            if reduction == 'mean':
                result = mx.mean(result, axis=ax, keepdims=keepdims)
            elif reduction == 'sum':
                result = mx.sum(result, axis=ax, keepdims=keepdims)
        return result
    
    if reduction == 'mean':
        return mx.mean(squared_diff, axis=axis, keepdims=keepdims)
    elif reduction == 'sum':
        return mx.sum(squared_diff, axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def mean_absolute_error(y_true: TensorLike, y_pred: TensorLike,
                        axis: Optional[Union[int, Sequence[int]]] = None,
                        keepdims: bool = False) -> mx.array:
    """
    Compute the mean absolute error between predictions and targets.
    
    Args:
        y_true: Target values
        y_pred: Predicted values
        axis: Axis or axes along which to compute the MAE
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Mean absolute error
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    # Calculate absolute difference
    abs_diff = mx.abs(mx.subtract(y_pred_tensor, y_true_tensor))
    
    # Reduce along specified axes
    if axis is None:
        return mx.mean(abs_diff, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = abs_diff
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.mean(abs_diff, axis=axis, keepdims=keepdims)


def binary_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                       from_logits: bool = False,
                       axis: Optional[Union[int, Sequence[int]]] = None,
                       keepdims: bool = False) -> mx.array:
    """
    Compute the binary crossentropy loss between predictions and targets.
    
    Args:
        y_true: Target values (binary)
        y_pred: Predicted values or logits
        from_logits: Whether y_pred is logits (True) or probabilities (False)
        axis: Axis or axes along which to compute the loss
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Binary crossentropy loss
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    if from_logits:
        # Stable implementation: max(logits, 0) - logits * y_true + log(1 + exp(-abs(logits)))
        max_val = mx.maximum(y_pred_tensor, 0)
        log_exp_term = mx.log(1 + mx.exp(-mx.abs(y_pred_tensor)))
        loss = max_val - y_pred_tensor * y_true_tensor + log_exp_term
    else:
        # Clip predictions for numerical stability
        y_pred_tensor = mx.clip(y_pred_tensor, EPSILON, 1.0 - EPSILON)
        loss = - (y_true_tensor * mx.log(y_pred_tensor) +
                 (1.0 - y_true_tensor) * mx.log(1.0 - y_pred_tensor))
    
    # Reduce along specified axes
    if axis is None:
        return mx.mean(loss, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = loss
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.mean(loss, axis=axis, keepdims=keepdims)


def categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                            from_logits: bool = False,
                            axis: Optional[Union[int, Sequence[int]]] = None,
                            keepdims: bool = False) -> mx.array:
    """
    Compute the categorical crossentropy loss between predictions and targets.
    
    Args:
        y_true: Target values (one-hot encoded)
        y_pred: Predicted values or logits
        from_logits: Whether y_pred is logits (True) or probabilities (False)
        axis: Axis or axes along which to compute the loss
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Categorical crossentropy loss
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    if from_logits:
        # Implement log_softmax manually since mx.core doesn't have it
        # log_softmax(x) = x - log(sum(exp(x)))
        max_y_pred = mx.max(y_pred_tensor, axis=-1, keepdims=True)
        y_pred_shifted = mx.subtract(y_pred_tensor, max_y_pred)  # For numerical stability
        exp_y_pred = mx.exp(y_pred_shifted)
        sum_exp = mx.sum(exp_y_pred, axis=-1, keepdims=True)
        log_sum_exp = mx.log(sum_exp)
        log_probs = mx.subtract(y_pred_shifted, log_sum_exp)
    else:
        y_pred_tensor = mx.clip(y_pred_tensor, EPSILON, 1.0 - EPSILON)
        log_probs = mx.log(y_pred_tensor)
    
    cce = -mx.sum(y_true_tensor * log_probs, axis=-1)
    
    # Reduce along specified axes
    if axis is None:
        return mx.mean(cce, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = cce
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.mean(cce, axis=axis, keepdims=keepdims)


def sparse_categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                                   from_logits: bool = False,
                                   axis: Optional[Union[int, Sequence[int]]] = None,
                                   keepdims: bool = False) -> mx.array:
    """
    Compute the sparse categorical crossentropy loss between predictions and targets.
    
    Args:
        y_true: Target values (class indices)
        y_pred: Predicted values or logits
        from_logits: Whether y_pred is logits (True) or probabilities (False)
        axis: Axis or axes along which to compute the loss
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sparse categorical crossentropy loss
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true).astype(mx.int32)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    if not from_logits:
        y_pred_tensor = mx.clip(y_pred_tensor, EPSILON, 1.0 - EPSILON)
        y_pred_tensor = mx.log(y_pred_tensor)
    
    probs = mx.softmax(y_pred_tensor, axis=-1)
    log_probs = mx.log(probs)
    y_true_tensor_expanded = mx.expand_dims(y_true_tensor, axis=-1)
    neg_log_likelihood = -mx.take_along_axis(log_probs, y_true_tensor_expanded, axis=-1)
    loss = mx.squeeze(neg_log_likelihood, axis=-1)
    
    # Reduce along specified axes
    if axis is None:
        return mx.mean(loss, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = loss
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.mean(loss, axis=axis, keepdims=keepdims)


def huber_loss(y_true: TensorLike, y_pred: TensorLike, delta: float = 1.0,
              axis: Optional[Union[int, Sequence[int]]] = None,
              keepdims: bool = False) -> mx.array:
    """
    Compute the Huber loss between predictions and targets.
    
    Args:
        y_true: Target values
        y_pred: Predicted values
        delta: Threshold at which to change from quadratic to linear
        axis: Axis or axes along which to compute the loss
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Huber loss
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    error = mx.subtract(y_pred_tensor, y_true_tensor)
    abs_error = mx.abs(error)
    quadratic = mx.minimum(abs_error, delta)
    linear = mx.subtract(abs_error, quadratic)
    loss = mx.add(mx.multiply(0.5, mx.square(quadratic)), mx.multiply(delta, linear))
    
    # Reduce along specified axes
    if axis is None:
        return mx.mean(loss, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = loss
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.mean(loss, axis=axis, keepdims=keepdims)


def log_cosh_loss(y_true: TensorLike, y_pred: TensorLike,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False) -> mx.array:
    """
    Compute the log-cosh loss between predictions and targets.
    
    Args:
        y_true: Target values
        y_pred: Predicted values
        axis: Axis or axes along which to compute the loss
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Log-cosh loss
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    y_true_tensor = tensor_ops.convert_to_tensor(y_true)
    y_pred_tensor = tensor_ops.convert_to_tensor(y_pred)
    
    error = mx.subtract(y_pred_tensor, y_true_tensor)
    logcosh = mx.subtract(mx.logaddexp(error, mx.negative(error)), mx.log(mx.array(2.0, dtype=error.dtype)))
    
    # Reduce along specified axes
    if axis is None:
        return mx.mean(logcosh, keepdims=keepdims)
    
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = logcosh
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.mean(result, axis=ax, keepdims=keepdims)
        return result
    
    return mx.mean(logcosh, axis=axis, keepdims=keepdims)