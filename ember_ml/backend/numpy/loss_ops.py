"""NumPy backend implementation for loss operations."""

from typing import Optional, Union, Sequence
import numpy as np

from ember_ml.backend.numpy.types import TensorLike, default_int

# Epsilon for numerical stability in crossentropy
EPSILON = 1e-7

# --- Standalone Loss Functions ---

def mse(y_true: TensorLike, y_pred: TensorLike,
                       axis: Optional[Union[int, Sequence[int]]] = None,
                       keepdims: bool = False,
                       reduction: str = 'mean') -> np.ndarray:
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
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_arr = tensor.convert_to_tensor(data=y_true)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)
    
    # Calculate squared difference
    squared_diff = np.square(y_pred_arr - y_true_arr)
    
    # Apply reduction
    if reduction == 'none':
        return squared_diff
    elif reduction == 'mean':
        return np.mean(squared_diff, axis=axis, keepdims=keepdims)
    elif reduction == 'sum':
        return np.sum(squared_diff, axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def mean_absolute_error(y_true: TensorLike, y_pred: TensorLike,
                         axis: Optional[Union[int, Sequence[int]]] = None,
                         keepdims: bool = False) -> np.ndarray:
    """NumPy implementation of mean absolute error."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_arr = tensor.convert_to_tensor(data=y_true)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)
    abs_diff = np.abs(y_pred_arr - y_true_arr)
    return np.mean(abs_diff, axis=axis, keepdims=keepdims)

def binary_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                        from_logits: bool = False,
                        axis: Optional[Union[int, Sequence[int]]] = None,
                        keepdims: bool = False) -> np.ndarray:
    """NumPy implementation of binary crossentropy."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_arr = tensor.convert_to_tensor(data=y_true)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)

    if from_logits:
        # Apply sigmoid if input is logits
        from ember_ml.backend.numpy import math_ops # Lazy import
        y_pred_arr = math_ops.sigmoid(y_pred_arr) # Assuming sigmoid exists in numpy math_ops

    # Clip predictions for numerical stability
    y_pred_arr = np.clip(y_pred_arr, EPSILON, 1.0 - EPSILON)

    bce = - (y_true_arr * np.log(y_pred_arr) +
             (1.0 - y_true_arr) * np.log(1.0 - y_pred_arr))
    return np.mean(bce, axis=axis, keepdims=keepdims)

def categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                             from_logits: bool = False,
                             axis: Optional[Union[int, Sequence[int]]] = None,
                             keepdims: bool = False) -> np.ndarray:
    """NumPy implementation of categorical crossentropy."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_arr = tensor.convert_to_tensor(data=y_true)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)

    if from_logits:
        # Apply softmax if input is logits
        from ember_ml.backend.numpy import math_ops # Lazy import
        y_pred_arr = math_ops.softmax(y_pred_arr, axis=-1) # Assuming softmax exists

    # Clip predictions for numerical stability
    y_pred_arr = np.clip(y_pred_arr, EPSILON, 1.0 - EPSILON)

    # Sum over the class axis (assumed to be the last axis)
    cce = -np.sum(y_true_arr * np.log(y_pred_arr), axis=-1)

    # Mean over remaining axes
    if axis is not None:
         # Adjust axis if the last dim was summed over
         if isinstance(axis, int) and axis < 0:
             adjusted_axis = axis + 1 if axis < -1 else None
         elif isinstance(axis, Sequence):
             adjusted_axis = [a + 1 if a < -1 else None for a in axis] # Needs careful adjustment based on final shape
             adjusted_axis = [a for a in adjusted_axis if a is not None]
             if not adjusted_axis: adjusted_axis = None
         else:
             adjusted_axis = axis
         # Need to handle case where adjusted_axis becomes None after adjustment
         if adjusted_axis is None:
              return np.mean(cce, keepdims=keepdims)
         else:
              return np.mean(cce, axis=adjusted_axis, keepdims=keepdims)
    else:
         return np.mean(cce, keepdims=keepdims)

def sparse_categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                                    from_logits: bool = False,
                                    axis: Optional[Union[int, Sequence[int]]] = None,
                                    keepdims: bool = False) -> np.ndarray:
    """NumPy implementation of sparse categorical crossentropy."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_int = tensor.convert_to_tensor(data=y_true).astype(default_int)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)
    num_classes = y_pred_arr.shape[-1]

    # Convert y_true to one-hot
    y_true_one_hot = np.eye(num_classes, dtype=y_pred_arr.dtype)[y_true_int]
    # Ensure shape matches y_pred if y_true has fewer dimensions
    if y_true_one_hot.ndim < y_pred_arr.ndim:
        # This reshape might be too simplistic if shapes differ in more than just the class dim
        try:
            y_true_one_hot = y_true_one_hot.reshape(y_pred_arr.shape)
        except ValueError:
             raise ValueError(f"Cannot reshape one-hot y_true {y_true_one_hot.shape} to match y_pred {y_pred_arr.shape}. Check input dimensions.")

    # Reuse categorical crossentropy implementation
    return categorical_crossentropy(y_true_one_hot, y_pred_arr, from_logits, axis, keepdims)

def huber_loss(y_true: TensorLike, y_pred: TensorLike, delta: float = 1.0,
               axis: Optional[Union[int, Sequence[int]]] = None,
               keepdims: bool = False) -> np.ndarray:
    """NumPy implementation of Huber loss."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_arr = tensor.convert_to_tensor(data=y_true)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)
    error = y_pred_arr - y_true_arr
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * np.square(quadratic) + delta * linear
    return np.mean(loss, axis=axis, keepdims=keepdims)

def log_cosh_loss(y_true: TensorLike, y_pred: TensorLike,
                  axis: Optional[Union[int, Sequence[int]]] = None,
                  keepdims: bool = False) -> np.ndarray:
    """NumPy implementation of log-cosh loss."""
    from ember_ml.backend.numpy.tensor import NumpyTensor # Lazy load
    tensor = NumpyTensor()
    y_true_arr = tensor.convert_to_tensor(data=y_true)
    y_pred_arr = tensor.convert_to_tensor(data=y_pred)
    error = y_pred_arr - y_true_arr
    # log(cosh(x)) = log((e^x + e^-x)/2) = log(e^x + e^-x) - log(2)
    logcosh = np.logaddexp(error, -error) - np.log(2.0)
    return np.mean(logcosh, axis=axis, keepdims=keepdims)

# Removed NumpyLossOps class