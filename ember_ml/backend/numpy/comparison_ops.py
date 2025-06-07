"""
NumPy implementation of comparison operations.

This module provides NumPy implementations of comparison operations.
"""

import numpy as np
from typing import Any
from ember_ml.backend.numpy.types import TensorLike


def equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if two tensors are equal element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x == y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def not_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if two tensors are not equal element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x != y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.not_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def less(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is less than another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x < y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.less(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def less_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is less than or equal to another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x <= y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.less_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def greater(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is greater than another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x > y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.greater(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def greater_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Check if one tensor is greater than or equal to another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x >= y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.greater_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def logical_and(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical AND of two tensors element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x AND y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_and(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def logical_or(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical OR of two tensors element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x OR y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_or(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def logical_not(x: Any) -> np.ndarray:
    """
    Compute the logical NOT of a tensor element-wise.

    Args:
        x: Input tensor

    Returns:
        Boolean NumPy array with True where NOT x
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_not(Tensor.convert_to_tensor(x))


def logical_xor(x: Any, y: Any) -> np.ndarray:
    """
    Compute the logical XOR of two tensors element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean NumPy array with True where x XOR y
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.logical_xor(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))


def allclose(x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if all elements of two tensors are close within a tolerance.

    Args:
        x: First tensor
        y: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Boolean indicating if all elements are close
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.allclose(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y), rtol=rtol, atol=atol)

def isclose(x: Any, y: Any, rtol: float = 1e-5, atol: float = 1e-8) -> np.ndarray:
    """
    Check if elements of two tensors are close within a tolerance element-wise.

    Args:
        x: First tensor
        y: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Boolean NumPy array with True where elements are close
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.isclose(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y), rtol=rtol, atol=atol)


def all(x: Any, axis: Any = None) -> Any:
    """
    Check if all elements in a tensor are True.

    Args:
        x: Input tensor
        axis: Axis or axes along which to perform the reduction.
            If None, reduce over all dimensions.

    Returns:
        Boolean tensor with True if all elements are True, False otherwise.
        If axis is specified, the result is a tensor with the specified
        axes reduced.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.all(Tensor.convert_to_tensor(x), axis=axis)


def any(x: Any, axis: Any = None) -> Any:
    """
    Check if any elements in a tensor are True.

    Args:
        x: Input tensor
        axis: Axis or axes along which to perform the reduction.
            If None, reduce over all dimensions.

    Returns:
        Boolean tensor with True if any elements are True, False otherwise.
        If axis is specified, the result is a tensor with the specified
        axes reduced.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.any(Tensor.convert_to_tensor(x), axis=axis)


def where(condition: TensorLike, x: TensorLike, y: TensorLike) -> np.ndarray:
    """
    Return elements chosen from x or y depending on condition.

    Args:
        condition: Boolean tensor
        x: Tensor with values to use where condition is True
        y: Tensor with values to use where condition is False

    Returns:
        Tensor with values from x where condition is True, and values from y elsewhere
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    condition_tensor = Tensor.convert_to_tensor(condition)
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    return np.where(condition_tensor, x_tensor, y_tensor)


def isnan(x: TensorLike) -> np.ndarray:
    """
    Test element-wise for NaN values.

    Args:
        x: Input tensor

    Returns:
        Boolean tensor with True where x is NaN, False otherwise
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return np.isnan(Tensor.convert_to_tensor(x))


# Removed NumpyComparisonOps class as it's redundant with standalone functions
