"""
PyTorch math operations for ember_ml.

This module provides PyTorch implementations of math operations.
"""

import torch
from typing import Union, Optional, List, Tuple, Any

from ember_ml.backend.torch.types import TensorLike, ShapeLike

# We avoid creating global instances to prevent circular imports
# Each function will create its own instances when needed

def gather(x: TensorLike, indices: TensorLike, axis: int = 0) -> Any: # Changed torch.Tensor to Any
    """
    Gather slices from x along the specified axis according to indices.
    
    Args:
        x: The input array from which to gather values
        indices: The indices of the values to extract
        axis: The axis along which to index (default: 0)
        
    Returns:
        Array of gathered values with the same type as x
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_array = tensor_ops.convert_to_tensor(x)
    indices_array = tensor_ops.convert_to_tensor(indices)
    
    # Ensure indices are integer type
    # Check if the dtype contains 'int' in its name
    if isinstance(indices_array, torch.Tensor) and indices_array.dtype != torch.int64:
        # If not, cast to int64
        indices_array = indices_array.to(torch.int64)
    
    # Use torch.index_select to gather values along the specified axis
    return torch.index_select(x_array, dim=axis, index=indices_array)

def add(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Add two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise sum
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    return torch.add(tensor_ops.convert_to_tensor(x), tensor_ops.convert_to_tensor(y))


def subtract(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Subtract two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise difference
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.subtract(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def multiply(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Multiply two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Element-wise product
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.mul(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def divide(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Divide two tensors element-wise.
    
    Args:
        x: First tensor (numerator)
        y: Second tensor (denominator)
        
    Returns:
        Element-wise quotient
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    return torch.div(tensor.convert_to_tensor(x), tensor.convert_to_tensor(y))


def dot(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the dot product of two PyTorch tensors.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Dot product
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_array = tensor_ops.convert_to_tensor(x)
    y_array = tensor_ops.convert_to_tensor(y)
    
    # Handle different dimensions
    if torch.equal(torch.tensor(len(x_array.shape)), torch.tensor(1)) and torch.equal(torch.tensor(len(y_array.shape)), torch.tensor(1)):
        return torch.sum(torch.multiply(x_array, y_array))
    else:
        return torch.matmul(x_array, y_array)


def matmul(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Multiply two tensors as matrices.

    Args:
        x: First array
        y: Second array

    Returns:
        Matrix product
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    
    # Convert inputs to tensors
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    
    # Check if we're on MPS device and convert non-float inputs to float32
    if str(x_tensor.device).startswith('mps') and not torch.is_floating_point(x_tensor):
        x_tensor = x_tensor.to(dtype=torch.float32)
    if str(y_tensor.device).startswith('mps') and not torch.is_floating_point(y_tensor):
        y_tensor = y_tensor.to(dtype=torch.float32)
    
    return torch.matmul(x_tensor, y_tensor)


def mean(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> Any: # Changed torch.Tensor to Any
    """
    Compute mean of tensor elements along specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute mean.
            If None, compute mean over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Mean of tensor elements
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    # Ensure float dtype for mean calculation if input is integer
    if not x_tensor.dtype.is_floating_point and not x_tensor.dtype.is_complex:
        x_tensor = x_tensor.to(torch.float32) # Cast to float32

    if axis is None:
        # torch.mean doesn't accept keepdim when axis is None (mean over all elements)
        result = torch.mean(x_tensor)
        # If keepdims is True, we need to manually reshape the scalar result
        if keepdims:
            # Create a shape of all ones with the same ndim as input
            target_shape = (1,) * x_tensor.ndim
            return result.reshape(target_shape)
        else:
            return result
    elif isinstance(axis, tuple):
        result = x_tensor
        for dim in sorted(axis, reverse=True):
            result = torch.mean(result, dim=dim, keepdim=keepdims)
        return result
    else:
        return torch.mean(x_tensor, dim=axis, keepdim=keepdims)


def sum(x: TensorLike, axis: Optional[ShapeLike] = None, keepdim: bool = False) -> Any: # Changed torch.Tensor to Any
    """
    Compute sum of tensor elements along specified axis.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute sum.
            If None, compute sum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Sum of tensor elements
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        return torch.sum(x_tensor)
    elif isinstance(axis, tuple):
        # Sort axes in reverse order to avoid reshaping issues
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for dim in sorted(axis, reverse=True):
            result = torch.sum(result, dim=dim, keepdim=keepdim)
        return result
    else:
        return torch.sum(x_tensor, dim=axis, keepdim=keepdim)

def var(x: TensorLike, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Any: # Changed torch.Tensor to Any
    """
    Compute variance of tensor elements along specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute variance.
            If None, compute variance over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Variance of tensor elements
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        result = torch.var(x_tensor, unbiased=False)
        # Handle keepdims for None axis
        if keepdim:
            # Add back all dimensions as size 1
            for _ in range(x_tensor.dim()):
                result = result.unsqueeze(0)
        return result
    
    elif isinstance(axis, tuple):
        # Sort axes in reverse order to avoid reshaping issues
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.var(result, dim=ax, unbiased=False, keepdim=keepdim)
        return result
    else:
        return torch.var(x_tensor, dim=axis, unbiased=False, keepdim=keepdim)


def exp(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute exponential of all elements in the input tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Exponential of each element in the input tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.exp(TorchTensor().convert_to_tensor(x))


def log(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute natural logarithm of all elements in the input tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Natural logarithm of each element in the input tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.log(TorchTensor().convert_to_tensor(x))


def pow(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute x raised to the power of y for all elements.
    
    Args:
        x: Base tensor
        y: Exponent tensor or scalar
        
    Returns:
        Tensor with elements of x raised to the power of y
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    return torch.pow(x_tensor, y_tensor)


def sqrt(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the square root of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Square root of each element in the input tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sqrt(TorchTensor().convert_to_tensor(x))


def clip(x: TensorLike, 
         min_val: TensorLike, 
         max_val: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Clip tensor elements to a specified range.
    
    Args:
        x: Input tensor
        min_val: Minimum value for clipping, can be None for no lower bound
        max_val: Maximum value for clipping, can be None for no upper bound
        
    Returns:
        Tensor with clipped values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_array = tensor.convert_to_tensor(x)
    min_val = tensor.convert_to_tensor(min_val)
    max_val = tensor.convert_to_tensor(max_val)

    return torch.clamp(x_array, min=min_val, max=max_val)


def abs(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute absolute value of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with absolute values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.abs(TorchTensor().convert_to_tensor(x))


def negative(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the negative of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with negated values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.negative(TorchTensor().convert_to_tensor(x))


def sign(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the sign of tensor elements.
    
    Returns -1 for negative values, 0 for zero, and 1 for positive values.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sign values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sign(TorchTensor().convert_to_tensor(x))




def argmax(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> Any: # Changed torch.Tensor to Any
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to find the indices of maximum values.
            If None, the index is for the flattened tensor.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Indices of maximum values along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        result = torch.argmax(x_tensor.flatten())
        # For None axis with keepdims=True, we need a scalar tensor with shape (1,)
        if keepdims:
            return result.reshape(1)
        return result
    else:
        result = torch.argmax(x_tensor, dim=axis)
        # Handle keepdims by unsqueezing the reduced dimension if needed
        if keepdims:
            return result.unsqueeze(dim=axis)
        return result


def sin(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute sine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sin(TorchTensor().convert_to_tensor(x))


def cos(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute cosine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with cosine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.cos(TorchTensor().convert_to_tensor(x))


def tan(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute tangent of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with tangent of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.tan(TorchTensor().convert_to_tensor(x))




def sinh(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute hyperbolic sine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with hyperbolic sine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sinh(TorchTensor().convert_to_tensor(x))


def cosh(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute hyperbolic cosine of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with hyperbolic cosine of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.cosh(TorchTensor().convert_to_tensor(x))



def log10(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute base-10 logarithm of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with base-10 logarithm of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.log10(TorchTensor().convert_to_tensor(x))




def log2(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute base-2 logarithm of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with base-2 logarithm of values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.log2(TorchTensor().convert_to_tensor(x))


def square(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute square of tensor elements.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with squared values
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.square(TorchTensor().convert_to_tensor(x))

def mod(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute element-wise remainder of division.
    
    Args:
        x: Input tensor (dividend)
        y: Input tensor (divisor)
        
    Returns:
        Tensor with element-wise remainder
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    
    return torch.remainder(x_tensor, y_tensor)

def floor_divide(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute element-wise integer division.
    
    Args:
        x: Input tensor (dividend)
        y: Input tensor (divisor)
        
    Returns:
        Tensor with element-wise integer division result
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()

    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)

    # Use floor_divide to perform integer division    
    return torch.floor_divide(x_tensor, y_tensor)


def floor(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Return the floor of the input, element-wise.
    
    The floor of the scalar x is the largest integer i, such that i <= x.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise floor of the input
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    # Use floor from PyTorch
    return torch.floor(x_tensor)


def ceil(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Return the ceiling of the input, element-wise.
    
    The ceiling of the scalar x is the smallest integer i, such that i >= x.
    
    Args:
        x: Input tensor
        
    Returns:
        Element-wise ceiling of the input
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    # Use ceil from PyTorch
    return torch.ceil(x_tensor)


def min(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> Any: # Changed torch.Tensor to Any
    """
    Compute minimum of tensor elements along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to find minimum.
            If None, find minimum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Minimum values along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        # Find the minimum
        return torch.min(x_tensor)
    return torch.min(x_tensor, dim=axis, keepdim=keepdims).values


def max(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False) -> Any: # Changed torch.Tensor to Any
    """
    Compute maximum of tensor elements along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to find maximum.
            If None, find maximum over all elements.
        keepdims: Whether the output tensor has dim retained or not.
        
    Returns:
        Maximum values along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)

    if axis is None:
        return torch.max(x_tensor)
    return torch.max(x_tensor, dim=axis, keepdim=keepdims).values

# Removed softmax function definition


def sort(x: TensorLike, axis: int = -1) -> Any: # Changed torch.Tensor to Any
    """
    Sort tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort (default: -1)
        
    Returns:
        Sorted tensor
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    return torch.sort(TorchTensor().convert_to_tensor(x), dim=axis).values 


def gradient(f: TensorLike, *varargs, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             edge_order: int = 1) -> Union[Any, List[Any]]: # Changed torch.Tensor to Any
    """
    Return the gradient of an N-dimensional tensor.
    
    The gradient is computed using finite differences. This implementation
    approximates numpy.gradient using PyTorch operations.
    
    Args:
        f: An N-dimensional tensor containing samples of a scalar function.
        *varargs: Spacing between f values. Default unitary spacing for all dimensions.
        axis: Gradient is calculated only along the given axis or axes.
            The default (axis = None) is to calculate the gradient for all the axes.
        edge_order: Gradient is calculated using N-th order accurate differences at the boundaries.
            Must be 1 or 2.
            
    Returns:
        A tensor or tuple of tensors corresponding to the derivatives of f with respect to each dimension.
        Each derivative has the same shape as f.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    f_array = tensor_ops.convert_to_tensor(f)

    # Get the shape of the input tensor
    f_shape = f_array.shape
    ndim = len(f_shape)

    # Determine the axes along which to compute the gradient
    if axis is None:
        axes = tuple(range(f_array.dim()))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = axis
        
    # Initialize spacing for each dimension
    spacings = []
    if len(varargs) == 0:
        # Default: unitary spacing for all dimensions
        spacings = [torch.tensor(1.0)] * len(axes)
    elif len(varargs) == 1:
        # Same spacing for all dimensions
        spacings = [tensor_ops.convert_to_tensor(varargs[0])] * len(axes)
    else:
        # Different spacing for each dimension
        if len(varargs) != len(axes):
            raise ValueError("If spacing is specified for each axis, the number of "
                            "spacing values must match the number of axes.")
        spacings = [tensor_ops.convert_to_tensor(spacing) for spacing in varargs]
    
    # Compute the gradient along each specified axis
    result = []
    
    for i, axis_i in enumerate(axes):
        # Get the spacing for this axis
        dx = spacings[i]
        
        # Create slices for forward and backward differences
        slice_prev = [slice(None)] * f_array.dim()
        slice_next = [slice(None)] * f_array.dim()
        slice_center = [slice(None)] * f_array.dim()
        
        # Compute the gradient using finite differences
        if edge_order == 1:
            # Forward difference at the beginning
            slice_prev[axis_i] = slice(0, 1)
            slice_next[axis_i] = slice(1, 2)
            
            # Use torch operations instead of Python operators
            forward_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), 
                dx
            )
            
            # Backward difference at the end
            slice_prev[axis_i] = slice(-2, -1)
            slice_next[axis_i] = slice(-1, None)
            
            # Use torch operations instead of Python operators
            backward_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), 
                dx
            )
            
            # Central difference in the middle
            slice_prev[axis_i] = slice(0, -2)
            slice_center[axis_i] = slice(1, -1)
            slice_next[axis_i] = slice(2, None)
            
            # Use torch operations instead of Python operators
            central_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]), 
                torch.multiply(torch.tensor(2.0), dx)
            )
            
            # Combine the differences
            grad = torch.zeros_like(f_array)
            
            # Assign the forward difference at the beginning
            slice_center[axis_i] = slice(0, 1)
            grad[tuple(slice_center)] = forward_diff
            
            # Assign the central difference in the middle
            slice_center[axis_i] = slice(1, -1)
            grad[tuple(slice_center)] = central_diff
            
            # Assign the backward difference at the end
            slice_center[axis_i] = slice(-1, None)
            grad[tuple(slice_center)] = backward_diff
            
        elif edge_order == 2:
            # Second-order accurate differences
            # For simplicity, we'll implement a basic version here
            # A more accurate implementation would use higher-order finite differences
            
            # Central difference for interior points
            slice_prev[axis_i] = slice(0, -2)
            slice_center[axis_i] = slice(1, -1)
            slice_next[axis_i] = slice(2, None)
            
            # Use torch operations instead of Python operators
            central_diff = torch.div(
                torch.subtract(f_array[tuple(slice_next)], f_array[tuple(slice_prev)]),
                torch.multiply(torch.tensor(2.0), dx)
            )
            
            # Second-order accurate differences at the boundaries
            # For the beginning
            slice_0 = [slice(None)] * f_array.dim()
            slice_1 = [slice(None)] * f_array.dim()
            slice_2 = [slice(None)] * f_array.dim()
            
            slice_0[axis_i] = slice(0, 1)
            slice_1[axis_i] = slice(1, 2)
            slice_2[axis_i] = slice(2, 3)
            
            if f_shape[axis_i] > 2:
                # Use torch operations instead of Python operators
                term1 = torch.multiply(torch.tensor(-3.0), f_array[tuple(slice_0)])
                term2 = torch.multiply(torch.tensor(4.0), f_array[tuple(slice_1)])
                term3 = torch.negative(f_array[tuple(slice_2)])
                
                sum_terms = torch.add(torch.add(term1, term2), term3)
                forward_diff = torch.div(sum_terms, torch.multiply(torch.tensor(2.0), dx))
            else:
                # Use torch operations instead of Python operators
                forward_diff = torch.div(
                    torch.subtract(f_array[tuple(slice_1)], f_array[tuple(slice_0)]),
                    dx
                )
            
            # For the end
            slice_n2 = [slice(None)] * f_array.dim()
            slice_n1 = [slice(None)] * f_array.dim()
            slice_n = [slice(None)] * f_array.dim()
            
            slice_n2[axis_i] = slice(-3, -2)
            slice_n1[axis_i] = slice(-2, -1)
            slice_n[axis_i] = slice(-1, None)
            
            if f_shape[axis_i] > 2:
                # Use torch operations instead of Python operators
                term1 = torch.multiply(torch.tensor(3.0), f_array[tuple(slice_n)])
                term2 = torch.multiply(torch.tensor(-4.0), f_array[tuple(slice_n1)])
                term3 = f_array[tuple(slice_n2)]
                
                sum_terms = torch.add(torch.add(term1, term2), term3)
                backward_diff = torch.div(sum_terms, torch.multiply(torch.tensor(2.0), dx))
            else:
                # Use torch operations instead of Python operators
                backward_diff = torch.div(
                    torch.subtract(f_array[tuple(slice_n)], f_array[tuple(slice_n1)]),
                    dx
                )
            
            # Combine the differences
            grad = torch.zeros_like(f_array)
            
            # Assign the forward difference at the beginning
            slice_center[axis_i] = slice(0, 1)
            grad[tuple(slice_center)] = forward_diff
            
            # Assign the central difference in the middle
            slice_center[axis_i] = slice(1, -1)
            grad[tuple(slice_center)] = central_diff
            
            # Assign the backward difference at the end
            slice_center[axis_i] = slice(-1, None)
            grad[tuple(slice_center)] = backward_diff
            
        else:
            raise ValueError("Edge order must be 1 or 2.")
        
        result.append(grad)
    
    # Return a single tensor if only one axis was specified
    if len(result) == 1:
        return result[0]
    else:
        return result
    

def cumsum(x: TensorLike, axis: Optional[int] = None) -> Any: # Changed torch.Tensor to Any
    """
    Compute the cumulative sum of tensor elements along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the cumulative sum. If None, the
            flattened tensor's cumulative sum is returned.
        
    Returns:
        Tensor with cumulative sum along the specified axis
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    x_tensor = TorchTensor().convert_to_tensor(x)
    
    if axis is None:
        return torch.cumsum(x_tensor.flatten(), dim=0)
    else:
        return torch.cumsum(x_tensor, dim=axis)


def eigh(a: TensorLike) -> Tuple[Any, Any]: # Changed torch.Tensor to Any
    """
    Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    
    Args:
        a: Input Hermitian or symmetric matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    a_tensor = TorchTensor().convert_to_tensor(a)
    return torch.linalg.eigh(a_tensor)


# Define the pi constant using Chudnovsky algorithm
def _calculate_pi_value(precision_digits=15) -> Any: # Changed torch.Tensor to Any
    """
    Calculate pi using the Chudnovsky algorithm.
    
    The Chudnovsky algorithm is one of the most efficient algorithms for calculating π,
    with a time complexity of O(n log(n)^3). It converges much faster than other series.
    
    Formula:
    1/π = (12/426880√10005) * Σ (6k)!(13591409 + 545140134k) / ((3k)!(k!)^3 * (-640320)^(3k))
    
    Args:
        precision_digits: Number of decimal places to calculate
        
    Returns:
        Value of pi with the specified precision
    """
    import math
    # Check if torch and necessary attributes are available
    if not hasattr(torch, 'as_tensor') or not hasattr(torch, 'tensor') or not hasattr(torch, 'divide'): # Add more checks if needed
        # Fallback to math.pi if torch operations are not available
        # This ensures the function can run even if torch is not fully functional,
        # especially when the torch backend is not the active one.
        print("Warning: PyTorch tensor operations for PI calculation are not available. Falling back to math.pi.")
        # Return as a tensor-like shape if possible, or just the float
        try:
            # Try to return in a way that downstream code might expect (e.g., a 1-element tensor)
            return torch.tensor([math.pi], dtype=torch.float32) # Try torch.tensor first
        except AttributeError: # If torch.tensor is not available
            try:
                return torch.Tensor([math.pi]).to(torch.float32) # Try torch.Tensor as another fallback
            except AttributeError: # If torch.Tensor is also not available
                return math.pi # Raw float as last resort
        except Exception: # Broad exception if all torch tensor creation fails
            return math.pi # Raw float as last resort


    # Get the default device for the current backend inside the function
    from ember_ml.backend.torch.device_ops import get_device # Use backend's get_device
    device = get_device()

    # Constants in the Chudnovsky algorithm (on the correct device)
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    C = torch.as_tensor(640320, device=device)
    C3_OVER_24 = torch.divide(torch.pow(C, 3), torch.as_tensor(24, device=device))
    DIGITS_PER_TERM = torch.as_tensor(14.1816474627254776555, device=device)  # Approx. digits per iteration

    def binary_split(a, b):
        """Recursive binary split for the Chudnovsky algorithm."""
        from ember_ml.backend.torch.tensor import TorchTensor
        # Ensure tensors created here are on the correct device
        inner_tensor = TorchTensor()
        a_tensor = inner_tensor.convert_to_tensor(a) # convert_to_tensor handles device
        b_tensor = inner_tensor.convert_to_tensor(b) # convert_to_tensor handles device
        diff = torch.subtract(b_tensor, a_tensor)

        # Ensure comparisons happen on the same device
        one_tensor = torch.as_tensor(1, device=device) # Changed torch.tensor to torch.as_tensor
        zero_tensor = torch.as_tensor(0, device=device) # Changed torch.tensor to torch.as_tensor
        two_tensor = torch.as_tensor(2, device=device) # Changed torch.tensor to torch.as_tensor
        five_tensor = torch.as_tensor(5, device=device) # Changed torch.tensor to torch.as_tensor
        six_tensor = torch.as_tensor(6, device=device) # Changed torch.tensor to torch.as_tensor
        base_term_tensor = torch.as_tensor(13591409, device=device) # Changed torch.tensor to torch.as_tensor
        multiplier_tensor = torch.as_tensor(545140134, device=device) # Changed torch.tensor to torch.as_tensor

        if torch.equal(diff, one_tensor):
            # Base case
            if torch.equal(a_tensor, zero_tensor):
                Pab = torch.as_tensor(1, device=device) # Changed torch.tensor to torch.as_tensor
                Qab = torch.as_tensor(1, device=device) # Changed torch.tensor to torch.as_tensor
            else:
                term1 = torch.subtract(torch.multiply(six_tensor, a_tensor), five_tensor)
                term2 = torch.subtract(torch.multiply(two_tensor, a_tensor), one_tensor)
                term3 = torch.subtract(torch.multiply(six_tensor, a_tensor), one_tensor)
                Pab = torch.multiply(torch.multiply(term1, term2), term3)
                Qab = torch.multiply(torch.pow(a_tensor, 3), C3_OVER_24) # C3_OVER_24 is already on device

            term = torch.add(base_term_tensor, torch.multiply(multiplier_tensor, a_tensor))
            Tab = torch.multiply(Pab, term)

            # Check if a is odd using remainder comparison
            remainder = torch.remainder(a_tensor, two_tensor)
            is_odd = torch.eq(remainder, one_tensor)

            # If a is odd, negate Tab
            Tab = torch.where(is_odd, torch.negative(Tab), Tab)

            return Pab, Qab, Tab
        
        # Recursive case
        m = torch.divide(torch.add(a_tensor, b_tensor), two_tensor) # Use two_tensor
        m = torch.floor(m)  # Ensure m is an integer
        
        Pam, Qam, Tam = binary_split(a, m)
        Pmb, Qmb, Tmb = binary_split(m, b)
        
        Pab = torch.multiply(Pam, Pmb)
        Qab = torch.multiply(Qam, Qmb)
        term1 = torch.multiply(Qmb, Tam)
        term2 = torch.multiply(Pam, Tmb)
        Tab = torch.add(term1, term2)
        
        return Pab, Qab, Tab
    
    # Number of terms needed for the desired precision
    precision_tensor = tensor_ops.convert_to_tensor(precision_digits)
    terms_float = torch.divide(precision_tensor, DIGITS_PER_TERM) # DIGITS_PER_TERM is already on device
    terms_float = torch.add(terms_float, torch.as_tensor(1, device=device)) # Use device, Changed torch.tensor to torch.as_tensor
    terms = torch.floor(terms_float)  # Convert to integer
    terms_int = terms.to(torch.int32)  # Convert to int32 using PyTorch's to() method
    
    # Compute the binary split
    P, Q, T = binary_split(0, terms_int)
    
    # Calculate pi (ensure constants are on the correct device)
    sqrt_10005 = torch.sqrt(torch.as_tensor(10005, device=device)) # Changed torch.tensor to torch.as_tensor
    numerator = torch.multiply(Q, torch.as_tensor(426880, device=device)) # Changed torch.tensor to torch.as_tensor
    numerator = torch.multiply(numerator, sqrt_10005)
    pi_approx = torch.divide(numerator, T)
    
    # Return as PyTorch tensor with shape (1,)
    return torch.reshape(pi_approx, (1,))

# Calculate pi with appropriate precision for PyTorch (float32)
# Ensure it's a scalar with shape (1,) as per PyTorch conventions
PI_CONSTANT = _calculate_pi_value(15)  # Increased precision to match reference value

# Attempt to convert to torch.float32 if PI_CONSTANT is a tensor, otherwise assign as float
if hasattr(PI_CONSTANT, 'to') and hasattr(torch, 'float32'):
    pi : Any = PI_CONSTANT.to(torch.float32)
elif hasattr(torch, 'tensor') and hasattr(torch, 'float32'): # If it's a float, try to make it a tensor
    try:
        pi : Any = torch.tensor(PI_CONSTANT, dtype=torch.float32)
    except AttributeError: # If torch.tensor or torch.float32 is missing
        pi : Any = float(PI_CONSTANT) # Fallback to float
else: # If torch itself is missing attributes
    pi : Any = float(PI_CONSTANT)


def binary_split(a: TensorLike, b: TensorLike) -> Tuple[Any, Any]: # Changed torch.Tensor to Any
    """
    Recursive binary split for the Chudnovsky algorithm.
    
    This is used in the implementation of PI calculation for the PyTorch backend.
    
    Args:
        a: Start value
        b: End value
        
    Returns:
        Tuple of intermediate values for PI calculation
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    a_tensor = tensor.convert_to_tensor(a)
    b_tensor = tensor.convert_to_tensor(b)
    
    # Use torch operations
    diff = torch.subtract(b_tensor, a_tensor)
    
    if torch.equal(diff, torch.as_tensor(1.0, device=a_tensor.device)): # Changed torch.tensor to torch.as_tensor and ensure device consistency
        # Base case
        if torch.equal(a_tensor, torch.as_tensor(0.0, device=a_tensor.device)): # Changed torch.tensor to torch.as_tensor and ensure device consistency
            Pab = torch.as_tensor(1.0, device=a_tensor.device) # Changed torch.tensor to torch.as_tensor
            Qab = torch.as_tensor(1.0, device=a_tensor.device) # Changed torch.tensor to torch.as_tensor
        else:
            # Calculate terms using torch operations
            term1 = torch.subtract(torch.multiply(torch.as_tensor(6.0, device=a_tensor.device), a_tensor), torch.as_tensor(5.0, device=a_tensor.device)) # Changed torch.tensor to torch.as_tensor
            term2 = torch.subtract(torch.multiply(torch.as_tensor(2.0, device=a_tensor.device), a_tensor), torch.as_tensor(1.0, device=a_tensor.device)) # Changed torch.tensor to torch.as_tensor
            term3 = torch.subtract(torch.multiply(torch.as_tensor(6.0, device=a_tensor.device), a_tensor), torch.as_tensor(1.0, device=a_tensor.device)) # Changed torch.tensor to torch.as_tensor
            Pab = torch.multiply(torch.multiply(term1, term2), term3)
            
            # Define C3_OVER_24
            C = torch.as_tensor(640320.0, device=a_tensor.device) # Changed torch.tensor to torch.as_tensor
            C3_OVER_24 = torch.div(torch.pow(C, torch.as_tensor(3.0, device=a_tensor.device)), torch.as_tensor(24.0, device=a_tensor.device)) # Changed torch.tensor to torch.as_tensor
            
            Qab = torch.multiply(torch.pow(a_tensor, torch.as_tensor(3.0, device=a_tensor.device)), C3_OVER_24) # Changed torch.tensor to torch.as_tensor
        
        return Pab, Qab
    else:
        # Recursive case
        m = torch.div(torch.add(a_tensor, b_tensor), torch.as_tensor(2.0, device=a_tensor.device)) # Changed torch.tensor to torch.as_tensor
        Pam, Qam = binary_split(a_tensor, m)
        Pmb, Qmb = binary_split(m, b_tensor)
        
        Pab = torch.multiply(Pam, Pmb)
        Qab = torch.add(torch.multiply(Qam, Pmb), torch.multiply(Pam, Qmb))
        
        return Pab, Qab

# Removed TorchMathOps class as it's redundant with standalone functions
