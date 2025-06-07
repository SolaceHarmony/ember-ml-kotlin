"""PyTorch tensor manipulation operations."""

from typing import Union, Optional, Sequence, Any, List

import torch
import torch.nn.functional as F

# Type aliases
Shape = Sequence[int]
TensorLike = Any

def reshape(data: TensorLike, shape: Shape) -> torch.Tensor:
    """
    Reshape a tensor.
    
    Args:
        data: The tensor to reshape
        shape: The new shape
        
    Returns:
        Reshaped tensor
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    return tensor.reshape(shape)

def transpose(data: TensorLike, axes: Optional[List[int]] = None) -> torch.Tensor:
    """
    Transpose a tensor.
    
    Args:
        data: The tensor to transpose
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    
    if axes is None:
        if tensor.dim() <= 2:
            return tensor.t()
        else:
            # Reverse the dimensions for default transpose
            axes = list(reversed(range(tensor.dim())))
    return torch.transpose(tensor, *axes)

def concatenate(data: List[TensorLike], axis: int = 0) -> torch.Tensor:
    """
    Concatenate tensors along a specified axis.
    
    Args:
        data: The tensors to concatenate
        axis: The axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    # Convert to PyTorch tensors
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    torch_tensors = [tensor_ops.convert_to_tensor(t) for t in data]
    return torch.cat(torch_tensors, dim=axis)

def vstack(data: List[TensorLike]) -> torch.Tensor:
    """
    Stack arrays vertically (row wise).
    
    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape (N,) have been reshaped to (1,N). Rebuilds arrays divided by vsplit.
    
    Args:
        data: The tensors to stack
        
    Returns:
        Stacked tensor
    """
    # Convert to PyTorch tensors
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    torch_tensors = []
    
    for t in data:
        tensor = tensor_ops.convert_to_tensor(t)
        # If 1D tensor, reshape to (1, N)
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, -1)
        torch_tensors.append(tensor)
    
    return torch.cat(torch_tensors, dim=0)

def hstack(data: List[TensorLike]) -> torch.Tensor:
    """
    Stack arrays horizontally (column wise).
    
    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided by hsplit.
    
    Args:
        data: The tensors to stack
        
    Returns:
        Stacked tensor
    """
    # Convert to PyTorch tensors
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    torch_tensors = [tensor_ops.convert_to_tensor(t) for t in data]
    
    # Check if tensors are 1D
    if all(t.dim() == 1 for t in torch_tensors):
        # For 1D tensors, concatenate along axis 0
        return torch.cat(torch_tensors, dim=0)
    else:
        # For nD tensors, concatenate along axis 1
        return torch.cat(torch_tensors, dim=1)

def stack(data: List[TensorLike], axis: int = 0) -> torch.Tensor:
    """
    Stack tensors along a new axis.
    
    Args:
        data: The tensors to stack
        axis: The axis along which to stack
        
    Returns:
        Stacked tensor
    """
    # Convert to PyTorch tensors
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    torch_tensors = [tensor_ops.convert_to_tensor(t) for t in data]
    return torch.stack(torch_tensors, dim=axis)

def split(data: TensorLike, num_or_size_splits: Union[int, List[int]], axis: int = 0) -> List[torch.Tensor]:
    """
    Split a tensor into sub-tensors.
    
    Args:
        data: The tensor to split
        num_or_size_splits: Number of splits or sizes of each split
        axis: The axis along which to split
        
    Returns:
        List of sub-tensors
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    
    if isinstance(num_or_size_splits, int):
        # Get the size of the dimension we're splitting along
        dim_size = tensor.shape[axis]
        
        # Check if the dimension is evenly divisible by num_or_size_splits
        # Use torch.remainder instead of % operator
        is_divisible = torch.eq(torch.remainder(torch.tensor(dim_size), torch.tensor(num_or_size_splits)), torch.tensor(0))
        
        if is_divisible:
            # If evenly divisible, use a simple split
            # Use torch.div instead of // operator
            split_size = torch.div(torch.tensor(dim_size), torch.tensor(num_or_size_splits), rounding_mode='trunc')
            # Convert to int to avoid type error
            split_size_int = int(split_size.item())
            return list(torch.split(tensor, split_size_int, dim=axis))
        else:
            # If not evenly divisible, create a list of split sizes
            # Use torch.div instead of // operator
            base_size = torch.div(torch.tensor(dim_size), torch.tensor(num_or_size_splits), rounding_mode='trunc')
            # Use torch.remainder instead of % operator
            remainder = torch.remainder(torch.tensor(dim_size), torch.tensor(num_or_size_splits))
            
            # Create a list where the first 'remainder' chunks have size 'base_size + 1'
            # and the rest have size 'base_size'
            split_sizes = []
            for i in range(num_or_size_splits):
                if i < remainder.item():
                    # Use torch.add instead of + operator
                    split_sizes.append(torch.add(base_size, torch.tensor(1)).item())
                else:
                    split_sizes.append(base_size.item())
            
            return list(torch.split(tensor, split_sizes, dim=axis))
    
    return list(torch.split(tensor, num_or_size_splits, dim=axis))

def expand_dims(data: TensorLike, axis: Union[int, List[int]]) -> torch.Tensor:
    """
    Insert new axes into a tensor's shape.
    
    Args:
        data: The tensor to expand
        axis: The axis or axes at which to insert the new dimension(s)
        
    Returns:
        Expanded tensor
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    
    if isinstance(axis, int):
        return tensor.unsqueeze(axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis):
        result = result.unsqueeze(ax)
    return result

def squeeze(data: TensorLike, axis: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        data: The tensor to squeeze
        axis: The axis or axes to remove
        
    Returns:
        Squeezed tensor
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    
    if axis is None:
        return tensor.squeeze()
    
    if isinstance(axis, int):
        return tensor.squeeze(axis)
    
    # Handle multiple axes
    result = tensor
    for ax in sorted(axis, reverse=True):
        result = result.squeeze(ax)
    return result

def tile(data: TensorLike, reps: List[int]) -> torch.Tensor:
    """
    Construct a tensor by tiling a given tensor.
    
    Args:
        data: Input tensor
        reps: Number of repetitions along each dimension
        
    Returns:
        Tiled tensor
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    return tensor.repeat(tuple(reps))

def pad(data: TensorLike, paddings: List[List[int]], mode: str = 'constant', constant_values: int = 0) -> torch.Tensor:
    """
    Pad a tensor with a constant value.
    
    Args:
        data: Input tensor
        paddings: List of lists of integers specifying the padding for each dimension
                Each inner list should contain two integers: [pad_before, pad_after]
        mode: Padding mode. PyTorch supports 'constant', 'reflect', 'replicate', and 'circular'.
               Default is 'constant'.
        constant_values: Value to pad with when mode is 'constant'
        
    Returns:
        Padded tensor
    """
    # Convert to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    
    # Convert paddings to the format expected by torch.nn.functional.pad
    # PyTorch expects (pad_left, pad_right, pad_top, pad_bottom, ...)
    # We need to reverse the order and flatten
    pad_list = []
    for pad_pair in reversed(paddings):
        pad_list.extend(pad_pair)
    
    # Pad the tensor
    return F.pad(tensor, pad_list, mode=mode, value=constant_values)

__all__ = [
    "reshape",
    "transpose",
    "concatenate",
    "vstack",
    "hstack",
    "stack",
    "split",
    "expand_dims",
    "squeeze",
    "tile",
    "pad",
]