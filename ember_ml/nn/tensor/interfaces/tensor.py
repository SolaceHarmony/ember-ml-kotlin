"""
Tensor interface definition.

This module defines the abstract interface for tensor operations that must be
implemented by all backend tensor implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Tuple, Sequence, Callable, Iterator

class TensorInterface(ABC):
    """Abstract interface for tensor operations."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the tensor."""
        pass

    @abstractmethod
    def __init__(
        self,
        data: Optional[Any] = None,
        *,
        dtype: Optional[Union[Any, str, Callable[[], Any]]] = None,
        device: Optional[str] = None,
        requires_grad: bool = False
    ) -> None:
        """
        Initialize a tensor.

        Args:
            data: Input data to create tensor from
            dtype: Optional dtype for the tensor (can be a DType, string, or callable)
            device: Optional device to place the tensor on
            requires_grad: Whether the tensor requires gradients
        """
        pass

    @abstractmethod
    def to_backend_tensor(self) -> Any:
        """
        Convert to the native backend tensor type.

        Returns:
            The native backend tensor (NumPy array, tensor.convert_to_tensor, mlx.array)
        """
        pass

    @abstractmethod
    def __array__(self) -> Any:
        """NumPy array interface."""
        pass
    
    @abstractmethod
    def __getitem__(self, key) -> Any:
        """
        Get values at specified indices.
        
        Args:
            key: Index or slice
            
        Returns:
            Tensor with values at specified indices
        """
        pass
    
    @abstractmethod
    def __setitem__(self, key, value) -> None:
        """
        Set values at specified indices.
        
        Args:
            key: Index or slice
            value: Value to set
        """
        pass

    @abstractmethod
    def item(self) -> Union[int, float, bool]:
        """Get the value of a scalar tensor."""
        pass

    @abstractmethod
    def tolist(self) -> List[Any]:
        """Convert tensor to a (nested) list."""
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """Get the dtype of the tensor."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device the tensor is on."""
        pass
    
    @property
    @abstractmethod
    def backend(self) -> str:
        """Get the backend the tensor is using."""
        pass

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        """Get whether the tensor requires gradients."""
        pass

    @abstractmethod
    def detach(self) -> Any:
        """Create a new tensor detached from the computation graph."""
        pass

    @abstractmethod
    def to_numpy(self) -> Any:
        """Convert tensor to NumPy array."""
        pass
    
    @abstractmethod
    def zeros(self, shape: Union[int, Sequence[int]], dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of zeros.
        
        Args:
            shape: Shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the specified shape
        """
        pass
    
    @abstractmethod
    def ones(self, shape: Union[int, Sequence[int]], dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of ones.
        
        Args:
            shape: Shape of the tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the specified shape
        """
        pass
    
    @abstractmethod
    def zeros_like(self, x: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of zeros with the same shape as the input.
        
        Args:
            x: Input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of zeros with the same shape as x
        """
        pass
    
    @abstractmethod
    def ones_like(self, x: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor of ones with the same shape as the input.
        
        Args:
            x: Input tensor
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor of ones with the same shape as x
        """
        pass
    
    @abstractmethod
    def eye(self, n: int, m: Optional[int] = None, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create an identity matrix.
        
        Args:
            n: Number of rows
            m: Number of columns (default: n)
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Identity matrix of shape (n, m)
        """
        pass
    
    @abstractmethod
    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (exclusive)
            step: Spacing between values
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with evenly spaced values
        """
        pass
    
    @abstractmethod
    def linspace(self, start: float, stop: float, num: int, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with evenly spaced values within a given interval.
        
        Args:
            start: Start of interval (inclusive)
            stop: End of interval (inclusive)
            num: Number of values to generate
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with evenly spaced values
        """
        pass
    
    @abstractmethod
    def full(self, shape: Union[int, Sequence[int]], fill_value: Union[float, int], dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor filled with a scalar value.
        
        Args:
            shape: Shape of the tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor filled with the specified value
        """
        pass
    
    @abstractmethod
    def full_like(self, x: Any, fill_value: Union[float, int], dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor filled with a scalar value with the same shape as the input.
        
        Args:
            x: Input tensor
            fill_value: Value to fill the tensor with
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor filled with the specified value with the same shape as x
        """
        pass
    
    @abstractmethod
    def reshape(self, x: Any, shape: Union[int, Sequence[int]]) -> Any:
        """
        Reshape a tensor to a new shape.
        
        Args:
            x: Input tensor
            shape: New shape
            
        Returns:
            Reshaped tensor
        """
        pass
    
    @abstractmethod
    def transpose(self, x: Any, axes: Optional[Sequence[int]] = None) -> Any:
        """
        Permute the dimensions of a tensor.
        
        Args:
            x: Input tensor
            axes: Optional permutation of dimensions
            
        Returns:
            Transposed tensor
        """
        pass
    
    @abstractmethod
    def concatenate(self, tensors: Sequence[Any], axis: int = 0) -> Any:
        """
        Concatenate tensors along a specified axis.
        
        Args:
            tensors: Sequence of tensors
            axis: Axis along which to concatenate
            
        Returns:
            Concatenated tensor
        """
        pass
    
    @abstractmethod
    def stack(self, tensors: Sequence[Any], axis: int = 0) -> Any:
        """
        Stack tensors along a new axis.
        
        Args:
            tensors: Sequence of tensors
            axis: Axis along which to stack
            
        Returns:
            Stacked tensor
        """
        pass
    
    @abstractmethod
    def split(self, x: Any, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[Any]:
        """
        Split a tensor into sub-tensors.
        
        Args:
            x: Input tensor
            num_or_size_splits: Number of splits or sizes of each split
            axis: Axis along which to split
            
        Returns:
            List of sub-tensors
        """
        pass
    
    @abstractmethod
    def expand_dims(self, x: Any, axis: Union[int, Sequence[int]]) -> Any:
        """
        Insert new axes into a tensor's shape.
        
        Args:
            x: Input tensor
            axis: Position(s) where new axes should be inserted
            
        Returns:
            Tensor with expanded dimensions
        """
        pass
    
    @abstractmethod
    def squeeze(self, x: Any, axis: Optional[Union[int, Sequence[int]]] = None) -> Any:
        """
        Remove single-dimensional entries from a tensor's shape.
        
        Args:
            x: Input tensor
            axis: Position(s) where dimensions should be removed
            
        Returns:
            Tensor with squeezed dimensions
        """
        pass
    
    @abstractmethod
    def tile(self, x: Any, reps: Sequence[int]) -> Any:
        """
        Construct a tensor by tiling a given tensor.
        
        Args:
            x: Input tensor
            reps: Number of repetitions along each dimension
            
        Returns:
            Tiled tensor
        """
        pass
    
    @abstractmethod
    def gather(self, x: Any, indices: Any, axis: int = 0) -> Any:
        """
        Gather slices from a tensor along an axis.
        
        Args:
            x: Input tensor
            indices: Indices of slices to gather
            axis: Axis along which to gather
            
        Returns:
            Gathered tensor
        """
        pass
    
    @abstractmethod
    def convert_to_tensor(self, x: Any, dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Convert input to a tensor.
        
        Args:
            x: Input data (array, tensor, scalar)
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor representation of the input
        """
        pass
    
    @abstractmethod
    def cast(self, x: Any, dtype: Union[Any, str, Callable[[], Any]]) -> Any:
        """
        Cast a tensor to a different data type.
        
        Args:
            x: Input tensor
            dtype: Target data type (can be a DType, string, or callable)
            
        Returns:
            Tensor with the target data type
        """
        pass
    
    @abstractmethod
    def copy(self, x: Any) -> Any:
        """
        Create a copy of a tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Copy of the tensor
        """
        pass
    
    
    
    
    @abstractmethod
    def slice_tensor(self, x: Any, starts: Sequence[int], sizes: Sequence[int]) -> Any:
        """
        Extract a slice from a tensor.
        
        Args:
            x: Input tensor
            starts: Starting indices for each dimension
            sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
            
        Returns:
            Sliced tensor
        """
        pass
    
    @abstractmethod
    def sort(self, x: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Sorted tensor
        """
        pass
    
    @abstractmethod
    def argsort(self, x: Any, axis: int = -1, descending: bool = False) -> Any:
        """
        Return the indices that would sort a tensor along a specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to sort
            descending: Whether to sort in descending order
            
        Returns:
            Indices that would sort the tensor
        """
        pass
    
    @abstractmethod
    def slice_update(self, x: Any, slices: Union[List, Tuple], updates: Any) -> Any:
        """
        Update a tensor at specific indices.
        
        Args:
            x: Input tensor to update
            slices: List or tuple of slice objects or indices
            updates: Values to insert at the specified indices
            
        Returns:
            Updated tensor
        """
        pass
    
    @abstractmethod
    def pad(self, x: Any, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0) -> Any:
        """
        Pad a tensor with a constant value.
        
        Args:
            x: Input tensor
            paddings: Sequence of sequences of integers specifying the padding for each dimension
                     Each inner sequence should contain two integers: [pad_before, pad_after]
            constant_values: Value to pad with
            
        Returns:
            Padded tensor
        """
        pass

    @abstractmethod
    def tensor_scatter_nd_update(self, tensor: Any, indices: Any, updates: Any) -> Any:
        """
        Updates values of a tensor at specified indices.

        Args:
            tensor: Input tensor to update
            indices: Indices at which to update values (N-dimensional indices)
            updates: Values to insert at the specified indices

        Returns:
            Updated tensor
        """
        pass
        
    @abstractmethod
    def scatter(self, data: Any, indices: Any, dim_size: Optional[Any] = None,
                aggr: str = 'sum', axis: int = 0) -> Any:
        """
        Scatter data according to indices into a new tensor.
        
        Args:
            data: The data to scatter
            indices: The indices to scatter the data to
            dim_size: The size of the output tensor along the specified axis
            aggr: The aggregation method ('sum', 'mean', 'max', 'min')
            axis: The axis along which to scatter
            
        Returns:
            The scattered tensor
        """
        pass
    
    @abstractmethod
    def maximum(self, x1: Any, x2: Any) -> Any:
        """
        Element-wise maximum of two tensors.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Element-wise maximum
        """
        pass
    
    @abstractmethod
    def random_normal(self, shape: Union[int, Sequence[int]], mean: float = 0.0, stddev: float = 1.0,
                     dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a normal distribution.
        
        Args:
            shape: Shape of the tensor
            mean: Mean of the normal distribution
            stddev: Standard deviation of the normal distribution
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random normal values
        """
        pass
    
    @abstractmethod
    def random_uniform(self, shape: Union[int, Sequence[int]], minval: float = 0.0, maxval: float = 1.0,
                      dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a uniform distribution.
        
        Args:
            shape: Shape of the tensor
            minval: Minimum value
            maxval: Maximum value
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random uniform values
        """
        pass
    
    @abstractmethod
    def random_binomial(self, shape: Union[int, Sequence[int]], p: float = 0.5,
                       dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a binomial distribution.
        
        Args:
            shape: Shape of the tensor
            p: Probability of success
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random binomial values
        """
        pass
    
    @abstractmethod
    def random_gamma(self, shape: Union[int, Sequence[int]], alpha: float = 1.0, beta: float = 1.0,
                    dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Generate random values from a gamma distribution.
        
        Args:
            shape: Shape of the output tensor
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random values from a gamma distribution
        """
        pass
    
    @abstractmethod
    def random_exponential(self, shape: Union[int, Sequence[int]], scale: float = 1.0,
                          dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Generate random values from an exponential distribution.
        
        Args:
            shape: Shape of the output tensor
            scale: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random values from an exponential distribution
        """
        pass
    
    @abstractmethod
    def random_poisson(self, shape: Union[int, Sequence[int]], lam: float = 1.0,
                      dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Generate random values from a Poisson distribution.
        
        Args:
            shape: Shape of the output tensor
            lam: Rate parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random values from a Poisson distribution
        """
        pass
    
    @abstractmethod
    def random_categorical(self, logits: Any, num_samples: int,
                          dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Draw samples from a categorical distribution.
        
        Args:
            logits: 2D tensor with unnormalized log probabilities
            num_samples: Number of samples to draw
            dtype: Optional data type
            device: Optional device to place the tensor on
        
        Returns:
            Tensor with random categorical values
        """
        pass
    
    @abstractmethod
    def random_permutation(self, x: Union[int, Any], dtype: Optional[Any] = None, device: Optional[str] = None) -> Any:
        """
        Randomly permute a sequence or return a permuted range.
        
        Args:
            x: If x is an integer, randomly permute tensor.arange(x).
               If x is an array, make a copy and shuffle the elements randomly.
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Permuted tensor
        """
        pass
    

    @abstractmethod
    def shuffle(self, x: Any) -> Any:
        """
        Randomly shuffle a tensor along the first dimension.
        
        Args:
            x: Input tensor
        
        Returns:
            Shuffled tensor
        """
        pass
    
    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        pass
    
    @abstractmethod
    def get_seed(self) -> Optional[int]:
        """
        Get the current random seed.
        
        Returns:
            Current random seed (None if not set)
        """
        pass
    
    @abstractmethod
    def __getstate__(self) -> dict:
        """
        Get the state of the tensor for serialization.
        
        Returns:
            Dictionary containing the tensor state
        """
        pass
    
    @abstractmethod
    def __setstate__(self, state: dict) -> None:
        """
        Restore the tensor from a serialized state.
        
        Args:
            state: Dictionary containing the tensor state
        """
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """
        Make the tensor iterable.
        
        Returns:
            Iterator over the tensor elements, where each element is a TensorInterface
        """
        pass