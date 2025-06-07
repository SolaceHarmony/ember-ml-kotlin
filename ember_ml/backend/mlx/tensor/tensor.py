"""MLX tensor class and operations."""

import mlx.core as mx
import numpy
from typing import Union, Optional, Sequence, Any, Literal, List, TYPE_CHECKING, Tuple
from ember_ml.backend.mlx.tensor.dtype import MLXDType

# Conditionally import backend types for type checking only
import mlx.core

# Basic type aliases
Numeric = Union[int, float]

# Type definitions for MLX dtypes
DTypeStr = str
DTypeClass = Union[mlx.core.Dtype, str, None]

# Type alias for dtype arguments that maintains compatibility
# with both MLX's dtype system and tensor.py's DType
DType = Any  # Using Any for maximum compatibility

# MLX array types
MLXArray = mlx.core.array
ArrayLike = Union[MLXArray, Numeric, List[Any], Tuple[Any, ...]]

TensorTypes = Any
# Conditional imports
if TYPE_CHECKING:
    import numpy as np
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    
    TensorTypes = Union[
        np.ndarray,
        MLXArray,
        MLXTensor,
        EmberTensor
    ]

# Main type definitions
TensorLike = Optional[Union[
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    TensorTypes
]]

# Shape types
Shape = Sequence[int]
ShapeType = Union[int, Tuple[int, ...], List[int]]
ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]

# Dimension types
DimSize = Union[int, MLXArray]
Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
ScalarLike = Optional[Union[
    Numeric,
    bool,
    MLXArray,
    TensorTypes
]]

# Device type
Device = Optional[str]

# Index types
IndexType = Union[int, Sequence[int], MLXArray]
Indices = Union[Sequence[int], MLXArray]


class MLXTensor:
    """MLX tensor operations."""

    def __init__(self):
        """Initialize MLX tensor operations."""
        self._dtype_cls = MLXDType()

    def zeros(self, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array of zeros.

        Args:
            shape: Shape of the array
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array of zeros
        """
        from ember_ml.backend.mlx.tensor.ops.creation import zeros as zeros_func
        return zeros_func(shape, dtype, device)

    def ones(self, shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array of ones.

        Args:
            shape: Shape of the array
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array of ones
        """
        from ember_ml.backend.mlx.tensor.ops.creation import ones as ones_func
        return ones_func(shape, dtype, device)

    def convert_to_tensor(self, data: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Convert input to MLX array.

        Args:
            data: Input data
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import _convert_to_tensor as convert_to_tensor_func
        return convert_to_tensor_func(data, dtype, device)
        
    def convert(self, data: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Convert input to MLX array with validated dtype and device in one step.
        
        This is a simplified helper method that combines dtype validation and tensor conversion,
        which is a common pattern when handling dtype and device parameters in operations.
        
        Args:
            data: Input data
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            MLX array with the specified dtype and device
        """
        # Use lazy imports to avoid circular dependencies
        # Validate the dtype first
        if dtype is not None:
            from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
            mlx_dtype = _validate_and_get_mlx_dtype(dtype)
        else:
            mlx_dtype = None
        
        # Convert input to MLX array with the validated dtype and device
        # Use the convert_to_tensor method which already has lazy imports
        return self.convert_to_tensor(data, dtype=mlx_dtype, device=device)

    def slice_tensor(self, data: TensorLike, starts: Sequence[int], sizes: Sequence[int]) -> mx.array:
        """
        Extract a slice from a tensor.

        Args:
            data: Input tensor
            starts: Starting indices for each dimension
            sizes: Sizes of slice in each dimension

        Returns:
            Sliced tensor
        """
        from ember_ml.backend.mlx.tensor.ops.indexing import slice_tensor as slice_func
        return slice_func(data, starts, sizes)

    def slice_update(self, data: TensorLike, slices: Any, updates: Any) -> mx.array:
        """
        Update a tensor at specific indices.

        Args:
            data: Input tensor to update
            slices: List or tuple of slice objects or indices
            updates: Values to insert at the specified indices

        Returns:
            Updated tensor
        """
        from ember_ml.backend.mlx.tensor.ops.indexing import slice_update as slice_update_func
        return slice_update_func(data, slices, updates)

    def tensor_scatter_nd_update(self, data: TensorLike, indices: Any, updates: Any) -> mx.array:
        """
        Update tensor elements at given indices.

        Args:
            data: Input tensor
            indices: Index tensor
            updates: Values to insert

        Returns:
            Updated tensor
        """
        from ember_ml.backend.mlx.tensor.ops.indexing import tensor_scatter_nd_update as tensor_scatter_nd_update_func
        return tensor_scatter_nd_update_func(data, indices, updates)

    def item(self, data: TensorLike) -> Union[int, float, bool]:
        """
        Extract the scalar value from a tensor.

        Args:
            data: Input tensor containing a single element

        Returns:
            Standard Python scalar (int, float, or bool)
        """
        from ember_ml.backend.mlx.tensor.ops.utility import item as item_func
        return item_func(data)

    def shape(self, data: TensorLike) -> Shape:
        """
        Get the shape of a tensor.

        Args:
            data: Input array

        Returns:
            Shape of the array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import shape as shape_func
        return shape_func(data)

    def dtype(self, data: TensorLike) -> DType:
        """
        Get the data type of a tensor.

        Args:
            data: Input array

        Returns:
            Data type of the array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import dtype as dtype_func
        return dtype_func(data)

    def zeros_like(self, data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array of zeros with the same shape as the input.

        Args:
            data: Input array
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array of zeros with the same shape as data
        """
        from ember_ml.backend.mlx.tensor.ops.creation import zeros_like as zeros_like_func
        return zeros_like_func(data, dtype, device)

    def ones_like(self, data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array of ones with the same shape as the input.

        Args:
            data: Input array
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array of ones with the same shape as data
        """
        from ember_ml.backend.mlx.tensor.ops.creation import ones_like as ones_like_func
        return ones_like_func(data, dtype, device)

    def eye(self, n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX identity matrix.

        Args:
            n: Number of rows
            m: Number of columns (default: n)
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX identity matrix of shape (n, m)
        """
        from ember_ml.backend.mlx.tensor.ops.creation import eye as eye_func
        return eye_func(n, m, dtype, device)

    def reshape(self, data: TensorLike, shape: Shape) -> mx.array:
        """
        Reshape an MLX array to a new shape.

        Args:
            data: Input array
            shape: New shape

        Returns:
            Reshaped MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import reshape as reshape_func
        return reshape_func(data, shape)

    def transpose(self, data: TensorLike, axes: Optional[List[int]] = None) -> mx.array:
        """
        Permute the dimensions of an MLX array.

        Args:
            data: Input array
            axes: Optional permutation of dimensions

        Returns:
            Transposed MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import transpose as transpose_func
        return transpose_func(data, axes)

    def concatenate(self, data: List[TensorLike], axis: int = 0) -> mx.array:
        """
        Concatenate MLX arrays along a specified axis.

        Args:
            data: Sequence of arrays
            axis: Axis along which to concatenate

        Returns:
            Concatenated MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import concatenate as concatenate_func
        return concatenate_func(data, axis)

    def stack(self, data: List[TensorLike], axis: int = 0) -> mx.array:
        """
        Stack MLX arrays along a new axis.

        Args:
            data: Sequence of arrays
            axis: Axis along which to stack

        Returns:
            Stacked MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import stack as stack_func
        return stack_func(data, axis)

    def split(self, data: TensorLike, num_or_size_splits: Union[int, List[int]], axis: int = 0) -> List[mx.array]:
        """
        Split an MLX array into sub-arrays.

        Args:
            data: Input array
            num_or_size_splits: Number of splits or sizes of each split
            axis: Axis along which to split

        Returns:
            List of sub-arrays
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import split as split_func
        return split_func(data, num_or_size_splits, axis)
    
    def split_tensor(self, data: TensorLike, num_splits: int, axis: int = 0) -> List[mx.array]:
        """
        Split a tensor into num_splits equal parts along the given axis.
        
        Args:
            data: Input tensor
            num_splits: Number of equal parts to split into
            axis: Axis along which to split
            
        Returns:
            List of sub-tensors
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import split_tensor as split_tensor_func
        return split_tensor_func(data, num_splits, axis)

    def expand_dims(self, data: TensorLike, axis: Union[int, List[int]]) -> mx.array:
        """
        Insert new axes into an MLX array's shape.

        Args:
            data: Input array
            axis: Position(s) where new axes should be inserted

        Returns:
            MLX array with expanded dimensions
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import expand_dims as expand_dims_func
        return expand_dims_func(data, axis)

    def squeeze(self, data: TensorLike, axis: Optional[Union[int, List[int]]] = None) -> mx.array:
        """
        Remove single-dimensional entries from an MLX array's shape.

        Args:
            data: Input array
            axis: Position(s) where dimensions should be removed

        Returns:
            MLX array with squeezed dimensions
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import squeeze as squeeze_func
        return squeeze_func(data, axis)

    def copy(self, data: TensorLike) -> mx.array:
        """
        Create a copy of an MLX array.

        Args:
            data: Input array

        Returns:
            Copy of the array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import copy as copy_func
        return copy_func(data)

    def full(self, shape: Shape, fill_value: Union[int, float, bool], dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array filled with a scalar value.

        Args:
            shape: Shape of the array
            fill_value: Value to fill the array with
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array filled with the specified value
        """
        from ember_ml.backend.mlx.tensor.ops.creation import full as full_func
        return full_func(shape, fill_value, dtype, device)



    def cast(self, data: TensorLike, dtype: DType) -> mx.array:
        """
        Cast a tensor to a new data type.

        Args:
            data: Input tensor
            dtype: Target data type

        Returns:
            Tensor with new data type
        """
        from ember_ml.backend.mlx.tensor.ops.casting import cast as cast_func
        return cast_func(data, dtype)


    def full_like(self, data: TensorLike, fill_value: Union[int, float, bool], dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array filled with a scalar value with the same shape as the input.

        Args:
            data: Input array
            fill_value: Value to fill the array with
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array filled with the specified value with the same shape as data
        """
        from ember_ml.backend.mlx.tensor.ops.creation import full_like as full_like_func
        return full_like_func(data, fill_value, dtype, device)

    def arange(self, start: Union[int, float], stop: Optional[Union[int, float]] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array with evenly spaced values within a given interval.

        Args:
            start: Start of interval (inclusive)
            stop: End of interval (exclusive)
            step: Spacing between values
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with evenly spaced values
        """
        from ember_ml.backend.mlx.tensor.ops.creation import arange as arange_func
        return arange_func(start, stop, step, dtype, device)

    def linspace(self, start: Union[int, float], stop: Union[int, float], num: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create an MLX array with evenly spaced values within a given interval.

        Args:
            start: Start of interval (inclusive)
            stop: End of interval (inclusive)
            num: Number of values to generate
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with evenly spaced values
        """
        from ember_ml.backend.mlx.tensor.ops.creation import linspace as linspace_func
        return linspace_func(start, stop, num, dtype, device)

    def tile(self, data: TensorLike, reps: List[int]) -> mx.array:
        """
        Construct an MLX array by tiling a given array.

        Args:
            data: Input array
            reps: Number of repetitions for each dimension

        Returns:
            Tiled MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import tile as tile_func
        return tile_func(data, reps)

    def gather(self, data: TensorLike, indices: Any, axis: int = 0) -> mx.array:
        """
        Gather slices from an MLX array along an axis.

        Args:
            data: Input array
            indices: Indices to gather
            axis: Axis along which to gather

        Returns:
            Gathered tensor
        """
        from ember_ml.backend.mlx.tensor.ops.indexing import gather as gather_func
        return gather_func(data, indices, axis)

    def pad(self, data: TensorLike, paddings: List[List[int]], constant_values: int = 0) -> mx.array:
        """
        Pad a tensor with a constant value.

        Args:
            data: Input tensor
            paddings: List of lists of integers specifying the padding for each dimension
                     Each inner list should contain two integers: [pad_before, pad_after]
            constant_values: Value to pad with

        Returns:
            Padded tensor
        """
        from ember_ml.backend.mlx.tensor.ops.manipulation import pad as pad_func
        return pad_func(data, paddings, constant_values)


    def to_numpy(self, data: Optional[TensorLike] = None) -> Optional[numpy.ndarray]:
        """
        Convert an MLX array to a NumPy array.

        IMPORTANT: This function is provided ONLY for visualization/plotting libraries
        that specifically require NumPy arrays. It should NOT be used for general tensor
        conversions or operations. Ember ML has a zero backend design where EmberTensor
        relies entirely on the selected backend for representation.

        Args:
            data: Input MLX array

        Returns:
            NumPy array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import to_numpy as to_numpy_func
        if data is not None:
            return to_numpy_func(data)
        else:
            return 

    def var(self, data: TensorLike, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> mx.array:
        """
        Compute the variance of a tensor.

        Args:
            data: Input array
            axis: Axis or axes along which to compute the variance
            keepdims: Whether to keep the dimensions or not

        Returns:
            Variance of the array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import var as var_func
        return var_func(data, axis, keepdims)

    def sort(self, data: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
        """
        Sort a tensor along the given axis.

        Args:
            data: Input array
            axis: Axis along which to sort
            descending: Whether to sort in descending order

        Returns:
            Sorted array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import sort as sort_func
        return sort_func(data, axis, descending)

    def argsort(self, data: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
        """
        Return the indices that would sort a tensor along the given axis.

        Args:
            data: Input array
            axis: Axis along which to sort
            descending: Whether to sort in descending order

        Returns:
            Indices that would sort the array
        """
        from ember_ml.backend.mlx.tensor.ops.utility import argsort as argsort_func
        return argsort_func(data, axis, descending)

    def maximum(self, data1: TensorLike, data2: TensorLike) -> mx.array:
        """
        Element-wise maximum of two arrays.

        Args:
            data1: First input array
            data2: Second input array

        Returns:
            Element-wise maximum
        """
        from ember_ml.backend.mlx.tensor.ops.utility import maximum as maximum_func
        return maximum_func(data1, data2)

    def random_normal(self, shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create a tensor with random values from a normal distribution.

        Args:
            shape: Shape of the tensor
            mean: Mean of the normal distribution
            stddev: Standard deviation of the normal distribution
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random normal values
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_normal as random_normal_func
        return random_normal_func(shape, mean, stddev, dtype, device)

    def random_uniform(self, shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create a tensor with random values from a uniform distribution.

        Args:
            shape: Shape of the tensor
            minval: Minimum value
            maxval: Maximum value
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random uniform values
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_uniform as random_uniform_func
        return random_uniform_func(shape, minval, maxval, dtype, device)

    def random_binomial(self, shape: Shape, p: float = 0.5,
                        dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Create a tensor with random values from a binomial distribution.

        Args:
            shape: Shape of the tensor
            p: Probability of success
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random binomial values
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_binomial as random_binomial_func
        return random_binomial_func(shape, p, dtype, device)

    def random_gamma(self, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                     dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Generate random values from a gamma distribution.

        Args:
            shape: Shape of the output array
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random values from a gamma distribution
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_gamma as random_gamma_func
        return random_gamma_func(shape, alpha, beta, dtype, device)

    def random_exponential(self, shape: Shape, scale: float = 1.0,
                           dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Generate random values from an exponential distribution.

        Args:
            shape: Shape of the output array
            scale: Scale parameter
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random values from an exponential distribution
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_exponential as random_exponential_func
        return random_exponential_func(shape, scale, dtype, device)

    def random_poisson(self, shape: Shape, lam: float = 1.0,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Generate random values from a Poisson distribution.

        Args:
            shape: Shape of the output array
            lam: Rate parameter
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random values from a Poisson distribution
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_poisson as random_poisson_func
        return random_poisson_func(shape, lam, dtype, device)

    def random_categorical(self, data: TensorLike, num_samples: int,
                                dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Draw samples from a categorical distribution.

        Args:
            data: 2D tensor with unnormalized log probabilities
            num_samples: Number of samples to draw
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            MLX array with random categorical values
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_categorical as random_categorical_func
        return random_categorical_func(data, num_samples, dtype, device)

    def random_permutation(self, data: Union[int, TensorLike], dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
        """
        Randomly permute a sequence or return a permuted range.

        Args:
            data: If data is an integer, randomly permute mx.arange(data).
                 If data is an array, make a copy and shuffle the elements randomly.
            dtype: Optional data type
            device: Ignored for MLX backend

        Returns:
            Permuted array
        """
        from ember_ml.backend.mlx.tensor.ops.random import random_permutation as random_permutation_func
        return random_permutation_func(data, dtype, device)

    def shuffle(self, data: TensorLike) -> mx.array:
        """
        Randomly shuffle an MLX array along the first dimension.

        Args:
            data: Input array

        Returns:
            Shuffled MLX array
        """
        from ember_ml.backend.mlx.tensor.ops.random import shuffle as shuffle_func
        return shuffle_func(data)

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed: Random seed
        """
        from ember_ml.backend.mlx.tensor.ops.random import set_seed as set_seed_func
        return set_seed_func(seed)

    def get_seed(self) -> Optional[int]:
        """
        Get the current random seed.

        Returns:
            Current random seed (None if not set)
        """
        from ember_ml.backend.mlx.tensor.ops.random import get_seed as get_seed_func
        return get_seed_func()

    def scatter(self, data: TensorLike, indices: Any, dim_size: Optional[int] = None,
                      aggr: Literal["add", "max", "mean", "softmax", "min"] = "add", axis: int = 0) -> mx.array:
        """
        Scatter values from data into a new tensor of size dim_size along the given axis.

        Args:
            data: Source tensor containing values to scatter
            indices: Indices where to scatter the values
            dim_size: Size of the output tensor along the given axis. If None, uses the maximum index + 1
            aggr: Aggregation method to use for duplicate indices ("add", "max", "mean", "softmax", "min")
            axis: Axis along which to scatter

        Returns:
            Tensor with scattered values
        """
        from ember_ml.backend.mlx.tensor.ops.indexing import scatter as scatter_func
        return scatter_func(data, indices, dim_size, aggr)