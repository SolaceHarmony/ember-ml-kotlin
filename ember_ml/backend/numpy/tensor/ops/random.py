"""NumPy tensor random operations."""

import numpy as np
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.numpy.tensor.dtype import NumpyDType
from ember_ml.backend.numpy.types import Shape, TensorLike, DType

# Create single instances to reuse throughout the module
DTypeHandler = NumpyDType()

def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor with random values from a normal distribution.
    
    Args:
        shape: Shape of the tensor
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random normal values
    """
    # Convert shape to a sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    
    # Use NumPy's normal function
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(numpy_dtype) if numpy_dtype is not None else np.random.normal(loc=mean, scale=stddev, size=shape)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor with random values from a uniform distribution.
    
    Args:
        shape: Shape of the tensor
        minval: Minimum value
        maxval: Maximum value
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random uniform values
    """
    # Convert shape to a sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    
    # Use NumPy's uniform function
    tensor = np.random.uniform(low=minval, high=maxval, size=shape)
    
    # Cast to the specified dtype if needed
    if numpy_dtype is not None:
        tensor = tensor.astype(numpy_dtype)
    
    return tensor

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array with random binomial values
    """
    # Convert shape to a sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    
    # Use NumPy's binomial function
    result = np.random.binomial(n=1, p=p, size=shape)
    
    # Convert to the specified dtype if needed
    if numpy_dtype is not None:
        result = result.astype(numpy_dtype)
    
    return result

def random_exponential(shape: Shape, scale: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from an exponential distribution.
    
    Args:
        shape: Shape of the output array
        scale: Scale parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random values from an exponential distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Generate exponential random values
    result = np.random.exponential(scale=scale, size=shape)
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    if numpy_dtype is not None:
        result = result.astype(numpy_dtype)
    
    return result

def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a gamma distribution.
    
    Args:
        shape: Shape of the output array
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random values from a gamma distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")
    
    # Generate gamma random values
    result = np.random.gamma(shape=alpha, scale=beta, size=shape)
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    if numpy_dtype is not None:
        result = result.astype(numpy_dtype)
    
    return result

def random_poisson(shape: Shape, lam: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Generate random values from a Poisson distribution.
    
    Args:
        shape: Shape of the output array
        lam: Rate parameter
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random values from a Poisson distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)
    
    # Generate poisson random values
    result = np.random.poisson(lam=lam, size=shape)
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    if numpy_dtype is not None and numpy_dtype != np.int32:
        result = result.astype(numpy_dtype)
    
    return result

def random_categorical(data: TensorLike, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Draw samples from a categorical distribution.
    
    Args:
        data: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Ignored for NumPy backend
    
    Returns:
        NumPy array with random categorical values
    """
    # Convert to NumPy array if needed
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    logits_tensor = Tensor.convert_to_tensor(data)
    
    # Convert to probabilities
    max_logits = np.max(logits_tensor, axis=-1, keepdims=True)
    exp_logits = np.exp(np.subtract(logits_tensor, max_logits))
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = np.divide(exp_logits, sum_exp_logits)
    
    # Sample from the categorical distribution
    result = np.zeros((logits_tensor.shape[0], num_samples), dtype=np.int64)
    for i in range(logits_tensor.shape[0]):
        result[i] = np.random.choice(logits_tensor.shape[1], size=num_samples, p=probs[i])
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    if numpy_dtype is not None:
        result = result.astype(numpy_dtype)
    
    return result

def random_permutation(data: Union[int, TensorLike], dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Randomly permute a sequence or return a permuted range.
    
    Args:
        data: If data is an integer, randomly permute np.arange(data).
             If data is an array, make a copy and shuffle the elements randomly.
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        Permuted array
    """
    if isinstance(data, int):
        # Create a range and permute it
        result = np.random.permutation(data)
    else:
        # Convert to NumPy array if needed
        from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
        Tensor = NumpyTensor()
        arr = Tensor.convert_to_tensor(data)
        
        # Get the shape of the array
        shape = arr.shape
        
        # If the array is empty or has only one element, return it as is
        if shape[0] <= 1:
            return arr
        
        # Generate random indices
        indices = np.random.permutation(shape[0])
        
        # Gather along the first dimension
        result = arr[indices]
    
    # Validate dtype
    numpy_dtype = DTypeHandler.validate_dtype(dtype)
    if numpy_dtype is not None:
        result = result.astype(numpy_dtype)
    
    return result

def shuffle(data: TensorLike) -> np.ndarray:
    """
    Randomly shuffle a NumPy array along the first dimension.
    
    Args:
        data: Input array
    
    Returns:
        Shuffled NumPy array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    
    data_tensor = Tensor.convert_to_tensor(data)
    
    # Get the shape of the tensor
    shape = data_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return data_tensor
    
    # Generate random indices
    indices = np.random.permutation(shape[0])
    
    # Gather along the first dimension
    return data_tensor[indices]

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)

def get_seed() -> Optional[int]:
    """
    Get the current random seed.
    
    Returns:
        Current random seed (None if not set)
    """
    # NumPy doesn't provide a way to get the current seed
    return None