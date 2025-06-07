"""MLX tensor random operations."""

from typing import Union, Optional, Any
import mlx.core as mx

from ember_ml.backend.mlx.types import Shape, TensorLike, DType,default_int, default_float
from ember_ml.backend.mlx.tensor.ops.utility import _create_new_tensor # Import helper
# Create single instances to reuse throughout the module
# DTypeHandler instance removed, logic moved to helper/local

def random_normal(shape: Shape, mean: float = 0.0, stddev: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None, seed: Optional[int] = None) -> 'mx.array':
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
        :param device:
        :param dtype:
        :param stddev:
        :param mean:
        :param shape:
        :param seed:
    """
    # Set seed if provided
    if seed is not None:
        mx.random.seed(seed)
        
    # Use the helper function, passing mx.random.normal and its specific args
    return _create_new_tensor(mx.random.normal, dtype=dtype, device=device, shape=shape, loc=mean, scale=stddev)

def random_uniform(shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None, seed: Optional[int] = None) -> mx.array:
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
        :param device:
        :param dtype:
        :param maxval:
        :param minval:
        :param shape:
        :param seed:
    """
    from ember_ml.backend.mlx.tensor.dtype import MLXDType
    
    # Set seed if provided
    if seed is not None:
        mx.random.seed(seed)
        
    # Automatically work with int types or floats
    mlx_dtype_handler = MLXDType()
    if dtype is not None and 'float' in mlx_dtype_handler.to_dtype_str(dtype):
        # For float types, use mx.random.uniform directly
        return _create_new_tensor(mx.random.uniform, dtype=dtype, device=device, 
                                shape=shape, low=minval, high=maxval)
    elif dtype is not None and 'int' in mlx_dtype_handler.to_dtype_str(dtype):
        # For integer types, generate uniform floats and then cast to int
        # MLX doesn't have a direct randint function, so we need to use uniform and then cast
        # Generate uniform values in [minval, maxval+1) to include maxval
        from ember_ml.backend.mlx.tensor import MLXTensor
        tensor = MLXTensor()
        high = mx.add(tensor.convert_to_tensor(maxval, dtype=mx.float32), tensor.convert_to_tensor(1, dtype=mx.float32)) if maxval == int(maxval) else mx.add(tensor.convert_to_tensor(int(maxval), dtype=mx.float32), tensor.convert_to_tensor(1, dtype=mx.float32))
        low = tensor.convert_to_tensor(int(minval), dtype=mx.float32)
        
        # Generate uniform floats in [low, high)
        # MLX's random_uniform only supports float types, so we need to generate floats and then cast to int
        floats = mx.random.uniform(shape=shape, low=low, high=high)
        
        # Cast to integer type
        if dtype is not None:
            mlx_dtype = mlx_dtype_handler.get_dtype(dtype)
            return mx.floor(floats).astype(mlx_dtype)
        else:
            return mx.floor(floats).astype(default_int)
    else:
        # For float types, use mx.random.uniform directly
        return _create_new_tensor(mx.random.uniform,low=minval, high=maxval,shape=shape, dtype=default_float)

def random_binomial(shape: Shape, p: float = 0.5,
                   dtype: Optional[DType] = None, device: Optional[str] = None, seed: Optional[int] = None) -> mx.array:
    """
    Create a tensor with random values from a binomial distribution.
    
    Args:
        shape: Shape of the tensor
        p: Probability of success
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with random binomial values
        :param device:
        :param dtype:
        :param p:
        :param shape:
        :param seed:
    """
    # Set seed if provided
    if seed is not None:
        mx.random.seed(seed)
    
    # Use the helper function, passing mx.random.bernoulli and its specific args
    # Note: MLX's bernoulli function expects p and shape parameters
    return _create_new_tensor(mx.random.bernoulli, dtype=dtype, device=device, p=p, shape=shape)

def random_exponential(shape: Shape, scale: float = 1.0,
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
    # MLX doesn't have direct exponential. Sample uniform and transform: -scale * log(1-U)
    # Use helper for uniform sampling first.
    u = _create_new_tensor(mx.random.uniform, shape=shape, dtype=dtype, device=device) # Use target dtype for intermediate if specified

    # Perform transformation using mx ops
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    scale_tensor = tensor.convert_to_tensor(scale, dtype=u.dtype)
    one_tensor = tensor.convert_to_tensor(1.0, dtype=u.dtype)
    log_input = mx.maximum(mx.subtract(one_tensor, u), tensor.convert_to_tensor(1e-9, dtype=u.dtype))
    result = mx.multiply(mx.negative(scale_tensor), mx.log(log_input))

    # The result should already have the correct dtype from u or inference
    return result

def random_gamma(shape: Shape, alpha: float = 1.0, beta: float = 1.0,
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
    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")
    # Use the helper function, passing mx.random.gamma and its specific args
    # Note: mx.random.gamma takes shape_param (alpha) and scale (beta)
    return _create_new_tensor(mx.random.gamma, dtype=dtype, device=device, shape=shape, shape_param=alpha, scale=beta)

def random_poisson(shape: Shape, lam: float = 1.0,
                   dtype: Optional[DType] = None, seed: Optional[int] = None) -> mx.array:
    """
    Generate random values from a Poisson distribution using Knuth's algorithm.
    
    Args:
        shape: Shape of the output array
        lam: Rate parameter (mean of the distribution)
        dtype: Optional data type
        seed: Optional random seed
    
    Returns:
        MLX array with random values from a Poisson distribution
    """
    # Set seed if provided
    if seed is not None:
        mx.random.seed(seed)
    
    # Import necessary dependencies
    from ember_ml.backend.mlx.tensor import MLXTensor
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    tensor = MLXTensor()
    
    # Validate and get the MLX dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Ensure shape is a tuple
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
    
    # For very small lambda values, optimize by returning zeros
    # This is an optimization to avoid unnecessary computation
    if lam <= 0:
        return mx.zeros(shape, dtype=mlx_dtype or mx.int32)
    
    # For large lambda values (> 15), use Normal approximation
    if lam > 15:
        # Normal approximation for large lambda
        # Poisson(λ) ≈ Normal(λ, λ)
        normal_samples = mx.random.normal(shape=shape, loc=lam, scale=mx.sqrt(tensor.convert_to_tensor(lam)))
        # Round and ensure non-negative
        return mx.maximum(mx.round(normal_samples), 0).astype(mlx_dtype or mx.int32)
    
    # Implementation of Knuth's algorithm:
    # 1. Initialize: Set k = 0 and p = 1
    # 2. Repeat: k = k + 1, p = p * u where u is a uniform random number
    # 3. Until: p < e^(-λ)
    # 4. Return: k - 1
    
    # Initialize result array with zeros
    result = mx.zeros(shape, dtype=mx.int32)
    
    # Calculate limit
    L = mx.exp(-lam)
    
    # Generate uniform random samples for initial step
    u = mx.random.uniform(shape=shape)
    k = mx.zeros(shape, dtype=mx.int32)
    p = mx.ones(shape, dtype=mx.float32)
    
    # Create a mask for values that need to continue
    continue_mask = p >= L
    
    # Continue until all values are completed
    max_iterations = max(100, int(lam * 5))  # Avoid infinite loops with a safety limit
    
    for _ in range(max_iterations):
        # If no values need to continue, break
        if not mx.any(continue_mask):
            break
        
        # Increment k where needed
        k = mx.where(continue_mask, k + 1, k)
        
        # Generate new uniform samples
        u = mx.random.uniform(shape=shape)
        
        # Update product
        p = mx.where(continue_mask, p * u, p)
        
        # Update mask
        continue_mask = p >= L
    
    # Ensure k does not exceed a reasonable maximum
    # For Poisson, values beyond 5*lambda are extremely unlikely
    max_reasonable_value = int(lam * 5) + 10
    result = mx.minimum(k, tensor.convert_to_tensor(max_reasonable_value, dtype=mx.int32))
    
    # Cast to desired dtype if provided
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_categorical(logits: TensorLike, num_samples: int,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Draw samples from a categorical distribution.
    
    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Ignored for MLX backend
    
    Returns:
        MLX array with random categorical values
    """
    # Import here to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor


    logits_tensor = MLXTensor().convert_to_tensor(logits)
    
    # MLX's categorical function takes num_samples parameter
    result = mx.random.categorical(logits=logits_tensor, num_samples=num_samples)
    
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    if mlx_dtype is not None:
        result = result.astype(mlx_dtype)
    
    return result

def random_permutation(x: Union[int, TensorLike], dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Randomly permute a sequence or return a permuted range.
    
    Args:
        x: If x is an integer, randomly permute mx.arange(x).
           If x is an array, make a copy and shuffle the elements randomly.
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        Permuted array
    """
    if isinstance(x, int):
        # Create a range and permute it
        from ember_ml.backend.mlx.tensor import MLXTensor
        tensor = MLXTensor()
        arr = mx.arange(tensor.convert_to_tensor(x, dtype=mx.int32))
        indices = mx.random.permutation(tensor.convert_to_tensor(x, dtype=mx.int32))
        return arr[indices]
    else:
        from ember_ml.backend.mlx.tensor import MLXTensor
        tensor = MLXTensor()
        arr = tensor.convert_to_tensor(x)
        indices = mx.random.permutation(tensor.convert_to_tensor(arr.shape[0], dtype=mx.int32))
        return arr[indices]

def random_shuffle(data: TensorLike) -> mx.array:
    """
    Randomly shuffle an MLX array along the first dimension.
    Similar to shuffle but specifically for shuffling indices.
    
    Args:
        data: Input array
    
    Returns:
        Shuffled MLX array
    """
    # Import here to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor

    data_tensor = MLXTensor().convert_to_tensor(data)
    
    # Get the shape of the tensor
    shape = data_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return data_tensor
    
    # Generate random indices
    indices = mx.random.permutation(shape[0])
    
    # Gather along the first dimension
    return mx.take(data_tensor, indices, axis=0)

def shuffle(x: TensorLike) -> mx.array:
    """
    Randomly shuffle an MLX array along the first dimension.
    
    Args:
        x: Input array
    
    Returns:
        Shuffled MLX array
    """
    # Import here to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor

    x_tensor = MLXTensor().convert_to_tensor(x)
    
    # Get the shape of the tensor
    shape = x_tensor.shape
    
    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor
    
    # Generate random indices
    indices = mx.random.permutation(shape[0])
    
    # Gather along the first dimension
    return mx.take(x_tensor, indices, axis=0)

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    mx.random.seed(seed)

def get_seed() -> Any:
    """
    Get the current random seed.
    
    Returns:
        Current random seed (None if not set)
    """
    # MLX doesn't provide a way to get the current seed
    return None