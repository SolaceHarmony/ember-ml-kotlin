"""
MLX implementation of vector operations.

This module provides MLX implementations of vector operations.
"""

import mlx.core as mx
from typing import Optional, Tuple, List, Union, Any # Keep Any for now
from ember_ml.backend.mlx.types import TensorLike, Shape, Axis, default_float # Remove unused default_int

# We avoid creating global instances to prevent circular imports
# Each function will create its own instances when needed


def normalize_vector(input_vector: TensorLike, axis: Optional[int] = None) -> mx.array:
    """
    Normalize an input vector or matrix to unit length (L2 norm).

    If the vector's norm is zero, the original vector is returned.

    Args:
        input_vector: The vector or matrix to normalize.
        axis: Axis along which to normalize. If None, the entire input is normalized.

    Returns:
        The normalized vector or matrix as an MLX array.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    vector_tensor = tensor_ops.convert_to_tensor(input_vector)
    norm = mx.linalg.norm(vector_tensor, axis=axis, keepdims=True)

    # Avoid division by zero
    norm_safe = mx.maximum(norm, tensor_ops.convert_to_tensor(1e-8))
    return mx.divide(vector_tensor, norm_safe)


def euclidean_distance(vector1: TensorLike, vector2: TensorLike) -> mx.array:
    """
    Compute the Euclidean (L2) distance between two vectors.

    Args:
        vector1: The first input vector.
        vector2: The second input vector.

    Returns:
        An MLX scalar representing the Euclidean distance.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(vector1)
    y_tensor = tensor_ops.convert_to_tensor(vector2)

    return mx.sqrt(mx.sum(mx.square(mx.subtract(x_tensor, y_tensor))))


def cosine_similarity(vector1: TensorLike, vector2: TensorLike) -> mx.array:
    """
    Compute the cosine similarity between two vectors.

    Measures the cosine of the angle between two non-zero vectors.
    Result ranges from -1 (exactly opposite) to 1 (exactly the same).

    Args:
        vector1: The first input vector.
        vector2: The second input vector.

    Returns:
        An MLX scalar representing the cosine similarity.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(vector1)
    y_tensor = tensor_ops.convert_to_tensor(vector2)

    dot_product = mx.sum(mx.multiply(x_tensor, y_tensor))
    norm_x = mx.sqrt(mx.sum(mx.square(x_tensor)))
    norm_y = mx.sqrt(mx.sum(mx.square(y_tensor)))
    return mx.divide(dot_product, mx.add(mx.multiply(norm_x, norm_y), tensor_ops.convert_to_tensor(1e-8)))


def exponential_decay(initial_value: TensorLike, decay_rate: TensorLike, time_step: Optional[TensorLike] = None) -> mx.array:
    """
    Compute exponential decay.

    If `time_step` is provided, applies uniform decay:
        value = initial * exp(-rate * time_step)
    If `time_step` is None, applies index-based decay to the input array:
        value[i] = initial[i] * exp(-rate * i)

    Args:
        initial_value: The starting value(s) (TensorLike).
        decay_rate: The rate of decay (must be positive, TensorLike).
        time_step: The elapsed time (TensorLike) for uniform decay,
                   or None for index-based decay. Defaults to None.

    Returns:
        The value(s) after exponential decay as an MLX array.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    initial_value_tensor = tensor_ops.convert_to_tensor(initial_value)
    decay_rate_tensor = tensor_ops.convert_to_tensor(decay_rate)
    if time_step is not None:
        # Uniform time step decay
        time_tensor = tensor_ops.convert_to_tensor(time_step)
        decay_factor = mx.exp(mx.multiply(mx.negative(decay_rate_tensor), time_tensor))
    else:
        # Index-based decay
        # Ensure indices have a compatible float type for multiplication
        indices = mx.arange(initial_value_tensor.shape[0]).astype(default_float)
        decay_factor = mx.exp(mx.multiply(mx.negative(decay_rate_tensor), indices))

    return mx.multiply(initial_value_tensor, decay_factor)



def compute_energy_stability(input_wave: TensorLike, window_size: int = 100) -> mx.array:
    """
    Compute the energy stability of a wave signal.

    Calculates stability based on the variance of energy across sliding windows.
    A value closer to 1.0 indicates higher stability.

    Args:
        input_wave: The input wave signal.
        window_size: The size of the sliding window used for energy calculation.

    Returns:
        An MLX scalar representing the energy stability metric (0.0 to 1.0).
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    wave_tensor = tensor_ops.convert_to_tensor(input_wave)
    if len(wave_tensor.shape) == 0 or wave_tensor.shape[0] < window_size:
        return mx.array(1.0, dtype=default_float) # Return MLX scalar

    # Compute energy in windows
    num_windows = mx.floor_divide(wave_tensor.shape[0], window_size) # Use mx.floor_divide
    energies: List[Union[float,Any]] = []
    
    # Convert num_windows MLX array to Python int for range()
    # Convert num_windows MLX array to Python int for range()
    # Use .item() as confirmed by CLI test and MLX convention
    num_windows_item = num_windows.item()
    # Mypy might still complain here due to static analysis limits
    for i in range(num_windows_item):
        start = i * window_size
        end = start + window_size
        window = wave_tensor[start:end]
        energy = mx.sum(mx.square(window))
        energies.append(energy.item())

    if len(energies) <= 1:
        return mx.array(1.0, dtype=default_float) # Return MLX scalar

    energies_tensor = tensor_ops.convert_to_tensor(energies)
    energy_mean = mx.mean(energies_tensor)
    
    if energy_mean == 0:
        return mx.array(1.0, dtype=default_float) # Return MLX scalar

    energy_var = mx.var(energies_tensor)
    stability = mx.divide(tensor_ops.convert_to_tensor(1.0), 
                        mx.add(tensor_ops.convert_to_tensor(1.0), 
                                mx.divide(energy_var, energy_mean)))

    # Return as MLX scalar with default float type
    return stability.astype(default_float)


def compute_interference_strength(input_wave1: TensorLike, input_wave2: TensorLike) -> mx.array:
    """
    Compute the interference strength between two wave signals.

    Combines correlation and phase difference to quantify interference.
    A value closer to 1 suggests strong constructive interference,
    closer to -1 suggests strong destructive interference, and near 0
    suggests low interference or incoherence.

    Args:
        input_wave1: The first input wave signal.
        input_wave2: The second input wave signal.

    Returns:
        An MLX scalar representing the interference strength metric.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    from ember_ml.backend.mlx.math_ops import pi as math_pi
    tensor_ops = MLXTensor()
    
    # Convert to MLX tensors
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute correlation
    wave1_mean = mx.mean(wave1_tensor)
    wave2_mean = mx.mean(wave2_tensor)
    wave1_centered = mx.subtract(wave1_tensor, wave1_mean)
    wave2_centered = mx.subtract(wave2_tensor, wave2_mean)

    numerator = mx.sum(mx.multiply(wave1_centered, wave2_centered))
    denominator = mx.multiply(
        mx.sqrt(mx.sum(mx.multiply(wave1_centered, wave1_centered))),
        mx.sqrt(mx.sum(mx.multiply(wave2_centered, wave2_centered)))
    )
    denominator = mx.add(denominator, tensor_ops.convert_to_tensor(1e-8))
    correlation = mx.divide(numerator, denominator)

    # Compute phase difference using FFT
    fft1 = mx.fft.fft(wave1_tensor)
    fft2 = mx.fft.fft(wave2_tensor)

    # MLX doesn't have angle() directly, compute phase using arctan2
    phase1 = mx.arctan2(mx.imag(fft1), mx.real(fft1))
    phase2 = mx.arctan2(mx.imag(fft2), mx.real(fft2))
    phase_diff = mx.abs(mx.subtract(phase1, phase2))
    mean_phase_diff = mx.mean(phase_diff)

    # Normalize phase difference to [0, 1]
    pi_tensor = tensor_ops.convert_to_tensor(math_pi)
    normalized_phase_diff = mx.divide(mean_phase_diff, pi_tensor)

    # Compute interference strength
    interference_strength = mx.multiply(correlation, mx.subtract(tensor_ops.convert_to_tensor(1.0), normalized_phase_diff))

    # Return as MLX scalar with default float type
    return interference_strength.astype(default_float)


def compute_phase_coherence(input_wave1: TensorLike, input_wave2: TensorLike,
                            freq_range: Optional[Tuple[float, float]] = None) -> mx.array:
    """
    Compute the phase coherence between two wave signals.

    Calculates the consistency of the phase difference between two signals,
    optionally within a specified frequency range. Uses circular statistics.
    A value closer to 1 indicates high phase coherence.

    Args:
        input_wave1: The first input wave signal.
        input_wave2: The second input wave signal.
        freq_range: Optional tuple (min_freq, max_freq) to filter frequencies
                    before computing coherence.

    Returns:
        An MLX scalar representing the phase coherence metric (0.0 to 1.0).
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    # Convert to MLX tensors
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute FFT using MLX
    fft1 = mx.fft.fft(wave1_tensor)
    fft2 = mx.fft.fft(wave2_tensor)

    # Get phases using arctan2 since angle() isn't available
    phase1 = mx.arctan2(mx.imag(fft1), mx.real(fft1))
    phase2 = mx.arctan2(mx.imag(fft2), mx.real(fft2))

    # Compute frequencies using manual calculation since fftfreq isn't available
    n = wave1_tensor.shape[0]
    freqs = mx.divide(mx.arange(n), n)

    # Apply frequency range filter if specified
    if freq_range is not None:
        min_freq, max_freq = freq_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        phase1 = mx.where(freq_mask, phase1, mx.zeros_like(phase1))
        phase2 = mx.where(freq_mask, phase2, mx.zeros_like(phase2))

    # Compute phase difference
    phase_diff = mx.subtract(phase1, phase2)

    # Use Euler's formula for complex phase calculation
    complex_real = mx.cos(phase_diff)
    complex_imag = mx.sin(phase_diff)
    coherence = mx.sqrt(mx.add(mx.power(mx.mean(complex_real), 2), mx.power(mx.mean(complex_imag), 2)))

    # Return as MLX scalar with default float type
    return coherence.astype(default_float)


def partial_interference(input_wave1: TensorLike, input_wave2: TensorLike, window_size: int = 100) -> mx.array:
    """
    Compute the partial interference between two wave signals over sliding windows.

    Calculates interference strength for overlapping windows of the signals.

    Args:
        input_wave1: The first input wave signal.
        input_wave2: The second input wave signal.
        window_size: The size of the sliding window.

    Returns:
        An MLX array containing the interference strength for each window.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    from ember_ml.backend.mlx.math_ops import pi as math_pi
    tensor_ops = MLXTensor()
    
    # Convert to MLX tensors
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute number of windows
    num_windows = min_length - window_size + 1

    # Initialize result array
    result: List[Union[float,Any]] = []

    # Compute interference for each window
    for i in range(num_windows):
        window1 = wave1_tensor[i:i+window_size]
        window2 = wave2_tensor[i:i+window_size]

        # Compute correlation
        window1_centered = mx.subtract(window1, mx.mean(window1))
        window2_centered = mx.subtract(window2, mx.mean(window2))
        correlation = mx.divide(
            mx.sum(mx.multiply(window1_centered, window2_centered)),
            mx.add(
                mx.multiply(
                    mx.sqrt(mx.sum(mx.power(window1_centered, 2))),
                    mx.sqrt(mx.sum(mx.power(window2_centered, 2)))
                ),
                tensor_ops.convert_to_tensor(1e-8)
            )
        )

        # Compute FFT for this window
        fft1 = mx.fft.fft(window1)
        fft2 = mx.fft.fft(window2)

        # Get phases using arctan2
        phase1 = mx.arctan2(mx.imag(fft1), mx.real(fft1))
        phase2 = mx.arctan2(mx.imag(fft2), mx.real(fft2))
        phase_diff = mx.abs(mx.subtract(phase1, phase2))
        mean_phase_diff = mx.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        pi_tensor = tensor_ops.convert_to_tensor(math_pi)
        normalized_phase_diff = mx.divide(mean_phase_diff, pi_tensor)

        # Compute interference strength
        val = mx.multiply(correlation, mx.subtract(tensor_ops.convert_to_tensor(1.0), normalized_phase_diff))
        
        result.append(val.item())

    return mx.array(result)

# Note: Removing duplicate definition of euclidean_distance
# Note: Duplicate definition removed by previous diff, assuming this is the intended one
# Note: Removing duplicate definition of cosine_similarity
# Note: Removing orphaned code block causing NameErrors
def fft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> mx.array:
    """
    One dimensional discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_length: Length of the transformed axis
        axis: Axis over which to compute the FFT
        
    Returns:
        The transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    return mx.fft.fft(a_tensor, n=output_length, axis=axis)


def ifft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> mx.array:
    """
    One dimensional inverse discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_length: Length of the transformed axis
        axis: Axis over which to compute the IFFT
        
    Returns:
        The inverse transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    return mx.fft.ifft(a_tensor, n=output_length, axis=axis)


def fft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> mx.array:
    """
    Two dimensional discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the FFT
        
    Returns:
        The transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    
    # Use native mx.fft.fft2
    # Map output_shape -> s and axes -> axes
    return mx.fft.fft2(a_tensor, s=output_shape, axes=axes)


def ifft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> mx.array:
    """
    Two dimensional inverse discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IFFT
        
    Returns:
        The inverse transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Use native mx.fft.ifft2
    # Map output_shape -> s and axes -> axes
    return mx.fft.ifft2(a_tensor, s=output_shape, axes=axes)


def fftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> mx.array:
    """
    N-dimensional discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the FFT
        
    Returns:
        The transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    
    # Use native mx.fft.fftn
    # Map output_shape -> s and axes -> axes
    return mx.fft.fftn(a_tensor, s=output_shape, axes=axes)


def ifftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> mx.array:
    """
    N-dimensional inverse discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IFFT
        
    Returns:
        The inverse transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Use native mx.fft.ifftn
    # Map output_shape -> s and axes -> axes
    return mx.fft.ifftn(a_tensor, s=output_shape, axes=axes)


def rfft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> mx.array:
    """
    One dimensional discrete Fourier Transform for real input.
    
    Args:
        input_array: Input array (real)
        output_length: Length of the transformed axis
        axis: Axis over which to compute the RFFT
        
    Returns:
        The transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    # MLX doesn't have direct rfft, so we implement it using regular fft
    # and taking only the non-redundant half
    result = mx.fft.fft(a_tensor, n=output_length, axis=axis)
    
    # Restore MLX calculation for output size
    input_size = a_tensor.shape[axis]
    # Ensure addition uses mx.add for backend consistency
    output_size = mx.add(mx.floor_divide((input_size if output_length is None else output_length), 2), mx.array(1))
    output_size_item = output_size.item() # Get Python int size

    # Construct arguments for mx.slice
    start_indices = mx.array(0) # Start at index 0 along the specified axis
    slice_axes = (axis,) # Specify the axis to slice
    
    # Determine the full slice_size tuple for all dimensions
    slice_size_list = list(result.shape)
    # Ensure axis is handled correctly if negative
    actual_axis = axis if axis >= 0 else result.ndim + axis
    if 0 <= actual_axis < result.ndim:
        slice_size_list[actual_axis] = output_size_item
    else:
        raise ValueError(f"Invalid axis {axis} for tensor with {result.ndim} dimensions")
    slice_size_tuple = tuple(slice_size_list)

    # Use mx.slice instead of Python slice objects
    return mx.slice(result, start_indices=start_indices, axes=slice_axes, slice_size=slice_size_tuple)


def irfft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> mx.array:
    """
    One dimensional inverse discrete Fourier Transform for real input.
    
    Args:
        input_array: Input array
        output_length: Length of the transformed axis
        axis: Axis over which to compute the IRFFT
        
    Returns:
        The inverse transformed array (real)
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Handle the size parameter
    input_shape = a_tensor.shape
    if output_length is None:
        output_length = 2 * (input_shape[axis] - 1)

    # Reconstruct conjugate symmetric Fourier components
    # Take advantage of conjugate symmetry: F(-f) = F(f)*
    middle_slices = [slice(None)] * a_tensor.ndim
    middle_slices[axis] = slice(1, -1)

    if input_shape[axis] > 1:
        # Since MLX doesn't have flip, we'll reverse the array using array indexing
        rev_idx = mx.arange(input_shape[axis] - 2, -1, -1)

        # Create broadcasted indices for other dimensions
        rev_shape = [1] * a_tensor.ndim
        rev_shape[axis] = rev_idx.shape[0]
        rev_idx = rev_idx.reshape(rev_shape)

        # Broadcast rev_idx to match tensor shape for other dimensions
        broadcast_shape = list(input_shape)
        broadcast_shape[axis] = rev_idx.shape[axis]
        rev_idx = mx.broadcast_to(rev_idx, broadcast_shape)

        # Get reversed data
        reversed_data = mx.take(a_tensor[tuple(middle_slices)], rev_idx, axis=axis)

        # Concatenate with conjugate
        full_fourier = mx.concatenate([
            a_tensor,
            mx.conj(reversed_data)
        ], axis=axis)
    else:
        full_fourier = a_tensor

    # Perform inverse FFT
    return mx.real(mx.fft.ifft(full_fourier, n=output_length, axis=axis))


def rfft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> mx.array:
    """
    Two dimensional real discrete Fourier Transform.
    
    Args:
        input_array: Input array (real)
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the RFFT2
        
    Returns:
        The transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Use native mx.fft.rfft2
    # Map output_shape -> s and axes -> axes
    return mx.fft.rfft2(a_tensor, s=output_shape, axes=axes)


def irfft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> mx.array:
    """
    Two dimensional inverse real discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IRFFT2
        
    Returns:
        The inverse transformed array (real)
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Use native mx.fft.irfft2
    # Map output_shape -> s and axes -> axes
    return mx.fft.irfft2(a_tensor, s=output_shape, axes=axes)


def rfftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> mx.array:
    """
    N-dimensional real discrete Fourier Transform.
    
    Args:
        input_array: Input array (real)
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the RFFTN
        
    Returns:
        The transformed array
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Use native mx.fft.rfftn
    # Map output_shape -> s and axes -> axes
    return mx.fft.rfftn(a_tensor, s=output_shape, axes=axes)


def irfftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> mx.array:
    """
    N-dimensional inverse real discrete Fourier Transform.
    
    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IRFFTN
        
    Returns:
        The inverse transformed array (real)
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    
    a_tensor = tensor_ops.convert_to_tensor(input_array)

    # Use native mx.fft.irfftn
    # Map output_shape -> s and axes -> axes
    return mx.fft.irfftn(a_tensor, s=output_shape, axes=axes)


# Removed MLXVectorOps class as it's redundant with standalone functions
