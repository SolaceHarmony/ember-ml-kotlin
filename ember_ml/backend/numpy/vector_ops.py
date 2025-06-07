"""
NumPy implementation of vector operations.

This module provides NumPy implementations of vector operations.
"""

import numpy as np
from typing import Optional, Tuple
from ember_ml.backend.numpy.types import TensorLike, Shape, default_float, Axis


def fft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> np.ndarray:
    """
    Compute the one-dimensional discrete Fourier Transform.

    Args:
        input_array: Input array.
        output_length: Length of the transformed axis of the output.
                       If None, the length of the input along the axis is used.
        axis: Axis over which to compute the FFT.

    Returns:
        The transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    return np.fft.fft(input_tensor, n=output_length, axis=axis)

def ifft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> np.ndarray:
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Args:
        input_array: Input array.
        output_length: Length of the transformed axis of the output.
                       If None, the length of the input along the axis is used.
        axis: Axis over which to compute the inverse FFT.

    Returns:
        The inverse transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    return np.fft.ifft(input_tensor, n=output_length, axis=axis)

def fft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> np.ndarray:
    """
    Compute the two-dimensional discrete Fourier Transform.

    Args:
        input_array: Input array.
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the FFT.

    Returns:
        The transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int]
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.fft2(input_tensor, s=output_shape, axes=axes)

def ifft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> np.ndarray:
    """
    Compute the two-dimensional inverse discrete Fourier Transform.

    Args:
        input_array: Input array.
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the inverse FFT.

    Returns:
        The inverse transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int]
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.ifft2(input_tensor, s=output_shape, axes=axes)

def fftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> np.ndarray:
    """
    Compute the N-dimensional discrete Fourier Transform.

    Args:
        input_array: Input array.
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the FFT. If None, all axes are used.

    Returns:
        The transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int] | None
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.fftn(input_tensor, s=output_shape, axes=axes)

def ifftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> np.ndarray:
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    Args:
        input_array: Input array.
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the inverse FFT. If None, all axes are used.

    Returns:
        The inverse transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int] | None
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.ifftn(input_tensor, s=output_shape, axes=axes)

def rfft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> np.ndarray:
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    Args:
        input_array: Input array (must be real).
        output_length: Length of the transformed axis of the output.
                       If None, the length of the input along the axis is used.
        axis: Axis over which to compute the FFT.

    Returns:
        The transformed array. The output has the same shape as input,
        except along the specified axis, where the length is (output_length // 2) + 1.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    return np.fft.rfft(input_tensor, n=output_length, axis=axis)

def irfft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> np.ndarray:
    """
    Compute the one-dimensional inverse discrete Fourier Transform for real input.

    Args:
        input_array: Input array.
        output_length: Length of the output array along the transformed axis.
                       If None, it defaults to 2 * (input_array.shape[axis] - 1).
        axis: Axis over which to compute the inverse FFT.

    Returns:
        The inverse transformed array (real).
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    return np.fft.irfft(input_tensor, n=output_length, axis=axis)

def rfft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> np.ndarray:
    """
    Compute the two-dimensional discrete Fourier Transform for real input.

    Args:
        input_array: Input array (must be real).
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the FFT.

    Returns:
        The transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int]
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.rfft2(input_tensor, s=output_shape, axes=axes)

def irfft2(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = (-2, -1)) -> np.ndarray:
    """
    Compute the two-dimensional inverse discrete Fourier Transform for real input.

    Args:
        input_array: Input array.
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the inverse FFT.

    Returns:
        The inverse transformed array (real).
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int]
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.irfft2(input_tensor, s=output_shape, axes=axes)

def rfftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> np.ndarray:
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    Args:
        input_array: Input array (must be real).
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the FFT. If None, all axes are used.

    Returns:
        The transformed array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int] | None
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.rfftn(input_tensor, s=output_shape, axes=axes)

def irfftn(input_array: TensorLike, output_shape: Optional[Shape] = None, axes: Axis = None) -> np.ndarray:
    """
    Compute the N-dimensional inverse discrete Fourier Transform for real input.

    Args:
        input_array: Input array.
        output_shape: Shape (length of each transformed axis) of the output.
                      If None, the shape of the input along the axes is used.
        axes: Axes over which to compute the inverse FFT. If None, all axes are used.

    Returns:
        The inverse transformed array (real).
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    input_tensor = tensor_ops.convert_to_tensor(input_array)
    # Handle potential int for axes, though numpy expects Sequence[int] | None
    if isinstance(axes, int):
        axes = (axes,)
    return np.fft.irfftn(input_tensor, s=output_shape, axes=axes)

def normalize_vector(input_vector: TensorLike, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize an input vector to unit length (L2 norm).

    If the vector's norm is zero, the original vector is returned.

    Args:
        input_vector: The vector to normalize.
        axis: Axis along which to normalize. If None, the entire vector is normalized.

    Returns:
        The normalized vector as a NumPy array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    vector_tensor = tensor_ops.convert_to_tensor(input_vector)
    norm = np.linalg.norm(vector_tensor, axis=axis, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_vector = np.divide(vector_tensor, norm, where=(norm > 0))
    return normalized_vector

def compute_energy_stability(input_wave: TensorLike, window_size: int = 100) -> np.ndarray:
    """
    Compute the energy stability of a wave signal.

    Calculates stability based on the variance of energy across sliding windows.
    A value closer to 1.0 indicates higher stability.

    Args:
        input_wave: The input wave signal.
        window_size: The size of the sliding window used for energy calculation.

    Returns:
        A NumPy scalar representing the energy stability metric (0.0 to 1.0).
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    wave_tensor = tensor_ops.convert_to_tensor(input_wave)
    if len(wave_tensor) < window_size:
        return np.array(1.0, dtype=default_float) # Return NumPy scalar using default float

    # Compute energy in windows
    num_windows = np.floor_divide(len(wave_tensor), window_size)
    energies = []

    for i in range(num_windows):
        start = np.multiply(i, window_size) # Use np.multiply
        end = np.add(start, window_size) # Use np.add
        window = wave_tensor[start:end]
        energy = np.sum(np.square(window))
        energies.append(energy)

    # Compute stability as inverse of energy variance
    if len(energies) <= 1:
        return np.array(1.0, dtype=default_float) # Return NumPy scalar using default float

    energy_mean = np.mean(energies)
    if energy_mean == 0:
        return np.array(1.0, dtype=default_float) # Return NumPy scalar using default float

    energy_var = np.var(energies)
    # Use np functions for arithmetic
    one = np.array(1.0, dtype=default_float) # Use default_float
    stability = np.divide(one, np.add(one, np.divide(energy_var, energy_mean)))

    return np.array(stability, dtype=default_float) # Return NumPy scalar using default float

def compute_interference_strength(input_wave1: TensorLike, input_wave2: TensorLike) -> np.ndarray:
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
        A NumPy scalar representing the interference strength metric.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    from ember_ml.backend.numpy.math_ops import NumpyMathOps # Import for pi
    tensor_ops = NumpyTensor()
    math_ops = NumpyMathOps() # Instance for pi
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(len(wave1_tensor), len(wave2_tensor))
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute correlation
    correlation = np.corrcoef(wave1_tensor, wave2_tensor)[0, 1]

    # Compute phase difference
    fft1 = np.fft.fft(wave1_tensor)
    fft2 = np.fft.fft(wave2_tensor)
    phase1 = np.angle(fft1)
    phase2 = np.angle(fft2)
    phase_diff = np.abs(np.subtract(phase1, phase2))
    mean_phase_diff = np.mean(phase_diff)

    # Normalize phase difference to [0, 1]
    normalized_phase_diff = np.divide(mean_phase_diff, math_ops.pi) # Use math_ops.pi

    # Compute interference strength
    one = np.array(1.0, dtype=default_float) # Use default_float
    interference_strength = np.multiply(correlation, np.subtract(one, normalized_phase_diff))

    return np.array(interference_strength, dtype=default_float) # Return NumPy scalar using default float

def compute_phase_coherence(input_wave1: TensorLike, input_wave2: TensorLike, freq_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
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
        A NumPy scalar representing the phase coherence metric (0.0 to 1.0).
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(len(wave1_tensor), len(wave2_tensor))
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute FFT
    fft1 = np.fft.fft(wave1_tensor)
    fft2 = np.fft.fft(wave2_tensor)

    # Get phases
    phase1 = np.angle(fft1)
    phase2 = np.angle(fft2)

    # Get frequencies
    freqs = np.fft.fftfreq(len(wave1_tensor))

    # Apply frequency range filter if specified
    if freq_range is not None:
        min_freq, max_freq = freq_range
        freq_mask = (np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq)
        phase1 = phase1[freq_mask]
        phase2 = phase2[freq_mask]

    # Compute phase difference
    phase_diff = np.subtract(phase1, phase2)

    # Compute phase coherence using circular statistics
    # Convert phase differences to complex numbers on the unit circle
    complex_one_j = np.array(1j, dtype=np.complex64) # Match complex type if needed
    complex_phase = np.exp(np.multiply(complex_one_j, phase_diff))

    # Compute mean vector length (phase coherence)
    coherence = np.abs(np.mean(complex_phase))

    return np.array(coherence, dtype=default_float) # Return NumPy scalar using default float

def partial_interference(input_wave1: TensorLike, input_wave2: TensorLike, window_size: int = 100) -> np.ndarray:
    """
    Compute the partial interference between two wave signals over sliding windows.

    Calculates interference strength for overlapping windows of the signals.

    Args:
        input_wave1: The first input wave signal.
        input_wave2: The second input wave signal.
        window_size: The size of the sliding window.

    Returns:
        A NumPy array containing the interference strength for each window.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    from ember_ml.backend.numpy.math_ops import NumpyMathOps # Import for pi
    tensor_ops = NumpyTensor()
    math_ops = NumpyMathOps() # Instance for pi
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(len(wave1_tensor), len(wave2_tensor))
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute number of windows
    num_windows = np.add(np.subtract(min_length, window_size), 1)

    # Initialize result array using convert_to_tensor
    interference = tensor_ops.convert_to_tensor([0.0] * num_windows)

    # Compute interference for each window
    for i in range(num_windows):
        window1 = wave1_tensor[i:i + window_size]
        window2 = wave2_tensor[i:i + window_size]

        # Compute correlation
        correlation = np.corrcoef(window1, window2)[0, 1]

        # Compute phase difference
        fft1 = np.fft.fft(window1)
        fft2 = np.fft.fft(window2)
        phase1 = np.angle(fft1)
        phase2 = np.angle(fft2)
        phase_diff = np.abs(np.subtract(phase1, phase2))
        mean_phase_diff = np.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        normalized_phase_diff = np.divide(mean_phase_diff, math_ops.pi) # Use math_ops.pi

        # Compute interference strength
        one = np.array(1.0, dtype=default_float) # Use default_float
        interference[i] = np.multiply(correlation, np.subtract(one, normalized_phase_diff))

    return interference

def euclidean_distance(vector1: TensorLike, vector2: TensorLike) -> np.ndarray:
    """
    Compute the Euclidean (L2) distance between two vectors.

    Args:
        vector1: The first input vector.
        vector2: The second input vector.

    Returns:
        A NumPy scalar representing the Euclidean distance.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_tensor = tensor_ops.convert_to_tensor(vector1)
    y_tensor = tensor_ops.convert_to_tensor(vector2)
    return np.sqrt(np.sum(np.square(np.subtract(x_tensor, y_tensor))))

def cosine_similarity(vector1: TensorLike, vector2: TensorLike) -> np.ndarray:
    """
    Compute the cosine similarity between two vectors.

    Measures the cosine of the angle between two non-zero vectors.
    Result ranges from -1 (exactly opposite) to 1 (exactly the same).

    Args:
        vector1: The first input vector.
        vector2: The second input vector.

    Returns:
        A NumPy scalar representing the cosine similarity.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_tensor = tensor_ops.convert_to_tensor(vector1)
    y_tensor = tensor_ops.convert_to_tensor(vector2)
    dot_product = np.sum(np.multiply(x_tensor, y_tensor))
    norm_x = np.sqrt(np.sum(np.square(x_tensor)))
    norm_y = np.sqrt(np.sum(np.square(y_tensor)))
    denominator = np.add(np.multiply(norm_x, norm_y), 1e-8)
    return np.divide(dot_product, denominator)

def exponential_decay(initial_value: TensorLike, decay_rate: TensorLike, time_step: Optional[TensorLike] = None) -> np.ndarray:
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
        The value(s) after exponential decay as a NumPy array.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    initial_tensor = tensor_ops.convert_to_tensor(initial_value)
    decay_tensor = tensor_ops.convert_to_tensor(decay_rate)

    if time_step is not None:
        # Uniform time step decay
        time_tensor = tensor_ops.convert_to_tensor(time_step)
        decay_factor = np.exp(np.multiply(np.negative(decay_tensor), time_tensor))
    else:
        # Index-based decay
        indices = np.arange(initial_tensor.shape[0], dtype=default_float)
        decay_factor = np.exp(np.multiply(np.negative(decay_tensor), indices))

    return np.multiply(initial_tensor, decay_factor)


# Removed NumpyVectorOps class as it's redundant with standalone functions
