"""
PyTorch implementation of vector operations.

This module provides PyTorch implementations of vector operations including FFT transforms
and wave signal analysis functions.
"""

import torch
from typing import Optional, Tuple, Sequence # Remove unused List, Union, Any
from ember_ml.backend.torch.types import TensorLike, Shape, Axis, default_float


def normalize_vector(input_vector: TensorLike) -> torch.Tensor:
    """
    Normalize an input vector to unit length (L2 norm).

    If the vector's norm is zero, the original vector is returned.

    Args:
        input_vector: The vector to normalize.

    Returns:
        The normalized vector as a PyTorch tensor.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    vector_tensor = tensor_ops.convert_to_tensor(input_vector)
    norm = torch.linalg.norm(vector_tensor)
    if norm > 0:
        return torch.divide(vector_tensor, norm)
    return vector_tensor


def euclidean_distance(vector1: TensorLike, vector2: TensorLike) -> torch.Tensor:
    """
    Compute the Euclidean (L2) distance between two vectors.

    Args:
        vector1: The first input vector.
        vector2: The second input vector.

    Returns:
        A PyTorch scalar tensor representing the Euclidean distance.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(vector1)
    y_tensor = tensor_ops.convert_to_tensor(vector2)
    return torch.sqrt(torch.sum(torch.square(torch.subtract(x_tensor, y_tensor))))

def cosine_similarity(vector1: TensorLike, vector2: TensorLike) -> torch.Tensor:
    """
    Compute the cosine similarity between two vectors.

    Measures the cosine of the angle between two non-zero vectors.
    Result ranges from -1 (exactly opposite) to 1 (exactly the same).

    Args:
        vector1: The first input vector.
        vector2: The second input vector.

    Returns:
        A PyTorch scalar tensor representing the cosine similarity.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(vector1)
    y_tensor = tensor_ops.convert_to_tensor(vector2)

    dot_product = torch.sum(torch.multiply(x_tensor, y_tensor))
    norm_x = torch.sqrt(torch.sum(torch.square(x_tensor)))
    norm_y = torch.sqrt(torch.sum(torch.square(y_tensor)))
    return torch.divide(dot_product, torch.add(torch.multiply(norm_x, norm_y), tensor_ops.convert_to_tensor(1e-8)))

def exponential_decay(initial_value: TensorLike, decay_rate: TensorLike, time_step: Optional[TensorLike] = None) -> torch.Tensor:
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
        The value(s) after exponential decay as a PyTorch tensor.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    initial_tensor = tensor_ops.convert_to_tensor(initial_value)
    decay_tensor = tensor_ops.convert_to_tensor(decay_rate)
    if time_step is not None:
        # Uniform time step decay
        time_tensor = tensor_ops.convert_to_tensor(time_step)
        decay_factor = torch.exp(torch.multiply(torch.negative(decay_tensor), time_tensor))
    else:
        # Index-based decay
        indices = torch.arange(initial_tensor.shape[0], dtype=default_float, device=initial_tensor.device) # Match device and dtype
        decay_factor = torch.exp(torch.multiply(torch.negative(decay_tensor), indices))

    return torch.multiply(initial_tensor, decay_factor)

def compute_energy_stability(input_wave: TensorLike, window_size: int = 100) -> torch.Tensor:
    """
    Compute the energy stability of a wave signal.

    Calculates stability based on the variance of energy across sliding windows.
    A value closer to 1.0 indicates higher stability.

    Args:
        input_wave: The input wave signal.
        window_size: The size of the sliding window used for energy calculation.

    Returns:
        A PyTorch scalar tensor representing the energy stability metric (0.0 to 1.0).
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    wave_tensor = tensor_ops.convert_to_tensor(input_wave)
    if len(wave_tensor.shape) == 0 or wave_tensor.shape[0] < window_size:
        return torch.tensor(1.0, dtype=default_float) # Return torch scalar

    # Compute energy in windows
    # Ensure integer division for range
    num_windows = wave_tensor.shape[0] // window_size
    energies = []
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = wave_tensor[start:end]
        energy = torch.sum(torch.square(window))
        energies.append(energy.item())

    if len(energies) <= 1:
        return torch.tensor(1.0, dtype=default_float) # Return torch scalar

    energies_tensor = tensor_ops.convert_to_tensor(energies)
    energy_mean = torch.mean(energies_tensor)
    
    if energy_mean == 0:
        return torch.tensor(1.0, dtype=default_float) # Return torch scalar

    energy_var = torch.var(energies_tensor)
    stability = torch.divide(tensor_ops.convert_to_tensor(1.0), 
                            torch.add(tensor_ops.convert_to_tensor(1.0), 
                                    torch.divide(energy_var, energy_mean)))

    return stability.to(dtype=default_float) # Return torch scalar with correct dtype

def compute_interference_strength(input_wave1: TensorLike, input_wave2: TensorLike) -> torch.Tensor:
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
        A PyTorch scalar tensor representing the interference strength metric.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    from ember_ml.backend.torch.math_ops import pi
    tensor_ops = TorchTensor()
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute correlation
    wave1_mean = torch.mean(wave1_tensor)
    wave2_mean = torch.mean(wave2_tensor)
    wave1_centered = torch.subtract(wave1_tensor, wave1_mean)
    wave2_centered = torch.subtract(wave2_tensor, wave2_mean)

    numerator = torch.sum(torch.multiply(wave1_centered, wave2_centered))
    denominator = torch.multiply(
        torch.sqrt(torch.sum(torch.square(wave1_centered))),
        torch.sqrt(torch.sum(torch.square(wave2_centered)))
    )
    denominator = torch.add(denominator, tensor_ops.convert_to_tensor(1e-8))
    correlation = torch.divide(numerator, denominator)

    # Compute phase difference using FFT
    fft1 = torch.fft.fft(wave1_tensor)
    fft2 = torch.fft.fft(wave2_tensor)
    phase1 = torch.angle(fft1)
    phase2 = torch.angle(fft2)
    phase_diff = torch.abs(torch.subtract(phase1, phase2))
    mean_phase_diff = torch.mean(phase_diff)

    # Normalize phase difference to [0, 1]
    pi_tensor = tensor_ops.convert_to_tensor(pi)
    normalized_phase_diff = torch.divide(mean_phase_diff, pi_tensor)

    # Compute interference strength
    interference_strength = torch.multiply(
        correlation, 
        torch.subtract(tensor_ops.convert_to_tensor(1.0), normalized_phase_diff)
    )

    return interference_strength.to(dtype=default_float) # Return torch scalar with correct dtype

def compute_phase_coherence(input_wave1: TensorLike, input_wave2: TensorLike,
                            freq_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
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
        A PyTorch scalar tensor representing the phase coherence metric (0.0 to 1.0).
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    from ember_ml.backend.torch.math_ops import pi
    tensor_ops = TorchTensor()
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute FFT
    fft1 = torch.fft.fft(wave1_tensor)
    fft2 = torch.fft.fft(wave2_tensor)

    # Get phases
    phase1 = torch.angle(fft1)
    phase2 = torch.angle(fft2)

    # Get frequencies
    freqs = torch.fft.fftfreq(len(wave1_tensor))

    # Apply frequency range filter if specified
    if freq_range is not None:
        min_freq, max_freq = freq_range
        min_freq_tensor = tensor_ops.convert_to_tensor(min_freq)
        max_freq_tensor = tensor_ops.convert_to_tensor(max_freq)
        freq_mask = torch.logical_and(
            torch.greater_equal(torch.abs(freqs), min_freq_tensor),
            torch.less_equal(torch.abs(freqs), max_freq_tensor)
        )
        phase1 = torch.where(freq_mask, phase1, torch.zeros_like(phase1))
        phase2 = torch.where(freq_mask, phase2, torch.zeros_like(phase2))

    # Compute phase difference
    phase_diff = torch.subtract(phase1, phase2)

    # Use Euler's formula for complex phase calculation
    complex_real = torch.cos(phase_diff)
    complex_imag = torch.sin(phase_diff)
    coherence = torch.sqrt(torch.add(
        torch.square(torch.mean(complex_real)),
        torch.square(torch.mean(complex_imag))
    ))

    return coherence.to(dtype=default_float) # Return torch scalar with correct dtype

def partial_interference(input_wave1: TensorLike, input_wave2: TensorLike, window_size: int = 100) -> torch.Tensor:
    """
    Compute the partial interference between two wave signals over sliding windows.

    Calculates interference strength for overlapping windows of the signals.

    Args:
        input_wave1: The first input wave signal.
        input_wave2: The second input wave signal.
        window_size: The size of the sliding window.

    Returns:
        A PyTorch tensor containing the interference strength for each window.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    from ember_ml.backend.torch.math_ops import pi
    tensor_ops = TorchTensor()
    wave1_tensor = tensor_ops.convert_to_tensor(input_wave1)
    wave2_tensor = tensor_ops.convert_to_tensor(input_wave2)

    # Ensure waves are the same length
    min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]

    # Compute number of windows
    num_windows = min_length - window_size + 1 # Python ops okay for range
    interference = []

    # Compute interference for each window
    for i in range(num_windows):
        # Python ops okay for index calculation
        start = i * window_size
        end = start + window_size
        window1 = wave1_tensor[start:end]
        window2 = wave2_tensor[start:end]

        # Compute correlation
        window1_mean = torch.mean(window1)
        window2_mean = torch.mean(window2)
        window1_centered = torch.subtract(window1, window1_mean)
        window2_centered = torch.subtract(window2, window2_mean)

        correlation = torch.divide(
            torch.sum(torch.multiply(window1_centered, window2_centered)),
            torch.add(torch.multiply(
                torch.sqrt(torch.sum(torch.square(window1_centered))),
                torch.sqrt(torch.sum(torch.square(window2_centered)))
            ), tensor_ops.convert_to_tensor(1e-8))
        )

        # Compute FFT for this window
        fft1 = torch.fft.fft(window1)
        fft2 = torch.fft.fft(window2)

        # Get phases
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)
        phase_diff = torch.abs(torch.subtract(phase1, phase2))
        mean_phase_diff = torch.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        pi_tensor = tensor_ops.convert_to_tensor(pi)
        normalized_phase_diff = torch.divide(mean_phase_diff, pi_tensor)

        # Compute interference strength
        interference.append(torch.multiply(
            correlation, 
            torch.subtract(tensor_ops.convert_to_tensor(1.0), normalized_phase_diff)
        ))

    return tensor_ops.convert_to_tensor(interference)

def fft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> torch.Tensor:
    """
    One dimensional discrete Fourier Transform.

    Args:
        input_array: Input array
        output_length: Length of the transformed axis
        axis: Axis over which to compute the FFT

    Returns:
        The transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    return torch.fft.fft(a_tensor, n=output_length, dim=axis)

def ifft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> torch.Tensor:
    """
    One dimensional inverse discrete Fourier Transform.

    Args:
        input_array: Input array
        output_length: Length of the transformed axis
        axis: Axis over which to compute the IFFT

    Returns:
        The inverse transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    return torch.fft.ifft(a_tensor, n=output_length, dim=axis)

def fft2(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = (-2, -1)) -> torch.Tensor:
    """
    Two dimensional discrete Fourier Transform.

    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the FFT2

    Returns:
        The transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    # PyTorch fft2 dim expects Tuple[int, ...]
    current_axes = axes if isinstance(axes, tuple) else (axes,) if isinstance(axes, int) else (-2,-1)
    return torch.fft.fft2(a_tensor, s=output_shape, dim=current_axes)

def ifft2(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = (-2, -1)) -> torch.Tensor:
    """
    Two dimensional inverse discrete Fourier Transform.

    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IFFT2

    Returns:
        The inverse transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    # PyTorch ifft2 dim expects Tuple[int, ...]
    current_axes = axes if isinstance(axes, tuple) else (axes,) if isinstance(axes, int) else (-2,-1)
    return torch.fft.ifft2(a_tensor, s=output_shape, dim=current_axes)

def fftn(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = None) -> torch.Tensor:
    """
    N-dimensional discrete Fourier Transform.

    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the FFTN

    Returns:
        The transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    # PyTorch fftn dim expects Optional[List[int]]
    current_axes = list(axes) if isinstance(axes, Sequence) else [axes] if isinstance(axes, int) else None
    return torch.fft.fftn(a_tensor, s=output_shape, dim=current_axes)

def ifftn(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = None) -> torch.Tensor:
    """
    N-dimensional inverse discrete Fourier Transform.

    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IFFTN

    Returns:
        The inverse transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    # PyTorch ifftn dim expects Optional[List[int]]
    current_axes = list(axes) if isinstance(axes, Sequence) else [axes] if isinstance(axes, int) else None
    return torch.fft.ifftn(a_tensor, s=output_shape, dim=current_axes)

def rfft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> torch.Tensor:
    """
    One dimensional real discrete Fourier Transform.

    Args:
        input_array: Input array (real)
        output_length: Length of the transformed axis
        axis: Axis over which to compute the RFFT

    Returns:
        The transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    return torch.fft.rfft(a_tensor, n=output_length, dim=axis)

def irfft(input_array: TensorLike, output_length: Optional[int] = None, axis: int = -1) -> torch.Tensor:
    """
    One dimensional inverse real discrete Fourier Transform.

    Args:
        input_array: Input array
        output_length: Length of the transformed axis
        axis: Axis over which to compute the IRFFT

    Returns:
        The inverse transformed array (real)
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    return torch.fft.irfft(a_tensor, n=output_length, dim=axis)

def rfft2(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = (-2, -1)) -> torch.Tensor:
    """
    Two dimensional real discrete Fourier Transform.

    Args:
        input_array: Input array (real)
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the RFFT2

    Returns:
        The transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    current_axes = axes if isinstance(axes, tuple) else (axes,) if isinstance(axes, int) else (-2,-1)
    return torch.fft.rfft2(a_tensor, s=output_shape, dim=current_axes)

def irfft2(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = (-2, -1)) -> torch.Tensor:
    """
    Two dimensional inverse real discrete Fourier Transform.

    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IRFFT2

    Returns:
        The inverse transformed array (real)
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    current_axes = axes if isinstance(axes, tuple) else (axes,) if isinstance(axes, int) else (-2,-1)
    return torch.fft.irfft2(a_tensor, s=output_shape, dim=current_axes)

def rfftn(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = None) -> torch.Tensor:
    """
    N-dimensional real discrete Fourier Transform.

    Args:
        input_array: Input array (real)
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the RFFTN

    Returns:
        The transformed array
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    current_axes = list(axes) if isinstance(axes, Sequence) else [axes] if isinstance(axes, int) else None
    return torch.fft.rfftn(a_tensor, s=output_shape, dim=current_axes)

def irfftn(input_array: TensorLike, output_shape: Optional[Shape] = None,
            axes: Axis = None) -> torch.Tensor:
    """
    N-dimensional inverse real discrete Fourier Transform.

    Args:
        input_array: Input array
        output_shape: Shape of the transformed axes
        axes: Axes over which to compute the IRFFTN

    Returns:
        The inverse transformed array (real)
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    a_tensor = tensor_ops.convert_to_tensor(input_array)
    current_axes = list(axes) if isinstance(axes, Sequence) else [axes] if isinstance(axes, int) else None
    return torch.fft.irfftn(a_tensor, s=output_shape, dim=current_axes)


# Removed TorchVectorOps class as it's redundant with standalone functions
