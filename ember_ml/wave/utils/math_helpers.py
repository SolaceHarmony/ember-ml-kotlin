"""
Math helper functions for wave processing.

This module provides math helper functions for wave processing,
using the ops abstraction layer for backend-agnostic operations.
"""

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor
# Define math functions using ops abstraction layer
sigmoid = lambda x: ops.sigmoid(tensor.convert_to_tensor(x))
tanh = lambda x: ops.tanh(tensor.convert_to_tensor(x))
relu = lambda x: ops.relu(tensor.convert_to_tensor(x))
def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU of a tensor."""
    x_tensor = tensor.convert_to_tensor(x)
    alpha_tensor = tensor.convert_to_tensor(alpha)
    return ops.where(
        ops.greater(x_tensor, tensor.convert_to_tensor(0.0)),
        x_tensor,
        ops.multiply(alpha_tensor, x_tensor)
    )
softmax = lambda x, axis=-1: ops.softmax(tensor.convert_to_tensor(x), axis=axis)

def normalize(x, axis=-1):
    """Normalize a tensor along the specified axis."""
    x_tensor = tensor.convert_to_tensor(x)
    norm = ops.sqrt(stats.sum(ops.square(x_tensor), axis=axis, keepdims=True))
    return ops.divide(x_tensor, ops.add(norm, tensor.convert_to_tensor(1e-8)))

def standardize(x, axis=-1):
    """Standardize a tensor to have zero mean and unit variance."""
    x_tensor = tensor.convert_to_tensor(x)
    mean = ops.stats.mean(x_tensor, axis=axis, keepdims=True)
    std = ops.sqrt(ops.stats.mean(ops.square(ops.subtract(x_tensor, mean)), axis=axis, keepdims=True))
    return ops.divide(ops.subtract(x_tensor, mean), ops.add(std, tensor.convert_to_tensor(1e-8)))

def euclidean_distance(x, y):
    """Compute the Euclidean distance between two vectors."""
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    return ops.sqrt(stats.sum(ops.square(ops.subtract(x_tensor, y_tensor))))

def cosine_similarity(x, y):
    """Compute the cosine similarity between two vectors."""
    x_tensor = tensor.convert_to_tensor(x)
    y_tensor = tensor.convert_to_tensor(y)
    dot_product = stats.sum(ops.multiply(x_tensor, y_tensor))
    norm_x = ops.sqrt(stats.sum(ops.square(x_tensor)))
    norm_y = ops.sqrt(stats.sum(ops.square(y_tensor)))
    return ops.divide(dot_product, ops.add(ops.multiply(norm_x, norm_y), tensor.convert_to_tensor(1e-8)))

def exponential_decay(initial_value, decay_rate, time_step):
    """Compute exponential decay."""
    return ops.multiply(initial_value, ops.exp(ops.multiply(ops.negative(decay_rate), time_step)))

def gaussian(x, mu=0.0, sigma=1.0):
    """Compute the Gaussian function."""
    x_tensor = tensor.convert_to_tensor(x)
    mu_tensor = tensor.convert_to_tensor(mu)
    sigma_tensor = tensor.convert_to_tensor(sigma)
    return ops.divide(
        ops.exp(ops.multiply(tensor.convert_to_tensor(-0.5), ops.square(ops.divide(ops.subtract(x_tensor, mu_tensor), sigma_tensor)))),
        ops.multiply(sigma_tensor, ops.sqrt(ops.multiply(tensor.convert_to_tensor(2.0), ops.pi)))
    )

def normalize_vector(vector):
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    return linearalg.normalize_vector(vector)

def compute_energy_stability(wave, window_size: int = 100) -> float:
    """
    Compute the energy stability of a wave signal.
    
    Args:
        wave: Wave signal
        window_size: Window size for stability computation
        
    Returns:
        Energy stability metric
    """
    wave_tensor = tensor.convert_to_tensor(wave)
    wave_length = tensor.shape(wave_tensor)[0]
    
    if wave_length < window_size:
        return 1.0  # Perfectly stable for short signals
        
    # Compute energy in windows
    num_windows = tensor.cast(ops.floor_divide(wave_length, tensor.convert_to_tensor(window_size)), tensor.int32)
    energies = []
    
    for i in range(tensor.item(num_windows)):
        start = ops.multiply(tensor.convert_to_tensor(i), tensor.convert_to_tensor(window_size))
        end = ops.add(start, tensor.convert_to_tensor(window_size))
        window = wave_tensor[start:end]
        energy = stats.sum(ops.square(window))
        energies.append(tensor.item(energy))
    
    # Convert energies to tensor
    energies_tensor = tensor.convert_to_tensor(energies)
    
    # Compute stability as inverse of energy variance
    if len(energies) <= 1:
        return 1.0
        
    energy_mean = ops.stats.mean(energies_tensor)
    if tensor.item(energy_mean) == 0:
        return 1.0
        
    energy_var = tensor.var(energies_tensor)
    stability = 1.0 / (1.0 + tensor.item(energy_var) / tensor.item(energy_mean))
    
    return stability

def compute_interference_strength(wave1, wave2) -> float:
    """
    Compute the interference strength between two wave signals.
    
    Args:
        wave1: First wave signal
        wave2: Second wave signal
        
    Returns:
        Interference strength metric
    """
    # Convert to tensors
    wave1_tensor = tensor.convert_to_tensor(wave1)
    wave2_tensor = tensor.convert_to_tensor(wave2)
    
    # Ensure waves are the same length
    wave1_length = tensor.shape(wave1_tensor)[0]
    wave2_length = tensor.shape(wave2_tensor)[0]
    min_length = min(tensor.item(wave1_length), tensor.item(wave2_length))
    
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]
    
    # Compute correlation
    # Since ops doesn't have a direct corrcoef function, we'll compute it manually
    wave1_mean = ops.stats.mean(wave1_tensor)
    wave2_mean = ops.stats.mean(wave2_tensor)
    wave1_centered = ops.subtract(wave1_tensor, wave1_mean)
    wave2_centered = ops.subtract(wave2_tensor, wave2_mean)
    
    numerator = stats.sum(ops.multiply(wave1_centered, wave2_centered))
    denominator = ops.sqrt(ops.multiply(
        stats.sum(ops.square(wave1_centered)),
        stats.sum(ops.square(wave2_centered))
    ))
    
    correlation = ops.divide(numerator, ops.add(denominator, tensor.convert_to_tensor(1e-8)))
    
    # For phase difference, we would need FFT operations
    # Since ops doesn't have FFT functions, we'll use a simplified approach
    # This is a placeholder and should be replaced with proper FFT operations
    # when they become available in ops
    
    # For now, we'll use a simplified phase difference calculation
    phase_diff = ops.stats.mean(ops.abs(ops.subtract(
        ops.divide(wave1_tensor, ops.add(ops.abs(wave1_tensor), tensor.convert_to_tensor(1e-8))),
        ops.divide(wave2_tensor, ops.add(ops.abs(wave2_tensor), tensor.convert_to_tensor(1e-8)))
    )))
    
    # Normalize phase difference to [0, 1]
    normalized_phase_diff = ops.divide(phase_diff, ops.pi)
    
    # Compute interference strength
    interference_strength = ops.multiply(
        correlation,
        ops.subtract(tensor.convert_to_tensor(1.0), normalized_phase_diff)
    )
    
    return tensor.item(interference_strength)

def compute_phase_coherence(wave1, wave2, freq_range=None) -> float:
    """
    Compute the phase coherence between two wave signals.
    
    Args:
        wave1: First wave signal
        wave2: Second wave signal
        freq_range: Optional frequency range to consider (min_freq, max_freq)
        
    Returns:
        Phase coherence metric
    """
    # This function requires FFT operations which are not available in ops
    # We'll implement a simplified version that approximates phase coherence
    
    # Convert to tensors
    wave1_tensor = tensor.convert_to_tensor(wave1)
    wave2_tensor = tensor.convert_to_tensor(wave2)
    
    # Ensure waves are the same length
    wave1_length = tensor.shape(wave1_tensor)[0]
    wave2_length = tensor.shape(wave2_tensor)[0]
    min_length = min(tensor.item(wave1_length), tensor.item(wave2_length))
    
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]
    
    # Compute a simplified phase coherence metric
    # This is a placeholder and should be replaced with proper FFT-based
    # phase coherence calculation when FFT operations become available in ops
    
    # Normalize the waves
    wave1_norm = ops.divide(wave1_tensor, ops.add(ops.sqrt(ops.stats.mean(ops.square(wave1_tensor))), tensor.convert_to_tensor(1e-8)))
    wave2_norm = ops.divide(wave2_tensor, ops.add(ops.sqrt(ops.stats.mean(ops.square(wave2_tensor))), tensor.convert_to_tensor(1e-8)))
    
    # Compute dot product as a measure of coherence
    coherence = ops.abs(ops.stats.mean(ops.multiply(wave1_norm, wave2_norm)))
    
    return tensor.item(coherence)

def partial_interference(wave1, wave2, window_size: int = 100):
    """
    Compute the partial interference between two wave signals over sliding windows.
    
    Args:
        wave1: First wave signal
        wave2: Second wave signal
        window_size: Size of the sliding window
        
    Returns:
        Array of partial interference values for each window
    """
    # Convert to tensors
    wave1_tensor = tensor.convert_to_tensor(wave1)
    wave2_tensor = tensor.convert_to_tensor(wave2)
    
    # Ensure waves are the same length
    wave1_length = tensor.shape(wave1_tensor)[0]
    wave2_length = tensor.shape(wave2_tensor)[0]
    min_length = min(tensor.item(wave1_length), tensor.item(wave2_length))
    
    wave1_tensor = wave1_tensor[:min_length]
    wave2_tensor = wave2_tensor[:min_length]
    
    # Compute number of windows
    num_windows = tensor.cast(ops.subtract(ops.add(min_length, 1), window_size), tensor.int32)
    
    # Initialize result array
    interference = tensor.zeros((num_windows,))
    
    # Compute interference for each window
    for i in range(num_windows):
        window1 = wave1_tensor[i:i+window_size]
        window2 = wave2_tensor[i:i+window_size]
        
        # Compute correlation
        window1_mean = ops.stats.mean(window1)
        window2_mean = ops.stats.mean(window2)
        window1_centered = ops.subtract(window1, window1_mean)
        window2_centered = ops.subtract(window2, window2_mean)
        
        numerator = stats.sum(ops.multiply(window1_centered, window2_centered))
        denominator = ops.sqrt(ops.multiply(
            stats.sum(ops.square(window1_centered)),
            stats.sum(ops.square(window2_centered))
        ))
        
        correlation = ops.divide(numerator, ops.add(denominator, tensor.convert_to_tensor(1e-8)))
        
        # Simplified phase difference calculation
        phase_diff = ops.stats.mean(ops.abs(ops.subtract(
            ops.divide(window1, ops.add(ops.abs(window1), tensor.convert_to_tensor(1e-8))),
            ops.divide(window2, ops.add(ops.abs(window2), tensor.convert_to_tensor(1e-8)))
        )))
        
        # Normalize phase difference to [0, 1]
        normalized_phase_diff = ops.divide(phase_diff, ops.pi)
        
        # Compute interference strength
        interference_val = ops.multiply(
            correlation,
            ops.subtract(tensor.convert_to_tensor(1.0), normalized_phase_diff)
        )
        
        interference = tensor.tensor_scatter_nd_update(
            interference,
            tensor.convert_to_tensor([[i]]),
            tensor.convert_to_tensor([tensor.item(interference_val)])
        )
    
    return interference

__all__ = [
    'sigmoid',
    'tanh',
    'relu',
    'leaky_relu',
    'softmax',
    'normalize',
    'standardize',
    'euclidean_distance',
    'cosine_similarity',
    'exponential_decay',
    'gaussian',
    'normalize_vector',
    'compute_energy_stability',
    'compute_interference_strength',
    'compute_phase_coherence',
    'partial_interference'
]