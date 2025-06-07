import numpy as np

def harmonic_wave(params, t, batch_size):
    """
    Generate a harmonic wave based on parameters.
    Handles batch processing for multiple embeddings.
    
    Args:
        params (TensorLike): Array of shape (batch_size, 3*n_components) containing
                           amplitudes, frequencies, and phases for each component
        t (TensorLike): Time points at which to evaluate the wave
        batch_size (int): Number of waves to generate in parallel
        
    Returns:
        TensorLike: Generated harmonic waves of shape (batch_size, len(t))
    """
    harmonics = []
    for i in range(batch_size):
        amplitudes, frequencies, phases = tensor.split(params[i], 3)
        harmonic = (
            amplitudes[:, None] * ops.sin(2 * ops.pi * frequencies[:, None] * t + phases[:, None])
        )
        harmonics.append(harmonic.sum(axis=0))
    return tensor.vstack(harmonics)

def map_embeddings_to_harmonics(embeddings):
    """
    Initialize harmonic parameters for all embeddings in a batch.
    
    Args:
        embeddings (TensorLike): Input embeddings of shape (batch_size, embedding_dim)
        
    Returns:
        TensorLike: Initialized parameters of shape (batch_size, 3*embedding_dim)
                   containing amplitudes, frequencies, and phases
    """
    batch_size, embedding_dim = embeddings.shape
    params = []
    for i in range(batch_size):
        params.append(np.random.rand(3 * embedding_dim))  # Amplitudes, Frequencies, Phases
    return tensor.vstack(params)