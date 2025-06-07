import matplotlib.pyplot as plt
import numpy as np

class HarmonicVisualizer:
    """Class to handle visualization of embeddings and harmonic waves."""
    
    @staticmethod
    def plot_embeddings_comparison(target_embeddings, learned_waves, figsize=(12, 6)):
        """
        Visualize target embeddings and learned harmonic embeddings side by side.
        
        Args:
            target_embeddings (TensorLike): Original embeddings
            learned_waves (TensorLike): Generated harmonic waves
            figsize (tuple): Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Plot target embeddings
        plt.subplot(211)
        plt.imshow(target_embeddings, aspect="auto", cmap="viridis")
        plt.title("Target Embeddings")
        plt.colorbar()
        
        # Plot learned harmonic embeddings
        plt.subplot(212)
        plt.imshow(learned_waves, aspect="auto", cmap="viridis")
        plt.title("Learned Harmonic Embeddings")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(history, figsize=(10, 5)):
        """
        Plot training loss history.
        
        Args:
            history (list): List of loss values during training
            figsize (tuple): Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        plt.plot(history, 'b-')
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_wave_components(params, t, n_components=3, figsize=(12, 8)):
        """
        Visualize individual harmonic components for a single embedding.
        
        Args:
            params (TensorLike): Wave parameters for a single embedding
            t (TensorLike): Time points
            n_components (int): Number of components to plot
            figsize (tuple): Figure size (width, height)
        """
        amplitudes, frequencies, phases = tensor.split(params, 3)
        
        plt.figure(figsize=figsize)
        
        # Plot individual components
        for i in range(min(n_components, len(amplitudes))):
            wave = amplitudes[i] * ops.sin(2 * ops.pi * frequencies[i] * t + phases[i])
            plt.plot(t, wave, label=f'Component {i+1}')
        
        # Plot combined wave
        combined = sum(amplitudes[i] * ops.sin(2 * ops.pi * frequencies[i] * t + phases[i])
                      for i in range(len(amplitudes)))
        plt.plot(t, combined, 'k--', linewidth=2, label='Combined Wave')
        
        plt.title('Harmonic Wave Components')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_embedding_similarity(target_embeddings, learned_waves, figsize=(8, 6)):
        """
        Plot similarity matrix between target embeddings and learned waves.
        
        Args:
            target_embeddings (TensorLike): Original embeddings
            learned_waves (TensorLike): Generated harmonic waves
            figsize (tuple): Figure size (width, height)
        """
        # Compute cosine similarity
        similarity = np.corrcoef(target_embeddings, learned_waves)
        
        plt.figure(figsize=figsize)
        plt.imshow(similarity, cmap='coolwarm', aspect='auto')
        plt.title('Embedding Similarity Matrix')
        plt.colorbar(label='Correlation')
        plt.xlabel('Learned Wave Index')
        plt.ylabel('Target Embedding Index')
        plt.show()