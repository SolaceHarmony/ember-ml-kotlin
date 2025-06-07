import numpy as np
from .wave_generator import harmonic_wave, map_embeddings_to_harmonics

class HarmonicTrainer:
    """Class to handle training of harmonic wave parameters to match embeddings."""
    
    def __init__(self, learning_rate=0.01, epochs=100, epsilon=1e-5):
        """
        Initialize the trainer with hyperparameters.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            epochs (int): Number of training epochs
            epsilon (float): Small value for numerical gradient computation
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        
    def loss_function(self, params, t, target_embedding):
        """
        Compute the Mean Squared Error loss between target embedding and generated wave.
        
        Args:
            params (TensorLike): Wave parameters (amplitudes, frequencies, phases)
            t (TensorLike): Time points
            target_embedding (TensorLike): Target embedding to match
            
        Returns:
            float: MSE loss value
        """
        amplitudes, frequencies, phases = tensor.split(params, 3)
        harmonic = (
            amplitudes[:, None] * ops.sin(2 * ops.pi * frequencies[:, None] * t + phases[:, None])
        ).sum(axis=0)
        
        return ((target_embedding - harmonic) ** 2).mean()
    
    def compute_gradients(self, params, t, target_embedding):
        """
        Compute numerical gradients for the harmonic parameters using finite differences.
        
        Args:
            params (TensorLike): Current parameter values
            t (TensorLike): Time points
            target_embedding (TensorLike): Target embedding to match
            
        Returns:
            TensorLike: Computed gradients for all parameters
        """
        gradients = tensor.zeros_like(params)
        for i in range(len(params)):
            params_step = params.copy()
            
            # Positive perturbation
            params_step[i] += self.epsilon
            loss_plus = self.loss_function(params_step, t, target_embedding)
            
            # Negative perturbation
            params_step[i] -= 2 * self.epsilon
            loss_minus = self.loss_function(params_step, t, target_embedding)
            
            # Compute gradient
            gradients[i] = (loss_plus - loss_minus) / (2 * self.epsilon)
            
        return gradients
    
    def train(self, embeddings, t):
        """
        Train harmonic wave parameters to match transformer embeddings.
        
        Args:
            embeddings (TensorLike): Target embeddings of shape (batch_size, embedding_dim)
            t (TensorLike): Time points for wave generation
            
        Returns:
            TensorLike: Trained parameters
            list: Training history (losses)
        """
        batch_size = embeddings.shape[0]
        params = map_embeddings_to_harmonics(embeddings)
        history = []
        
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(batch_size):
                # Compute loss
                loss = self.loss_function(params[i], t, embeddings[i])
                
                # Compute gradients
                gradients = self.compute_gradients(params[i], t, embeddings[i])
                
                # Update parameters
                params[i] -= self.learning_rate * gradients
                
                # Accumulate loss
                total_loss += loss
                
            avg_loss = total_loss / batch_size
            history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:  # Print progress every 10 epochs
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
                
        return params, history
    
    def generate_learned_waves(self, params, t):
        """
        Generate harmonic waves using the learned parameters.
        
        Args:
            params (TensorLike): Learned parameters
            t (TensorLike): Time points
            
        Returns:
            TensorLike: Generated harmonic waves
        """
        batch_size = params.shape[0]
        return harmonic_wave(params, t, batch_size)