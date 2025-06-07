"""
CPU-Friendly Restricted Boltzmann Machine Implementation

This module provides an efficient implementation of Restricted Boltzmann Machines
optimized for CPU usage with minimal computational requirements.
"""

import time
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from ember_ml.ops import stats
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike # Corrected import for TensorLike
from ember_ml.nn.modules.activations import sigmoid as sigmoid_activation
class RestrictedBoltzmannMachine:
    """
    CPU-friendly implementation of a Restricted Boltzmann Machine.
    
    This implementation focuses on computational efficiency while still
    providing powerful feature learning capabilities. It includes:
    
    - Optimized matrix operations using NumPy
    - Mini-batch training for better memory usage
    - Efficient contrastive divergence with k=1 by default
    - State tracking for visualization and "dreaming" capabilities
    - Anomaly detection through reconstruction error
    """
    
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        batch_size: int = 10,
        use_binary_states: bool = False,
        track_states: bool = True,
        max_tracked_states: int = 50
    ):
        """
        Initialize the RBM with optimized parameters for CPU usage.
        
        Args:
            n_visible: Number of visible units (input features)
            n_hidden: Number of hidden units (learned features)
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for gradient updates
            weight_decay: L2 regularization coefficient
            batch_size: Size of mini-batches for training
            use_binary_states: Whether to use binary states (True) or probabilities (False)
            track_states: Whether to track states for visualization
            max_tracked_states: Maximum number of states to track (to limit memory usage)
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_binary_states = use_binary_states
        self.track_states = track_states
        self.max_tracked_states = max_tracked_states
        
        # Initialize weights and biases
        # Use small random values for weights to break symmetry
        # Scale by 1/sqrt(n_visible) for better initial convergence
        self.weights = tensor.random_normal(0, 0.01 / ops.sqrt(n_visible), (n_visible, n_hidden))
        self.visible_bias = tensor.zeros(n_visible)
        self.hidden_bias = tensor.zeros(n_hidden)
        
        # Initialize momentum terms
        self.weights_momentum = tensor.zeros((n_visible, n_hidden))
        self.visible_bias_momentum = tensor.zeros(n_visible)
        self.hidden_bias_momentum = tensor.zeros(n_hidden)
        
        # For tracking training progress
        self.training_errors: List[float] = [] # Added type hint
        self.training_states: Optional[List[Dict[str, Any]]] = [] if track_states else None # Added type hint
        self.dream_states: List[TensorLike] = [] # Added type hint
        
        # For anomaly detection
        self.reconstruction_error_threshold = None
        self.free_energy_threshold = None
        
        # Training metadata
        self.training_time = 0
        self.n_epochs_trained = 0
        self.last_batch_error = float('inf')
    
    def sigmoid(self, x: TensorLike) -> TensorLike:
        """
        Compute sigmoid function with numerical stability improvements.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid of input array
        """
        # Clip values to avoid overflow/underflow
        x = ops.clip(x, -15, 15)
        return sigmoid_activation(x)
    
    def compute_hidden_probabilities(self, visible_states: TensorLike) -> TensorLike:
        """
        Compute probabilities of hidden units given visible states.
        
        Args:
            visible_states: States of visible units [batch_size, n_visible]
            
        Returns:
            Probabilities of hidden units [batch_size, n_hidden]
        """
        # Compute activations: visible_states @ weights + hidden_bias
        hidden_activations = ops.add(
            ops.dot(visible_states, self.weights),
            self.hidden_bias
        )
        return self.sigmoid(hidden_activations)
    
    def sample_hidden_states(self, hidden_probs: TensorLike) -> TensorLike:
        """
        Sample binary hidden states from their probabilities.
        
        Args:
            hidden_probs: Probabilities of hidden units [batch_size, n_hidden]
            
        Returns:
            Binary hidden states [batch_size, n_hidden]
        """
        if not self.use_binary_states:
            return hidden_probs
        
        return tensor.cast(
            ops.greater(hidden_probs, tensor.random_uniform(tensor.shape(hidden_probs))),
            dtype=tensor.float32
        )
    
    def compute_visible_probabilities(self, hidden_states: TensorLike) -> TensorLike:
        """
        Compute probabilities of visible units given hidden states.
        
        Args:
            hidden_states: States of hidden units [batch_size, n_hidden]
            
        Returns:
            Probabilities of visible units [batch_size, n_visible]
        """
        # Compute activations: hidden_states @ weights.T + visible_bias
        visible_activations = ops.add(
            ops.dot(hidden_states, tensor.transpose(self.weights)),
            self.visible_bias
        )
        return self.sigmoid(visible_activations)
    
    def sample_visible_states(self, visible_probs: TensorLike) -> TensorLike:
        """
        Sample binary visible states from their probabilities.
        
        Args:
            visible_probs: Probabilities of visible units [batch_size, n_visible]
            
        Returns:
            Binary visible states [batch_size, n_visible]
        """
        if not self.use_binary_states:
            return visible_probs
        
        return tensor.cast(
            ops.greater(visible_probs, tensor.random_uniform(tensor.shape(visible_probs))),
            dtype=tensor.float32
        )
    
    def contrastive_divergence(self, batch_data: TensorLike, k: int = 1) -> float:
        """
        Perform contrastive divergence algorithm for a single batch.
        
        This is an efficient implementation of CD-k that minimizes
        memory usage and computational complexity.
        
        Args:
            batch_data: Batch of training data [batch_size, n_visible]
            k: Number of Gibbs sampling steps (default: 1)
            
        Returns:
            Reconstruction error for this batch
        """
        batch_size = tensor.shape(batch_data)[0]
        
        # Positive phase
        # Compute hidden probabilities and states
        pos_hidden_probs = self.compute_hidden_probabilities(batch_data)
        pos_hidden_states = self.sample_hidden_states(pos_hidden_probs)
        
        # Compute positive associations
        pos_associations = ops.dot(tensor.transpose(batch_data), pos_hidden_probs)
        
        # Negative phase
        # Start with the hidden states from positive phase
        neg_hidden_states = tensor.copy(pos_hidden_states)
        
        # Perform k steps of Gibbs sampling
        for _ in range(k):
            # Compute visible probabilities and states
            neg_visible_probs = self.compute_visible_probabilities(neg_hidden_states)
            neg_visible_states = self.sample_visible_states(neg_visible_probs)
            
            # Compute hidden probabilities and states
            neg_hidden_probs = self.compute_hidden_probabilities(neg_visible_states)
            neg_hidden_states = self.sample_hidden_states(neg_hidden_probs)
        
        # Compute negative associations
        neg_associations = ops.dot(tensor.transpose(neg_visible_states), neg_hidden_probs)
        
        # Compute gradients
        weights_gradient = ops.divide(ops.subtract(pos_associations, neg_associations),
                                     tensor.convert_to_tensor(batch_size, dtype=tensor.float32))
        visible_bias_gradient = stats.mean(ops.subtract(batch_data, neg_visible_states), axis=0)
        hidden_bias_gradient = stats.mean(ops.subtract(pos_hidden_probs, neg_hidden_probs), axis=0)
        
        # Update with momentum and weight decay
        self.weights_momentum = ops.add(
            ops.multiply(self.momentum, self.weights_momentum),
            weights_gradient
        )
        self.visible_bias_momentum = ops.add(
            ops.multiply(self.momentum, self.visible_bias_momentum),
            visible_bias_gradient
        )
        self.hidden_bias_momentum = ops.add(
            ops.multiply(self.momentum, self.hidden_bias_momentum),
            hidden_bias_gradient
        )
        
        # Apply updates
        self.weights = ops.add(
            self.weights,
            ops.multiply(
                self.learning_rate,
                ops.subtract(
                    self.weights_momentum,
                    ops.multiply(self.weight_decay, self.weights)
                )
            )
        )
        self.visible_bias = ops.add(
            self.visible_bias,
            ops.multiply(self.learning_rate, self.visible_bias_momentum)
        )
        self.hidden_bias = ops.add(
            self.hidden_bias,
            ops.multiply(self.learning_rate, self.hidden_bias_momentum)
        )
        
        # Compute reconstruction error
        squared_diff = ops.square(ops.subtract(batch_data, neg_visible_probs))
        reconstruction_error = stats.mean(stats.sum(squared_diff, axis=1))
        reconstruction_error_value = tensor.item(reconstruction_error)
        
        # Track state if enabled
        if self.track_states and self.training_states is not None and len(self.training_states) < self.max_tracked_states: # Added None check
            self.training_states.append({
                'weights': tensor.copy(self.weights),
                'visible_bias': tensor.copy(self.visible_bias),
                'hidden_bias': tensor.copy(self.hidden_bias),
                'error': reconstruction_error_value,
                'visible_sample': tensor.copy(neg_visible_states[0]) if batch_size > 0 else None,
                'hidden_sample': tensor.copy(neg_hidden_states[0]) if batch_size > 0 else None
            })
        
        return reconstruction_error_value
    
    def train(
        self,
        data: TensorLike,
        epochs: int = 10,
        k: int = 1,
        validation_data: Optional[TensorLike] = None,
        early_stopping_patience: int = 5,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the RBM on the provided data.
        
        Args:
            data: Training data [n_samples, n_visible]
            epochs: Number of training epochs
            k: Number of Gibbs sampling steps
            validation_data: Optional validation data for early stopping
            early_stopping_patience: Number of epochs to wait for improvement
            verbose: Whether to print progress
            
        Returns:
            List of reconstruction errors per epoch
        """
        n_samples = tensor.shape(data)[0]
        n_batches = max(n_samples // self.batch_size, 1)
        
        start_time = time.time()
        best_validation_error = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = tensor.random_permutation(n_samples)
            shuffled_data = tensor.gather(data, indices)
            
            epoch_error: float = 0.0 # Initialized as float
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, n_samples)
                batch = shuffled_data[batch_start:batch_end]
                
                # Skip empty batches
                if tensor.shape(batch)[0] == 0:
                    continue
                
                # Train on batch
                batch_error = self.contrastive_divergence(batch, k)
                epoch_error += batch_error
                self.last_batch_error = batch_error
            
            # Compute average epoch error
            avg_epoch_error = epoch_error / n_batches
            self.training_errors.append(avg_epoch_error)
            self.n_epochs_trained += 1
            
            # Check validation error if provided
            validation_error = None
            if validation_data is not None:
                validation_error = self.reconstruction_error(validation_data)
                
                # Early stopping check
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                val_str = f", validation error: {validation_error:.4f}" if validation_data is not None else ""
                print(f"Epoch {epoch+1}/{epochs}: reconstruction error = {avg_epoch_error:.4f}{val_str}")
        
        self.training_time += time.time() - start_time
        
        # Compute threshold for anomaly detection based on training data
        if self.reconstruction_error_threshold is None:
            errors = self.reconstruction_error(data, per_sample=True)
            # Use ops.stats.percentile when available, for now use a simple threshold
            self.reconstruction_error_threshold = stats.mean(errors) + 2 * stats.std(errors)
        
        if self.free_energy_threshold is None:
            energies = self.free_energy(data)
            # Use ops.stats.percentile when available, for now use a simple threshold
            self.free_energy_threshold = stats.mean(energies) - 2 * stats.std(energies)
        
        return self.training_errors
    
    def transform(self, data: TensorLike) -> TensorLike:
        """
        Transform data to hidden representation.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Hidden representation [n_samples, n_hidden]
        """
        return self.compute_hidden_probabilities(data)
    
    def reconstruct(self, data: TensorLike) -> TensorLike:
        """
        Reconstruct input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Reconstructed data [n_samples, n_visible]
        """
        hidden_probs = self.compute_hidden_probabilities(data)
        hidden_states = self.sample_hidden_states(hidden_probs)
        visible_probs = self.compute_visible_probabilities(hidden_states)
        return visible_probs
    
    def reconstruction_error(self, data: TensorLike, per_sample: bool = False) -> Union[float, TensorLike]:
        """
        Compute reconstruction error for input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            per_sample: Whether to return error per sample
            
        Returns:
            Reconstruction error (mean or per sample)
        """
        reconstructed = self.reconstruct(data)
        squared_error = stats.sum(ops.square(ops.subtract(data, reconstructed)), axis=1)
        
        if per_sample:
            return squared_error
        
        return tensor.item(stats.mean(squared_error))
    
    def free_energy(self, data: TensorLike) -> TensorLike:
        """
        Compute free energy for input data.
        
        The free energy is a measure of how well the RBM models the data.
        Lower values indicate better fit.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Free energy for each sample [n_samples]
        """
        visible_bias_term = ops.dot(data, self.visible_bias)
        hidden_activations = ops.add(ops.dot(data, self.weights), self.hidden_bias)
        hidden_term = stats.sum(
            ops.log(ops.add(1.0, ops.exp(hidden_activations))),
            axis=1
        )
        
        return ops.negative(ops.add(hidden_term, visible_bias_term))
    
    def anomaly_score(self, data: TensorLike, method: str = 'reconstruction') -> TensorLike:
        """
        Compute anomaly score for input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            method: Method to use ('reconstruction' or 'free_energy')
            
        Returns:
            Anomaly scores [n_samples]
        """
        if method == 'reconstruction':
            return self.reconstruction_error(data, per_sample=True)
        elif method == 'free_energy':
            # For free energy, lower is better, so we negate
            return ops.negative(self.free_energy(data))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def is_anomaly(self, data: TensorLike, method: str = 'reconstruction') -> TensorLike:
        """
        Determine if input data is anomalous.
        
        Args:
            data: Input data [n_samples, n_visible]
            method: Method to use ('reconstruction' or 'free_energy')
            
        Returns:
            Boolean array indicating anomalies [n_samples]
        """
        scores = self.anomaly_score(data, method)
        
        if method == 'reconstruction':
            return ops.greater(scores, self.reconstruction_error_threshold)
        elif method == 'free_energy':
            return ops.less(scores, self.free_energy_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def dream(self, n_steps: int = 100, start_data: Optional[TensorLike] = None) -> List[TensorLike]:
        """
        Generate a sequence of "dream" states by Gibbs sampling.
        
        This is the RBM's generative mode, where it "dreams" by
        sampling from its learned distribution.
        
        Args:
            n_steps: Number of Gibbs sampling steps
            start_data: Optional starting data (if None, random initialization)
            
        Returns:
            List of visible states during dreaming
        """
        # Initialize visible state
        if start_data is not None:
            visible_state = tensor.copy(start_data)
        else:
            visible_state = tensor.random_uniform((1, self.n_visible))
        
        # Clear previous dream states
        self.dream_states = []
        
        # Perform Gibbs sampling
        for _ in range(n_steps):
            # Compute hidden probabilities and sample states
            hidden_probs = self.compute_hidden_probabilities(visible_state)
            hidden_state = self.sample_hidden_states(hidden_probs)
            
            # Compute visible probabilities and sample states
            visible_probs = self.compute_visible_probabilities(hidden_state)
            visible_state = self.sample_visible_states(visible_probs)
            
            # Store state
            self.dream_states.append(tensor.copy(visible_state))
        
        return self.dream_states
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert tensors to numpy for saving
        weights_np = tensor.to_numpy(self.weights)
        visible_bias_np = tensor.to_numpy(self.visible_bias)
        hidden_bias_np = tensor.to_numpy(self.hidden_bias)
        
        # Prepare model data
        model_data = {
            'weights': weights_np,
            'visible_bias': visible_bias_np,
            'hidden_bias': hidden_bias_np,
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'use_binary_states': self.use_binary_states,
            'training_errors': self.training_errors,
            'reconstruction_error_threshold': tensor.to_numpy(self.reconstruction_error_threshold) if self.reconstruction_error_threshold is not None else None,
            'free_energy_threshold': tensor.to_numpy(self.free_energy_threshold) if self.free_energy_threshold is not None else None,
            'training_time': self.training_time,
            'n_epochs_trained': self.n_epochs_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        # Use ops.save when available, for now use numpy
        ops.save(filepath, model_data, allow_pickle=True) # Replaced ops.save with ops.save
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RestrictedBoltzmannMachine':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded RBM model
        """
        # Use ops.load when available, for now use numpy
        import numpy as np
        model_data = ops.load(filepath, allow_pickle=True).item()
        
        # Create model
        rbm = cls(
            n_visible=model_data['n_visible'],
            n_hidden=model_data['n_hidden'],
            learning_rate=model_data['learning_rate'],
            momentum=model_data['momentum'],
            weight_decay=model_data['weight_decay'],
            batch_size=model_data['batch_size'],
            use_binary_states=model_data['use_binary_states']
        )
        
        # Set model parameters
        rbm.weights = tensor.convert_to_tensor(model_data['weights'])
        rbm.visible_bias = tensor.convert_to_tensor(model_data['visible_bias'])
        rbm.hidden_bias = tensor.convert_to_tensor(model_data['hidden_bias'])
        rbm.training_errors = model_data['training_errors']
        rbm.reconstruction_error_threshold = tensor.convert_to_tensor(model_data['reconstruction_error_threshold']) if model_data['reconstruction_error_threshold'] is not None else None
        rbm.free_energy_threshold = tensor.convert_to_tensor(model_data['free_energy_threshold']) if model_data['free_energy_threshold'] is not None else None
        rbm.training_time = model_data['training_time']
        rbm.n_epochs_trained = model_data['n_epochs_trained']
        
        return rbm
    
    def summary(self) -> str:
        """
        Get a summary of the model.
        
        Returns:
            Summary string
        """
        summary = [
            "Restricted Boltzmann Machine Summary",
            "====================================",
            f"Visible units: {self.n_visible}",
            f"Hidden units: {self.n_hidden}",
            f"Parameters: {self.n_visible * self.n_hidden + self.n_visible + self.n_hidden}",
            f"Learning rate: {self.learning_rate}",
            f"Momentum: {self.momentum}",
            f"Weight decay: {self.weight_decay}",
            f"Batch size: {self.batch_size}",
            f"Binary states: {self.use_binary_states}",
            f"Epochs trained: {self.n_epochs_trained}",
            f"Training time: {self.training_time:.2f} seconds",
            f"Current reconstruction error: {self.last_batch_error:.4f}",
            f"Anomaly threshold (reconstruction): {self.reconstruction_error_threshold}",
            f"Anomaly threshold (free energy): {self.free_energy_threshold}"
        ]
        
        return "\n".join(summary)


class RBM:
    """
    PyTorch implementation of a Restricted Boltzmann Machine.
    
    This implementation uses PyTorch tensors and operations for GPU acceleration.
    """
    
    def __init__(
        self,
        visible_size: int,
        hidden_size: int,
        device: str = "cpu"
    ):
        """
        Initialize the RBM.
        
        Args:
            visible_size: Number of visible units
            hidden_size: Number of hidden units
            device: Device to place tensors on ('cpu', 'cuda', or 'mps')
        """
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Initialize weights and biases
        self.weights = ops.multiply(tensor.random_normal(visible_size, hidden_size, device=device), 0.1)
        self.visible_bias = tensor.zeros(visible_size, device=device)
        self.hidden_bias = tensor.zeros(hidden_size, device=device)
    
    def forward(self, x):
        """
        Forward pass (visible to hidden).
        
        Args:
            x: Visible layer activations
            
        Returns:
            (hidden_probs, hidden_states)
        """
        # Compute hidden activations
        from ember_ml.nn.modules.activations import sigmoid as sigmoid_activation
        hidden_activations = ops.matmul(x, self.weights) + self.hidden_bias
        hidden_probs = sigmoid_activation(hidden_activations)
        hidden_states = tensor.random_bernoulli(hidden_probs)
        
        return hidden_probs, hidden_states
    
    def backward(self, h):
        """
        Backward pass (hidden to visible).
        
        Args:
            h: Hidden layer activations
            
        Returns:
            (visible_probs, visible_states)
        """
        # Compute visible activations
        visible_activations = ops.matmul(h, self.weights.t()) + self.visible_bias
        visible_probs = sigmoid_activation(visible_activations)
        visible_states = tensor.random_bernoulli(visible_probs)
        
        return visible_probs, visible_states
    
    def free_energy(self, v):
        """
        Compute the free energy of a visible vector.
        
        Args:
            v: Visible vector
            
        Returns:
            Free energy
        """
        vbias_term = ops.matmul(v, self.visible_bias)
        wx_b = ops.matmul(v, self.weights) + self.hidden_bias
        hidden_term = stats.sum(ops.log(ops.add(1, ops.exp(wx_b))), dim=1)
        
        return -hidden_term - vbias_term
    
    def contrastive_divergence(self, v_pos, k=1, learning_rate=0.1):
        """
        Perform k-step contrastive divergence.
        
        Args:
            v_pos: Positive phase visible units
            k: Number of Gibbs sampling steps
            learning_rate: Learning rate
            
        Returns:
            None
        """
        # Positive phase
        h_pos_probs, h_pos = self.forward(v_pos)
        
        # Negative phase
        h_neg = h_pos.clone()
        v_neg = v_pos.clone()
        
        for _ in range(k):
            v_neg_probs, v_neg = self.backward(h_neg)
            h_neg_probs, h_neg = self.forward(v_neg)
        
        # Compute gradients
        pos_associations = ops.matmul(v_pos.t(), h_pos_probs)
        neg_associations = ops.matmul(v_neg.t(), h_neg_probs)
        
        # Update parameters
        self.weights += learning_rate * (pos_associations - neg_associations) / v_pos.size(0)
        self.visible_bias += learning_rate * stats.mean(v_pos - v_neg, dim=0)
        self.hidden_bias += learning_rate * stats.mean(h_pos_probs - h_neg_probs, dim=0)
    
    def train(self, data, epochs=10, batch_size=10, learning_rate=0.1, k=1):
        """
        Train the RBM.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            k: Number of Gibbs sampling steps
            
        Returns:
            List of reconstruction errors
        """
        data = tensor.convert_to_tensor(data, device=self.device)
        n_samples = data.size(0)
        n_batches = n_samples // batch_size
        
        errors = []
        
        for epoch in range(epochs):
            epoch_error = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            data = data[indices]
            
            for i in range(n_batches):
                batch = data[i*batch_size:(i+1)*batch_size]
                self.contrastive_divergence(batch, k, learning_rate)
                
                # Compute reconstruction error
                h_probs, h_states = self.forward(batch)
                v_probs, v_states = self.backward(h_states)
                batch_error = stats.mean(stats.sum((batch - v_probs) ** 2, dim=1))
                epoch_error += batch_error.item()
            
            # Average error over batches
            avg_error = epoch_error / n_batches
            errors.append(avg_error)
            
            print(f"Epoch {epoch+1}/{epochs}: error = {avg_error:.4f}")
        
        return errors