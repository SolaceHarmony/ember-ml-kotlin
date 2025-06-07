"""
Backend-agnostic Restricted Boltzmann Machine Implementation

This module provides an efficient implementation of Restricted Boltzmann Machines
that can use different backends (NumPy, PyTorch, MLX) for tensor operations.
"""

import time
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Union, Any

from ember_ml import ops
from ember_ml.nn import tensor

class RBM:
    """
    Backend-agnostic implementation of a Restricted Boltzmann Machine.
    
    This implementation can use different backends (NumPy, PyTorch, MLX)
    for tensor operations, allowing it to leverage hardware acceleration
    when available.
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
        max_tracked_states: int = 50,
        backend: str = '',
        device: str = ''
    ):
        """
        Initialize the RBM with optimized parameters.
        
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
            backend: Backend to use ('numpy', 'torch', 'mlx', or None for default)
            device: Device to use (e.g., 'cpu', 'cuda', 'mps', or None for default)
        """
        # Set backend if specified
        if backend is not None:
            ops.set_backend(backend)
            
        self.backend = ops.get_backend()
        self.device = device
        
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
        self.weights = tensor.random_normal(
            (n_visible, n_hidden), 
            mean=0.0, 
            stddev=ops.div(0.01, ops.sqrt(tensor.convert_to_tensor(n_visible)))
        )
        self.visible_bias = tensor.zeros(n_visible)
        self.hidden_bias = tensor.zeros(n_hidden)
 
        # Initialize momentum terms
        self.weights_momentum = tensor.zeros_like(self.weights)
        self.visible_bias_momentum = tensor.zeros_like(self.visible_bias)
        self.hidden_bias_momentum = tensor.zeros_like(self.hidden_bias)
        
        # For tracking training progress
        self.training_errors: List[float] = []
        self.training_states: Optional[List[Dict[str, Any]]] = [] if track_states else None
        self.dream_states: List[tensor.EmberTensor] = []
        
        # For anomaly detection
        self.reconstruction_error_threshold: Optional[float] = None
        self.free_energy_threshold: Optional[float] = None
        
        # Training metadata
        self.training_time: float = 0.0
        self.n_epochs_trained: int = 0
        self.last_batch_error: float = float('inf')
    
    def sigmoid(self, x):
            """
            Compute sigmoid function with numerical stability improvements.
            
            Args:
                x: Input tensor
                
            Returns:
                Sigmoid of input tensor
            """
            return ops.sigmoid(ops.clip(x, -15, 15))
    
    def compute_hidden_probabilities(self, visible_states):
        """
        Compute probabilities of hidden units given visible states.
        
        Args:
            visible_states: States of visible units [batch_size, n_visible]
            
        Returns:
            Probabilities of hidden units [batch_size, n_hidden]
        """
        hidden_activations = ops.add(ops.dot(visible_states, self.weights), self.hidden_bias)
        return self.sigmoid(hidden_activations)
    
    def sample_hidden_states(self, hidden_probs):
        """
        Sample binary hidden states from their probabilities.
        
        Args:
            hidden_probs: Probabilities of hidden units [batch_size, n_hidden]
            
        Returns:
            Binary hidden states [batch_size, n_hidden]
        """
        if not self.use_binary_states:
            return hidden_probs
        
        random_values = tensor.random_uniform(tensor.shape(hidden_probs))
        return tensor.cast(ops.greater(hidden_probs, random_values), hidden_probs.dtype)
    
    def compute_visible_probabilities(self, hidden_states):
        """
        Compute probabilities of visible units given hidden states.
        
        Args:
            hidden_states: States of hidden units [batch_size, n_hidden]
            
        Returns:
            Probabilities of visible units [batch_size, n_visible]
        """
        visible_activations = ops.add(ops.dot(hidden_states, tensor.transpose(self.weights)), self.visible_bias)
        return self.sigmoid(visible_activations)
    
    def sample_visible_states(self, visible_probs):
        """
        Sample binary visible states from their probabilities.
        
        Args:
            visible_probs: Probabilities of visible units [batch_size, n_visible]
            
        Returns:
            Binary visible states [batch_size, n_visible]
        """
        if not self.use_binary_states:
            return visible_probs
        
        random_values = tensor.random_uniform(tensor.shape(visible_probs))
        return tensor.cast(ops.greater(visible_probs, random_values), visible_probs.dtype)
    
    def contrastive_divergence(self, batch_data, k=1):
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
        batch_size = len(batch_data)
        
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
        weights_gradient = ops.divide(ops.subtract(pos_associations, neg_associations), batch_size)
        visible_bias_gradient = ops.stats.mean(ops.subtract(batch_data, neg_visible_states), axis=0)
        hidden_bias_gradient = ops.stats.mean(ops.subtract(pos_hidden_probs, neg_hidden_probs), axis=0)
        
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
            ops.subtract(
                ops.multiply(self.learning_rate, self.weights_momentum),
                ops.multiply(self.learning_rate * self.weight_decay, self.weights)
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
        reconstruction_error = ops.stats.mean(
            stats.sum(ops.pow(ops.subtract(batch_data, neg_visible_probs), 2), axis=1)
        )
        
        # Track state if enabled
        if self.track_states and len(self.training_states) < self.max_tracked_states:
            self.training_states.append({
                'weights': tensor.to_numpy(self.weights),
                'visible_bias': tensor.to_numpy(self.visible_bias),
                'hidden_bias': tensor.to_numpy(self.hidden_bias),
                'error': float(tensor.to_numpy(reconstruction_error)),
                'visible_sample': tensor.to_numpy(neg_visible_states[0]) if batch_size > 0 else None,
                'hidden_sample': tensor.to_numpy(neg_hidden_states[0]) if batch_size > 0 else None
            })
        
        return reconstruction_error
    
    def train(
        self,
        data,
        epochs: int = 10,
        k: int = 1,
        validation_data = None,
        early_stopping_patience: int = 5,
        verbose: bool = True
    ):
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
        # Convert data to tensors
        data = tensor.convert_to_tensor(data)
        if validation_data is not None:
            validation_data = tensor.convert_to_tensor(validation_data)
        
        n_samples = len(data)
        n_batches = max(n_samples // self.batch_size, 1)
        
        start_time = time.time()
        best_validation_error = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = tensor.to_numpy(tensor.random_uniform((n_samples,)))
            indices = tensor.convert_to_tensor(indices.argsort())
            shuffled_data = tensor.convert_to_tensor([data[i] for i in tensor.to_numpy(indices)])
            
            epoch_error = 0
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, n_samples)
                batch = shuffled_data[batch_start:batch_end]
                
                # Skip empty batches
                if len(batch) == 0:
                    continue
                
                # Train on batch
                batch_error = self.contrastive_divergence(batch, k)
                epoch_error += tensor.to_numpy(batch_error)
                self.last_batch_error = float(tensor.to_numpy(batch_error))
            
            # Compute average epoch error
            avg_epoch_error = epoch_error / n_batches
            self.training_errors.append(avg_epoch_error)
            self.n_epochs_trained += 1
            
            # Check validation error if provided
            validation_error = None
            if validation_data is not None:
                validation_error = float(tensor.to_numpy(self.reconstruction_error(validation_data)))
                
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
            self.reconstruction_error_threshold = float(tensor.to_numpy(
                tensor.convert_to_tensor(sorted(tensor.to_numpy(errors)))[int(0.95 * len(errors))]
            ))
        
        if self.free_energy_threshold is None:
            energies = self.free_energy(data)
            self.free_energy_threshold = float(tensor.to_numpy(
                tensor.convert_to_tensor(sorted(tensor.to_numpy(energies)))[int(0.05 * len(energies))]
            ))
        
        return self.training_errors
    
    def transform(self, data):
        """
        Transform data to hidden representation.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Hidden representation [n_samples, n_hidden]
        """
        data = tensor.convert_to_tensor(data)
        return self.compute_hidden_probabilities(data)
    
    def reconstruct(self, data):
        """
        Reconstruct input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Reconstructed data [n_samples, n_visible]
        """
        data = tensor.convert_to_tensor(data)
        hidden_probs = self.compute_hidden_probabilities(data)
        hidden_states = self.sample_hidden_states(hidden_probs)
        visible_probs = self.compute_visible_probabilities(hidden_states)
        return visible_probs
    
    def reconstruction_error(self, data, per_sample=False):
        """
        Compute reconstruction error for input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            per_sample: Whether to return error per sample
            
        Returns:
            Reconstruction error (mean or per sample)
        """
        data = tensor.convert_to_tensor(data)
        reconstructed = self.reconstruct(data)
        squared_error = stats.sum(ops.pow(ops.subtract(data, reconstructed), 2), axis=1)
        
        if per_sample:
            return squared_error
        
        return ops.stats.mean(squared_error)
    
    def free_energy(self, data):
        """
        Compute free energy for input data.
        
        The free energy is a measure of how well the RBM models the data.
        Lower values indicate better fit.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Free energy for each sample [n_samples]
        """
        data = tensor.convert_to_tensor(data)
        visible_bias_term = ops.dot(data, self.visible_bias)
        hidden_term = stats.sum(
            ops.log(ops.add(1, ops.exp(ops.add(ops.dot(data, self.weights), self.hidden_bias)))),
            axis=1
        )
        
        return ops.subtract(ops.negative(hidden_term), visible_bias_term)
    
    def anomaly_score(self, data, method='reconstruction'):
        """
        Compute anomaly score for input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            method: Method to use ('reconstruction' or 'free_energy')
            
        Returns:
            Anomaly scores [n_samples]
        """
        data = tensor.convert_to_tensor(data)
        
        if method == 'reconstruction':
            return self.reconstruction_error(data, per_sample=True)
        elif method == 'free_energy':
            # For free energy, lower is better, so we negate
            return ops.negative(self.free_energy(data))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def is_anomaly(self, data, method='reconstruction'):
        """
        Determine if input data is anomalous.
        
        Args:
            data: Input data [n_samples, n_visible]
            method: Method to use ('reconstruction' or 'free_energy')
            
        Returns:
            Boolean array indicating anomalies [n_samples]
        """
        data = tensor.convert_to_tensor(data)
        scores = self.anomaly_score(data, method)
        
        if method == 'reconstruction':
            return ops.greater(scores, self.reconstruction_error_threshold)
        elif method == 'free_energy':
            return ops.less(scores, self.free_energy_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def dream(self, n_steps=100, start_data=None):
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
            visible_state = tensor.convert_to_tensor(start_data)
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
            self.dream_states.append(tensor.to_numpy(visible_state))
        
        return self.dream_states
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'weights': tensor.to_numpy(self.weights),
            'visible_bias': tensor.to_numpy(self.visible_bias),
            'hidden_bias': tensor.to_numpy(self.hidden_bias),
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'use_binary_states': self.use_binary_states,
            'training_errors': self.training_errors,
            'reconstruction_error_threshold': self.reconstruction_error_threshold,
            'free_energy_threshold': self.free_energy_threshold,
            'training_time': self.training_time,
            'n_epochs_trained': self.n_epochs_trained,
            'backend': self.backend,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file using NumPy
        import numpy as np
        ops.save(filepath, model_data, allow_pickle=True)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, backend=None, device=None):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            backend: Backend to use (default: same as saved model)
            device: Device to use (default: None)
            
        Returns:
            Loaded RBM model
        """
        # Load model data
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
            use_binary_states=model_data['use_binary_states'],
            backend=backend or model_data.get('backend', 'numpy'),
            device=device
        )
        
        # Set model parameters
        rbm.weights = tensor.convert_to_tensor(model_data['weights'])
        rbm.visible_bias = tensor.convert_to_tensor(model_data['visible_bias'])
        rbm.hidden_bias = tensor.convert_to_tensor(model_data['hidden_bias'])
        rbm.training_errors = model_data['training_errors']
        rbm.reconstruction_error_threshold = model_data['reconstruction_error_threshold']
        rbm.free_energy_threshold = model_data['free_energy_threshold']
        rbm.training_time = model_data['training_time']
        rbm.n_epochs_trained = model_data['n_epochs_trained']
        
        # Move to device if specified
        if device:
            rbm.weights = ops.to_device(rbm.weights, device)
            rbm.visible_bias = ops.to_device(rbm.visible_bias, device)
            rbm.hidden_bias = ops.to_device(rbm.hidden_bias, device)
        
        return rbm
    
    def summary(self):
        """
        Get a summary of the model.
        
        Returns:
            Summary string
        """
        summary = [
            "Restricted Boltzmann Machine Summary",
            "====================================",
            f"Backend: {self.backend}",
            f"Device: {self.device or 'default'}",
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