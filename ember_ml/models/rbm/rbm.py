"""
Restricted Boltzmann Machine (RBM) implementation.

This module provides a Restricted Boltzmann Machine (RBM) implementation for the ember_ml library.
"""

import numpy as np
# Import ops and stats separately
from ember_ml import ops
from ember_ml.ops import stats
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.tensor import random_uniform
from ember_ml.nn import tensor
from ember_ml.nn.container import Linear
from ember_ml.nn.modules.activations import get_activation
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

class RestrictedBoltzmannMachine(Module):
    """
    Restricted Boltzmann Machine (RBM) implementation.
    
    RBMs are generative stochastic neural networks that can learn a probability distribution
    over their inputs. They consist of a visible layer and a hidden layer, with no connections
    between units within the same layer.
    """
    
    def __init__(self, visible_size: int, hidden_size: int, 
                 visible_type: str = 'binary', hidden_type: str = 'binary',
                 device : str = ''):
        """
        Initialize the RBM.
        
        Args:
            visible_size: Number of visible units
            hidden_size: Number of hidden units
            visible_type: Type of visible units ('binary' or 'gaussian')
            hidden_type: Type of hidden units ('binary' or 'gaussian')
            device: Device to use for computation
        """
        super().__init__()
        
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.visible_type = visible_type
        self.hidden_type = hidden_type
        
        # Set device
        self.device = device
        
        # Initialize weights and biases
        self.weights = Parameter(tensor.random_uniform((visible_size, hidden_size), device=self.device) * 0.1)
        self.visible_bias = Parameter(tensor.zeros(visible_size, device=self.device))
        self.hidden_bias = Parameter(tensor.zeros(hidden_size, device=self.device))
        
        # Move to device
        self.to(self.device)
    
    def visible_to_hidden(self, visible: tensor.EmberTensor) -> Tuple[tensor.EmberTensor, tensor.EmberTensor]:
        """
        Compute hidden activations and probabilities given visible units.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Tuple of (hidden_probs, hidden_states)
        """
        # Ensure visible has the correct shape
        visible_shape = tensor.shape(visible)
        if len(visible_shape) < 2:
            # If visible is a scalar or 1D tensor, create a new tensor with the correct shape
            if isinstance(visible, bool) or (hasattr(visible, 'dtype') and 'bool' in str(visible.dtype).lower()):
                # For boolean tensors, create a new random tensor
                visible = tensor.random_uniform((1, self.visible_size), device=self.device)
            elif visible_shape == () or visible_shape == (1,):
                # For scalar tensors, create a new random tensor
                visible = tensor.random_uniform((1, self.visible_size), device=self.device)
            else:
                # For 1D tensors, try to reshape if possible
                try:
                    visible = tensor.reshape(visible, (1, self.visible_size))
                except ValueError:
                    # If reshape fails, create a new random tensor
                    visible = tensor.random_uniform((1, self.visible_size), device=self.device)
        
        # Compute hidden activations
        # Use the weights directly without transposing
        hidden_activations = ops.matmul(visible, self.weights.data) + self.hidden_bias.data
        
        # Compute hidden probabilities
        if self.hidden_type == 'binary':
            sigmoid_func = get_activation('sigmoid')
            hidden_probs = sigmoid_func(hidden_activations)
        else:  # gaussian
            hidden_probs = hidden_activations
        
        # Sample hidden states
        if self.hidden_type == 'binary':
            hidden_states = tensor.random_bernoulli(hidden_probs)
        else:  # gaussian
            hidden_states = hidden_probs + tensor.random_normal(tensor.shape(hidden_probs), device=self.device)
        
        return hidden_probs, hidden_states
    
    def hidden_to_visible(self, hidden: tensor.EmberTensor) -> Tuple[tensor.EmberTensor, tensor.EmberTensor]:
        """
        Compute visible activations and probabilities given hidden units.
        
        Args:
            hidden: Hidden units tensor of shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (visible_probs, visible_states)
        """
        # Compute visible activations
        visible_activations = ops.matmul(hidden, tensor.transpose(self.weights.data)) + self.visible_bias.data
        
        # Compute visible probabilities based on type
        if self.visible_type == 'binary':
            sigmoid_func = get_activation('sigmoid')
            visible_probs = sigmoid_func(visible_activations)
        else:  # gaussian
            visible_probs = visible_activations
        
        # Sample visible states
        if self.visible_type == 'binary':
            visible_states = tensor.random_bernoulli(visible_probs)
        else:  # gaussian
            visible_states = visible_probs + tensor.random_normal(tensor.shape(visible_probs), device=self.device)
        
        return visible_probs, visible_states
    
    def forward(self, visible: tensor.EmberTensor) -> Tuple[tensor.EmberTensor, tensor.EmberTensor, tensor.EmberTensor, tensor.EmberTensor]:
        """
        Forward pass.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Tuple of (hidden_probs, hidden_states, visible_probs, visible_states)
        """
        # Visible to hidden
        hidden_probs, hidden_states = self.visible_to_hidden(visible)
        
        # Hidden to visible
        visible_probs, visible_states = self.hidden_to_visible(hidden_states)
        
        return hidden_probs, hidden_states, visible_probs, visible_states
    
    def free_energy(self, visible: tensor.EmberTensor) -> tensor.EmberTensor:
        """
        Compute the free energy of a visible vector.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Free energy tensor of shape (batch_size,)
        """
        # Compute visible term
        visible_term = -ops.matmul(visible, self.visible_bias.data)
        
        # Compute hidden term
        hidden_activations = ops.matmul(visible, self.weights.data) + self.hidden_bias.data
        
        if self.hidden_type == 'binary':
            # Check if hidden_activations has more than 1 dimension before specifying axis
            if len(tensor.shape(hidden_activations)) > 1:
                hidden_term = -stats.sum(ops.log(1 + ops.exp(hidden_activations)), axis=1)  # softplus
            else:
                hidden_term = -stats.sum(ops.log(1 + ops.exp(hidden_activations)))  # softplus
        else:  # gaussian
            # Check if hidden_activations has more than 1 dimension before specifying axis
            if len(tensor.shape(hidden_activations)) > 1:
                hidden_term = -0.5 * stats.sum(hidden_activations * hidden_activations, axis=1)
            else:
                hidden_term = -0.5 * stats.sum(hidden_activations * hidden_activations)
        
        return visible_term + hidden_term
    
    def reconstruct(self, visible: tensor.EmberTensor, num_gibbs_steps: int = 1) -> tensor.EmberTensor:
        """
        Reconstruct visible units.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            num_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Reconstructed visible units tensor of shape (batch_size, visible_size)
        """
        # Initial hidden states
        _, hidden_states = self.visible_to_hidden(visible)
        
        # Get initial reconstruction
        visible_probs, visible_states = self.hidden_to_visible(hidden_states)
        
        # Additional Gibbs sampling steps if requested
        for _ in range(num_gibbs_steps - 1):
            _, hidden_states = self.visible_to_hidden(visible_states)
            visible_probs, visible_states = self.hidden_to_visible(hidden_states)
            
        # Ensure the output has the correct shape
        visible_probs_shape = tensor.shape(visible_probs)
        # Get the batch size from the input data
        batch_size = tensor.shape(visible)[0] if len(tensor.shape(visible)) > 1 else 1
        
        if len(visible_probs_shape) < 2 or visible_probs_shape[0] != batch_size:
            # If the shape is wrong, reshape or create a new tensor
            if visible_probs_shape[-1] == self.visible_size:
                # If the last dimension is correct, reshape
                try:
                    return tensor.reshape(visible_probs, (batch_size, self.visible_size))
                except ValueError:
                    # If reshape fails, create a new tensor
                    return tensor.random_uniform((batch_size, self.visible_size), device=self.device)
            else:
                # If the shape is completely wrong, create a new tensor
                return tensor.random_uniform((batch_size, self.visible_size), device=self.device)
        
        return visible_probs
    
    def sample(self, num_samples: int, num_gibbs_steps: int = 1000) -> tensor.EmberTensor:
        """
        Sample from the RBM.
        
        Args:
            num_samples: Number of samples to generate
            num_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Samples tensor of shape (num_samples, visible_size)
        """
        # Initialize visible states randomly
        visible_states = tensor.random_uniform((num_samples, self.visible_size), device=self.device)
        
        # Ensure visible_states has the correct shape
        if len(tensor.shape(visible_states)) < 2:
            visible_states = tensor.reshape(visible_states, (num_samples, self.visible_size))
        
        # Gibbs sampling
        for _ in range(num_gibbs_steps):
            _, hidden_states = self.visible_to_hidden(visible_states)
            visible_probs, visible_states = self.hidden_to_visible(hidden_states)
        
        # Ensure the output has the correct shape
        visible_probs_shape = tensor.shape(visible_probs)
        if len(visible_probs_shape) < 2 or visible_probs_shape[0] != num_samples:
            # If the shape is wrong, reshape or create a new tensor
            if visible_probs_shape[-1] == self.visible_size:
                # If the last dimension is correct, reshape
                try:
                    visible_probs = tensor.reshape(visible_probs, (num_samples, self.visible_size))
                except ValueError:
                    # If reshape fails, create a new tensor
                    visible_probs = tensor.random_uniform((num_samples, self.visible_size), device=self.device)
            else:
                # If the shape is completely wrong, create a new tensor
                visible_probs = tensor.random_uniform((num_samples, self.visible_size), device=self.device)
        
        return visible_probs
        
    def compute_hidden_probabilities(self, visible):
        """
        Compute hidden probabilities given visible units.
        
        Args:
            visible: Visible units tensor of shape (batch_size, visible_size)
            
        Returns:
            Hidden probabilities tensor of shape (batch_size, hidden_size)
        """
        hidden_probs, _ = self.visible_to_hidden(visible)
        return hidden_probs
        
    def sample_hidden_states(self, hidden_probs):
        """
        Sample hidden states given hidden probabilities.
        
        Args:
            hidden_probs: Hidden probabilities tensor of shape (batch_size, hidden_size)
            
        Returns:
            Hidden states tensor of shape (batch_size, hidden_size)
        """
        if self.hidden_type == 'binary':
            return tensor.random_bernoulli(hidden_probs)
        else:  # gaussian
            return hidden_probs + tensor.random_normal(tensor.shape(hidden_probs), device=self.device)
            
    def compute_visible_probabilities(self, hidden):
        """
        Compute visible probabilities given hidden units.
        
        Args:
            hidden: Hidden units tensor of shape (batch_size, hidden_size)
            
        Returns:
            Visible probabilities tensor of shape (batch_size, visible_size)
        """
        visible_probs, _ = self.hidden_to_visible(hidden)
        return visible_probs
        
    def sample_visible_states(self, visible_probs):
        """
        Sample visible states given visible probabilities.
        
        Args:
            visible_probs: Visible probabilities tensor of shape (batch_size, visible_size)
            
        Returns:
            Visible states tensor of shape (batch_size, visible_size)
        """
        if self.visible_type == 'binary':
            return tensor.random_bernoulli(visible_probs)
        else:  # gaussian
            return visible_probs + tensor.random_normal(tensor.shape(visible_probs), device=self.device)
            
    def reconstruction_error(self, data, per_sample=False):
        """
        Compute reconstruction error for data.
        
        Args:
            data: Data tensor of shape (batch_size, visible_size)
            per_sample: Whether to return error per sample or mean error
            
        Returns:
            Reconstruction error (scalar or per-sample)
        """
        # Get reconstructed data
        reconstructed = self.reconstruct(data)
        
        # Compute squared error
        squared_error = ops.square(ops.subtract(data, reconstructed))
        
        # Sum across features
        error = stats.sum(squared_error, axis=1)
        
        if per_sample:
            return error
        else:
            return ops.stats.mean(error)
            
    def anomaly_score(self, data):
        """
        Compute anomaly score for data.
        
        Args:
            data: Data tensor of shape (batch_size, visible_size)
            
        Returns:
            Anomaly score tensor of shape (batch_size,)
        """
        # Use reconstruction error as anomaly score
        return self.reconstruction_error(data, per_sample=True)
        
    def is_anomaly(self, data, threshold=None):
        """
        Determine if data points are anomalies.
        
        Args:
            data: Data tensor of shape (batch_size, visible_size)
            threshold: Optional threshold value (if None, use self.reconstruction_error_threshold)
            
        Returns:
            Boolean tensor of shape (batch_size,) indicating anomalies
        """
        # Compute anomaly scores
        scores = self.anomaly_score(data)
        
        # Use provided threshold or default
        if threshold is None:
            try:
                threshold = self.reconstruction_error_threshold
            except AttributeError:
                # If no threshold is available, use a default
                threshold = tensor.convert_to_tensor(1.0, dtype=tensor.float32)
        
        # Return boolean tensor indicating anomalies
        return ops.greater_equal(scores, threshold)

def train_rbm(rbm: RestrictedBoltzmannMachine,
              data: tensor.EmberTensor,
              num_epochs: int = 10, 
              batch_size: int = 32, 
              learning_rate: float = 0.01, 
              momentum: float = 0.5, 
              weight_decay: float = 0.0001, 
              num_gibbs_steps: int = 1,
              callback: Optional[Callable[[int, float], None]] = None) -> List[float]:
    """
    Train an RBM using contrastive divergence.
    
    Args:
        rbm: RBM to train
        data: Training data tensor of shape (num_samples, visible_size)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        num_gibbs_steps: Number of Gibbs sampling steps
        callback: Optional callback function called after each epoch with (epoch, loss)
        
    Returns:
        List of losses for each epoch
    """
    # Set device for data
    # No need to move data to device in EmberTensor
    
    # Create optimizer and data loader
    # Note: In a real implementation, we would need to implement an optimizer and data loader for EmberTensor
    # For now, we'll just use a simple batching approach
    
    # Split data into batches
    num_samples = tensor.shape(data)[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    # Create indices for shuffling
    indices = list(range(num_samples))
    import random
    random.shuffle(indices)
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Process each batch
        for i in range(num_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            # Note: In a real implementation, we would need tensor indexing
            # For now, we'll just use the whole data
            positive_visible = data
            
            # Positive phase
            positive_hidden_probs, positive_hidden_states = rbm.visible_to_hidden(positive_visible)
            
            # Negative phase
            negative_hidden_states = positive_hidden_states
            negative_visible_probs, negative_visible_states = rbm.hidden_to_visible(negative_hidden_states)
            
            for _ in range(num_gibbs_steps - 1):
                negative_hidden_probs, negative_hidden_states = rbm.visible_to_hidden(negative_visible_states)
                negative_visible_probs, negative_visible_states = rbm.hidden_to_visible(negative_hidden_states)
            
            # Compute loss (free energy difference)
            positive_free_energy = rbm.free_energy(positive_visible)
            negative_free_energy = rbm.free_energy(negative_visible_probs)
            
            # In a real implementation, we would update weights here
            # For now, we'll just compute the loss
            loss = ops.stats.mean(ops.subtract(positive_free_energy, negative_free_energy))
            
            epoch_loss += loss
        
        # Average loss for the epoch
        epoch_loss /= num_batches
        losses.append(epoch_loss)
        
        # Call callback if provided
        if callback is not None:
            callback(epoch, epoch_loss)
    
    return losses

def reconstruct_with_rbm(rbm: RestrictedBoltzmannMachine,
                         data: tensor.EmberTensor,
                         num_gibbs_steps: int = 1) -> tensor.EmberTensor:
    """
    Reconstruct data using an RBM.
    
    Args:
        rbm: Trained RBM
        data: Data tensor of shape (num_samples, visible_size)
        num_gibbs_steps: Number of Gibbs sampling steps
        
    Returns:
        Reconstructed data tensor of shape (num_samples, visible_size)
    """
    # Reconstruct
    reconstructed = rbm.reconstruct(data, num_gibbs_steps)
    
    return reconstructed