"""
RBM Module Implementation

This module provides an implementation of Restricted Boltzmann Machines
using the ember_ml Module system.
"""

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module, Parameter

class RBMModule(Module):
    """
    Restricted Boltzmann Machine implemented using the ember_ml Module system.
    
    This module provides a backend-agnostic implementation of RBMs with support
    for feature extraction, reconstruction, and anomaly detection.
    """
    
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        use_binary_states: bool = False,
        **kwargs
    ):
        """
        Initialize the RBM module.
        
        Args:
            n_visible: Number of visible units (input features)
            n_hidden: Number of hidden units (learned features)
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for gradient updates
            weight_decay: L2 regularization coefficient
            use_binary_states: Whether to use binary states (True) or probabilities (False)
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_binary_states = use_binary_states
        
        # Initialize weights and biases with optimized scaling
        # Calculate standard deviation for weight initialization
        # Use 0.01 / sqrt(n_visible) as a good initialization for RBMs
        std_dev = ops.divide(
            tensor.convert_to_tensor(0.01, dtype=tensor.float32),
            ops.sqrt(tensor.convert_to_tensor(n_visible, dtype=tensor.float32))
        )
        # Create weights with random normal distribution
        weights_tensor = tensor.random_normal((n_visible, n_hidden), stddev=std_dev)
        self.weights = Parameter(weights_tensor)
        self.visible_bias = Parameter(tensor.zeros(n_visible))
        self.hidden_bias = Parameter(tensor.zeros(n_hidden))
        
        # For tracking training progress
        self.register_buffer('training_errors', tensor.convert_to_tensor(0, dtype=tensor.float32))
        self.register_buffer('reconstruction_error_threshold', None)
        self.register_buffer('free_energy_threshold', None)
        self.register_buffer('n_epochs_trained', tensor.convert_to_tensor(0, dtype=tensor.int32))
    
    def forward(self, visible_states):
        """
        Transform visible states to hidden representation.
        
        Args:
            visible_states: Visible states [batch_size, n_visible]
            
        Returns:
            Hidden probabilities [batch_size, n_hidden]
        """
        return self.compute_hidden_probabilities(visible_states)
    
    def compute_hidden_probabilities(self, visible_states):
        """
        Compute probabilities of hidden units given visible states.
        
        Args:
            visible_states: States of visible units [batch_size, n_visible]
            
        Returns:
            Probabilities of hidden units [batch_size, n_hidden]
        """
        # Compute activations: visible_states @ weights + hidden_bias
        hidden_activations = ops.add(
            ops.matmul(visible_states, self.weights.data),
            self.hidden_bias.data
        )
        from ember_ml.nn.modules.activations import sigmoid
        return sigmoid(hidden_activations)
    
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
        
        return tensor.cast(
            ops.greater(hidden_probs, tensor.random_uniform(tensor.shape(hidden_probs))),
            dtype=tensor.float32
        )
    
    def compute_visible_probabilities(self, hidden_states):
        """
        Compute probabilities of visible units given hidden states.
        
        Args:
            hidden_states: States of hidden units [batch_size, n_hidden]
            
        Returns:
            Probabilities of visible units [batch_size, n_visible]
        """
        # Compute activations: hidden_states @ weights.T + visible_bias
        visible_activations = ops.add(
            ops.matmul(hidden_states, tensor.transpose(self.weights.data)),
            self.visible_bias.data
        )
        from ember_ml.nn.modules.activations import sigmoid
        return sigmoid(visible_activations)
    
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
        
        return tensor.cast(
            ops.greater(visible_probs, tensor.random_uniform(tensor.shape(visible_probs))),
            dtype=tensor.float32
        )
    
    def reconstruct(self, visible_states):
        """
        Reconstruct visible states.
        
        Args:
            visible_states: Visible states [batch_size, n_visible]
            
        Returns:
            Reconstructed visible states [batch_size, n_visible]
        """
        hidden_probs = self.compute_hidden_probabilities(visible_states)
        hidden_states = self.sample_hidden_states(hidden_probs)
        visible_probs = self.compute_visible_probabilities(hidden_states)
        return visible_probs
    
    def reconstruction_error(self, visible_states, per_sample=False):
        """
        Compute reconstruction error.
        
        Args:
            visible_states: Visible states [batch_size, n_visible]
            per_sample: Whether to return error per sample
            
        Returns:
            Reconstruction error (mean or per sample)
        """
        reconstructed = self.reconstruct(visible_states)
        squared_error = stats.sum(ops.square(ops.subtract(visible_states, reconstructed)), axis=1)
        
        if per_sample:
            return squared_error
        else:
            return ops.stats.mean(squared_error)
    
    def free_energy(self, visible_states):
        """
        Compute free energy.
        
        Args:
            visible_states: Visible states [batch_size, n_visible]
            
        Returns:
            Free energy [batch_size]
        """
        visible_bias_term = ops.matmul(visible_states, self.visible_bias.data)
        hidden_term = stats.sum(
            ops.log(ops.add(1.0, ops.exp(ops.add(ops.matmul(visible_states, self.weights.data), self.hidden_bias.data)))),
            axis=1
        )
        
        return ops.negative(ops.add(hidden_term, visible_bias_term))
    
    def anomaly_score(self, visible_states, method='reconstruction'):
        """
        Compute anomaly score.
        
        Args:
            visible_states: Visible states [batch_size, n_visible]
            method: Method to use ('reconstruction' or 'free_energy')
            
        Returns:
            Anomaly scores [batch_size]
        """
        if method == 'reconstruction':
            return self.reconstruction_error(visible_states, per_sample=True)
        elif method == 'free_energy':
            # For free energy, lower is better, so we negate
            return ops.negative(self.free_energy(visible_states))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def is_anomaly(self, visible_states, method='reconstruction'):
        """
        Determine if visible states are anomalous.
        
        Args:
            visible_states: Visible states [batch_size, n_visible]
            method: Method to use ('reconstruction' or 'free_energy')
            
        Returns:
            Boolean array indicating anomalies [batch_size]
        """
        scores = self.anomaly_score(visible_states, method)
        
        if method == 'reconstruction':
            try:
                threshold = self.reconstruction_error_threshold
                if threshold is None:
                    # Default threshold if not set
                    threshold = tensor.convert_to_tensor(1.0, dtype=tensor.float32)
                return ops.greater(scores, threshold)
            except (AttributeError, TypeError):
                # If the attribute doesn't exist, use a default threshold
                return ops.greater(scores, tensor.convert_to_tensor(1.0, dtype=tensor.float32))
        elif method == 'free_energy':
            try:
                threshold = self.free_energy_threshold
                if threshold is None:
                    # Default threshold if not set
                    threshold = tensor.convert_to_tensor(0.0, dtype=tensor.float32)
                return ops.less(scores, threshold)
            except (AttributeError, TypeError):
                # If the attribute doesn't exist, use a default threshold
                return ops.less(scores, tensor.convert_to_tensor(0.0, dtype=tensor.float32))
        else:
            raise ValueError(f"Unknown method: {method}")