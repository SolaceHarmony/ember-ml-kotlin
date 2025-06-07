"""
Optimized Restricted Boltzmann Machine for Large-Scale Feature Learning

This module provides an optimized implementation of Restricted Boltzmann Machines
designed for processing large-scale data with efficient memory usage and support
for chunked training.
"""

import time
from datetime import datetime
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from ember_ml.ops import stats
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optimized_rbm')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU acceleration will be disabled.")


class OptimizedRBM:
    """
    Optimized Restricted Boltzmann Machine for large-scale feature learning.
    
    This implementation focuses on:
    - Memory efficiency for large datasets
    - Chunked training support
    - Optional GPU acceleration
    - Efficient parameter initialization
    - Comprehensive monitoring and logging
    """
    
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        batch_size: int = 100,
        use_binary_states: bool = False,
        use_gpu: bool = False,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the optimized RBM.
        
        Args:
            n_visible: Number of visible units (input features)
            n_hidden: Number of hidden units (learned features)
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for gradient updates
            weight_decay: L2 regularization coefficient
            batch_size: Size of mini-batches for training
            use_binary_states: Whether to use binary states (True) or probabilities (False)
            use_gpu: Whether to use GPU acceleration if available
            device: Specific device to use ('cuda:0', 'cuda:1', etc.)
            verbose: Whether to print progress information
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_binary_states = use_binary_states
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.verbose = verbose
        
        # Initialize weights and biases with optimized scaling
        scale = 0.01 / ops.sqrt(n_visible)
        self.weights = tensor.random_normal(0, scale, (n_visible, n_hidden))
        self.visible_bias = tensor.zeros(n_visible)
        self.hidden_bias = tensor.zeros(n_hidden)
        
        # Initialize momentum terms
        self.weights_momentum = tensor.zeros((n_visible, n_hidden))
        self.visible_bias_momentum = tensor.zeros(n_visible)
        self.hidden_bias_momentum = tensor.zeros(n_hidden)
        
        # For tracking training progress
        self.training_errors = []
        self.training_time = 0
        self.n_epochs_trained = 0
        self.last_batch_error = float('inf')
        
        # For anomaly detection
        self.reconstruction_error_threshold = None
        self.free_energy_threshold = None
        
        # Move to GPU if requested and available
        self.device = None
        if self.use_gpu:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            
            if self.device.type == "cuda":
                logger.info(f"Using GPU acceleration for RBM: {self.device}")
                # Convert numpy arrays to torch tensors on GPU
                self._to_gpu()
            else:
                logger.info("GPU requested but not available, using CPU")
                self.use_gpu = False
        
        logger.info(f"Initialized OptimizedRBM with {n_visible} visible units, {n_hidden} hidden units")
    
    def _to_gpu(self):
        """Convert numpy arrays to torch tensors on GPU."""
        from ember_ml.nn import tensor
        self.weights_torch = tensor.convert_to_tensor(self.weights, dtype=torch.float32, device=self.device)
        self.visible_bias_torch = tensor.convert_to_tensor(self.visible_bias, dtype=torch.float32, device=self.device)
        self.hidden_bias_torch = tensor.convert_to_tensor(self.hidden_bias, dtype=torch.float32, device=self.device)
        self.weights_momentum_torch = tensor.convert_to_tensor(self.weights_momentum, dtype=tensor.float32, device=self.device)
        self.visible_bias_momentum_torch = tensor.convert_to_tensor(self.visible_bias_momentum, dtype=tensor.float32, device=self.device)
        self.hidden_bias_momentum_torch = tensor.convert_to_tensor(self.hidden_bias_momentum, dtype=tensor.float32, device=self.device)
    
    def _to_cpu(self):
        """Convert torch tensors back to numpy arrays."""
        if self.use_gpu:
            self.weights = self.weights_torch.cpu().numpy()
            self.visible_bias = self.visible_bias_torch.cpu().numpy()
            self.hidden_bias = self.hidden_bias_torch.cpu().numpy()
            self.weights_momentum = self.weights_momentum_torch.cpu().numpy()
            self.visible_bias_momentum = self.visible_bias_momentum_torch.cpu().numpy()
            self.hidden_bias_momentum = self.hidden_bias_momentum_torch.cpu().numpy()
    
    def sigmoid(self, x: Union[TensorLike]) -> Union[TensorLike, tensor.EmberTensor]:
        """
        Compute sigmoid function with numerical stability improvements.
        
        Args:
            x: Input array or tensor
            
        Returns:
            Sigmoid of input
        """
        if self.use_gpu and isinstance(x, tensor.EmberTensor):
            # Clip values to avoid overflow/underflow
            x = torch.clamp(x, -15, 15)
            return 1.0 / (1.0 + torch.exp(-x))
        else:
            # Clip values to avoid overflow/underflow
            x = ops.clip(x, -15, 15)
            return 1.0 / (1.0 + ops.exp(-x))
    
    def compute_hidden_probabilities(self, visible_states: Union[TensorLike, tensor.EmberTensor]) -> Union[TensorLike, tensor.EmberTensor]:
        """
        Compute probabilities of hidden units given visible states.
        
        Args:
            visible_states: States of visible units [batch_size, n_visible]
            
        Returns:
            Probabilities of hidden units [batch_size, n_hidden]
        """
        if self.use_gpu and isinstance(visible_states, tensor.EmberTensor):
            # Compute activations: visible_states @ weights + hidden_bias
            hidden_activations = ops.matmul(visible_states, self.weights_torch) + self.hidden_bias_torch
            return self.sigmoid(hidden_activations)
        else:
            # Convert to numpy if needed
            if isinstance(visible_states, tensor.EmberTensor):
                visible_states = visible_states.cpu().numpy()
            
            # Compute activations: visible_states @ weights + hidden_bias
            hidden_activations = ops.dot(visible_states, self.weights) + self.hidden_bias
            return self.sigmoid(hidden_activations)
    
    def sample_hidden_states(self, hidden_probs: Union[TensorLike, tensor.EmberTensor]) -> Union[TensorLike, tensor.EmberTensor]:
        """
        Sample binary hidden states from their probabilities.
        
        Args:
            hidden_probs: Probabilities of hidden units [batch_size, n_hidden]
            
        Returns:
            Binary hidden states [batch_size, n_hidden]
        """
        if not self.use_binary_states:
            return hidden_probs
        
        if self.use_gpu and isinstance(hidden_probs, tensor.EmberTensor):
            return (hidden_probs > torch.rand_like(hidden_probs)).float()
        else:
            # Convert to numpy if needed
            if isinstance(hidden_probs, tensor.EmberTensor):
                hidden_probs = hidden_probs.cpu().numpy()
            
            return (hidden_probs > tensor.random_normal(hidden_probs.shape)).astype(tensor.float32)
    
    def compute_visible_probabilities(self, hidden_states: Union[TensorLike, tensor.EmberTensor]) -> Union[TensorLike, tensor.EmberTensor]:
        """
        Compute probabilities of visible units given hidden states.
        
        Args:
            hidden_states: States of hidden units [batch_size, n_hidden]
            
        Returns:
            Probabilities of visible units [batch_size, n_visible]
        """
        if self.use_gpu and isinstance(hidden_states, tensor.EmberTensor):
            # Compute activations: hidden_states @ weights.T + visible_bias
            visible_activations = ops.matmul(hidden_states, self.weights_torch.t()) + self.visible_bias_torch
            return self.sigmoid(visible_activations)
        else:
            # Convert to numpy if needed
            if isinstance(hidden_states, tensor.EmberTensor):
                hidden_states = hidden_states.cpu().numpy()
            
            # Compute activations: hidden_states @ weights.T + visible_bias
            visible_activations = ops.dot(hidden_states, self.weights.T) + self.visible_bias
            return self.sigmoid(visible_activations)
    
    def sample_visible_states(self, visible_probs: Union[TensorLike, tensor.EmberTensor]) -> Union[TensorLike, tensor.EmberTensor]:
        """
        Sample binary visible states from their probabilities.
        
        Args:
            visible_probs: Probabilities of visible units [batch_size, n_visible]
            
        Returns:
            Binary visible states [batch_size, n_visible]
        """
        if not self.use_binary_states:
            return visible_probs
        
        if self.use_gpu and isinstance(visible_probs, tensor.EmberTensor):
            return (visible_probs > torch.rand_like(visible_probs)).float()
        else:
            # Convert to numpy if needed
            if isinstance(visible_probs, tensor.EmberTensor):
                visible_probs = visible_probs.cpu().numpy()
            
            return (visible_probs > tensor.random_normal(visible_probs.shape)).astype(tensor.float32)
    
    def contrastive_divergence(self, batch_data: Union[TensorLike, tensor.EmberTensor], k: int = 1) -> float:
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
        # Convert to torch tensor if using GPU
        if self.use_gpu and not isinstance(batch_data, tensor.EmberTensor):
            batch_data = tensor.convert_to_tensor(batch_data, dtype=torch.float32, device=self.device)
        
        batch_size = len(batch_data)
        
        # Positive phase
        # Compute hidden probabilities and states
        if self.use_gpu:
            pos_hidden_probs = self.compute_hidden_probabilities(batch_data)
            pos_hidden_states = self.sample_hidden_states(pos_hidden_probs)
            
            # Compute positive associations
            pos_associations = ops.matmul(batch_data.t(), pos_hidden_probs)
            
            # Negative phase
            # Start with the hidden states from positive phase
            neg_hidden_states = pos_hidden_states.clone()
            
            # Perform k steps of Gibbs sampling
            for _ in range(k):
                # Compute visible probabilities and states
                neg_visible_probs = self.compute_visible_probabilities(neg_hidden_states)
                neg_visible_states = self.sample_visible_states(neg_visible_probs)
                
                # Compute hidden probabilities and states
                neg_hidden_probs = self.compute_hidden_probabilities(neg_visible_states)
                neg_hidden_states = self.sample_hidden_states(neg_hidden_probs)
            
            # Compute negative associations
            neg_associations = ops.matmul(neg_visible_states.t(), neg_hidden_probs)
            
            # Compute gradients
            weights_gradient = (pos_associations - neg_associations) / batch_size
            visible_bias_gradient = torch.mean(batch_data - neg_visible_states, dim=0)
            hidden_bias_gradient = torch.mean(pos_hidden_probs - neg_hidden_probs, dim=0)
            
            # Update with momentum and weight decay
            self.weights_momentum_torch = self.momentum * self.weights_momentum_torch + weights_gradient
            self.visible_bias_momentum_torch = self.momentum * self.visible_bias_momentum_torch + visible_bias_gradient
            self.hidden_bias_momentum_torch = self.momentum * self.hidden_bias_momentum_torch + hidden_bias_gradient
            
            # Apply updates
            self.weights_torch += self.learning_rate * (self.weights_momentum_torch - self.weight_decay * self.weights_torch)
            self.visible_bias_torch += self.learning_rate * self.visible_bias_momentum_torch
            self.hidden_bias_torch += self.learning_rate * self.hidden_bias_momentum_torch
            
            # Compute reconstruction error
            reconstruction_error = torch.mean(stats.sum((batch_data - neg_visible_probs) ** 2, dim=1)).item()
        else:
            # CPU implementation
            pos_hidden_probs = self.compute_hidden_probabilities(batch_data)
            pos_hidden_states = self.sample_hidden_states(pos_hidden_probs)
            
            # Compute positive associations
            pos_associations = ops.dot(batch_data.T, pos_hidden_probs)
            
            # Negative phase
            # Start with the hidden states from positive phase
            neg_hidden_states = pos_hidden_states.copy()
            
            # Perform k steps of Gibbs sampling
            for _ in range(k):
                # Compute visible probabilities and states
                neg_visible_probs = self.compute_visible_probabilities(neg_hidden_states)
                neg_visible_states = self.sample_visible_states(neg_visible_probs)
                
                # Compute hidden probabilities and states
                neg_hidden_probs = self.compute_hidden_probabilities(neg_visible_states)
                neg_hidden_states = self.sample_hidden_states(neg_hidden_probs)
            
            # Compute negative associations
            neg_associations = ops.dot(neg_visible_states.T, neg_hidden_probs)
            
            # Compute gradients
            weights_gradient = (pos_associations - neg_associations) / batch_size
            visible_bias_gradient = stats.mean(batch_data - neg_visible_states, axis=0)
            hidden_bias_gradient = stats.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
            
            # Update with momentum and weight decay
            self.weights_momentum = self.momentum * self.weights_momentum + weights_gradient
            self.visible_bias_momentum = self.momentum * self.visible_bias_momentum + visible_bias_gradient
            self.hidden_bias_momentum = self.momentum * self.hidden_bias_momentum + hidden_bias_gradient
            
            # Apply updates
            self.weights += self.learning_rate * (self.weights_momentum - self.weight_decay * self.weights)
            self.visible_bias += self.learning_rate * self.visible_bias_momentum
            self.hidden_bias += self.learning_rate * self.hidden_bias_momentum
            
            # Compute reconstruction error
            reconstruction_error = stats.mean(stats.sum((batch_data - neg_visible_probs) ** 2, axis=1))
        
        self.last_batch_error = reconstruction_error
        return reconstruction_error
    
    def train_in_chunks(
        self,
        data_generator: Generator,
        epochs: int = 10,
        k: int = 1,
        validation_data: Optional[TensorLike] = None,
        early_stopping_patience: int = 5,
        callback: Optional[callable] = None
    ) -> List[float]:
        """
        Train the RBM using a data generator to handle large datasets.
        
        Args:
            data_generator: Generator yielding batches of training data
            epochs: Number of training epochs
            k: Number of Gibbs sampling steps
            validation_data: Optional validation data for early stopping
            early_stopping_patience: Number of epochs to wait for improvement
            callback: Optional callback function for monitoring
            
        Returns:
            List of reconstruction errors per epoch
        """
        training_errors = []
        
        start_time = time.time()
        best_validation_error = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_error = 0
            n_batches = 0
            
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Process each batch from the generator
            for batch_idx, batch_data in enumerate(data_generator):
                # Skip empty batches
                if len(batch_data) == 0:
                    continue
                
                batch_start_time = time.time()
                
                # Train on batch
                batch_error = self.contrastive_divergence(batch_data, k)
                epoch_error += batch_error
                n_batches += 1
                
                batch_time = time.time() - batch_start_time
                
                # Log progress
                if self.verbose and (batch_idx % 10 == 0 or batch_idx < 5):
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}: "
                               f"error = {batch_error:.4f}, time = {batch_time:.2f}s")
                
                # Call callback if provided
                if callback:
                    callback(epoch, batch_idx, batch_error)
            
            # Compute average epoch error
            avg_epoch_error = epoch_error / max(n_batches, 1)
            training_errors.append(avg_epoch_error)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{epochs} completed: "
                       f"avg_error = {avg_epoch_error:.4f}, time = {epoch_time:.2f}s")
            
            # Check validation error if provided
            validation_error = None
            if validation_data is not None:
                validation_error = self.reconstruction_error(validation_data)
                logger.info(f"Validation error: {validation_error:.4f}")
                
                # Early stopping check
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    patience_counter = 0
                    logger.info(f"New best validation error: {best_validation_error:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} epochs")
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.n_epochs_trained += 1
        
        self.training_time += time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f}s")
        
        # Compute threshold for anomaly detection based on training data
        if self.reconstruction_error_threshold is None:
            logger.info("Computing anomaly detection thresholds")
            # Use the generator to compute errors
            errors = []
            energies = []
            
            for batch_data in data_generator:
                if len(batch_data) == 0:
                    continue
                
                batch_errors = self.reconstruction_error(batch_data, per_sample=True)
                batch_energies = self.free_energy(batch_data)
                
                errors.extend(batch_errors)
                energies.extend(batch_energies)
            
            self.reconstruction_error_threshold = stats.percentile(errors, 95)  # 95th percentile
            self.free_energy_threshold = stats.percentile(energies, 5)  # 5th percentile
            
            logger.info(f"Reconstruction error threshold: {self.reconstruction_error_threshold:.4f}")
            logger.info(f"Free energy threshold: {self.free_energy_threshold:.4f}")
        
        # Sync GPU tensors to CPU if needed
        if self.use_gpu:
            self._to_cpu()
        
        return training_errors
    
    def transform(self, data: Union[TensorLike, tensor.EmberTensor]) -> TensorLike:
        """
        Transform data to hidden representation.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Hidden representation [n_samples, n_hidden]
        """
        # Convert to torch tensor if using GPU
        if self.use_gpu and not isinstance(data, tensor.EmberTensor):
            data = tensor.convert_to_tensor(data, dtype=torch.float32, device=self.device)
        
        # Compute hidden probabilities
        hidden_probs = self.compute_hidden_probabilities(data)
        
        # Convert to numpy if needed
        if self.use_gpu and isinstance(hidden_probs, tensor.EmberTensor):
            hidden_probs = hidden_probs.cpu().numpy()
        
        return hidden_probs
    
    def transform_in_chunks(self, data_generator: Generator, chunk_size: int = 1000) -> TensorLike:
        """
        Transform data to hidden representation in chunks.
        
        Args:
            data_generator: Generator yielding batches of data
            chunk_size: Size of chunks for processing
            
        Returns:
            Hidden representation [n_samples, n_hidden]
        """
        hidden_probs_list = []
        
        for batch_data in data_generator:
            if len(batch_data) == 0:
                continue
            
            # Transform batch
            batch_hidden_probs = self.transform(batch_data)
            hidden_probs_list.append(batch_hidden_probs)
        
        # Combine all hidden probabilities
        if hidden_probs_list:
            return tensor.vstack(hidden_probs_list)
        else:
            from ember_ml.nn.tensor import tensor
            return tensor.convert_to_tensor([])
    
    def reconstruct(self, data: Union[TensorLike, tensor.EmberTensor]) -> TensorLike:
        """
        Reconstruct input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Reconstructed data [n_samples, n_visible]
        """
        # Convert to torch tensor if using GPU
        if self.use_gpu and not isinstance(data, tensor.EmberTensor):
            data = tensor.convert_to_tensor(data, dtype=torch.float32, device=self.device)
        
        # Compute hidden probabilities and sample states
        hidden_probs = self.compute_hidden_probabilities(data)
        hidden_states = self.sample_hidden_states(hidden_probs)
        
        # Compute visible probabilities
        visible_probs = self.compute_visible_probabilities(hidden_states)
        
        # Convert to numpy if needed
        if self.use_gpu and isinstance(visible_probs, tensor.EmberTensor):
            visible_probs = visible_probs.cpu().numpy()
        
        return visible_probs
    
    def reconstruction_error(self, data: Union[TensorLike, tensor.EmberTensor], per_sample: bool = False) -> Union[float, TensorLike]:
        """
        Compute reconstruction error for input data.
        
        Args:
            data: Input data [n_samples, n_visible]
            per_sample: Whether to return error per sample
            
        Returns:
            Reconstruction error (mean or per sample)
        """
        # Convert to torch tensor if using GPU
        if self.use_gpu and not isinstance(data, tensor.EmberTensor):
            data = tensor.convert_to_tensor(data, dtype=torch.float32, device=self.device)
        
        # Reconstruct data
        reconstructed = self.reconstruct(data)
        
        # Compute squared error
        if self.use_gpu and isinstance(data, tensor.EmberTensor):
            # Convert reconstructed to tensor if it's not already
            if not isinstance(reconstructed, tensor.EmberTensor):
                reconstructed = tensor.convert_to_tensor(reconstructed, dtype=torch.float32, device=self.device)
            
            squared_error = stats.sum((data - reconstructed) ** 2, dim=1)
            
            if per_sample:
                return squared_error.cpu().numpy()
            else:
                return torch.mean(squared_error).item()
        else:
            # Convert data to numpy if it's a tensor
            if isinstance(data, tensor.EmberTensor):
                data = data.cpu().numpy()
            
            squared_error = stats.sum((data - reconstructed) ** 2, axis=1)
            
            if per_sample:
                return squared_error
            else:
                return stats.mean(squared_error)
    
    def free_energy(self, data: Union[TensorLike, tensor.EmberTensor]) -> TensorLike:
        """
        Compute free energy for input data.
        
        The free energy is a measure of how well the RBM models the data.
        Lower values indicate better fit.
        
        Args:
            data: Input data [n_samples, n_visible]
            
        Returns:
            Free energy for each sample [n_samples]
        """
        # Convert to torch tensor if using GPU
        if self.use_gpu and not isinstance(data, tensor.EmberTensor):
            data = tensor.convert_to_tensor(data, dtype=torch.float32, device=self.device)
        
        if self.use_gpu:
            visible_bias_term = ops.matmul(data, self.visible_bias_torch)
            hidden_term = stats.sum(
                torch.log(1 + torch.exp(ops.matmul(data, self.weights_torch) + self.hidden_bias_torch)),
                dim=1
            )
            
            result = -hidden_term - visible_bias_term
            return result.cpu().numpy()
        else:
            # Convert data to numpy if it's a tensor
            if isinstance(data, tensor.EmberTensor):
                data = data.cpu().numpy()
            
            visible_bias_term = ops.dot(data, self.visible_bias)
            hidden_term = stats.sum(
                ops.log(1 + ops.exp(ops.dot(data, self.weights) + self.hidden_bias)),
                axis=1
            )
            
            return -hidden_term - visible_bias_term
    
    def anomaly_score(self, data: Union[TensorLike, tensor.EmberTensor], method: str = 'reconstruction') -> TensorLike:
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
            return -self.free_energy(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def is_anomaly(self, data: Union[TensorLike, tensor.EmberTensor], method: str = 'reconstruction') -> TensorLike:
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
            return scores > self.reconstruction_error_threshold
        elif method == 'free_energy':
            return scores < self.free_energy_threshold
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        # Sync GPU tensors to CPU if needed
        if self.use_gpu:
            self._to_cpu()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'weights': self.weights,
            'visible_bias': self.visible_bias,
            'hidden_bias': self.hidden_bias,
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
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        ops.save(filepath, model_data, allow_pickle=True)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, use_gpu: bool = False, device: Optional[str] = None) -> 'OptimizedRBM':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            use_gpu: Whether to use GPU acceleration
            device: Specific device to use
            
        Returns:
            Loaded RBM model
        """
        # Load model data
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
            use_gpu=use_gpu,
            device=device
        )
        
        # Set model parameters
        rbm.weights = model_data['weights']
        rbm.visible_bias = model_data['visible_bias']
        rbm.hidden_bias = model_data['hidden_bias']
        rbm.training_errors = model_data['training_errors']
        rbm.reconstruction_error_threshold = model_data['reconstruction_error_threshold']
        rbm.free_energy_threshold = model_data['free_energy_threshold']
        rbm.training_time = model_data['training_time']
        rbm.n_epochs_trained = model_data['n_epochs_trained']
        
        # Move to GPU if requested
        if use_gpu and rbm.use_gpu:
            rbm._to_gpu()
        
        logger.info(f"Model loaded from {filepath}")
        return rbm
    
    def summary(self) -> str:
        """
        Get a summary of the model.
        
        Returns:
            Summary string
        """
        summary = [
            "Optimized Restricted Boltzmann Machine Summary",
            "============================================",
            f"Visible units: {self.n_visible}",
            f"Hidden units: {self.n_hidden}",
            f"Parameters: {self.n_visible * self.n_hidden + self.n_visible + self.n_hidden}",
            f"Learning rate: {self.learning_rate}",
            f"Momentum: {self.momentum}",
            f"Weight decay: {self.weight_decay}",
            f"Batch size: {self.batch_size}",
            f"Binary states: {self.use_binary_states}",
            f"GPU acceleration: {self.use_gpu}",
            f"Epochs trained: {self.n_epochs_trained}",
            f"Training time: {self.training_time:.2f} seconds",
            f"Current reconstruction error: {self.last_batch_error:.4f}",
            f"Anomaly threshold (reconstruction): {self.reconstruction_error_threshold}",
            f"Anomaly threshold (free energy): {self.free_energy_threshold}"
        ]
        
        return "\n".join(summary)


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = tensor.random_normal(1000, 20)
    
    # Create RBM
    rbm = OptimizedRBM(
        n_visible=20,
        n_hidden=10,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        batch_size=100,
        use_binary_states=False,
        use_gpu=True
    )
    
    # Define a generator to yield data in batches
    def data_generator(data, batch_size=100):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    # Train RBM
    rbm.train_in_chunks(
        data_generator(data, batch_size=100),
        epochs=10,
        k=1
    )
    
    # Transform data
    features = rbm.transform(data)
    
    print(f"Transformed data shape: {features.shape}")
    print(rbm.summary())