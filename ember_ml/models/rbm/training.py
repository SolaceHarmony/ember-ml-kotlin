"""
RBM Training Functions

This module provides functions for training and using RBM modules.
"""

import os
import json

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.container import Linear

def contrastive_divergence_step(rbm, batch_data, k=1):
    """
    Perform one step of contrastive divergence.
    
    Args:
        rbm: RBMModule instance
        batch_data: Batch of training data [batch_size, n_visible]
        k: Number of Gibbs sampling steps
        
    Returns:
        Tuple of (gradients, reconstruction_error)
    """
    batch_size = tensor.shape(batch_data)[0]
    
    # Positive phase
    pos_hidden_probs = rbm.compute_hidden_probabilities(batch_data)
    pos_hidden_states = rbm.sample_hidden_states(pos_hidden_probs)
    
    # Compute positive associations
    pos_associations = ops.matmul(tensor.transpose(batch_data), pos_hidden_probs)
    
    # Negative phase
    neg_hidden_states = pos_hidden_states
    
    # Perform k steps of Gibbs sampling
    neg_visible_probs = None
    neg_visible_states = None
    neg_hidden_probs = None
    
    for _ in range(k):
        neg_visible_probs = rbm.compute_visible_probabilities(neg_hidden_states)
        neg_visible_states = rbm.sample_visible_states(neg_visible_probs)
        neg_hidden_probs = rbm.compute_hidden_probabilities(neg_visible_states)
        neg_hidden_states = rbm.sample_hidden_states(neg_hidden_probs)
    
    # Compute negative associations
    neg_associations = ops.matmul(tensor.transpose(neg_visible_states), neg_hidden_probs)
    
    # Compute gradients
    weights_gradient = ops.divide(
        ops.subtract(pos_associations, neg_associations),
        tensor.convert_to_tensor(batch_size, dtype=tensor.float32)
    )
    visible_bias_gradient = ops.stats.mean(ops.subtract(batch_data, neg_visible_states), axis=0)
    hidden_bias_gradient = ops.stats.mean(ops.subtract(pos_hidden_probs, neg_hidden_probs), axis=0)
    
    # Compute reconstruction error
    reconstruction_error = ops.stats.mean(
        stats.sum(ops.square(ops.subtract(batch_data, neg_visible_probs)), axis=1)
    )
    
    return (weights_gradient, visible_bias_gradient, hidden_bias_gradient), reconstruction_error

def train_rbm(
    rbm,
    data_generator,
    epochs=10,
    k=1,
    validation_data=None,
    early_stopping_patience=5,
    callback=None
):
    """
    Train the RBM using a data generator.
    
    Args:
        rbm: RBMModule instance
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
    
    # Initialize momentum buffers
    weights_momentum = tensor.zeros_like(rbm.weights.data)
    visible_bias_momentum = tensor.zeros_like(rbm.visible_bias.data)
    hidden_bias_momentum = tensor.zeros_like(rbm.hidden_bias.data)
    
    # Use a very large number instead of infinity to avoid float() cast
    best_validation_error = tensor.convert_to_tensor(1e38, dtype=tensor.float32)
    patience_counter = tensor.convert_to_tensor(0, dtype=tensor.int32)
    
    for epoch in range(epochs):
        epoch_error = tensor.convert_to_tensor(0.0, dtype=tensor.float32)
        n_batches = tensor.convert_to_tensor(0, dtype=tensor.int32)
        
        # Process each batch from the generator
        for batch_idx, batch_data in enumerate(data_generator):
            # Skip empty batches
            if len(batch_data) == 0:
                continue
            
            # Convert batch data to tensor
            batch_data = tensor.convert_to_tensor(batch_data, dtype=tensor.float32)
            
            # Perform contrastive divergence
            gradients, batch_error = contrastive_divergence_step(rbm, batch_data, k)
            weights_gradient, visible_bias_gradient, hidden_bias_gradient = gradients
            
            # Update with momentum and weight decay
            weights_momentum = ops.add(
                ops.multiply(rbm.momentum, weights_momentum),
                weights_gradient
            )
            visible_bias_momentum = ops.add(
                ops.multiply(rbm.momentum, visible_bias_momentum),
                visible_bias_gradient
            )
            hidden_bias_momentum = ops.add(
                ops.multiply(rbm.momentum, hidden_bias_momentum),
                hidden_bias_gradient
            )
            
            # Apply updates
            rbm.weights.data = ops.add(
                rbm.weights.data,
                ops.multiply(
                    rbm.learning_rate,
                    ops.subtract(
                        weights_momentum,
                        ops.multiply(rbm.weight_decay, rbm.weights.data)
                    )
                )
            )
            rbm.visible_bias.data = ops.add(
                rbm.visible_bias.data,
                ops.multiply(rbm.learning_rate, visible_bias_momentum)
            )
            rbm.hidden_bias.data = ops.add(
                rbm.hidden_bias.data,
                ops.multiply(rbm.learning_rate, hidden_bias_momentum)
            )
            
            epoch_error = ops.add(epoch_error, batch_error)
            n_batches = ops.add(n_batches, tensor.convert_to_tensor(1, dtype=tensor.int32))
            
            # Call callback if provided
            if callback:
                callback(epoch, batch_idx, batch_error)
        
        # Compute average epoch error
        avg_epoch_error = ops.divide(
            epoch_error,
            tensor.convert_to_tensor(max(n_batches, 1), dtype=tensor.float32)
        )
        training_errors.append(avg_epoch_error)
        
        # Check validation error if provided
        if validation_data is not None:
            validation_error = rbm.reconstruction_error(validation_data)
            
            # Early stopping check
            if ops.less(validation_error, best_validation_error):
                best_validation_error = validation_error
                patience_counter = tensor.convert_to_tensor(0, dtype=tensor.int32)
            else:
                patience_counter = ops.add(patience_counter, tensor.convert_to_tensor(1, dtype=tensor.int32))
                if ops.greater_equal(patience_counter, tensor.convert_to_tensor(early_stopping_patience, dtype=tensor.int32)):
                    break
        
        # Update epochs trained
        try:
            rbm.n_epochs_trained = ops.add(rbm.n_epochs_trained, 1)
        except (AttributeError, TypeError):
            # If the attribute doesn't exist, we can't update it
            pass
    
    # Compute threshold for anomaly detection based on training data
    try:
        has_threshold = rbm.reconstruction_error_threshold is not None
    except (AttributeError, TypeError):
        has_threshold = False
        
    if not has_threshold:
        # Use the generator to compute errors
        errors = []
        energies = []
        
        for batch_data in data_generator:
            if len(batch_data) == 0:
                continue
            
            batch_data = tensor.convert_to_tensor(batch_data, dtype=tensor.float32)
            batch_errors = rbm.reconstruction_error(batch_data, per_sample=True)
            batch_energies = rbm.free_energy(batch_data)
            
            # Store errors and energies for computing thresholds
            # We'll use a different approach than percentiles to avoid NumPy
            errors.append(batch_errors)
            energies.append(batch_energies)
        
        # Combine all errors and energies
        all_errors = tensor.concatenate(errors, axis=0) if errors else tensor.zeros((0,))
        all_energies = tensor.concatenate(energies, axis=0) if energies else tensor.zeros((0,))
        
        # Sort values to compute thresholds (95th percentile for errors, 5th for energies)
        sorted_errors = tensor.sort(all_errors)
        sorted_energies = tensor.sort(all_energies)
        
        # Compute indices for percentiles
        error_idx_float = ops.multiply(
            tensor.convert_to_tensor(0.95, dtype=tensor.float32),
            tensor.convert_to_tensor(tensor.shape(sorted_errors)[0], dtype=tensor.float32)
        )
        energy_idx_float = ops.multiply(
            tensor.convert_to_tensor(0.05, dtype=tensor.float32),
            tensor.convert_to_tensor(tensor.shape(sorted_energies)[0], dtype=tensor.float32)
        )
        
        # Convert to integer indices using floor
        error_idx = tensor.cast(error_idx_float, dtype=tensor.int32)
        energy_idx = tensor.cast(energy_idx_float, dtype=tensor.int32)
        
        # Set thresholds (handle empty case)
        if ops.logical_and(
            ops.greater(error_idx, tensor.convert_to_tensor(0, dtype=tensor.int32)),
            ops.greater(tensor.shape(sorted_errors)[0], tensor.convert_to_tensor(0, dtype=tensor.int32))
        ):
            error_threshold = sorted_errors[error_idx]
        else:
            error_threshold = tensor.convert_to_tensor(1.0, dtype=tensor.float32)
            
        if ops.logical_and(
            ops.greater(energy_idx, tensor.convert_to_tensor(0, dtype=tensor.int32)),
            ops.greater(tensor.shape(sorted_energies)[0], tensor.convert_to_tensor(0, dtype=tensor.int32))
        ):
            energy_threshold = sorted_energies[energy_idx]
        else:
            energy_threshold = tensor.convert_to_tensor(0.0, dtype=tensor.float32)
            
        # Try to set the thresholds as attributes, but handle the case where they don't exist
        try:
            rbm.reconstruction_error_threshold = error_threshold
        except (AttributeError, TypeError):
            # If the attribute doesn't exist, we can't set it
            pass
            
        try:
            rbm.free_energy_threshold = energy_threshold
        except (AttributeError, TypeError):
            # If the attribute doesn't exist, we can't set it
            pass
    
    return training_errors

def transform_in_chunks(rbm, data_generator):
    """
    Transform data to hidden representation in chunks.
    
    Args:
        rbm: RBMModule instance
        data_generator: Generator yielding batches of data
        
    Returns:
        Hidden representation [n_samples, n_hidden]
    """
    hidden_probs_list = []
    
    for batch_data in data_generator:
        if len(batch_data) == 0:
            continue
        
        # Convert batch data to tensor
        batch_data = tensor.convert_to_tensor(batch_data, dtype=tensor.float32)
        
        # Transform batch
        batch_hidden_probs = rbm(batch_data)
        hidden_probs_list.append(batch_hidden_probs)
    
    # Combine all hidden probabilities
    if hidden_probs_list:
        return tensor.concatenate(hidden_probs_list, axis=0)
    else:
        return tensor.zeros((0, rbm.n_hidden), dtype=tensor.float32)

def save_rbm(rbm, filepath):
    """
    Save RBM to file.
    
    Args:
        rbm: RBMModule instance
        filepath: Path to save model
    """
    # Create directory if it doesn't exist and if there is a directory component
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    # Extract model parameters and metadata
    model_data = {
        'n_visible': rbm.n_visible,
        'n_hidden': rbm.n_hidden,
        'learning_rate': rbm.learning_rate,
        'momentum': rbm.momentum,
        'weight_decay': rbm.weight_decay,
        'use_binary_states': rbm.use_binary_states,
        'weights': rbm.weights.data.tolist(),
        'visible_bias': rbm.visible_bias.data.tolist(),
        'hidden_bias': rbm.hidden_bias.data.tolist()
    }
    
    # Try to save additional attributes if they exist
    try:
        model_data['n_epochs_trained'] = rbm.n_epochs_trained.tolist()
    except (AttributeError, TypeError):
        pass
        
    try:
        model_data['reconstruction_error_threshold'] = rbm.reconstruction_error_threshold.tolist()
    except (AttributeError, TypeError):
        pass
        
    try:
        model_data['free_energy_threshold'] = rbm.free_energy_threshold.tolist()
    except (AttributeError, TypeError):
        pass
    
    # Save metadata and tensor data to a JSON file
    with open(filepath, 'w') as f:
        json.dump(model_data, f)

def load_rbm(filepath):
    """
    Load RBM from file.
    
    Args:
        filepath: Path to load model from
        
    Returns:
        Loaded RBMModule instance
    """
    # Import here to avoid circular imports
    from ember_ml.models.rbm import RBMModule
    
    # Load metadata and tensor data from JSON file
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    # Create a new RBM instance with the same parameters
    rbm = RBMModule(
        n_visible=model_data['n_visible'],
        n_hidden=model_data['n_hidden'],
        learning_rate=model_data['learning_rate'],
        momentum=model_data['momentum'],
        weight_decay=model_data['weight_decay'],
        use_binary_states=model_data['use_binary_states']
    )
    
    # Load tensor data
    rbm.weights.data = tensor.convert_to_tensor(model_data['weights'])
    rbm.visible_bias.data = tensor.convert_to_tensor(model_data['visible_bias'])
    rbm.hidden_bias.data = tensor.convert_to_tensor(model_data['hidden_bias'])
    
    # Load additional attributes if they exist
    if 'n_epochs_trained' in model_data:
        rbm.n_epochs_trained = tensor.convert_to_tensor(model_data['n_epochs_trained'])
        
    if 'reconstruction_error_threshold' in model_data:
        rbm.reconstruction_error_threshold = tensor.convert_to_tensor(model_data['reconstruction_error_threshold'])
        
    if 'free_energy_threshold' in model_data:
        rbm.free_energy_threshold = tensor.convert_to_tensor(model_data['free_energy_threshold'])
    
    return rbm