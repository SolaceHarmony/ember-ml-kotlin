"""
Spatially Embedded Closed-form Continuous-time (seCfC) Example.

This script demonstrates how to use the seCfC framework for
spatially embedded continuous-time neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import EnhancedNCPMap
from ember_ml.nn.modules.rnn import seCfC

# Set random seed for reproducibility
tensor.set_seed(42)
# No need for np.random.seed as we're using tensor operations

def generate_sine_wave_data(num_samples, sequence_length, num_features):
    """Generate sine wave data for demonstration."""
    # Generate time points using tensor operations
    time_start = tensor.convert_to_tensor(0.0)
    time_end = tensor.convert_to_tensor(4.0 * ops.pi)
    time = tensor.linspace(time_start, time_end, sequence_length)
    
    # Generate sine waves with different frequencies
    data = tensor.zeros((num_samples, sequence_length, num_features))
    
    for i in range(num_samples):
        for j in range(num_features):
            frequency = tensor.convert_to_tensor(1.0 + 0.1 * j)
            phase = tensor.convert_to_tensor(0.5 * i)
            
            # Calculate sine wave
            angle = ops.add(ops.multiply(frequency, time), phase)
            sine_wave = ops.sin(angle)
            
            # Update data tensor using tensor indexing
            # Create indices for the update
            batch_idx = i
            feature_idx = j
            
            # Use tensor scatter update to set values
            # Create indices for all time steps for this batch and feature
            indices = []
            for t in range(sequence_length):
                indices.append([batch_idx, t, feature_idx])
            
            indices_tensor = tensor.convert_to_tensor(indices)
            updates = sine_wave
            
            # Update the tensor
            data = tensor.tensor_scatter_nd_update(data, indices_tensor, updates)
    
    # Split into train and test using tensor operations
    split_idx = tensor.cast(ops.multiply(0.8, num_samples), dtype=tensor.int32)
    # Use slice_tensor with start indices and sizes. Assumes 3D input (batch, seq, features).
    train_data = tensor.slice_tensor(data, [0, 0, 0], [split_idx, tensor.shape(data)[1], tensor.shape(data)[2]])
    test_data = tensor.slice_tensor(data, [split_idx, 0, 0], [ops.subtract(num_samples, split_idx), tensor.shape(data)[1], tensor.shape(data)[2]]) # Use ops.subtract for size
    
    return train_data, test_data

def create_se_cfc_model():
    """Create a seCfC model for demonstration."""
    # Create an enhanced NCP map
    neuron_map = EnhancedNCPMap(
        inter_neurons=16,
        command_neurons=8,
        motor_neurons=4,
        sensory_neurons=4,
        neuron_type="cfc",
        time_scale_factor=1.0,
        activation="tanh",
        recurrent_activation="sigmoid",
        sparsity_level=0.5,
        seed=42
    )
    
    # Create a seCfC model
    model = seCfC(
        neuron_map=neuron_map,
        return_sequences=True,
        return_state=False,
        go_backwards=False,
        regularization_strength=0.01
    )
    
    return model

def train_model(model, train_data, num_epochs=10, learning_rate=0.01):
    """Train the model on the provided data."""
    # Convert data to tensors
    train_inputs = tensor.convert_to_tensor(train_data[:, :-1, :])
    train_targets = tensor.convert_to_tensor(train_data[:, 1:, :])
    from ember_ml.training import Adam
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate) # Pass parameters and use correct arg name
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        # Define function for gradient calculation
        def compute_loss_and_grads(model, inputs, targets):
            predictions = model(inputs)
            mse_loss = ops.mse(targets, predictions)
            # Assuming get_regularization_loss exists and returns a tensor
            reg_loss = model.get_regularization_loss()
            total_loss = ops.add(mse_loss, reg_loss)
            # Calculate gradients using ops.gradients
            grads = ops.gradients(total_loss, model.parameters())
            return total_loss, grads

        # Calculate loss and gradients
        total_loss, gradients = compute_loss_and_grads(model, train_inputs, train_targets)

        # Apply gradients
        # Assuming Adam optimizer has apply_gradients or similar (might need step method)
        # If Adam follows pattern from other examples, it might need a step(grads) method
        # For now, assuming apply_gradients exists based on the original code structure
        optimizer.apply_gradients(zip(gradients, model.parameters())) # Or optimizer.step(gradients) if API matches
        
        # Record loss using tensor.to_numpy
        losses.append(tensor.to_numpy(total_loss))
        
        # Print progress using tensor.to_numpy
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {tensor.to_numpy(total_loss):.4f}")
    
    return losses

def evaluate_model(model, test_data):
    """Evaluate the model on test data."""
    # Convert data to tensors
    test_inputs = tensor.convert_to_tensor(test_data[:, :-1, :])
    test_targets = tensor.convert_to_tensor(test_data[:, 1:, :])
    
    # Forward pass
    predictions = model(test_inputs)
    
    # Calculate loss
    mse_loss = ops.mse(test_targets, predictions)

    print(f"Test MSE: {tensor.to_numpy(mse_loss):.4f}")

    # Use tensor.to_numpy for returning numpy arrays
    return tensor.to_numpy(predictions), tensor.to_numpy(test_targets)

def visualize_results(predictions, targets):
    """Visualize the model predictions against targets."""
    # Plot the first sample, first feature
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(targets[0, :, 0], 'b-', label='Target')
    plt.plot(predictions[0, :, 0], 'r--', label='Prediction')
    plt.title('Feature 0')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(targets[0, :, 1], 'b-', label='Target')
    plt.plot(predictions[0, :, 1], 'r--', label='Prediction')
    plt.title('Feature 1')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('se_cfc_predictions.png')
    plt.show()

def visualize_neuron_map(model):
    """Visualize the neuron map connectivity."""
    # Get the neuron map
    neuron_map = model.neuron_map
    
    # Get the recurrent mask
    recurrent_mask = tensor.to_numpy(neuron_map.get_recurrent_mask())
    
    # Get neuron groups
    neuron_groups = neuron_map.get_neuron_groups()
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot the recurrent mask
    plt.imshow(recurrent_mask, cmap='viridis')
    plt.colorbar(label='Connection Strength')
    plt.title('Recurrent Mask')
    
    # Add lines to separate neuron groups
    group_sizes = [
        len(neuron_groups['sensory']),
        len(neuron_groups['inter']),
        len(neuron_groups['command']),
        len(neuron_groups['motor'])
    ]
    
    # Calculate cumulative sizes using tensor operations
    group_sizes_tensor = tensor.convert_to_tensor(group_sizes)
    cum_sizes_tensor = tensor.cumsum(group_sizes_tensor)
    cum_sizes = tensor.to_numpy(cum_sizes_tensor)
    
    # Add vertical and horizontal lines
    for size in cum_sizes[:-1]:
        plt.axvline(x=size - 0.5, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=size - 0.5, color='r', linestyle='-', alpha=0.3)
    
    # Add labels
    group_names = ['Sensory', 'Inter', 'Command', 'Motor']
    
    # Calculate group centers using tensor operations
    group_sizes_half = ops.divide(group_sizes_tensor, 2.0)
    group_centers_tensor = ops.subtract(cum_sizes_tensor, group_sizes_half)
    group_centers = tensor.to_numpy(group_centers_tensor)
    
    plt.xticks(group_centers, group_names)
    plt.yticks(group_centers, group_names)
    
    plt.tight_layout()
    plt.savefig('se_cfc_connectivity.png')
    plt.show()

def main():
    """Main function to run the example."""
    # Generate data
    train_data, test_data = generate_sine_wave_data(
        num_samples=100,
        sequence_length=50,
        num_features=4
    )
    
    # Create model
    model = create_se_cfc_model()
    
    # Train model
    losses = train_model(model, train_data, num_epochs=20)
    
    # Evaluate model
    predictions, targets = evaluate_model(model, test_data)
    
    # Visualize results
    visualize_results(predictions, targets)
    
    # Visualize neuron map
    visualize_neuron_map(model)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('se_cfc_training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()