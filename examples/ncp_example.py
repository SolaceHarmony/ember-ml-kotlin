"""
Neural Circuit Policy (NCP) example.

This example demonstrates how to use the NCP and AutoNCP classes
to create and train a neural circuit policy.
"""

import matplotlib.pyplot as plt
# NumPy import might still be needed for evaluation/plotting, keep it for now.
import numpy as np

from ember_ml import ops
from ember_ml.nn.modules.wiring import NCPMap # Import NCPMap instead of NCPWiring
from ember_ml.nn.modules import NCP, AutoNCP
from ember_ml.nn import tensor

def main():
    """Run the NCP example."""
    print("Neural Circuit Policy (NCP) Example")
    print("===================================")
    
    # Create a simple dataset
    print("\nCreating dataset...")
    X = tensor.reshape(tensor.linspace(0, ops.multiply(2.0, ops.pi), 100), (-1, 1)) # Use ops.pi
    y = ops.sin(X)
    
    # Convert to numpy for splitting
    X_np = tensor.to_numpy(X)
    y_np = tensor.to_numpy(y)
    
    # Split into train and test sets
    X_train, X_test = X_np[:80], X_np[80:]
    y_train, y_test = y_np[:80], y_np[80:]
    
    # Create a wiring configuration
    print("\nCreating wiring configuration...")
    wiring = NCPMap( # Use NCPMap
        inter_neurons=10,
        motor_neurons=1,
        sensory_neurons=0,
        sparsity_level=0.5,
        seed=42
    )
    
    # Create an NCP model
    print("\nCreating NCP model...")
    model = NCP(
        neuron_map=wiring, # Use neuron_map argument
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros"
    )
    
    # Train the model
    print("\nTraining NCP model...")
    learning_rate = 0.01
    epochs = 10  # Reduced from 100 to 10 for a smoke test
    batch_size = 16
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle the data using tensor functions
        # Convert train data to tensors first for shuffling
        X_train_tensor = tensor.convert_to_tensor(X_train)
        y_train_tensor = tensor.convert_to_tensor(y_train)
        indices = tensor.random_permutation(tensor.shape(X_train_tensor)[0])
        X_shuffled = tensor.gather(X_train_tensor, indices)
        y_shuffled = tensor.gather(y_train_tensor, indices)
        
        # Train in batches using tensor slicing
        for i in range(0, tensor.shape(X_train_tensor)[0], batch_size):
            # Use tensor slicing instead of NumPy slicing
            end_idx = min(i + batch_size, tensor.shape(X_train_tensor)[0])
            # Use slice_tensor with start indices and sizes. Assumes 2D input (batch, features).
            X_batch = tensor.slice_tensor(X_shuffled, [i, 0], [end_idx - i, tensor.shape(X_shuffled)[1]])
            y_batch = tensor.slice_tensor(y_shuffled, [i, 0], [end_idx - i, tensor.shape(y_shuffled)[1]])
            
            # Forward pass
            model.reset_state()
            # X_batch is already a tensor
            y_pred = model(X_batch)
            
            # Compute loss using ops.subtract
            # y_batch is already a tensor
            loss = ops.stats.mean(ops.square(ops.subtract(y_pred, y_batch)))
            
            # Compute gradients
            params = list(model.parameters())
            grads = ops.gradients(loss, params)
            
            # Update parameters
            for param, grad in zip(params, grads):
                param.data = ops.subtract(param.data, ops.multiply(tensor.convert_to_tensor(learning_rate), grad))
            
            epoch_loss += tensor.to_numpy(loss)

        # Calculate average loss using ops functions
        num_batches = ops.floor_divide(tensor.shape(X_train_tensor)[0], batch_size)
        # Avoid division by zero
        if tensor.item(num_batches) > 0:
            avg_loss = ops.divide(epoch_loss, float(tensor.item(num_batches)))
        else:
            avg_loss = 0.0 # Or handle as appropriate
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}") # Print avg_loss
    
    # Evaluate the model
    print("\nEvaluating NCP model...")
    model.reset_state()
    y_pred = tensor.to_numpy(model(tensor.convert_to_tensor(X_test)))
    test_loss = stats.mean(np.square(y_pred - y_test))
    print(f"Test Loss: {test_loss:.6f}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot the loss
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Plot the predictions
    plt.subplot(2, 1, 2)
    plt.plot(X_test, y_test, label="True")
    plt.plot(X_test, y_pred, label="Predicted")
    plt.title("Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ncp_example.png")
    plt.show()
    
    # Create an AutoNCP model
    print("\nCreating AutoNCP model...")
    auto_model = AutoNCP(
        units=20,
        output_size=1,
        sparsity_level=0.5,
        seed=42,
        activation="tanh",
        use_bias=True
    )
    
    print("\nAutoNCP model created successfully!")
    print(f"Units: {auto_model.units}")
    print(f"Output size: {auto_model.output_size}")
    print(f"Sparsity level: {auto_model.sparsity_level}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()