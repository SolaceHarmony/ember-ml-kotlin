"""
Recurrent Neural Network (RNN) Example.

This example demonstrates how to use the RNN and RNNCell classes
to create and train a basic recurrent neural network for sequence prediction.
"""

import matplotlib.pyplot as plt

from ember_ml import ops
from ember_ml.nn.modules.rnn import RNN
from ember_ml.nn import Sequential, tensor
from ember_ml.training import Optimizer, Adam # Import Adam directly
from ember_ml.training.loss import Loss, MSELoss # Import Loss and MSELoss from correct module
ops.set_backend("mlx")
def generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1):
    """Generate sine wave data for sequence prediction."""
    # Generate time points
    t = tensor.linspace(0, ops.multiply(2, ops.pi), seq_length)
    
    # Generate sine waves with random phase shifts
    X = tensor.zeros((num_samples, seq_length, num_features))
    y = tensor.zeros((num_samples, seq_length, num_features))
    
    for i in range(num_samples):
        # Random phase shift
        phase_shift = tensor.random_uniform(0, ops.multiply(2, ops.pi))
        
        # Generate sine wave with phase shift
        signal = ops.sin(ops.add(t, phase_shift))
        
        # Add some noise
        noise = tensor.random_normal((seq_length,), mean=0.0, stddev=0.1) # Ensure shape is a tuple
        noisy_signal = ops.add(signal, noise)
        
        # Store input and target
        X = tensor.tensor_scatter_nd_update(
            X,
            tensor.stack([
                ops.multiply(tensor.ones((seq_length,), dtype=tensor.int32), i),
                tensor.arange(seq_length),
                tensor.zeros((seq_length,), dtype=tensor.int32)
            ], axis=1),
            noisy_signal
        )
        
        y = tensor.tensor_scatter_nd_update(
            y,
            tensor.stack([
                ops.multiply(tensor.ones((seq_length,), dtype=tensor.int32), i),
                tensor.arange(seq_length),
                tensor.zeros((seq_length,), dtype=tensor.int32)
            ], axis=1),
            signal
        )
    
    return X, y

def train_rnn_model(model, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001):
    """Train an RNN model."""
    # Convert data to tensors
    X_train_tensor = tensor.convert_to_tensor(X_train, dtype=tensor.float32)
    y_train_tensor = tensor.convert_to_tensor(y_train, dtype=tensor.float32)
    
    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), learning_rate=learning_rate) # Use learning_rate argument
    loss_fn = MSELoss() # Instantiate MSELoss class directly
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle the data
        indices = tensor.random_permutation(tensor.shape(X_train)[0])
        shuffled_X = tensor.gather(X_train_tensor, indices)
        shuffled_y = tensor.gather(y_train_tensor, indices)
        
        # Train in batches
        for i in range(0, tensor.shape(X_train)[0], batch_size):
            batch_X = shuffled_X[i:i+batch_size]
            batch_y = shuffled_y[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = loss_fn(outputs, batch_y)
            
            # Backward pass and optimize
            grads = ops.gradients(loss, model.parameters())
            optimizer.step(grads)
            
            epoch_loss += tensor.to_numpy(loss)
        
        # Print progress using ops functions
        batches_per_epoch = ops.floor_divide(tensor.shape(X_train)[0], batch_size)
        # Avoid division by zero
        if tensor.item(batches_per_epoch) > 0:
            avg_loss = ops.divide(epoch_loss, float(tensor.item(batches_per_epoch)))
        else:
            avg_loss = 0.0 # Or handle as appropriate
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def evaluate_model(model, X_test, y_test):
    """Evaluate an RNN model."""
    # Convert data to tensors
    X_test_tensor = tensor.convert_to_tensor(X_test, dtype=tensor.float32)
    y_test_tensor = tensor.convert_to_tensor(y_test, dtype=tensor.float32)
    
    # Make predictions
    y_pred = model(X_test_tensor)
    
    # Compute loss
    loss_fn = MSELoss() # Instantiate MSELoss correctly
    loss = loss_fn(y_pred, y_test_tensor)
    
    # Convert predictions to numpy
    y_pred_np = tensor.to_numpy(y_pred)
    
    return tensor.to_numpy(loss), y_pred_np

def main():
    """Run the RNN example."""
    print("Recurrent Neural Network (RNN) Example")
    print("======================================")
    
    # Generate data
    print("\nGenerating data...")
    X, y = generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1)
    
    # Split data into train and test sets using ops and tensor functions
    train_size = tensor.cast(ops.multiply(0.8, tensor.shape(X)[0]), dtype=tensor.int32)
    # Slicing works directly on tensors
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train data shape: {tensor.shape(X_train)}")
    print(f"Test data shape: {tensor.shape(X_test)}")
    
    # Create RNN models with different activation functions
    print("\nCreating RNN models with different activation functions...")
    
    # Tanh activation
    tanh_model = Sequential([
        RNN(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            activation="tanh",
            dropout=0.2,
            bidirectional=True,
            return_sequences=True
        )
    ])
    
    # ReLU activation
    relu_model = Sequential([
        RNN(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            activation="relu",
            dropout=0.2,
            bidirectional=True,
            return_sequences=True
        )
    ])
    
    # Train the tanh model
    print("\nTraining RNN with tanh activation...")
    tanh_losses = train_rnn_model(
        tanh_model,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate the tanh model
    print("\nEvaluating RNN with tanh activation...")
    tanh_loss, tanh_preds = evaluate_model(tanh_model, X_test, y_test)
    print(f"Tanh RNN Test Loss: {tanh_loss:.6f}")
    
    # Train the relu model
    print("\nTraining RNN with ReLU activation...")
    relu_losses = train_rnn_model(
        relu_model,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate the relu model
    print("\nEvaluating RNN with ReLU activation...")
    relu_loss, relu_preds = evaluate_model(relu_model, X_test, y_test)
    print(f"ReLU RNN Test Loss: {relu_loss:.6f}")
    
    # Convert to numpy for visualization
    X_test_np = tensor.to_numpy(X_test)
    y_test_np = tensor.to_numpy(y_test)
    tanh_losses_np = tensor.to_numpy(tensor.convert_to_tensor(tanh_losses))
    relu_losses_np = tensor.to_numpy(tensor.convert_to_tensor(relu_losses))
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot the training losses
    plt.subplot(2, 1, 1)
    plt.plot(tanh_losses_np, label='Tanh RNN')
    plt.plot(relu_losses_np, label='ReLU RNN')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot the predictions for a sample
    plt.subplot(2, 1, 2)
    sample_idx = 0
    plt.plot(X_test_np[sample_idx, :, 0], 'b-', label='Input (Noisy)')
    plt.plot(y_test_np[sample_idx, :, 0], 'g-', label='Target')
    plt.plot(tanh_preds[sample_idx, :, 0], 'r--', label='Tanh RNN')
    plt.plot(relu_preds[sample_idx, :, 0], 'm--', label='ReLU RNN')
    plt.title('Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_example.png')
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()