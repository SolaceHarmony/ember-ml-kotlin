"""
Long Short-Term Memory (LSTM) Neural Network Example.

This example demonstrates how to use the LSTM and LSTMCell classes
to create and train a recurrent neural network for sequence prediction.
"""

import matplotlib.pyplot as plt

from ember_ml import ops
from ember_ml.nn.modules.rnn import LSTM, LSTMCell
from ember_ml.nn import Module, Sequential, tensor
from ember_ml.training import Optimizer, Loss

def generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1):
    """Generate sine wave data for sequence prediction."""
    # Generate time points
    t = tensor.linspace(0, 2 * ops.pi, seq_length)
    
    # Generate sine waves with random phase shifts
    X = tensor.zeros((num_samples, seq_length, num_features))
    y = tensor.zeros((num_samples, seq_length, num_features))
    
    for i in range(num_samples):
        # Random phase shift
        phase_shift = tensor.random_uniform(0, 2 * ops.pi)
        
        # Generate sine wave with phase shift
        signal = ops.sin(ops.add(t, phase_shift)) # Use ops.add
        
        # Add some noise
        noise = tensor.random_normal(0, 0.1, seq_length)
        noisy_signal = ops.add(signal, noise)
        
        # Store input and target
        X = tensor.tensor_scatter_nd_update(
            X,
            tensor.stack([
                ops.multiply(tensor.ones((seq_length,), dtype=tensor.int32), i), # Use ops.multiply
                tensor.arange(seq_length),
                tensor.zeros((seq_length,), dtype=tensor.int32)
            ], axis=1),
            noisy_signal
        )
        
        y = tensor.tensor_scatter_nd_update(
            y,
            tensor.stack([
                ops.multiply(tensor.ones((seq_length,), dtype=tensor.int32), i), # Use ops.multiply
                tensor.arange(seq_length),
                tensor.zeros((seq_length,), dtype=tensor.int32)
            ], axis=1),
            signal
        )
    
    return X, y

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001):
    """Train an LSTM model."""
    # Convert data to tensors
    X_train_tensor = tensor.convert_to_tensor(X_train, dtype=tensor.float32)
    y_train_tensor = tensor.convert_to_tensor(y_train, dtype=tensor.float32)
    
    # Define optimizer and loss function
    optimizer = Optimizer.adam(model.parameters(), learning_rate=learning_rate)
    loss_fn = Loss.mse()
    
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
        
        # Print progress
        batches_per_epoch = ops.floor_divide(tensor.shape(X_train)[0], batch_size) # Use ops.floor_divide
        avg_loss = ops.divide(epoch_loss, batches_per_epoch) # Use ops.divide
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def evaluate_model(model, X_test, y_test):
    """Evaluate an LSTM model."""
    # Convert data to tensors
    X_test_tensor = tensor.convert_to_tensor(X_test, dtype=tensor.float32)
    y_test_tensor = tensor.convert_to_tensor(y_test, dtype=tensor.float32)
    
    # Make predictions
    y_pred = model(X_test_tensor)
    
    # Compute loss
    loss_fn = Loss.mse()
    loss = loss_fn(y_pred, y_test_tensor)
    
    # Convert predictions to numpy
    y_pred_np = tensor.to_numpy(y_pred)
    
    return tensor.to_numpy(loss), y_pred_np

def main():
    """Run the LSTM example."""
    print("Long Short-Term Memory (LSTM) Neural Network Example")
    print("===================================================")
    
    # Generate data
    print("\nGenerating data...")
    X, y = generate_sine_wave_data(num_samples=1000, seq_length=100, num_features=1)
    
    # Split data into train and test sets
    train_size = tensor.cast(ops.multiply(0.8, tensor.shape(X)[0]), tensor.int32) # Use ops.multiply and tensor.cast
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train data shape: {tensor.shape(X_train)}")
    print(f"Test data shape: {tensor.shape(X_test)}")
    
    # Create a single-layer LSTM model
    print("\nCreating single-layer LSTM model...")
    single_layer_model = Sequential([
        LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            return_sequences=True
        )
    ])
    
    # Train the single-layer model
    print("\nTraining single-layer LSTM model...")
    single_layer_losses = train_lstm_model(
        single_layer_model,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate the single-layer model
    print("\nEvaluating single-layer LSTM model...")
    single_layer_loss, single_layer_preds = evaluate_model(single_layer_model, X_test, y_test)
    print(f"Single-layer LSTM Test Loss: {single_layer_loss:.6f}")
    
    # Create a multi-layer bidirectional LSTM model
    print("\nCreating multi-layer bidirectional LSTM model...")
    multi_layer_model = Sequential([
        LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            return_sequences=True
        )
    ])
    
    # Train the multi-layer model
    print("\nTraining multi-layer bidirectional LSTM model...")
    multi_layer_losses = train_lstm_model(
        multi_layer_model,
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate the multi-layer model
    print("\nEvaluating multi-layer bidirectional LSTM model...")
    multi_layer_loss, multi_layer_preds = evaluate_model(multi_layer_model, X_test, y_test)
    print(f"Multi-layer Bidirectional LSTM Test Loss: {multi_layer_loss:.6f}")
    
    # Convert to numpy for visualization
    X_test_np = tensor.to_numpy(X_test)
    y_test_np = tensor.to_numpy(y_test)
    single_layer_losses_np = tensor.to_numpy(tensor.convert_to_tensor(single_layer_losses))
    multi_layer_losses_np = tensor.to_numpy(tensor.convert_to_tensor(multi_layer_losses))
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot the training losses
    plt.subplot(2, 1, 1)
    plt.plot(single_layer_losses_np, label='Single-layer LSTM')
    plt.plot(multi_layer_losses_np, label='Multi-layer Bidirectional LSTM')
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
    plt.plot(single_layer_preds[sample_idx, :, 0], 'r--', label='Single-layer LSTM')
    plt.plot(multi_layer_preds[sample_idx, :, 0], 'm--', label='Multi-layer Bidirectional LSTM')
    plt.title('Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_example.png')
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()