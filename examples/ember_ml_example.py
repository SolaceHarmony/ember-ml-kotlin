"""
Backend-agnostic example of using Ember ML.
This example demonstrates how to create and use neural network components
in a backend-agnostic way using the ops abstraction layer.
"""
from typing import Tuple, Any # Add Any for type hint

from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn import tensor # Importing the Ember ML neural network module
import ember_ml.nn as nn

def create_model() -> nn.Sequential:
    """
    Create a simple neural network model.
    
    Returns:
        A sequential model with two linear layers and ReLU activation
    """
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

def train_step(model: nn.Sequential, x: EmberTensor, y: EmberTensor, learning_rate: float = 0.01) -> Any: # Change return type hint
    """
    Perform a single training step.
    
    Args:
        model: Neural network model
        x: Input data
        y: Target data
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Loss value as a tensor
    """
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass (not implemented yet)
    # loss.backward()
    
    # Update parameters (not implemented yet)
    # for param in model.parameters():
    #     param.data = ops.subtract(param.data, ops.multiply(tensor.convert_to_tensor(learning_rate), param.grad))
    
    return loss

def main() -> None:
    """Main function to demonstrate neural network components."""
    # Let Ember ML choose the best available backend
    ops.auto_select_backend() # Use ops re-export
    print(f"Using {ops.get_backend()} backend") # Use ops re-export
    
    # Create random data
    x = tensor.random_normal((32, 10))  # 32 samples, 10 features (Remove EmberTensor wrapper)
    y = tensor.random_normal((32, 1))   # 32 samples, 1 target (Remove EmberTensor wrapper)
    
    # Create model
    model = create_model()
    print(f"Model architecture:\n{model}")
    
    # Forward pass
    y_pred = model(x)
    print(f"Output shape: {tensor.shape(y_pred)}")
    
    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)
    print(f"Loss: {loss}")
    
    # Training step (will be limited without backward pass)
    loss = train_step(model, x, y)
    print(f"Loss after training step: {loss}")

if __name__ == "__main__":
    main()