# Neural Network Modules

This directory contains neural network module implementations for ember_ml, with a focus on Neural Circuit Policies (NCPs) and other specialized neural network architectures.

## Overview

Neural network modules are the building blocks of neural network models. They encapsulate specific neural network architectures and provide a unified interface for forward and backward passes.

## Available Modules

### Neural Circuit Policy (NCP)

The `NCP` class implements a Neural Circuit Policy, which is a recurrent neural network with a specific connectivity pattern defined by a wiring configuration.

```python
from ember_ml.nn.wirings import NCPWiring
from ember_ml.nn.modules import NCP

# Create a wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)

# Create an NCP model
model = NCP(
    wiring=wiring,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros"
)

# Forward pass
outputs = model(inputs)
```

### Auto NCP

The `AutoNCP` class is a convenience wrapper around the `NCP` class that automatically configures the wiring based on the number of units and outputs.

```python
from ember_ml.nn.modules import AutoNCP

# Create an AutoNCP model
model = AutoNCP(
    units=20,
    output_size=5,
    sparsity_level=0.5,
    seed=42,
    activation="tanh",
    use_bias=True
)

# Forward pass
outputs = model(inputs)
```

## Implementation Details

### NCP Module

The NCP module applies the wiring masks during the forward pass to constrain the connectivity of the network. This ensures that the network follows the specified connectivity pattern.

The forward pass consists of the following steps:

1. Apply the input mask to the input tensor
2. Compute the recurrent state using the recurrent mask
3. Apply the activation function
4. Apply the output mask to the state tensor
5. Return the output tensor

### AutoNCP Module

The AutoNCP module automatically creates an NCPWiring configuration based on the specified number of units and outputs. It then creates an NCP module using this wiring configuration.

## Usage Examples

### Training an NCP Model

```python
import numpy as np
from ember_ml import ops
from ember_ml.nn.wirings import NCPWiring
from ember_ml.nn.modules import NCP

# Create a simple dataset
X = ops.reshape(ops.linspace(0, 2 * np.pi, 100), (-1, 1))
y = ops.sin(X)

# Convert to numpy for splitting
X_np = tensor.to_numpy(X)
y_np = tensor.to_numpy(y)

# Split into train and test sets
X_train, X_test = X_np[:80], X_np[80:]
y_train, y_test = y_np[:80], y_np[80:]

# Create a wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=1,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)

# Create an NCP model
model = NCP(
    wiring=wiring,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros"
)

# Train the model
learning_rate = 0.01
epochs = 100
batch_size = 16

for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Train in batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward pass
        model.reset_state()
        y_pred = model(tensor.convert_to_tensor(X_batch))
        
        # Compute loss
        loss = ops.stats.mean(ops.square(y_pred - tensor.convert_to_tensor(y_batch)))
        
        # Compute gradients
        params = list(model.parameters())
        grads = ops.gradients(loss, params)
        
        # Update parameters
        for param, grad in zip(params, grads):
            param.data = ops.subtract(param.data, ops.multiply(tensor.convert_to_tensor(learning_rate), grad))
        
        epoch_loss += tensor.to_numpy(loss)
    
    epoch_loss /= (len(X_train) // batch_size)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")

# Evaluate the model
model.reset_state()
y_pred = tensor.to_numpy(model(tensor.convert_to_tensor(X_test)))
test_loss = np.mean(np.square(y_pred - y_test))
print(f"Test Loss: {test_loss:.6f}")
```

### Custom NCP Modules

You can create custom NCP modules by subclassing the `NCP` class and overriding the `forward` method:

```python
from ember_ml.nn.modules import NCP
from ember_ml import ops

class CustomNCP(NCP):
    """
    Custom Neural Circuit Policy module.
    """
    
    def __init__(self, wiring, activation="tanh", use_bias=True, **kwargs):
        """
        Initialize a custom NCP module.
        
        Args:
            wiring: Wiring configuration
            activation: Activation function to use
            use_bias: Whether to use bias
            **kwargs: Additional arguments
        """
        super().__init__(wiring, activation, use_bias, **kwargs)
    
    def forward(self, inputs, state=None, return_state=False):
        """
        Forward pass of the custom NCP module.
        
        Args:
            inputs: Input tensor
            state: Optional state tensor
            return_state: Whether to return the state
            
        Returns:
            Output tensor, or tuple of (output, state) if return_state is True
        """
        # Custom forward pass
        # ...
        
        # Call the parent forward method
        return super().forward(inputs, state, return_state)
```

## Related Documentation

For more detailed information on Neural Circuit Policies and their implementation, see the [Neural Circuit Policies documentation](../../../docs/neural_circuit_policies.md).

For information on the wiring configurations used with NCP modules, see the [Neural Network Wirings documentation](../wirings/README.md).