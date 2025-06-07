# Neural Network Modules Architecture

This document provides a comprehensive overview of the neural network modules in Ember ML, with a focus on the Neural Circuit Policies (NCPs) and their implementation.

## Neural Circuit Policies (NCPs)

### Overview

Neural Circuit Policies (NCPs) are a class of biologically-inspired neural networks that use structured connectivity patterns to implement control policies. They were originally introduced by Mathias Lechner et al. in their paper "Neural Circuit Policies Enabling Auditable Autonomy" (Nature Machine Intelligence, 2020).

Ember ML includes an implementation of NCPs based on the original work, with adaptations to fit into the Ember ML architecture and backend-agnostic design.

### Acknowledgment

The NCP implementation in Ember ML is based on the original work by Mathias Lechner and the ncps library (https://github.com/mlech26l/ncps), which is licensed under the Apache-2.0 License. We acknowledge and thank the original authors for their contribution to the field.

### Architecture

The NCP architecture in Ember ML consists of several key components:

1. **Wiring**: Defines the connectivity pattern between neurons
2. **NCP Module**: Implements the neural network using the wiring configuration
3. **AutoNCP**: A convenience wrapper that automatically configures the wiring

#### Wiring

The `Wiring` class is the base class for all wiring configurations. It defines the interface for creating connectivity patterns between neurons:

```python
class Wiring:
    def __init__(self, units, output_dim, input_dim=None, sparsity_level=0.5, seed=None):
        self.units = units
        self.output_dim = output_dim
        self.input_dim = input_dim or units
        self.sparsity_level = sparsity_level
        self.seed = seed
        
    def build(self, input_dim=None):
        """Build the wiring configuration."""
        pass
        
    def get_config(self):
        """Get the configuration of the wiring."""
        pass
        
    @classmethod
    def from_config(cls, config):
        """Create a wiring configuration from a configuration dictionary."""
        pass
```

The `NCPWiring` class extends `Wiring` to implement the specific connectivity pattern used in Neural Circuit Policies:

```python
class NCPWiring(Wiring):
    def __init__(
        self, 
        inter_neurons,
        motor_neurons,
        sensory_neurons=0,
        sparsity_level=0.5, 
        seed=None,
        # ... other parameters
    ):
        # Initialize the wiring
        super().__init__(
            units=inter_neurons + motor_neurons + sensory_neurons,
            output_dim=motor_neurons,
            input_dim=None,
            sparsity_level=sparsity_level,
            seed=seed
        )
        
        self.inter_neurons = inter_neurons
        self.motor_neurons = motor_neurons
        self.sensory_neurons = sensory_neurons
        # ... other initialization
```

In an NCP wiring, neurons are divided into three groups:
- **Sensory neurons**: Receive input from the environment
- **Inter neurons**: Process information internally
- **Motor neurons**: Produce output to the environment

The connectivity pattern between these groups is defined by the sparsity level and can be customized.

#### NCP Module

The `NCP` class implements a neural network using the wiring configuration:

```python
class NCP(Module):
    def __init__(
        self,
        wiring,
        activation="tanh",
        use_bias=True,
        # ... other parameters
    ):
        super().__init__()
        
        self.wiring = wiring
        self.activation_name = activation
        self.activation = nn.modules.activation.get_activation(activation)
        self.use_bias = use_bias
        # ... other initialization
        
        # Get masks from wiring
        self.input_mask = tensor.convert_to_tensor(self.wiring.get_input_mask())
        self.recurrent_mask = tensor.convert_to_tensor(self.wiring.get_recurrent_mask())
        self.output_mask = tensor.convert_to_tensor(self.wiring.get_output_mask())
        
        # Initialize weights
        self._kernel = Parameter(...)
        self._recurrent_kernel = Parameter(...)
        self._bias = Parameter(...) if self.use_bias else None
```

The `NCP` class uses the masks from the wiring configuration to implement the connectivity pattern in the neural network. The forward pass applies these masks to ensure that only the connections defined in the wiring are used:

```python
def forward(self, inputs, state=None, return_state=False):
    # Apply input mask
    masked_inputs = ops.multiply(inputs, self.input_mask)
    
    # Apply recurrent mask
    masked_state = ops.matmul(state, self.recurrent_mask)
    
    # Compute new state
    new_state = ops.matmul(masked_inputs, self.kernel)
    if self.use_bias:
        new_state = ops.add(new_state, self.bias)
    new_state = ops.add(new_state, ops.matmul(masked_state, self.recurrent_kernel))
    new_state = self.activation(new_state)
    
    # Compute output - only include motor neurons
    masked_output = ops.multiply(new_state, self.output_mask)
    
    # Extract only the motor neurons
    output = masked_output[:, :self.wiring.output_dim]
    
    # ... return output and optionally state
```

#### AutoNCP

The `AutoNCP` class is a convenience wrapper around the `NCP` class that automatically configures the wiring based on the number of units and outputs:

```python
class AutoNCP(NCP):
    def __init__(
        self,
        units,
        output_size,
        sparsity_level=0.5,
        seed=None,
        # ... other parameters
    ):
        # Calculate the number of inter and command neurons
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons
        
        # Create the wiring
        wiring = NCPWiring(
            inter_neurons=inter_neurons,
            motor_neurons=output_size,
            sensory_neurons=0,  # No sensory neurons in AutoNCP
            sparsity_level=sparsity_level,
            seed=seed,
        )
        
        # Initialize the NCP module
        super().__init__(
            wiring=wiring,
            # ... other parameters
        )
```

### Control Theory and AutoNCP

The AutoNCP implementation in Ember ML is designed to facilitate the application of control theory principles to neural networks. By automatically configuring the wiring based on the number of units and outputs, AutoNCP makes it easier to create neural networks that can be used for control tasks.

The key aspects of control theory implemented in AutoNCP include:

1. **Separation of Concerns**: The division of neurons into sensory, inter, and motor groups allows for a clear separation of input processing, internal computation, and output generation.

2. **Sparse Connectivity**: The sparsity level parameter controls the density of connections between neurons, allowing for more efficient computation and better generalization.

3. **Recurrent Connections**: The recurrent connections in the network enable the implementation of dynamic systems that can maintain state over time, which is essential for control tasks.

4. **Output Masking**: The output mask ensures that only the motor neurons contribute to the output, providing a clear interface for the control policy.

### Usage

Here's an example of how to use the AutoNCP module:

```python
from ember_ml.nn.modules import AutoNCP

# Create an AutoNCP module
model = AutoNCP(
    units=64,
    output_size=4,
    sparsity_level=0.5,
    activation="tanh"
)

# Forward pass
input_tensor = EmberTensor.random_normal((32, 64))  # Batch of 32 samples with 64 features
output = model(input_tensor)  # Shape: (32, 4)
```

## Other Neural Network Modules

Ember ML includes a variety of other neural network modules, including:

### Basic Modules

- **Linear**: A linear transformation module
- **Activation**: An activation function module
- **Sequential**: A sequential container module

### Recurrent Networks

- **RNN**: A simple recurrent neural network
- **LSTM**: Long Short-Term Memory network
- **GRU**: Gated Recurrent Unit network
- **LTC**: Liquid Time-Constant network
- **CFC**: Closed-form Continuous-time network

### Stride-Aware Cells

Stride-aware cells are a special type of recurrent cell that can process inputs at different time scales. They are particularly useful for processing time series data with varying sampling rates.

The `StrideAwareCell` class is the base class for all stride-aware cells:

```python
class StrideAwareCell(Module):
    def __init__(self, stride_length=1, **kwargs):
        super().__init__(**kwargs)
        self.stride_length = stride_length
        
    def forward(self, inputs, state=None, **kwargs):
        # Process inputs at the current time step
        # ...
        
        # Update state based on stride length
        # ...
        
        return output, new_state
```

The `StrideAwareCfC` class extends `StrideAwareCell` to implement a stride-aware version of the Closed-form Continuous-time cell:

```python
class StrideAwareCfC(Module):
    def __init__(
        self,
        units_or_cell,
        stride_length=1,
        # ... other parameters
    ):
        super().__init__()
        
        if isinstance(units_or_cell, StrideAwareWiredCfCCell):
            self.cell = units_or_cell
        elif isinstance(units_or_cell, Wiring):
            self.cell = StrideAwareWiredCfCCell(wiring=units_or_cell, **kwargs)
        else:
            # Create a cell with a default wiring
            # ...
```

## Integration with Backend Abstraction

All neural network modules in Ember ML are designed to work with the backend abstraction layer, allowing them to run on any supported backend (NumPy, PyTorch, MLX) without modification.

The modules use the `ops` module for all tensor operations, which delegates to the current backend:

```python
from ember_ml import ops

# Create a tensor
tensor = tensor.random_normal((10, 10))

# Perform operations
result = ops.matmul(tensor, tensor)
```

This design allows the neural network modules to be backend-agnostic, making it easy to switch between backends without changing the model code.

## Conclusion

The neural network modules in Ember ML provide a flexible and powerful framework for building and training neural networks. The implementation of Neural Circuit Policies (NCPs) based on the work by Mathias Lechner et al. adds a biologically-inspired approach to neural network design, with a focus on control theory principles.

The backend-agnostic design of the modules allows them to run on any supported backend, making it easy to switch between backends without changing the model code.