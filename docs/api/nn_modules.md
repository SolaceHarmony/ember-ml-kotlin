# Neural Network Modules (nn.modules)

The `ember_ml.nn.modules` package provides a comprehensive set of backend-agnostic neural network modules for building machine learning models. These modules follow a consistent API across different backends and are designed to be composable and extensible, supporting various connectivity patterns including advanced and spatial wiring configurations through the `NeuronMap` hierarchy.

## Importing

```python
from ember_ml.nn import modules
```

## Base Classes

### Module

`Module` is the base class for all neural network modules in Ember ML. It provides common functionality like parameter management, forward pass, and state tracking.

```python
from ember_ml.nn.modules import Module

class MyModule(Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters
        
    def forward(self, x):
        # Implement forward pass
        return x
```

### Parameter

`Parameter` represents a trainable parameter in a neural network module. It wraps a tensor and indicates that it should be included when collecting trainable variables. When using a `Parameter` in calculations, pass it directly to `ops` module functions (e.g., `ops.add`, `ops.matmul`); the backend abstraction layer will handle accessing the underlying tensor data. Avoid accessing the `.data` attribute directly unless necessary for specific low-level operations.

```python
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(tensor.random_normal((in_features, out_features)))
        self.bias = Parameter(tensor.zeros((out_features,)))
        
    def forward(self, x):
        # Use ops functions for all operations involving Parameters
        y = ops.matmul(x, self.weight)
        return ops.add(y, self.bias)
```

### BaseModule

`BaseModule` extends `Module` with additional functionality for building more complex neural network modules.


### Deferred Initialization (Build Pattern)

Many Ember ML modules, particularly those whose internal structure depends on the shape of the input data (like layers using `NeuronMap` where `input_dim` might not be known at initialization), follow a deferred initialization pattern.

- **Initialization (`__init__`)**: Only parameters independent of the input shape are defined. Parameters like weights or biases that depend on the input dimension are typically set to `None`. The `input_size` attribute might also be initialized to `None`.
- **Build (`build(input_shape)`)**: This method is automatically called by the base `Module` class during the *first* forward pass (`__call__`). It receives the shape of the first input tensor. Inside `build`, the module finalizes its initialization:
    - It determines the input dimension from `input_shape`.
    - If using a `NeuronMap`, it calls the map's `build(input_dim)` method (which should set the map's `input_dim`).
    - It sets the module's `input_size` attribute.
    - It initializes all remaining parameters (weights, biases) whose shapes depend on the now-known input dimension.
- **`self.built` Flag**: A `built` flag (managed by the base class) tracks whether the `build` method has been executed, preventing re-initialization.

This pattern allows for flexible module creation where input dimensions don't need to be specified upfront. Modules like `NCP`, `LTC`, and those using neuron maps utilize this pattern. Standard layers like `Dense` that require `input_size` during `__init__` typically initialize all parameters immediately and do not rely on deferred building (their `build` method might be empty or only call `super().build`).

## Core Modules

### Dense

`Dense` implements a fully connected layer, also known as a linear or dense layer.

```python
from ember_ml.nn.modules import Dense
from ember_ml.nn import tensor

# Create a dense layer
layer = Dense(in_features=10, out_features=5, activation='relu')

# Forward pass
x = tensor.random_normal((32, 10))  # Batch of 32 samples with 10 features each
y = layer(x)  # Shape: (32, 5)
```

### NCP (Neural Circuit Policy)

`NCP` implements a neural circuit policy, a type of neural network with custom connectivity patterns.

```python
from ember_ml.nn.modules import NCP
from ember_ml.nn.modules.wiring import NCPMap

# Create a wiring configuration
wiring = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)

# Create an NCP
# Note: NCP internally uses ops functions with Parameters
ncp = NCP(neuron_map=wiring, activation='tanh') # Use neuron_map argument

# Forward pass
x = tensor.random_normal((32, 8))  # Batch of 32 samples with 8 features each
y = ncp(x)  # Shape: (32, 3)
```

Note: The `neuron_map` parameter can accept any derivative of `NeuronMap`, allowing for various connectivity patterns, including those defined by spatial maps like `EnhancedNeuronMap` and `EnhancedNCPMap`. Refer to the [Neuron Maps (Wiring) Documentation](nn_modules_wiring.md) for more details.

### Sequential

`Sequential` is a container module that chains multiple modules together, applying them sequentially to the input.

```python
from ember_ml.nn.modules import Sequential, Dense, ReLU, Dropout
from ember_ml.nn import tensor

# Create a sequential model
model = Sequential([
    Dense(in_features=10, out_features=64),
    ReLU(),
    Dropout(rate=0.5),
    Dense(in_features=64, out_features=1)
])

# Forward pass
# Note: The 'training' argument is automatically passed down to modules
# like Dropout within the Sequential container.
x = tensor.random_normal((32, 10))
output = model(x, training=True) # Shape: (32, 1)
```

### BatchNormalization

`BatchNormalization` applies Batch Normalization, a technique to stabilize and accelerate training by normalizing the activations of the previous layer. It maintains running estimates of the mean and variance of activations during training, which are then used for normalization during inference.

```python
from ember_ml.nn.modules import BatchNormalization, Dense, Sequential
from ember_ml.nn import tensor

# Example usage within a Sequential model
model = Sequential([
    Dense(in_features=10, out_features=64),
    BatchNormalization(), # Automatically infers feature dimension
    ReLU(),
    Dense(in_features=64, out_features=1)
])

# Forward pass (Batch Normalization uses batch stats during training)
x = tensor.random_normal((32, 10))
output = model(x, training=True) # Training=True is important for BN

# During inference (model.eval() mode typically), Batch Normalization
# would use its moving averages for mean and variance.
# output_eval = model(x, training=False)
```

### AutoNCP

`AutoNCP` provides a convenient way to create an NCP with automatic wiring configuration.

```python
from ember_ml.nn.modules import AutoNCP

# Create an AutoNCP
auto_ncp = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5,
    seed=42
)

# Forward pass
x = tensor.random_normal((32, 16))  # Batch of 32 samples with 16 features each
y = auto_ncp(x)  # Shape: (32, 10)
```

Note: `AutoNCP` automatically configures an `NCPMap`. For more advanced or spatial wiring, you can create an instance of the desired `NeuronMap` derivative (e.g., `EnhancedNCPMap`) and pass it directly to the `NCP` class instead of using `AutoNCP`. Refer to the [Neuron Maps (Wiring) Documentation](nn_modules_wiring.md) for more details.


## Activation Functions

The following activation functions are available:

| Activation | Description |
|------------|-------------|
| `ReLU` | Rectified Linear Unit activation |
| `Tanh` | Hyperbolic Tangent activation |
| `Sigmoid` | Sigmoid activation |
| `Softmax` | Softmax activation |
| `Softplus` | Softplus activation |
| `LeCunTanh` | LeCun Tanh activation |
| `Dropout` | Dropout regularization |

```python
from ember_ml.nn.modules import Dense, ReLU, Dropout
from ember_ml.nn import tensor

# Create a dense layer with ReLU activation
layer1 = Dense(in_features=10, out_features=5)
activation = ReLU()
dropout = Dropout(rate=0.2)

# Forward pass
x = tensor.random_normal((32, 10))
y = layer1(x)
y = activation(y)
y = dropout(y, training=True)
```

## Neuron Maps

Neuron maps (formerly called wirings) define the connectivity patterns between neurons in a neural network, including advanced configurations with spatial properties. For detailed documentation on all available neuron map implementations, please refer to the [Neuron Maps Documentation](nn_modules_wiring.md).

Here is a brief overview of some key neuron map implementations:

### NeuronMap

`NeuronMap` is the base class for all neuron maps.

### NCPMap

`NCPMap` implements a Neural Circuit Policy connectivity pattern.

```python
from ember_ml.nn.modules.wiring import NCPMap

# Create an NCP wiring configuration
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)
```

### FullyConnectedMap

`FullyConnectedMap` implements a fully connected connectivity pattern.

```python
from ember_ml.nn.modules.wiring import FullyConnectedMap

# Create a fully connected wiring configuration
neuron_map = FullyConnectedMap(
    units=10,
    output_size=5,
    input_size=8
)
```

### RandomMap

`RandomMap` implements a random connectivity pattern.

```python
from ember_ml.nn.modules.wiring import RandomMap

# Create a random wiring configuration
neuron_map = RandomMap(
    units=10,
    output_size=5,
    input_size=8,
    sparsity_level=0.5,
    seed=42
)
```

## Recurrent Neural Networks (RNN)

The `modules` package provides various recurrent neural network implementations.

### RNN

`RNN` implements a basic recurrent neural network.

```python
from ember_ml.nn.modules import RNN
from ember_ml.nn import tensor

# Create an RNN
rnn = RNN(
    input_size=10,
    hidden_size=20,
    activation='tanh'
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = rnn(x)  # y: (32, 5, 20), h: (32, 20)
```

### LSTM

`LSTM` implements a Long Short-Term Memory network.

```python
from ember_ml.nn.modules import LSTM
from ember_ml.nn import tensor

# Create an LSTM
lstm = LSTM(
    input_size=10,
    hidden_size=20
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, (h, c) = lstm(x)  # y: (32, 5, 20), h: (32, 20), c: (32, 20)
```

### GRU

`GRU` implements a Gated Recurrent Unit network.

```python
from ember_ml.nn.modules import GRU
from ember_ml.nn import tensor

# Create a GRU
gru = GRU(
    input_size=10,
    hidden_size=20
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = gru(x)  # y: (32, 5, 20), h: (32, 20)
```

### CfC

`CfC` implements a Closed-form Continuous-time network.

```python
from ember_ml.nn.modules import CfC
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a NeuronMap
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=5,
    sensory_neurons=10,
    seed=42
)

# Create a CfC layer with the neuron map
cfc = CfC(neuron_map=neuron_map)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = cfc(x)  # y: (32, 5, 5), h: (32, 20)
```

### LTC

`LTC` implements a Liquid Time-Constant network.

```python
from ember_ml.nn.modules import LTC
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a NeuronMap
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=5,
    sensory_neurons=10,
    seed=42
)

# Create an LTC layer with the neuron map
ltc = LTC(neuron_map=neuron_map)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = ltc(x)  # y: (32, 5, 5), h: (32, 20)
```

## Stride-Aware Modules

Stride-aware modules are specialized for processing temporal data with variable strides.

### StrideAware

`StrideAware` is the base class for stride-aware modules.


### StrideAwareCfC

`StrideAwareCfC` implements a stride-aware Closed-form Continuous-time network.

```python
from ember_ml.nn.modules import StrideAwareCfC
from ember_ml.nn import tensor

# Create a StrideAwareCfC
# Create a NeuronMap first for the StrideAwareCfC layer
from ember_ml.nn.modules.wiring import FullyConnectedMap
neuron_map = FullyConnectedMap(units=20, input_dim=10, output_dim=20)
stride_cfc = StrideAwareCfC(
    neuron_map_or_cell=neuron_map, # Example assuming it takes a map
    stride_lengths=[1, 2, 4]
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = stride_cfc(x)  # y: (32, 5, 20), h: (32, 20)
```


## Building Complex Models

You can combine these modules to build complex neural network architectures:

```python
from ember_ml.nn.modules import Module, Dense, LSTM, Dropout
from ember_ml.nn import tensor

class SequenceClassifier(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.dropout = Dropout(rate=0.2)
        self.dense = Dense(in_features=hidden_size, out_features=num_classes, activation='softmax')
        
    def forward(self, x, training=False):
        # x shape: (batch_size, sequence_length, input_size)
        y, (h, _) = self.lstm(x)
        # Use the last hidden state
        # Assume h is the last hidden state tensor (or stacked states)
        # If h is stacked (num_layers*dirs, batch, hidden), take the last layer's state
        # Example: h_last = h[-1] if h.ndim == 3 else h
        h_last = h[-1] if tensor.ndim(h) == 3 else h # Get last layer state if stacked
        h_dropout = self.dropout(h_last, training=training)
        # Pass through the dense layer
        output = self.dense(h_dropout)
        return output

# Create a sequence classifier
model = SequenceClassifier(input_size=10, hidden_size=20, num_classes=5)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y = model(x, training=True)  # Shape: (32, 5)
```

## Backend Support

All modules are backend-agnostic and work with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.

```python
from ember_ml.nn.modules import Dense
from ember_ml.ops import set_backend

# Use NumPy backend
set_backend('numpy')
dense_numpy = Dense(in_features=10, out_features=5)

# Use PyTorch backend
set_backend('torch')
dense_torch = Dense(in_features=10, out_features=5)

# Use MLX backend
set_backend('mlx')
dense_mlx = Dense(in_features=10, out_features=5)
```

## Implementation Details

The neural network modules are implemented using a layered architecture:

1. **Base Classes**: Provide common functionality for all modules.
2. **Core Modules**: Implement basic neural network components, often integrating with the `NeuronMap` hierarchy to define internal structure and connectivity, including spatial aspects.
3. **Advanced Modules**: Implement specialized neural network architectures.

This architecture allows Ember ML to provide a consistent API across different backends while still leveraging the unique capabilities of each backend.

## Additional Resources

For more detailed information on specific modules, see the following resources:

- [RNN Modules Documentation](nn_modules_rnn.md): Detailed documentation on recurrent neural network modules
- [Neuron Maps Documentation](nn_modules_wiring.md): Detailed documentation on neuron maps
- [Tensor Module Documentation](nn_tensor.md): Documentation on tensor operations used by the modules