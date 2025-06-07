# Recurrent Neural Network Modules (nn.modules.rnn)

The `ember_ml.nn.modules.rnn` package provides a comprehensive set of recurrent neural network (RNN) modules for sequential data processing. These modules are backend-agnostic and follow a consistent API across different backends. The implementation has been simplified by removing cell-based architecture and integrating all functionality directly into layer classes, with some modules capable of utilizing advanced neuron maps, including those with spatial properties.

## Importing

```python
from ember_ml.nn.modules import rnn
# or import specific modules
from ember_ml.nn.modules.rnn import LSTM, GRU, CfC, LQNet, CTRQNet
```

## Basic RNN Modules

### RNN

`RNN` implements a basic recurrent neural network layer.

```python
from ember_ml.nn.modules.rnn import RNN
from ember_ml.nn import tensor

# Create an RNN layer
rnn_layer = RNN(input_size=10, hidden_size=20, activation='tanh')

# Forward pass (sequence)
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
# Layer returns outputs by default. Set return_state=True to get final state.
# y, [h] = rnn_layer(x, return_state=True)
# If return_sequences=True (default for RNN layer? Check impl): y shape=(32, 5, 20)
# If return_sequences=False: y shape=(32, 20)
# Final state h shape=(num_layers*num_directions, 32, 20) -> (1, 32, 20) for default
y = rnn_layer(x) # Example without state return
```

## LSTM Modules

### LSTM

`LSTM` implements a Long Short-Term Memory layer.

```python
from ember_ml.nn.modules.rnn import LSTM
from ember_ml.nn import tensor

# Create an LSTM layer
lstm_layer = LSTM(input_size=10, hidden_size=20)

# Forward pass (sequence)
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
# Layer returns outputs by default. Set return_state=True to get final state.
# y, (h, c) = lstm_layer(x, return_state=True)
# If return_sequences=True (default): y shape=(32, 5, 20)
# If return_sequences=False: y shape=(32, 20)
# Final state h shape=(num_layers*num_directions, 32, 20) -> (1, 32, 20) for default
# Final state c shape=(num_layers*num_directions, 32, 20) -> (1, 32, 20) for default
y = lstm_layer(x) # Example without state return
```

## GRU Modules

### GRU

`GRU` implements a Gated Recurrent Unit layer.

```python
from ember_ml.nn.modules.rnn import GRU
from ember_ml.nn import tensor

# Create a GRU layer
gru_layer = GRU(input_size=10, hidden_size=20)

# Forward pass (sequence)
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
# Layer returns outputs by default. Set return_state=True to get final state.
# y, [h] = gru_layer(x, return_state=True)
# If return_sequences=True (default): y shape=(32, 5, 20)
# If return_sequences=False: y shape=(32, 20)
# Final state h shape=(num_layers*num_directions, 32, 20) -> (1, 32, 20) for default
y = gru_layer(x) # Example without state return
```

## Closed-form Continuous-time (CfC) Modules

CfC modules implement closed-form continuous-time recurrent neural networks, which are particularly effective for modeling irregular time series. These modules can integrate with various `NeuronMap` implementations to define their internal connectivity.

### CfC

`CfC` implements a Closed-form Continuous-time layer.

```python
from ember_ml.nn.modules.rnn import CfC
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

# Create a CfC layer, passing the NeuronMap instance
# The neuron_map can be any derivative of NeuronMap, including spatial maps like EnhancedNeuronMap
cfc_layer = CfC(neuron_map=neuron_map)

# Forward pass (sequence)
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
# Layer returns outputs by default. Set return_state=True to get final state.
# y, [h, t] = cfc_layer(x, return_state=True)
# If return_sequences=False (default): y shape=(32, 20)
# If return_sequences=True: y shape=(32, 5, 20)
# Final state h shape=(32, 20), t shape=(32, 20)
y = cfc_layer(x) # Example without state return
```


## Liquid Time-Constant (LTC) Modules

LTC modules implement liquid time-constant recurrent neural networks, which model continuous-time neuronal dynamics. These modules can integrate with various `NeuronMap` implementations to define their internal connectivity.

### LTC

`LTC` is a layer that applies an LTC cell to a sequence of inputs.

```python
from ember_ml.nn.modules.rnn import LTC
from ember_ml.nn import tensor

# Create a NeuronMap (e.g., FullyConnectedMap)
from ember_ml.nn.modules.wiring import FullyConnectedMap
neuron_map = FullyConnectedMap(units=20, input_dim=10, output_dim=20)

# Create an LTC layer by passing the NeuronMap instance
# The neuron_map can be any derivative of NeuronMap, including spatial maps like EnhancedNeuronMap
ltc_layer = LTC(neuron_map=neuron_map)

# Forward pass (sequence)
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
# Layer returns outputs by default. Set return_state=True to get final state.
# y, h = ltc_layer(x, return_state=True)
# If return_sequences=True (default): y shape=(32, 5, 20)
# If return_sequences=False: y shape=(32, 20)
# Final state h shape=(32, 20)
y = ltc_layer(x) # Example without state return
```

## Stride-Aware Modules

Stride-aware modules are specialized for processing temporal data with variable strides, which is particularly useful for multi-scale time series analysis. Note that the architecture of these modules, particularly the use of explicit "Cell" and "Wired" classes, may represent an older pattern compared to the current approach of integrating `NeuronMap` directly into layer implementations.

### StrideAware

Stride-aware modules are specialized for processing temporal data with variable strides, which is particularly useful for multi-scale time series analysis.

### StrideAware

`StrideAware` is the base class for stride-aware modules.

### StrideAwareCfC

`StrideAwareCfC` implements a stride-aware Closed-form Continuous-time network.

```python
from ember_ml.nn.modules.rnn import StrideAwareCfC
from ember_ml.nn import tensor

# Create a NeuronMap (assuming FullyConnected for this example)
# The StrideAwareCfC layer likely expects a NeuronMap or a pre-configured stride-aware cell
from ember_ml.nn.modules.wiring import FullyConnectedMap
neuron_map = FullyConnectedMap(units=20, input_dim=10, output_dim=20)

# Create a StrideAwareCfC layer (Verify exact init signature if different)
stride_cfc = StrideAwareCfC(
    neuron_map=neuron_map,
    stride_lengths=[1, 2, 4]
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
# Layer returns outputs by default. Check implementation for state return details.
# y, h = stride_cfc(x, return_state=True)
y = stride_cfc(x)
```


## Advanced Usage

### Working with Time Deltas

CfC and LTC modules support time deltas between inputs, which is useful for irregular time series:

```python
from ember_ml.nn.modules.rnn import CfC
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

# Create a CfC layer
cfc_layer = CfC(neuron_map=neuron_map)

# Create input sequence and time deltas
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
time_deltas = tensor.random_uniform((32, 5, 1), minval=0.1, maxval=1.0)  # Time deltas between inputs

# Forward pass with time deltas
# Assuming return_sequences=False, return_state=False by default
y = cfc_layer(x, time_deltas=time_deltas)
# Or to get state:
# y, [h, t] = cfc_layer(x, time_deltas=time_deltas, return_state=True)
```

### Multi-Scale Time Series Processing

Stride-aware modules can process time series at multiple scales simultaneously:

```python
from ember_ml.nn.modules.rnn import StrideAwareCfC
from ember_ml.nn import tensor

# Create a NeuronMap first
from ember_ml.nn.modules.wiring import FullyConnectedMap
neuron_map = FullyConnectedMap(units=20, input_dim=10, output_dim=20)
# Create a StrideAwareCfC with multiple stride lengths
stride_cfc = StrideAwareCfC(
    neuron_map=neuron_map,
    stride_lengths=[1, 2, 4, 8],
    backbone_units=32,
    backbone_layers=2
)

# Create a long input sequence
x = tensor.random_normal((32, 100, 10))  # Batch of 32 sequences of length 100 with 10 features each

# Forward pass
# Assuming return_sequences=False, return_state=False by default
y = stride_cfc(x)
# Or to get state:
# y, h = stride_cfc(x, return_state=True) # Check state structure

# Each stride processes the sequence at a different time scale:
# - Stride 1: Processes every time step
# - Stride 2: Processes every other time step
# - Stride 4: Processes every fourth time step
# - Stride 8: Processes every eighth time step
```

### Building a Multi-Layer RNN

You can stack multiple RNN layers to create deep recurrent networks:

```python
from ember_ml.nn.modules import Module
from ember_ml.nn.modules.rnn import LSTM
from ember_ml.nn import tensor

class DeepLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(LSTM(input_size=layer_input_size, hidden_size=hidden_size))
        
    def forward(self, x, initial_states=None):
        batch_size, seq_length, _ = tensor.shape(x)
        
        if initial_states is None:
            initial_states = [(None, None)] * len(self.layers)
        
        outputs = x
        final_states = []
        
        for i, layer in enumerate(self.layers):
            outputs, states = layer(outputs, initial_state=initial_states[i])
            final_states.append(states)
        
        return outputs, final_states

# Create a deep LSTM
deep_lstm = DeepLSTM(input_size=10, hidden_size=20, num_layers=3)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, states = deep_lstm(x)
```

## Implementation Details

The RNN modules are implemented using a simplified architecture:

1. **Layer Classes**: Implement the recurrent neural network logic directly, often integrating a `NeuronMap` to define internal connectivity, including spatial patterns.
2. **Advanced Modules**: Implement specialized recurrent architectures.

This architecture allows Ember ML to provide a consistent API across different backends while still leveraging the unique capabilities of each backend. The cell-based architecture has been removed to simplify the codebase and improve maintainability.

## Performance Considerations

When working with RNN modules, consider the following performance optimizations:

1. **Batch Processing**: Process data in batches to maximize computational efficiency
2. **Sequence Packing**: Pack sequences of variable length to avoid unnecessary computations
3. **Mixed Precision**: Use mixed precision training for faster computation on GPUs
4. **Memory Management**: Use gradient checkpointing for long sequences to reduce memory usage

## Theoretical Background

### CfC (Closed-form Continuous-time)

CfC modules are based on the paper "Closed-form Continuous-time Neural Networks" by Gu et al. They model continuous-time dynamics using a closed-form solution to the differential equations, which allows for efficient training and inference.

### LTC (Liquid Time-Constant)

LTC modules are based on the paper "Liquid Time-Constant Networks" by Hasani et al. They model continuous-time neuronal dynamics using a liquid time-constant, which allows for adaptive time scales and improved modeling of irregular time series.

### Quantum-Inspired Modules

Quantum-inspired modules combine principles from quantum computing with classical neural networks to enhance temporal processing capabilities. These modules also utilize `NeuronMap` instances to define their connectivity.

Quantum-inspired modules combine principles from quantum computing with classical neural networks to enhance temporal processing capabilities.

#### LQNet (Liquid Quantum Neural Network)

`LQNet` implements a quantum-inspired recurrent neural network that combines liquid neural networks with quantum computing concepts using classical hardware.

```python
from ember_ml.nn.modules.rnn import LQNet
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a neuron map for connectivity
neuron_map = NCPMap(
    inter_neurons=32,
    command_neurons=16,
    motor_neurons=8,
    sensory_neurons=10,
    seed=42
)

# Create LQNet model, passing the NeuronMap instance
# The neuron_map can be any derivative of NeuronMap
lqnet = LQNet(
    neuron_map=neuron_map,
    nu_0=1.0,
    beta=0.1,
    noise_scale=0.05,
    return_sequences=True,
    return_state=False,
    batch_first=True
)

# Forward pass
inputs = tensor.random_normal((32, 100, 10))  # (batch_size, seq_length, input_dim)
outputs = lqnet(inputs)
```

#### CTRQNet (Continuous-Time Recurrent Quantum Neural Network)

`CTRQNet` extends LQNet with continuous-time dynamics and enhanced quantum-inspired features.

```python
from ember_ml.nn.modules.rnn import CTRQNet
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a neuron map for connectivity
neuron_map = NCPMap(
    inter_neurons=32,
    command_neurons=16,
    motor_neurons=8,
    sensory_neurons=10,
    seed=42
)

# Create CTRQNet model, passing the NeuronMap instance
# The neuron_map can be any derivative of NeuronMap
ctrqnet = CTRQNet(
    neuron_map=neuron_map,
    nu_0=1.0,
    beta=0.1,
    noise_scale=0.05,
    time_scale_factor=1.0,
    use_harmonic_embedding=True,
    return_sequences=True,
    return_state=False,
    batch_first=True
)

# Forward pass
inputs = tensor.random_normal((32, 100, 10))  # (batch_size, seq_length, input_dim)
outputs = ctrqnet(inputs)
```

For more details on quantum-inspired modules, see [Quantum-Inspired Neural Networks](nn_modules_rnn_quantum.md).

### Stride-Aware Modules

Stride-aware modules extend the CfC and LTC architectures with multi-scale processing capabilities, which allow for efficient modeling of time series at multiple time scales simultaneously.

## References

1. Gu, A., Goel, K., & Réa, C. (2020). Closed-form Continuous-time Neural Networks. [arXiv:2003.09346](https://arxiv.org/abs/2003.09346)
2. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2020). Liquid Time-Constant Networks. [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
4. Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
5. Barandes, J. A., & Kagan, D. (2020). Measurement and Quantum Dynamics in the Minimal Modal Interpretation of Quantum Theory. Foundations of Physics, 50(10), 1189-1218.
6. Markidis, S. (2021). The Old and the New: Can Quantum Computing Become a Reality? ACM Computing Surveys, 54(8), 1-36.

## See Also

- [Neural Network Modules](nn_modules.md): Documentation on base neural network modules
- [Neuron Maps Documentation](nn_modules_wiring.md): Documentation on neuron maps used by RNN modules
- [Tensor Module Documentation](nn_tensor.md): Documentation on tensor operations used by the RNN modules
- [Quantum-Inspired Neural Networks](nn_modules_rnn_quantum.md): Documentation on quantum-inspired neural network modules