# Recurrent Neural Network (RNN) Module

This module provides implementations of various RNN layers for the ember_ml framework, including:

- **CfC (Closed-form Continuous-time)**: A continuous-time RNN with closed-form solution for the hidden state dynamics
- **WiredCfC**: A CfC cell with custom wiring (e.g., Neural Circuit Policies)
- **LTC (Liquid Time-Constant)**: A biologically-inspired continuous-time RNN with liquid neural dynamics
- **LTCCell**: The cell implementation of the LTC layer
- **LSTM (Long Short-Term Memory)**: A standard LSTM implementation with support for multiple layers and bidirectionality
- **LSTMCell**: The cell implementation of the LSTM layer
- **GRU (Gated Recurrent Unit)**: A standard GRU implementation with support for multiple layers and bidirectionality
- **GRUCell**: The cell implementation of the GRU layer
- **RNN (Recurrent Neural Network)**: A basic RNN implementation with support for multiple layers and bidirectionality
- **RNNCell**: The cell implementation of the RNN layer

## Overview

The RNN module is designed to be backend-agnostic, using ember_ml's ops module for all tensor operations. This allows the same code to run on different backends (PyTorch, NumPy, MLX) without modification.

## Components

### CfCCell

The `CfCCell` is the basic building block for Closed-form Continuous-time RNNs. It implements a continuous-time recurrent neural network with closed-form solution for the hidden state dynamics.

```python
from ember_ml.nn.modules.rnn import CfCCell

# Create a CfC cell with 32 units
cell = CfCCell(
    units=32,
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    mixed_memory=True
)
```

### WiredCfCCell

The `WiredCfCCell` extends the `CfCCell` with support for custom wiring, such as Neural Circuit Policies (NCPs).

```python
from ember_ml.nn.modules.rnn import WiredCfCCell
from ember_ml.nn.wirings import AutoNCP

# Create an AutoNCP wiring
wiring = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5
)

# Create a wired CfC cell
cell = WiredCfCCell(
    wiring=wiring,
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    mixed_memory=True
)
```

### CfC

The `CfC` layer wraps a `CfCCell` or `WiredCfCCell` to create a recurrent layer that can process sequences.

```python
from ember_ml.nn.modules.rnn import CfC
from ember_ml.nn.wirings import AutoNCP

# Create a CfC layer with 32 units
cfc_layer = CfC(
    units=32,
    return_sequences=True,
    mixed_memory=True
)

# Create a CfC layer with custom wiring
wiring = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5
)

wired_cfc_layer = CfC(
    wiring,
    return_sequences=True,
    mixed_memory=True
)
```

### LTCCell

The `LTCCell` implements a Liquid Time-Constant cell with biologically-inspired dynamics. It requires a wiring configuration to define the connectivity between neurons.

```python
from ember_ml.nn.modules.rnn import LTCCell
from ember_ml.nn.wirings import AutoNCP

# Create an AutoNCP wiring
wiring = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5
)

# Create an LTC cell
cell = LTCCell(
    wiring=wiring,
    input_mapping="affine",
    output_mapping="affine",
    ode_unfolds=6,
    implicit_param_constraints=True
)
```

### LTC

The `LTC` layer wraps an `LTCCell` to create a recurrent layer that can process sequences.

```python
from ember_ml.nn.modules.rnn import LTC
from ember_ml.nn.wirings import AutoNCP

# Create an LTC layer with fully connected wiring
ltc_layer = LTC(
    input_size=20,
    units=32,
    return_sequences=True,
    mixed_memory=True
)

# Create an LTC layer with custom wiring
wiring = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5
)

wired_ltc_layer = LTC(
    input_size=20,
    units=wiring,
    return_sequences=True,
    mixed_memory=True
)
```

### LSTMCell

The `LSTMCell` implements a standard Long Short-Term Memory cell with input, forget, and output gates.

```python
from ember_ml.nn.modules.rnn import LSTMCell

# Create an LSTM cell
cell = LSTMCell(
    input_size=20,
    hidden_size=32,
    bias=True
)
```

### LSTM

The `LSTM` layer wraps one or more `LSTMCell` instances to create a recurrent layer that can process sequences. It supports multiple layers, bidirectionality, and dropout.

```python
from ember_ml.nn.modules.rnn import LSTM

# Create a single-layer LSTM
lstm_layer = LSTM(
    input_size=20,
    hidden_size=32,
    num_layers=1,
    return_sequences=True
)

# Create a multi-layer bidirectional LSTM
bidirectional_lstm = LSTM(
    input_size=20,
    hidden_size=32,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,
    return_sequences=True,
    return_state=True
)
```

### GRUCell

The `GRUCell` implements a standard Gated Recurrent Unit cell with reset and update gates.

```python
from ember_ml.nn.modules.rnn import GRUCell

# Create a GRU cell
cell = GRUCell(
    input_size=20,
    hidden_size=32,
    bias=True
)
```

### GRU

The `GRU` layer wraps one or more `GRUCell` instances to create a recurrent layer that can process sequences. It supports multiple layers, bidirectionality, and dropout.

```python
from ember_ml.nn.modules.rnn import GRU

# Create a single-layer GRU
gru_layer = GRU(
    input_size=20,
    hidden_size=32,
    num_layers=1,
    return_sequences=True
)

# Create a multi-layer bidirectional GRU
bidirectional_gru = GRU(
    input_size=20,
    hidden_size=32,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,
    return_sequences=True,
    return_state=True
)
```

### RNNCell

The `RNNCell` implements a basic recurrent neural network cell with a single activation function.

```python
from ember_ml.nn.modules.rnn import RNNCell

# Create an RNN cell with tanh activation
cell = RNNCell(
    input_size=20,
    hidden_size=32,
    activation="tanh",
    bias=True
)

# Create an RNN cell with ReLU activation
cell = RNNCell(
    input_size=20,
    hidden_size=32,
    activation="relu",
    bias=True
)
```

### RNN

The `RNN` layer wraps one or more `RNNCell` instances to create a recurrent layer that can process sequences. It supports multiple layers, bidirectionality, and dropout.

```python
from ember_ml.nn.modules.rnn import RNN

# Create a single-layer RNN with tanh activation
rnn_layer = RNN(
    input_size=20,
    hidden_size=32,
    num_layers=1,
    activation="tanh",
    return_sequences=True
)

# Create a multi-layer bidirectional RNN with ReLU activation
bidirectional_rnn = RNN(
    input_size=20,
    hidden_size=32,
    num_layers=2,
    activation="relu",
    dropout=0.2,
    bidirectional=True,
    return_sequences=True,
    return_state=True
)
```

## Usage

### Basic Usage

```python
import numpy as np
from ember_ml import ops
from ember_ml.nn.modules.rnn import CfC, LTC, LSTM, GRU, RNN
from ember_ml.nn import Sequential

# Create a CfC model
cfc_model = Sequential([
    CfC(
        units=32,
        return_sequences=True,
        mixed_memory=True
    )
])

# Create an LTC model
ltc_model = Sequential([
    LTC(
        input_size=5,
        units=32,
        return_sequences=True,
        mixed_memory=True
    )
])

# Create an LSTM model
lstm_model = Sequential([
    LSTM(
        input_size=5,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        return_sequences=True
    )
])

# Create a GRU model
gru_model = Sequential([
    GRU(
        input_size=5,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        return_sequences=True
    )
])

# Create an RNN model
rnn_model = Sequential([
    RNN(
        input_size=5,
        hidden_size=32,
        num_layers=2,
        activation="tanh",
        dropout=0.2,
        bidirectional=True,
        return_sequences=True
    )
])

# Generate some data
x = tensor.random_normal((32, 10, 5))  # (batch_size, sequence_length, features)

# Forward pass
y_cfc = cfc_model(x)
y_ltc = ltc_model(x)
y_lstm = lstm_model(x)
y_gru = gru_model(x)
y_rnn = rnn_model(x)
```

### Training

```python
from ember_ml.training import Optimizer, Loss

# Define optimizer and loss function
optimizer = Optimizer.adam(model.parameters(), learning_rate=0.001)
loss_fn = Loss.mse()

# Training loop
for epoch in range(epochs):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    
    # Compute loss
    loss = loss_fn(outputs, targets)
    
    # Backward pass and optimize
    grads = ops.gradients(loss, model.parameters())
    optimizer.step(grads)
```

## Examples

See the `examples/cfc_example.py`, `examples/ltc_example.py`, `examples/lstm_example.py`, `examples/gru_example.py`, and `examples/rnn_example.py` files for complete examples of using these RNN networks for time series prediction.

## References

- [Neural Circuit Policies Enabling Auditable Autonomy](https://www.nature.com/articles/s42256-020-00237-3) by Lechner et al.
- [Liquid Time-constant Networks](https://ojs.aaai.org/index.php/AAAI/article/view/16936) by Hasani et al.
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) by Hochreiter and Schmidhuber
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) by Cho et al. (GRU)
- [NCPS: Neural Circuit Policy Search](https://github.com/mlech26l/ncps) - Original implementation