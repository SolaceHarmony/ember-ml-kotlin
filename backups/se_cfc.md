# Spatially Embedded Closed-form Continuous-time (seCfC) Framework

The seCfC framework integrates spatial embedding with continuous-time dynamics, creating neural networks that are both structurally and functionally constrained. This document provides an overview of the framework and how to use it.

## Overview

The seCfC framework combines three key elements:

1. **Spatial structure**: Neurons are embedded in a 3D space, with connectivity constrained by physical distance.
2. **Continuous-time dynamics**: Neurons operate in continuous time, with state updates governed by differential equations.
3. **Neural circuit organization**: Neurons are organized into functional groups (sensory, inter, command, motor) with specific connectivity patterns.

This integration creates neural networks that exhibit properties similar to biological neural circuits, including:

- Low entropy modularity
- Distance-dependent connectivity
- Regular communication topology
- Heterogeneous spectral dynamics

## Components

The seCfC framework consists of three main components:

### 1. EnhancedNeuronMap

The `EnhancedNeuronMap` class extends the basic `NeuronMap` with support for:

- Arbitrary neuron types and dynamics
- Spatial embedding in 3D space
- Dynamic properties that affect temporal processing

```python
from ember_ml.nn.modules.wiring import EnhancedNeuronMap

neuron_map = EnhancedNeuronMap(
    units=100,
    output_dim=10,
    input_dim=5,
    neuron_type="cfc",
    neuron_params={
        "time_scale_factor": 1.0,
        "activation": "tanh",
        "recurrent_activation": "sigmoid"
    },
    network_structure=(5, 5, 4),
    distance_metric="euclidean",
    distance_power=1.0
)
```

### 2. EnhancedNCPMap

The `EnhancedNCPMap` class implements a Neural Circuit Policy connectivity pattern with support for arbitrary neuron types and dynamics:

```python
from ember_ml.nn.modules.wiring import EnhancedNCPMap

neuron_map = EnhancedNCPMap(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=4,
    sensory_neurons=5,
    neuron_type="cfc",
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    sparsity_level=0.5
)
```

### 3. seCfC

The `seCfC` class implements a spatially embedded CfC neural network that integrates spatial constraints with continuous-time dynamics:

```python
from ember_ml.nn.modules.rnn import seCfC

model = seCfC(
    neuron_map=neuron_map,
    return_sequences=True,
    return_state=False,
    go_backwards=False,
    regularization_strength=0.01
)
```

## Usage

### Basic Usage

```python
import numpy as np
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import EnhancedNCPMap
from ember_ml.nn.modules.rnn import seCfC

# Create a neuron map
neuron_map = EnhancedNCPMap(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=4,
    sensory_neurons=5,
    neuron_type="cfc",
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    sparsity_level=0.5
)

# Create a seCfC model
model = seCfC(
    neuron_map=neuron_map,
    return_sequences=True,
    return_state=False,
    go_backwards=False,
    regularization_strength=0.01
)

# Generate some data
x = tensor.random_normal((32, 10, 5))  # (batch_size, time_steps, features)
y = tensor.random_normal((32, 10, 4))  # (batch_size, time_steps, output_dim)

# Train the model
optimizer = ops.optimizers.Adam(learning_rate=0.01)

for epoch in range(10):
    with ops.GradientTape() as tape:
        # Forward pass
        y_pred = model(x)
        
        # Calculate loss
        mse_loss = ops.stats.mean(ops.square(y_pred - y))
        reg_loss = model.get_regularization_loss()
        total_loss = mse_loss + reg_loss
    
    # Backward pass
    gradients = tape.gradient(total_loss, model.parameters())
    optimizer.apply_gradients(zip(gradients, model.parameters()))
    
    print(f"Epoch {epoch+1}, Loss: {total_loss.numpy():.4f}")
```

### Advanced Usage

For more advanced usage, see the example script at `examples/se_cfc_example.py`.

## Customization

### Custom Neuron Types

You can use different neuron types by specifying the `neuron_type` parameter:

```python
neuron_map = EnhancedNCPMap(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=4,
    sensory_neurons=5,
    neuron_type="ltc",  # Use LTC neurons instead of CfC
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    sparsity_level=0.5
)
```

### Custom Spatial Embedding

You can provide custom coordinates for neurons:

```python
import numpy as np

# Generate random coordinates in 3D space
coordinates = [
    np.random.uniform(0, 1, 100),  # x coordinates
    np.random.uniform(0, 1, 100),  # y coordinates
    np.random.uniform(0, 1, 100)   # z coordinates
]

neuron_map = EnhancedNeuronMap(
    units=100,
    output_dim=10,
    input_dim=5,
    coordinates_list=coordinates
)
```

### Custom Connectivity Patterns

You can customize the connectivity patterns between neuron groups:

```python
neuron_map = EnhancedNCPMap(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=4,
    sensory_neurons=5,
    sensory_to_inter_sparsity=0.3,  # Sparse connections from sensory to inter
    inter_to_inter_sparsity=0.7,    # Dense connections within inter neurons
    inter_to_motor_sparsity=0.5,    # Medium connections from inter to motor
    motor_to_motor_sparsity=0.9     # Very dense connections within motor neurons
)
```

## Theoretical Background

The seCfC framework is based on several theoretical frameworks:

1. **Hamiltonian Cognitive Dynamics (HCD)**: The continuous-time dynamics of seCfC neurons follow trajectories through a phase space, similar to physical systems governed by Hamiltonian mechanics.

2. **Wave-Based Cognition (GUCE)**: Information propagates through the network like waves, with interference patterns encoding and processing information.

3. **Multi-Scale Processing (FHE)**: The network operates at multiple spatial and temporal scales, with different neuron groups and time constants.

Recent research has shown that spatial embedding promotes a specific form of modularity with low entropy and heterogeneous spectral dynamics. The seCfC framework leverages these findings to create neural networks that are both structurally and functionally constrained, leading to more interpretable and biologically plausible models.

## References

1. Sheeran, C., Ham, A. S., Astle, D. E., Achterberg, J., & Akarca, D. (2025). Spatial embedding promotes a specific form of modularity with low entropy and heterogeneous spectral dynamics.

2. Achterberg, J., Akarca, D., Strouse, D. J., Duncan, J., & Astle, D. E. (2023). Spatially embedded recurrent neural networks reveal widespread links between structural and functional neuroscience findings. Nature Machine Intelligence, 5, 1369-1381.

3. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2020). Liquid Time-constant Networks. arXiv preprint arXiv:2006.04439.