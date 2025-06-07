# Neuron Maps API (nn.modules.wiring)

The `ember_ml.nn.modules.wiring` package provides a comprehensive set of neuron map implementations for defining custom connectivity patterns in neural networks, including those with spatial properties. These were formerly called "wirings" and have been refactored and enhanced as "neuron maps" to better reflect their role in defining the connectivity structure of neural networks.

## Importing

```python
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap, EnhancedNeuronMap, EnhancedNCPMap
```

## Core Concepts

Neuron maps define how neurons in a neural network are connected to each other. They specify:

1. **Connectivity Patterns**: Which neurons are connected to which other neurons, potentially influenced by spatial relationships.
2. **Connection Weights**: The initial weights of connections between neurons.
3. **Neuron Types**: Different functional roles for neurons (sensory, motor, inter, command), and potentially neuron-specific dynamics.
4. **Input/Output Mapping**: How external inputs map to internal neurons and how internal neurons map to outputs.
5. **Spatial Properties**: Optional spatial coordinates and metrics that can influence connectivity and dynamics.

## Base Class

### NeuronMap

`NeuronMap` is the base class for all neuron maps. It defines the core API and functionality.

```python
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn import tensor

class CustomNeuronMap(NeuronMap):
    def __init__(self, units, output_size, input_size=None):
        super().__init__(units, output_size, input_size)
        # Initialize custom connectivity pattern
        self.build_adjacency_matrix()
        self.build_sensory_adjacency_matrix()
        
    def build_adjacency_matrix(self):
        # Define connectivity between internal neurons
        self.adjacency_matrix = tensor.eye(self.units)
        
    def build_sensory_adjacency_matrix(self):
        # Define connectivity from inputs to internal neurons
        if self.input_size is not None:
            self.sensory_adjacency_matrix = tensor.ones((self.input_size, self.units))
```

## Neuron Map Implementations

### NCPMap

`NCPMap` implements a Neural Circuit Policy connectivity pattern, which is inspired by biological neural circuits.

```python
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create an NCP neuron map
neuron_map = NCPMap(
    inter_neurons=10,  # Number of interneurons
    command_neurons=5,  # Number of command neurons
    motor_neurons=3,    # Number of motor neurons
    sensory_neurons=8,  # Number of sensory neurons (input size)
    seed=42             # Random seed for reproducibility
)

# Access properties
print(f"Units: {neuron_map.units}")                # 18 (10 + 5 + 3)
print(f"Output size: {neuron_map.output_size}")    # 3 (motor_neurons)
print(f"Input size: {neuron_map.input_size}")      # 8 (sensory_neurons)
print(f"Sparsity level: {neuron_map.sparsity_level}")  # Default: 0.5
```

#### Neural Circuit Policy Architecture

The NCPMap divides neurons into three functional categories:

1. **Interneurons**: Hidden neurons that process information within the network
2. **Command Neurons**: Neurons that influence the behavior of motor neurons
3. **Motor Neurons**: Output neurons that produce the final network output

Additionally, it defines:

4. **Sensory Neurons**: Input neurons that receive external signals (not counted in total units)

The connectivity follows specific patterns:
- Sensory neurons connect to interneurons and command neurons
- Interneurons connect to other interneurons, command neurons, and motor neurons
- Command neurons connect to motor neurons
- Motor neurons are output neurons and don't connect to other neurons

### FullyConnectedMap

`FullyConnectedMap` implements a fully connected connectivity pattern, where every neuron is connected to every other neuron.

```python
from ember_ml.nn.modules.wiring import FullyConnectedMap
from ember_ml.nn import tensor

# Create a fully connected neuron map
neuron_map = FullyConnectedMap(
    units=10,         # Number of internal neurons
    output_size=5,    # Number of output neurons
    input_size=8      # Number of input neurons
)

# Access properties
print(f"Units: {neuron_map.units}")                # 10
print(f"Output size: {neuron_map.output_size}")    # 5
print(f"Input size: {neuron_map.input_size}")      # 8
```

### RandomMap

`RandomMap` implements a random connectivity pattern, where connections between neurons are established randomly based on a sparsity level.

```python
from ember_ml.nn.modules.wiring import RandomMap
from ember_ml.nn import tensor

# Create a random neuron map
neuron_map = RandomMap(
    units=10,              # Number of internal neurons
    output_size=5,         # Number of output neurons
    input_size=8,          # Number of input neurons
    sparsity_level=0.5,    # Sparsity level (0.0 = dense, 1.0 = no connections)
    seed=42                # Random seed for reproducibility
)

# Access properties
print(f"Units: {neuron_map.units}")                # 10
print(f"Output size: {neuron_map.output_size}")    # 5
print(f"Input size: {neuron_map.input_size}")      # 8
print(f"Sparsity level: {neuron_map.sparsity_level}")  # 0.5
```

### EnhancedNeuronMap

`EnhancedNeuronMap` extends `NeuronMap` to incorporate neuron-type specific parameters, dynamic properties, and spatial embedding.

```python
from ember_ml.nn.modules.wiring import EnhancedNeuronMap
import numpy as np

# Create an enhanced neuron map with spatial properties
neuron_map = EnhancedNeuronMap(
    units=20,
    output_dim=5,
    input_dim=10,
    neuron_type="cfc", # Specify neuron type
    neuron_params={"time_scale_factor": 0.5}, # Neuron-specific parameters
    network_structure=(4, 5, 1), # 3D structure for spatial embedding
    distance_metric="euclidean",
    distance_power=1.0,
    seed=42
)

# Access properties
print(f"Units: {neuron_map.units}")                # 20
print(f"Output size: {neuron_map.output_dim}")    # 5
print(f"Input size: {neuron_map.input_dim}")      # 10
print(f"Neuron type: {neuron_map.neuron_type}")    # cfc
print(f"Distance matrix shape: {neuron_map.distance_matrix.shape}") # (20, 20)
```

#### Spatial Properties

`EnhancedNeuronMap` allows defining neurons in a spatial arrangement, which can influence connectivity.

-   **Coordinates**: Neurons can be assigned spatial coordinates, either explicitly via `coordinates_list` or generated from a `network_structure`.
-   **Distance Matrix**: A distance matrix is computed based on the neuron coordinates and a specified `distance_metric` and `distance_power`.
-   **Communicability Matrix**: A communicability matrix is calculated during the `build` process, reflecting the ease of information flow between neurons based on connectivity and potentially spatial distance.

#### Dynamic Properties

This map can also store parameters specific to the type of neuron being used (e.g., time constants for CfC neurons), allowing the map to define both the structure and some aspects of the neuron dynamics.

```python
# Access dynamic properties
dynamic_props = neuron_map.get_dynamic_properties()
print(f"Dynamic properties: {dynamic_props}")
```

#### Spatial Properties Access

```python
# Access spatial properties
spatial_props = neuron_map.get_spatial_properties()
print(f"Spatial properties keys: {spatial_props.keys()}")
```

### EnhancedNCPMap

`EnhancedNCPMap` extends `EnhancedNeuronMap` to implement the Neural Circuit Policy (NCP) connectivity pattern with support for spatial properties and neuron-type specific dynamics. It combines the structured grouping of NCPMap with the spatial and dynamic features of EnhancedNeuronMap.

```python
from ember_ml.nn.modules.wiring import EnhancedNCPMap

# Create an enhanced NCP map with spatial properties
neuron_map = EnhancedNCPMap(
    sensory_neurons=8,
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    neuron_type="ltc", # Specify neuron type
    time_scale_factor=0.8, # Neuron-specific parameter
    network_structure=(4, 5, 1), # 3D structure for spatial embedding
    seed=42
)

# Access properties
print(f"Units: {neuron_map.units}")                # 26 (8 + 10 + 5 + 3)
print(f"Output size: {neuron_map.output_dim}")    # 3
print(f"Input size: {neuron_map.input_dim}")      # 8
print(f"Neuron type: {neuron_map.neuron_type}")    # ltc
print(f"Distance matrix shape: {neuron_map.distance_matrix.shape}") # (26, 26)
```

#### Integration of Spatial and NCP Properties

`EnhancedNCPMap` allows defining the classic NCP neuron groups (sensory, inter, command, motor) while also embedding these neurons in a spatial layout. Connectivity can then be influenced by both the NCP group structure and the spatial distances between neurons.

#### Dynamic Properties

Similar to `EnhancedNeuronMap`, this class supports defining neuron-type specific parameters, which are then used by layers that utilize this map.

## Key Properties and Methods

### Properties

| Property | Description |
|----------|-------------|
| `units` | Number of internal neurons |
| `output_size` | Number of output neurons |
| `input_size` | Number of input neurons |
| `sparsity_level` | Sparsity level of the connectivity pattern |
| `adjacency_matrix` | Connectivity matrix between internal neurons |
| `sensory_adjacency_matrix` | Connectivity matrix from inputs to internal neurons |
| `synapse_count` | Number of synapses between internal neurons |
| `sensory_synapse_count` | Number of synapses from inputs to internal neurons |

### Methods

| Method | Description |
|--------|-------------|
| `build_adjacency_matrix()` | Build the connectivity matrix between internal neurons |
| `build_sensory_adjacency_matrix()` | Build the connectivity matrix from inputs to internal neurons |
| `get_config()` | Get the configuration of the neuron map |
| `from_config(config)` | Create a neuron map from a configuration |

## Advanced Usage

### Creating Custom Neuron Maps

You can create custom neuron maps by extending the `NeuronMap` base class:

```python
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn import tensor
from ember_ml import ops

class SmallWorldNeuronMap(NeuronMap):
    def __init__(self, units, output_size, input_size=None, k=2, beta=0.1, seed=None):
        super().__init__(units, output_size, input_size)
        self.k = k          # Number of nearest neighbors
        self.beta = beta    # Rewiring probability
        self.seed = seed
        
        # Set the random seed
        if seed is not None:
            tensor.set_seed(seed)
        
        # Build the connectivity matrices
        self.build_adjacency_matrix()
        self.build_sensory_adjacency_matrix()
    
    def build_adjacency_matrix(self):
        # Create a ring lattice with k nearest neighbors
        adjacency = tensor.zeros((self.units, self.units))
        
        for i in range(self.units):
            for j in range(1, self.k + 1):
                # Connect to k nearest neighbors in both directions
                adjacency[i, (i + j) % self.units] = 1
                adjacency[i, (i - j) % self.units] = 1
        
        # Rewire connections with probability beta
        for i in range(self.units):
            for j in range(self.units):
                if adjacency[i, j] == 1 and i != j:
                    if tensor.random_uniform(()) < self.beta:
                        # Remove this connection
                        adjacency[i, j] = 0
                        
                        # Add a connection to a random neuron
                        possible_targets = []
                        for k in range(self.units):
                            if k != i and adjacency[i, k] == 0:
                                possible_targets.append(k)
                        
                        if len(possible_targets) > 0:
                            target = possible_targets[tensor.random_uniform((), maxval=len(possible_targets), dtype=tensor.int32)]
                            adjacency[i, target] = 1
        
        self.adjacency_matrix = adjacency
    
    def build_sensory_adjacency_matrix(self):
        if self.input_size is not None:
            # Connect each input to a random subset of neurons
            self.sensory_adjacency_matrix = tensor.random_bernoulli(
                (self.input_size, self.units),
                p=0.3,
                seed=self.seed
            )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'k': self.k,
            'beta': self.beta,
            'seed': self.seed
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

### Using Neuron Maps with RNN Layers

Neuron maps are primarily used with RNN layers that support wired connectivity:

```python
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import CfC
from ember_ml.nn import tensor

# Create a neuron map
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)

# Create a CfC layer with the neuron map
cfc_layer = CfC(
    neuron_map=neuron_map,
    return_sequences=True
)

# Forward pass
x = tensor.random_normal((32, 10, 8))  # Batch of 32 sequences of length 10 with 8 features
output = cfc_layer(x)  # Shape: (32, 10, 3) if return_sequences=True
```

### Visualizing Neuron Maps

You can visualize neuron maps to better understand the connectivity patterns:

```python
import matplotlib.pyplot as plt
import numpy as np
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a neuron map
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)

# Convert adjacency matrices to numpy arrays
adjacency = tensor.to_numpy(neuron_map.adjacency_matrix)
sensory_adjacency = tensor.to_numpy(neuron_map.sensory_adjacency_matrix)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the internal adjacency matrix
im1 = ax1.imshow(adjacency, cmap='viridis')
ax1.set_title('Internal Connectivity')
ax1.set_xlabel('Target Neuron')
ax1.set_ylabel('Source Neuron')
fig.colorbar(im1, ax=ax1)

# Plot the sensory adjacency matrix
im2 = ax2.imshow(sensory_adjacency, cmap='viridis')
ax2.set_title('Sensory Connectivity')
ax2.set_xlabel('Target Neuron')
ax2.set_ylabel('Source Sensory Neuron')
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
```

## Implementation Details

The neuron map module is implemented using a layered architecture:

1. **Base Class**: Provides common functionality for all neuron maps
2. **Specific Implementations**: Implement different connectivity patterns
3. **Integration with RNN Layers**: Neuron maps are used by RNN layers to define connectivity

This architecture allows Ember ML to provide a consistent API for defining custom connectivity patterns in neural networks.

## Migration from Old Wiring API to Neuron Maps API

The neuron map module was previously called "wirings" and has been refactored and enhanced. If you have code that uses the old wirings API, you can migrate to the new neuron maps API as follows:

### Old API (Wiring)

```python
from ember_ml.nn.wirings import NCPWiring
from ember_ml.nn.modules.rnn import CfC

# Create a wiring
wiring = NCPWiring(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)

# Create a CfC layer with the wiring
cfc_layer = CfC(
    wiring=wiring,
    return_sequences=True
)
```

### New API (Neuron Maps)

```python
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import CfC

# Create a neuron map
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)

# Create a CfC layer with the neuron map
cfc_layer = CfC(
    neuron_map=neuron_map,
    return_sequences=True
)
```

The key changes are:
1. Import from `ember_ml.nn.modules.wiring` instead of `ember_ml.nn.wirings`
2. Use `NCPMap` instead of `NCPWiring`
3. Use `neuron_map` parameter instead of `wiring` parameter

## References

1. Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., & Grosu, R. (2020). Neural Circuit Policies Enabling Auditable Autonomy. Nature Machine Intelligence, 2(10), 642-652.
2. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442.
3. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509-512.

## See Also

- [Neural Network Modules](nn_modules.md): Documentation on base neural network modules
- [RNN Modules Documentation](nn_modules_rnn.md): Documentation on recurrent neural network modules that use neuron maps
- [Tensor Module Documentation](nn_tensor.md): Documentation on tensor operations used by neuron maps