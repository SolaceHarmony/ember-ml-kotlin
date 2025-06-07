# Wiring and Neuron Maps

This section documents the components within Ember ML responsible for defining connectivity patterns within neural circuits, often referred to as "wiring" or "neuron maps". These patterns dictate how neurons are connected to each other and to inputs/outputs.

## Core Concepts

Wiring configurations are fundamental to Neural Circuit Policies (NCPs) and other structured neural architectures. They define the network topology, including:

*   **Neuron Groups:** Dividing neurons into functional groups (e.g., sensory, inter, command, motor).
*   **Connectivity:** Specifying the connections between individual neurons or groups of neurons.
*   **Sparsity:** Controlling the density of connections.
*   **Spatial Embedding:** Incorporating spatial relationships between neurons to influence connectivity (in enhanced variants).
*   **Dynamic Properties:** Including parameters that affect the dynamics of the neurons within the wiring.

## Components

### Base Neuron Map (`ember_ml.nn.modules.wiring.neuron_map`)

*   **`NeuronMap`**: The abstract base class for all wiring configurations.
    *   `__init__(units, output_dim, input_dim, sparsity_level, seed)`: Initializes basic properties like the number of units, input/output dimensions, sparsity, and random seed. Initializes adjacency matrices and masks (as None).
    *   `build(input_dim)`: Abstract method that must be overridden by subclasses to implement the specific wiring pattern and create the input, recurrent, and output masks.
    *   `set_input_dim(input_dim)`: Sets the input dimension and initializes the sensory adjacency matrix.
    *   `is_built()`: Checks if the neuron map has been built.
    *   `get_input_mask()`, `get_recurrent_mask()`, `get_output_mask()`: Methods to retrieve the generated masks (calls `build` if not already built).
    *   `get_config()`: Returns the basic configuration.
    *   `erev_initializer()`, `sensory_erev_initializer()`: Initializers for reversal potentials based on adjacency matrices.
    *   `synapse_count`, `sensory_synapse_count`: Properties to count the number of synapses.
    *   `add_synapse(src, dest, polarity)`: Adds a synapse between two neurons in the internal adjacency matrix.
    *   `add_sensory_synapse(src, dest, polarity)`: Adds a sensory synapse between an input and a neuron in the sensory adjacency matrix.
    *   `from_config(config)`: Class method to create a `NeuronMap` instance from configuration.

### NCP Map (`ember_ml.nn.modules.wiring.ncp_map`)

*   **`NCPMap(NeuronMap)`**: Implements the standard Neural Circuit Policy wiring configuration with distinct sensory, inter, command, and motor neuron groups. Includes cell-specific parameters.
    *   `__init__(inter_neurons, command_neurons, motor_neurons, sensory_neurons, sparsity_level, seed, time_scale_factor, activation, recurrent_activation, mode, use_bias, kernel_initializer, recurrent_initializer, bias_initializer, mixed_memory, ode_unfolds, epsilon, implicit_param_constraints, input_mapping, output_mapping, sensory_to_inter_sparsity, ..., motor_to_inter_sparsity)`: Initializes neuron counts, default and custom sparsity levels, and cell-specific parameters.
    *   `build(input_dim)`: Builds the NCP connectivity pattern based on neuron group counts and sparsity levels. Creates input, recurrent, and output masks. Uses helper methods (`_build_sensory_connections`, `_build_inter_connections`, `_build_command_connections`, `_build_motor_connections`) and an internal `create_random_connections` helper.
    *   `get_neuron_groups()`: Returns a dictionary mapping group names to lists of neuron indices.
    *   `get_config()`: Returns the configuration including structural and cell-specific parameters.
    *   `from_config(config)`: Class method to create an `NCPMap` instance from configuration.

### Enhanced Neuron Map (`ember_ml.nn.modules.wiring.enhanced_neuron_map`)

*   **`EnhancedNeuronMap(NeuronMap)`**: Extends `NeuronMap` with support for arbitrary neuron types, dynamic properties, and spatial embedding.
    *   `__init__(..., neuron_type, neuron_params, coordinates_list, network_structure, distance_metric, distance_power)`: Initializes base properties and adds parameters for neuron type, neuron-specific parameters, spatial coordinates, network structure (if coordinates not provided), distance metric, and distance power.
    *   `_initialize_spatial_properties(...)`: Initializes spatial properties, calculating the distance matrix (using SciPy temporarily) and initializing the communicability matrix.
    *   `build(input_dim)`: Abstract method (must be implemented by subclasses).
    *   `get_neuron_factory()`: Returns a factory function for creating neurons of the specified `neuron_type` (currently supports "cfc" and "ltc").
    *   `get_dynamic_properties()`: Returns a dictionary of dynamic properties (neuron type and parameters).
    *   `get_spatial_properties()`: Returns a dictionary of spatial properties (coordinates, distance matrix, communicability matrix).
    *   `_get_network_structure()`: Helper to infer network structure from coordinates.
    *   `from_config(config)`: Class method to create an `EnhancedNeuronMap` instance from configuration.

### Enhanced NCP Map (`ember_ml.nn.modules.wiring.enhanced_ncp_map`)

*   **`EnhancedNCPMap(NCPMap)`**: Implements an enhanced NCP map inheriting from `EnhancedNeuronMap`.
    *   `__init__(..., neuron_type, time_scale_factor, activation, recurrent_activation, mode, ...)`: Initializes base NCPMap parameters and adds enhanced properties from `EnhancedNeuronMap` and cell-specific parameters.
    *   `build(input_dim)`: Builds the NCP connectivity pattern, incorporating sparsity and spatial constraints using an internal `create_random_connections` helper. Updates the communicability matrix based on the built recurrent mask (using SciPy temporarily for matrix exponential).
    *   `get_neuron_groups()`: Returns a dictionary mapping group names to lists of neuron indices.
    *   `get_config()`: Returns the configuration including all inherited and specific parameters.
    *   `from_config(config)`: Class method to create an `EnhancedNCPMap` instance from configuration.

### Automatic GUCE NCP Wiring (`ember_ml.nn.modules.wiring.guce_ncp`)

*   **`GUCENCP(NeuronMap)`**: Implements a NeuronMap with a structured connectivity pattern based on NCP architecture, intended for use with GUCE neurons.
    *   `__init__(inter_neurons, command_neurons, motor_neurons, sensory_neurons, sensory_fanout, ..., dt)`: Initializes neuron counts, fanout/fanin parameters, and GUCE-specific parameters (state\_dim, step\_size, nu\_0, beta, theta\_freq, gamma\_freq, dt). Creates GUCE neuron instances.
    *   `build(input_dim)`: Builds the connectivity pattern based on neuron group counts and fanout/fanin parameters using helper methods (`_build_sensory_connections`, `_build_inter_connections`, `_build_command_connections`, `_build_motor_connections`) and an internal `create_random_connections` helper.
    *   `forward(inputs)`: Processes inputs through the GUCE neurons and returns sensory outputs (simplified implementation).
    *   `get_config()`: Returns the configuration.
    *   `from_config(config)`: Class method to create a `GUCENCP` instance from configuration.
    *   `get_neuron_groups()`: Returns a dictionary mapping group names to lists of neuron indices.
*   **`AutoGUCENCP(GUCENCP)`**: Automatic GUCE NCP wiring that determines neuron counts based on total units and output size.
    *   `__init__(units, output_size, sparsity_level, state_dim, ...)`: Calculates neuron counts (sensory, inter, command, motor) and connectivity parameters based on total `units`, `output_size`, and `sparsity_level`. Calls the base `GUCENCP` constructor with these calculated values.
    *   `get_config()`: Returns the configuration including calculated and passed parameters.
    *   `from_config(config)`: Class method to create an `AutoGUCENCP` instance from configuration.

### Language Wiring (`ember_ml.nn.modules.wiring.language_wiring`)

*   **`LanguageWiring(NeuronMap)`**: Specialized wiring for language processing tasks, implementing a structure similar to transformer architectures (token embeddings, multi-head attention, position-wise processing).
    *   `__init__(hidden_size, num_heads, vocab_size, max_seq_length, **kwargs)`: Initializes dimensions for hidden size, attention heads, vocabulary size, and max sequence length. Calculates total units needed for Q, K, V projections and output. Defines ranges for these components.
    *   `build(input_dim)`: Builds the connectivity pattern, including attention connections (`_build_attention_connections`) and position-wise connections (`_build_position_connections`).
    *   `_build_attention_connections()`: Builds connections for multi-head attention between Q, K, V, and output ranges.
    *   `_build_position_connections()`: Builds connections for position-wise processing and skip connections.
    *   `get_config()`: Returns the configuration.
    *   `from_config(config)`: Class method to create a `LanguageWiring` instance from configuration.

### Random Wiring (`ember_ml.nn.modules.wiring.random_map`)

*   **`RandomMap(NeuronMap)`**: Implements a random wiring configuration where connections are randomly generated based on the sparsity level.
    *   `__init__(units, output_dim, input_dim, sparsity_level, seed)`: Initializes base properties.
    *   `build(input_dim)`: Builds the random wiring by creating random input, recurrent, and output masks based on the `sparsity_level`.
    *   `get_config()`: Returns the configuration.
    *   `from_config(config)`: Class method to create a `RandomMap` instance from configuration.

### Robotics Wiring (`ember_ml.nn.modules.wiring.robotics_wiring`)

*   **`RoboticsWiring(NeuronMap)`**: Specialized wiring for robotics applications, implementing sensor processing, state estimation, and control layers.
    *   `__init__(sensor_neurons, state_neurons, control_neurons, sensor_fanout, state_recurrent, control_fanin, reflex_probability, **kwargs)`: Initializes neuron counts for each layer and connectivity parameters (fanout, recurrent connections, fanin, reflex probability). Defines ranges for each neuron layer.
    *   `build(input_dim)`: Builds the connectivity pattern based on the defined layers and connectivity parameters using helper methods (`_build_sensor_connections`, `_build_state_connections`, `_build_control_connections`).
    *   `_build_sensor_connections()`: Builds connections from the sensor layer to state estimation and control (reflexes).
    *   `_build_state_connections()`: Builds recurrent connections within the state estimation layer.
    *   `_build_control_connections()`: Builds connections from state estimation to the control layer.
    *   `get_config()`: Returns the configuration.
    *   `from_config(config)`: Class method to create a `RoboticsWiring` instance from configuration.

### Signal Wiring (`ember_ml.nn.modules.wiring.signal_wiring`)

*   **`SignalWiring(NeuronMap)`**: Specialized wiring for multi-scale signal processing, implementing multiple frequency bands and cross-band interactions.
    *   `__init__(input_size, num_bands, neurons_per_band, output_size, **kwargs)`: Initializes input size, number of bands, neurons per band, and output size. Calculates total units and defines ranges for each band.
    *   `build(input_dim)`: Builds the connectivity pattern, including connections within each band (`_build_band_connections`), between adjacent bands (`_build_cross_band_connections`), and to the output neurons (`_build_output_connections`).
    *   `_build_band_connections()`: Builds dense connections within each frequency band.
    *   `_build_cross_band_connections()`: Builds sparse connections between adjacent frequency bands.
    *   `_build_output_connections()`: Builds connections from each band to the output neurons.
    *   `get_config()`: Returns the configuration.
    *   `from_config(config)`: Class method to create a `SignalWiring` instance from configuration.

### Vision Wiring (`ember_ml.nn.modules.wiring.vision_wiring`)

*   **`VisionWiring(NeuronMap)`**: Specialized wiring for computer vision tasks, implementing local receptive fields and feature hierarchies.
    *   `__init__(input_height, input_width, channels, kernel_size, stride, **kwargs)`: Initializes input dimensions, channel sizes for each layer, kernel size, and stride. Calculates feature map sizes for each layer.
    *   `_get_feature_maps(h, w, channels, stride)`: Helper to calculate feature map sizes.
    *   `_get_receptive_field(h, w, layer)`: Helper to get neuron positions in a local receptive field.
    *   `build(input_dim)`: Builds the connectivity pattern, including local connections (`_build_local_connections`) and skip connections (`_build_skip_connections`).
    *   `_build_local_connections()`: Builds connections based on local receptive fields, mimicking convolutional layers.
    *   `_build_skip_connections()`: Builds sparse skip connections between layers.
    *   `get_config()`: Returns the configuration.
    *   `from_config(config)`: Class method to create a `VisionWiring` instance from configuration.