# Container Modules

This section documents the container modules within Ember ML, which are used to hold and manage other layers or components in a structured way.

## Components

### `ember_ml.nn.container.sequential`

*   **`Sequential(Module)`**: A sequential container that holds modules in an ordered list.
    *   `__init__(layers)`: Initializes the container with an optional list of layers.
    *   `forward(x)`: Processes the input `x` sequentially through each layer in the container.
    *   `add(layer)`: Adds a layer to the end of the sequence.
    *   `build(input_shape)`: Builds each layer in the sequence based on the input shape, propagating the shape through the layers.
    *   `get_config()`: Returns the configuration of the sequential container, including the configurations of its layers.
    *   `state_dict()`: Returns a dictionary containing the state dictionaries of all layers.
    *   `load_state_dict(state_dict)`: Loads the state dictionaries into the layers.
    *   `train(mode)`: Sets the training mode for the container and all its layers.
    *   `eval()`: Sets the evaluation mode for the container and all its layers.
    *   `extra_repr()`: Returns a string with extra information (number of layers).
    *   `__repr__()`: Provides a string representation of the sequential container and its layers.
    *   `__getitem__(idx)`: Allows accessing a layer by index or a slice of layers as a new `Sequential` container.
    *   `__len__()`: Returns the number of layers.

### `ember_ml.nn.container.linear`

*   **`Linear(Module)`**: Applies a linear transformation (`y = x @ W.T + b`).
    *   `__init__(in_features, out_features, bias, device, dtype)`: Initializes input and output dimensions, bias, and optionally device and dtype. Initializes `weight` and `bias` (if used) as `Parameter` objects.
    *   `forward(x)`: Computes the linear transformation and adds bias if present.
    *   `add(layer)`: Not implemented; Linear layer does not support adding layers.
    *   `build(input_shape)`: Does nothing; parameters are initialized in `__init__`.
    *   `get_config()`: Returns the configuration (in\_features, out\_features, bias presence).
    *   `state_dict()`: Returns a dictionary containing the `weight` and `bias` (if present) parameters.
    *   `load_state_dict(state_dict)`: Loads the `weight` and `bias` parameters.
    *   `extra_repr()`: Returns a string with extra information (in\_features, out\_features, bias presence).
    *   `__repr__()`: Provides a string representation including dimensions and bias presence.

### `ember_ml.nn.container.dropout`

*   **`Dropout(Module)`**: Applies dropout to the input during training.
    *   `__init__(rate, seed)`: Initializes the dropout rate and optional random seed.
    *   `forward(x)`: Applies dropout by randomly zeroing elements with probability `rate` and scaling the remaining elements. Returns input unchanged if not in training mode or rate is 0.
    *   `add(layer)`: Not implemented; Dropout layer does not support adding layers.
    *   `build(input_shape)`: Does nothing; dropout layer does not need to be built.
    *   `get_config()`: Returns the configuration (rate, seed).
    *   `state_dict()`: Returns the state (rate, seed, training mode).
    *   `load_state_dict(state_dict)`: Loads the state.
    *   `train(mode)`: Sets the training mode.
    *   `eval()`: Sets the evaluation mode.
    *   `extra_repr()`: Returns a string with extra information (rate, seed).
    *   `__repr__()`: Provides a string representation including rate and seed.

### `ember_ml.nn.container.batch_normalization`

*   **`BatchNormalization(Module)`**: Normalizes activations across the batch.
    *   `__init__(axis, momentum, epsilon, center, scale)`: Initializes normalization parameters (axis, momentum, epsilon) and flags for centering and scaling.
    *   `forward(x, training)`: Performs batch normalization. If `training` is True, computes batch statistics and updates moving averages; otherwise, uses moving averages. Centers and scales the input.
    *   `add(layer)`: Not implemented; BatchNormalization layer does not support adding layers.
    *   `build(input_shape)`: Initializes `gamma`, `beta` (if used), `moving_mean`, and `moving_variance` as `Parameter` or buffer objects based on the input shape.
    *   `get_config()`: Returns the configuration (axis, momentum, epsilon, center, scale).
    *   `state_dict()`: Returns the state (gamma, beta, moving\_mean, moving\_variance).
    *   `load_state_dict(state_dict)`: Loads the state.
    *   `train(mode)`: Sets the training mode (affects whether batch or moving statistics are used).
    *   `eval()`: Sets the evaluation mode.
    *   `extra_repr()`: Returns a string with extra information (axis, momentum, epsilon).
    *   `__repr__()`: Provides a string representation including parameters.