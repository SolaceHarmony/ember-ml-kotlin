# Initializers

This section documents the parameter initialization methods available in Ember ML. Initializers are used to set the initial values of trainable parameters (weights and biases) in neural network layers.

## Core Concepts

Proper initialization is crucial for training neural networks effectively. Ember ML provides backend-agnostic initializers that can be applied to `Parameter` objects, ensuring consistent behavior regardless of the underlying computation backend.

## Components

### `ember_ml.nn.initializers`

This module exposes various initializer functions and classes.

*   **`glorot_uniform`**: Glorot (Xavier) uniform initializer. Draws samples from a uniform distribution within `[-limit, limit]`, where `limit = sqrt(6 / (fan_in + fan_out))`.
*   **`glorot_normal`**: Glorot (Xavier) normal initializer. Draws samples from a truncated normal distribution centered on 0 with standard deviation `stddev = sqrt(2 / (fan_in + fan_out))`.
*   **`orthogonal`**: Orthogonal initializer. Initializes weights as a random orthogonal matrix. Used for recurrent weights to help maintain gradient norms.
*   **`BinomialInitializer`**: A class for initializing weights with binary values (0 or 1) based on a specified probability.
    *   `__init__(probability, seed)`: Initializes with the probability of a value being 1 and an optional random seed.
    *   `__call__(shape, dtype, device)`: Generates a tensor of the specified `shape` with binary values sampled according to the `probability`.
*   **`binomial`**: A function (likely a helper or alias) related to binomial initialization. (Based on `ember_ml/nn/initializers/binomial.py`, this function is the core logic used by `BinomialInitializer`).
*   **`get_initializer(name)`**: A helper function to retrieve an initializer function or class by its string name.

*(Note: The implementations of `glorot_uniform`, `glorot_normal`, and `orthogonal` are not in the analyzed files but are exposed by the `__init__.py`, suggesting they are either implemented directly within the backend modules or imported from another internal location.)*