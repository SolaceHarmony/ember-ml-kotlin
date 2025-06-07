# Geometric Liquid Neuron Variants

This section describes implementations of liquid neurons that operate on non-Euclidean manifolds, specifically the unit sphere (S² or higher dimensions).

## Core Concepts

Geometric variants adapt the dynamics of continuous-time RNNs (like LTC) to operate on curved spaces. Instead of standard vector addition and subtraction in Euclidean space, updates are performed using concepts from differential geometry, such as logarithmic maps (projecting differences onto the tangent space) and exponential maps (projecting tangent vectors back onto the manifold). This allows the neuron state to remain constrained to the manifold (e.g., the surface of a sphere).

## Components

### `ember_ml.nn.modules.rnn.geometric`

*   **`normalize_sphere(vec)`**: Helper function to normalize a vector to unit length (project onto the unit sphere). Uses `ops.linalg.norm`.
*   **`GeometricNeuron(Module)`**: Abstract base class for geometry-aware neurons.
    *   Requires subclasses to implement `_initialize_manifold_state` and `_manifold_update`.
    *   `update(input_signal, **kwargs)`: Updates the neuron's `manifold_state` by calling `_manifold_update`.

### `ember_ml.nn.modules.rnn.spherical_ltc`

*   **`SphericalLTCConfig`**: Dataclass holding configuration specific to Spherical LTC neurons (`tau`, `gleak`, `dt`).
*   **`log_map_sphere(p, q)`**: Computes the logarithmic map from point `p` to point `q` on the unit sphere. This gives the tangent vector at `p` pointing along the geodesic towards `q`. Uses `ops.arccos`, `ops.dot`, `ops.norm`.
*   **`exp_map_sphere(p, v)`**: Computes the exponential map of tangent vector `v` at point `p` on the unit sphere. This follows the geodesic from `p` in the direction `v` for a distance equal to the norm of `v`. Uses `ops.cos`, `ops.sin`, `ops.norm`.
*   **`SphericalLTCNeuron(GeometricNeuron)`**: Implements an LTC neuron operating on the unit sphere (default S²).
    *   `__init__(..., dim)`: Initializes the neuron, setting its state dimension and baseline (North pole).
    *   `_initialize_manifold_state()`: Initializes state as a random point on the unit sphere.
    *   `_manifold_update(current_state, target_state, **kwargs)`: Implements the LTC update rule using `log_map_sphere` and `exp_map_sphere`. It calculates the update vector in the tangent space based on the difference between the current and target states (derived from input), scales it by `dt/tau`, and applies it using the exponential map. It also applies a leak towards the baseline state using the same geometric operations.
*   **`SphericalLTCChain(BaseChain)`**: Implements a chain of connected `SphericalLTCNeuron` instances. (Note: `BaseChain` is not defined in the provided files, suggesting it might be an external or missing dependency/class).
    *   `__init__(..., base_tau_or_config, ...)`: Initializes the chain, accepting either a base tau or a `SphericalLTCConfig`. Uses a factory pattern to create `SphericalLTCNeuron` instances.
    *   `__call__(input_batch)`: Processes an input batch through the chain (currently seems to process each batch item sequentially rather than in parallel).
    *   `reset_states(batch_size)`: Resets the state of neurons in the chain.
    *   `get_forgetting_times(states, threshold)`: Analyzes state history to estimate how long each neuron retains information about an initial pattern.
    *   `update(input_signals)`: Updates the chain state for one time step, propagating state from one neuron to the next as input.
*   **`create_spherical_ltc_chain(...)`**: Factory function to create a `SphericalLTCChain`.