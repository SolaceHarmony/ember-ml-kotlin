# Attention State and Memory

This section describes components used for managing state and memory within certain attention mechanisms in Ember ML, particularly `CausalAttention` and `PredictionAttention`.

## Components

### `ember_ml.nn.attention.mechanisms.state`

*   **`AttentionState`**: A simple dataclass used to store the different components of attention weight for a single neuron or item.
    *   Attributes: `temporal_weight`, `causal_weight`, `novelty_weight` (all floats, default to 0.0).
    *   `compute_total()`: Calculates the total attention value by averaging the three weight components.

### `ember_ml.nn.attention.causal`

*   **`CausalMemory`**: Manages a memory buffer of cause-effect relationships and associated prediction accuracies. Used by `PredictionAttention`.
    *   `__init__(max_size)`: Initializes the memory with a maximum capacity. Stores pairs as `List[Tuple[Tensor, Tensor]]` and accuracies as `List[float]`.
    *   `add(cause, effect, accuracy)`: Adds a new cause-effect pair and its prediction accuracy to the memory, removing the oldest entry if capacity is exceeded.
    *   `_compute_cosine_similarity(a, b)`: Computes cosine similarity between two tensors using `ops` functions.
    *   `get_similar_causes(current_state, threshold)`: Finds indices of past causes in memory that are similar (based on cosine similarity) to the `current_state`.
    *   `get_prediction(current_state, k)`: Predicts the likely effect based on the `current_state` by finding the `k` most similar past causes and computing a weighted average of their corresponding effects, weighted by similarity. Also returns a confidence score based on the prediction accuracies associated with the retrieved memories.
    *   `clear()`: Empties the memory buffer.

*(Note: The `CausalAttention` class itself also maintains state in the form of a dictionary mapping neuron IDs to `AttentionState` objects and a `history` list, but these are managed internally within the class described in `implementations.md`.)*