# Attention Mechanism Implementations

This section details the specific implementations of various attention mechanisms found within Ember ML.

## Causal Attention (`ember_ml.nn.attention.causal`)

*   **`CausalAttention(BaseAttention)`**: Implements an attention mechanism that incorporates causality, temporal dynamics, and novelty.
    *   `__init__(hidden_size, decay_rate, novelty_threshold, memory_length)`: Initializes parameters and state tracking (`states` dictionary, `history` list). Includes learnable linear projections (`temporal_proj`, `causal_proj`, `novelty_proj`).
    *   `forward(query, key, value, mask)`: Computes attention output. It iterates through the batch, calling the `update` method for each item to get attention weights, then applies these weights to the `value` tensor.
    *   `update(neuron_id, current_state, target_state)`: Updates the `AttentionState` for a specific neuron (or batch item). Calculates temporal weight based on history decay, causal weight based on prediction accuracy (difference between target and current state), and novelty weight based on the magnitude of state change compared to `novelty_threshold`. Stores the updated state and history.
    *   `reset()`: Clears the internal `states` and `history`.
    *   `save_state()` / `load_state()`: Methods for saving and loading the mechanism's state, including configuration parameters, internal states, history, and projection layer weights.
*   **`create_causal_attention(...)`**: Factory function to create a `CausalAttention` instance.
*   **`PredictionAttention(BaseAttention)`**: An attention mechanism where scores are modulated by prediction accuracy.
    *   `__init__(hidden_size, num_heads, dropout, memory_size)`: Initializes standard multi-head attention projections (Q, K, V, Out), a `predictor` network (Sequential Linear layers), and a `CausalMemory` instance.
    *   `forward(query, key, value, mask)`: Computes attention. It predicts values based on keys using the `predictor`, calculates prediction error against actual values, and uses this error to derive `prediction_weights`. Standard attention scores (Q\*K.T) are then multiplied by these `prediction_weights` before softmax normalization and application to the `value` tensor. Updates the `CausalMemory` with cause (key), effect (value), and prediction accuracy.
    *   `get_attention_weights()`: Returns the last computed attention weights.

## Temporal Attention (`ember_ml.nn.attention.temporal`)

*   **`PositionalEncoding(Module)`**: Implements standard sinusoidal positional encoding.
    *   `__init__(hidden_size, dropout, max_len)`: Initializes dropout and creates the positional encoding matrix `pe` up to `max_len`.
    *   `forward(x, times)`: Adds positional encoding to the input tensor `x`. If `times` are provided, it scales the positional encoding based on normalized time differences before adding. Applies dropout.
*   **`TemporalAttention(BaseAttention)`**: A multi-head attention mechanism specialized for temporal sequences.
    *   `__init__(hidden_size, num_heads, dropout, max_len, use_time_embedding)`: Initializes standard multi-head attention projections (Q, K, V, Out), an optional `PositionalEncoding` (`time_embedding`), and a `time_gate` network (Linear + Sigmoid).
    *   `forward(query, key, value, mask, times)`: Computes attention. If `use_time_embedding` and `times` are provided, it first adds positional encodings to Q, K, V. It computes standard attention scores. If `times` are provided, it calculates time differences between query and key positions, feeds these differences along with the query representation into the `time_gate`, and multiplies the attention scores by the resulting time gates before applying softmax and dropout.
    *   `get_attention_weights()`: Returns the last computed attention weights.
*   **`create_temporal_attention(...)`**: Factory function to create a `TemporalAttention` instance.

## Multi-Head Attention (`ember_ml.wave.models.wave_transformer`)

*   **`WaveMultiHeadAttention(nn.Module)`**: Implements standard multi-head self-attention as found in Transformers (using PyTorch's `nn.Linear`, `nn.Softmax`, etc.).
    *   `__init__(d_model, num_heads, dropout)`: Initializes Q, K, V, and output linear projections.
    *   `forward(query, key, value, mask)`: Performs the standard multi-head attention calculation: projects Q, K, V; computes scaled dot-product attention scores; applies optional mask; applies softmax and dropout; computes weighted sum of values; applies final output projection.

*(Note: `BaseAttention` is imported but not defined in the analyzed files, suggesting it might be an abstract base class defined elsewhere or missing.)*
*(Note: Some attention implementations, like `AttentionLayer` in `hybrid.py` and `WaveMultiHeadAttention`, use PyTorch directly, violating backend purity.)*