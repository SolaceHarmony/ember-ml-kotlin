# Attention Mechanism Integration

This section provides examples of how attention mechanisms are integrated with other neural components within Ember ML.

## Components

### `ember_ml.nn.attention.attention`

*   **`LTCNeuronWithAttention`**: This class demonstrates the integration of an attention mechanism (`CausalAttention`) with the dynamics of a Liquid Time-Constant (LTC) neuron.
    *   **Initialization**: Takes standard LTC parameters (`tau`, `dt`) and optional `attention_params` which are passed to the internal `CausalAttention` instance.
    *   **Update Logic**:
        1.  Calculates a `prediction_error` based on the difference between the current input and the neuron's `last_prediction`.
        2.  Calls the internal `attention.update()` method, providing the `prediction_error`, `current_state`, and the input signal (as `target_state`).
        3.  Retrieves the computed `attention_value` from the attention mechanism.
        4.  Modulates the neuron's effective time constant (`effective_tau`) based on the `attention_value` (higher attention leads to a smaller effective tau, i.e., faster response).
        5.  Modulates the input signal based on the `attention_value` (higher attention increases the effective input strength).
        6.  Updates the neuron's `state` using the standard LTC differential equation (`d_state = (1/effective_tau) * (weighted_input - state) * dt`), but with the modulated time constant and input.
        7.  Stores the updated state as `last_prediction` for the next step.
    *   **Attention Access**: Provides a `get_attention_value()` method to retrieve the current total attention weight from the internal `CausalAttention` state.

This example illustrates how attention, driven by factors like prediction error and novelty (handled within `CausalAttention`), can dynamically modulate the core parameters (time constant, input sensitivity) of another neuron type like LTC.