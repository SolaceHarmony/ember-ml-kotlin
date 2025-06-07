# Hybrid Liquid Neuron Models

This section describes architectures that combine liquid neurons (LTC, CfC) with other neural network mechanisms like attention or standard RNN cells (LSTM).

## Core Concepts

Hybrid models aim to leverage the strengths of different architectures. For instance, combining the adaptive temporal processing of LTC/CfC with the sequence modeling capabilities of LSTMs or the context-focusing ability of attention mechanisms.

## Components

### `ember_ml.nn.modules.rnn.hybrid`

*   **`HybridNeuron(BaseNeuron)`**: Combines LTC dynamics with a temporal attention mechanism.
    *   Maintains a `memory_buffer` of recent inputs.
    *   Uses an `AttentionLayer` to compute an `attended` representation based on the current state (query) and the memory buffer (key, value).
    *   The LTC state update `dh` incorporates this `attended` input instead of the raw input signal.
*   **`AttentionLayer(nn.Module)`**: A standard multi-head attention layer (using PyTorch's `nn.Linear`, `nn.Softmax`, etc.). Takes query, key, and value tensors and computes an attention-weighted output. *(Note: This implementation uses PyTorch directly, violating backend purity rules).*
*   **`HybridLNNModel(nn.Module)`**: A complex hybrid architecture combining parallel LTC chains (`ImprovedLiquidTimeConstantCell`), an LSTM layer, and an `AttentionLayer`.
    *   Processes input through multiple parallel LTC chains.
    *   The outputs of the LTC chains (or some representation derived from them) likely feed into the LSTM.
    *   Attention is potentially applied to the LSTM outputs or states.
    *   Includes an `_integrate_ode` method using `torchdiffeq` for solving the LTC cell dynamics. *(Note: This also uses PyTorch directly).*
*   **`ImprovedLiquidTimeConstantCell(nn.Module)`**: An enhanced LTC cell implementation (using PyTorch) with potentially non-linear dynamics (`torch.sqrt(x + 1)` term in the derivative). Used within `HybridLNNModel`. *(Note: PyTorch dependency).*

### `ember_ml.models.stride_aware_cfc` (Relevant Components)

*   **`LiquidNetworkWithMotorNeuron(Module)`**: Combines a stride-aware CfC cell (`StrideAwareCfCCell`) with a `MotorNeuron`.
    *   Processes input through the CfC cell.
    *   Includes optional `mixed_memory` using a sigmoid gate applied to the cell output.
    *   Feeds the cell output to a `MotorNeuron` to generate trigger signals.
*   **`MotorNeuron(Module)`**: A simple module that takes input, applies a linear layer with sigmoid activation to produce an output, and compares this output to a (potentially adaptive) threshold to generate a binary trigger signal.
*   **`LSTMGatedLiquidNetwork(Module)`**: Combines a CfC cell and an LSTM cell.
    *   Processes input through both cells in parallel.
    *   Uses the LSTM output to compute a gate (`sigmoid` activation).
    *   Applies the gate multiplicatively to the CfC output.
    *   Concatenates the gated CfC output and the LSTM output before feeding to a final output projection layer.
*   **`MultiStrideLiquidNetwork(Module)`**: Processes input through multiple `StrideAwareCfCCell` instances, each configured with a different stride length.
    *   Applies separate input projections for each stride.
    *   Processes input through the corresponding cell only at time steps divisible by the stride length (otherwise reuses the previous state/output).
    *   Concatenates the outputs from all stride-specific cells.
    *   Feeds the combined output to a final output projection layer.