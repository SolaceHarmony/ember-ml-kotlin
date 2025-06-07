# Liquid Neuron Cells and Layers

This section details the specific implementations of various continuous-time RNN cells and layers within Ember ML, including LTC, CfC, ELTC, CTRNN, CTGRU, LQNet, and their stride-aware variants.

## Core Implementations

### `ember_ml.nn.modules.rnn.cfc`

*   **`CfC(Module)`**: Implements the Closed-form Continuous-time RNN layer.
    *   Integrates directly with `NCPMap` for structure and dynamics.
    *   Uses a closed-form solution for state updates, potentially involving decay factors based on time constants and time deltas.
    *   Supports different modes (`default`, `pure`, `no_gate`) affecting the gating mechanism.
    *   Includes parameters like `time_scale_factor`, `activation`, `recurrent_activation`, initializers, and `mixed_memory`.
    *   `build(input_shape)`: Initializes weights (kernel, recurrent\_kernel, bias, time\_scale) based on input shape and neuron map.
    *   `forward(inputs, initial_state, time_deltas)`: Processes the input sequence using the closed-form update rule, handling time deltas if provided. Returns sequence or final output, and optionally the final state.
    *   `reset_state(batch_size)`: Returns the initial zero state.

### `ember_ml.nn.modules.rnn.ltc`

*   **`LTC(Module)`**: Implements the Liquid Time-Constant RNN layer. (Note: The specific mechanism for adaptive time constants needs further inspection of the code, as the provided file might be incomplete or simplified).
    *   Similar structure to `CfC`, likely using `NCPMap`.
    *   Expected to implement adaptive time constants based on input or state, distinguishing it from CfC.
    *   Includes parameters like `solver`, `ode_unfolds`, `epsilon`, `implicit_param_constraints`.
    *   `build(input_shape)`: Initializes weights and parameters.
    *   `forward(inputs, initial_state, timespans)`: Processes the input sequence using a numerical ODE solver (`_ode_solver`) based on the selected `solver` type and `ode_unfolds`.
    *   `_ode_solver(inputs, state, elapsed_time)`: Internal method performing the ODE integration step.
    *   `reset_state(batch_size)`: Returns the initial zero state.

### `ember_ml.nn.modules.rnn.eltc`

*   **`ODESolver(Enum)`**: Defines available ODE solvers (`SEMI_IMPLICIT`, `EXPLICIT`, `RUNGE_KUTTA`).
*   **`ELTC(Module)`**: Implements the Enhanced Liquid Time-Constant RNN layer.
    *   Extends LTC by explicitly allowing selection of ODE solvers (`solver` parameter) and the number of integration steps per RNN time step (`ode_unfolds`).
    *   Defines the core ODE: `dy/dt = σ(Wx + Uh + b) - y`.
    *   `build(input_shape)`: Initializes weights (gleak, vleak, cm, sigma, mu, w, erev, sensory\_*, input\_*, output\_*) and potentially a `memory_cell` if `mixed_memory` is True.
    *   `_create_memory_cell(...)`: Helper to create a simple LSTM-like memory cell.
    *   `_explicit_euler_solve`, `_semi_implicit_solve`, `_rk4_solve`: Internal methods implementing different ODE solvers.
    *   `_ode_solver(inputs, state, elapsed_time)`: Solves the ODE using the selected solver over `ode_unfolds` steps.
    *   `_map_inputs`, `_map_outputs`: Applies linear or affine transformations to inputs/outputs based on `neuron_map` settings.
    *   `forward(inputs, initial_state, timespans)`: Processes the input sequence, applying input mapping, ODE solving, and output mapping. Handles `mixed_memory`.
    *   `reset_state(batch_size)`: Returns the initial zero state (or tuple of states if `mixed_memory`).

### `ember_ml.nn.modules.rnn.ctrnn`

*   **`CTRNN(Module)`**: Implements a classic Continuous-Time Recurrent Neural Network layer.
    *   Uses the ODE: `dh/dt = (-h + σ(W*x + U*h + b)) / τ`.
    *   Integrates directly with `NCPMap`.
    *   `build(input_shape)`: Initializes weights (kernel, recurrent\_kernel, bias, tau).
    *   `_update_state(inputs, state, elapsed_time)`: Performs one step of the CTRNN ODE update using explicit Euler.
    *   `forward(inputs, initial_state, timespans)`: Processes the input sequence by repeatedly calling `_update_state`.
    *   `reset_state(batch_size)`: Returns the initial zero state.

### `ember_ml.nn.modules.rnn.ctgru`

*   **`CTGRU(Module)`**: Implements a Continuous-Time Gated Recurrent Unit layer.
    *   Adapts the standard GRU update equations into a continuous-time ODE framework: `dh/dt = (-h + target_state) / tau`, where `target_state` is calculated using GRU gating mechanisms (update `z`, reset `r`, candidate `c`).
    *   Integrates directly with `NCPMap`.
    *   `build(input_shape)`: Initializes weights for the three gates (kernel\_z/r/h, recurrent\_kernel\_z/r/h, bias\_z/r/h) and the time constant `tau`.
    *   `_update_state(inputs, state, elapsed_time)`: Performs one step of the CTGRU ODE update using explicit Euler.
    *   `forward(inputs, initial_state, timespans)`: Processes the input sequence by repeatedly calling `_update_state`.
    *   `reset_state(batch_size)`: Returns the initial zero state.

### `ember_ml.nn.modules.rnn.lqnet`

*   **`LQNet(Module)`**: Implements the Liquid Quantum Network layer. (Note: The exact "quantum" inspiration or mechanism needs further clarification from the code or documentation).
    *   Appears to maintain separate classical (`h_c`) and stochastic (`h_s`) state components.
    *   Uses b-symplectic integration (`apply_b_symplectic`) and a stochastic-quantum mapping (`apply_stochastic_quantum`).
    *   Optionally uses harmonic embedding for inputs (`apply_harmonic_embedding`).
    *   Includes parameters like `W_symplectic`, `A_matrix`, `B_matrix`, `nu_0`, `beta`, `noise_scale`.
    *   `build(input_shape)`: Initializes the various weight matrices and parameters.
    *   `forward(inputs, initial_state, time_deltas)`: Processes the input sequence, updating `h_c`, `h_s`, and time `t` at each step using the specialized update rules.
    *   `reset_state(batch_size)`: Returns the initial zero state tuple `(h_c, h_s, t)`.

## Stride-Aware Variants

### `ember_ml.nn.modules.rnn.stride_aware`

*   **`StrideAware(Module)`**: A basic stride-aware RNN layer implementation.
    *   Directly implements stride-aware dynamics without a separate cell class.
    *   Uses `stride_length` and `time_scale_factor` to scale the effective time step in its update rule: `state = state + (1 / (tau * effective_time)) * (input_signal + hidden_signal - state)`.
    *   `_initialize_parameters()`: Initializes input, hidden, output kernels, biases, and time constant `tau`.
    *   `forward(inputs, initial_state, elapsed_time)`: Processes the sequence, applying the time-scaled update rule at each step.
    *   `reset_state(batch_size)`: Returns the initial zero state.

### `ember_ml.nn.modules.rnn.stride_aware_cfc`

*   **`StrideAwareCfC(Module)`**: Implements a stride-aware CfC layer, seemingly intended to wrap a cell but currently contains the cell logic directly.
    *   Inherits from `Module` (previously `ModuleWiredCell`).
    *   Includes parameters like `stride_length`, `time_scale_factor`, `mode`, `activation`, `backbone_units`, `backbone_layers`.
    *   `_initialize_weights()`: Initializes weights for input, recurrent, backbone, time gate, and gating connections. Also initializes sparsity masks.
    *   `_compute_time_scaling(inputs, kwargs)`: Calculates the effective time `t` based on stride and time scale factor.
    *   `forward(inputs, states, **kwargs)`: Processes input using CfC gating logic (`default`, `pure`, `no_gate`) combined with the calculated time scaling. Applies wiring masks.
    *   `get_initial_state(batch_size)`: Returns the initial zero state.

### `ember_ml.models.attention.testfile` (Keras-based Implementations)

*Note: These are Keras implementations found in a test file, likely prototypes or experiments.*
*   **`StrideAwareWiredCfCCell(keras.layers.Layer)`**: A Keras Layer implementing a stride-aware CfC cell that uses a `wirings.Wiring` object for connectivity. Re-implements CfC logic with time scaling based on `stride_length` and `time_scale_factor`.
*   **`StrideAwareCfC(keras.layers.RNN)`**: A Keras RNN layer that wraps either `StrideAwareWiredCfCCell` (if `units` is a Wiring object) or an internally defined `StrideAwareCfCCell` (if `units` is an int). Supports `mixed_memory`.
*   **`MixedMemoryRNN(keras.layers.Layer)`**: A Keras Layer used internally by `StrideAwareCfC` when `mixed_memory` is True. It wraps an RNN cell (like the stride-aware CfC cell) and adds an LSTM-like memory gate mechanism.