# Liquid Neuron Core Concepts

This section outlines the fundamental ideas behind Liquid Time-Constant (LTC), Closed-form Continuous-time (CfC), and related continuous-time recurrent neural network models within Ember ML.

## Continuous-Time Dynamics

Unlike traditional RNNs (like LSTMs or GRUs) that operate on discrete time steps, continuous-time models like LTC, CfC, ELTC, CTRNN, and CTGRU define their state evolution using Ordinary Differential Equations (ODEs).

*   **ODE Representation:** The core dynamic is typically expressed as `dh/dt = f(h, x, t)`, where `h` is the hidden state, `x` is the input, `t` is time, and `f` is a function parameterized by the network's weights.
*   **Handling Irregular Time Series:** This formulation naturally handles irregularly sampled data, as the time difference (`dt`) between inputs can be explicitly incorporated into the ODE solver step.
*   **Numerical Solvers:** The ODE is solved numerically using methods like Euler (explicit or semi-implicit) or Runge-Kutta (RK4) to update the state over a given time interval `dt`. The choice of solver impacts the trade-off between computational cost, accuracy, and stability.

## LTC (Liquid Time-Constant)

*   **Adaptive Time Constants:** The key idea behind LTC networks is that the time constant (`tau`) of each neuron is not fixed but dynamically adapts based on the input. This allows neurons to adjust their integration timescale, responding quickly to rapid changes or slowly integrating information over longer periods as needed.
*   **Biological Inspiration:** Inspired by the dynamics of biological neurons, particularly the Caenorhabditis elegans nervous system.
*   **Implementation:** Often involves a separate gating mechanism or a specific formulation of the ODE where the effective time constant depends on the neuron's state or input. (Note: The specific `LTC` implementation details in `ember_ml/nn/modules/rnn/ltc.py` need further examination to confirm the exact mechanism used).

## CfC (Closed-form Continuous-time)

*   **Closed-Form Solution:** CfC models utilize a specific type of linear ODE whose solution can be expressed in closed form (often involving matrix exponentials). This avoids the need for iterative numerical ODE solvers during the forward pass, potentially leading to faster computation and improved stability.
*   **Linear Dynamics Assumption:** The closed-form solution typically relies on an assumption of linear dynamics within the time step, although non-linearities are introduced through activation functions applied to the state or output.
*   **Implementation (`ember_ml/nn/modules/rnn/cfc.py`):** The `CfC` class in Ember ML implements these dynamics, likely using the closed-form update rule involving decay factors derived from the time constants and time deltas. It integrates directly with `NeuronMap` (specifically `NCPMap`) for its structure.

## ELTC (Enhanced LTC)

*   **Configurable ODE Solvers:** ELTC extends the LTC concept by allowing the user to explicitly choose the numerical ODE solver (`ODESolver.SEMI_IMPLICIT`, `ODESolver.EXPLICIT`, `ODESolver.RUNGE_KUTTA`).
*   **Unfolding:** Introduces the `ode_unfolds` parameter, which determines how many solver steps are performed within a single conceptual time step (`dt`) of the RNN. More unfolds increase accuracy at the cost of computation.
*   **Implementation (`ember_ml/nn/modules/rnn/eltc.py`):** The `ELTC` class implements the core ODE `dy/dt = σ(Wx + Uh + b) - y` and uses helper methods (`_explicit_euler_solve`, `_semi_implicit_solve`, `_rk4_solve`) to perform the chosen integration method over `ode_unfolds` steps.

## Stride Awareness

*   **Multi-Timescale Processing:** Stride-aware variants (e.g., `StrideAwareCfCCell`, `StrideAwareWiredCfCCell`, `StrideAware`) are designed to process input sequences at multiple temporal resolutions simultaneously.
*   **Stride Length:** Each stride-aware cell or layer is associated with a `stride_length`. It processes inputs only at intervals defined by this stride.
*   **Time Scaling:** Often incorporates a `time_scale_factor` (potentially learnable) that interacts with the `stride_length` to adjust the effective time constant for that specific timescale.
*   **Implementation:** Achieved either by modifying the cell's internal dynamics based on stride and time scale factors (as seen in the Keras-based `StrideAwareWiredCfCCell` in `testfile.py`) or by controlling how inputs are fed to different cells/layers in a larger network architecture (as suggested by `MultiStrideLiquidNetwork` in `stride_aware_cfc.py`).