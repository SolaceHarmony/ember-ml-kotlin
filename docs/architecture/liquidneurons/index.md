# Ember ML Liquid Neuron Architectures

This section details the architecture and components related to Liquid Time-Constant (LTC), Closed-form Continuous-time (CfC), and related continuous-time recurrent neural network models within the Ember ML framework. These models often leverage principles from Neural ODEs and specialized wiring patterns.

## Core Concepts

*   **Continuous-Time Dynamics:** These models operate based on differential equations, allowing them to handle irregularly sampled time series and potentially capture finer temporal dependencies compared to discrete RNNs. ([Details](core_concepts.md))
*   **LTC (Liquid Time-Constant):** Neurons with time constants that adapt based on input, allowing for flexible temporal processing. ([Details](cells_layers.md#ltc))
*   **CfC (Closed-form Continuous-time):** A variant that uses a closed-form solution for the continuous-time dynamics, potentially offering computational advantages. ([Details](cells_layers.md#cfc))
*   **ELTC (Enhanced LTC):** Extends LTC with configurable ODE solvers (Euler, RK4) for varying accuracy and stability trade-offs. ([Details](cells_layers.md#eltc))
*   **Stride Awareness:** Modifications to handle data at multiple timescales simultaneously by processing inputs with different strides. ([Details](cells_layers.md#stride-aware-variants))

## Implementations

*   **Core Cells and Layers:** Base implementations of LTC, CfC, ELTC, CTRNN, CTGRU, LQNet, and related RNN layers. ([Details](cells_layers.md))
*   **Geometric Variants:** Implementations operating on non-Euclidean manifolds, such as Spherical LTC. ([Details](geometric_variants.md))
*   **Hybrid Models:** Architectures combining liquid neurons with other mechanisms like Attention or LSTMs. ([Details](hybrid_models.md))

## Applications & Utilities

*   **Specific Applications:** Models tailored for tasks like anomaly detection or forecasting using liquid networks. ([Details](applications.md))
*   **Training Utilities:** Helper classes for processing data or monitoring training specific to liquid models. ([Details](training_utils.md))