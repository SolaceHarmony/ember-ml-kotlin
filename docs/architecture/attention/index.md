# Ember ML Attention Mechanisms

This section details the architecture and components related to various attention mechanisms implemented within the Ember ML framework. Attention allows models to dynamically focus on relevant parts of the input or internal state.

## Core Concepts

*   **General Attention:** The fundamental idea of computing attention weights based on query, key, and value representations.
*   **Causal Attention:** Incorporates factors like temporal decay, prediction accuracy (causality), and novelty to modulate attention weights dynamically. ([Details](implementations.md#causal-attention))
*   **Temporal Attention:** Specialized attention for sequences, incorporating positional or time-based information. ([Details](implementations.md#temporal-attention))
*   **Multi-Head Attention:** Standard transformer-style attention mechanism involving multiple parallel attention computations ("heads"). ([Details](implementations.md#multi-head-attention))
*   **Prediction Attention:** An attention variant where weights are influenced by the accuracy of internal predictions. ([Details](implementations.md#prediction-attention))

## Implementations

*   **Attention Mechanisms:** Details on the specific implementations like `CausalAttention`, `TemporalAttention`, `WaveMultiHeadAttention`, and `PredictionAttention`. ([Details](implementations.md))
*   **State and Memory:** Components for managing the state required by certain attention mechanisms, like `AttentionState` and `CausalMemory`. ([Details](state_memory.md))

## Integration

*   **Hybrid Neurons:** Examples of how attention mechanisms are integrated with other neuron types, such as `LTCNeuronWithAttention`. ([Details](integration.md))