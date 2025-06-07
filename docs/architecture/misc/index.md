# Ember ML Miscellaneous Frontend Components

This section documents various frontend components of the Ember ML framework that are not specifically part of the Wave, Liquid Neuron, or Attention architectures. These components provide foundational structures, utility functions, and implementations for other model types and operations.

## Core Structures

*   **Base Module and Parameters:** The fundamental building blocks for creating trainable neural network components. ([Details](core_structures.md))

## Foundational Components

*   **Initializers:** Methods for initializing the weights and biases of neural network parameters. ([Details](initializers.md))
*   **Features:** Components for preprocessing and engineering features from various data types. ([Details](features.md))
*   **Wiring and Neuron Maps:** Definitions for connectivity patterns within neural circuits. ([Details](wiring_neuron_maps.md))
*   **Tensor Interfaces and Types:** Abstract definitions and common implementations for backend-agnostic tensor operations and data types. ([Details](tensor_interfaces_types.md))

## Other Model Implementations

*   **Restricted Boltzmann Machines (RBMs):** Implementations of RBMs for feature learning and anomaly detection. ([Details](rbms.md))

## Operations Interfaces

*   **Linear Algebra Operations:** Abstract interface for linear algebra functions. ([Details](ops_interfaces.md#linear-algebra-operations))
*   **Statistical Operations:** Abstract interface for statistical functions. ([Details](ops_interfaces.md#statistical-operations))