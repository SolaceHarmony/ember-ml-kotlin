# API Reference

This section contains detailed API documentation for Ember ML.

## Core Modules

- **`ember_ml.ops`**: Core operations for tensor manipulation
  - `tensor`: Tensor creation and manipulation
  - `math`: Mathematical operations
  - `random`: Random number generation
  - `solver`: Linear algebra operations

- **`ember_ml.backend`**: Backend abstraction system
  - `numpy`: NumPy backend implementation
  - `torch`: PyTorch backend implementation
  - `mlx`: MLX backend implementation
  - `ember`: Ember backend implementation

- **`ember_ml.features`**: Feature extraction and processing
  - `TerabyteFeatureExtractor`: Extracts features from large datasets
  - `TemporalStrideProcessor`: Processes temporal data with variable strides
  - `GenericFeatureEngineer`: General-purpose feature engineering
  - `GenericTypeDetector`: Automatic column type detection

- **`ember_ml.models`**: Machine learning models
  - `liquid`: Liquid neural networks
  - `rbm`: Restricted Boltzmann Machines

## Neural Network Components

- **`ember_ml.core`**: Core neural implementations
  - `ltc`: Liquid Time Constant neurons
  - `geometric`: Geometric processing
  - `spherical_ltc`: Spherical variants
  - `stride_aware_cfc`: Stride-Aware Continuous-time Fully Connected cells

- **`ember_ml.attention`**: Attention mechanisms
  - `temporal`: Time-based attention
  - `causal`: Causal attention
  - `multiscale_ltc`: Multiscale LTC attention

- **`ember_ml.nn`**: Neural network components
  - `wirings`: Network connectivity patterns
  - `modules`: Neural network modules

## Utility Modules

- **`ember_ml.utils`**: Utility functions
  - `math_helpers`: Mathematical utilities
  - `metrics`: Evaluation metrics
  - `visualization`: Plotting tools

- **`ember_ml.initializers`**: Weight initialization
  - `glorot`: Glorot/Xavier initialization
  - `binomial`: Binomial initialization

For detailed documentation on specific functions and classes, refer to the docstrings in the source code.