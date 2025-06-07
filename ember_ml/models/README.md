# ember_ml Models

This directory contains pre-built machine learning models implemented in ember_ml. These models are ready-to-use implementations that can be applied to various tasks.

## Overview

The models module provides high-level, ready-to-use machine learning models that are not specific to a particular domain. These models are built using the lower-level components from the `ember_ml.nn` module and other core modules.

## Available Models

### Restricted Boltzmann Machine (RBM)

The `RBM` class implements a Restricted Boltzmann Machine, which is a generative stochastic neural network that can learn a probability distribution over its set of inputs.

```python
from ember_ml.models.rbm import RBM

# Create an RBM model
rbm = RBM(
    visible_size=784,
    hidden_size=256,
    learning_rate=0.01
)

# Train the model
rbm.train(training_data, epochs=10, batch_size=32)

# Generate samples
samples = rbm.generate(num_samples=10)
```

### RBM Anomaly Detector

The `RBMAnomalyDetector` class implements an anomaly detection model based on a Restricted Boltzmann Machine. It learns the normal patterns in the data and can detect anomalies by measuring the reconstruction error.

```python
from ember_ml.models.rbm_anomaly_detector import RBMAnomalyDetector

# Create an RBM anomaly detector
detector = RBMAnomalyDetector(
    visible_size=784,
    hidden_size=256,
    learning_rate=0.01,
    threshold=0.1
)

# Train the model on normal data
detector.train(normal_data, epochs=10, batch_size=32)

# Detect anomalies
anomalies = detector.detect(test_data)
```

### Liquid State Machine

The Liquid State Machine (LSM) models are located in the `liquid` subdirectory. These models implement liquid state machines, which are a type of reservoir computing model.

```python
from ember_ml.models.liquid.lsm import LiquidStateMachine

# Create a liquid state machine
lsm = LiquidStateMachine(
    input_size=10,
    reservoir_size=100,
    output_size=5,
    connectivity=0.1,
    spectral_radius=0.9
)

# Train the model
lsm.train(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
predictions = lsm.predict(X_test)
```

## Implementation Details

### RBM

The RBM implementation uses a backend-agnostic approach, allowing it to work with different backends (NumPy, PyTorch, MLX) without any code changes. This is achieved through the use of the `ops` module, which provides a unified interface for tensor operations across different backends.

The RBM training process uses Contrastive Divergence (CD) to approximate the gradient of the log-likelihood. The implementation supports both CD-1 (single-step Gibbs sampling) and CD-k (k-step Gibbs sampling).

### RBM Anomaly Detector

The RBM Anomaly Detector extends the base RBM model to detect anomalies by measuring the reconstruction error. It learns the normal patterns in the data during training and can detect anomalies by comparing the reconstruction error to a threshold.

### Liquid State Machine

The Liquid State Machine implementation uses a reservoir of recurrently connected neurons to process temporal information. The reservoir is a randomly connected network of neurons that transforms the input into a high-dimensional state space. The output is then computed using a linear readout layer trained on the reservoir states.

## Usage Examples

### Training an RBM

```python
import numpy as np
from ember_ml.models.rbm import RBM
from ember_ml import ops

# Create random data
data = ops.random_uniform((1000, 784))

# Create an RBM model
rbm = RBM(
    visible_size=784,
    hidden_size=256,
    learning_rate=0.01
)

# Train the model
rbm.train(data, epochs=10, batch_size=32)

# Generate samples
samples = rbm.generate(num_samples=10)

# Reconstruct data
reconstructed = rbm.reconstruct(data[:10])

# Compute reconstruction error
error = ops.stats.mean(ops.square(data[:10] - reconstructed))
print(f"Reconstruction error: {error}")
```

### Using RBM for Anomaly Detection

```python
import numpy as np
from ember_ml.models.rbm_anomaly_detector import RBMAnomalyDetector
from ember_ml import ops

# Create normal data
normal_data = ops.random_uniform((1000, 784))

# Create anomalous data
anomalous_data = ops.random_uniform((100, 784)) * 2.0

# Create an RBM anomaly detector
detector = RBMAnomalyDetector(
    visible_size=784,
    hidden_size=256,
    learning_rate=0.01,
    threshold=0.1
)

# Train the model on normal data
detector.train(normal_data, epochs=10, batch_size=32)

# Detect anomalies
normal_scores = detector.score(normal_data)
anomalous_scores = detector.score(anomalous_data)

print(f"Normal data scores: mean={ops.stats.mean(normal_scores)}, std={ops.std(normal_scores)}")
print(f"Anomalous data scores: mean={ops.stats.mean(anomalous_scores)}, std={ops.std(anomalous_scores)}")

# Detect anomalies
normal_anomalies = detector.detect(normal_data)
anomalous_anomalies = detector.detect(anomalous_data)

print(f"Normal data anomalies: {ops.stats.sum(normal_anomalies)}/{len(normal_data)}")
print(f"Anomalous data anomalies: {ops.stats.sum(anomalous_anomalies)}/{len(anomalous_data)}")
```

## Relationship with Other Modules

The models in this directory are built using the lower-level components from other modules:

- **ember_ml.nn**: Provides the neural network components used to build the models
- **ember_ml.ops**: Provides the tensor operations used by the models
- **ember_ml.backend**: Provides the backend-specific implementations used by the models

For domain-specific models, see the respective domain modules:

- **ember_ml.wave.models**: Contains models specifically designed for wave-based neural processing
- **ember_ml.audio.models**: Contains models specifically designed for audio processing

## Module Organization

The models module is organized as follows:

- **ember_ml.models**: Contains general-purpose machine learning models
  - `RBM`: Restricted Boltzmann Machine implementation
  - `RBMAnomalyDetector`: Anomaly detection model based on RBM
  - **liquid**: Contains Liquid State Machine implementations
  - **rbm**: Contains additional RBM-related implementations

For more information on the distinction between the `ember_ml.nn` and `ember_ml.models` modules, see the [Module Architecture documentation](../../docs/module_architecture.md).