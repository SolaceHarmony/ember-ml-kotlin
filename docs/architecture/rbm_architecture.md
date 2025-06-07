# Restricted Boltzmann Machine Architecture

This document provides an architectural overview of the Restricted Boltzmann Machine (RBM) implementation in the Ember ML project.

## Overview

The Ember ML project includes two implementations of Restricted Boltzmann Machines:

1. **CPU-Friendly Implementation** (`RestrictedBoltzmannMachine`): Optimized for computational efficiency on CPU with minimal requirements.
2. **PyTorch Implementation** (`RBM`): Leverages PyTorch tensors and operations for potential GPU acceleration.

Both implementations provide the core functionality of RBMs while focusing on different use cases and performance characteristics.

## Core Concepts

Restricted Boltzmann Machines are a type of stochastic neural network that can learn a probability distribution over its set of inputs. Key characteristics include:

- **Bipartite Graph Structure**: RBMs consist of a visible layer and a hidden layer, with no connections within each layer.
- **Energy-Based Model**: RBMs define an energy function over the joint configuration of visible and hidden units.
- **Generative Capabilities**: Once trained, RBMs can generate new samples that resemble the training data.
- **Unsupervised Learning**: RBMs learn without labeled data, extracting features and patterns from the input.

## CPU-Friendly Implementation

The `RestrictedBoltzmannMachine` class provides a comprehensive implementation optimized for CPU usage with minimal computational requirements.

### Key Features

- **Optimized Matrix Operations**: Uses NumPy for efficient matrix computations.
- **Mini-Batch Training**: Processes data in small batches to manage memory usage.
- **Contrastive Divergence**: Implements CD-k algorithm (default k=1) for efficient training.
- **State Tracking**: Maintains history of model states for visualization and analysis.
- **Anomaly Detection**: Provides methods for detecting anomalies through reconstruction error or free energy.
- **Generative Capabilities**: Includes "dreaming" functionality to generate samples from the learned distribution.

### Architecture Components

#### Initialization and Configuration

The RBM is initialized with parameters that control its structure and learning behavior:

```python
def __init__(
    self,
    n_visible: int,
    n_hidden: int,
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0.0001,
    batch_size: int = 10,
    use_binary_states: bool = False,
    track_states: bool = True,
    max_tracked_states: int = 50
)
```

Key parameters include:
- `n_visible`: Number of visible units (input features)
- `n_hidden`: Number of hidden units (learned features)
- `learning_rate`: Controls the step size during gradient descent
- `momentum`: Adds a fraction of the previous weight update to the current one
- `weight_decay`: L2 regularization coefficient to prevent overfitting
- `use_binary_states`: Whether to use binary states or probabilities

#### Core Operations

The implementation provides several core operations:

1. **Forward Pass** (visible to hidden):
   ```python
   def compute_hidden_probabilities(self, visible_states: np.ndarray) -> np.ndarray
   def sample_hidden_states(self, hidden_probs: np.ndarray) -> np.ndarray
   ```

2. **Backward Pass** (hidden to visible):
   ```python
   def compute_visible_probabilities(self, hidden_states: np.ndarray) -> np.ndarray
   def sample_visible_states(self, visible_probs: np.ndarray) -> np.ndarray
   ```

3. **Training** (contrastive divergence):
   ```python
   def contrastive_divergence(self, batch_data: np.ndarray, k: int = 1) -> float
   def train(self, data: np.ndarray, epochs: int = 10, k: int = 1, ...) -> List[float]
   ```

4. **Evaluation and Analysis**:
   ```python
   def reconstruction_error(self, data: np.ndarray, per_sample: bool = False) -> Union[float, np.ndarray]
   def free_energy(self, data: np.ndarray) -> np.ndarray
   def anomaly_score(self, data: np.ndarray, method: str = 'reconstruction') -> np.ndarray
   def is_anomaly(self, data: np.ndarray, method: str = 'reconstruction') -> np.ndarray
   ```

5. **Generation**:
   ```python
   def dream(self, n_steps: int = 100, start_data: Optional[np.ndarray] = None) -> List[np.ndarray]
   ```

#### Persistence

The implementation includes methods for saving and loading models:

```python
def save(self, filepath: str) -> None
@classmethod
def load(cls, filepath: str) -> 'RestrictedBoltzmannMachine'
```

### Training Process

The training process follows these steps:

1. **Initialization**: Initialize weights with small random values and biases with zeros.
2. **Mini-Batch Processing**: Divide training data into mini-batches.
3. **Contrastive Divergence**:
   - **Positive Phase**: Compute hidden probabilities and states from visible data.
   - **Negative Phase**: Perform k steps of Gibbs sampling to obtain reconstructed data.
   - **Update Parameters**: Adjust weights and biases based on the difference between positive and negative phases.
4. **Regularization**: Apply weight decay to prevent overfitting.
5. **Momentum**: Use momentum to accelerate training and avoid local minima.
6. **Early Stopping**: Optionally stop training when validation error stops improving.

### Anomaly Detection

The RBM can be used for anomaly detection through two methods:

1. **Reconstruction Error**: Samples with high reconstruction error are considered anomalies.
2. **Free Energy**: Samples with unusually low free energy are considered anomalies.

Thresholds for anomaly detection are automatically computed based on the training data.

## PyTorch Implementation

The `RBM` class provides a more concise implementation using PyTorch, which enables potential GPU acceleration.

### Key Features

- **GPU Support**: Can leverage CUDA or MPS for accelerated computation.
- **PyTorch Integration**: Uses PyTorch tensors and operations for efficient computation.
- **Simplified Interface**: Provides a more streamlined API compared to the CPU implementation.

### Architecture Components

#### Initialization

```python
def __init__(
    self,
    visible_size: int,
    hidden_size: int,
    device: str = "cpu"
)
```

The PyTorch implementation is initialized with the network structure and the device to use (CPU, CUDA, or MPS).

#### Core Operations

1. **Forward Pass**:
   ```python
   def forward(self, x)
   ```

2. **Backward Pass**:
   ```python
   def backward(self, h)
   ```

3. **Free Energy Calculation**:
   ```python
   def free_energy(self, v)
   ```

4. **Training**:
   ```python
   def contrastive_divergence(self, v_pos, k=1, learning_rate=0.1)
   def train(self, data, epochs=10, batch_size=10, learning_rate=0.1, k=1)
   ```

### Training Process

The training process is similar to the CPU implementation but leverages PyTorch operations:

1. **Positive Phase**: Compute hidden probabilities and states from visible data.
2. **Negative Phase**: Perform k steps of Gibbs sampling.
3. **Parameter Updates**: Update weights and biases based on the difference between positive and negative phases.

## Usage Patterns

### Feature Extraction

RBMs can be used to extract meaningful features from raw data:

```python
# Train the RBM
rbm = RestrictedBoltzmannMachine(n_visible=784, n_hidden=256)
rbm.train(training_data, epochs=20)

# Extract features
features = rbm.transform(test_data)
```

### Anomaly Detection

RBMs can detect anomalies in data:

```python
# Train the RBM on normal data
rbm = RestrictedBoltzmannMachine(n_visible=784, n_hidden=256)
rbm.train(normal_data, epochs=20)

# Detect anomalies
anomaly_scores = rbm.anomaly_score(test_data)
is_anomaly = rbm.is_anomaly(test_data)
```

### Generative Modeling

RBMs can generate new samples:

```python
# Train the RBM
rbm = RestrictedBoltzmannMachine(n_visible=784, n_hidden=256)
rbm.train(training_data, epochs=20)

# Generate new samples
dream_states = rbm.dream(n_steps=100)
```

## Integration with Ember ML

The RBM implementations integrate with the broader Ember ML framework:

1. **Backend Agnosticism**: The PyTorch implementation can leverage different backends (CPU, CUDA, MPS).
2. **Tensor Operations**: Uses Ember ML's tensor operations for the test suite.
3. **Model Persistence**: Includes save/load functionality compatible with the Ember ML ecosystem.

## Conclusion

The Ember ML project provides two complementary implementations of Restricted Boltzmann Machines:

1. The `RestrictedBoltzmannMachine` class offers a comprehensive, CPU-optimized implementation with extensive features for training, evaluation, anomaly detection, and generation.

2. The `RBM` class provides a more concise PyTorch-based implementation that can leverage GPU acceleration for improved performance.

Both implementations follow the core principles of RBMs while catering to different use cases and performance requirements.