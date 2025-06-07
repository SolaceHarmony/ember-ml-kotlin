# Ember ML Documentation

Welcome to the Ember ML documentation. Ember ML is a hardware-optimized neural network library that supports multiple backends (PyTorch, MLX, NumPy) to run efficiently on different hardware platforms (CUDA, Apple Metal, and other platforms).

## Documentation Sections

- [API Reference](api/index.md): Detailed API documentation for all modules
- [Tutorials](tutorials/index.md): Step-by-step guides for common tasks
- [Examples](examples/index.md): Code examples and use cases
- [Plans](plans/): Development plans and roadmaps

## Quick Start

### Installation

```bash
pip install ember-ml
```

### Basic Usage

```python
import ember_ml as eh
from ember_ml import ops

# Set the backend (optional, auto-selects by default)
eh.set_backend('torch')  # or 'numpy', 'mlx'

# Create a liquid neural network
model = eh.models.LiquidNeuralNetwork(
    input_size=10,
    hidden_size=32,
    output_size=1
)

# Create input tensor
x = ops.random.normal(shape=(100, 10))

# Forward pass
output = model(x)
```

For more detailed instructions, see the [Getting Started](tutorials/getting_started.md) guide.

## Key Features

- **Backend Abstraction**: Automatically selects the optimal computational backend (MLX, PyTorch, or NumPy)
- **Neural Network Architectures**: Implementation of cutting-edge neural network architectures like LTC, NCP, and more
- **Feature Extraction**: Tools for extracting features from large datasets
- **Hardware Optimization**: Optimized for different hardware platforms (CUDA, Apple Metal, etc.)

## Core Components

### Neural Network Architectures

The project implements various cutting-edge neural network architectures:

- Liquid Neural Networks (LNN): Dynamic networks with adaptive connectivity
- Neural Circuit Policies (NCP): Biologically-inspired neural architectures
- Stride-Aware Continuous-time Fully Connected (CfC) networks
- Specialized attention mechanisms and temporal processing units

### Multi-Backend Support

The project implements backend-agnostic tensor operations that can use different computational backends:

- MLX (optimized for Apple Silicon)
- PyTorch (for CUDA and other GPU platforms)
- NumPy (for CPU computation)
- Future support for additional backends

### Feature Extraction

The project includes tools for extracting features from large datasets, including:

- `TerabyteFeatureExtractor`: Extracts features from large datasets
- `TemporalStrideProcessor`: Processes temporal data with variable strides

## Getting Help

If you encounter any issues or have questions:

1. Check the tutorials and examples in this documentation
2. Search for similar issues in the GitHub repository
3. Ask a question in the Discussion forum

## License

Ember ML is released under the MIT License.