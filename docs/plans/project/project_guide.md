# Project Guide

This document provides a comprehensive overview of the Ember ML project, including its purpose, features, installation instructions, and contribution guidelines.

## 1. Project Overview

Ember ML is a machine learning framework designed to provide a flexible, backend-agnostic API for tensor operations and neural network components. It supports multiple backends (NumPy, PyTorch, MLX) and provides a consistent API across all of them.

### 1.1 Key Features

- **Backend Agnostic**: Works with multiple backends (NumPy, PyTorch, MLX)
- **Consistent API**: Provides a consistent API across all backends
- **Flexible Tensor Operations**: Supports a wide range of tensor operations
- **Neural Network Components**: Provides building blocks for neural networks
- **Biologically-Inspired Architectures**: Supports biologically-inspired neural network architectures

### 1.2 Project Structure

The project is organized into the following main components:

- **ember_ml/nn/tensor**: Tensor operations and data types
- **ember_ml/nn/wirings**: Wiring patterns for neural networks
- **ember_ml/backend**: Backend-specific implementations
- **ember_ml/core**: Core functionality and base classes
- **ember_ml/models**: Pre-defined model architectures
- **ember_ml/training**: Training utilities and algorithms
- **ember_ml/utils**: Utility functions and helpers

## 2. Installation

### 2.1 Requirements

- Python 3.8 or higher
- NumPy, PyTorch, or MLX (depending on the backend you want to use)

### 2.2 Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ember-ml.git
   cd ember-ml
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Install the required backend:
   ```bash
   # For NumPy backend
   pip install numpy

   # For PyTorch backend
   pip install torch

   # For MLX backend
   pip install mlx
   ```

## 3. Usage

### 3.1 Basic Usage

```python
import ember_ml
from ember_ml.nn.tensor import zeros, ones, cast, float32

# Set the backend
ember_ml.backend.set_backend('numpy')  # or 'torch', 'mlx'

# Create tensors
x = zeros((2, 3), dtype=float32)
y = ones((2, 3), dtype=float32)

# Perform operations
z = x + y
z = cast(z, float32)
```

### 3.2 Creating Models

```python
from ember_ml.nn.tensor import EmberTensor
from ember_ml.models import LSTM

# Create a model
model = LSTM(units=64, return_sequences=True)

# Process input
x = EmberTensor(...)  # Input tensor
output = model(x)
```

### 3.3 Training Models

```python
from ember_ml.training import Trainer

# Create a trainer
trainer = Trainer(model, loss='mse', optimizer='adam')

# Train the model
trainer.fit(x_train, y_train, epochs=10, batch_size=32)
```

## 4. Contribution Guidelines

### 4.1 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Write docstrings for all functions, classes, and modules
- Use type hints where appropriate

### 4.2 Pull Request Process

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

### 4.3 Testing

- Write unit tests for all new functionality
- Run existing tests to ensure your changes don't break existing functionality
- Use pytest for running tests

### 4.4 Documentation

- Update documentation for all new functionality
- Write clear and concise documentation
- Include examples where appropriate

### 4.5 Code Review

- All pull requests will be reviewed by at least one maintainer
- Address all comments and suggestions from reviewers
- Make requested changes and push them to your branch

## 5. Development Roadmap

### 5.1 Current Focus

- Refactoring tensor operations to support both function and method calling patterns
- Ensuring frontend compatibility with the refactored backend
- Implementing dynamic dtype properties
- Fixing wiring module issues

### 5.2 Future Plans

- Adding more cell types
- Adding more wiring patterns
- Supporting dynamic wiring
- Adding visualization tools
- Improving performance
- Adding more examples and tutorials

## 6. Resources

- [GitHub Repository](https://github.com/yourusername/ember-ml)
- [Documentation](https://ember-ml.readthedocs.io)
- [Issue Tracker](https://github.com/yourusername/ember-ml/issues)
- [Discussion Forum](https://github.com/yourusername/ember-ml/discussions)