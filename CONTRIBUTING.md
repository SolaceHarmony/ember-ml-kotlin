# Contributing to Ember ML

Thank you for your interest in contributing to Ember ML! This guide will help you understand our development practices and how to maintain the architectural integrity of the framework.

## Core Principles

Ember ML is built on a few key principles that ensure it works consistently across different computational backends:

1. **Backend Agnosticism**: All frontend code must be backend-agnostic, using the ops abstraction layer
2. **Clean Architecture**: Strict separation between frontend abstractions and backend implementations
3. **Consistent Experience**: Providing a unified API regardless of the underlying backend

## Backend Abstraction Architecture

Ember ML uses a clean separation between frontend abstractions and backend implementations:

```
ember_ml/
├── ops/           # Frontend abstractions (math, tensor, random operations)
│   ├── interfaces/  # Abstract interfaces defining the API
│   └── ...
├── backend/       # Backend implementations
│   ├── numpy/     # NumPy backend
│   ├── torch/     # PyTorch backend
│   ├── mlx/       # MLX backend
│   └── ember/     # Ember backend
└── ...            # Other modules
```

### Key Architectural Rules

1. **Frontend-Only Rule**: Frontend code must NEVER contain backend-specific implementations
2. **Backend-Only Rule**: Backend implementations must ONLY reside in the backend directory
3. **Abstraction-Only Rule**: All interaction with tensors and neural network components must go through the abstraction layer
4. **No Mixing Rule**: Never mix different backends in the same computation graph
5. **No Circular Dependencies**: Never use frontend ops in backend code

## Backend Purity Guidelines

### Using the Ops Abstraction Layer

Always use the ops abstraction layer for tensor operations:

```python
# ✅ CORRECT
from ember_ml import ops
from ember_ml.nn import tensor

def process_data(data):
    Tensor = tensor.convert_to_tensor(data)
    return ops.sin(Tensor)
```

### Avoiding Direct Backend Usage

Never import or use backend-specific libraries directly in frontend code:

```python
# ❌ INCORRECT
import numpy as np

def process_data(data):
    return np.sin(data)
```

### Using Proper Type Handling

Use tensor.cast() with appropriate dtype instead of Python's built-in type conversion:

```python
# ✅ CORRECT
from ember_ml import ops

def normalize(x):
    x_tensor = tensor.convert_to_tensor(x)
    return ops.divide(x_tensor, tensor.convert_to_tensor(255.0))

# ❌ INCORRECT
def normalize(x):
    return float(x) / 255.0
```

### Using Ops Functions for Operations

Use ops functions for all operations instead of Python operators:

```python
# ✅ CORRECT
from ember_ml import ops

def add_tensors(a, b):
    return ops.add(a, b)

# ❌ INCORRECT
def add_tensors(a, b):
    return a + b
```

## Adding New Functionality

When adding new functionality to Ember ML, follow these steps:

1. **Define the Interface**: Add the new method to the appropriate interface in `ops/interfaces/`
2. **Expose in Frontend**: Add the function to `ops/__init__.py`
3. **Implement in Each Backend**: Add implementations in each backend directory
4. **Test Thoroughly**: Create tests that verify the function works with all backends

## Limited Exceptions

The only permitted exception to backend purity is for visualization/plotting libraries that specifically require NumPy. Even in these cases:

- Thoroughly test to confirm the necessity
- Isolate the NumPy usage to minimize its scope
- Clearly document why direct NumPy usage is necessary

## Testing Guidelines

- Test against the frontend abstraction layer to exercise the full stack
- Verify behavior across all supported backends
- Test edge cases and error conditions
- Ensure consistent behavior regardless of the backend

## Documentation Standards

- Use Google-style docstrings for all functions, classes, and modules
- Include type annotations for all function parameters and return values
- Document any non-obvious behavior or implementation details

## Code Style

- Follow PEP 8 for Python code style
- Group imports (standard library, third-party, internal)
- Use meaningful variable and function names
- Keep functions focused on a single responsibility

## Getting Help

If you're unsure about how to implement something in a backend-agnostic way, please:

1. Check the existing code for similar patterns
2. Ask in our community channels
3. Open a discussion issue on GitHub

We're here to help you contribute successfully to Ember ML!

Thank you for helping make Ember ML better!