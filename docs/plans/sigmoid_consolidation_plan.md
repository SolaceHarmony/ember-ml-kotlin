# Sigmoid Function Consolidation Plan

## Current State Analysis

1. **Core Abstraction Layer**:
   - `ember_ml/ops/__init__.py` defines `sigmoid = lambda *args, **kwargs: math_ops().sigmoid(*args, **kwargs)`
   - This is the correct, backend-agnostic implementation that delegates to the appropriate backend

2. **Backend-specific Implementations**:
   - Each backend has its own implementation in its respective math_ops.py file:
     - `ember_ml/backend/torch/math_ops.py`: Uses `torch.sigmoid`
     - `ember_ml/backend/numpy/math_ops.py`: NumPy implementation
     - `ember_ml/backend/mlx/math_ops.py`: Uses `mx.sigmoid`

3. **Neural Network Layer/Activation**:
   - `ember_ml/nn/activation.py`: Contains a `Sigmoid` class that uses `K.sigmoid(x)` in its forward method
   - This file is not part of the intended design and needs to be migrated to the proper structure

4. **Problematic Implementations**:
   - `ember_ml/utils/math_helpers.py`: Contains a direct NumPy implementation of sigmoid (should be removed)
   - Direct framework calls (e.g., `torch.sigmoid()`, `mx.sigmoid()`) throughout the codebase
   - `from ember_ml import backend as K` usage, which violates the backend purity principle

## Issues to Address

1. **Redundant Math Helpers**: The `math_helpers.py` file contains functionality that should be covered by the ops abstraction layer.
2. **Direct Framework Calls**: Many places in the code directly call framework-specific functions.
3. **Backend as K**: Some code uses `from ember_ml import backend as K`, which bypasses the ops abstraction layer.
4. **Multiple Implementations**: Having multiple sigmoid implementations can lead to inconsistencies and maintenance issues.
5. **Improper Module Structure**: The Sigmoid implementation is in `nn/activation.py` but should be in the `nn/activations` module with proper interfaces and common implementations.

## Consolidation Strategy

### 1. Standardize on `ops.sigmoid`

The central `ops.sigmoid` function should be the only way to access sigmoid functionality in the codebase. This function already correctly delegates to the appropriate backend implementation.

### 2. Remove Redundant Math Helpers

The `math_helpers.py` file should be removed entirely, as its functionality should be covered by the ops abstraction layer in `math_ops.py` and the corresponding backend implementations.

### 3. Create Proper Activation Module Structure

Following the project's architecture, we need to:

1. Create a Sigmoid implementation in `ember_ml/nn/activations/common`:

```python
# ember_ml/nn/activations/common/sigmoid.py
from ember_ml import ops
from ember_ml.ops.tensor import EmberTensor
from ember_ml.nn.activations.interfaces.activation import ActivationInterface

class Sigmoid(ActivationInterface):
    """
    Applies the Sigmoid function element-wise:
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    
    def forward(self, x: EmberTensor) -> EmberTensor:
        """
        Forward pass of the Sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid activation applied
        """
        return ops.sigmoid(x)
    
    def __repr__(self):
        return "Sigmoid()"
```

3. Update `ember_ml/nn/activations/__init__.py` to expose the Sigmoid class:

```python
"""
Activation functions for neural networks.

This module provides various activation functions that can be used
in neural network components.
"""

from ember_ml.nn.activations.common.tanh import Tanh
from ember_ml.nn.activations.common.softmax import Softmax
from ember_ml.nn.activations.common.dropout import Dropout
from ember_ml.nn.activations.common.sigmoid import Sigmoid

__all__ = [
    'Tanh',
    'Softmax',
    'Dropout',
    'Sigmoid',
]
```

### 4. Decommission Factory Pattern

The current factory pattern in `ember_ml/nn/factory.py` should be decommissioned rather than fixed, as it uses direct backend access which violates the backend purity principle. Instead, activations should be instantiated directly from the `nn.activations` module:

```python
from ember_ml.nn.activations import Sigmoid

# Use the activation directly
sigmoid_activation = Sigmoid()
```

### 5. Replace Direct Framework Calls

Search for and replace all direct framework calls (e.g., `torch.sigmoid()`, `mx.sigmoid()`) with `ops.sigmoid()`.

### 6. Replace Backend as K Usage

Search for and replace all instances of `from ember_ml import backend as K` with `from ember_ml import ops`, and update the corresponding function calls.

## Implementation Plan

1. **Create New Module Structure**:
   - Create `ember_ml/nn/activations/common/sigmoid.py`
   - Update `ember_ml/nn/activations/__init__.py`
2. **Fix Core Issues**:
   - Remove `math_helpers.py` and ensure all its functionality is covered by the ops abstraction layer
   - Decommission `factory.py` and update code to use activations directly from the `nn.activations` module

3. **Systematic Replacement**:
   - Replace direct framework calls with `ops.sigmoid()`
   - Replace `K.sigmoid()` with `ops.sigmoid()`

4. **Remove Old Implementation**:
   - Remove `ember_ml/nn/activation.py` entirely

5. **Testing**:
   - Ensure all tests pass after the changes
   - Add tests for the new Sigmoid implementation

6. **Documentation**:
   - Update documentation to emphasize the use of `ops.sigmoid()` as the standard way to access sigmoid functionality
   - Document the new module structure

## Specific Files to Modify

1. **New Files to Create**:
   - `ember_ml/nn/activations/common/sigmoid.py`

2. **Files to Update**:
   - `ember_ml/nn/activations/__init__.py`: Add Sigmoid to imports and __all__
   - `ember_ml/nn/factory.py`: Mark as deprecated with a warning message directing users to use activations directly from the `nn.activations` module

3. **Files to Remove**:
   - `ember_ml/utils/math_helpers.py`: Remove this file as its functionality should be covered by the ops abstraction layer
   - `ember_ml/nn/activation.py`: Remove this file as it's being replaced by the proper module structure in `nn/activations`

4. **Files with Direct Framework Calls to Fix**:
   - Various files using direct calls to `torch.sigmoid()`, `mx.sigmoid()`, etc.
   - Files using `K.sigmoid()`

## Validation

After making these changes, we should run EmberLint on all modified files to ensure they comply with the project's coding standards:

```bash
python utils/emberlint.py path/to/modified/file.py --verbose
```

## Benefits

1. **Backend Purity**: All code will be backend-agnostic, using the ops abstraction layer.
2. **Consistency**: There will be a single, consistent way to access sigmoid functionality.
3. **Maintainability**: Changes to sigmoid implementation will only need to be made in one place.
4. **Testability**: Tests can focus on a single implementation.
5. **Architectural Alignment**: The solution will align with the proper module structure of the project.
6. **Reduced Redundancy**: Removing redundant implementations reduces code duplication and potential inconsistencies.