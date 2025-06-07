# Architecture Documentation

This section provides comprehensive documentation on the architecture of Ember ML, including design principles, component organization, and implementation patterns.

## Core Architecture Documents

- [Ember ML Architecture](architecture.md): Comprehensive overview of the Ember ML architecture, including core design principles, component organization, and implementation patterns
- [Function-First Design](function_first_design.md): Detailed explanation of the function-first design pattern used in Ember ML, focusing on how it enables memory optimization and efficient code organization

## Component Architecture

- [Neural Network Modules](neural_network_modules.md): Detailed documentation on the neural network modules in Ember ML, including Neural Circuit Policies (NCPs) and their implementation
- [RBM Architecture](rbm_architecture.md): Documentation on the Restricted Boltzmann Machine architecture in Ember ML
- [Wiring and Neuron Maps](wiring.md): Documentation on connectivity patterns and neuron mapping.
- [Attention Mechanisms](attention/index.md): Architecture of attention components.
- [Liquid Neurons](liquidneurons/index.md): Architecture of liquid neuron implementations.
- [Miscellaneous Components](misc/index.md): Architecture details for other components.

## Tensor Architecture

- [Tensor Architecture](tensor_architecture.md): Detailed explanation of the tensor architecture within Ember ML.
- [Tensor Implementation Guide](../plans/tensor_implementation/tensor_implementation_guide.md): Comprehensive guide for implementing tensor operations across all backends
- [MLX Tensor Guide](../plans/tensor_implementation/mlx_tensor_guide.md): MLX-specific tensor implementation guide
- [NumPy Tensor Guide](../plans/tensor_implementation/numpy_tensor_guide.md): NumPy-specific tensor implementation guide
- [PyTorch Tensor Guide](../plans/tensor_implementation/torch_tensor_guide.md): PyTorch-specific tensor implementation guide

## Specific Topics

- [Bizarromath Integration](bizarromath_integration.md): Details on integrating the Bizarromath library.
- [Bizarromath Concepts](bizarromath.md): Explanation of Bizarromath concepts used.
- [Gradient Tape (MLX)](gradient_tape_mlx.md): Documentation on gradient tape implementation for the MLX backend.

## Design Principles

The architecture of Ember ML is guided by several key design principles:

### 1. Function-First Design Pattern

The most distinctive aspect of Ember ML is its function-first design pattern, particularly in the tensor operations framework:

- **Standalone Functions**: Each operation is implemented first as a standalone function
- **Method Wrappers**: Class methods are thin wrappers around these standalone functions
- **Self as First Argument**: The class instance (`self`) is passed as the first argument to the function
- **Consistent API**: The function and method interfaces maintain the same parameter order and names

For more details, see the [Function-First Design](function_first_design.md) document.

### 2. Backend Abstraction

Ember ML supports multiple computational backends through a clean abstraction layer:

- **Backend Agnosticism**: The frontend code is completely agnostic of the backend implementation
- **Backend Purity**: Each backend implementation is kept pure, with no direct imports of backend-specific libraries in the frontend code
- **Backend Switching**: The backend can be switched at runtime, with tensors automatically converted to the new backend
- **Consistent API**: The same API is available regardless of the backend

### 3. Modular Component Architecture

The framework is organized into modular components that can be combined to create complex models:

- **Base Module System**: Foundation for building neural network components
- **Cell-Based Recurrent Networks**: Various RNN implementations with a consistent interface
- **Neural Circuit Policies**: Biologically-inspired neural networks with custom wiring
- **Tensor Operations**: Comprehensive set of tensor operations with a consistent API

For more details, see the [Ember ML Architecture](ember_ml_architecture.md) document.

## Implementation Patterns

Several implementation patterns are used consistently throughout the framework:

### 1. Function-First Implementation

Each tensor operation follows this implementation pattern:

```python
# In ember_ml/backend/{backend_name}/tensor/ops/{module}.py
def operation_name(tensor_obj, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        tensor_obj: The tensor object (instance of the backend's tensor class)
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    # Implementation
    return result
```

### 2. Method as Passthrough

The class methods are thin wrappers that call the standalone functions:

```python
# In ember_ml/backend/{backend_name}/tensor/tensor.py
def operation_name(self, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    from ember_ml.backend.{backend_name}.tensor.ops.{module} import operation_name as op_func
    return op_func(self, *args, **kwargs)
```

For more implementation patterns, see the [Ember ML Architecture](ember_ml_architecture.md) document.