# Ember ML Architecture

This document provides a comprehensive overview of the Ember ML architecture, focusing on the core design principles, component organization, and implementation patterns.

## Core Design Principles

Ember ML is built on several key design principles that guide its implementation:

### 1. Function-First Design Pattern

The most distinctive aspect of Ember ML is its function-first design pattern, particularly in the tensor operations framework:

- **Standalone Functions**: Each operation is implemented first as a standalone function
- **Method Wrappers**: Class methods are thin wrappers around these standalone functions
- **Self as First Argument**: The class instance (`self`) is passed as the first argument to the function
- **Consistent API**: The function and method interfaces maintain the same parameter order and names

This approach offers several benefits:
- **Flexibility**: Operations can be called as functions or methods
- **Consistency**: Unified implementation for each operation
- **Maintainability**: Easier to add new operations or modify existing ones
- **Testability**: Functions can be tested independently
- **Discoverability**: Better IDE support for discovering available operations

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

## Component Organization

### Tensor Operations Framework

The tensor operations framework is organized into several layers:

#### 1. Frontend Layer

The frontend provides a unified interface through `ember_ml.nn.tensor`:

- **EmberTensor**: The main tensor class that users interact with
- **EmberDType**: A backend-agnostic data type class
- **Standalone Functions**: Functions for tensor operations that can be called directly

#### 2. Backend Abstraction Layer

The backend abstraction layer manages backend selection and switching:

- **Backend Selection**: Functions for selecting and getting the current backend
- **Backend Registration**: Mechanism for registering new backends
- **Device Management**: Functions for managing devices (CPU, GPU, etc.)

#### 3. Backend Implementations

Each backend implements the same API through a consistent structure:

```
ember_ml/backend/{backend_name}/tensor/
  ├── __init__.py           # Exports the tensor class and operations
  ├── tensor.py             # Contains the tensor class with method interfaces
  ├── dtype.py              # Contains the data type class
  ├── ops/                  # Directory for operation modules
  │   ├── __init__.py       # Exports all operations
  │   ├── casting.py        # Contains cast() and related functions
  │   ├── creation.py       # Contains zeros(), ones(), etc.
  │   ├── manipulation.py   # Contains reshape(), transpose(), etc.
  │   ├── indexing.py       # Contains slice(), gather(), etc.
  │   ├── utility.py        # Contains convert_to_tensor(), to_numpy(), etc.
  │   └── random.py         # Contains random_normal(), random_uniform(), etc.
```

### Neural Network Framework

The neural network framework is organized into several components:

#### 1. Base Module System

The `BaseModule` class provides the foundation for building neural network components:

- **Parameter Management**: Tracking and updating parameters
- **Module Composition**: Building complex modules from simpler ones
- **Training/Evaluation Modes**: Switching between training and evaluation modes
- **Device and Dtype Handling**: Managing devices and data types

#### 2. Cell-Based Recurrent Networks

The framework includes various recurrent neural network implementations:

- **Basic RNN**: Simple recurrent cells
- **LSTM**: Long Short-Term Memory cells
- **GRU**: Gated Recurrent Units
- **CFC**: Closed-form Continuous-time cells
- **LTC**: Liquid Time-Constant cells
- **Stride-Aware Cells**: Cells for multi-scale temporal processing

These implementations follow a cell-layer pattern:
- **Cell Classes**: Implement single-step computation
- **Layer Classes**: Handle sequence processing

#### 3. Neural Circuit Policies

The framework includes Neural Circuit Policy (NCP) implementations:

- **NCP**: Neural circuit policies with custom wiring
- **AutoNCP**: Automatically configured neural circuit policies

These implementations use wiring configurations to define connectivity patterns.

#### 4. Restricted Boltzmann Machines

The framework includes Restricted Boltzmann Machine (RBM) implementations:

- **CPU-Friendly RBM**: Optimized for CPU computation
- **PyTorch-Based RBM**: Leveraging PyTorch for potential GPU acceleration

## Implementation Patterns

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

### 3. Frontend Implementation

The frontend EmberTensor class delegates to the backend implementation:

```python
# In ember_ml/nn/tensor/common/ember_tensor.py
def operation_name(self, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    # Convert EmberTensor arguments to backend tensors if needed
    backend_args = [arg.to_backend_tensor() if isinstance(arg, EmberTensor) else arg for arg in args]
    backend_kwargs = {k: v.to_backend_tensor() if isinstance(v, EmberTensor) else v for k, v in kwargs.items()}
    
    # Call the backend operation
    result = operation_name(self._tensor, *backend_args, **backend_kwargs)
    
    # Wrap the result in an EmberTensor if needed
    return EmberTensor(result, device=self.device, requires_grad=self._requires_grad)
```

### 4. Cell-Layer Pattern

Recurrent networks follow a cell-layer pattern:

```python
# Cell implementation
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Initialize parameters
        
    def forward(self, input, state):
        # Implement single-step computation
        return output, new_state

# Layer implementation
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        # Create cells
        self.cells = [LSTMCell(input_size, hidden_size) for _ in range(num_layers)]
        
    def forward(self, input_sequence):
        # Process sequence through cells
        return sequence_output
```

### 5. Wiring Configuration Pattern

Neural circuit policies use wiring configurations:

```python
# Create wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5
)

# Create NCP with wiring
model = NCP(wiring=wiring)
```

## Memory Optimization

A key aspect of Ember ML's design is memory optimization, particularly in how abstract classes and backend functions are implemented:

### 1. Separation of Functions from Class Implementations

Functions are defined separately from classes and then imported when needed:

```python
# In ember_ml/backend/{backend_name}/tensor/tensor.py
def operation_name(self, *args, **kwargs):
    from ember_ml.backend.{backend_name}.tensor.ops.{module} import operation_name as op_func
    return op_func(self, *args, **kwargs)
```

This approach has several memory benefits:
- **Lazy Loading**: Functions are only loaded when they're actually used
- **Reduced Memory Footprint**: The class definition itself remains small
- **Garbage Collection**: Functions that aren't being used can be garbage collected

### 2. Backend Function Implementation

Backend functions are implemented to minimize memory usage:

- **No Unnecessary Copies**: Operations avoid creating unnecessary copies of tensors
- **In-place Operations**: Where appropriate, operations modify tensors in-place
- **Memory Reuse**: Temporary buffers are reused where possible

### 3. Abstract Class Design

Abstract classes define interfaces without implementing functionality:

```python
class TensorInterface(ABC):
    @abstractmethod
    def zeros(self, shape, dtype=None, device=None):
        pass
        
    @abstractmethod
    def ones(self, shape, dtype=None, device=None):
        pass
        
    # Other abstract methods
```

This approach allows:
- **Minimal Memory Overhead**: Abstract classes have minimal memory footprint
- **Implementation Flexibility**: Concrete implementations can optimize for their specific backend
- **Clear Contracts**: The interface clearly defines what methods must be implemented

## Backend Selection and Switching

Ember ML allows for dynamic backend selection and switching:

```python
from ember_ml.backend import set_backend, get_backend

# Set the backend to PyTorch
set_backend('torch')

# Create a tensor using PyTorch backend
x = EmberTensor([1, 2, 3])

# Switch to MLX backend
set_backend('mlx')

# x is automatically converted to MLX backend
y = x + 1  # Uses MLX operations
```

This is implemented through:
- **Backend Registry**: A registry of available backends
- **Current Backend Tracking**: Tracking the currently selected backend
- **Tensor Conversion**: Converting tensors between backends when needed

## Planned Enhancements

### 1. Operator Overloading

A planned enhancement to the tensor operations framework is the implementation of operator overloading for EmberTensor:

```python
# Planned operator overloading implementation
def __add__(self, other):
    from ember_ml import ops
    if isinstance(other, EmberTensor):
        other = other.to_backend_tensor()
    result = ops.add(self._tensor, other)
    return EmberTensor(result, device=self.device, requires_grad=self._requires_grad)

def __sub__(self, other):
    from ember_ml import ops
    if isinstance(other, EmberTensor):
        other = other.to_backend_tensor()
    result = ops.subtract(self._tensor, other)
    return EmberTensor(result, device=self.device, requires_grad=self._requires_grad)

def __mul__(self, other):
    from ember_ml import ops
    if isinstance(other, EmberTensor):
        other = other.to_backend_tensor()
    result = ops.multiply(self._tensor, other)
    return EmberTensor(result, device=self.device, requires_grad=self._requires_grad)
```

This will enable more intuitive tensor operations:

```python
# Current approach
c = ops.add(a, b)

# Planned approach
c = a + b
```

The operator overloading will maintain the function-first design pattern by delegating to the existing standalone functions, ensuring consistency and maintainability.

### 2. Static Methods

Another planned enhancement is the implementation of static methods for common tensor operations:

```python
# Current approach
tensor_obj = EmberTensor([0])
a = tensor_obj.zeros((2, 3))

# Planned approach
a = EmberTensor.zeros((2, 3))
```

This will provide a more intuitive API for tensor creation and manipulation, while still maintaining the function-first design pattern.

## Conclusion

The Ember ML architecture is designed for flexibility, consistency, and performance. The function-first design pattern, backend abstraction, and modular component architecture enable a clean, maintainable codebase that can easily adapt to new requirements and backends.

The separation of functions from class implementations and the careful design of abstract classes contribute to memory efficiency, allowing the framework to handle large models and datasets without excessive memory usage.