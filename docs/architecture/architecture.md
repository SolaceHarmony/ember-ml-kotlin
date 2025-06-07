# Ember ML Architecture

## Overview

Ember ML is a modern machine learning library designed for hardware-optimized neural networks with multi-backend support. The project implements various cutting-edge neural network architectures and provides a flexible, backend-agnostic tensor operations framework that can run efficiently on different hardware platforms (CUDA, Apple Metal, and other platforms).

## Core Design Principles

Ember ML is built on several key design principles that guide its implementation:

### 1. Function-First Design Pattern and absolute agnosticism to the backend

The most distinctive aspect of Ember ML is its function-first design pattern, particularly in the tensor operations framework:

- **Standalone Functions**: Each operation is implemented first as a standalone function.
- **Method Wrappers**: Class methods are thin wrappers around these standalone functions. (nn.tensor.EmberTensor and backend tensors backend.mlx.tensor.MLXTensor)

### 2. Backend Abstraction

Ember ML supports multiple computational backends through a clean abstraction layer:

- **Backend Agnosticism**: The frontend code is completely agnostic of the backend implementation with the only exception being the __init__ as they dynamically load the backend functions. __init__.pyi files keep linters happy.
- **Backend Purity**: Each backend implementation is kept pure, with no direct imports of backend-specific libraries in the frontend code and no NumPy whatsoever. Exceptions are made in *extreme* cases like needing to have a ".to_numpy()" function. We do not mix backend tensors and no backend knows about the others. utils/emberlint.py was created to check purity and linting.
- **Backend Switching**: The backend can be switched at runtime using ops.set_backend. This changes the return values of the backend tensors.
- **Consistent API**: The same API is available regardless of the backend.
- **NumPy forbidden on front-end** Absolute purity from NumPy is enforced.

### 3. Modular Component Architecture

The framework is organized into modular components that can be combined to create complex models. Key components include:

- **Base Module System**: The foundation for all neural network components, providing core functionality like parameter management.
- **Neuron Maps**: Configuration objects that define the connectivity patterns and structure of wired neural networks.
- **Neural Network Modules**: Implementations of neural network layers and components. This includes:
    *   **Wired Modules**: Layers (like `NCP`, `LTC`, `CfC`, `LQNet`, `CTRQNet`) that utilize Neuron Maps to define their architecture and dynamics. The implementation of these modules has moved away from a cell-based architecture, integrating functionality directly into the layer classes.
    *   **Spatially Aware Wiring**: Novel 3 dimensional connectomes through enhanced versions of Module and NCP exist.
    *   **Standard Modules**: Various other layers and components (like standard RNNs, LSTMs, GRUs, and Dense layers) that may or may not use Neuron Maps. A NeuronMap is optional. se_cfc.py is a spatially aware neuron.
- **Tensor Operations**: A comprehensive set of backend-agnostic tensor operations in ember_ml.nn.tensor

### 4. Memory Optimization

A key aspect of Ember ML's design is memory optimization, particularly in how abstract classes and backend functions are implemented:

- **Separation of Functions from Class Implementations**: Functions are defined separately from classes and then imported when needed. This allows for lazy loading, reduced memory footprint, and efficient garbage collection.
- **Backend Function Implementation**: Backend functions are implemented to minimize memory usage by avoiding unnecessary copies, using in-place operations where appropriate, and reusing temporary buffers.
- **Abstract Class Design**: Abstract classes define interfaces without implementing functionality, resulting in minimal memory overhead and allowing implementation flexibility.

### 5. Strong Typing

The framework emphasizes strong typing for improved code reliability and clarity:

Backend: backend.<backendnname>.types(.py) contains the strong typing - this is required import.
Frontend: nn.tensor.types(.py) contains the typing for front-end. 

- **Explicit Type Annotations**: Use explicit type annotations for tensor operations.
- **Type Validation**: Validation of input types before operations.
- **Explicit Rejection**: Explicit rejection of incompatible types with informative error messages.
- **Consistent Conversion**: Consistent conversion of inputs to the appropriate backend tensor type.

## Architecture Components

### Tensor Operations Framework

The tensor operations framework is organized into three main layers:

#### Frontend Layer

The frontend provides a unified interface through `ember_ml.nn.tensor`:

- **EmberTensor**: The main tensor class that users interact with.
- **EmberDType**: A backend-agnostic data type class.
- **Standalone Functions**: Functions for tensor operations that can be called directly.

EmberTensor will be depreciated at some point, so doing this:
from ember_ml.nn import tensor
tensor.convert_to_tensor([],dtype=tensor.int32)

Is the prefered way to create a tensor. All operations return *backend* tensors. This is to allow graphing optimizations on the backend.

#### Backend Abstraction Layer

The backend abstraction layer manages backend selection and switching:

- **Backend Selection**: Functions for selecting and getting the current backend.
- **Backend Registration**: Mechanism for registering new backends.
- **Device Management**: Functions for managing devices (CPU, GPU, etc.).

#### Backend Implementations

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

#### Native Backend Tensor Preservation

A key architectural enhancement is the preservation of native backend tensor types throughout the computation pipeline to maintain GPU acceleration and performance. This is achieved by:

1.  **Direct Function Calls**: Functions in `nn.tensor.__init__` return native backend tensors (e.g., `mx.array` for MLX).
2.  **EmberTensor Methods**: Methods on EmberTensor objects return EmberTensor objects to maintain object type consistency.
3.  **Backend Tensor Extraction**: EmberTensor extracts the native backend tensor before passing to backend functions.
4.  **No EmberTensor in Backend**: Backend functions never receive EmberTensor objects, only native backend tensors.

This approach ensures GPU acceleration is maintained, type safety is preserved, object consistency is maintained, and performance is optimized.

#### Strong Typing Implementation

The strong typing implementation, particularly evident in the MLX backend, provides type safety, clear error messages, explicit conversions, and performance optimization. Key components include explicit type annotations, type validation and conversion, explicit rejection of incompatible types, type-specific function signatures, and consistent type conversion (often via NumPy as a universal intermediate).

#### EmberTensor Python Operator Support

EmberTensor implements numerous dunder methods to allow seamless operations with Python operators, enabling intuitive, Pythonic code while delegating to the underlying backend operations.

### Backend Architecture Details

The backend management system handles backend selection and switching, configuration persistence, device management, and automatic backend detection. Dynamic function aliasing maintains a consistent API across backends and updates function references when the backend changes. Specialized module-level aliasing exists for modules like statistics, linear algebra, and activations. The framework primarily uses a function-based approach with dynamic imports for backend implementations, with some modules employing a hybrid approach combining factory function aliasing for stateful components and direct function aliasing for stateless operations.

### Frontend-Backend Interaction

Frontend calls are routed to appropriate backend implementations, with backend selection handled through environment variables. Frontend code remains pure by never importing backend libraries directly, using only the ops abstraction layer, and maintaining backend-agnostic data types.

### Mathematical Operations

The framework provides a comprehensive set of backend-agnostic mathematical operations, including basic arithmetic, linear algebra, statistical functions, exponential and logarithmic functions, trigonometric functions, activation functions, advanced manipulation and reduction operations, and solver operations.

### Pipeline Architecture

Ember ML is evolving towards a comprehensive pipeline architecture that integrates both neural and non-neural components. The current implementation provides a sequential processing pipeline. The future evolution is towards a more flexible "Tasks with handoffs" approach with integration of NLP and data processing blocks, asynchronous communication and an actor model, and cross-language support. Auto-wiring capabilities will be extended to non-neural tasks, and the pipeline architecture will support mixing different backend implementations.

### Feature Extraction Framework

Ember ML includes a comprehensive feature extraction framework designed to handle various data types and scales. Core components include column-based and terabyte-scale processing tools. The framework provides specialized processing for numeric, categorical, datetime, text, and temporal data, with a backend-agnostic implementation leveraging the backend abstraction.

### Neural Network Framework

The Ember ML neural network framework is built around a modular system that leverages **Neuron Maps** to define the structure of wired networks. A key architectural shift has been the **removal of a separate cell-based architecture** in favor of integrating functionality directly into layer classes, simplifying the codebase and improving maintainability.

1.  **BaseModule (Module)**: The fundamental building block for all neural network layers and modules. It provides essential functionalities such as parameter management, submodule management, and device/dtype handling.
2.  **Neuron Maps**: Defined in `ember_ml.nn.modules.wiring`, Neuron Maps are configuration objects that specify the connectivity pattern (like adjacency matrices) and structural properties of a neural network. They are used by wired modules to determine their architecture. This concept replaces the older "Wiring" terminology. Neuron Maps can also incorporate **spatial properties**, allowing neurons to be embedded in a multi-dimensional space and influencing connectivity based on distance. `EnhancedNeuronMap` and `EnhancedNCPMap` are key classes for defining spatially aware neuron maps.
3.  **Neural Network Modules**: These are `BaseModule`-based layers and components. This includes:
    *   **Wired Modules**: Layers (such as `NCP`, `LTC`, `CfC`, `LQNet`, `CTRQNet`) that take a `NeuronMap` instance as input. These modules implement network dynamics based on the connectivity and properties defined by the Neuron Map. During their deferred `build` phase, they use the Neuron Map to determine the shapes of their trainable parameters (weights and biases) and obtain masks that define the connections between neurons. The forward pass of these modules then applies these parameters and masks using backend-agnostic operations (`ops`) to compute the network's output.
    *   **Standard Modules**: Various other layers and components (like standard RNNs, LSTMs, GRUs, and Dense layers) that may or may not use Neuron Maps.
    *   **Stride-Aware Modules**: Specialized modules for multi-scale time series processing. *Note: The documentation for these modules may still reflect an older architecture pattern using explicit "Cell" and "Wired" classes, which needs to be updated.*
    *   Restricted Boltzmann Machines (RBMs).
    *   Activation functions and Container modules.

The framework is evolving towards a more control-theory friendly architecture with automatic wiring capabilities, further integrating the Neuron Map concept into the design of advanced neural networks.

## Implementation Patterns

Key implementation patterns include:

-   **Function-First Implementation**: Operations implemented as standalone functions in backend directories.
-   **Method as Passthrough**: Class methods acting as thin wrappers around standalone functions.
-   **Frontend Implementation**: Frontend classes like EmberTensor delegating to backend implementations.
-   **Cell-Layer Pattern**: Recurrent networks structured with cell classes for single steps and layer classes for sequences.
-   **Neuron Map Configuration Pattern**: Wired neural networks using explicit Neuron Map configurations to define connectivity and structure. This replaces the older "Wiring Configuration Pattern".
-   **Module Hierarchy Pattern**: Clear inheritance structure for specialized modules.
-   **Feature Extraction Pattern**: Consistent approach to processing different data types with type detection and specialized processing.
-   **Asynchronous Task Pattern**: Emerging pattern for parallel processing and efficient resource utilization.

## Backend Selection and Switching

Ember ML allows for dynamic backend selection and switching through a backend registry, tracking the currently selected backend, and converting tensors between backends when needed.

## Supported Neural Network Architectures

-   Liquid Neural Networks (LNN) - Can be implemented using wired structures defined by Neuron Maps, including spatially aware ones.
-   Neural Circuit Policies (NCP) - Implemented as modules that use Neuron Maps, including spatially aware ones (`EnhancedNCPMap`).
-   Stride-Aware Continuous-time Fully Connected (CfC) networks - Wired implementations use Neuron Maps, including spatially aware ones (`seCfC` utilizes spatially embedded maps).
-   Restricted Boltzmann Machines (RBM)
-   Specialized attention mechanisms and temporal processing units
-   Quantum-Inspired Modules (LQNet, CTRQNet) - Implemented as modules that use Neuron Maps.

## Supported Backends

-   MLX (optimized for Apple Silicon)
-   PyTorch (for CUDA and other GPU platforms)
-   NumPy (for CPU computation)
-   Future support for additional backends

## Future Enhancements

Planned enhancements include operator overloading for EmberTensor, static methods for common tensor operations, a registry system, Mixture of Experts (MoE), FFT Convolution, self-training and self-tuning capabilities, a block architecture, a configuration system, distributed training support, and cross-language support (Swift and Kotlin).

## Best Practices and Common Pitfalls

**Best Practices:**
- Always use the abstraction layer (`ops` module).
- Use `tensor.convert_to_tensor()` for input conversion.
- Use proper type annotations.
- Keep frontend code backend-agnostic.
- Implement backend-specific optimizations behind the abstraction layer.
- Test across all supported backends.

**Common Pitfalls to Avoid:**
- Direct backend imports.
- Using backend-specific features in frontend code.
- Mixing backend types in the same computation.
- Precision-reducing casts.
- Direct Python operators on tensors (prefer `ops` functions or EmberTensor operator overloading).
- Using the outdated "Wiring" terminology instead of "NeuronMap".

## Testing Considerations and Strategy

Testing focuses on backend equivalence (output matching), edge case testing, and type safety verification. The strategy involves verifying operation equivalence across backends, testing edge cases for various operations, and verifying type safety through explicit checks for type mismatches.