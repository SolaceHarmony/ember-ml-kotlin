# Tensor Interfaces and Types

This section documents the abstract interfaces and common type definitions for tensors within Ember ML. These components are crucial for ensuring backend-agnostic tensor operations.

## Core Concepts

Ember ML's tensor system is designed to provide a consistent API for tensor operations regardless of the underlying computation backend (NumPy, PyTorch, MLX). This is achieved through:

*   **Abstract Interfaces:** Defining the expected methods and properties for tensor objects and operations.
*   **Common Implementations:** Providing a base `EmberTensor` class that wraps backend-specific tensors and delegates operations through the `ops` abstraction layer.
*   **Type Definitions:** Standardizing type aliases for clarity and static analysis.

## Components

### Tensor Interfaces (`ember_ml.nn.tensor.interfaces`)

*   **`TensorInterface(ABC)`**: The abstract base class defining the interface for tensor objects. All backend-specific tensor wrappers (like `EmberTensor`) should adhere to this interface.
    *   Includes abstract methods for standard tensor operations and properties (e.g., `__repr__`, `__str__`, `__init__`, `to_backend_tensor`, `__array__`, `__getitem__`, `__setitem__`, `item`, `tolist`, `shape`, `dtype`, `device`, `backend`, `requires_grad`, `detach`, `to_numpy`, creation functions like `zeros`, `ones`, `arange`, `linspace`, manipulation functions like `reshape`, `transpose`, `concatenate`, `stack`, `split`, `expand_dims`, `squeeze`, `tile`, `gather`, `scatter`, `tensor_scatter_nd_update`, `slice`, `sort`, `argsort`, `slice_update`, `pad`, `maximum`, random functions, state serialization (`__getstate__`, `__setstate__`), and iteration (`__iter__`)).
*   **`DTypeInterface(ABC)`**: The abstract base class defining the interface for data type objects and operations. Backend-specific data type implementations should adhere to this interface.
    *   Includes abstract properties for standard data types (e.g., `float32`, `int64`, `bool_`) and abstract methods for data type conversion (e.g., `get_dtype`, `to_dtype_str`, `from_dtype_str`).

### Common Tensor Implementation (`ember_ml.nn.tensor.common.ember_tensor`)

*   **`EmberTensor(TensorInterface)`**: A concrete implementation of `TensorInterface` that wraps a backend-specific tensor and delegates operations.
    *   `__init__(data, dtype, device, requires_grad)`: Initializes by converting input `data` to a backend tensor using `_convert_to_backend_tensor` and stores it internally (`_tensor`). Stores device, requires\_grad, and backend name.
    *   Implements all abstract methods from `TensorInterface` by calling the corresponding backend operation via the `ops` abstraction layer or internal helper functions (e.g., `to_numpy`, `item`, `shape`, `dtype`, `convert_to_tensor`, `cast`, `copy`, `slice`, `sort`, `argsort`, `slice_update`, `pad`, `tensor_scatter_nd_update`, `maximum`, random functions, `tolist`, `scatter`).
    *   `to_backend_tensor()`: Returns the wrapped backend tensor.
    *   `__array__()`: Implements the NumPy array interface by converting the internal backend tensor to a NumPy array using `to_numpy`.
    *   `__setitem__(key, value)`: Implements item assignment using `slice_update`.
    *   `__getstate__()` / `__setstate__()`: Implement serialization by converting the internal tensor to NumPy for storage and converting back on load.
    *   `__iter__()`: Makes the `EmberTensor` iterable by iterating over the internal backend tensor.

### Common Data Types (`ember_ml.nn.tensor.common.dtypes`)

*   **`EmberDType`**: A backend-agnostic data type representation.
    *   `__init__(name)`: Initializes with a string name and attempts to find the corresponding backend-specific dtype.
    *   `name`: Property returning the string name.
    *   `__repr__()` / `__str__()`: String representations.
    *   `__eq__(other)`: Compares equality with another object (EmberDType, string, or backend dtype).
    *   `__call__()`: Returns the backend-specific data type (makes the instance callable).
*   **`_get_backend_dtype()`**: Internal helper function to get the backend-specific DType implementation instance.
*   **`get_dtype(name)`**: Gets a data type by name from the current backend.
*   **`to_dtype_str(dtype)`**: Converts a data type (EmberDType or backend dtype) to a string.
*   **`from_dtype_str(dtype_str)`**: Converts a string to a data type.
*   **`DTypes`**: A class providing dynamic property access to standard backend data types (e.g., `DTypes().float32`).
*   **`dtype`**: A singleton instance of `DTypes` for convenient access (e.g., `dtype.float32`).
*   **Standard DType Instances**: `float32`, `float64`, `int32`, `int64`, `bool_`, `int8`, `int16`, `uint8`, `uint16`, `uint32`, `uint64`, `float16` (instances of `EmberDType`).

### Type Definitions (`ember_ml.nn.tensor.types`)

*   **Type Aliases**: Defines standard type aliases for clarity and static analysis, including `Numeric`, `TensorLike`, `Scalar`, `Vector`, `Matrix`, `Shape`, `ShapeLike`, `DType`, `Device`, `Axis`, `ScalarLike`. These are primarily used for type hinting.