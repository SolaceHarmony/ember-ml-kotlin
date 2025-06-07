# Core Structures

This section documents the fundamental building blocks for creating trainable neural network components within Ember ML.

## Components

### `ember_ml.nn.modules.base_module`

*   **`Parameter`**: A special type of tensor that represents a trainable parameter in a neural network.
    *   `__init__(data, requires_grad)`: Initializes a parameter with initial data (converted to a backend tensor) and a flag indicating if gradients are required.
    *   `data`: Property providing access to the underlying backend tensor data.
    *   `requires_grad`: Property indicating if gradients are required.
    *   `grad`: Attribute to store the gradient (initialized to None).
    *   `__repr__()`: Provides a string representation including shape and dtype.
*   **`BaseModule`**: The base class for all neural network modules in Ember ML. Custom modules should subclass this.
    *   `__init__()`: Initializes internal dictionaries for storing parameters (`_parameters`), submodules (`_modules`), and buffers (`_buffers`). Sets `training` mode to True and `built` flag to False.
    *   `forward(*args, **kwargs)`: Abstract method that defines the computation performed by the module. Must be overridden by subclasses.
    *   `build(input_shape)`: Method called before the first `forward` pass to create weights based on input shape. Subclasses should override this and set `self.built = True`.
    *   `__call__(*args, **kwargs)`: Handles deferred building (calls `build` if not already built and inputs are provided) and then calls the `forward` method.
    *   `register_parameter(name, param)`: Registers a `Parameter` with the module.
    *   `register_buffer(name, buffer)`: Registers a buffer (non-parameter tensor state) with the module.
    *   `add_module(name, module)`: Registers a submodule.
    *   `named_parameters(prefix, recurse)`: Iterator over module parameters, yielding name and parameter.
    *   `parameters(recurse)`: Returns a list of module parameters.
    *   `named_buffers(prefix, recurse)`: Iterator over module buffers, yielding name and buffer.
    *   `buffers(recurse)`: Iterator over module buffers.
    *   `named_modules(prefix, memo)`: Iterator over all modules in the network.
    *   `modules()`: Iterator over all modules in the network.
    *   `train(mode)`: Sets the module and its submodules to training or evaluation mode.
    *   `eval()`: Sets the module and its submodules to evaluation mode.
    *   `to(device, dtype)`: Moves and/or casts parameters and buffers to a specified device and/or data type.
    *   `__repr__()`: Provides a string representation of the module structure.
    *   `__setattr__(name, value)`: Custom attribute setting to automatically register `Parameter` and `BaseModule` instances.
    *   `__getattr__(name)`: Custom attribute getting to retrieve parameters, buffers, or submodules.
    *   `zero_grad()`: Sets gradients of all parameters to zero.
    *   `get_config()`: Returns the module's configuration (should be overridden by subclasses).
    *   `from_config(config)`: Class method to create a module instance from a configuration dictionary.