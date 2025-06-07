# Activation Modules (nn.modules.activations)

The `ember_ml.nn.modules.activations` package provides both **activation modules** (classes inheriting from `Module`) and dynamically aliased **functional activations** (plain functions). These are backend-agnostic, drawing their implementation from the currently active backend (e.g., MLX, PyTorch, NumPy).

## Importing

```python
# Import Module classes (e.g., for Sequential)
from ember_ml.nn.modules.activations import ReLU, Tanh, Sigmoid, Softmax, Softplus, LeCunTanh, Dropout

# Import functional activations (e.g., for internal use in custom modules)
from ember_ml.nn.modules import activations
# Examples: activations.relu, activations.tanh, activations.sigmoid

# Import the activation getter function
from ember_ml.nn.modules.activations import get_activation
```

## Available Activation Modules

### ReLU

`ReLU` implements the Rectified Linear Unit activation function.

```python
from ember_ml.nn.modules.activations import ReLU
from ember_ml.nn import tensor

relu_activation = ReLU()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0, 2.0])
output = relu_activation(input_tensor) # Output: [0., 0., 1., 2.]
```

### Tanh

`Tanh` implements the Hyperbolic Tangent activation function.

```python
from ember_ml.nn.modules.activations import Tanh
from ember_ml.nn import tensor

tanh_activation = Tanh()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = tanh_activation(input_tensor) # Output: [-0.76159, 0.        , 0.76159]
```

### Sigmoid

`Sigmoid` implements the Sigmoid activation function.

```python
from ember_ml.nn.modules.activations import Sigmoid
from ember_ml.nn import tensor

sigmoid_activation = Sigmoid()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = sigmoid_activation(input_tensor) # Output: [0.26894, 0.5      , 0.73105]
```

### Softmax

`Softmax` implements the Softmax activation function, typically used for multi-class classification output layers.

```python
from ember_ml.nn.modules.activations import Softmax
from ember_ml.nn import tensor

softmax_activation = Softmax(axis=-1)
input_tensor = tensor.convert_to_tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
output = softmax_activation(input_tensor)
# Output: [[0.09003, 0.24472, 0.66524],
#          [0.33333, 0.33333, 0.33333]]
```

**Arguments:**
*   `axis` (int): The axis along which the softmax normalization is applied. Defaults to -1.

### Softplus

`Softplus` implements the Softplus activation function: `log(exp(x) + 1)`.

```python
from ember_ml.nn.modules.activations import Softplus
from ember_ml.nn import tensor

softplus_activation = Softplus()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = softplus_activation(input_tensor) # Output: [0.31326, 0.69314, 1.31326]
```

### LeCunTanh

`LeCunTanh` implements the scaled hyperbolic tangent activation function proposed by LeCun et al. (1998): `1.7159 * tanh(2/3 * x)`.

```python
from ember_ml.nn.modules.activations import LeCunTanh
from ember_ml.nn import tensor

lecun_tanh_activation = LeCunTanh()
input_tensor = tensor.convert_to_tensor([-1.0, 0.0, 1.0])
output = lecun_tanh_activation(input_tensor) # Output: [-1.000..., 0.       , 1.000...] (approx)
```

### Dropout

`Dropout` implements the dropout regularization technique.

Dropout randomly sets a fraction `rate` of input units to 0 during training updates, helping prevent overfitting. Units not zeroed are scaled up by `1 / (1 - rate)` to maintain the expected sum. Dropout is only active when the `training=True` argument is passed to the `forward` call.

```python
from ember_ml.nn.modules.activations import Dropout
from ember_ml.nn import tensor

dropout_layer = Dropout(rate=0.5, seed=42)
input_tensor = tensor.ones((2, 2))

# During training
output_train = dropout_layer(input_tensor, training=True)
# Example Output (stochastic): [[2., 0.], [0., 2.]]

# During inference
output_eval = dropout_layer(input_tensor, training=False)
# Output: [[1., 1.], [1., 1.]]
```

**Arguments:**
*   `rate` (float): Fraction of input units to drop (0 <= rate < 1).
*   `seed` (Optional[int]): Seed for the random number generator for reproducibility.

## Activation Modules vs. Functional Activations

Ember ML provides two ways to use activations:

1.  **Activation Modules (Classes):**
    *   Instances of classes like `ReLU()`, `Tanh()`, etc.
    *   Inherit from `ember_ml.nn.modules.Module`.
    *   Primarily intended for use as distinct layers within model structures, especially `Sequential` containers.
    *   Manage their own state (though most activations are stateless).

2.  **Functional Activations (Functions):**
    *   Plain functions like `activations.relu`, `activations.tanh`, etc.
    *   Dynamically aliased from the active backend's implementation into the `ember_ml.nn.modules.activations` namespace.
    *   Intended for direct use *within* the `forward` method of complex modules where applying an activation is part of an internal computation, not a separate layer.

### Choosing Between Modules and Functions

*   Use **Activation Modules** when adding an activation step between distinct layers (e.g., after a `Dense` layer in a `Sequential` model).
*   Use **Functional Activations** when applying an activation to an intermediate result inside a module's `forward` method (e.g., activating gates in a recurrent network).

## Activation Modules Usage

Activation modules are commonly used within `Sequential` containers:

Activation modules are commonly used within `Sequential` containers:

```python
from ember_ml.nn.container import Sequential
from ember_ml.nn.modules import Dense
from ember_ml.nn.modules.activations import ReLU, Dropout

model = Sequential([
    Dense(units=128),
    ReLU(),
    Dropout(0.2),
    Dense(units=10)
])

# Forward pass (training=True implicitly passed to Dropout within Sequential)
output = model(input_tensor, training=True)
```

## Functional Activations Usage

Functional activations are typically used inside the `forward` method of custom modules.

```python
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules import activations # Import the module namespace
from ember_ml import ops

class CustomLayer(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(...) # Example parameter

    def forward(self, x):
        # ... some computation ...
        intermediate = ops.matmul(x, self.weight)
        # Apply functional activation directly
        output = activations.tanh(intermediate)
        return output
```

### Dynamic Activation Lookup

For modules that accept an activation function specified by a string name during initialization (e.g., `activation="relu"`), the recommended pattern is to use the `get_activation` helper function *within the `forward` pass* to retrieve the corresponding functional activation dynamically based on the stored name. This avoids potential issues with backend state or serialization associated with storing function references directly.

```python
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.activations import get_activation # Import the helper
from ember_ml import ops

class ConfigurableActivationLayer(Module):
    def __init__(self, activation_name: str = "relu"):
        super().__init__()
        self.weight = Parameter(...) # Example parameter
        # Store the *name* of the activation
        self.activation_name = activation_name

    def forward(self, x):
        # ... some computation ...
        intermediate = ops.matmul(x, self.weight)
        # Look up the function dynamically using the stored name
        activation_fn = get_activation(self.activation_name)
        # Apply the retrieved functional activation
        output = activation_fn(intermediate)
        return output

# Example usage:
layer_relu = ConfigurableActivationLayer(activation_name="relu")
layer_tanh = ConfigurableActivationLayer(activation_name="tanh")
```

#### `get_activation(name: str)`

Retrieves a functional activation by its string name from the currently aliased backend functions available in `ember_ml.nn.modules.activations`.

**Arguments:**
*   `name` (str): The name of the activation function (e.g., `"relu"`, `"tanh"`, `"sigmoid"`, `"lecun_tanh"`).

**Returns:**
*   (Callable): The corresponding activation function.

**Raises:**
*   `AttributeError`: If the activation function name is not found or aliased for the current backend.

## Backend Support

All activation modules are backend-agnostic and work with any backend (NumPy, PyTorch, MLX) selected via `ember_ml.backend.set_backend`.

## See Also

*   [Core NN Modules](nn_modules.md): Documentation on base modules and other core layers.
*   [Operations (ops)](ops.md): Documentation on the underlying backend-agnostic operations.
*   Backend Implementations (e.g., `ember_ml.backend.mlx.activations`): Specific backend functions providing the implementations (consult backend-specific documentation).