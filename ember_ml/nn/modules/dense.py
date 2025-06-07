"""
Dense (fully connected) module for ember_ml.

This module provides a backend-agnostic implementation of a dense (fully connected)
layer that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Any, Dict # Removed Union, Tuple, Callable

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.activations import get_activation
from ember_ml.nn import tensor
class Dense(Module):
    # Explicitly type hint attributes for clarity and type checking
    input_dim: int
    units: int
    activation: Optional[str]
    use_bias: bool
    kernel: Parameter
    bias: Optional[Parameter] # Crucial for mypy when use_bias=False

    """
    Dense (fully connected) layer.
    
    Implements the operation: output = activation(dot(input, kernel) + bias)
    
    Attributes:
        units: Dimensionality of the output space
        activation: Activation function to use.
        use_bias: Whether the layer uses a bias vector.
        input_dim: Dimensionality of the input space (derived from first input).
        kernel: Parameter # Weight matrix (Parameter).
        bias: Optional[Parameter] # Bias vector (Parameter, if use_bias is True).
    """
    
    def __init__(self, input_dim: int, units: int, activation: Optional[str] = None, use_bias: bool = True):
        """
        Initialize a dense layer.

        Args:
            input_dim: Dimensionality of the input space.
            units: Dimensionality of the output space.
            activation: Activation function to use (e.g., 'relu', 'tanh').
            use_bias: Whether the layer uses a bias vector.
        """
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.activation = activation
        self.use_bias = use_bias

        # Initialize weights using Glorot uniform initialization
        # Use ops for backend agnosticism
        denominator = ops.add(tensor.convert_to_tensor(self.input_dim), tensor.convert_to_tensor(self.units))
        sqrt_val = ops.sqrt(ops.divide(tensor.convert_to_tensor(6.0), denominator))
        self.kernel = Parameter(
            tensor.random_uniform(
                (self.input_dim, self.units),
                minval=-sqrt_val,
                maxval=sqrt_val
            )
        )

        if self.use_bias:
            self.bias = Parameter(tensor.zeros((self.units,)))
        else:
            self.bias = None
    
    def forward(self, x: Any) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get input shape
        input_shape = tensor.shape(x)
        
        # Validate input shape
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, "
                f"but received input with shape {input_shape}"
            )
        
        # Reshape input if needed
        original_shape = input_shape
        if len(input_shape) > 2:
            # Flatten all dimensions except the last one
            x = tensor.reshape(x, (-1, input_shape[-1]))
        
        # Linear transformation
        # Linear transformation
        output = ops.matmul(x, self.kernel.data) # Use Parameter's data
        if self.bias is not None:
            output = ops.add(output, self.bias.data) # Use Parameter's data
        
        # Apply activation if specified
        if self.activation is not None:
            try:
                # Use the get_activation helper to retrieve the activation function
                activation_fn = get_activation(self.activation)
                output = activation_fn(output)
            except (AttributeError, ValueError) as e:
                # Fallback to ops module for backward compatibility
                activation_fn = getattr(ops, self.activation, None)
                if activation_fn is not None:
                    output = activation_fn(output)
                else:
                    raise ValueError(f"Unknown activation function: {self.activation}") from e
        
        # Reshape output if needed
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1])
            output_shape.append(self.units) # Avoid '+' operator for list concatenation
            output = tensor.reshape(output, output_shape)
        
        return output
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return (f"Dense(input_dim={self.input_dim}, units={self.units}, "
                f"activation={self.activation}, use_bias={self.use_bias})")

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the Dense layer."""
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
        })
        return config

    # from_config can rely on BaseModule implementation