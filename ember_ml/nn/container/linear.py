"""
Linear layer implementation for ember_ml.

This module provides a backend-agnostic implementation of a fully connected
(linear) layer that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Union, Tuple, Any, Dict, List

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = x @ W.T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        device: Device to place the parameters on
        dtype: Data type of the parameters
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[Any] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization (He initialization)
        # This is a good default for layers followed by ReLU
        std = ops.sqrt(ops.divide(2.0, in_features))
        weight_data = tensor.random_normal(
            (out_features, in_features),
            mean=0.0,
            stddev=std,
            dtype=dtype,
            device=device
        )
        self.weight = Parameter(weight_data)
        
        if bias:
            # Initialize bias to zeros
            bias_data = tensor.zeros(out_features, dtype=dtype, device=device)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Ensure x is a tensor
        x = tensor.convert_to_tensor(x)
        
        # Compute the linear transformation
        output = ops.matmul(x, tensor.transpose(self.weight))
        
        # Add bias if present
        if self.bias is not None:
            output = ops.add(output, self.bias)
        
        return output
    
    def add(self, layer: Any) -> None:
        """
        Add a layer to the container.
        
        Args:
            layer: Layer to add
            
        Raises:
            NotImplementedError: Linear layer does not support adding layers
        """
        raise NotImplementedError("Linear layer does not support adding layers")
    
    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None:
        """
        Build the container for a specific input shape.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Linear layer is already built during initialization
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the container.
        
        Returns:
            Dictionary containing the configuration
        """
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias is not None
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the container.
        
        Returns:
            Dictionary containing the state
        """
        state = {
            'weight': self.weight
        }
        if self.bias is not None:
            state['bias'] = self.bias
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary into the container.
        
        Args:
            state_dict: Dictionary containing the state
        """
        self.weight = state_dict['weight']
        if 'bias' in state_dict and self.bias is not None:
            self.bias = state_dict['bias']
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"