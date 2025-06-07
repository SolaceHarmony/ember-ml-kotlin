"""
Sequential container implementation for ember_ml.

This module provides a backend-agnostic implementation of a sequential container
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Union, Tuple, Any, Dict, List, Sequence

# Import necessary layer modules for from_config reconstruction
from . import linear
from ..modules import activations
from typing import Dict, Any, List # Ensure these are imported if not already
from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor
class Sequential(Module):
    """
    A sequential container.
    
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can be passed in.
    
    Args:
        layers: An ordered list of modules to add
    """
    
    def __init__(
        self,
        layers: Optional[List[Any]] = None
    ):
        super().__init__()
        self.layers = []
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
    
    def forward(self, x):
        """
        Forward pass through the sequential container.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all layers
        """
        # Ensure x is a tensor
        x = tensor.convert_to_tensor(x)
        
        # Pass input through each layer in sequence
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def add(self, layer: Any) -> None:
        """
        Add a layer to the container.
        
        Args:
            layer: Layer to add
        """
        self.layers.append(layer)
    
    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None:
        """
        Build the container for a specific input shape.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Build each layer in sequence
        shape = input_shape
        for layer in self.layers:
            if hasattr(layer, 'build'):
                layer.build(shape)
            
            # Update shape for next layer
            if hasattr(layer, 'compute_output_shape'):
                shape = layer.compute_output_shape(shape)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the container.
        
        Returns:
            Dictionary containing the configuration
        """
        return {
            'layers': [
                {
                    'class_name': layer.__class__.__name__,
                    'config': layer.get_config() if hasattr(layer, 'get_config') else {}
                }
                for layer in self.layers
            ]
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Sequential':
        """
        Creates a Sequential container from its config.

        This method reconstructs the container and its layers based on the
        provided configuration dictionary.

        Args:
            config: Dictionary containing the configuration, typically obtained
                    from `get_config()`. Expected format: {'layers': [...]}.

        Returns:
            A new Sequential instance configured according to the input config.

        Raises:
            ValueError: If the configuration is invalid or a layer class
                        cannot be found or instantiated.
        """
        reconstructed_layers = []
        layer_configs = config.get('layers', [])

        # Map class names to actual classes (adjust imports as needed)
        # TODO: Implement a more robust layer registry/lookup mechanism
        available_modules = {
            'Linear': linear.Linear,
            'ReLU': activations.ReLU,
            # Add other potential layer classes from container/modules here
            # e.g., 'BatchNormalization': batch_normalization.BatchNormalization,
        }

        for layer_config in layer_configs:
            class_name = layer_config.get('class_name')
            layer_specific_config = layer_config.get('config', {})

            if not class_name:
                raise ValueError("Layer configuration missing 'class_name'.")

            LayerClass = available_modules.get(class_name)
            if LayerClass is None:
                raise ValueError(f"Unknown layer class '{class_name}'. Ensure it's imported and added to 'available_modules'.")

            try:
                # Check if the layer class has its own from_config method
                if hasattr(LayerClass, 'from_config') and callable(getattr(LayerClass, 'from_config')):
                     layer_instance = LayerClass.from_config(layer_specific_config)
                else:
                     # Otherwise, instantiate directly using the config kwargs
                     layer_instance = LayerClass(**layer_specific_config)
                reconstructed_layers.append(layer_instance)
            except Exception as e:
                raise ValueError(f"Failed to instantiate layer '{class_name}' from config: {e}")

        return cls(layers=reconstructed_layers)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the container.
        
        Returns:
            Dictionary containing the state
        """
        return {
            f'layer_{i}': layer.state_dict() if hasattr(layer, 'state_dict') else {}
            for i, layer in enumerate(self.layers)
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary into the container.
        
        Args:
            state_dict: Dictionary containing the state
        """
        for i, layer in enumerate(self.layers):
            layer_key = f'layer_{i}'
            if layer_key in state_dict and hasattr(layer, 'load_state_dict'):
                layer.load_state_dict(state_dict[layer_key])
    
    def train(self, mode: bool = True) -> 'Sequential':
        """
        Set the container in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            Self
        """
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train(mode)
        return self
    
    def eval(self) -> 'Sequential':
        """
        Set the container in evaluation mode.
        
        Returns:
            Self
        """
        return self.train(False)
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"layers={len(self.layers)}"
    
    def __repr__(self):
        layer_reprs = [f"  ({i}): {repr(layer)}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(layer_reprs) + "\n)"
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Any, 'Sequential']:
        """
        Get a layer or a new Sequential container with the specified layers.
        
        Args:
            idx: Index or slice
            
        Returns:
            Layer or new Sequential container
        """
        if isinstance(idx, slice):
            return Sequential(self.layers[idx])
        else:
            return self.layers[idx]
    
    def __len__(self) -> int:
        """
        Get the number of layers in the container.
        
        Returns:
            Number of layers
        """
        return len(self.layers)