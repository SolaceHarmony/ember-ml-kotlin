# ember_ml/nn/modules/rnn/stride_aware_cfc_layer.py

"""
Stride-Aware Continuous-time Fully Connected (CfC) Layer.
"""

from typing import Union, Optional, Dict, Any # Added Dict, Any

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class
# Import the cell this layer uses
from ember_ml.nn.modules.rnn.stride_aware_cfc import StrideAwareCfC # Remove lecun_tanh import
# Import specific wirings needed for default behavior
from ember_ml.nn.modules.wiring import FullyConnectedMap # Use renamed map class
from ember_ml.nn.modules import activations # Import activations module
from ember_ml.nn.modules.activations import get_activation # Import the new helper

class StrideAwareCfC(Module):
    """
    Stride-Aware Continuous-time Fully Connected (CfC) layer.

    This layer implements a continuous-time recurrent neural network
    with closed-form solution for the hidden state dynamics,
    specialized for multi-timescale processing.

    Args:
        units_or_cell: Number of units, a Wiring object, or a StrideAwareWiredCfCCell
        stride_length: Length of stride for time-scaling
        time_scale_factor: Factor to scale the time constant
        mixed_memory: Whether to use mixed memory
        mode: Mode of operation ("default", "pure", or "no_gate")
        activation: Activation function for the output
        backbone_units: Number of units in the backbone
        backbone_layers: Number of layers in the backbone
        backbone_dropout: Dropout rate for the backbone
        fully_recurrent: Whether to use full recurrent connections
        return_sequences: Whether to return the full sequence
        return_state: Whether to return the state
    """

    def __init__(
        self,
        neuron_map_or_cell: Union[NeuronMap, StrideAwareCfC], # Remove int option
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        mixed_memory: bool = False,
        mode: str = "default",
        activation = "lecun_tanh", # Use string name as default
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[float] = None,
        fully_recurrent: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        **kwargs
    ):
        """
        Initialize the StrideAwareCfC layer.

        Args:
            neuron_map_or_cell: A NeuronMap object or a StrideAwareWiredCfCCell instance
            stride_length: Length of stride for time-scaling
            time_scale_factor: Factor to scale the time constant
            mixed_memory: Whether to use mixed memory
            mode: Mode of operation ("default", "pure", or "no_gate")
            activation: Activation function name string for the output (e.g., "lecun_tanh")
            backbone_units: Number of units in the backbone
            backbone_layers: Number of layers in the backbone
            backbone_dropout: Dropout rate for the backbone
            fully_recurrent: Whether to use full recurrent connections
            return_sequences: Whether to return the full sequence
            return_state: Whether to return the state
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        # Store parameters
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.mixed_memory = mixed_memory
        self.mode = mode
        self._activation = activation
        self.backbone_units = backbone_units or 128
        self.backbone_layers = backbone_layers or 1
        self.backbone_dropout = backbone_dropout or 0.0
        self.fully_recurrent = fully_recurrent
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Create the cell
        # Check argument type
        if isinstance(neuron_map_or_cell, StrideAwareWiredCfC): # Check for cell
            self.cell = neuron_map_or_cell
        elif isinstance(neuron_map_or_cell, NeuronMap): # Check for NeuronMap
            if any([backbone_units, backbone_layers, backbone_dropout]):
                raise ValueError("Cannot use backbone parameters with a Wiring object.")
            # Ensure the wiring is built before passing it to the cell
            # The cell's __init__ expects a built wiring or input_size
            # We might not know input_size here, so let the cell handle build if needed
            self.cell = StrideAwareWiredCfC(
                neuron_map=neuron_map_or_cell, # Pass map with correct name
                stride_length=stride_length,
                time_scale_factor=time_scale_factor,
                fully_recurrent=fully_recurrent,
                mode=mode,
                activation=activation,
                backbone_units=128, # Default if not specifiable
                backbone_layers=1  # Default if not specifiable
            )
        else:
            raise ValueError("neuron_map_or_cell must be a NeuronMap object or a StrideAwareWiredCfCCell instance")

        # Add mixed memory if requested
        if mixed_memory:
            self.cell = self._create_mixed_memory_cell(self.cell)

    # Removed _create_simple_cfc_cell method as it's no longer needed

    def _create_mixed_memory_cell(self, cell):
        """Create a mixed memory cell that wraps the given cell."""
        # This would be a wrapper around the cell that adds mixed memory
        # For now, we'll just return the cell as is
        return cell

    def forward(self, inputs, initial_state=None, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: Input tensor
            initial_state: Initial state
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor or tuple of (output, state) if return_state is True
        """
        # Process the sequence
        batch_size = tensor.shape(inputs)[0]
        time_steps = tensor.shape(inputs)[1]

        # Initialize state if not provided
        if initial_state is None:
             # Use the cell's get_initial_state method
             state = self.cell.get_initial_state(batch_size=batch_size)
        else:
            state = initial_state

        # Process each time step
        outputs = []
        for t in range(time_steps):
            # Get input at time t
            x_t = inputs[:, t, :]

            # Process with cell
            # Pass **kwargs down to the cell's forward method
            output, state = self.cell.forward(x_t, state, **kwargs)

            # Store output
            outputs.append(output)

        # Stack outputs
        if self.return_sequences:
            outputs_tensor = tensor.stack(outputs, axis=1)
        else:
            outputs_tensor = outputs[-1]

        # Return output and state if requested
        if self.return_state:
            return outputs_tensor, state
        else:
            return outputs_tensor


    def get_config(self):
        """
        Get configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "mixed_memory": self.mixed_memory,
            "mode": self.mode,
            "activation": self._activation,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout,
            "fully_recurrent": self.fully_recurrent,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
        }

        # If the cell has a wiring, save its config
        # If the cell has a neuron_map, save its config
        if hasattr(self.cell, 'neuron_map') and hasattr(self.cell.neuron_map, 'get_config'):
             map_config = self.cell.neuron_map.get_config()
             # Add class name for deserialization
             map_config['class_name'] = self.cell.neuron_map.__class__.__name__
             config["neuron_map_or_cell"] = map_config # Changed key name
        else:
            # Otherwise, save the cell's units
            # Use hidden_size from parent ModuleCell if available
            units = getattr(self.cell, 'hidden_size', getattr(self.cell, 'units', None))
            if units is None:
                 raise ValueError("Cannot determine units for cell in get_config")
            config["neuron_map_or_cell"] = units # Changed key name


        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a StrideAwareCfC from a configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            StrideAwareCfC instance
        """
        config_copy = config.copy()

        # Handle wiring configuration
        # Use updated key name
        neuron_map_or_cell_config = config_copy.get("neuron_map_or_cell")
        if isinstance(neuron_map_or_cell_config, dict):
            # It's a map config
            # Import from the modules.wiring package where map classes reside
            from ember_ml.nn.modules import wiring as map_module
            import importlib

            # Get the wiring class name and config params
            map_class_name = neuron_map_or_cell_config.pop("class_name", "NeuronMap") # Use NeuronMap as default
            map_params = neuron_map_or_cell_config # Remaining items are params

            # Try to get class from modules package
            # Try to get class directly from the map module (nn.modules.wiring)
            try:
                 map_class = getattr(map_module, map_class_name)
            except AttributeError:
                 # This fallback might not be needed if __init__ exports all maps
                 raise ImportError(f"Could not find NeuronMap class '{map_class_name}' in {map_module.__name__}")



            # Create the wiring from the config
            # Create the map instance
            neuron_map_or_cell = map_class(**map_params)
            del config_copy["neuron_map_or_cell"] # Use updated key name
        else:
            # It's a simple integer (units)
            # It's a simple integer (units)
            neuron_map_or_cell = config_copy.pop("neuron_map_or_cell") # Use updated key name

        # Remove backbone parameters if units_or_cell is a Wiring object
        # (Unless the cell itself handles backbone params internally)
        # Check if created object is a NeuronMap
        if isinstance(neuron_map_or_cell, NeuronMap):
            config_copy.pop('backbone_units', None)
            config_copy.pop('backbone_layers', None)
            config_copy.pop('backbone_dropout', None)

        # Create the layer
        # Pass the created map or units int to constructor
        return cls(neuron_map_or_cell, **config_copy)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the StrideAwareCfC layer."""
        config = super().get_config()
        # Save layer's direct __init__ args
        config.update({
            # Save the config needed to reconstruct the cell later
            "cell_config": self.cell.get_config(),
            "cell_class_name": self.cell.__class__.__name__,
            # Save other layer args
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "mixed_memory": self.mixed_memory,
            "mode": self.mode,
            # Save activation name/repr
            "activation": self._activation if isinstance(self._activation, str) else getattr(self._activation, '__name__', str(self._activation)),
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout,
            "fully_recurrent": self.fully_recurrent,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
        })
        # 'neuron_map_or_cell' (original arg) is implicitly handled by saving cell_config
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StrideAwareCfC':
        """Creates a StrideAwareCfC layer from its configuration."""
        cell_config = config.pop("cell_config")
        cell_class_name = config.pop("cell_class_name")

        # Import cell class dynamically (assuming it's StrideAwareWiredCfCCell)
        if cell_class_name == "StrideAwareWiredCfCCell":
             from .stride_aware_cfc import StrideAwareWiredCfCCell as cell_cls
        else:
             # Add logic for other potential cell types if needed
             raise TypeError(f"Unsupported cell type for StrideAwareCfC: {cell_class_name}")

        # Reconstruct the cell instance
        cell = cell_cls.from_config(cell_config)

        # Prepare config for StrideAwareCfC layer __init__
        layer_config = config # Start with remaining config
        layer_config['neuron_map_or_cell'] = cell # Pass the reconstructed cell

        # Handle activation reconstruction if name was serialized
        activation_config_val = layer_config.get('activation')
        # Remove special case for 'lecun_tanh', rely on get_activation
        if isinstance(activation_config_val, str):
            try:
                # Use new helper to reconstruct from name
                layer_config['activation'] = get_activation(activation_config_val)
            except (AttributeError, TypeError):
                # If reconstruction fails, remove it so __init__ uses default
                layer_config.pop('activation', None)
        elif activation_config_val is not None and not callable(activation_config_val):
            # If it's somehow serialized but not a string or callable, remove it
            layer_config.pop('activation', None)

        # Call the base from_config which calls cls(**layer_config)
        return super(StrideAwareCfC, cls).from_config(layer_config)