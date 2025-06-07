"""
Stride-Aware Wired CfC Cell and Layer

This module provides implementations of StrideAwareWiredCfCCell and StrideAwareCfC,
which are specialized recurrent neural network components for multi-timescale processing.
"""

from typing import Union, Optional, Dict, Any # Added Dict, Any

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Parameter # Module removed, inheriting from ModuleWiredCell instead
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class
from ember_ml.nn.modules import Module # Import parent class
from ember_ml.nn.initializers import glorot_uniform, orthogonal, BinomialInitializer
from ember_ml.nn.modules import activations # Import activations module
from ember_ml.nn.modules.activations import get_activation # Import the new helper

# Local lecun_tanh function removed, use activations.lecun_tanh instead


class StrideAwareCfC(Module): # Inherit from ModuleWiredCell
    """
    Stride-Aware Wired CfC Cell.
    
    This cell implements a continuous-time recurrent neural network
    with closed-form solution for the hidden state dynamics,
    specialized for multi-timescale processing with custom wiring.
    
    Args:
        wiring: Wiring configuration (e.g., AutoNCP)
        stride_length: Length of stride for time-scaling
        time_scale_factor: Factor to scale the time constant
        fully_recurrent: Whether to use full recurrent connections
        mode: Mode of operation ("default", "pure", or "no_gate")
        activation: Activation function for the output
        backbone_units: Number of units in the backbone
        backbone_layers: Number of layers in the backbone
    """
    
    def __init__(
            self,
            neuron_map: NeuronMap, # Corrected type hint and name
            stride_length: int = 1,
            time_scale_factor: float = 1.0,
            fully_recurrent: bool = True,
            mode: str = "default",
            activation = "lecun_tanh", # Use string name as default
            backbone_units: int = 128,
            backbone_layers: int = 1,
            **kwargs
    ):
        """
        Initialize the Stride-Aware Wired CfC cell.
        
        Args:
            neuron_map: NeuronMap configuration object
            stride_length: Length of stride for time-scaling
            time_scale_factor: Factor to scale the time constant
            fully_recurrent: Whether to use full recurrent connections
            mode: Mode of operation ("default", "pure", or "no_gate")
            activation: Activation function name string for the output (e.g., "lecun_tanh")
            backbone_units: Number of units in the backbone
            backbone_layers: Number of layers in the backbone
            **kwargs: Additional keyword arguments
        """
        # Call ModuleWiredCell's __init__
        # ModuleWiredCell needs input_size, which is wiring.input_dim
        # ModuleWiredCell calls ModuleCell init, passing wiring.units as hidden_size
        # Pass input_size and neuron_map to ModuleWiredCell parent
        # ModuleWiredCell will handle building map if needed and calling ModuleCell init
        # Handle the case where neuron_map is an AutoNCP instance that doesn't have is_built()
        input_size = None
        if hasattr(neuron_map, 'is_built') and neuron_map.is_built():
            input_size = neuron_map.input_dim
        
        super().__init__(
            input_size=input_size,
            neuron_map=neuron_map,
            mode=mode,
            # Pass other relevant kwargs up
            **kwargs
        )
        # If map wasn't built, ModuleWiredCell's init would have failed earlier,
        # or it needs to handle deferred building based on first forward pass.
        # Assuming ModuleWiredCell handles build correctly based on input_size.
        # self.wiring is set by parent init
        # self.input_size, self.hidden_size (units) are set by parent init
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.fully_recurrent = fully_recurrent
        # self.mode is set by parent init
        self.activation = activation # Store the actual function
        self._activation = activation # Keep for serialization if needed, or serialize name
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        # self.units, self.input_dim, self.output_dim are available via self.wiring or parent properties like self.hidden_size, self.input_size, self.output_size

        self.activation = activation
        
        # Initialize weights
        self._initialize_weights()
        
        # State size is defined as a property

    def _initialize_weights(self):
        """Initialize the weights for the cell with wiring constraints."""
        # Get input dimension from wiring
        # Access input_dim via parent property, which gets it from neuron_map
        # Access input_dim via the map stored in the parent class
        # Access input_dim via parent property, which gets it from neuron_map
        input_dim = self.input_size
        
        # Input weights
        self.kernel = Parameter(tensor.zeros((input_dim, self.backbone_units)))
        self.kernel.data = glorot_uniform((input_dim, self.backbone_units))
        
        # Recurrent weights
        # Get units from neuron_map instead of trying to access self.units
        units = self.neuron_map.units
        self.recurrent_kernel = Parameter(tensor.zeros((units, self.backbone_units)))
        self.recurrent_kernel.data = orthogonal((units, self.backbone_units))
        
        # Backbone weights (multiple layers)
        self.backbone_kernels = []
        self.backbone_biases = []
        for i in range(self.backbone_layers):
            backbone_kernel = Parameter(tensor.zeros((self.backbone_units, self.backbone_units)))
            backbone_kernel.data = glorot_uniform((self.backbone_units, self.backbone_units))
            self.backbone_kernels.append(backbone_kernel)
            
            backbone_bias = Parameter(tensor.zeros((self.backbone_units,)))
            self.backbone_biases.append(backbone_bias)
        
        # Output projection
        # Get units from neuron_map instead of trying to access self.units
        units = self.neuron_map.units
        self.backbone_out = Parameter(tensor.zeros((self.backbone_units, units)))
        self.backbone_out.data = glorot_uniform((self.backbone_units, units))
        
        # Time gate weights
        self.time_kernel = Parameter(tensor.zeros((1, units)))
        
        # Biases
        self.bias = Parameter(tensor.zeros((self.backbone_units,)))
        # Get units from neuron_map instead of trying to access self.units
        units = self.neuron_map.units
        self.recurrent_bias = Parameter(tensor.zeros((units,)))
        
        # Gate weights (for default and pure modes)
        if self.mode != "no_gate":
            # Get units from neuron_map instead of trying to access self.units
            units = self.neuron_map.units
            self.gate_kernel = Parameter(tensor.zeros((input_dim, units)))
            self.gate_kernel.data = glorot_uniform((input_dim, units))
            
            self.gate_recurrent_kernel = Parameter(tensor.zeros((units, units)))
            self.gate_recurrent_kernel.data = orthogonal((units, units))
            
            # Get units from neuron_map instead of trying to access self.units
            units = self.neuron_map.units
            self.gate_bias = Parameter(tensor.ones((units,)))  # Initialize with ones for open gates
        
        # Sparsity masks
        # Access sparsity via the map stored in the parent class
        # Access sparsity via the map stored in the parent class
        # Access sparsity level via the map stored in parent
        # Access sparsity level via the map stored in parent
        sparsity = self.neuron_map.sparsity_level
        # Use float32 dtype for masks to ensure compatibility with all backends
        mask_dtype = 'float32'
        self.input_mask = Parameter(BinomialInitializer(probability=sparsity, seed=42)((input_dim,), dtype=mask_dtype))
        self.input_mask.requires_grad = False  # Not trainable
        
        # Get units from neuron_map instead of trying to access self.units
        units = self.neuron_map.units
        self.recurrent_mask = Parameter(BinomialInitializer(probability=sparsity, seed=43)((units, units), dtype=mask_dtype))
        self.recurrent_mask.requires_grad = False  # Not trainable
        
        # Get units from neuron_map instead of trying to access self.units
        units = self.neuron_map.units
        self.output_mask = Parameter(BinomialInitializer(probability=sparsity, seed=44)((units,), dtype=mask_dtype))
        self.output_mask.requires_grad = False  # Not trainable

    def _compute_time_scaling(self, inputs, kwargs):
        """Helper function to compute time scaling."""
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = ops.multiply(ops.multiply(t, self.stride_length), self.time_scale_factor)
        else:
            t = kwargs.get("time", 1.0)
            t = ops.multiply(ops.multiply(t, self.stride_length), self.time_scale_factor)
            t = tensor.cast(t, dtype='float32')
        return inputs, t

    def forward(self, inputs, states=None, **kwargs):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state]
            **kwargs: Additional keyword arguments including time
            
        Returns:
            Tuple of (output, [new_hidden_state])
        """
        # Initialize states if not provided
        if states is None:
            # Get units from neuron_map instead of trying to access self.units
            units = self.neuron_map.units
            h_prev = tensor.zeros((tensor.shape(inputs)[0], units))
        else:
            h_prev = states[0]
        
        # Apply time scaling
        inputs, t = self._compute_time_scaling(inputs, kwargs)
        
        # Apply wiring masks
        # Extract data from Parameter objects to avoid dtype inference issues
        # Access masks via the neuron_map property inherited from ModuleWiredCell/BaseModule?
        # Or are they intended to be direct attributes set by _initialize_weights?
        # Assuming they are direct attributes set by _initialize_weights for now.
        input_mask_data = self.input_mask.data # Assume Parameter object
        recurrent_mask_data = self.recurrent_mask.data # Assume Parameter object
        
        # Use string dtypes for consistent behavior across backends
        float_dtype = 'float32'
        
        # Cast masks to float32
        input_mask_tensor = tensor.cast(input_mask_data, dtype=float_dtype)
        recurrent_mask_tensor = tensor.cast(recurrent_mask_data, dtype=float_dtype)
        
        # Apply masks
        masked_inputs = ops.multiply(inputs, input_mask_tensor)
        
        # Handle broadcasting for recurrent_mask_tensor (shape 8,8) with h_prev (shape batch_size,8)
        # We need to reshape the mask to apply it correctly across the batch dimension
        batch_size = tensor.shape(h_prev)[0]
        
        # Apply the mask using matmul instead of element-wise multiply
        # This handles the batch dimension correctly
        masked_h_prev = ops.matmul(h_prev, recurrent_mask_tensor)
        
        # Backbone computation
        x = ops.add(ops.matmul(masked_inputs, self.kernel), self.bias)
        x = self.activation(ops.add(x, ops.matmul(masked_h_prev, self.recurrent_kernel)))
        
        # Apply backbone layers
        for i in range(self.backbone_layers):
            x = self.activation(ops.add(ops.matmul(x, self.backbone_kernels[i]), self.backbone_biases[i]))
        
        # Compute candidate hidden state
        h_candidate = ops.add(ops.matmul(x, self.backbone_out), self.recurrent_bias)
        
        # Compute time gate
        time_gate = ops.exp(ops.multiply(-ops.abs(t), ops.exp(self.time_kernel)))
        
        # Apply gating mechanism based on mode
        if self.mode == "no_gate":
            h_new = ops.add(
                ops.multiply(h_prev, time_gate),
                ops.multiply(h_candidate, ops.subtract(tensor.ones_like(time_gate), time_gate))
            )
        else:
            # Compute gate values
            gate_in = ops.matmul(inputs, self.gate_kernel)
            gate_rec = ops.matmul(h_prev, self.gate_recurrent_kernel)
            gate = ops.sigmoid(ops.add(ops.add(gate_in, gate_rec), self.gate_bias))
            
            if self.mode == "pure":
                # Pure mode: h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
                gate_time = ops.multiply(gate, time_gate)
                h_new = ops.add(
                    ops.multiply(h_prev, gate_time),
                    ops.multiply(h_candidate, ops.subtract(tensor.ones_like(gate_time), gate_time))
                )
            else:
                # Default mode: h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)
                h_new = ops.add(
                    ops.multiply(h_prev, gate),
                    ops.multiply(
                        ops.multiply(h_candidate, ops.subtract(tensor.ones_like(gate), gate)),
                        ops.subtract(tensor.ones_like(time_gate), time_gate)
                    )
                )
        
        # Apply output mask
        # Extract data from Parameter objects to avoid dtype inference issues
        output_mask_data = self.output_mask.data # Assume Parameter object
        
        # Use string dtypes for consistent behavior across backends
        output_mask_tensor = tensor.cast(output_mask_data, dtype='float32')
        output = ops.multiply(h_new, output_mask_tensor)
        
        return output, [h_new]

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the StrideAwareWiredCfC cell."""
        # Get config from ModuleWiredCell (map config, map class, mode, etc.)
        config = super().get_config()
        # Add StrideAware specific args from __init__
        config.update({
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "fully_recurrent": self.fully_recurrent,
            # "activation": ..., # Removing activation serialization for now
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
        })
        # Remove args from parent configs if they aren't direct __init__ args for this class
        config.pop('hidden_size', None)
        config.pop('input_size', None)
        config.pop('use_bias', None)
        # Activation name saved above might override parent's, which is fine

        return config
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StrideAwareWiredCfCCell':
        """Creates a StrideAwareWiredCfC cell from its configuration."""
        # Handle activation reconstruction if name was serialized
        activation_config_val = config.get('activation')
        # Remove special case for 'lecun_tanh', rely on get_activation
        if isinstance(activation_config_val, str):
             try:
                 # Use new helper to reconstruct from name
                 config['activation'] = get_activation(activation_config_val)
             except (AttributeError, TypeError):
                 # If reconstruction fails, remove it so __init__ uses default
                 config.pop('activation', None)
        elif activation_config_val is not None and not callable(activation_config_val):
             # If it's somehow serialized but not a string or callable, remove it
             config.pop('activation', None)

        # Clean up the config for proper reconstruction
        # Remove input_size from config to avoid duplicate parameter error
        input_size = config.pop('input_size', None)
        
        # Check if neuron_map needs reconstruction
        neuron_map = config.get('neuron_map')
        if isinstance(neuron_map, dict):
            # Import NeuronMap classes
            from ember_ml.nn.modules.wiring import NeuronMap
            from ember_ml.nn.modules.wiring.fully_connected_map import FullyConnectedMap
            from ember_ml.nn.modules.wiring.ncp_map import NCPMap
            from ember_ml.nn.modules.wiring.random_map import RandomMap
            
            # Get the map class
            map_class_name = neuron_map.pop('class_name', 'FullyConnectedMap')
            map_classes = {
                'NeuronMap': NeuronMap,
                'FullyConnectedMap': FullyConnectedMap,
                'NCPMap': NCPMap,
                'RandomMap': RandomMap
            }
            
            map_cls = map_classes.get(map_class_name)
            if map_cls:
                config['neuron_map'] = map_cls.from_config(neuron_map)
        
        # Create a new instance using a direct approach
        try:
            # Don't pass input_size in kwargs to avoid duplicate parameter error
            return cls(**config)
        except Exception as e:
            # If there's an error in direct instantiation, try the parent approach
            # but make sure to clean the config again
            config.pop('input_size', None)  # Ensure input_size is removed
            return super(StrideAwareWiredCfCCell, cls).from_config(config)


    def get_initial_state(self, batch_size=1):
        """
        Get the initial state for the cell.

        Args:
            batch_size: Batch size

        Returns:
            Initial state
        """
        # Get units from neuron_map
        units = self.neuron_map.units
        h = tensor.zeros((batch_size, units))
        return [h]

    # Removed duplicated get_config and from_config methods


# StrideAwareCfC class definition removed and placed in stride_aware_cfc_layer.py
# Visualization function and main block removed and placed in examples/rnn/stride_aware_cfc_visualization.py