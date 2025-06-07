"""
Neural Circuit Policy (NCP) module.

This module provides the NCP class, which implements a neural circuit policy
using a wiring configuration.
"""

from typing import Optional, Tuple, Dict, Any, Union

from ember_ml import ops
from ember_ml.nn import initializers  # Import initializers module
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import get_activation  # Keep helper import
from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap  # Use renamed base class


class NCP(Module):
    """
    Neural Circuit Policy (NCP) module.
    
    This module implements a neural circuit policy using a wiring configuration.
    It consists of a recurrent neural network with a specific connectivity pattern
    defined by the wiring configuration.
    """
    
    def __init__(
        self,
        neuron_map: NeuronMap, # Changed argument name
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        dtype: Optional[Any] = None,
    ):
        """
        Initialize an NCP module.
        
        Args:
            neuron_map: NeuronMap configuration object
            activation: Activation function to use
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias weights
            dtype: Data type for the weights
        """
        super().__init__()
        
        self.neuron_map = neuron_map # Changed attribute name
        self.activation = activation # Store the string name only
        # self._activation_fn removed
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dtype = dtype
        
        # Defer mask and weight initialization to build method
        self.input_mask = None
        self.recurrent_mask = None
        self.output_mask = None
        self._kernel = None
        self._recurrent_kernel = None
        self._bias = None
        self.built = False # Track build status of the layer
        
        # Initialize state
        self.state = tensor.zeros((1, self.neuron_map.units))
    
    @property
    def kernel(self):
        """Get the kernel parameter."""
        return self._kernel.data
    
    @property
    def recurrent_kernel(self):
        """Get the recurrent kernel parameter."""
        return self._recurrent_kernel.data
    
    @property
    def bias(self):
        """Get the bias parameter."""
        if self.use_bias and self._bias is not None:
            return self._bias.data
        return None
    

    def build(self, input_shape):
        """
        Build the layer's weights and masks based on the input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return

        # Ensure input_shape is a tuple or list
        if not isinstance(input_shape, (tuple, list)):
             raise TypeError(f"Expected input_shape to be a tuple or list, got {type(input_shape)}")

        if len(input_shape) < 1:
             raise ValueError(f"Input shape must have at least one dimension, got {input_shape}")

        input_dim = input_shape[-1]

        # Build the neuron map if it hasn't been built or if input_dim changed
        if not self.neuron_map.is_built() or self.neuron_map.input_dim != input_dim:
            # build() method in NeuronMap subclasses should set self._built = True
            self.neuron_map.build(input_dim)

        # Check if map build was successful (it should set input_dim)
        if self.neuron_map.input_dim is None:
             raise RuntimeError("NeuronMap failed to set input_dim during build.")

        # Get masks after map is built (these are backend tensors, likely int32)
        print(f"DEBUG: Getting masks from neuron map. input_dim={self.neuron_map.input_dim}, units={self.neuron_map.units}")
        input_mask_int = tensor.convert_to_tensor(self.neuron_map.get_input_mask())
        recurrent_mask_int = tensor.convert_to_tensor(self.neuron_map.get_recurrent_mask())
        output_mask_int = tensor.convert_to_tensor(self.neuron_map.get_output_mask())
        print(f"DEBUG: Got input_mask_int with shape {tensor.shape(input_mask_int)} dtype {tensor.dtype(input_mask_int)}")
        print(f"DEBUG: Got recurrent_mask_int with shape {tensor.shape(recurrent_mask_int)} dtype {tensor.dtype(recurrent_mask_int)}")
        print(f"DEBUG: Got output_mask_int with shape {tensor.shape(output_mask_int)} dtype {tensor.dtype(output_mask_int)}")

        # Cast masks to the default float type (float32) for use in matmul/multiply operations
        # These masks act as multiplicative factors, so they need to match the float dtype of the data/state/kernels
        float_dtype = tensor.float32 # Use the standard float type defined in nn.tensor

        self.input_mask = tensor.cast(input_mask_int, float_dtype)
        self.recurrent_mask = tensor.cast(recurrent_mask_int, float_dtype)
        self.output_mask = tensor.cast(output_mask_int, float_dtype)
        print(f"DEBUG: Cast input_mask to shape {tensor.shape(self.input_mask)} dtype {tensor.dtype(self.input_mask)}")
        print(f"DEBUG: Cast recurrent_mask to shape {tensor.shape(self.recurrent_mask)} dtype {tensor.dtype(self.recurrent_mask)}")
        print(f"DEBUG: Cast output_mask to shape {tensor.shape(self.output_mask)} dtype {tensor.dtype(self.output_mask)}")

        # Initialize weights now that input_dim is known
        try:
            print(f"DEBUG: Initializing kernel with shape ({self.neuron_map.input_dim}, {self.neuron_map.units}) using {self.kernel_initializer}")
            tensor_data = self._initialize_tensor(
                (self.neuron_map.input_dim, self.neuron_map.units),
                self.kernel_initializer
            )
            print(f"DEBUG: Initialized tensor data with shape {tensor.shape(tensor_data)}")
            # Create the parameter and assign it to self._kernel
            param = Parameter(tensor_data)
            print(f"DEBUG: Created kernel Parameter: {param}")
            print(f"DEBUG: Parameter data type: {type(param.data)}")
            print(f"DEBUG: Parameter data shape: {tensor.shape(param.data)}")
            # Assign the parameter to self._kernel
            self._kernel = param
            print(f"DEBUG: Assigned parameter to self._kernel: {self._kernel}")
            print(f"DEBUG: self._kernel type: {type(self._kernel)}")
            if hasattr(self._kernel, 'data'):
                print(f"DEBUG: self._kernel.data type: {type(self._kernel.data)}")
                print(f"DEBUG: self._kernel.data shape: {tensor.shape(self._kernel.data)}")
            else:
                print(f"DEBUG: self._kernel has no data attribute")
        except Exception as e:
            print(f"DEBUG: Error creating kernel Parameter: {e}")
            raise

        try:
            print(f"DEBUG: Initializing recurrent_kernel with shape ({self.neuron_map.units}, {self.neuron_map.units}) using {self.recurrent_initializer}")
            tensor_data = self._initialize_tensor(
                (self.neuron_map.units, self.neuron_map.units),
                self.recurrent_initializer
            )
            print(f"DEBUG: Initialized recurrent tensor data with shape {tensor.shape(tensor_data)}")
            # Create the parameter and assign it to self._recurrent_kernel
            param = Parameter(tensor_data)
            print(f"DEBUG: Created recurrent_kernel Parameter: {param}")
            print(f"DEBUG: Parameter data type: {type(param.data)}")
            print(f"DEBUG: Parameter data shape: {tensor.shape(param.data)}")
            # Assign the parameter to self._recurrent_kernel
            self._recurrent_kernel = param
            print(f"DEBUG: Assigned parameter to self._recurrent_kernel: {self._recurrent_kernel}")
            print(f"DEBUG: self._recurrent_kernel type: {type(self._recurrent_kernel)}")
            if hasattr(self._recurrent_kernel, 'data'):
                print(f"DEBUG: self._recurrent_kernel.data type: {type(self._recurrent_kernel.data)}")
                print(f"DEBUG: self._recurrent_kernel.data shape: {tensor.shape(self._recurrent_kernel.data)}")
            else:
                print(f"DEBUG: self._recurrent_kernel has no data attribute")
        except Exception as e:
            print(f"DEBUG: Error creating recurrent_kernel Parameter: {e}")
            raise

        if self.use_bias:
            try:
                print(f"DEBUG: Initializing bias with shape ({self.neuron_map.units},) using {self.bias_initializer}")
                tensor_data = self._initialize_tensor(
                    (self.neuron_map.units,),
                    self.bias_initializer
                )
                print(f"DEBUG: Initialized bias tensor data with shape {tensor.shape(tensor_data)}")
                # Create the parameter and assign it to self._bias
                param = Parameter(tensor_data)
                print(f"DEBUG: Created bias Parameter: {param}")
                print(f"DEBUG: Parameter data type: {type(param.data)}")
                print(f"DEBUG: Parameter data shape: {tensor.shape(param.data)}")
                # Assign the parameter to self._bias
                self._bias = param
                print(f"DEBUG: Assigned parameter to self._bias: {self._bias}")
                print(f"DEBUG: self._bias type: {type(self._bias)}")
                if hasattr(self._bias, 'data'):
                    print(f"DEBUG: self._bias.data type: {type(self._bias.data)}")
                    print(f"DEBUG: self._bias.data shape: {tensor.shape(self._bias.data)}")
                else:
                    print(f"DEBUG: self._bias has no data attribute")
            except Exception as e:
                print(f"DEBUG: Error creating bias Parameter: {e}")
                raise
        else:
            self._bias = None
            print("DEBUG: Bias not used")

        self.built = True
        print("DEBUG: NCP module built successfully")
        # It's good practice to call super().build, although BaseModule's build is empty
        super().build(input_shape)


    def _initialize_tensor(self, shape, initializer):
        """Initialize a tensor with the specified shape and initializer."""
        print(f"DEBUG: _initialize_tensor called with shape={shape}, initializer={initializer}")
        try:
            # Use the get_initializer helper function from the initializers module
            print(f"DEBUG: Getting initializer function for '{initializer}'")
            initializer_fn = initializers.get_initializer(initializer)
            print(f"DEBUG: Got initializer function: {initializer_fn}")
            
            # Call the initializer function
            print(f"DEBUG: Calling initializer function with shape={shape}, dtype={self.dtype}")
            result = initializer_fn(shape, dtype=self.dtype)
            print(f"DEBUG: Initializer returned result of type {type(result)} with shape {tensor.shape(result)}")
            
            return result
        except ValueError as e:
            # If initializer is not recognized, provide a helpful error message
            print(f"DEBUG: Error getting initializer: {e}")
            raise ValueError(f"Unknown initializer '{initializer}' for NCP module. "
                           f"Available initializers: {', '.join(initializers._INITIALIZERS.keys())}") from e
        except Exception as e:
            print(f"DEBUG: Unexpected error in _initialize_tensor: {e}")
            raise
    
    def forward(
        self,
        inputs: Any,
        state: Optional[Any] = None,
        return_state: bool = False
    ) -> Union[Any, Tuple[Any, Any]]:
        # Ensure the layer is built before proceeding
        if not self.built:
             # Need the shape of the input tensor to build
             self.build(tensor.shape(inputs))

        """
        Forward pass of the NCP module.
        
        Args:
            inputs: Input tensor
            state: Optional state tensor
            return_state: Whether to return the state
            
        Returns:
            Output tensor, or tuple of (output, state) if return_state is True
        """
        if state is None:
            state = self.state
            
        # Apply input mask
        masked_inputs = ops.multiply(inputs, self.input_mask)
        
        # Apply recurrent mask
        masked_state = ops.matmul(state, self.recurrent_mask)
        
        # Compute new state
        print(f"DEBUG: Forward - masked_inputs shape: {tensor.shape(masked_inputs)}")
        print(f"DEBUG: Forward - self._kernel type: {type(self._kernel)} value: {self._kernel}")
        if self._kernel is not None:
            print(f"DEBUG: Forward - self._kernel.data shape: {tensor.shape(self._kernel.data)}")
        new_state = ops.matmul(masked_inputs, self._kernel)
        print(f"DEBUG: Forward - new_state after kernel matmul shape: {tensor.shape(new_state)}")
        
        if self.use_bias:
            print(f"DEBUG: Forward - self._bias type: {type(self._bias)} value: {self._bias}")
            if self._bias is not None:
                print(f"DEBUG: Forward - self._bias.data shape: {tensor.shape(self._bias.data)}")
            new_state = ops.add(new_state, self._bias)
            print(f"DEBUG: Forward - new_state after bias add shape: {tensor.shape(new_state)}")
            
        print(f"DEBUG: Forward - masked_state shape: {tensor.shape(masked_state)}")
        print(f"DEBUG: Forward - self._recurrent_kernel type: {type(self._recurrent_kernel)} value: {self._recurrent_kernel}")
        if self._recurrent_kernel is not None:
            print(f"DEBUG: Forward - self._recurrent_kernel.data shape: {tensor.shape(self._recurrent_kernel.data)}")
        recurrent_state = ops.matmul(masked_state, self._recurrent_kernel)
        print(f"DEBUG: Forward - recurrent_state shape: {tensor.shape(recurrent_state)}")
        
        new_state = ops.add(new_state, recurrent_state)
        print(f"DEBUG: Forward - new_state after recurrent add shape: {tensor.shape(new_state)}")
        
        # Apply activation function dynamically
        activation_fn = get_activation(self.activation) # Lookup happens here
        new_state = activation_fn(new_state)
        
        # Compute output - only include motor neurons
        # The output mask is a binary mask that selects only the motor neurons
        masked_output = ops.multiply(new_state, self.output_mask)
        
        # Extract only the motor neurons (first output_dim neurons)
        output = masked_output[:, :self.neuron_map.output_dim]
        
        # Update state
        self.state = new_state
        
        if return_state:
            return output, new_state
        else:
            return output
    
    def reset_state(self) -> None:
        """
        Reset the state of the NCP module.
        """
        self.state = tensor.zeros((1, self.neuron_map.units))
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the NCP module.
        
        Returns:
            Dictionary containing the configuration
        """
        config = {
            # Save map config and class name
            "neuron_map": self.neuron_map.get_config(),
            "neuron_map_class": self.neuron_map.__class__.__name__,
            "activation": self.activation, # Save the string name
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "dtype": self.dtype,
            "state": self.state
        }
        return config
    
    @classmethod
    # Need to update the constructor signature first before fixing from_config call
    # Let's update the __init__ signature in a separate step first.
    # For now, just fixing the import logic part.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCP':
        """
        Create an NCP module from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NCP module
        """
        # Config should contain the map configuration under 'neuron_map' key
        # and the map class name under 'neuron_map_class'
        map_config = config.pop("neuron_map") # Changed key from "wiring"
        map_class_name = config.pop("neuron_map_class") # Changed key from "wiring_class"

        # Directly import known map classes from their location in nn.modules.wiring
        from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap
        # AutoNCP is a layer, not a map, so it's not loaded here.

        # Map class name string to class object
        neuron_map_class_map = {
            "NeuronMap": NeuronMap,
            "NCPMap": NCPMap,
            "FullyConnectedMap": FullyConnectedMap,
            "RandomMap": RandomMap,
        }

        neuron_map_class_obj = neuron_map_class_map.get(map_class_name)
        if neuron_map_class_obj is None:
             raise ImportError(f"Unknown NeuronMap class '{map_class_name}' specified in config.")

        # Create the map instance using the remaining config params
        neuron_map = neuron_map_class_obj.from_config(map_config)

        # Create the NCP module
        config.pop('state', None) # Remove state before passing to constructor
        # Pass the created map object using the new argument name
        return cls(neuron_map=neuron_map, **config)