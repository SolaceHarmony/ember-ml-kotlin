"""
Base Module class for neural network components.

This module provides the foundation for building neural network components
that work with any backend (NumPy, PyTorch, MLX).

The BaseModule class is the base class for all neural network modules in ember_ml.
"""

from collections import OrderedDict
from typing import Dict, Iterator, Optional, Set, Tuple, Union, Any, List

from ember_ml import ops
from ember_ml.nn import tensor

class Parameter:
    """
    A special kind of tensor that represents a trainable parameter.
    
    Parameters are tensors that require gradients and are updated during
    the optimization process.
    """
    def __init__(self, data, requires_grad=True):
        """
        Initialize a parameter with data.

        Args:
            data: Initial data for the parameter
            requires_grad: Whether the parameter requires gradients
        """
        from ember_ml.nn import tensor # Use lazy import

        # Convert data to EmberTensor first
        ember_tensor = tensor.convert_to_tensor(data)
        
        # Get the underlying backend tensor (TensorLike, tensor.convert_to_tensor, TensorLike)
        # This ensures .data is the native backend tensor, which is what the tests expect
        self.data = ember_tensor._tensor
        
        # Store requires_grad and initialize grad
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        # Use try-except for robustness as self.data might not be initialized
        # when repr is called (e.g., during debugging or errors in __init__)
        try:
            # Attempt to get shape using lazy imported tensor object
            from ember_ml.nn import tensor
            shape_val = tensor.shape(self.data)
        except AttributeError:
            shape_val = "[AttributeError: data not found]" # Provide informative fallback
        except Exception as e: # Catch other potential errors during shape access
            shape_val = f"[Error getting shape: {e}]"
        try:
            # Attempt to get dtype using lazy imported tensor object
            from ember_ml.nn import tensor
            dtype_val = tensor.dtype(self.data)
            return f"Parameter(shape={shape_val}, dtype={dtype_val})"
        except AttributeError:
             # This might happen if shape failed and shape_val is a string
             return f"Parameter(shape={shape_val})"
        except Exception as e: # Catch other potential errors during dtype access
             return f"Parameter(shape={shape_val}, dtype_error: {e})"
        try:
            dtype_val = tensor.dtype(self.data)
            return f"Parameter(shape={shape_val}, dtype={dtype_val})"
        except AttributeError:
             return f"Parameter(shape={shape_val})" # Fallback if dtype access also fails
        except Exception as e: # Catch other potential errors during repr
             return f"Parameter(shape={shape_val}, repr_error: {e})"


class BaseModule:
    """
    Base class for all neural network modules in ember_ml.
    
    All custom modules should subclass this class and override the forward method.
    This class provides the foundation for building neural network components
    that work with any backend (NumPy, PyTorch, MLX).
    """
    
    def __init__(self):
        """Initialize the module."""
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True
        self.built = False # Flag for deferred build
    
    def forward(self, *args, **kwargs):
        """
        Define the computation performed at every call.
        
        This method should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def build(self, input_shape):
        """
        Creates the layer's weights.

        This method should be overridden by subclasses that need to create
        weights based on the shape of the input tensor. The framework calls
        this method automatically before the first `forward` pass.

        Args:
            input_shape: A shape tuple (or list of tuples) reporting the
                         shape of the input tensor(s).
        """
        # Base implementation does nothing. Subclasses override this.
        # Subclasses should set self.built = True if they implement build.
        # However, the main built flag is set in __call__.
        pass

    def __call__(self, *args, **kwargs):
        """
        Call the module on inputs, handling deferred build.

        This method calls the forward method after ensuring the module is built.
        """
        # Check if built, and if inputs are provided to trigger build
        if not self.built and args:
            # Determine input shape from the first argument primarily
            # Assumes the first argument is the main input tensor or a tuple/list of them
            # More complex input structures might need custom handling in __call__ overrides

            if isinstance(args[0], (list, tuple)):
                 # Handle multiple inputs - pass list/tuple of shapes
                 try:
                     input_shape = [tensor.shape(inp) for inp in args[0]]
                     # Convert list of tuples to tuple of tuples if needed, or handle as list
                     input_shape = tuple(input_shape) if len(input_shape) > 1 else input_shape[0]
                 except Exception as e:
                     raise ValueError(f"Could not determine input shape from list/tuple input {args[0]}: {e}")
            else:
                 # Assume single input tensor
                 try:
                     input_shape = tensor.shape(args[0])
                 except Exception as e:
                     raise ValueError(f"Could not determine input shape from input {args[0]}: {e}")

            # Call the build method
            self.build(input_shape)
            # Mark as built AFTER the build method completes successfully
            self.built = True

        # Execute the forward pass
        return self.forward(*args, **kwargs)
    
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Register a parameter with the module.
        
        Args:
            name: Name of the parameter
            param: Parameter to register, or None to remove
        """
        if param is None:
            self._parameters.pop(name, None)
        else:
            self._parameters[name] = param
    
    def register_buffer(self, name: str, buffer: Any) -> None:
        """
        Register a buffer with the module.
        
        Buffers are tensors that are not considered parameters but are part of the
        module's state, such as running means in batch normalization.
        
        Args:
            name: Name of the buffer
            buffer: Buffer to register, or None to remove
        """
        if buffer is None:
            self._buffers.pop(name, None)
        else:
            self._buffers[name] = tensor.convert_to_tensor(buffer)
    
    def add_module(self, name: str, module: Optional['BaseModule']) -> None:
        """
        Register a submodule with the module.
        
        Args:
            name: Name of the submodule
            module: Module to register, or None to remove
        """
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Return an iterator over module parameters, yielding both the name and the parameter.
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, yield parameters of submodules
            
        Yields:
            (name, parameter) pairs
        """
        for name, param in self._parameters.items():
            yield prefix + ('.' if prefix else '') + name, param
        
        if recurse:
            for module_name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + module_name
                for name, param in module.named_parameters(submodule_prefix, recurse):
                    yield name, param
    
    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """
        Return a list of module parameters.
        
        Args:
            recurse: If True, include parameters of submodules
            
        Returns:
            A list of module parameters
        """
        params_list = []
        for _, param in self.named_parameters(recurse=recurse):
            params_list.append(param)
        return params_list
    
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Any]]:
        """
        Return an iterator over module buffers, yielding both the name and the buffer.
        
        Args:
            prefix: Prefix to prepend to buffer names
            recurse: If True, yield buffers of submodules
            
        Yields:
            (name, buffer) pairs
        """
        for name, buf in self._buffers.items():
            yield prefix + ('.' if prefix else '') + name, buf
        
        if recurse:
            for module_name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + module_name
                for name, buf in module.named_buffers(submodule_prefix, recurse):
                    yield name, buf
    
    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        """
        Return an iterator over module buffers.
        
        Args:
            recurse: If True, yield buffers of submodules
            
        Yields:
            Module buffers
        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf
    
    def named_modules(self, prefix: str = '', memo: Optional[Set['BaseModule']] = None) -> Iterator[Tuple[str, 'BaseModule']]:
        """
        Return an iterator over all modules in the network, yielding both the name and the module.
        
        Args:
            prefix: Prefix to prepend to module names
            memo: Set of modules already yielded
            
        Yields:
            (name, module) pairs
        """
        if memo is None:
            memo = set()
        
        if self not in memo:
            memo.add(self)
            yield prefix, self
            
            for module_name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + module_name
                for name, mod in module.named_modules(submodule_prefix, memo):
                    yield name, mod
    
    def modules(self) -> Iterator['BaseModule']:
        """
        Return an iterator over all modules in the network.
        
        Yields:
            Modules in the network
        """
        for _, module in self.named_modules():
            yield module
    
    def train(self, mode: bool = True) -> 'BaseModule':
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        self.training = mode
        for module in self.modules():
            module.training = mode
        return self
    
    def eval(self) -> 'BaseModule':
        """
        Set the module in evaluation mode.
        
        Returns:
            self
        """
        return self.train(False)
    
    def to(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> 'BaseModule':
        """
        Move and/or cast the parameters and buffers.
        
        Args:
            device: Device to move parameters and buffers to
            dtype: Data type to cast parameters and buffers to
            
        Returns:
            self
        """
        for param in self.parameters():
            if dtype is not None:
                param.data = tensor.cast(param.data, dtype)
            if device is not None:
                param.data = ops.to_device(param.data, device)
        
        for key, buf in self._buffers.items():
            if dtype is not None:
                self._buffers[key] = tensor.cast(buf, dtype)
            if device is not None:
                self._buffers[key] = ops.to_device(buf, device)
        
        return self
    
    def __repr__(self):
        """Return a string representation of the module."""
        lines = [self.__class__.__name__ + '(']
        
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '  ' + mod_str.replace('\n', '\n  ')
            lines.append('(' + name + '): ' + mod_str)
        
        for name, param in self._parameters.items():
            lines.append('(' + name + '): ' + repr(param))
        
        lines.append(')')
        return '\n'.join(lines)
    
    def __setattr__(self, name, value):
        """Set an attribute on the module."""
        if isinstance(value, Parameter):
            # Register in parameter dictionary
            self.register_parameter(name, value)
            # ALSO set the actual attribute for direct access
            object.__setattr__(self, name, value)
        elif isinstance(value, BaseModule):
            self.add_module(name, value)
            # Similarly for modules
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the module."""
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = tensor.zeros_like(param.grad)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the module.

        Subclasses should override this method to return a dictionary
        containing the arguments needed to reconstruct the module.

        Returns:
            A dictionary containing the module's configuration.
        """
        # Base implementation returns an empty dict.
        # Subclasses should call super().get_config() and update the dict.
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModule':
        """Creates a module from its configuration.

        This method is the reverse of get_config().

        Args:
            config: A Python dictionary containing the configuration of the module.

        Returns:
            A module instance.
        """
        # Simple base implementation assumes config directly maps to __init__ args.
        # Subclasses might need to override this if reconstruction is more complex
        # (e.g., reconstructing nested modules/maps first).
        # We remove 'class_name' if present, as it's often added during saving.
        config.pop('class_name', None)
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when creating {cls.__name__} from config: {config}\n"
                f"Exception: {e}"
            )