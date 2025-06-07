"""
Data types for ember_ml.nn.tensor.

This module provides a backend-agnostic data type system that can be used
across different backends.
"""

from typing import Any

from ember_ml.backend import get_backend, get_backend_module

# Cache for backend instances
_CURRENT_INSTANCES = {}

def _get_backend_module():
    """Get the current backend module."""
    return get_backend_module()

# Create a class that implements the DTypeInterface
class EmberDType:
    """
    A backend-agnostic data type.
    
    This class represents data types that can be used across different backends.
    """
    
    def __init__(self, name: str):
        """
        Initialize an EmberDType.
        
        Args:
            name: The name of the data type
        """
        self._name = name
        try:
            backend_module = _get_backend_module()
            if hasattr(backend_module, self._name):
                self._backend_dtype = getattr(backend_module, self._name)
            elif hasattr(backend_module, 'dtype_ops'):
                # Try to get the dtype from the dtype_ops
                dtype_ops = getattr(backend_module, 'dtype_ops')
                if hasattr(dtype_ops, self._name):
                    self._backend_dtype = getattr(dtype_ops, self._name)
                else:
                    # If all else fails, use the name as a string
                    self._backend_dtype = self._name
            else:
                # If all else fails, use the name as a string
                self._backend_dtype = self._name
        except:
            # If there's an error, use the name as a string
            self._backend_dtype = None
    @property
    def name(self) -> str:
        """Get the name of the data type."""
        return self._name
    
    def __repr__(self) -> str:
        """Return a string representation of the data type."""
        return self._name
    
    def __str__(self) -> str:
        """Return a string representation of the data type."""
        return self._name
    
    def __eq__(self, other: Any) -> bool:
        """Check if this EmberDType is equal to another object."""
        if isinstance(other, EmberDType):
            # Compare names directly if other is also an EmberDType
            return self._name == other._name
        elif isinstance(other, str):
            # Compare name if other is a string
            return self._name == other
        else:
            # Attempt to convert 'other' to its string representation using the backend
            try:
                # Get the backend helper's to_dtype_str method
                to_str_func = _get_backend_dtype().to_dtype_str
                other_str = to_str_func(other)
                # Compare names
                return self._name == other_str
            except (ValueError, TypeError, AttributeError):
                # If conversion fails or type is incompatible, they are not equal
                return False
    
    def __call__(self) -> Any:
        """Return the backend-specific data type."""
        # This is a convenience method to make the EmberDType callable
        # It allows us to use EmberDType instances as functions
        # For example: float32() instead of float32
        if self._backend_dtype is None:
            # Try to get the backend-specific dtype
            try:
                backend_module = _get_backend_module()
                if hasattr(backend_module, self._name):
                    self._backend_dtype = getattr(backend_module, self._name)
                elif hasattr(backend_module, 'dtype_ops'):
                    # Try to get the dtype from the dtype_ops
                    dtype_ops = getattr(backend_module, 'dtype_ops')
                    if hasattr(dtype_ops, self._name):
                        self._backend_dtype = getattr(dtype_ops, self._name)
                    else:
                        # If all else fails, use the name as a string
                        self._backend_dtype = self._name
                else:
                    # If all else fails, use the name as a string
                    self._backend_dtype = self._name
            except:
                # If there's an error, use the name as a string
                self._backend_dtype = self._name
        
        return self._backend_dtype

# Get the backend dtype class
def _get_backend_dtype():
    """Get the dtype class for the current backend."""
    backend = get_backend()
    if backend == 'torch':
        from ember_ml.backend.tensor.convert_to_tensor import TorchDType
        return TorchDType()
    elif backend == 'numpy':
        # This will be updated when numpy backend is refactored
        from ember_ml.backend.numpy.tensor import NumpyDType
        return NumpyDType()
    elif backend == 'mlx':
        # This will be updated when mlx backend is refactored
        from ember_ml.backend.mlx.tensor import MLXDType
        return MLXDType()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

# Define a function to get a data type from the backend
def get_dtype(name):
    """Get a data type by name from the current backend."""
    return _get_backend_dtype().get_dtype(name)

# Define a function to convert a dtype to a string
def to_dtype_str(dtype):
    """Convert a data type to a string."""
    # Handle EmberDType instances
    if isinstance(dtype, EmberDType):
        return dtype.name
    # Handle string dtypes
    return _get_backend_dtype().to_dtype_str(dtype)

# Define a function to convert a string to a dtype
def from_dtype_str(dtype_str):
    """Convert a string to a data type."""
    return _get_backend_dtype().from_dtype_str(dtype_str)

# Define a class to dynamically get data types from the backend
class DTypes:
    """Dynamic access to backend data types."""
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return _get_backend_dtype().float32
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return _get_backend_dtype().float64
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return _get_backend_dtype().int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return _get_backend_dtype().int64
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        return _get_backend_dtype().bool_
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return _get_backend_dtype().int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return _get_backend_dtype().int16
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return _get_backend_dtype().uint8
    
    @property
    def uint16(self):
        """Get the uint16 data type."""
        return _get_backend_dtype().uint16
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        return _get_backend_dtype().uint32
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        return _get_backend_dtype().uint64
    
    @property
    def float16(self):
        """Get the float16 data type."""
        return _get_backend_dtype().float16

# Create a singleton instance
dtypes = DTypes()

# Create a DType class with dynamic properties for each data type
class DType:
    """
    Data type class with dynamic properties for each data type.
    
    This class provides properties for each data type that can be used
    directly without calling them. The properties are dynamically created
    based on the available data types in the current backend.
    """
    
    def __init__(self):
        """Initialize the DType class with dynamic properties."""
        # Initialize the data dictionary
        self._data = {}
        
        # Define the standard data types that should be available in all backends
        self._standard_dtypes = [
            'float32', 'float64', 'int32', 'int64', 'bool_',
            'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16'
        ]
        
        # Create EmberDType instances for each standard data type
        for dtype_name in self._standard_dtypes:
            self._data[dtype_name] = EmberDType(dtype_name)
    
    def __getattr__(self, name):
        """
        Get a data type by name.
        
        This method is called when an attribute is not found. It tries to get
        the data type from the data dictionary.
        
        Args:
            name: The name of the data type
            
        Returns:
            The data type
            
        Raises:
            AttributeError: If the data type is not found
        """
        if name in self._data:
            return self._data[name]
        else:
            # If the data type is not in the dictionary, create it
            self._data[name] = EmberDType(name)
            return self._data[name]
    
    def __setattr__(self, name, value):
        """
        Set a data type by name.
        
        This method is called when an attribute is set. It sets the data type
        in the data dictionary.
        
        Args:
            name: The name of the data type
            value: The data type value
        """
        if name.startswith('_'):
            # Allow setting private attributes directly
            super().__setattr__(name, value)
        else:
            # Set the data type in the data dictionary
            self._data[name] = value

# Create a singleton instance
dtype = DType()

# Create EmberDType instances for common data types
float32 = EmberDType('float32')
float64 = EmberDType('float64')
int32 = EmberDType('int32')
int64 = EmberDType('int64')
bool_ = EmberDType('bool_')
int8 = EmberDType('int8')
int16 = EmberDType('int16')
uint8 = EmberDType('uint8')
uint16 = EmberDType('uint16')
uint32 = EmberDType('uint32')
uint64 = EmberDType('uint64')
float16 = EmberDType('float16')
# Define a list of all available data types and functions
__all__ = [
    'EmberDType',
    'DType',
    'dtype',
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16',
]