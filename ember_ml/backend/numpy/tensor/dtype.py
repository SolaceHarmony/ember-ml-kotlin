"""
NumPy data type implementation for ember_ml.

This module provides NumPy implementations of data type operations.
"""

import numpy as np
from typing import Union, Any, Optional

class NumpyDType:
    """NumPy implementation of data type operations."""

    @property
    def float16(self):
        """Get the float16 data type."""
        return np.float16
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return np.float32
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return np.float64
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return np.int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return np.int16
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return np.int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return np.int64
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return np.uint8
    
    @property
    def uint16(self):
        """Get the uint16 data type."""
        return np.uint16
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        return np.uint32
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        return np.uint64
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        return np.bool_
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return self.from_dtype_str(name)

    def to_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[str]:
        """
        Convert a NumPy data type to a dtype string.
        
        Args:
            dtype: The NumPy data type to convert
            
        Returns:
            The corresponding dtype string
        """
        if dtype is None:
            return None

        # If it's already a string, return it
        if isinstance(dtype, str):
            return dtype
            
        # Map standard NumPy dtype names (strings) to EmberDType names (strings)
        # This avoids issues with comparing type objects directly.
        dtype_name_map = {
            'float16': 'float16',
            'float32': 'float32',
            'float64': 'float64',
            'int8': 'int8',
            'int16': 'int16',
            'int32': 'int32',
            'int64': 'int64',
            'uint8': 'uint8',
            'uint16': 'uint16',
            'uint32': 'uint32',
            'uint64': 'uint64',
            'bool': 'bool_', # Map numpy 'bool' to standard 'bool_'
            'bool_': 'bool_' # Handle np.bool_ type as well
            # Add complex types if needed
            # 'complex64': 'complex64',
            # 'complex128': 'complex128',
        }

        # Canonicalize the input dtype and get its standard NumPy name
        try:
            input_dtype_obj = np.dtype(dtype)
            input_dtype_name = input_dtype_obj.name
        except TypeError:
             raise ValueError(f"Input '{dtype}' cannot be interpreted as a NumPy dtype.")

        # Perform lookup using the canonical name string
        if input_dtype_name in dtype_name_map:
            return dtype_name_map[input_dtype_name]
        else:
            # Add more detail to the error message
            raise ValueError(f"Cannot convert NumPy dtype '{input_dtype_obj}' (name: {input_dtype_name}) to standardized string representation.")
    
    def validate_dtype(self, dtype: Optional[Any]) -> Optional[Any]:
        """
        Validate and convert dtype to NumPy format.
        
        Args:
            dtype: Input dtype to validate
            
        Returns:
            Validated NumPy dtype or None
        """
        if dtype is None:
            return None
        
        # Handle string dtypes
        if isinstance(dtype, str):
            return self.from_dtype_str(dtype)
            
        # Handle EmberDType objects
        if hasattr(dtype, 'name'):
            return self.from_dtype_str(str(dtype.name))
            
        # If it's already a NumPy dtype, return as is
        if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64,
                                                  np.bool_, np.int8, np.int16, np.uint8,
                                                  np.uint16, np.uint32, np.uint64, np.float16]:
            return dtype
            
        raise ValueError(f"Invalid dtype: {dtype}")
    
    def from_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[np.dtype]: # Ensure return type is native NumPy dtype object
        """
        Convert a dtype string or EmberDType object to a native NumPy data type object.

        Args:
            dtype: The dtype string or EmberDType object to convert

        Returns:
            The corresponding native NumPy data type object (e.g., np.float32, np.int64)
        """
        if dtype is None:
            return None

        # If it's already a NumPy dtype object or type, return it
        if isinstance(dtype, np.dtype):
             # Already a dtype object, return as is
             return dtype
        if isinstance(dtype, type) and issubclass(dtype, np.generic):
             # It's a type like np.float32, convert to dtype object
             return np.dtype(dtype)

        dtype_name_str = None
        # If it's a string, use it directly
        if isinstance(dtype, str):
            dtype_name_str = dtype
        # If it's an EmberDType instance, get its name attribute
        elif hasattr(dtype, '__class__') and dtype.__class__.__name__ == 'EmberDType' and hasattr(dtype, 'name'):
             dtype_name_str = dtype.name
        else:
            # If it's some other object that might represent a dtype (like directly passing np.float32 class)
            # try getting its name, otherwise raise error
            try:
                 # Attempt to get a standard name if possible (e.g., from np.float32.__name__)
                 if hasattr(dtype, '__name__'):
                      dtype_name_str = dtype.__name__
                 else: # Last resort, convert to string and hope it matches
                      dtype_name_str = str(dtype)
            except Exception:
                 raise ValueError(f"Cannot convert input '{dtype}' (type: {type(dtype)}) to a NumPy data type name.")

        # Map standard string names to NumPy native type objects
        dtype_map = {
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'int64': np.int64,
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'bool_': np.bool_,
            'bool': np.bool_, # Allow 'bool' as well
            'complex64': np.complex64,
            'complex128': np.complex128,
        }

        if dtype_name_str in dtype_map:
            return dtype_map[dtype_name_str]
        else:
            # Maybe it's a direct numpy name like 'i4', try np.dtype()
            try:
                 return np.dtype(dtype_name_str) # Return the dtype object (not the type)
            except TypeError:
                 raise ValueError(f"Unknown or uninterpretable data type string: '{dtype_name_str}'")