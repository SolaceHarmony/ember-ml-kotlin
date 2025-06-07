"""
MLX data type implementation for ember_ml.

This module provides MLX implementations of data type operations.
"""

import mlx.core as mx
from typing import Union, Any, Optional

class MLXDType:
    """MLX implementation of data type operations."""

    @property
    def float16(self):
        """Get the float16 data type."""
        return mx.float16
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return mx.float32
    
    @property
    def float64(self):
         """Get the float64 data type."""
         return mx.float32
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return mx.int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return mx.int16
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return mx.int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return mx.int64
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return mx.uint8
    
    @property 
    def uint16(self):
        """Get the uint16 data type."""
        return mx.uint16
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        return mx.uint32
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        return mx.uint64
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        # MLX now has bool_ type, but we need to ensure it's used consistently
        if hasattr(mx, 'bool_'):
            return mx.bool_
        else:
            return mx.uint8  # Fallback to uint8 if bool_ not available
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return self.from_dtype_str(name)

    def to_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[str]:
        """
        Convert an MLX data type to a dtype string.
        
        Args:
            dtype: The MLX data type to convert
            
        Returns:
            The corresponding dtype string
        """
        if dtype is None:
            return None

        # If it's already a string, return it
        if isinstance(dtype, str):
            return dtype
            
        # Handle EmberDType objects
        if hasattr(dtype, 'name'):
            return dtype.name
            
        # Map MLX dtypes to EmberDType names
        dtype_map = {
            mx.float16: 'float16',
            mx.float32: 'float32',
            mx.int8: 'int8',
            mx.int16: 'int16',
            mx.int32: 'int32',
            mx.int64: 'int64',
            mx.uint8: 'uint8',
            mx.uint16: 'uint16',
            mx.uint32: 'uint32',
            mx.uint64: 'uint64'
        }

        # Add bool type if available
        if hasattr(mx, 'bool_'):
            dtype_map[mx.bool_] = 'bool_'  # Use 'bool_' to match EmberDType naming
        
        # Special handling for MLX dtype objects
        if hasattr(dtype, 'name') and isinstance(dtype.name, str):
            # Map MLX dtype names to EmberDType names
            mlx_to_ember = {
                'float16': 'float16',
                'float32': 'float32',
                'int8': 'int8',
                'int16': 'int16',
                'int32': 'int32',
                'int64': 'int64',
                'uint8': 'uint8',
                'uint16': 'uint16',
                'uint32': 'uint32',
                'uint64': 'uint64',
                'bool': 'bool_',
                'bool_': 'bool_'
            }
            if dtype.name in mlx_to_ember:
                return mlx_to_ember[dtype.name]

        if dtype in dtype_map:
            return dtype_map[dtype]
        else:
            # If we can't determine the dtype, return a default
            return str(dtype)
    # Removed validate_dtype method, logic moved to utility.py
    def from_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[Any]:
        """
        Convert a dtype string to an MLX data type.
        
        Args:
            dtype: The dtype string to convert
            
        Returns:
            The corresponding MLX data type
        """
        if dtype is None:
            return None
            
        # If it's already an MLX dtype, return it
        if hasattr(dtype, '__class__') and dtype.__class__.__module__ == 'mlx.core' and hasattr(dtype, 'name'):
            return dtype
            
        # If it's a string, use it directly
        if isinstance(dtype, str):
            dtype_name = dtype
        # If it has a name attribute, use that
        elif hasattr(dtype, 'name'):
            dtype_name = dtype.name
        else:
            raise ValueError(f"Cannot convert {dtype} to MLX data type")
            
        # Map dtype names to MLX dtypes
        dtype_map = {
            'float16': mx.float16,
            'float32': mx.float32,
            'float64': mx.float32,  # Map float64 to float32 for MLX
            'int8': mx.int8,
            'int16': mx.int16,
            'int32': mx.int32,
            'int64': mx.int64,
            'uint8': mx.uint8,
            'uint16': mx.uint16,
            'uint32': mx.uint32,
            'uint64': mx.uint64,
            'complex64': mx.complex64 if hasattr(mx, 'complex64') else None,
            'bool': mx.bool_ if hasattr(mx, 'bool_') else mx.uint8,
            'bool_': mx.bool_ if hasattr(mx, 'bool_') else mx.uint8
        }
        
        if dtype_name in dtype_map:
            result = dtype_map[dtype_name]
            if result is None:
                raise ValueError(f"Data type {dtype_name} is not supported in this MLX version")
            return result
        else:
            raise ValueError(f"Unknown data type: {dtype_name}")