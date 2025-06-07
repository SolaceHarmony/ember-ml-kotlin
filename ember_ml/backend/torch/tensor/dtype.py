"""
PyTorch data type implementation for ember_ml.

This module provides PyTorch implementations of data types.
"""

import torch
from typing import Union, Any, Optional


class TorchDType:
    """PyTorch implementation of data types."""
    
    @property
    def float16(self):
        """Get the float16 data type."""
        return torch.float16
    
    @property
    def float32(self):
        """Get the float32 data type."""
        return torch.float32
    
    @property
    def float64(self):
        """Get the float64 data type."""
        return torch.float64
    
    @property
    def int8(self):
        """Get the int8 data type."""
        return torch.int8
    
    @property
    def int16(self):
        """Get the int16 data type."""
        return torch.int16
    
    @property
    def int32(self):
        """Get the int32 data type."""
        return torch.int32
    
    @property
    def int64(self):
        """Get the int64 data type."""
        return torch.int64
    
    @property
    def uint8(self):
        """Get the uint8 data type."""
        return torch.uint8
    
    @property
    def uint16(self):
        """Get the uint16 data type."""
        # PyTorch doesn't have uint16, so we'll use int16 as a fallback
        return torch.int16
    
    @property
    def uint32(self):
        """Get the uint32 data type."""
        # PyTorch doesn't have uint32, so we'll use int32 as a fallback
        return torch.int32
    
    @property
    def uint64(self):
        """Get the uint64 data type."""
        # PyTorch doesn't have uint64, so we'll use int64 as a fallback
        return torch.int64
    
    @property
    def bool_(self):
        """Get the boolean data type."""
        return torch.bool
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return self.from_dtype_str(name)
    
    def to_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[str]: # Changed torch.dtype to Any
        """
        Convert a PyTorch data type to a dtype string.
        
        Args:
            dtype: The PyTorch data type to convert
            
        Returns:
            The corresponding dtype string
        """
        if dtype is None:
            return None
        
        # Map PyTorch dtypes to EmberDType names
        if dtype == torch.float32:
            return 'float32'
        elif dtype == torch.float64:
            return 'float64'
        elif dtype == torch.int32:
            return 'int32'
        elif dtype == torch.int64:
            return 'int64'
        elif dtype == torch.bool:
            return 'bool_' # Standardize to bool_ to match EmberDType/NumPy
        elif dtype == torch.int8:
            return 'int8'
        elif dtype == torch.int16:
            return 'int16'
        elif dtype == torch.uint8:
            return 'uint8'
        elif dtype == torch.float16:
            return 'float16'
        elif isinstance(dtype, str):
            # If it's already a string, return it
            return dtype
        else:
            raise ValueError(f"Cannot convert {dtype} to EmberDType")
    def validate_dtype(self, dtype: Union[Any, str, None]) -> Optional[Any]: # Changed torch.dtype to Any
        """
        Validate and convert dtype to Torch format.

        Args:
            dtype: Input dtype to validate (EmberDType, torch.dtype, str, or None).

        Returns:
            Validated torch.dtype object or None.

        Raises:
            ValueError: If the dtype is invalid or cannot be converted.
        """
        if dtype is None:
            return None

        # 1. Handle EmberDType instances
        if (hasattr(dtype, '__class__') and
            hasattr(dtype.__class__, '__name__') and
            dtype.__class__.__name__ == 'EmberDType'):
            backend_dtype_val = getattr(dtype, '_backend_dtype', None)
            if isinstance(backend_dtype_val, torch.dtype):
                return backend_dtype_val  # Already correct torch.dtype
            elif isinstance(backend_dtype_val, str):
                try:
                    # Attempt conversion from string representation inside EmberDType
                    return self.from_dtype_str(backend_dtype_val)
                except ValueError as e:
                    # If conversion fails, raise an error clearly indicating the source
                    raise ValueError(f"Invalid _backend_dtype string '{backend_dtype_val}' in EmberDType {dtype}: {e}") from e
            else:
                # If _backend_dtype is not torch.dtype or str (e.g., None, other type)
                 raise ValueError(f"Invalid _backend_dtype '{backend_dtype_val}' (type: {type(backend_dtype_val)}) in EmberDType {dtype}")

        # 2. Handle direct torch.dtype instances
        # Also check known torch types explicitly in case isinstance fails for some reason
        if isinstance(dtype, torch.dtype) or dtype in [
            torch.float16, torch.float32, torch.float64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool
        ]:
            # Ensure we return the actual torch.dtype object
            if not isinstance(dtype, torch.dtype):
                 pass # Assume 'dtype' is the correct torch.dtype object here
            return dtype  # type: ignore[return-value]

        # 3. Handle string representations
        if isinstance(dtype, str):
            try:
                return self.from_dtype_str(dtype)
            except ValueError as e:
                raise ValueError(f"Invalid dtype string: '{dtype}': {e}") from e

        # 4. If none of the above match, raise an error
        raise ValueError(f"Unsupported dtype type: {type(dtype)}, value: {dtype}")


    
    def from_dtype_str(self, dtype: Union[Any, str, None]) -> Optional[Any]: # Changed torch.dtype to Any
        """
        Convert a dtype string to a PyTorch data type.
        
        Args:
            dtype: The dtype string to convert
            
        Returns:
            The corresponding PyTorch data type
        """
        if dtype is None:
            return None
        
        # If it's already a PyTorch dtype, return it
        if isinstance(dtype, torch.dtype):
            return dtype
        
        # If it's a string, use it directly
        if isinstance(dtype, str):
            dtype_name = dtype
        # If it has a name attribute, use that
        elif hasattr(dtype, 'name'):
            dtype_name = dtype.name
        else:
            raise ValueError(f"Cannot convert {dtype} to PyTorch data type")
        
        # Map dtype names to PyTorch dtypes
        if dtype_name == 'float32':
            return torch.float32
        elif dtype_name == 'float64':
            return torch.float64
        elif dtype_name == 'int32':
            return torch.int32
        elif dtype_name == 'int64':
            return torch.int64
        elif dtype_name == 'bool' or dtype_name == 'bool_':
            return torch.bool
        elif dtype_name == 'int8':
            return torch.int8
        elif dtype_name == 'int16':
            return torch.int16
        elif dtype_name == 'uint8':
            return torch.uint8
        elif dtype_name == 'float16':
            return torch.float16
        else:
            raise ValueError(f"Unknown data type: {dtype_name}")