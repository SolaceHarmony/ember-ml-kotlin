"""
Test that the internal _convert_to_backend_tensor function is accessible to internal code
but not exported in the public API.
"""

import pytest
from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor


def test_internal_convert_to_backend_tensor_not_exported():
    """Test that _convert_to_backend_tensor is not exported in the public API."""
    # Import the tensor module
    import ember_ml.nn.tensor as tensor_module
    
    # Check that _convert_to_backend_tensor is not in the module's __all__ list
    assert '_convert_to_backend_tensor' not in tensor_module.__all__
    
    # Check that _convert_to_backend_tensor is not directly accessible from the module
    with pytest.raises(AttributeError):
        tensor_module._convert_to_backend_tensor


def test_internal_convert_to_backend_tensor_accessible_internally():
    """Test that _convert_to_backend_tensor is accessible to internal code."""
    # Import the internal function directly
    from ember_ml.nn.tensor.common import _convert_to_backend_tensor
    
    # Test that it works correctly
    data = [1, 2, 3]
    backend_tensor = _convert_to_backend_tensor(data)
    
    # Verify it's a backend tensor by wrapping it in an EmberTensor
    # and checking properties
    ember_tensor = EmberTensor(backend_tensor)
    assert ember_tensor.shape == (3,)