"""
Test the EmberTensor API structure after the changes to remove _convert_to_backend_tensor
from the public API.
"""

import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor


@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = ops.get_backend()
    yield original
    # Ensure original is not None before setting it
    if original is not None:
        ops.set_backend(original)
    else:
        # Default to 'numpy' if original is None
        ops.set_backend('numpy')


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_convert_to_tensor_api_structure(backend_name, original_backend):
    """Test the API structure of convert_to_tensor and EmberTensor."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # 1. Test that _convert_to_backend_tensor is not in the public API
        import ember_ml.nn.tensor as tensor_module
        assert '_convert_to_backend_tensor' not in tensor_module.__all__
        with pytest.raises(AttributeError):
            tensor_module._convert_to_backend_tensor
        
        # 2. Test that convert_to_tensor returns an EmberTensor object
        data = [1, 2, 3]
        t = tensor.convert_to_tensor(data)
        assert isinstance(t, EmberTensor)
        
        # 3. Test that the EmberTensor object has a backend property
        assert hasattr(t, 'backend')
        assert t.backend == backend_name
        
        # 4. Test that the EmberTensor object has a to_backend_tensor method
        assert hasattr(t, 'to_backend_tensor')
        backend_tensor = t.to_backend_tensor()
        
        # 5. Test that basic operations work with the EmberTensor
        t2 = tensor.convert_to_tensor([4, 5, 6])
        result = ops.add(t, t2)
        assert isinstance(result, EmberTensor)
        
        # 6. Test that the internal _convert_to_backend_tensor is still accessible to internal code
        from ember_ml.nn.tensor.common import _convert_to_backend_tensor
        backend_tensor_direct = _convert_to_backend_tensor(data)
        # We can't directly compare backend tensors, but we can wrap it in an EmberTensor
        # and verify it has the same shape
        wrapped = EmberTensor(backend_tensor_direct)
        assert wrapped.shape == (3,)
        
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")