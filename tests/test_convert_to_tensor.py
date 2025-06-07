"""
Test to check if tensor.convert_to_tensor returns an EmberTensor object.
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


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_convert_to_tensor_returns_ember_tensor(backend_name, original_backend):
    """Test if tensor.convert_to_tensor returns an EmberTensor object."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor using convert_to_tensor
        data = [1, 2, 3]
        t = tensor.convert_to_tensor(data)
        
        # Check if it's an EmberTensor object
        print(f"Type of t: {type(t)}")
        print(f"Is instance of EmberTensor: {isinstance(t, EmberTensor)}")
        
        # Check if it has EmberTensor attributes
        print(f"Has shape attribute: {hasattr(t, 'shape')}")
        if hasattr(t, 'shape'):
            print(f"Shape: {t.shape}")
        
        print(f"Has dtype attribute: {hasattr(t, 'dtype')}")
        if hasattr(t, 'dtype'):
            print(f"Dtype: {t.dtype}")
        
        # This assertion will fail if t is not an EmberTensor
        assert isinstance(t, EmberTensor), f"Expected EmberTensor, got {type(t)}"
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")