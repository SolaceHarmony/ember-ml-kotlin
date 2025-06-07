"""
Test the ops module functionality across all backends.

This module tests the basic functionality of the ops module using pytest.
"""

import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.backend import set_backend, get_backend

# Test with all available backends
BACKENDS = ['numpy', 'torch', 'mlx']

@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = get_backend()
    yield original
    # Restore original backend
    if original is not None:
        set_backend(original)
    else:
        # Default to 'numpy' if original is None
        set_backend('numpy')

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_tensor_creation(backend_name, original_backend):
    """Test tensor creation operations."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Test zeros
        zeros = tensor.zeros((2, 3))
        assert tensor.shape(zeros) == (2, 3)
        assert ops.all(ops.equal(zeros, tensor.zeros_like(zeros)))
        
        # Test ones
        ones = tensor.ones((2, 3))
        assert tensor.shape(ones) == (2, 3)
        assert ops.all(ops.equal(ones, tensor.ones_like(ones)))
        
        # Test convert_to_tensor
        data = [1, 2, 3]
        t = tensor.convert_to_tensor(data)
        assert tensor.shape(t) == (3,)
        assert ops.all(ops.equal(t, tensor.convert_to_tensor([1, 2, 3])))
        
        # Test reshape
        reshaped = tensor.reshape(ones, (3, 2))
        assert tensor.shape(reshaped) == (3, 2)
        
        # Test eye
        eye = tensor.eye(3)
        assert tensor.shape(eye) == (3, 3)
        assert ops.equal(tensor.diag(eye), tensor.ones(3)).numpy().all()
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_math_operations(backend_name, original_backend):
    """Test math operations."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create test tensors
        a = tensor.convert_to_tensor([1, 2, 3])
        b = tensor.convert_to_tensor([4, 5, 6])
        
        # Test basic arithmetic
        assert ops.all(ops.equal(ops.add(a, b), tensor.convert_to_tensor([5, 7, 9])))
        assert ops.all(ops.equal(ops.subtract(a, b), tensor.convert_to_tensor([-3, -3, -3])))
        assert ops.all(ops.equal(ops.multiply(a, b), tensor.convert_to_tensor([4, 10, 18])))
        assert ops.allclose(ops.divide(a, b), tensor.convert_to_tensor([0.25, 0.4, 0.5]))
        
        # Test exponential and logarithmic functions
        assert ops.allclose(ops.exp(a), tensor.convert_to_tensor([2.718282, 7.389056, 20.085537]))
        assert ops.allclose(ops.log(a), tensor.convert_to_tensor([0., 0.693147, 1.098612]))
        
        # Test power functions
        assert ops.allclose(ops.pow(a, b), tensor.convert_to_tensor([1, 32, 729]))
        assert ops.allclose(ops.sqrt(tensor.convert_to_tensor([1, 4, 9])), tensor.convert_to_tensor([1, 2, 3]))
        
        # Test trigonometric functions
        x = tensor.convert_to_tensor([0, 1, 2])
        assert ops.allclose(ops.sin(x), tensor.convert_to_tensor([0, 0.841471, 0.909297]), atol=1e-5)
        assert ops.allclose(ops.cos(x), tensor.convert_to_tensor([1, 0.540302, -0.416147]), atol=1e-5)
        
        # Test activation functions
        assert ops.allclose(ops.sigmoid(x), tensor.convert_to_tensor([0.5, 0.731059, 0.880797]), atol=1e-5)
        assert ops.all(ops.equal(ops.relu(tensor.convert_to_tensor([-1, 0, 1])), tensor.convert_to_tensor([0, 0, 1])))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_comparison_operations(backend_name, original_backend):
    """Test comparison operations."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create test tensors
        a = tensor.convert_to_tensor([1, 2, 3])
        b = tensor.convert_to_tensor([3, 2, 1])
        
        # Test comparison operations
        assert ops.all(ops.equal(ops.equal(a, b), tensor.convert_to_tensor([False, True, False])))
        assert ops.all(ops.equal(ops.not_equal(a, b), tensor.convert_to_tensor([True, False, True])))
        assert ops.all(ops.equal(ops.less(a, b), tensor.convert_to_tensor([True, False, False])))
        assert ops.all(ops.equal(ops.less_equal(a, b), tensor.convert_to_tensor([True, True, False])))
        assert ops.all(ops.equal(ops.greater(a, b), tensor.convert_to_tensor([False, False, True])))
        assert ops.all(ops.equal(ops.greater_equal(a, b), tensor.convert_to_tensor([False, True, True])))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_random_operations(backend_name, original_backend):
    """Test random operations."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Set seed for reproducibility
        tensor.set_seed(42)
        
        # Test random normal
        rand_normal = tensor.random_normal((2, 3))
        assert tensor.shape(rand_normal) == (2, 3)
        
        # Test random uniform
        rand_uniform = tensor.random_uniform((2, 3))
        assert tensor.shape(rand_uniform) == (2, 3)
        
        # Test random binomial
        rand_binomial = tensor.random_binomial((2, 3), p=0.5)
        assert tensor.shape(rand_binomial) == (2, 3)
        
        # Test shuffle
        x = tensor.convert_to_tensor([1, 2, 3, 4, 5])
        shuffled = tensor.shuffle(x)
        assert tensor.shape(shuffled) == (5,)
        # Check that all elements are still present (order may be different)
        assert ops.all(ops.equal(ops.sort(shuffled), x))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_device_operations(backend_name, original_backend):
    """Test device operations."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a test tensor
        x = tensor.convert_to_tensor([1, 2, 3])
        
        # Test get_device
        device = ops.get_device(x)
        assert isinstance(device, str)
        
        # Test to_device (this should work even if the device is the same)
        y = ops.to_device(x, device)
        assert ops.all(ops.equal(x, y))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")