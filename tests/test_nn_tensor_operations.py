"""
Tests for tensor operations.

This module tests the functionality of tensor operations across different backends.
"""

import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor, float32, int32


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
def test_ember_tensor_creation(backend_name, original_backend):
    """Test EmberTensor creation."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor from a list
        t = EmberTensor([1, 2, 3, 4, 5])
        assert t.shape == (5,)
        assert str(t.dtype).endswith('int32') or str(t.dtype).endswith('int64')
        
        # Create a tensor with a specific dtype
        t = EmberTensor([1, 2, 3, 4, 5], dtype=float32)
        assert t.shape == (5,)
        assert str(t.dtype).endswith('float32')
        
        # Create a tensor from another tensor
        data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
        t = EmberTensor(data)
        assert t.shape == (2, 3)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_creation_functions(backend_name, original_backend):
    """Test tensor creation functions."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Test zeros
        shape = (3, 3)
        zeros_tensor = tensor.zeros(shape)
        assert tensor.shape(zeros_tensor) == shape
        assert ops.all(ops.equal(zeros_tensor, tensor.convert_to_tensor(0)))
        
        # Test ones
        ones_tensor = tensor.ones(shape)
        assert tensor.shape(ones_tensor) == shape
        assert ops.all(ops.equal(ones_tensor, tensor.convert_to_tensor(1)))
        
        # Test full
        value = 5
        full_tensor = tensor.full(shape, value)
        assert tensor.shape(full_tensor) == shape
        assert ops.all(ops.equal(full_tensor, tensor.convert_to_tensor(value)))
        
        # Test arange
        arange_tensor = tensor.arange(0, 5)
        assert tensor.shape(arange_tensor) == (5,)
        for i in range(5):
            assert ops.equal(tensor.slice(arange_tensor, [i], [1]), tensor.convert_to_tensor(i))
        
        # Test eye
        eye_tensor = tensor.eye(3)
        assert tensor.shape(eye_tensor) == (3, 3)
        for i in range(3):
            for j in range(3):
                expected = 1 if i == j else 0
                assert ops.equal(tensor.slice(eye_tensor, [i, j], [1, 1]), tensor.convert_to_tensor(expected))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_arithmetic_operations(backend_name, original_backend):
    """Test arithmetic operations."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create tensors
        t1 = tensor.convert_to_tensor([1, 2, 3, 4, 5])
        t2 = tensor.convert_to_tensor([5, 4, 3, 2, 1])
        
        # Test add
        result = ops.add(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([6, 6, 6, 6, 6])
        assert ops.all(ops.equal(result, expected))
        
        # Test subtract
        result = ops.subtract(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([-4, -2, 0, 2, 4])
        assert ops.all(ops.equal(result, expected))
        
        # Test multiply
        result = ops.multiply(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([5, 8, 9, 8, 5])
        assert ops.all(ops.equal(result, expected))
        
        # Test divide
        result = ops.divide(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([0.2, 0.5, 1.0, 2.0, 5.0])
        assert ops.all(ops.less(ops.abs(ops.subtract(result, expected)), 1e-5))
        
        # Test negative
        result = ops.negative(t1)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([-1, -2, -3, -4, -5])
        assert ops.all(ops.equal(result, expected))
        
        # Test absolute
        result = ops.abs(ops.negative(t1))
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([1, 2, 3, 4, 5])
        assert ops.all(ops.equal(result, expected))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_comparison_operations(backend_name, original_backend):
    """Test comparison operations."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create tensors
        t1 = tensor.convert_to_tensor([1, 2, 3, 4, 5])
        t2 = tensor.convert_to_tensor([5, 4, 3, 2, 1])
        
        # Test equal
        result = ops.equal(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([False, False, True, False, False])
        assert ops.all(ops.equal(result, expected))
        
        # Test not equal
        result = ops.not_equal(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([True, True, False, True, True])
        assert ops.all(ops.equal(result, expected))
        
        # Test less than
        result = ops.less(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([True, True, False, False, False])
        assert ops.all(ops.equal(result, expected))
        
        # Test less than or equal
        result = ops.less_equal(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([True, True, True, False, False])
        assert ops.all(ops.equal(result, expected))
        
        # Test greater than
        result = ops.greater(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([False, False, False, True, True])
        assert ops.all(ops.equal(result, expected))
        
        # Test greater than or equal
        result = ops.greater_equal(t1, t2)
        assert tensor.shape(result) == (5,)
        expected = tensor.convert_to_tensor([False, False, True, True, True])
        assert ops.all(ops.equal(result, expected))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_reduction_operations(backend_name, original_backend):
    """Test reduction operations."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor
        t = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Test sum
        result = ops.sum(t)
        assert ops.equal(result, tensor.convert_to_tensor(21))
        
        # Test sum along axis 0
        result = ops.sum(t, axis=0)
        assert tensor.shape(result) == (3,)
        expected = tensor.convert_to_tensor([5, 7, 9])
        assert ops.all(ops.equal(result, expected))
        
        # Test sum along axis 1
        result = ops.sum(t, axis=1)
        assert tensor.shape(result) == (2,)
        expected = tensor.convert_to_tensor([6, 15])
        assert ops.all(ops.equal(result, expected))
        
        # Test mean
        result = ops.mean(t)
        assert ops.equal(result, tensor.convert_to_tensor(3.5))
        
        # Test mean along axis 0
        result = ops.mean(t, axis=0)
        assert tensor.shape(result) == (3,)
        expected = tensor.convert_to_tensor([2.5, 3.5, 4.5])
        assert ops.all(ops.less(ops.abs(ops.subtract(result, expected)), 1e-5))
        
        # Test mean along axis 1
        result = ops.mean(t, axis=1)
        assert tensor.shape(result) == (2,)
        expected = tensor.convert_to_tensor([2.0, 5.0])
        assert ops.all(ops.less(ops.abs(ops.subtract(result, expected)), 1e-5))
        
        # Test max
        result = ops.max(t)
        assert ops.equal(result, tensor.convert_to_tensor(6))
        
        # Test max along axis 0
        result = ops.max(t, axis=0)
        assert tensor.shape(result) == (3,)
        expected = tensor.convert_to_tensor([4, 5, 6])
        assert ops.all(ops.equal(result, expected))
        
        # Test max along axis 1
        result = ops.max(t, axis=1)
        assert tensor.shape(result) == (2,)
        expected = tensor.convert_to_tensor([3, 6])
        assert ops.all(ops.equal(result, expected))
        
        # Test min
        result = ops.min(t)
        assert ops.equal(result, tensor.convert_to_tensor(1))
        
        # Test min along axis 0
        result = ops.min(t, axis=0)
        assert tensor.shape(result) == (3,)
        expected = tensor.convert_to_tensor([1, 2, 3])
        assert ops.all(ops.equal(result, expected))
        
        # Test min along axis 1
        result = ops.min(t, axis=1)
        assert tensor.shape(result) == (2,)
        expected = tensor.convert_to_tensor([1, 4])
        assert ops.all(ops.equal(result, expected))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_type_conversion(backend_name, original_backend):
    """Test type conversion operations."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor
        t = tensor.convert_to_tensor([1, 2, 3])
        
        # Test cast to float32
        result = tensor.cast(t, float32)
        assert 'float32' in str(tensor.dtype(result))
        
        # Test cast to int32
        result = tensor.cast(t, int32)
        assert 'int32' in str(tensor.dtype(result))
        
        # Test to_numpy
        np_array = tensor.to_numpy(t)
        
        # Handle the case where to_numpy returns a list (MLX backend)
        if isinstance(np_array, list):
            assert len(np_array) == 3
            assert np_array == [1, 2, 3]
        else:
            # NumPy array for other backends
            assert np_array.shape == (3,)
            assert np_array.tolist() == [1, 2, 3]
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")