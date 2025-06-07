"""
Test the tensor creation and basic operations.

This module tests the tensor creation and basic operations to ensure they work correctly
across different backends.
"""

import pytest

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import (
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16
)


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
def test_tensor_creation(backend_name, original_backend):
    """Test creating a tensor from a list."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor from a list
        data = [[1, 2, 3], [4, 5, 6]]
        t = tensor.convert_to_tensor(data)
        
        # Check the tensor properties
        assert tensor.shape(t) == (2, 3)
        # The dtype string representation includes the backend name
        # Get the dtype directly from the tensor object
        if hasattr(t, 'dtype'):
            dtype_str = str(t.dtype)
            assert 'float32' in dtype_str or 'int32' in dtype_str or 'int64' in dtype_str
        
        # Convert to list and check values
        # Use numpy array's tolist method
        t_np = tensor.to_numpy(t)
        if t_np is not None:
            t_list = t_np.tolist()
            assert t_list == data
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_zeros_ones(backend_name, original_backend):
    """Test creating tensors with zeros and ones."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Test zeros
        shape = (2, 3)
        zeros_tensor = tensor.zeros(shape)
        
        # Check shape
        assert tensor.shape(zeros_tensor) == shape
        
        # Check that all values are zeros
        assert ops.all(ops.equal(zeros_tensor, tensor.convert_to_tensor(0)))
        
        # Test ones
        ones_tensor = tensor.ones(shape)
        
        # Check shape
        assert tensor.shape(ones_tensor) == shape
        
        # Check that all values are ones
        assert ops.all(ops.equal(ones_tensor, tensor.convert_to_tensor(1)))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_reshape(backend_name, original_backend):
    """Test reshaping a tensor."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor
        data = [[1, 2, 3], [4, 5, 6]]
        t = tensor.convert_to_tensor(data)
        
        # Reshape the tensor
        new_shape = (3, 2)
        reshaped = tensor.reshape(t, new_shape)
        
        # Check the new shape
        assert tensor.shape(reshaped) == new_shape
        
        # Check that the data is preserved
        assert ops.all(ops.equal(
            tensor.reshape(reshaped, (2, 3)),
            t
        ))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_transpose(backend_name, original_backend):
    """Test transposing a tensor."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor
        data = [[1, 2, 3], [4, 5, 6]]
        t = tensor.convert_to_tensor(data)
        
        # Transpose the tensor
        transposed = tensor.transpose(t)
        
        # Check the new shape
        assert tensor.shape(transposed) == (3, 2)
        
        # Check specific values
        assert ops.equal(tensor.slice(transposed, [0, 0], [1, 1]), tensor.convert_to_tensor(1))
        assert ops.equal(tensor.slice(transposed, [0, 1], [1, 1]), tensor.convert_to_tensor(4))
        assert ops.equal(tensor.slice(transposed, [1, 0], [1, 1]), tensor.convert_to_tensor(2))
        assert ops.equal(tensor.slice(transposed, [1, 1], [1, 1]), tensor.convert_to_tensor(5))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_concatenate(backend_name, original_backend):
    """Test concatenating tensors."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create tensors
        data = [[1, 2, 3], [4, 5, 6]]
        t1 = tensor.convert_to_tensor(data)
        t2 = tensor.convert_to_tensor(data)
        
        # Concatenate along axis 0
        concat_0 = tensor.concatenate([t1, t2], axis=0)
        
        # Check the new shape
        assert tensor.shape(concat_0) == (4, 3)
        
        # Check specific values
        assert ops.all(ops.equal(
            tensor.slice(concat_0, [0, 0], [2, 3]),
            t1
        ))
        assert ops.all(ops.equal(
            tensor.slice(concat_0, [2, 0], [2, 3]),
            t2
        ))
        
        # Concatenate along axis 1
        concat_1 = tensor.concatenate([t1, t2], axis=1)
        
        # Check the new shape
        assert tensor.shape(concat_1) == (2, 6)
        
        # Check specific values
        assert ops.all(ops.equal(
            tensor.slice(concat_1, [0, 0], [2, 3]),
            t1
        ))
        assert ops.all(ops.equal(
            tensor.slice(concat_1, [0, 3], [2, 3]),
            t2
        ))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_stack(backend_name, original_backend):
    """Test stacking tensors."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create tensors
        data = [[1, 2, 3], [4, 5, 6]]
        t1 = tensor.convert_to_tensor(data)
        t2 = tensor.convert_to_tensor(data)
        
        # Stack along axis 0
        stacked_0 = tensor.stack([t1, t2], axis=0)
        
        # Check the new shape
        assert tensor.shape(stacked_0) == (2, 2, 3)
        
        # Check specific values
        assert ops.all(ops.equal(
            tensor.slice(stacked_0, [0, 0, 0], [1, 2, 3]),
            tensor.expand_dims(t1, 0)
        ))
        assert ops.all(ops.equal(
            tensor.slice(stacked_0, [1, 0, 0], [1, 2, 3]),
            tensor.expand_dims(t2, 0)
        ))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_tensor_split(backend_name, original_backend):
    """Test splitting a tensor."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor
        data = [[1, 2, 3], [4, 5, 6]]
        t = tensor.convert_to_tensor(data)
        
        # Split along axis 1
        split_tensors = tensor.split(t, 3, axis=1)
        
        # Check the number of split tensors
        assert len(split_tensors) == 3
        
        # Check the shapes of the split tensors
        for split_t in split_tensors:
            assert tensor.shape(split_t) == (2, 1)
        
        # Check specific values
        assert ops.all(ops.equal(split_tensors[0], tensor.convert_to_tensor([[1], [4]])))
        assert ops.all(ops.equal(split_tensors[1], tensor.convert_to_tensor([[2], [5]])))
        assert ops.all(ops.equal(split_tensors[2], tensor.convert_to_tensor([[3], [6]])))
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_data_types(backend_name, original_backend):
    """Test data types."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        # Test float32
        t = tensor.convert_to_tensor([1, 2, 3], dtype=float32)
        # Get the dtype directly from the tensor object
        if hasattr(t, 'dtype'):
            dtype_str = str(t.dtype)
            assert 'float32' in dtype_str
        
        # Test int64
        t = tensor.convert_to_tensor([1, 2, 3], dtype=int64)
        # Get the dtype directly from the tensor object
        if hasattr(t, 'dtype'):
            dtype_str = str(t.dtype)
            assert 'int64' in dtype_str
        
        # Test bool_
        t = tensor.convert_to_tensor([True, False, True], dtype=bool_)
        # Get the dtype directly from the tensor object
        if hasattr(t, 'dtype'):
            dtype_str = str(t.dtype)
            assert 'bool' in dtype_str
        assert 'bool' in dtype_str
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")