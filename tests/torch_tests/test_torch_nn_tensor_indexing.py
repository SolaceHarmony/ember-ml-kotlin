# tests/torch_tests/test_nn_tensor_indexing.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor

# Note: Assumes conftest.py provides the torch_backend fixture

# Helper function to generate indexing tensor
def _get_indexing_tensor():
    return tensor.arange(12).reshape((3, 4))

def test_tensor_gather_torch(torch_backend): # Use fixture
    """Tests tensor.gather with PyTorch backend."""
    params = _get_indexing_tensor()
    indices_axis0 = tensor.convert_to_tensor([0, 2])
    indices_axis1 = tensor.convert_to_tensor([1, 3])
    result_axis0 = tensor.gather(params, indices_axis0, axis=0)
    expected_axis0 = tensor.convert_to_tensor([[0, 1, 2, 3], [8, 9, 10, 11]])
    assert tensor.shape(result_axis0) == (2, 4), "Gather axis=0 shape failed"
    # Ensure the expected tensor has the same dtype as the result
    expected_axis0 = tensor.cast(expected_axis0, tensor.dtype(result_axis0))
    assert ops.allclose(result_axis0, expected_axis0), "Gather axis=0 failed"
    result_axis1 = tensor.gather(params, indices_axis1, axis=1)
    expected_axis1 = tensor.convert_to_tensor([[1, 3], [5, 7], [9, 11]])
    assert tensor.shape(result_axis1) == (3, 2), "Gather axis=1 shape failed"
    # Ensure the expected tensor has the same dtype as the result
    expected_axis1 = tensor.cast(expected_axis1, tensor.dtype(result_axis1))
    assert ops.allclose(result_axis1, expected_axis1), "Gather axis=1 failed"

def test_tensor_scatter_torch(torch_backend): # Use fixture
    """Tests tensor.scatter with PyTorch backend."""
    indices = tensor.convert_to_tensor([[0, 1], [2, 2]])
    updates = tensor.convert_to_tensor([100, 200])
    shape = (3, 4)
    scattered = tensor.tensor_scatter_nd_update(tensor.zeros(shape), indices, updates)
    assert tensor.shape(scattered) == shape, "tensor.scatter shape failed"
    expected_manual = tensor.convert_to_tensor([[0, 100, 0, 0],[0, 0,   0, 0],[0, 0, 200, 0]])
    expected_manual = tensor.cast(expected_manual, tensor.dtype(scattered))
    assert ops.allclose(scattered, expected_manual), "Scatter content check failed"

def test_tensor_scatter_nd_update_torch(torch_backend): # Use fixture
    """Tests tensor.tensor_scatter_nd_update with PyTorch backend."""
    t_to_update = _get_indexing_tensor()
    indices = tensor.convert_to_tensor([[0, 0], [1, 2]])
    updates = tensor.convert_to_tensor([-50, -60])
    updated_tensor = tensor.tensor_scatter_nd_update(t_to_update, indices, updates)
    assert tensor.shape(updated_tensor) == tensor.shape(t_to_update), "Shape failed"
    expected = tensor.convert_to_tensor([[-50, 1,  2,  3],[  4, 5, -60, 7],[  8, 9, 10, 11]])
    expected = tensor.cast(expected, tensor.dtype(updated_tensor))
    assert ops.allclose(updated_tensor, expected), "Content check failed"

def test_tensor_slice_torch(torch_backend): # Use fixture
    """Tests tensor.slice with PyTorch backend."""
    params = _get_indexing_tensor()
    sliced = tensor.slice(params, begin=[1, 0], size=[2, 2])
    expected_slice = tensor.convert_to_tensor([[4, 5], [8, 9]])
    assert tensor.shape(sliced) == (2, 2), "tensor.slice shape failed"
    expected_slice = tensor.cast(expected_slice, tensor.dtype(sliced))
    assert ops.allclose(sliced, expected_slice), "tensor.slice content failed"

def test_tensor_pad_torch(torch_backend): # Use fixture
    """Tests tensor.pad with PyTorch backend."""
    t = _get_indexing_tensor()
    paddings = [[1, 2], [0, 1]]
    constant_values = 99
    padded = tensor.pad(t, paddings, mode='constant', constant_values=constant_values)
    expected_pad_shape = (6, 5)
    assert tensor.shape(padded) == expected_pad_shape, "tensor.pad shape failed"
    expected_manual = tensor.convert_to_tensor([
        [99, 99, 99, 99, 99],
        [ 0,  1,  2,  3, 99],
        [ 4,  5,  6,  7, 99],
        [ 8,  9, 10, 11, 99],
        [99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99]
    ])
    expected_manual = tensor.cast(expected_manual, tensor.dtype(padded))
    assert ops.allclose(padded, expected_manual), "tensor.pad content check failed"