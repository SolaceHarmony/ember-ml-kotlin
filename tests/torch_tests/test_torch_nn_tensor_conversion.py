# tests/torch_tests/test_nn_tensor_conversion.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
import numpy as np
from ember_ml.backend.torch.types import TensorLike # Import TensorLike

# Note: Assumes conftest.py provides the torch_backend fixture

# Fixture providing pairs of dtypes for casting tests.
@pytest.fixture(params=[
    (tensor.int32, tensor.float32),
    (tensor.float32, tensor.int64),
    (tensor.float64, tensor.bool_),
    (tensor.bool_, tensor.int8),
    (tensor.float32, tensor.float64),
    (tensor.int16, tensor.int64),
])
def dtype_pair(request):
    return request.param

# Helper function
def get_allowed_tensor_types(backend_name: str) -> tuple:
    allowed_types = [EmberTensor]
    if backend_name == 'torch':
        try:
            import torch
            allowed_types.append(tensor.convert_to_tensor)  # type: ignore
        except ImportError: pass
    return tuple(allowed_types)

def test_tensor_cast_torch(torch_backend, dtype_pair): # Use fixture
    """Tests tensor.cast with PyTorch backend."""
    original_dtype, target_dtype = dtype_pair

    # Skip float64 on MPS if necessary
    if 'mps' in ops.get_available_devices() and ('float64' in [str(original_dtype), str(target_dtype)]):
         pytest.skip("Skipping float64 test on MPS device for PyTorch")

    # Create initial tensor
    if original_dtype == tensor.bool_:
         initial_data = [[True, False], [False, True]]
         t_original = tensor.array(initial_data, dtype=original_dtype)
    elif 'int' in str(original_dtype):
         initial_data = [[1, 0], [5, -2]]
         t_original = tensor.array(initial_data, dtype=original_dtype)
    else: # float
         initial_data = [[1.5, 0.0], [5.2, -2.9]]
         t_original = tensor.array(initial_data, dtype=original_dtype)

    # Cast the tensor
    t_casted = tensor.cast(t_original, target_dtype)

    # Assertions
    allowed_types = get_allowed_tensor_types('torch')
    assert isinstance(t_casted, allowed_types), f"Cast result type ({type(t_casted)}) invalid"
    assert tensor.shape(t_casted) == tensor.shape(t_original), "Cast changed shape"
    casted_dtype_str = tensor.to_dtype_str(tensor.dtype(t_casted))
    target_dtype_str = str(target_dtype)
    assert casted_dtype_str == target_dtype_str, f"Cast failed: expected '{target_dtype_str}', got '{casted_dtype_str}'"

    # Value checks
    np_original = tensor.to_numpy(t_original)
    np_casted = tensor.to_numpy(t_casted)
    if original_dtype == tensor.float32 and 'int' in str(target_dtype):
        assert np_casted[0, 0] == 1, "Cast float->int failed"
    if 'int' in str(original_dtype) and target_dtype == tensor.bool_:
         assert np_casted[1, 0] is True, "Cast int->bool (non-zero) failed"
         assert np_casted[0, 1] is False, "Cast int->bool (zero) failed"
    if original_dtype == tensor.bool_ and 'int' in str(target_dtype):
         assert np_casted[0, 0] == 1, "Cast bool->int (True) failed"
         assert np_casted[0, 1] == 0, "Cast bool->int (False) failed"

def test_tensor_to_numpy_torch(torch_backend): # Use fixture
    """Tests tensor.to_numpy with PyTorch backend."""
    data = [[1.1, 2.2], [3.3, 4.4]]
    t_ember = tensor.convert_to_tensor(data)
    t_numpy = tensor.to_numpy(t_ember)
    assert isinstance(t_numpy, TensorLike), "Not TensorLike"
    assert ops.allclose(t_numpy, tensor.convert_to_tensor(data)), "Content mismatch"
    assert t_numpy.shape == tuple(tensor.shape(t_ember)), "Shape mismatch"

def test_tensor_item_torch(torch_backend): # Use fixture
    """Tests tensor.item for scalar tensors with PyTorch backend."""
    t_int = tensor.convert_to_tensor(42)
    item_int = tensor.item(t_int)
    assert isinstance(item_int, (int, tensor.integer)), "Int type failed"
    assert item_int == 42, "Int value failed"

    t_float = tensor.convert_to_tensor(3.14)
    item_float = tensor.item(t_float)
    assert isinstance(item_float, (float, tensor.floating)), "Float type failed"
    assert ops.less(ops.abs(ops.subtract(item_float, 3.14)), 1e-6), "Float value failed"

    t_bool = tensor.convert_to_tensor(True)
    item_bool = tensor.item(t_bool)
    # Check if the value is truthy, not strictly True
    assert bool(item_bool) is True, "Bool value failed"

    t_non_scalar = tensor.convert_to_tensor([1, 2])
    with pytest.raises(Exception):
        tensor.item(t_non_scalar)