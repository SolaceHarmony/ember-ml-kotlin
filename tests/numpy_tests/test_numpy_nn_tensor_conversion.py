# tests/numpy_tests/test_nn_tensor_conversion.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
import numpy as np
from ember_ml.nn.tensor.types import TensorLike

# Note: Assumes conftest.py provides the numpy_backend fixture

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

# Helper function (can remain or be moved to a common conftest)
def get_allowed_tensor_types(backend_name: str) -> tuple:
    allowed_types = [EmberTensor]
    if backend_name == 'numpy':
        allowed_types.append(tensor.EmberTensor)
    # Removed torch/mlx checks as they are not relevant here
    return tuple(allowed_types)

def test_tensor_cast_numpy(numpy_backend, dtype_pair): # Use fixture
    """Tests tensor.cast with NumPy backend."""
    original_dtype, target_dtype = dtype_pair

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
    allowed_types = get_allowed_tensor_types('numpy')
    assert isinstance(t_casted, allowed_types), f"Cast result type ({type(t_casted)}) invalid"
    assert tensor.shape(t_casted) == tensor.shape(t_original), "Cast changed shape"
    # Compare the actual NumPy dtype object with the expected NumPy dtype object
    assert t_casted.dtype == target_dtype._backend_dtype, f"Cast failed: expected dtype '{target_dtype.name}', got '{tensor.to_dtype_str(t_casted.dtype)}'"

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

def test_tensor_to_numpy_numpy(numpy_backend): # Use fixture
    """Tests tensor.to_numpy with NumPy backend."""
    data = [[1.1, 2.2], [3.3, 4.4]]
    t_ember = tensor.convert_to_tensor(data)
    t_numpy = tensor.to_numpy(t_ember)
    assert isinstance(t_numpy, tensor.EmberTensor), "Not tensor.EmberTensor"
    assert ops.allclose(t_numpy, tensor.convert_to_tensor(data)), "Content mismatch"
    assert t_numpy.shape == tuple(tensor.shape(t_ember)), "Shape mismatch"

def test_tensor_item_numpy(numpy_backend): # Use fixture
    """Tests tensor.item for scalar tensors with NumPy backend."""
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
    assert isinstance(item_bool, (bool, tensor.bool)), "Bool type failed"
    assert bool(item_bool) is True, "Bool value failed"

    t_non_scalar = tensor.convert_to_tensor([1, 2])
    with pytest.raises(Exception):
        tensor.item(t_non_scalar)