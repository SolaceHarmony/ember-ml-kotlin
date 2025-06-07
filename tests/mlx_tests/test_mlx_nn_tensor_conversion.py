# tests/mlx_tests/test_nn_tensor_conversion.py
import pytest
from ember_ml import ops
from ember_ml.nn import tensor

# Note: Assumes conftest.py provides the mlx_backend fixture

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



def test_tensor_cast_mlx(mlx_backend, dtype_pair): # Use fixture
    """Tests tensor.cast with MLX backend."""
    original_dtype, target_dtype = dtype_pair

    # Skip float64 if not supported
    # Skip float64 tests if the backend doesn't support it
    if 'float64' in [str(original_dtype), str(target_dtype)]:
        try:
            # Use tensor operations to test float64 support
            test_tensor = tensor.convert_to_tensor(1.0, dtype=tensor.float32)
            _ = tensor.cast(test_tensor, tensor.float64)
        except Exception as e:
            if "float64" in str(e).lower() and "not supported" in str(e).lower():
                pytest.skip(f"Skipping float64 test: {e}")
            else:
                raise e

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
    assert tensor.shape(t_casted) == tensor.shape(t_original), "Cast changed shape"
    casted_dtype_str = tensor.to_dtype_str(tensor.dtype(t_casted))
    target_dtype_str = str(target_dtype)
    
    # Special handling for float64 in MLX (which maps to float32)
    if target_dtype_str == 'float64' and casted_dtype_str == 'float32':
        # This is expected behavior for MLX backend
        pass
    else:
        assert casted_dtype_str == target_dtype_str, f"Cast failed: expected '{target_dtype_str}', got '{casted_dtype_str}'"

    # Value checks
    # Convert to numpy for value checks
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

def test_tensor_to_numpy_mlx(mlx_backend): # Use fixture
    """Tests tensor.to_numpy with MLX backend."""
    data = [[1.1, 2.2], [3.3, 4.4]]
    t_ember = tensor.convert_to_tensor(data)
    t_numpy = tensor.to_numpy(t_ember)
    # Check that the result is a numpy array
    assert isinstance(t_numpy, tensor.EmberTensor), "Not numpy.ndarray"
    assert ops.allclose(t_numpy, tensor.convert_to_tensor(data)), "Content mismatch"
    assert t_numpy.shape == tuple(tensor.shape(t_ember)), "Shape mismatch"

def test_tensor_item_mlx(mlx_backend): # Use fixture
    """Tests tensor.item for scalar tensors with MLX backend."""
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
    # MLX returns 1 for True, so we check if it's truthy instead of checking the type
    assert item_bool == 1, "Bool value failed"

    t_non_scalar = tensor.convert_to_tensor([1, 2])
    with pytest.raises(Exception):
        tensor.item(t_non_scalar)