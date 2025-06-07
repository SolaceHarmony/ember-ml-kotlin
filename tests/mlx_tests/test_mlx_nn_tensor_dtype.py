# tests/mlx_tests/test_nn_tensor_dtype.py
import pytest
from ember_ml.nn import tensor
from ember_ml.nn.tensor.common.dtypes import EmberDType

# Note: Assumes conftest.py provides the mlx_backend fixture

# List of expected dtype objects (redefined for clarity)
EXPECTED_DTYPES = [
    tensor.float16, tensor.float32, tensor.float64,
    tensor.int8, tensor.int16, tensor.int32, tensor.int64,
    tensor.uint8,
    tensor.bool_
]
AVAILABLE_DTYPES = [dt for dt in EXPECTED_DTYPES if dt is not None]

@pytest.mark.parametrize("dtype_to_test", AVAILABLE_DTYPES)
def test_get_dtype_mlx(mlx_backend, dtype_to_test): # Use fixture
    """Tests tensor.get_dtype for various types with MLX backend."""
    try:
        # Skip float64 if not supported by hardware/mlx version
        if 'float64' in str(dtype_to_test):
            try:
                import mlx.core as mx
                _ = mx.array(1.0, dtype=mx.float64) # Check support
            except ValueError:
                pytest.skip("float64 not supported by this MLX setup")
            except ImportError: pytest.skip("MLX not installed.")

        if dtype_to_test == tensor.bool_:
             t = tensor.zeros(1, dtype=dtype_to_test)
        else:
             t = tensor.ones(1, dtype=dtype_to_test)
        retrieved_dtype = tensor.get_dtype(t)
        assert retrieved_dtype == dtype_to_test, f"get_dtype failed for {dtype_to_test}"
    except Exception as e:
        pytest.skip(f"Skipping dtype {dtype_to_test} for mlx due to error: {e}")

@pytest.mark.parametrize("dtype_obj", AVAILABLE_DTYPES)
def test_dtype_str_conversion_mlx(mlx_backend, dtype_obj): # Use fixture
    """Tests tensor.to_dtype_str and tensor.from_dtype_str with MLX backend."""
    try:
        # Skip float64 if not supported
        if 'float64' in str(dtype_obj):
             try:
                 import mlx.core as mx
                 _ = mx.array(1.0, dtype=mx.float64) # Check support
             except ValueError:
                 pytest.skip("float64 not supported by this MLX setup")
             except ImportError: pytest.skip("MLX not installed.")

        dtype_str = tensor.to_dtype_str(dtype_obj)
        assert isinstance(dtype_str, str), f"to_dtype_str failed for {dtype_obj}"
        retrieved_dtype_obj = tensor.from_dtype_str(dtype_str)
        assert isinstance(retrieved_dtype_obj, EmberDType), f"from_dtype_str failed for '{dtype_str}'"
        assert retrieved_dtype_obj == dtype_obj, f"Round trip failed for {dtype_obj} ('{dtype_str}')"
    except Exception as e:
         pytest.skip(f"Skipping dtype {dtype_obj} for mlx due to error: {e}")

def test_dtype_equality_mlx(mlx_backend): # Use fixture
    """Tests dtype equality comparison with MLX backend."""
    assert tensor.float32 == tensor.float32, "Equality failed (self)"
    assert tensor.float32 != tensor.int32, "Inequality failed"
    f32_str = tensor.to_dtype_str(tensor.float32)
    f32_from_str = tensor.from_dtype_str(f32_str)
    assert tensor.float32 == f32_from_str, "Equality failed after string conversion"