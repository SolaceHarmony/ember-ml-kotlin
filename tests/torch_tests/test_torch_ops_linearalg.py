import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for ops.linearalg functions

def test_matmul():
    # Test matrix multiplication
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result = ops.matmul(a, b)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (using numpy for expected values)
    expected_np = ops.matmul(tensor.to_numpy(a), tensor.to_numpy(b))
    assert ops.allclose(result_np, expected_np)

    # Test with different shapes
    c = tensor.convert_to_tensor([[1.0, 2.0, 3.0]]) # Shape (1, 3)
    d = tensor.convert_to_tensor([[4.0], [5.0], [6.0]]) # Shape (3, 1)
    result_cd = ops.matmul(c, d)
    result_dc = ops.matmul(d, c)

    assert ops.allclose(tensor.to_numpy(result_cd), ops.matmul(tensor.to_numpy(c), tensor.to_numpy(d)))
    assert ops.allclose(tensor.to_numpy(result_dc), ops.matmul(tensor.to_numpy(d), tensor.to_numpy(c)))


def test_det():
    # Test determinant calculation
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = ops.linearalg.det(a)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    expected_np = ops.linearalg.det(tensor.to_numpy(a))
    assert ops.allclose(result_np, expected_np)

    # Test with a singular matrix
    b = tensor.convert_to_tensor([[1.0, 2.0], [2.0, 4.0]])
    result_singular = ops.linearalg.det(b)
    assert ops.allclose(tensor.to_numpy(result_singular), 0.0)

# Add more test functions for other ops.linearalg functions:
# test_qr(), test_svd(), test_cholesky(), test_eig(), test_eigvals(),
# test_solve(), test_inv(), test_norm(), test_lstsq(), test_diag(), test_diagonal()

# Example structure for test_inv
# def test_inv():
#     # Test inverse calculation
#     a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
#     result = ops.linearalg.inv(a)
#
#     # Convert to numpy for assertion
#     result_np = tensor.to_numpy(result)
#
#     # Assert correctness (A @ A_inv should be identity)
#     identity_check = ops.matmul(a, result)
#     assert ops.allclose(tensor.to_numpy(identity_check), ops.eye(2))