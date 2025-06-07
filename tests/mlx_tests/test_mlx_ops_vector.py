import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for ops.vector functions

def test_normalize_vector():
    # Test vector normalization
    x = tensor.convert_to_tensor([1.0, 2.0, 2.0]) # L2 norm is sqrt(1+4+4) = 3
    result = ops.normalize_vector(x)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (should be unit vector)
    assert ops.allclose(ops.linearalg.norm(result_np), 1.0)
    assert ops.allclose(result_np, [1.0/3.0, 2.0/3.0, 2.0/3.0])

    # Test with axis
    matrix = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result_axis0 = ops.normalize_vector(matrix, axis=0)
    result_axis1 = ops.normalize_vector(matrix, axis=1)

    assert ops.allclose(ops.linearalg.norm(tensor.to_numpy(result_axis0), axis=0), [1.0, 1.0])
    assert ops.allclose(ops.linearalg.norm(tensor.to_numpy(result_axis1), axis=1), [1.0, 1.0])


def test_euclidean_distance():
    # Test Euclidean distance
    a = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    b = tensor.convert_to_tensor([4.0, 5.0, 6.0])
    result = ops.euclidean_distance(a, b)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9+9+9) = sqrt(27))
    assert ops.allclose(result_np, ops.sqrt(27.0))

    # Test with matrices
    matrix_a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    matrix_b = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result_matrix = ops.euclidean_distance(matrix_a, matrix_b)
    # Expected: sqrt((5-1)^2 + (6-2)^2) = sqrt(16+16) = sqrt(32) for first row pair
    # Expected: sqrt((7-3)^2 + (8-4)^2) = sqrt(16+16) = sqrt(32) for second row pair
    assert ops.allclose(tensor.to_numpy(result_matrix), [ops.sqrt(32.0), ops.sqrt(32.0)])


def test_cosine_similarity():
    # Test cosine similarity
    a = tensor.convert_to_tensor([1.0, 1.0]) # Angle 45 degrees
    b = tensor.convert_to_tensor([1.0, 0.0]) # Angle 0 degrees
    result = ops.cosine_similarity(a, b)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (cos(45) = 1/sqrt(2))
    assert ops.allclose(result_np, 1.0 / ops.sqrt(2.0))

    # Test orthogonal vectors
    c = tensor.convert_to_tensor([1.0, 0.0])
    d = tensor.convert_to_tensor([0.0, 1.0])
    result_orthogonal = ops.cosine_similarity(c, d)
    assert ops.allclose(tensor.to_numpy(result_orthogonal), 0.0)

# Add more test functions for other ops.vector functions:
# test_compute_energy_stability(), test_compute_interference_strength(),
# test_compute_phase_coherence(), test_partial_interference(), test_exponential_decay()

# Example structure for test_exponential_decay
# def test_exponential_decay():
#     x = tensor.convert_to_tensor([1.0, 2.0, 3.0])
#     rate = 0.5
#     result = ops.exponential_decay(x, rate=rate)
#     # Expected: [1.0 * exp(-0.5*0), 2.0 * exp(-0.5*1), 3.0 * exp(-0.5*2)]
#     expected_np = tensor.convert_to_tensor([1.0 * ops.exp(-0.5*0), 2.0 * ops.exp(-0.5*1), 3.0 * ops.exp(-0.5*2)])
#     assert ops.allclose(tensor.to_numpy(result), expected_np)