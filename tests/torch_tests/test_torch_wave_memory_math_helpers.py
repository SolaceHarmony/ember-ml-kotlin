import pytest
import numpy as np # For comparison with known correct results
import math # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.memory import math_helpers # Import the module
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

# Test cases for wave.memory.math_helpers functions

def test_normalize_vector():
    # Test normalize_vector
    x = tensor.convert_to_tensor([1.0, 2.0, 2.0]) # L2 norm is 3.0
    result = math_helpers.normalize_vector(x)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (should be unit vector)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (3,)
    assert ops.allclose(ops.linearalg.norm(result_np), 1.0)
    assert ops.allclose(result_np, [1.0/3.0, 2.0/3.0, 2.0/3.0])

    # Test with epsilon
    x_small = tensor.convert_to_tensor([1e-10, 1e-10])
    result_small = math_helpers.normalize_vector(x_small, epsilon=1e-9)
    assert isinstance(result_small, tensor.EmberTensor)
    assert tensor.shape(result_small) == (2,)
    assert ops.allclose(ops.linearalg.norm(tensor.to_numpy(result_small)), 1.0)


def test_compute_phase_angle():
    # Test compute_phase_angle
    # Test with vectors in different quadrants
    vec1 = tensor.convert_to_tensor([1.0, 0.0]) # Positive x-axis
    vec2 = tensor.convert_to_tensor([0.0, 1.0]) # Positive y-axis
    vec3 = tensor.convert_to_tensor([-1.0, 0.0]) # Negative x-axis
    vec4 = tensor.convert_to_tensor([0.0, -1.0]) # Negative y-axis
    vec5 = tensor.convert_to_tensor([1.0, 1.0]) # First quadrant

    angle1 = math_helpers.compute_phase_angle(vec1)
    angle2 = math_helpers.compute_phase_angle(vec2)
    angle3 = math_helpers.compute_phase_angle(vec3)
    angle4 = math_helpers.compute_phase_angle(vec4)
    angle5 = math_helpers.compute_phase_angle(vec5)

    # Convert to numpy for assertion
    angle1_np = tensor.item(angle1)
    angle2_np = tensor.item(angle2)
    angle3_np = tensor.item(angle3)
    angle4_np = tensor.item(angle4)
    angle5_np = tensor.item(angle5)

    # Assert correctness (using math.atan2 for expected values)
    assert abs(angle1_np - math.atan2(0.0, 1.0)) < 1e-6
    assert abs(angle2_np - math.atan2(1.0, 0.0)) < 1e-6
    assert abs(angle3_np - math.atan2(0.0, -1.0)) < 1e-6
    assert abs(angle4_np - math.atan2(-1.0, 0.0)) < 1e-6
    assert abs(angle5_np - math.atan2(1.0, 1.0)) < 1e-6


def test_compute_energy():
    # Test compute_energy (squared L2 norm)
    vec1 = tensor.convert_to_tensor([1.0, 2.0, 2.0])
    result1 = math_helpers.compute_energy(vec1)
    assert isinstance(result1, tensor.EmberTensor)
    assert tensor.shape(result1) == () # Should be scalar
    assert ops.allclose(result1, 9.0).item() # 1^2 + 2^2 + 2^2 = 1 + 4 + 4 = 9

    vec2 = tensor.convert_to_tensor([0.0, 0.0])
    result2 = math_helpers.compute_energy(vec2)
    assert ops.allclose(result2, 0.0).item()


def test_partial_interference():
    # Test partial_interference
    base = tensor.convert_to_tensor([1.0, 0.0]) # Vector along x-axis
    new = tensor.convert_to_tensor([0.0, 1.0]) # Vector along y-axis
    alpha = 0.5 # Interference strength

    # Partial interference depends on the angle and alpha.
    # For orthogonal vectors, the result might be a linear combination.
    # The exact formula is needed for precise assertion.
    # Assuming a simple linear interpolation for now based on the name "partial".
    # This test might need adjustment based on the actual implementation.
    result = math_helpers.partial_interference(base, new, alpha)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (2,)

    # A more detailed test would involve known inputs and expected outputs based on the formula.


# Add more test functions for other math_helpers functions:
# test_compute_energy_stability(), test_compute_interference_strength(),
# test_compute_phase_coherence(), test_create_rotation_matrix()

# Note: Testing functions like compute_energy_stability, compute_interference_strength,
# and compute_phase_coherence will require simulating sequences of wave states
# and verifying the calculated metrics.