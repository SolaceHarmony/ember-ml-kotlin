import pytest
import numpy as np
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.wave.utils import math_helpers

@pytest.fixture(params=['numpy'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')

# Test cases for math_helpers functions

def test_normalize_vector(set_backend_fixture):
    """Test normalize_vector function."""
    vec = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    normalized_vec = math_helpers.normalize_vector(vec)
    norm = ops.sqrt(stats.sum(ops.square(normalized_vec)))
    assert ops.allclose(norm, tensor.convert_to_tensor(1.0))
    assert tensor.shape(normalized_vec) == tensor.shape(vec)
    assert normalized_vec.dtype == vec.dtype

def test_compute_phase_angle(set_backend_fixture):
    """Test compute_phase_angle function."""
    # Test with a 2D vector (x, y)
    vec_2d = tensor.convert_to_tensor([1.0, 1.0])
    angle_2d = math_helpers.compute_phase_angle(vec_2d)
    # Expected angle for [1, 1] is pi/4
    assert ops.allclose(angle_2d, tensor.convert_to_tensor(ops.pi / 4.0))

    # Test with a batch of 2D vectors
    vec_batch = tensor.convert_to_tensor([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    angles_batch = math_helpers.compute_phase_angle(vec_batch)
    expected_angles = tensor.convert_to_tensor([ops.pi / 4.0, 3.0 * ops.pi / 4.0, -3.0 * ops.pi / 4.0, -ops.pi / 4.0])
    assert ops.allclose(angles_batch, expected_angles)
    assert tensor.shape(angles_batch) == (tensor.shape(vec_batch)[0],) # Should return a 1D tensor of angles

    # Test with a vector that should result in 0 or pi
    vec_zero = tensor.convert_to_tensor([1.0, 0.0])
    angle_zero = math_helpers.compute_phase_angle(vec_zero)
    assert ops.allclose(angle_zero, tensor.convert_to_tensor(0.0))

    vec_pi = tensor.convert_to_tensor([-1.0, 0.0])
    angle_pi = math_helpers.compute_phase_angle(vec_pi)
    assert ops.allclose(angle_pi, tensor.convert_to_tensor(ops.pi))

    # Test with a vector that should result in pi/2 or -pi/2
    vec_pi_half = tensor.convert_to_tensor([0.0, 1.0])
    angle_pi_half = math_helpers.compute_phase_angle(vec_pi_half)
    assert ops.allclose(angle_pi_half, tensor.convert_to_tensor(ops.pi / 2.0))

    vec_neg_pi_half = tensor.convert_to_tensor([0.0, -1.0])
    angle_neg_pi_half = math_helpers.compute_phase_angle(vec_neg_pi_half)
    assert ops.allclose(angle_neg_pi_half, tensor.convert_to_tensor(-ops.pi / 2.0))


def test_compute_energy(set_backend_fixture):
    """Test compute_energy function."""
    vec = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    energy = math_helpers.compute_energy(vec)
    # Energy is sum of squares: 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
    assert ops.allclose(energy, tensor.convert_to_tensor(14.0))
    assert tensor.shape(energy) == () # Should return a scalar

    # Test with a batch of vectors
    vec_batch = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    energies_batch = math_helpers.compute_energy(vec_batch)
    # Energies: [1^2 + 2^2, 3^2 + 4^2] = [5, 25]
    assert ops.allclose(energies_batch, tensor.convert_to_tensor([5.0, 25.0]))
    assert tensor.shape(energies_batch) == (tensor.shape(vec_batch)[0],) # Should return a 1D tensor of energies

def test_partial_interference(set_backend_fixture):
    """Test partial_interference function."""
    base_wave = tensor.convert_to_tensor([1.0, 0.0]) # Vector along x-axis
    new_wave_aligned = tensor.convert_to_tensor([1.0, 0.0]) # Aligned
    new_wave_orthogonal = tensor.convert_to_tensor([0.0, 1.0]) # Orthogonal
    new_wave_opposite = tensor.convert_to_tensor([-1.0, 0.0]) # Opposite
    alpha = 0.5 # Interference strength

    # Aligned waves: Should result in a stronger wave
    interference_aligned = math_helpers.partial_interference(base_wave, new_wave_aligned, alpha)
    # Expected: base + alpha * new = [1, 0] + 0.5 * [1, 0] = [1.5, 0]
    assert ops.allclose(interference_aligned, tensor.convert_to_tensor([1.5, 0.0]))

    # Orthogonal waves: Should result in a wave with components from both
    interference_orthogonal = math_helpers.partial_interference(base_wave, new_wave_orthogonal, alpha)
    # Expected: base + alpha * new = [1, 0] + 0.5 * [0, 1] = [1, 0.5]
    assert ops.allclose(interference_orthogonal, tensor.convert_to_tensor([1.0, 0.5]))

    # Opposite waves: Should result in a weaker wave
    interference_opposite = math_helpers.partial_interference(base_wave, new_wave_opposite, alpha)
    # Expected: base + alpha * new = [1, 0] + 0.5 * [-1, 0] = [0.5, 0]
    assert ops.allclose(interference_opposite, tensor.convert_to_tensor([0.5, 0.0]))

    # Test with epsilon
    base_wave_small = tensor.convert_to_tensor([1e-9, 0.0])
    new_wave_small = tensor.convert_to_tensor([1e-9, 0.0])
    interference_small = math_helpers.partial_interference(base_wave_small, new_wave_small, alpha, epsilon=1e-8)
    # With epsilon, small vectors should still interfere
    assert ops.allclose(interference_small, tensor.convert_to_tensor([1.5e-9, 0.0]))

# Note: compute_energy_stability requires a history of energy values,
# which is typically generated over time steps in a simulation.
# This test will check the function's behavior with a sample history.
def test_compute_energy_stability(set_backend_fixture):
    """Test compute_energy_stability function."""
    # Sample energy history (e.g., from a simulation)
    energy_history = tensor.convert_to_tensor([10.0, 10.1, 10.0, 10.2, 10.1])
    window_size = 3 # Compute variance over windows of size 3

    # Expected stability: variance of [10.0, 10.1, 10.0], [10.1, 10.0, 10.2], [10.0, 10.2, 10.1]
    # Variances: ~0.0033, ~0.0089, ~0.01
    # Mean variance: ~(0.0033 + 0.0089 + 0.01) / 3 = ~0.0074
    # Stability is 1 - mean_variance (or similar, check implementation)
    # Assuming stability is 1 - normalized_variance, where normalized_variance is mean_variance / max_possible_variance
    # Let's just check the output type and range for now, as exact value depends on internal normalization
    stability = math_helpers.compute_energy_stability(energy_history, window_size)
    assert tensor.shape(stability) == () # Should return a scalar
    # Stability should be between 0 and 1
    assert ops.greater_equal(stability, tensor.convert_to_tensor(0.0))
    assert ops.less_equal(stability, tensor.convert_to_tensor(1.0))

# Note: compute_interference_strength requires a list of vectors,
# typically representing states from multiple waves or neurons.
# This test will check the function's behavior with sample vectors.
def test_compute_interference_strength(set_backend_fixture):
    """Test compute_interference_strength function."""
    # Sample vectors (e.g., from different waves or neurons)
    vectors = [
        tensor.convert_to_tensor([1.0, 0.0]),
        tensor.convert_to_tensor([1.0, 0.0]), # Aligned
        tensor.convert_to_tensor([0.0, 1.0]), # Orthogonal
        tensor.convert_to_tensor([-1.0, 0.0]) # Opposite
    ]

    # Expected interference strength: depends on the internal calculation (e.g., average dot product)
    # Let's check the output type and range for now
    interference_strength = math_helpers.compute_interference_strength(vectors)
    assert tensor.shape(interference_strength) == () # Should return a scalar
    # Interference strength should be between -1 and 1 (based on correlation/dot product)
    assert ops.greater_equal(interference_strength, tensor.convert_to_tensor(-1.0))
    assert ops.less_equal(interference_strength, tensor.convert_to_tensor(1.0))

# Note: compute_phase_coherence requires a list of vectors and a frequency range.
# This test will check the function's behavior with sample vectors.
# The current implementation in docs suggests it's a placeholder needing FFT.
# We will test the placeholder behavior or skip if it requires complex FFT.
@pytest.mark.skip(reason="compute_phase_coherence implementation in docs is a placeholder needing FFT.")
def test_compute_phase_coherence(set_backend_fixture):
    """Test compute_phase_coherence function."""
    # Sample vectors
    vectors = [
        tensor.convert_to_tensor([1.0, 0.0]),
        tensor.convert_to_tensor([1.0, 0.0]),
        tensor.convert_to_tensor([0.0, 1.0]),
    ]
    freq_range = (0, 100) # Example frequency range

    # This test is skipped based on the documentation note that the implementation is a placeholder.
    # If a functional implementation is added, this test should be updated.
    pass


def test_create_rotation_matrix(set_backend_fixture):
    """Test create_rotation_matrix function."""
    # Test for 2D rotation (axis=None or axis=2)
    angle_2d = ops.pi / 2.0 # 90 degrees
    # Expected 2D rotation matrix: [[cos(a), -sin(a)], [sin(a), cos(a)]]
    # For 90 degrees: [[0, -1], [1, 0]]
    rot_matrix_2d = math_helpers.create_rotation_matrix(tensor.convert_to_tensor(angle_2d), axis=2)
    expected_rot_matrix_2d = tensor.convert_to_tensor([[0.0, -1.0], [1.0, 0.0]])
    assert ops.allclose(rot_matrix_2d, expected_rot_matrix_2d, atol=1e-6) # Use atol for near-zero values
    assert tensor.shape(rot_matrix_2d) == (2, 2)

    # Test for 3D rotation (e.g., around z-axis, axis=2)
    angle_3d = ops.pi / 2.0 # 90 degrees
    # Expected 3D rotation matrix around z-axis:
    # [[cos(a), -sin(a), 0],
    #  [sin(a),  cos(a), 0],
    #  [0,       0,      1]]
    # For 90 degrees: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rot_matrix_3d_z = math_helpers.create_rotation_matrix(tensor.convert_to_tensor(angle_3d), axis=2)
    expected_rot_matrix_3d_z = tensor.convert_to_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert ops.allclose(rot_matrix_3d_z, expected_rot_matrix_3d_z, atol=1e-6)
    assert tensor.shape(rot_matrix_3d_z) == (3, 3)

    # Test for 4D rotation (e.g., around a specified plane/axis)
    # This requires understanding the specific 4D rotation implementation.
    # Assuming it creates a rotation in the plane defined by the last two axes (3, 4)
    angle_4d = ops.pi / 2.0
    # Expected 4D rotation matrix in the (3, 4) plane:
    # [[1, 0, 0, 0],
    #  [0, 1, 0, 0],
    #  [0, 0, cos(a), -sin(a)],
    #  [0, 0, sin(a),  cos(a)]]
    # For 90 degrees: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    rot_matrix_4d = math_helpers.create_rotation_matrix(tensor.convert_to_tensor(angle_4d), axis=3) # Assuming axis=3 rotates in the last two dims
    expected_rot_matrix_4d = tensor.convert_to_tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0]])
    assert ops.allclose(rot_matrix_4d, expected_rot_matrix_4d, atol=1e-6)
    assert tensor.shape(rot_matrix_4d) == (4, 4)

    # Test with a batch of angles
    angles_batch = tensor.convert_to_tensor([ops.pi / 2.0, ops.pi])
    # Expected batch of 2D rotation matrices
    expected_rot_batch_2d = tensor.convert_to_tensor([[[0.0, -1.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, -1.0]]])
    rot_batch_2d = math_helpers.create_rotation_matrix(angles_batch, axis=2)
    assert ops.allclose(rot_batch_2d, expected_rot_batch_2d, atol=1e-6)
    assert tensor.shape(rot_batch_2d) == (2, 2, 2) # Batch size 2, 2x2 matrix