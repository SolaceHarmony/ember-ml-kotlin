import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.wave.memory import multi_sphere # Import the multi_sphere module
from ember_ml.wave.memory import sphere_overlap # Import sphere_overlap components
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

# Test cases for wave.memory.multi_sphere and sphere_overlap components

def test_sphereoverlap_dataclass():
    # Test SphereOverlap dataclass initialization and validation
    reflection = 0.8
    transmission = 0.2

    overlap = sphere_overlap.SphereOverlap(reflection, transmission)

    assert isinstance(overlap, sphere_overlap.SphereOverlap)
    assert overlap.reflection == reflection
    assert overlap.transmission == transmission

    # Test validation (reflection + transmission should be <= 1.0)
    with pytest.raises(ValueError):
        sphere_overlap.SphereOverlap(0.8, 0.3) # Sum > 1.0

    with pytest.raises(ValueError):
        sphere_overlap.SphereOverlap(1.1, 0.0) # Reflection > 1.0

    with pytest.raises(ValueError):
        sphere_overlap.SphereOverlap(0.0, 1.1) # Transmission > 1.1

    with pytest.raises(ValueError):
        sphere_overlap.SphereOverlap(-0.1, 0.5) # Negative value


def test_spherestate_dataclass():
    # Test SphereState dataclass initialization and validation
    state_dim = 4
    # Create dummy state vectors (NumPy arrays)
    fast_vec_np = np.random.rand(state_dim).astype(tensor.float32)
    slow_vec_np = np.random.rand(state_dim).astype(tensor.float32)
    noise_std = 0.1

    # Convert to tensors
    fast_vec = tensor.convert_to_tensor(fast_vec_np)
    slow_vec = tensor.convert_to_tensor(slow_vec_np)

    state = sphere_overlap.SphereState(fast_vec, slow_vec, noise_std)

    assert isinstance(state, sphere_overlap.SphereState)
    assert isinstance(state.fast_wave_state, tensor.EmberTensor)
    assert isinstance(state.slow_wave_state, tensor.EmberTensor)
    assert ops.allclose(state.fast_wave_state, fast_vec).item()
    assert ops.allclose(state.slow_wave_state, slow_vec).item()
    assert state.noise_std == noise_std

    # Test validation (vector shapes should match)
    fast_vec_wrong_shape = tensor.random_normal((state_dim + 1,))
    with pytest.raises(ValueError):
        sphere_overlap.SphereState(fast_vec_wrong_shape, slow_vec, noise_std)


def test_overlapnetwork_initialization():
    # Test OverlapNetwork initialization
    num_spheres = 5
    # Create dummy overlaps (list of SphereOverlap objects)
    overlaps = [
        sphere_overlap.SphereOverlap(0.5, 0.5), # Overlap between sphere 0 and 1
        sphere_overlap.SphereOverlap(0.6, 0.4), # Overlap between sphere 1 and 2
        sphere_overlap.SphereOverlap(0.7, 0.3), # Overlap between sphere 3 and 4
    ]
    # Overlaps are defined by their position in the list (implicit connection)
    # The OverlapNetwork constructor likely infers connections based on list order.
    # This might need clarification from the actual implementation.
    # Assuming overlaps[i] is between sphere i and i+1.

    network = sphere_overlap.OverlapNetwork(overlaps, num_spheres)

    assert isinstance(network, sphere_overlap.OverlapNetwork)
    assert len(network.overlaps) == len(overlaps)
    assert network.num_spheres == num_spheres


def test_overlapnetwork_get_neighbors():
    # Test OverlapNetwork get_neighbors
    num_spheres = 5
    overlaps = [
        sphere_overlap.SphereOverlap(0.5, 0.5), # 0-1
        sphere_overlap.SphereOverlap(0.6, 0.4), # 1-2
        sphere_overlap.SphereOverlap(0.7, 0.3), # 3-4
    ]
    network = sphere_overlap.OverlapNetwork(overlaps, num_spheres)

    assert network.get_neighbors(0) == [1]
    assert network.get_neighbors(1) == [0, 2]
    assert network.get_neighbors(2) == [1]
    assert network.get_neighbors(3) == [4]
    assert network.get_neighbors(4) == [3]


def test_overlapnetwork_get_overlap():
    # Test OverlapNetwork get_overlap
    num_spheres = 5
    overlap_0_1 = sphere_overlap.SphereOverlap(0.5, 0.5)
    overlap_1_2 = sphere_overlap.SphereOverlap(0.6, 0.4)
    overlaps = [overlap_0_1, overlap_1_2]
    network = sphere_overlap.OverlapNetwork(overlaps, num_spheres)

    assert network.get_overlap(0, 1) == overlap_0_1
    assert network.get_overlap(1, 0) == overlap_0_1 # Should be symmetric
    assert network.get_overlap(1, 2) == overlap_1_2
    assert network.get_overlap(2, 1) == overlap_1_2

    # Test for non-existent overlap
    assert network.get_overlap(0, 2) is None
    assert network.get_overlap(3, 4) is None # No overlap defined for 3-4


def test_multispherewavemodel_initialization():
    # Test MultiSphereWaveModel initialization
    num_spheres = 3
    state_dim = 4
    reflection = 0.5
    transmission = 0.5
    noise_std = 0.1

    model = multi_sphere.MultiSphereWaveModel(num_spheres, state_dim, reflection, transmission, noise_std)

    assert isinstance(model, multi_sphere.MultiSphereWaveModel)
    assert model.num_spheres == num_spheres
    assert model.state_dim == state_dim
    assert isinstance(model.overlap_network, sphere_overlap.OverlapNetwork)
    assert len(model.sphere_states) == num_spheres
    for state in model.sphere_states:
        assert isinstance(state, sphere_overlap.SphereState)
        assert tensor.shape(state.fast_wave_state) == (state_dim,)
        assert tensor.shape(state.slow_wave_state) == (state_dim,)


def test_multispherewavemodel_set_initial_state():
    # Test MultiSphereWaveModel set_initial_state
    num_spheres = 3
    state_dim = 4
    model = multi_sphere.MultiSphereWaveModel(num_spheres, state_dim, 0.5, 0.5, 0.1)

    # Create dummy initial state vectors
    initial_fast = tensor.random_normal((state_dim,))
    initial_slow = tensor.random_normal((state_dim,))

    model.set_initial_state(1, initial_fast, initial_slow)

    assert ops.allclose(model.sphere_states[1].fast_wave_state, initial_fast).item()
    assert ops.allclose(model.sphere_states[1].slow_wave_state, initial_slow).item()

    # Test setting state for an invalid sphere index
    with pytest.raises(IndexError):
        model.set_initial_state(num_spheres + 1, initial_fast, initial_slow)


def test_multispherewavemodel_run():
    # Test MultiSphereWaveModel run method (basic execution)
    # This test verifies that the run method executes for a given number of steps
    # and returns a history of states of the correct shape.
    # It does not verify the correctness of the simulation dynamics.
    num_spheres = 3
    state_dim = 4
    model = multi_sphere.MultiSphereWaveModel(num_spheres, state_dim, 0.5, 0.5, 0.1)

    steps = 10
    # Create dummy input waves and gating signals (NumPy arrays or lists)
    # The shape of input_waves_seq should be (steps, num_spheres, state_dim)
    input_waves_seq_np = np.random.rand(steps, num_spheres, state_dim).astype(tensor.float32)
    input_waves_seq = tensor.convert_to_tensor(input_waves_seq_np)

    # The shape of gating_seq should be (steps, num_spheres) or (steps,)
    gating_seq_np = np.random.rand(steps, num_spheres).astype(tensor.float32)
    gating_seq = tensor.convert_to_tensor(gating_seq_np)


    # The run method returns the history of fast states (NumPy array)
    history = model.run(steps, input_waves_seq, gating_seq)

    assert isinstance(history, TensorLike)
    # Expected history shape: (steps, num_spheres, state_dim)
    assert history.shape == (steps, num_spheres, state_dim)


# Add more test functions for other multi_sphere components:
# test_multispherewavemodel_update_sphere_state(), test_multispherewavemodel_process_overlaps(),
# test_multispherewavemodel_get_sphere_states()

# Note: Testing the internal update and overlap processing logic will require
# detailed knowledge of the simulation dynamics and comparing intermediate
# states and values.