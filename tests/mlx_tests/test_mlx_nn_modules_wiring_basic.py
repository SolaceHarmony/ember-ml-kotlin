import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import NCPMap, FullyConnectedMap, RandomMap # Import basic maps
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

# Test cases for nn.modules.wiring basic maps

def test_ncpmap_initialization():
    # Test NCPMap initialization
    sensory = 5
    inter = 10
    command = 4
    motor = 3
    ncp_map = NCPMap(sensory_neurons=sensory, inter_neurons=inter, command_neurons=command, motor_neurons=motor, seed=42)

    assert isinstance(ncp_map, NCPMap)
    assert ncp_map.sensory_neurons == sensory
    assert ncp_map.inter_neurons == inter
    assert ncp_map.command_neurons == command
    assert ncp_map.motor_neurons == motor
    assert ncp_map.units == inter + command + motor # Units are non-sensory neurons
    assert ncp_map.output_dim == motor # Output dim is motor neurons
    assert ncp_map.input_dim is None # Input dim is set during build

    # Test initialization with input_dim provided (should override sensory_neurons)
    input_dim_override = 7
    ncp_map_override = NCPMap(sensory_neurons=sensory, inter_neurons=inter, command_neurons=command, motor_neurons=motor, input_dim=input_dim_override, seed=42)
    assert ncp_map_override.input_dim == input_dim_override


def test_ncpmap_build():
    # Test NCPMap build method
    sensory = 5
    inter = 10
    command = 4
    motor = 3
    ncp_map = NCPMap(sensory_neurons=sensory, inter_neurons=inter, command_neurons=command, motor_neurons=motor, seed=42)

    input_dim = 5 # Should match sensory_neurons if not overridden
    ncp_map.build(input_dim)

    assert ncp_map.input_dim == input_dim
    assert hasattr(ncp_map, 'adjacency_matrix')
    assert hasattr(ncp_map, 'sensory_adjacency_matrix')
    assert tensor.shape(ncp_map.adjacency_matrix) == (ncp_map.units, ncp_map.units)
    assert tensor.shape(ncp_map.sensory_adjacency_matrix) == (input_dim, ncp_map.units)
    assert ncp_map.synapse_count > 0 # Should have some connections
    assert ncp_map.sensory_synapse_count > 0 # Should have some sensory connections

    # Test build with input_dim override
    input_dim_override = 7
    ncp_map_override = NCPMap(sensory_neurons=sensory, inter_neurons=inter, command_neurons=command, motor_neurons=motor, input_dim=input_dim_override, seed=42)
    ncp_map_override.build(input_dim_override)
    assert ncp_map_override.input_dim == input_dim_override
    assert tensor.shape(ncp_map_override.sensory_adjacency_matrix) == (input_dim_override, ncp_map_override.units)

def test_fullyconnectedmap_initialization():
    # Test FullyConnectedMap initialization
    units = 10
    input_dim = 5
    output_dim = 3
    fc_map = FullyConnectedMap(units=units, input_dim=input_dim, output_dim=output_dim)

    assert isinstance(fc_map, FullyConnectedMap)
    assert fc_map.units == units
    assert fc_map.input_dim == input_dim
    assert fc_map.output_dim == output_dim
    assert hasattr(fc_map, 'adjacency_matrix')
    assert hasattr(fc_map, 'sensory_adjacency_matrix')
    assert tensor.shape(fc_map.adjacency_matrix) == (units, units)
    assert tensor.shape(fc_map.sensory_adjacency_matrix) == (input_dim, units)
    assert fc_map.synapse_count == units * units # Fully connected
    assert fc_map.sensory_synapse_count == input_dim * units # Fully connected sensory

def test_randommap_initialization():
    # Test RandomMap initialization
    units = 10
    input_dim = 5
    output_dim = 3
    sparsity_level = 0.5
    random_map = RandomMap(units=units, input_dim=input_dim, output_dim=output_dim, sparsity_level=sparsity_level, seed=42)

    assert isinstance(random_map, RandomMap)
    assert random_map.units == units
    assert random_map.input_dim == input_dim
    assert random_map.output_dim == output_dim
    assert random_map.sparsity_level == sparsity_level
    assert hasattr(random_map, 'adjacency_matrix')
    assert hasattr(random_map, 'sensory_adjacency_matrix')
    assert tensor.shape(random_map.adjacency_matrix) == (units, units)
    assert tensor.shape(random_map.sensory_adjacency_matrix) == (input_dim, units)
    # Check that sparsity is applied (synapse count should be less than fully connected)
    assert random_map.synapse_count < units * units
    assert random_map.sensory_synapse_count < input_dim * units

# Add more test functions for other basic map methods if any exist and are testable