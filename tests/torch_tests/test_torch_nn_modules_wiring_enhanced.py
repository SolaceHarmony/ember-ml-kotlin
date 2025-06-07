import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import EnhancedNeuronMap, EnhancedNCPMap # Import enhanced maps
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

# Test cases for nn.modules.wiring enhanced maps

def test_enhancedneuronmap_initialization():
    # Test EnhancedNeuronMap initialization with spatial properties
    units = 20
    output_dim = 5
    input_dim = 10
    network_structure = (4, 5, 1) # Example 3D structure
    neuron_type = "cfc"
    time_scale_factor = 0.8
    seed = 42

    enhanced_map = EnhancedNeuronMap(
        units=units,
        output_dim=output_dim,
        input_dim=input_dim,
        neuron_type=neuron_type,
        neuron_params={"time_scale_factor": time_scale_factor},
        network_structure=network_structure,
        distance_metric="euclidean",
        distance_power=1.0,
        seed=seed
    )

    assert isinstance(enhanced_map, EnhancedNeuronMap)
    assert enhanced_map.units == units
    assert enhanced_map.output_dim == output_dim
    assert enhanced_map.input_dim == input_dim
    assert enhanced_map.neuron_type == neuron_type
    assert enhanced_map.network_structure == network_structure
    assert hasattr(enhanced_map, 'distance_matrix')
    assert tensor.shape(enhanced_map.distance_matrix) == (units, units)
    assert hasattr(enhanced_map, 'coordinates')
    assert tensor.shape(enhanced_map.coordinates) == (units, len(network_structure))

    # Test initialization without spatial properties
    enhanced_map_no_spatial = EnhancedNeuronMap(units=units, output_dim=output_dim, input_dim=input_dim, neuron_type=neuron_type)
    assert not hasattr(enhanced_map_no_spatial, 'distance_matrix')
    assert not hasattr(enhanced_map_no_spatial, 'coordinates')


def test_enhancedneuronmap_get_config_from_config():
    # Test get_config and from_config with EnhancedNeuronMap
    units = 25
    output_dim = 6
    input_dim = 12
    network_structure = (5, 5)
    neuron_type = "ltc"
    time_scale_factor = 1.2
    seed = 43

    original_map = EnhancedNeuronMap(
        units=units,
        output_dim=output_dim,
        input_dim=input_dim,
        neuron_type=neuron_type,
        neuron_params={"time_scale_factor": time_scale_factor},
        network_structure=network_structure,
        seed=seed
    )

    config = original_map.get_config()
    assert isinstance(config, dict)
    assert config['units'] == units
    assert config['output_size'] == output_dim # Check output_size in config
    assert config['input_size'] == input_dim # Check input_size in config
    assert config['neuron_type'] == neuron_type
    assert config['network_structure'] == network_structure
    assert config['seed'] == seed
    assert config['class_name'] == 'EnhancedNeuronMap'

    # Create new map from config
    new_map = EnhancedNeuronMap.from_config(config)
    assert isinstance(new_map, EnhancedNeuronMap)
    assert new_map.units == units
    assert new_map.output_dim == output_dim
    assert new_map.input_dim == input_dim
    assert new_map.neuron_type == neuron_type
    assert new_map.network_structure == network_structure
    assert new_map.seed == seed


def test_enhancedncpmap_initialization():
    # Test EnhancedNCPMap initialization with spatial properties
    sensory = 5
    inter = 10
    command = 4
    motor = 3
    network_structure = (4, 5, 1)
    neuron_type = "cfc"
    time_scale_factor = 0.8
    seed = 42

    enhanced_ncp_map = EnhancedNCPMap(
        sensory_neurons=sensory,
        inter_neurons=inter,
        command_neurons=command,
        motor_neurons=motor,
        neuron_type=neuron_type,
        time_scale_factor=time_scale_factor,
        network_structure=network_structure,
        seed=seed
    )

    assert isinstance(enhanced_ncp_map, EnhancedNCPMap)
    assert enhanced_ncp_map.sensory_neurons == sensory
    assert enhanced_ncp_map.inter_neurons == inter
    assert enhanced_ncp_map.command_neurons == command
    assert enhanced_ncp_map.motor_neurons == motor
    assert enhanced_ncp_map.units == inter + command + motor # Units are non-sensory neurons
    assert enhanced_ncp_map.output_dim == motor # Output dim is motor neurons
    assert enhanced_ncp_map.input_dim is None # Input dim is set during build
    assert enhanced_ncp_map.neuron_type == neuron_type
    assert enhanced_ncp_map.network_structure == network_structure
    assert hasattr(enhanced_ncp_map, 'distance_matrix')
    assert tensor.shape(enhanced_ncp_map.distance_matrix) == (enhanced_ncp_map.units + enhanced_ncp_map.sensory_neurons, enhanced_ncp_map.units + enhanced_ncp_map.sensory_neurons) # Distance matrix includes sensory neurons
    assert hasattr(enhanced_ncp_map, 'coordinates')
    assert tensor.shape(enhanced_ncp_map.coordinates) == (enhanced_ncp_map.units + enhanced_ncp_map.sensory_neurons, len(network_structure)) # Coordinates include sensory neurons


def test_enhancedncpmap_build():
    # Test EnhancedNCPMap build method
    sensory = 5
    inter = 10
    command = 4
    motor = 3
    network_structure = (4, 5, 1)
    seed = 42

    enhanced_ncp_map = EnhancedNCPMap(
        sensory_neurons=sensory,
        inter_neurons=inter,
        command_neurons=command,
        motor_neurons=motor,
        network_structure=network_structure,
        seed=seed
    )

    input_dim = 5 # Should match sensory_neurons if not overridden
    enhanced_ncp_map.build(input_dim)

    assert enhanced_ncp_map.input_dim == input_dim
    assert hasattr(enhanced_ncp_map, 'adjacency_matrix')
    assert hasattr(enhanced_ncp_map, 'sensory_adjacency_matrix')
    assert tensor.shape(enhanced_ncp_map.adjacency_matrix) == (enhanced_ncp_map.units, enhanced_ncp_map.units)
    assert tensor.shape(enhanced_ncp_map.sensory_adjacency_matrix) == (input_dim, enhanced_ncp_map.units)
    assert enhanced_ncp_map.synapse_count > 0 # Should have some connections
    assert enhanced_ncp_map.sensory_synapse_count > 0 # Should have some sensory connections
    assert hasattr(enhanced_ncp_map, 'communicability_matrix') # Should have communicability matrix after build
    assert tensor.shape(enhanced_ncp_map.communicability_matrix) == (enhanced_ncp_map.units, enhanced_ncp_map.units)


# Add more test functions for other enhanced map methods if any exist and are testable