import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import NeuronMap # Import the base class
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

# --- Concrete subclass for testing abstract NeuronMap ---
class ConcreteNeuronMap(NeuronMap):
    """A concrete subclass of NeuronMap for testing purposes."""
    def __init__(self, units, output_size, input_size=None, **kwargs):
        super().__init__(units, output_size, input_size, **kwargs)
        # Initialize dummy adjacency matrices for testing common methods
        self.adjacency_matrix = tensor.ones((self.units, self.units))
        if self.input_size is not None:
            self.sensory_adjacency_matrix = tensor.ones((self.input_size, self.units))
        else:
             self.sensory_adjacency_matrix = None

    def build_adjacency_matrix(self):
        # Dummy implementation
        pass

    def build_sensory_adjacency_matrix(self):
        # Dummy implementation
        pass

    def get_config(self):
        config = super().get_config()
        # Add any specific config for this concrete map if needed
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Test cases for nn.modules.wiring.NeuronMap base class

def test_neuronmap_initialization():
    # Test NeuronMap initialization with concrete subclass
    units = 10
    output_size = 3
    input_size = 5
    neuron_map = ConcreteNeuronMap(units, output_size, input_size)

    assert isinstance(neuron_map, NeuronMap)
    assert neuron_map.units == units
    assert neuron_map.output_dim == output_size # Check output_dim property
    assert neuron_map.input_dim == input_size # Check input_dim property
    assert hasattr(neuron_map, 'adjacency_matrix')
    assert hasattr(neuron_map, 'sensory_adjacency_matrix')
    assert tensor.shape(neuron_map.adjacency_matrix) == (units, units)
    assert tensor.shape(neuron_map.sensory_adjacency_matrix) == (input_size, units)

    # Test initialization without input_size
    neuron_map_no_input = ConcreteNeuronMap(units, output_size)
    assert neuron_map_no_input.input_dim is None
    assert not hasattr(neuron_map_no_input, 'sensory_adjacency_matrix')


def test_neuronmap_get_config_from_config():
    # Test get_config and from_config with concrete subclass
    units = 15
    output_size = 4
    input_size = 6
    original_map = ConcreteNeuronMap(units, output_size, input_size)

    config = original_map.get_config()
    assert isinstance(config, dict)
    assert config['units'] == units
    assert config['output_size'] == output_size
    assert config['input_size'] == input_size
    assert config['class_name'] == 'ConcreteNeuronMap' # Check class name in config

    # Create new map from config
    new_map = NeuronMap.from_config(config) # Use base class from_config
    assert isinstance(new_map, ConcreteNeuronMap)
    assert new_map.units == units
    assert new_map.output_dim == output_size
    assert new_map.input_dim == input_size

# Add more test functions for other common NeuronMap methods if any exist and are testable on the base class