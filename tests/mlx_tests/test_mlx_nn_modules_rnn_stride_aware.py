import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import FullyConnectedMap # StrideAwareCfC uses a NeuronMap
from ember_ml.nn.modules.rnn.stride_aware_cfc_layer import StrideAwareCfC # Import StrideAwareCfC
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

# Fixture providing test data
@pytest.fixture
def stride_aware_test_data():
    """Create test data for StrideAwareCfC tests."""
    batch_size = 2
    seq_length = 20 # Need a longer sequence for strides
    input_dim = 4
    hidden_size = 8
    stride_lengths = [1, 2, 4]

    # Create input tensor
    inputs = tensor.random_normal(
        (batch_size, seq_length, input_dim)
    )

    # Create a NeuronMap for the StrideAwareCfC
    neuron_map = FullyConnectedMap(units=hidden_size, input_dim=input_dim, output_dim=hidden_size)


    return {
        'batch_size': batch_size,
        'seq_length': seq_length,
        'input_dim': input_dim,
        'hidden_size': hidden_size,
        'stride_lengths': stride_lengths,
        'inputs': inputs,
        'neuron_map': neuron_map,
    }

# --- StrideAwareCfC Tests ---

def test_stride_aware_cfc_forward(stride_aware_test_data):
    """Test StrideAwareCfC forward pass (MLX backend)."""
    data = stride_aware_test_data
    # Create StrideAwareCfC model
    stride_cfc_model = StrideAwareCfC(
        neuron_map=data['neuron_map'],
        stride_lengths=data['stride_lengths'],
        return_sequences=True # StrideAwareCfC typically returns sequences
    )

    # Forward pass
    outputs = stride_cfc_model(data['inputs'])

    # Check output shape
    # The output shape depends on how the outputs from different strides are combined.
    # Assuming they are concatenated along the feature dimension.
    # Output features per stride = hidden_size
    # Total output features = len(stride_lengths) * hidden_size
    expected_output_features = len(data['stride_lengths']) * data['hidden_size']

    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        expected_output_features
    )

    # Check output is not None or all zeros
    assert outputs is not None
    assert not ops.allclose(outputs, tensor.zeros_like(outputs)).item()

def test_stride_aware_cfc_return_state(stride_aware_test_data):
    """Test StrideAwareCfC with return_state=True (MLX backend)."""
    data = stride_aware_test_data
    # Create StrideAwareCfC model with return_state=True
    stride_cfc_model = StrideAwareCfC(
        neuron_map=data['neuron_map'],
        stride_lengths=data['stride_lengths'],
        return_sequences=True,
        return_state=True
    )

    # Forward pass
    outputs, state = stride_cfc_model(data['inputs'])

    # Check output shape
    expected_output_features = len(data['stride_lengths']) * data['hidden_size']
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        expected_output_features
    )

    # Check state shape
    # The state should be a list of states, one for each stride.
    # Each state for a CfC-like cell is typically a list of two tensors (h and t).
    # So, the state should be a list of lists of tensors.
    assert isinstance(state, list)
    assert len(state) == len(data['stride_lengths']) # One state per stride

    # Check the shape of each stride's state
    for stride_state in state:
        assert isinstance(stride_state, list)
        assert len(stride_state) == 2 # h and t states
        assert tensor.shape(stride_state[0]) == (data['batch_size'], data['hidden_size']) # h state
        assert tensor.shape(stride_state[1]) == (data['batch_size'], data['hidden_size']) # t state


# Add more StrideAwareCfC tests: test_stride_aware_cfc_no_sequences,
# test_stride_aware_cfc_different_strides, etc.