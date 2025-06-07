import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import FullyConnectedMap, NCPMap # Import necessary maps
from ember_ml.nn.modules.rnn import CfC, LTC # Import continuous RNN modules
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("mlx")

# --- Helper to assert tensor equality using ops ---
def assert_ops_equal(result_tensor, expected_tensor, message=""):
    """
    Asserts tensor equality using ops.equal and ops.all.
    Assumes both result_tensor and expected_tensor are EmberTensors.
    """
    assert isinstance(result_tensor, tensor.EmberTensor), \
        f"{message}: Result is not an EmberTensor: {type(result_tensor)}"
    assert isinstance(expected_tensor, tensor.EmberTensor), \
        f"{message}: Expected value is not an EmberTensor: {type(expected_tensor)}"

    assert tensor.shape(result_tensor) == tensor.shape(expected_tensor), \
        f"{message}: Shape mismatch. Got {tensor.shape(result_tensor)}, expected {tensor.shape(expected_tensor)}"

    equality_check = ops.equal(result_tensor, expected_tensor)
    all_equal_tensor = ops.all(equality_check)
    assert all_equal_tensor.item(), \
        f"{message}: Value mismatch. Got {result_tensor}, expected {expected_tensor}"

# Fixture providing test data
@pytest.fixture
def rnn_continuous_test_data():
    """Create test data for continuous RNN tests."""
    batch_size = 4
    seq_length = 10
    input_dim = 5
    hidden_size = 8 # For FullyConnectedMap
    units = 12 # For NCPMap (e.g., 3 sensory + 4 inter + 3 command + 2 motor)

    # Create input tensor
    inputs = tensor.random_normal(
        (batch_size, seq_length, input_dim)
    )

    # Create FullyConnectedMap
    fc_map = FullyConnectedMap(units=hidden_size, input_dim=input_dim, output_dim=hidden_size)

    # Create NCPMap (example dimensions)
    ncp_map = NCPMap(
        sensory_neurons=input_dim,
        inter_neurons=4,
        command_neurons=3,
        motor_neurons=2,
        seed=42
    )
    # Ensure NCPMap is built to get correct units/output_dim
    ncp_map.build(input_dim)


    return {
        'batch_size': batch_size,
        'seq_length': seq_length,
        'input_dim': input_dim,
        'hidden_size': hidden_size,
        'units': units,
        'inputs': inputs,
        'fc_map': fc_map,
        'ncp_map': ncp_map,
    }

# --- CfC Tests ---

def test_cfc_forward_fc_map(rnn_continuous_test_data):
    """Test CfC forward pass with FullyConnectedMap (NumPy backend)."""
    data = rnn_continuous_test_data
    # Create CfC model with FullyConnectedMap
    cfc_model = CfC(neuron_map=data['fc_map'], return_sequences=True)

    # Forward pass
    outputs = cfc_model(data['inputs'])

    # Check output shape (should match hidden_size for FullyConnectedMap)
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        data['hidden_size']
    )

    # Check output is not None or all zeros
    assert outputs is not None
    assert not ops.allclose(outputs, tensor.zeros_like(outputs)).item()

def test_cfc_forward_ncp_map(rnn_continuous_test_data):
    """Test CfC forward pass with NCPMap (NumPy backend)."""
    data = rnn_continuous_test_data
    # Create CfC model with NCPMap
    cfc_model = CfC(neuron_map=data['ncp_map'], return_sequences=True)

    # Forward pass
    outputs = cfc_model(data['inputs'])

    # Check output shape (should match motor_neurons for NCPMap)
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        data['ncp_map'].motor_neurons # Use motor_neurons for output dim
    )

    # Check output is not None or all zeros
    assert outputs is not None
    assert not ops.allclose(outputs, tensor.zeros_like(outputs)).item()

def test_cfc_return_state(rnn_continuous_test_data):
    """Test CfC with return_state=True (NumPy backend)."""
    data = rnn_continuous_test_data
    # Create CfC model with return_state=True
    cfc_model = CfC(neuron_map=data['fc_map'], return_sequences=True, return_state=True)

    # Forward pass
    outputs, state = cfc_model(data['inputs'])

    # Check output shape
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        data['hidden_size']
    )

    # Check state shape (should be a list of two tensors: h and t)
    assert isinstance(state, list)
    assert len(state) == 2
    assert tensor.shape(state[0]) == (data['batch_size'], data['hidden_size']) # h state
    assert tensor.shape(state[1]) == (data['batch_size'], data['hidden_size']) # t state

# Add more CfC tests: test_cfc_no_sequences, test_cfc_with_time_deltas, etc.


# --- LTC Tests ---

def test_ltc_forward_fc_map(rnn_continuous_test_data):
    """Test LTC forward pass with FullyConnectedMap (NumPy backend)."""
    data = rnn_continuous_test_data
    # Create LTC model with FullyConnectedMap
    ltc_model = LTC(neuron_map=data['fc_map'], return_sequences=True)

    # Forward pass
    outputs = ltc_model(data['inputs'])

    # Check output shape (should match hidden_size for FullyConnectedMap)
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        data['hidden_size']
    )

    # Check output is not None or all zeros
    assert outputs is not None
    assert not ops.allclose(outputs, tensor.zeros_like(outputs)).item()

def test_ltc_forward_ncp_map(rnn_continuous_test_data):
    """Test LTC forward pass with NCPMap (NumPy backend)."""
    data = rnn_continuous_test_data
    # Create LTC model with NCPMap
    ltc_model = LTC(neuron_map=data['ncp_map'], return_sequences=True)

    # Forward pass
    outputs = ltc_model(data['inputs'])

    # Check output shape (should match motor_neurons for NCPMap)
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        data['ncp_map'].motor_neurons # Use motor_neurons for output dim
    )

    # Check output is not None or all zeros
    assert outputs is not None
    assert not ops.allclose(outputs, tensor.zeros_like(outputs)).item()

def test_ltc_return_state(rnn_continuous_test_data):
    """Test LTC with return_state=True (NumPy backend)."""
    data = rnn_continuous_test_data
    # Create LTC model with return_state=True
    ltc_model = LTC(neuron_map=data['fc_map'], return_sequences=True, return_state=True)

    # Forward pass
    outputs, state = ltc_model(data['inputs'])

    # Check output shape
    assert tensor.shape(outputs) == (
        data['batch_size'],
        data['seq_length'],
        data['hidden_size']
    )

    # Check state shape (should be a single tensor: h)
    assert isinstance(state, tensor.EmberTensor) # LTC state is a single tensor
    assert tensor.shape(state) == (data['batch_size'], data['hidden_size']) # h state

# Add more LTC tests: test_ltc_no_sequences, test_ltc_with_time_deltas, etc.