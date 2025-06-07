import pytest
import numpy as np
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.wave.models.wave_rnn import WaveGRUCell, WaveGRU, WaveRNN, create_wave_rnn
from ember_ml.nn.modules import Module # Needed for isinstance checks
from ember_ml.nn.modules.rnn import LSTM # For testing WaveRNN with LSTM

@pytest.fixture(params=['torch'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')

# Helper function to create dummy input data
def create_dummy_input_data(shape=(32, 10, 10)):
    """Creates a dummy input tensor for RNN models."""
    return tensor.random_normal(shape, dtype=tensor.float32)

# Test cases for initialization and forward pass shapes

def test_wavegrucell_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveGRUCell initialization and forward pass shape (single step)."""
    input_size = 10
    hidden_size = 20
    cell = WaveGRUCell(input_size=input_size, hidden_size=hidden_size)
    assert isinstance(cell, WaveGRUCell)
    assert isinstance(cell, Module)
    input_data = create_dummy_input_data(shape=(32, input_size)) # Batch, Features
    hidden_state = tensor.zeros((32, hidden_size), dtype=tensor.float32) # Batch, Hidden
    output, new_hidden_state = cell(input_data, hidden_state)
    assert tensor.shape(output) == (32, hidden_size)
    assert tensor.shape(new_hidden_state) == (32, hidden_size)

def test_wavegru_initialization_and_forward_shape(set_backend_fixture):
    """Test WaveGRU initialization and forward pass shape (sequence)."""
    input_size = 10
    hidden_size = 20
    gru_layer = WaveGRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    assert isinstance(gru_layer, WaveGRU)
    assert isinstance(gru_layer, Module)
    input_sequence = create_dummy_input_data(shape=(32, 5, input_size)) # Batch, Seq, Features
    output, final_state = gru_layer(input_sequence)
    # Default return_sequences=True, return_state=True
    assert tensor.shape(output) == (32, 5, hidden_size)
    assert tensor.shape(final_state) == (1, 32, hidden_size) # (num_layers * num_directions, Batch, Hidden)

def test_wavegru_forward_return_sequences_false(set_backend_fixture):
    """Test WaveGRU forward pass with return_sequences=False."""
    input_size = 10
    hidden_size = 20
    gru_layer = WaveGRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, return_sequences=False)
    input_sequence = create_dummy_input_data(shape=(32, 5, input_size))
    output, final_state = gru_layer(input_sequence)
    assert tensor.shape(output) == (32, hidden_size)
    assert tensor.shape(final_state) == (1, 32, hidden_size)

def test_wavegru_forward_return_state_false(set_backend_fixture):
    """Test WaveGRU forward pass with return_state=False."""
    input_size = 10
    hidden_size = 20
    gru_layer = WaveGRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, return_state=False)
    input_sequence = create_dummy_input_data(shape=(32, 5, input_size))
    output = gru_layer(input_sequence)
    assert tensor.shape(output) == (32, 5, hidden_size) # Default return_sequences=True

def test_wavernn_initialization_and_forward_shape_gru(set_backend_fixture):
    """Test WaveRNN initialization and forward pass shape with GRU."""
    input_size = 10
    hidden_size = 20
    rnn_model = WaveRNN(rnn_type='GRU', input_size=input_size, hidden_size=hidden_size, batch_first=True)
    assert isinstance(rnn_model, WaveRNN)
    assert isinstance(rnn_model, Module)
    assert isinstance(rnn_model.rnn_layer, WaveGRU) # Check if the correct layer type is wrapped
    input_sequence = create_dummy_input_data(shape=(32, 5, input_size))
    output, final_state = rnn_model(input_sequence)
    # Default return_sequences=True, return_state=True for the wrapped layer
    assert tensor.shape(output) == (32, 5, hidden_size)
    assert tensor.shape(final_state) == (1, 32, hidden_size)

def test_wavernn_initialization_and_forward_shape_lstm(set_backend_fixture):
    """Test WaveRNN initialization and forward pass shape with LSTM."""
    input_size = 10
    hidden_size = 20
    # Note: LSTM requires complex cell dynamics which might not be fully supported by NumPy.
    # This test is included but may need to be skipped or adapted for specific backends.
    if ops.get_backend() == 'numpy':
         pytest.skip("NumPy backend may not fully support complex RNN cell dynamics for LSTM.")

    rnn_model = WaveRNN(rnn_type='LSTM', input_size=input_size, hidden_size=hidden_size, batch_first=True)
    assert isinstance(rnn_model, WaveRNN)
    assert isinstance(rnn_model, Module)
    assert isinstance(rnn_model.rnn_layer, LSTM) # Check if the correct layer type is wrapped
    input_sequence = create_dummy_input_data(shape=(32, 5, input_size))
    output, final_state = rnn_model(input_sequence)
    # Default return_sequences=True, return_state=True for the wrapped layer
    assert tensor.shape(output) == (32, 5, hidden_size)
    assert isinstance(final_state, tuple) and len(final_state) == 2 # LSTM state is (h, c)
    assert tensor.shape(final_state[0]) == (1, 32, hidden_size)
    assert tensor.shape(final_state[1]) == (1, 32, hidden_size)


def test_create_wave_rnn_factory(set_backend_fixture):
    """Test create_wave_rnn factory function."""
    input_size = 10
    hidden_size = 20
    rnn_model = create_wave_rnn(rnn_type='GRU', input_size=input_size, hidden_size=hidden_size, batch_first=True)
    assert isinstance(rnn_model, WaveRNN)
    assert isinstance(rnn_model, Module)
    assert isinstance(rnn_model.rnn_layer, WaveGRU) # Check if the correct layer type is wrapped
    input_sequence = create_dummy_input_data(shape=(32, 5, input_size))
    output, final_state = rnn_model(input_sequence)
    assert tensor.shape(output) == (32, 5, hidden_size)
    assert tensor.shape(final_state) == (1, 32, hidden_size)

# TODO: Add tests for parameter registration
# TODO: Add tests for different activation functions and dropout rates in GRU/RNN
# TODO: Add tests for edge cases and invalid inputs
# TODO: Add tests for bidirectional RNNs
# TODO: Add tests for multi-layer RNNs
# Note: Testing the correctness of RNN cell dynamics might require
# complex operations which may not be fully supported by all backends (e.g., NumPy).
# These tests are included but may need to be skipped or adapted for specific backends.