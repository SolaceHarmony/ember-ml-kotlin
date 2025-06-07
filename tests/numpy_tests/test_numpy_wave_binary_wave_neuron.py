import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.binary import binary_wave_neuron # Import the module
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

# Test cases for wave.binary.binary_wave_neuron components

def test_binarywaveneuron_initialization():
    # Test BinaryWaveNeuron initialization
    # BinaryWaveNeuron uses tensors for state and parameters.
    wave_max = 1000.0
    binary_neuron = binary_wave_neuron.BinaryWaveNeuron(wave_max)

    assert isinstance(binary_neuron, binary_wave_neuron.BinaryWaveNeuron)
    assert binary_neuron.wave_max == wave_max
    assert hasattr(binary_neuron, 'state') # Should have a tensor state
    assert tensor.shape(binary_neuron.state) == () # Assuming scalar state
    assert ops.allclose(binary_neuron.state, 0.0).item() # Initial state should be 0.0


def test_binarywaveneuron_process_input():
    # Test BinaryWaveNeuron process_input
    wave_max = 100.0
    binary_neuron = binary_wave_neuron.BinaryWaveNeuron(wave_max)

    # Simulate processing some inputs
    input1 = tensor.convert_to_tensor(50.0)
    output1 = binary_neuron.process_input(input1)
    # State update logic depends on internal implementation (e.g., ion channels, leak)
    # We can check that the state and output are tensors of expected shape/dtype.
    assert isinstance(binary_neuron.state, tensor.EmberTensor)
    assert tensor.shape(binary_neuron.state) == ()
    assert isinstance(output1, tensor.EmberTensor)
    assert tensor.shape(output1) == ()

    input2 = tensor.convert_to_tensor(60.0)
    output2 = binary_neuron.process_input(input2)
    assert isinstance(binary_neuron.state, tensor.EmberTensor)
    assert tensor.shape(binary_neuron.state) == ()
    assert isinstance(output2, tensor.EmberTensor)
    assert tensor.shape(output2) == ()

    # More detailed tests would require understanding the exact internal dynamics
    # and comparing outputs/states to expected values based on those dynamics.


def test_binarywavenetwork_initialization():
    # Test BinaryWaveNetwork initialization
    num_neurons = 5
    wave_max = 1000.0
    binary_network = binary_wave_neuron.BinaryWaveNetwork(num_neurons, wave_max)

    assert isinstance(binary_network, binary_wave_neuron.BinaryWaveNetwork)
    assert len(binary_network.neurons) == num_neurons
    for neuron in binary_network.neurons:
        assert isinstance(neuron, binary_wave_neuron.BinaryWaveNeuron)
        assert neuron.wave_max == wave_max


# Add more test functions for other binary_wave_neuron components:
# test_binarywavenetwork_process_pcm(), test_create_test_signal()

# Note: Testing the network processing (process_pcm) will involve simulating
# PCM input data and verifying the output.