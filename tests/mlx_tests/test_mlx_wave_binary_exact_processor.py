import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.binary import binary_exact_processor # Import the module
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

# Test cases for wave.binary.binary_exact_processor components

def test_binarywavestate_initialization():
    # Test BinaryWaveState initialization
    # BinaryWaveState uses Python's arbitrary precision integers, not tensors directly.
    state_int = 12345
    state = binary_exact_processor.BinaryWaveState(state_int)

    assert isinstance(state, binary_exact_processor.BinaryWaveState)
    assert state.state == state_int


def test_exactbinaryneuron_initialization():
    # Test ExactBinaryNeuron initialization
    # ExactBinaryNeuron uses Python integers for state and parameters.
    threshold = 10000
    exact_neuron = binary_exact_processor.ExactBinaryNeuron(threshold)

    assert isinstance(exact_neuron, binary_exact_processor.ExactBinaryNeuron)
    assert exact_neuron.threshold == threshold
    assert exact_neuron.state == 0 # Initial state should be 0


def test_exactbinaryneuron_process_input():
    # Test ExactBinaryNeuron process_input
    threshold = 100
    exact_neuron = binary_exact_processor.ExactBinaryNeuron(threshold)

    # Simulate processing some inputs
    input1 = 50
    output1 = exact_neuron.process_input(input1)
    assert exact_neuron.state == 50 # State should update
    assert output1 == 0 # Output should be 0 if state < threshold

    input2 = 60
    output2 = exact_neuron.process_input(input2)
    assert exact_neuron.state == 110 # State should update (50 + 60)
    assert output2 == 1 # Output should be 1 if state >= threshold
    assert exact_neuron.state == 10 # State should reset after firing (110 - 100)


def test_exactbinarynetwork_initialization():
    # Test ExactBinaryNetwork initialization
    num_neurons = 5
    threshold = 100
    exact_network = binary_exact_processor.ExactBinaryNetwork(num_neurons, threshold)

    assert isinstance(exact_network, binary_exact_processor.ExactBinaryNetwork)
    assert len(exact_network.neurons) == num_neurons
    for neuron in exact_network.neurons:
        assert isinstance(neuron, binary_exact_processor.ExactBinaryNeuron)
        assert neuron.threshold == threshold


def test_binaryexactprocessor_initialization():
    # Test BinaryExactProcessor initialization
    processor = binary_exact_processor.BinaryExactProcessor()
    assert isinstance(processor, binary_exact_processor.BinaryExactProcessor)


# Add more test functions for other binary_exact_processor components:
# test_binaryexactprocessor_pcm_to_exact_binary(), test_binaryexactprocessor_exact_binary_to_pcm(),
# test_exactbinarynetwork_process_input_sequence(), test_exactbinarynetwork_process_pcm()

# Note: Testing the conversion functions (pcm_to_exact_binary, exact_binary_to_pcm)
# might require creating dummy PCM data and verifying the integer representation.
# Testing the network processing will involve simulating input sequences.