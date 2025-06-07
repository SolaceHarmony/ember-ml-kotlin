import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave import binary_wave # Import the binary_wave module
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

# Fixture providing sample binary wave data
@pytest.fixture
def sample_binary_wave_data():
    """Create sample binary wave data."""
    # Use a consistent seed for reproducibility
    tensor.set_seed(42)
    # Create a simple binary wave pattern (e.g., a few pulses)
    pattern = tensor.zeros((10, 10), dtype=tensor.int32)
    pattern = tensor.tensor_scatter_nd_update(pattern, tensor.convert_to_tensor([[2, 3], [5, 6], [7, 8]]), tensor.ones(3, dtype=tensor.int32))
    return pattern

# Test cases for wave.binary_wave components

def test_waveconfig_initialization():
    # Test WaveConfig initialization
    grid_size = (10, 10)
    num_phases = 8
    fade_rate = 0.1
    threshold = 0.5
    config = binary_wave.WaveConfig(grid_size, num_phases, fade_rate, threshold)

    assert isinstance(config, binary_wave.WaveConfig)
    assert config.grid_size == grid_size
    assert config.num_phases == num_phases
    assert config.fade_rate == fade_rate
    assert config.threshold == threshold


def test_binarywave_initialization():
    # Test BinaryWave initialization
    grid_size = (10, 10)
    num_phases = 8
    binary_wave_module = binary_wave.BinaryWave(grid_size, num_phases)

    assert isinstance(binary_wave_module, binary_wave.BinaryWave)
    assert binary_wave_module.grid_size == grid_size
    assert binary_wave_module.num_phases == num_phases
    assert hasattr(binary_wave_module, 'phase_shift') # Should have learnable parameters
    assert hasattr(binary_wave_module, 'amplitude_scale')


def test_binarywave_encode_decode(sample_binary_wave_data):
    # Test BinaryWave encode and decode (round trip)
    pattern = sample_binary_wave_data
    grid_size = tensor.shape(pattern)
    num_phases = 8
    binary_wave_module = binary_wave.BinaryWave(grid_size, num_phases)

    # Encode and decode
    encoded_wave = binary_wave_module.encode(pattern)
    decoded_pattern = binary_wave_module.decode(encoded_wave)

    assert isinstance(encoded_wave, tensor.EmberTensor)
    assert isinstance(decoded_pattern, tensor.EmberTensor)
    assert tensor.shape(decoded_pattern) == tensor.shape(pattern)

    # Note: Encode/decode might not be perfectly reversible due to the nature of the transformation.
    # We can check if the decoded pattern is close to the original or has similar properties.
    # For a simple test, check if the shapes and dtypes are consistent.
    assert tensor.shape(decoded_pattern) == tensor.shape(pattern)
    assert tensor.dtype(decoded_pattern) == tensor.dtype(pattern)


def test_binarywaveprocessor_initialization():
    # Test BinaryWaveProcessor initialization
    processor = binary_wave.BinaryWaveProcessor()
    assert isinstance(processor, binary_wave.BinaryWaveProcessor)


def test_binarywaveprocessor_wave_interference(sample_binary_wave_data):
    # Test BinaryWaveProcessor wave_interference
    wave1 = sample_binary_wave_data
    wave2 = tensor.random_uniform(tensor.shape(wave1), minval=0, maxval=2, dtype=tensor.int32) # Create another dummy wave

    # Test XOR interference
    result_xor = binary_wave.BinaryWaveProcessor.wave_interference(wave1, wave2, mode='XOR')
    assert isinstance(result_xor, tensor.EmberTensor)
    assert tensor.shape(result_xor) == tensor.shape(wave1)
    assert tensor.dtype(result_xor) == tensor.dtype(wave1)

    # Test AND interference
    result_and = binary_wave.BinaryWaveProcessor.wave_interference(wave1, wave2, mode='AND')
    assert isinstance(result_and, tensor.EmberTensor)
    assert tensor.shape(result_and) == tensor.shape(wave1)
    assert tensor.dtype(result_and) == tensor.dtype(wave1)

    # Test OR interference
    result_or = binary_wave.BinaryWaveProcessor.wave_interference(wave1, wave2, mode='OR')
    assert isinstance(result_or, tensor.EmberTensor)
    assert tensor.shape(result_or) == tensor.shape(wave1)
    assert tensor.dtype(result_or) == tensor.dtype(wave1)

    # Test with invalid mode
    with pytest.raises(ValueError):
        binary_wave.BinaryWaveProcessor.wave_interference(wave1, wave2, mode='INVALID')


# Add more test functions for other binary_wave components:
# test_binarywaveprocessor_phase_similarity(), test_binarywaveprocessor_extract_features(),
# test_binarywaveencoder_initialization(), test_binarywaveencoder_encode_char(),
# test_binarywaveencoder_encode_sequence(), test_binarywavenetwork_initialization(),
# test_binarywavenetwork_forward()

# Note: Some tests might require more complex setups or mock dependencies.
# Initial tests focus on instantiation and basic calls/shape checks.