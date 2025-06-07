import pytest
import numpy as np # For comparison with known correct results
import math # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.wave.utils import wave_conversion # Import the module
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

# Test cases for wave.utils.wave_conversion functions

def test_pcm_to_float_and_back():
    # Test pcm_to_float and float_to_pcm round trip
    # Create dummy 16-bit PCM data (NumPy array)
    pcm_data_int16 = tensor.convert_to_tensor([0, 10000, -10000, 32767, -32768], dtype=tensor.int16)

    # Convert to float
    float_data = wave_conversion.pcm_to_float(pcm_data_int16, dtype=tensor.float32)
    assert isinstance(float_data, TensorLike)
    assert float_data.dtype == tensor.float32
    assert ops.all(float_data >= -1.0)
    assert ops.all(float_data <= 1.0)
    assert ops.allclose(float_data, pcm_data_int16 / 32768.0) # Check conversion logic

    # Convert back to PCM
    reconstructed_pcm = wave_conversion.float_to_pcm(float_data, dtype=tensor.int16)
    assert isinstance(reconstructed_pcm, TensorLike)
    assert reconstructed_pcm.dtype == tensor.int16
    # Allow some tolerance for round trip due to floating point
    assert ops.allclose(reconstructed_pcm, pcm_data_int16, atol=1) # Allow small error


def test_pcm_to_db_and_back():
    # Test pcm_to_db and db_to_amplitude round trip
    # Create dummy PCM data (NumPy array, float32)
    pcm_data_float = tensor.convert_to_tensor([0.1, 0.5, 1.0, 0.01], dtype=tensor.float32)
    ref = 1.0
    min_db = -60.0

    # Convert to dB
    db_data = wave_conversion.pcm_to_db(pcm_data_float, ref=ref, min_db=min_db)
    assert isinstance(db_data, TensorLike)
    # Check range (should be <= 0 dB if ref is max amplitude)
    assert ops.all(db_data <= 0.0)
    assert ops.all(db_data >= min_db)

    # Convert back to amplitude
    reconstructed_amplitude = wave_conversion.db_to_amplitude(db_data)
    assert isinstance(reconstructed_amplitude, TensorLike)
    # Allow some tolerance for round trip
    assert ops.allclose(reconstructed_amplitude, pcm_data_float, atol=1e-5)


def test_amplitude_to_db():
    # Test amplitude_to_db
    amplitude_data = tensor.convert_to_tensor([0.1, 0.5, 1.0, 0.01], dtype=tensor.float32)
    min_db = -60.0

    db_data = wave_conversion.amplitude_to_db(amplitude_data, min_db=min_db)
    assert isinstance(db_data, TensorLike)
    assert ops.all(db_data <= 0.0)
    assert ops.all(db_data >= min_db)


def test_pcm_to_binary_and_back():
    # Test pcm_to_binary and binary_to_pcm round trip
    # Create dummy PCM data (NumPy array, float32)
    pcm_data_float = tensor.convert_to_tensor([-0.5, 0.1, 0.6, -0.8, 0.9], dtype=tensor.float32)
    threshold = 0.5

    # Convert to binary
    binary_data = wave_conversion.pcm_to_binary(pcm_data_float, threshold=threshold)
    assert isinstance(binary_data, TensorLike)
    assert ops.all(ops.logical_or(binary_data == 0, binary_data == 1)) # Should be binary
    # Check conversion logic
    expected_binary = tensor.convert_to_tensor([0, 0, 1, 0, 1], dtype=tensor.int32)
    assert tensor.convert_to_tensor_equal(binary_data, expected_binary)

    # Convert back to PCM
    amplitude = 1.0
    reconstructed_pcm = wave_conversion.binary_to_pcm(binary_data, amplitude=amplitude, dtype=tensor.float32)
    assert isinstance(reconstructed_pcm, TensorLike)
    assert reconstructed_pcm.dtype == tensor.float32
    # Check conversion logic
    expected_pcm = tensor.convert_to_tensor([0.0, 0.0, 1.0, 0.0, 1.0], dtype=tensor.float32) * amplitude
    assert ops.allclose(reconstructed_pcm, expected_pcm)


def test_pcm_to_phase_and_back():
    # Test pcm_to_phase and phase_to_pcm round trip
    # Create dummy PCM data (NumPy array, float32)
    # Use a signal with known phase properties (e.g., a simple sine wave)
    sample_rate = 1000
    duration = 1.0
    t = tensor.linspace(0, duration, sample_rate)
    pcm_data_float = ops.sin(2 * ops.pi * 10 * t).astype(tensor.float32)

    # pcm_to_phase relies on numpy.fft
    # We can test that it runs without errors and returns expected types/shapes if dependencies are met.
    try:
        phase_data = wave_conversion.pcm_to_phase(pcm_data_float)
        assert isinstance(phase_data, TensorLike)
        # The shape depends on the FFT size, which is related to the input length.
        # For real input, FFT result has length N/2 + 1. Phase should have the same shape.
        expected_shape = (sample_rate // 2 + 1,)
        assert phase_data.shape == expected_shape
        # Check phase range (-pi to pi)
        assert ops.all(phase_data >= -ops.pi)
        assert ops.all(phase_data <= ops.pi)

        # phase_to_pcm relies on numpy.fft.ifft
        # It also requires magnitude data, which is not returned by pcm_to_phase.
        # This round trip test is difficult without the magnitude.
        # We can test phase_to_pcm with dummy phase and magnitude data.
        # For now, just test pcm_to_phase.

    except ImportError:
        pytest.skip("Skipping pcm_to_phase test: numpy.fft not available")
    except Exception as e:
        pytest.fail(f"pcm_to_phase raised an exception: {e}")


# Add more test functions for other wave_conversion functions if any exist and are testable