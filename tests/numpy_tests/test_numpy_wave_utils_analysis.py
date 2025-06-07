import pytest
import numpy as np # For comparison with known correct results
import math # For comparison with known correct results

# Import Ember ML modules
from ember_ml.backend.numpy.types import TensorLike # Import TensorLike
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave.utils import wave_analysis # Import the module
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

# Fixture providing sample wave data for analysis tests
@pytest.fixture
def sample_analysis_wave_data():
    """Create sample wave data for analysis tests."""
    sample_rate = 1000
    duration = 1.0
    t = tensor.linspace(0, duration, int(duration * sample_rate))
    # Create a simple sine wave
    wave = ops.sin(2 * ops.pi * 50 * t) + 0.5 * ops.sin(2 * ops.pi * 150 * t)
    return tensor.convert_to_tensor(wave, dtype=tensor.float32), sample_rate

# Test cases for wave.utils.wave_analysis functions

def test_compute_fft(sample_analysis_wave_data):
    # Test compute_fft
    wave_data, sample_rate = sample_analysis_wave_data

    # compute_fft might rely on numpy.fft or scipy.fft
    # We can test that it runs without errors and returns expected types/shapes if dependencies are met.
    try:
        frequencies, magnitudes = wave_analysis.compute_fft(wave_data, sample_rate)
        assert isinstance(frequencies, TensorLike) # Assuming numpy array return
        assert isinstance(magnitudes, TensorLike) # Assuming numpy array return
        assert frequencies.shape == magnitudes.shape
        assert len(frequencies.shape) == 1 # Should be 1D arrays
        # Check frequency range (should go up to Nyquist frequency)
        assert ops.all(frequencies >= 0)
        assert ops.all(frequencies <= sample_rate / 2.0)
    except ImportError:
        pytest.skip("Skipping compute_fft test: numpy.fft or scipy.fft not available")
    except Exception as e:
        pytest.fail(f"compute_fft raised an exception: {e}")


def test_compute_stft(sample_analysis_wave_data):
    # Test compute_stft
    wave_data, sample_rate = sample_analysis_wave_data
    window_size = 256
    hop_length = 64

    # compute_stft might rely on scipy.signal.stft
    # We can test that it runs without errors and returns expected types/shapes if dependencies are met.
    try:
        frequencies, times, spectrogram = wave_analysis.compute_stft(wave_data, sample_rate, window_size, hop_length)
        assert isinstance(frequencies, TensorLike)
        assert isinstance(times, TensorLike)
        assert isinstance(spectrogram, TensorLike)
        assert spectrogram.shape == (len(frequencies), len(times)) # Shape (n_freqs, n_times)
    except ImportError:
        pytest.skip("Skipping compute_stft test: scipy not installed")
    except Exception as e:
        pytest.fail(f"compute_stft raised an exception: {e}")


def test_compute_rms(sample_analysis_wave_data):
    # Test compute_rms
    wave_data, _ = sample_analysis_wave_data
    result = wave_analysis.compute_rms(wave_data)

    assert isinstance(result, (float, tensor.floating)) # Should return a scalar float
    # Calculate expected RMS manually
    expected_rms = ops.sqrt(stats.mean(np.square(tensor.to_numpy(wave_data))))
    assert abs(result - expected_rms) < 1e-6


def test_compute_peak_amplitude(sample_analysis_wave_data):
    # Test compute_peak_amplitude
    wave_data, _ = sample_analysis_wave_data
    result = wave_analysis.compute_peak_amplitude(wave_data)

    assert isinstance(result, (float, tensor.floating)) # Should return a scalar float
    # Calculate expected peak amplitude manually
    expected_peak = stats.max(ops.abs(tensor.to_numpy(wave_data)))
    assert abs(result - expected_peak) < 1e-6


def test_compute_crest_factor(sample_analysis_wave_data):
    # Test compute_crest_factor
    wave_data, _ = sample_analysis_wave_data
    result = wave_analysis.compute_crest_factor(wave_data)

    assert isinstance(result, (float, tensor.floating)) # Should return a scalar float
    # Calculate expected crest factor manually
    peak_amplitude = stats.max(ops.abs(tensor.to_numpy(wave_data)))
    rms = ops.sqrt(stats.mean(np.square(tensor.to_numpy(wave_data))))
    # Avoid division by zero if RMS is zero
    expected_crest_factor = peak_amplitude / rms if rms != 0 else 0.0
    assert abs(result - expected_crest_factor) < 1e-6


def test_compute_dominant_frequency(sample_analysis_wave_data):
    # Test compute_dominant_frequency
    wave_data, sample_rate = sample_analysis_wave_data

    # compute_dominant_frequency relies on compute_fft
    # We can test that it runs without errors and returns a float if dependencies are met.
    try:
        dominant_freq = wave_analysis.compute_dominant_frequency(wave_data, sample_rate)
        assert isinstance(dominant_freq, (float, tensor.floating)) # Should return a scalar float
        # For the sample data (50Hz and 150Hz sine waves), the dominant frequency should be 50Hz.
        assert abs(dominant_freq - 50.0) < 1.0 # Allow some tolerance
    except ImportError:
        pytest.skip("Skipping compute_dominant_frequency test: numpy.fft or scipy.fft not available")
    except Exception as e:
        pytest.fail(f"compute_dominant_frequency raised an exception: {e}")


# Add more test functions for other wave_analysis functions:
# test_compute_mfcc(), test_compute_spectral_centroid(), test_compute_spectral_bandwidth(),
# test_compute_spectral_contrast(), test_compute_spectral_rolloff(), test_compute_zero_crossing_rate(),
# test_compute_harmonic_ratio(), test_compute_wave_features()

# Note: Many of these functions rely on external libraries like librosa or scipy.
# Tests for these functions should be skipped if the dependencies are not met.