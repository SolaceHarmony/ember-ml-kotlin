import pytest
import numpy as np # For comparison with known correct results
import math # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.wave import harmonic # Import the harmonic module
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

# Fixture providing sample wave data
@pytest.fixture
def sample_wave_data():
    """Create sample wave data."""
    sample_rate = 1000
    duration = 1.0
    t = tensor.linspace(0, duration, int(duration * sample_rate))
    # Create a simple sine wave
    wave = ops.sin(2 * ops.pi * 5 * t) + 0.5 * ops.sin(2 * ops.pi * 15 * t)
    return tensor.convert_to_tensor(wave, dtype=tensor.float32), sample_rate

# Test cases for wave.harmonic components

def test_frequencyanalyzer_initialization():
    # Test FrequencyAnalyzer initialization
    sample_rate = 1000
    window_size = 256
    overlap = 128
    analyzer = harmonic.FrequencyAnalyzer(sample_rate, window_size, overlap)

    assert isinstance(analyzer, harmonic.FrequencyAnalyzer)
    assert analyzer.sampling_rate == sample_rate
    assert analyzer.window_size == window_size
    assert analyzer.overlap == overlap


def test_frequencyanalyzer_compute_spectrum(sample_wave_data):
    # Test FrequencyAnalyzer compute_spectrum
    wave_data, sample_rate = sample_wave_data
    window_size = 256
    overlap = 128
    analyzer = harmonic.FrequencyAnalyzer(sample_rate, window_size, overlap)

    # compute_spectrum might rely on scipy.fft or numpy.fft
    # We can test that it runs without errors and returns expected types/shapes if dependencies are met.
    try:
        frequencies, magnitudes = analyzer.compute_spectrum(wave_data)
        assert isinstance(frequencies, TensorLike) # Assuming numpy array return
        assert isinstance(magnitudes, TensorLike) # Assuming numpy array return
        assert frequencies.shape == magnitudes.shape
        assert len(frequencies.shape) == 1 # Should be 1D arrays
    except ImportError:
        pytest.skip("Skipping compute_spectrum test: scipy or numpy.fft not available")
    except Exception as e:
        pytest.fail(f"compute_spectrum raised an exception: {e}")


def test_wavesynthesizer_initialization():
    # Test WaveSynthesizer initialization
    sample_rate = 1000
    synthesizer = harmonic.WaveSynthesizer(sample_rate)
    assert isinstance(synthesizer, harmonic.WaveSynthesizer)
    assert synthesizer.sampling_rate == sample_rate


def test_wavesynthesizer_sine_wave():
    # Test WaveSynthesizer sine_wave
    sample_rate = 1000
    synthesizer = harmonic.WaveSynthesizer(sample_rate)
    frequency = 10.0
    duration = 0.5
    amplitude = 1.0
    phase = 0.0

    sine_wave = synthesizer.sine_wave(frequency, duration, amplitude, phase)
    assert isinstance(sine_wave, TensorLike) # Assuming numpy array return
    assert len(sine_wave.shape) == 1 # Should be 1D array
    expected_length = int(duration * sample_rate)
    assert sine_wave.shape[0] == expected_length

    # Check values at specific points (basic check)
    t = tensor.linspace(0, duration, expected_length)
    expected_wave = amplitude * ops.sin(2 * ops.pi * frequency * t + phase)
    assert ops.allclose(sine_wave, expected_wave)


# Add more test functions for other harmonic components:
# test_frequencyanalyzer_find_peaks(), test_frequencyanalyzer_harmonic_ratio(),
# test_wavesynthesizer_harmonic_wave(), test_wavesynthesizer_apply_envelope(),
# test_harmonicprocessor_initialization(), test_harmonicprocessor_decompose(),
# test_harmonicprocessor_reconstruct(), test_harmonicprocessor_filter_harmonics(),
# test_embeddinggenerator_initialization(), test_embeddinggenerator_generate_embeddings(),
# test_harmonictrainer_initialization(), test_harmonictrainer_loss_function(),
# test_harmonictrainer_compute_gradients(), test_harmonictrainer_train(),
# test_harmonicvisualizer_initialization(), test_harmonicvisualizer_plot_embeddings(),
# test_harmonicvisualizer_plot_wave_comparison(), test_harmonicvisualizer_animate_training(),
# test_harmonic_wave_generator(), test_map_embeddings_to_harmonics_generator()

# Note: Some tests might require mocking external dependencies (like transformer models for EmbeddingGenerator)
# or complex setups (like training for HarmonicTrainer). Initial tests can focus on instantiation and basic calls.