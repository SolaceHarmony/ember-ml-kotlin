import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.wave import generator # Import the generator module
from ember_ml.ops import set_backend
from ember_ml.backend.mlx.stats import descriptive as mlx_stats

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for wave.generator components

def test_signalsynthesizer_initialization():
    # Test SignalSynthesizer initialization
    sample_rate = 1000
    synthesizer = generator.SignalSynthesizer(sample_rate)

    assert isinstance(synthesizer, generator.SignalSynthesizer)
    assert synthesizer.sampling_rate == sample_rate


def test_signalsynthesizer_sine_wave():
    # Test SignalSynthesizer sine_wave
    sample_rate = 1000
    synthesizer = generator.SignalSynthesizer(sample_rate)
    frequency = 10.0
    duration = 0.5
    amplitude = 1.0
    phase = 0.0

    sine_wave = synthesizer.sine_wave(frequency, duration, amplitude, phase)
    assert sine_wave is not None # Just check it's not None
    assert len(sine_wave.shape) == 1 # Should be 1D array
    expected_length = int(duration * sample_rate)
    assert sine_wave.shape[0] == expected_length

    # Check values at specific points (basic check)
    t = tensor.linspace(0, duration, expected_length)  # Remove endpoint parameter
    expected_wave = amplitude * ops.sin(2 * ops.pi * frequency * t + phase)
    assert ops.allclose(sine_wave, expected_wave, atol=1e-5)  # Add tolerance


def test_signalsynthesizer_square_wave():
    # Test SignalSynthesizer square_wave
    sample_rate = 1000
    synthesizer = generator.SignalSynthesizer(sample_rate)
    frequency = 10.0
    duration = 0.5
    amplitude = 1.0
    duty_cycle = 0.5

    square_wave = synthesizer.square_wave(frequency, duration, amplitude, duty_cycle)
    assert square_wave is not None
    assert len(square_wave.shape) == 1
    expected_length = int(duration * sample_rate)
    assert square_wave.shape[0] == expected_length

    # Check values (should be amplitude or -amplitude)
    assert ops.all(ops.logical_or(ops.isclose(square_wave, amplitude), ops.isclose(square_wave, -amplitude)))


def test_signalsynthesizer_sawtooth_wave():
    # Test SignalSynthesizer sawtooth_wave
    sample_rate = 1000
    synthesizer = generator.SignalSynthesizer(sample_rate)
    frequency = 10.0
    duration = 0.5
    amplitude = 1.0

    sawtooth_wave = synthesizer.sawtooth_wave(frequency, duration, amplitude)
    assert sawtooth_wave is not None
    assert len(sawtooth_wave.shape) == 1
    expected_length = int(duration * sample_rate)
    assert sawtooth_wave.shape[0] == expected_length

    # Check value range
    assert ops.all(sawtooth_wave >= -amplitude)
    assert ops.all(sawtooth_wave <= amplitude)


def test_signalsynthesizer_triangle_wave():
    # Test SignalSynthesizer triangle_wave
    sample_rate = 1000
    synthesizer = generator.SignalSynthesizer(sample_rate)
    frequency = 10.0
    duration = 0.5
    amplitude = 1.0

    triangle_wave = synthesizer.triangle_wave(frequency, duration, amplitude)
    assert triangle_wave is not None
    assert len(triangle_wave.shape) == 1
    expected_length = int(duration * sample_rate)
    assert triangle_wave.shape[0] == expected_length

    # Check value range
    assert ops.all(triangle_wave >= -amplitude)
    assert ops.all(triangle_wave <= amplitude)


def test_signalsynthesizer_noise():
    # Test SignalSynthesizer noise
    sample_rate = 1000
    synthesizer = generator.SignalSynthesizer(sample_rate)
    duration = 0.5
    amplitude = 1.0

    # Test uniform noise
    uniform_noise = synthesizer.noise(duration, amplitude, distribution='uniform')
    assert uniform_noise is not None
    expected_length = int(duration * sample_rate)
    assert uniform_noise.shape[0] == expected_length
    assert ops.all(uniform_noise >= -amplitude)
    assert ops.all(uniform_noise <= amplitude)

    # Test gaussian noise
    gaussian_noise = synthesizer.noise(duration, amplitude, distribution='gaussian')
    assert gaussian_noise is not None
    assert gaussian_noise.shape[0] == expected_length
    # Checking distribution properties is complex, just check type and shape


def test_patterngenerator_initialization():
    # Test PatternGenerator initialization
    grid_size = (10, 10)
    num_phases = 8
    fade_rate = 0.1
    threshold = 0.5
    config = generator.WaveConfig(grid_size, num_phases, fade_rate, threshold) # Use WaveConfig from generator
    pattern_gen = generator.PatternGenerator(config)

    assert isinstance(pattern_gen, generator.PatternGenerator)
    assert isinstance(pattern_gen.config, generator.WaveConfig)


def test_patterngenerator_binary_pattern():
    # Test PatternGenerator binary_pattern
    grid_size = (10, 10)
    num_phases = 8
    fade_rate = 0.1
    threshold = 0.5
    config = generator.WaveConfig(grid_size, num_phases, fade_rate, threshold)
    pattern_gen = generator.PatternGenerator(config)
    density = 0.5

    binary_pattern = pattern_gen.binary_pattern(density)

    # MLX arrays are returned directly, not wrapped in EmberTensor
    assert binary_pattern is not None
    assert tensor.shape(binary_pattern) == grid_size
    assert tensor.dtype(binary_pattern) == tensor.float32 # Using float32 for the binary values
    # Check density (should be close to target for a large pattern)
    # Use a larger tolerance since we're working with a small sample
    assert ops.less(ops.abs(mlx_stats.mean(binary_pattern) - density), 0.1).item()


def test_wavegenerator_initialization():
    # Test WaveGenerator initialization
    latent_dim = 10
    hidden_dim = 20
    grid_size = (10, 10)
    num_phases = 8
    fade_rate = 0.1
    threshold = 0.5
    config = generator.WaveConfig(grid_size, num_phases, fade_rate, threshold)
    wave_gen = generator.WaveGenerator(latent_dim, hidden_dim, config)

    assert isinstance(wave_gen, generator.WaveGenerator)
    assert wave_gen.latent_dim == latent_dim
    assert wave_gen.hidden_dim == hidden_dim
    assert isinstance(wave_gen.config, generator.WaveConfig)
    assert hasattr(wave_gen, 'net') # Should have neural network layers
    assert hasattr(wave_gen, 'phase_net')


def test_wavegenerator_forward():
    # Test WaveGenerator forward pass
    latent_dim = 10
    hidden_dim = 20
    grid_size = (10, 10)
    num_phases = 8
    fade_rate = 0.1
    threshold = 0.5
    config = generator.WaveConfig(grid_size, num_phases, fade_rate, threshold)
    wave_gen = generator.WaveGenerator(latent_dim, hidden_dim, config)

    batch_size = 5
    latent_input = tensor.random_normal((batch_size, latent_dim))

    # Test forward pass returning only pattern
    pattern_output = wave_gen(latent_input, return_phases=False)
    assert pattern_output is not None
    assert tensor.shape(pattern_output) == (batch_size,) + grid_size # Shape (batch, height, width)

    # Test forward pass returning pattern and phases
    pattern_output_phases, phases_output = wave_gen(latent_input, return_phases=True)
    assert pattern_output_phases is not None
    assert tensor.shape(pattern_output_phases) == (batch_size,) + grid_size
    assert phases_output is not None
    assert tensor.shape(phases_output) == (batch_size, num_phases) # Phases should have shape (batch, num_phases)


# Add more test functions for other generator components:
# test_patterngenerator_wave_pattern(), test_patterngenerator_interference_pattern(),
# test_wavegenerator_interpolate(), test_wavegenerator_random_sample()

# Note: Testing the pattern generation methods might require checking the properties
# of the generated patterns (e.g., frequency content, interference patterns) which
# can be complex. Initial tests focus on instantiation and basic calls/shape checks.