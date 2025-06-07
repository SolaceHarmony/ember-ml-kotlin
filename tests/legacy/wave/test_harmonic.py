"""
Tests for harmonic wave processing components.
"""

import pytest
from ember_ml.wave.harmonic import (
    HarmonicProcessor,
    FrequencyAnalyzer,
    WaveSynthesizer
)
from ember_ml.nn.tensor import tensor

@pytest.fixture
def sampling_rate():
    """Fixture providing sampling rate."""
    return 100.0

@pytest.fixture
def duration():
    """Fixture providing signal duration."""
    return 1.0

@pytest.fixture
def frequency():
    """Fixture providing test frequency."""
    return 10.0

@pytest.fixture
def analyzer(sampling_rate):
    """Fixture providing frequency analyzer."""
    return FrequencyAnalyzer(sampling_rate)

@pytest.fixture
def synthesizer(sampling_rate):
    """Fixture providing wave synthesizer."""
    return WaveSynthesizer(sampling_rate)

@pytest.fixture
def processor(sampling_rate):
    """Fixture providing harmonic processor."""
    return HarmonicProcessor(sampling_rate)

class TestFrequencyAnalyzer:
    """Test suite for FrequencyAnalyzer."""

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.sampling_rate > 0
        assert hasattr(analyzer, 'window_size')
        assert hasattr(analyzer, 'overlap')

    def test_frequency_spectrum(self, analyzer, synthesizer, frequency, duration):
        """Test frequency spectrum computation."""
        # Generate test signal
        signal = synthesizer.sine_wave(frequency, duration)
        spectrum = analyzer.compute_spectrum(signal)
        import mlx.core
        # Check spectrum properties
        assert isinstance(tensor(spectrum),mlx.core.array)
        assert len(spectrum.shape) == 1
        assert tensor.all(spectrum >= 0)

    def test_peak_frequencies(self, analyzer, synthesizer, frequency, duration):
        """Test peak frequency detection."""
        # Generate test signal
        signal = synthesizer.sine_wave(frequency, duration)
        peaks = analyzer.find_peaks(signal)
        
        # Check peak properties
        assert len(peaks) > 0
        assert frequency in [p['frequency'] for p in peaks]

    def test_harmonic_ratio(self, analyzer, synthesizer, frequency, duration):
        """Test harmonic ratio computation."""
        # Generate signal with harmonics
        fundamental = synthesizer.sine_wave(frequency, duration)
        harmonic = synthesizer.sine_wave(2 * frequency, duration)
        signal = fundamental + 0.5 * harmonic
        
        ratio = analyzer.harmonic_ratio(signal)
        
        # Check ratio is reasonable
        # The ratio should be less than 1 since harmonic amplitude is 0.5
        assert ratio < 1.0

class TestWaveSynthesizer:
    """Test suite for WaveSynthesizer."""

    def test_initialization(self, synthesizer):
        """Test synthesizer initialization."""
        assert synthesizer.sampling_rate > 0

    def test_sine_wave(self, synthesizer, frequency, duration):
        """Test sine wave generation."""
        wave = synthesizer.sine_wave(frequency, duration)
        
        # Check wave properties
        n_samples = int(duration * synthesizer.sampling_rate)
        assert wave.shape == (n_samples,)
        assert tensor.all(wave >= -1.0)
        assert tensor.all(wave <= 1.0)

    def test_harmonic_synthesis(self, synthesizer, frequency, duration):
        """Test harmonic wave synthesis."""
        frequencies = [frequency, 2 * frequency, 3 * frequency]
        amplitudes = [1.0, 0.5, 0.25]
        
        wave = synthesizer.harmonic_wave(frequencies, amplitudes, duration)
        
        # Check wave properties
        assert tensor.all(wave >= -1.0)
        assert tensor.all(wave <= 1.0)
        assert tensor.all(wave <= 1.0)

    def test_envelope_application(self, synthesizer, frequency, duration):
        """Test envelope application."""
        wave = synthesizer.sine_wave(frequency, duration)
        envelope = tensor.linspace(0, 1, len(wave))
        
        modulated = synthesizer.apply_envelope(wave, envelope)
        import ember_ml.ops as ops
        # Check modulation
        assert tensor.all(ops.less_equal(modulated, wave))
        assert tensor.all(ops.greater_equal(modulated, ops.multiply((-1, tensor.abs(wave)))))

class TestHarmonicProcessor:
    """Test suite for HarmonicProcessor."""

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.sampling_rate > 0
        assert hasattr(processor, 'analyzer')
        assert hasattr(processor, 'synthesizer')

    def test_decomposition(self, processor, frequency, duration):
        """Test harmonic decomposition."""
        # Generate test signal
        signal = processor.synthesizer.sine_wave(frequency, duration)
        components = processor.decompose(signal)
        
        # Check components
        assert 'frequencies' in components
        assert 'amplitudes' in components
        assert len(components['frequencies']) > 0
        assert len(components['amplitudes']) > 0

    def test_reconstruction(self, processor, frequency, duration):
        """Test signal reconstruction."""
        # Generate and decompose signal
        original = processor.synthesizer.sine_wave(frequency, duration)
        components = processor.decompose(original)
        
        # Reconstruct
        reconstructed = processor.reconstruct(
            components['frequencies'],
            components['amplitudes'],
            duration
        )
        
        # Check reconstruction quality
        assert tensor.allclose(original, reconstructed, atol=1e-1)

    def test_harmonic_filtering(self, processor, frequency, duration):
        """Test harmonic filtering."""
        # Generate signal with harmonics
        fundamental = processor.synthesizer.sine_wave(frequency, duration)
        harmonic = processor.synthesizer.sine_wave(2 * frequency, duration)
        signal = fundamental + harmonic
        
        filtered = processor.filter_harmonics(signal, [frequency])
        
        # Check filtering
        assert not tensor.allclose(signal, filtered)
        assert tensor.allclose(filtered, fundamental, atol=1e-1)