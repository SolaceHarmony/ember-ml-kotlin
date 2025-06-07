"""
Tests for wave pattern and signal generation components.
"""

import pytest
import torch
import math
from ember_ml.wave.generator import (
    WaveGenerator,
    PatternGenerator,
    SignalSynthesizer
)
from ember_ml.wave.binary_wave import WaveConfig

@pytest.fixture
def config():
    """Fixture providing wave configuration."""
    return WaveConfig(
        grid_size=4,
        num_phases=8,
        fade_rate=0.1,
        threshold=0.5
    )

@pytest.fixture
def sampling_rate():
    """Fixture providing sampling rate."""
    return 100.0

@pytest.fixture
def duration():
    """Fixture providing signal duration."""
    return 1.0

@pytest.fixture
def synthesizer(sampling_rate):
    """Fixture providing signal synthesizer."""
    return SignalSynthesizer(sampling_rate)

@pytest.fixture
def pattern_generator(config):
    """Fixture providing pattern generator."""
    return PatternGenerator(config)

@pytest.fixture
def wave_generator(config):
    """Fixture providing wave generator."""
    return WaveGenerator(
        latent_dim=8,
        hidden_dim=16,
        config=config
    )

class TestSignalSynthesizer:
    """Test suite for SignalSynthesizer."""

    def test_sine_wave(self, synthesizer, duration):
        """Test sine wave generation."""
        frequency = 10.0
        amplitude = 2.0
        phase = math.pi/4
        
        wave = synthesizer.sine_wave(
            frequency,
            duration,
            amplitude,
            phase
        )
        
        # Check signal properties
        n_samples = int(duration * synthesizer.sampling_rate)
        assert wave.shape == (n_samples,)
        assert torch.all(wave >= -amplitude)
        assert torch.all(wave <= amplitude)

    def test_square_wave(self, synthesizer, duration):
        """Test square wave generation."""
        frequency = 10.0
        amplitude = 2.0
        duty_cycle = 0.3
        
        wave = synthesizer.square_wave(
            frequency,
            duration,
            amplitude,
            duty_cycle
        )
        
        # Check signal properties
        assert torch.all(torch.abs(wave) <= amplitude)
        
        # Check duty cycle
        high_samples = torch.sum(wave > 0)
        total_samples = wave.size(0)
        actual_duty = high_samples / total_samples
        assert abs(actual_duty - duty_cycle) < 0.1

    def test_sawtooth_wave(self, synthesizer, duration):
        """Test sawtooth wave generation."""
        frequency = 10.0
        amplitude = 2.0
        
        wave = synthesizer.sawtooth_wave(
            frequency,
            duration,
            amplitude
        )
        
        # Check signal properties
        assert torch.all(wave >= -amplitude)
        assert torch.all(wave <= amplitude)

    def test_triangle_wave(self, synthesizer, duration):
        """Test triangle wave generation."""
        frequency = 10.0
        amplitude = 2.0
        
        wave = synthesizer.triangle_wave(
            frequency,
            duration,
            amplitude
        )
        
        # Check signal properties
        assert torch.all(wave >= -amplitude)
        assert torch.all(wave <= amplitude)

    def test_noise(self, synthesizer, duration):
        """Test noise generation."""
        amplitude = 1.0
        
        # Test uniform noise
        uniform_noise = synthesizer.noise(
            duration,
            amplitude,
            'uniform'
        )
        assert torch.all(uniform_noise >= -amplitude)
        assert torch.all(uniform_noise <= amplitude)
        
        # Test gaussian noise
        gaussian_noise = synthesizer.noise(
            duration,
            amplitude,
            'gaussian'
        )
        assert gaussian_noise.shape == uniform_noise.shape

class TestPatternGenerator:
    """Test suite for PatternGenerator."""

    def test_binary_pattern(self, pattern_generator):
        """Test binary pattern generation."""
        density = 0.3
        pattern = pattern_generator.binary_pattern(density)
        
        # Check pattern properties
        assert pattern.shape == (
            pattern_generator.config.grid_size,
            pattern_generator.config.grid_size
        )
        assert torch.all((pattern == 0) | (pattern == 1))
        
        # Check density
        actual_density = torch.mean(pattern.float())
        assert abs(actual_density - density) < 0.2

    def test_wave_pattern(self, pattern_generator, duration):
        """Test wave pattern generation."""
        frequency = 2.0
        pattern = pattern_generator.wave_pattern(frequency, duration)
        
        # Check pattern properties
        assert pattern.shape == (
            pattern_generator.config.grid_size,
            pattern_generator.config.grid_size
        )
        assert torch.all(pattern >= 0)
        assert torch.all(pattern <= 1)

    def test_interference_pattern(self, pattern_generator, duration):
        """Test interference pattern generation."""
        frequencies = [2.0, 4.0]
        amplitudes = [1.0, 0.5]
        
        pattern = pattern_generator.interference_pattern(
            frequencies,
            amplitudes,
            duration
        )
        
        # Check pattern properties
        assert pattern.shape == (
            pattern_generator.config.grid_size,
            pattern_generator.config.grid_size
        )
        assert torch.all(pattern >= 0)
        assert torch.all(pattern <= 1)

class TestWaveGenerator:
    """Test suite for WaveGenerator."""

    def test_initialization(self, wave_generator):
        """Test generator initialization."""
        assert isinstance(wave_generator, torch.nn.Module)
        assert hasattr(wave_generator, 'net')
        assert hasattr(wave_generator, 'phase_net')

    def test_forward_pass(self, wave_generator):
        """Test forward generation."""
        batch_size = 4
        z = torch.randn(batch_size, wave_generator.latent_dim)
        
        # Test without phases
        pattern = wave_generator(z)
        assert pattern.shape == (
            batch_size,
            wave_generator.config.grid_size,
            wave_generator.config.grid_size
        )
        assert torch.all(pattern >= 0)
        assert torch.all(pattern <= 1)
        
        # Test with phases
        pattern, phases = wave_generator(z, return_phases=True)
        assert phases.shape == (batch_size, wave_generator.config.num_phases)

    def test_interpolation(self, wave_generator):
        """Test pattern interpolation."""
        z1 = torch.randn(wave_generator.latent_dim)
        z2 = torch.randn(wave_generator.latent_dim)
        steps = 5
        
        patterns = wave_generator.interpolate(z1, z2, steps)
        
        # Check interpolation properties
        assert patterns.shape == (
            steps,
            wave_generator.config.grid_size,
            wave_generator.config.grid_size
        )
        assert torch.all(patterns >= 0)
        assert torch.all(patterns <= 1)

    def test_random_sampling(self, wave_generator):
        """Test random pattern sampling."""
        num_samples = 3
        patterns = wave_generator.random_sample(num_samples)
        
        # Check sample properties
        assert patterns.shape == (
            num_samples,
            wave_generator.config.grid_size,
            wave_generator.config.grid_size
        )
        assert torch.all(patterns >= 0)
        assert torch.all(patterns <= 1)
        
        # Check reproducibility with seed
        patterns1 = wave_generator.random_sample(num_samples, seed=42)
        patterns2 = wave_generator.random_sample(num_samples, seed=42)
        assert torch.allclose(patterns1, patterns2)