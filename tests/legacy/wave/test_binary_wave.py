"""
Tests for binary wave neural processing components.
"""

import pytest
import torch
import numpy as np
from ember_ml.wave.binary_wave import (
    WaveConfig,
    BinaryWaveEncoder,
    BinaryWaveProcessor,
    BinaryWaveNetwork
)

@pytest.fixture
def wave_config():
    """Fixture providing default wave configuration."""
    return WaveConfig(
        grid_size=4,
        num_phases=8,
        fade_rate=0.1,
        threshold=0.5
    )

@pytest.fixture
def encoder(wave_config):
    """Fixture providing binary wave encoder."""
    return BinaryWaveEncoder(wave_config)

@pytest.fixture
def processor(wave_config):
    """Fixture providing binary wave processor."""
    return BinaryWaveProcessor(wave_config)

@pytest.fixture
def network(wave_config):
    """Fixture providing binary wave network."""
    return BinaryWaveNetwork(
        input_size=10,
        hidden_size=16,
        output_size=2,
        config=wave_config
    )

class TestWaveConfig:
    """Test suite for WaveConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WaveConfig()
        assert config.grid_size == 4
        assert config.num_phases == 8
        assert config.fade_rate == 0.1
        assert config.threshold == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WaveConfig(
            grid_size=8,
            num_phases=16,
            fade_rate=0.2,
            threshold=0.7
        )
        assert config.grid_size == 8
        assert config.num_phases == 16
        assert config.fade_rate == 0.2
        assert config.threshold == 0.7

class TestBinaryWaveEncoder:
    """Test suite for BinaryWaveEncoder."""

    def test_encode_char(self, encoder):
        """Test character encoding."""
        char = 'A'
        wave = encoder.encode_char(char)
        
        # Check shape
        assert wave.shape == (8, 4, 4, 1)  # (num_phases, grid_size, grid_size, 1)
        assert torch.is_tensor(wave)
        
        # Check values are in valid range
        assert torch.all(wave >= 0.0)
        assert torch.all(wave <= 1.0)

    def test_encode_sequence(self, encoder):
        """Test sequence encoding."""
        sequence = "ABC"
        waves = encoder.encode_sequence(sequence)
        
        # Check shape
        assert waves.shape == (3, 8, 4, 4, 1)  # (seq_len, num_phases, grid_size, grid_size, 1)
        
        # Verify each character encoding
        for i, char in enumerate(sequence):
            char_wave = encoder.encode_char(char)
            assert torch.allclose(waves[i], char_wave)

    def test_fade_effect(self, encoder):
        """Test fade effect in phase shifts."""
        wave = encoder.encode_char('X')
        
        # Later phases should have lower amplitudes
        max_vals = [wave[i].max().item() for i in range(encoder.config.num_phases)]
        assert all(max_vals[i] >= max_vals[i+1] for i in range(len(max_vals)-1))

class TestBinaryWaveProcessor:
    """Test suite for BinaryWaveProcessor."""

    @pytest.fixture
    def sample_waves(self, encoder):
        """Fixture providing sample wave patterns."""
        wave1 = encoder.encode_char('A')
        wave2 = encoder.encode_char('B')
        return wave1, wave2

    def test_wave_interference_xor(self, processor, sample_waves):
        """Test XOR wave interference."""
        wave1, wave2 = sample_waves
        result = processor.wave_interference(wave1, wave2, mode='XOR')
        
        # Check shape preservation
        assert result.shape == wave1.shape
        
        # Convert to binary tensors
        binary1 = (wave1 > processor.config.threshold).float()
        binary2 = (wave2 > processor.config.threshold).float()
        
        # Compute expected XOR
        expected = torch.logical_xor(
            binary1.bool(),
            binary2.bool()
        ).float()
        
        assert torch.allclose(result, expected)

    def test_wave_interference_and(self, processor, sample_waves):
        """Test AND wave interference."""
        wave1, wave2 = sample_waves
        result = processor.wave_interference(wave1, wave2, mode='AND')
        
        # Convert to binary tensors
        binary1 = (wave1 > processor.config.threshold).float()
        binary2 = (wave2 > processor.config.threshold).float()
        
        # Compute expected AND
        expected = torch.logical_and(
            binary1.bool(),
            binary2.bool()
        ).float()
        
        assert torch.allclose(result, expected)

    def test_phase_similarity(self, processor, sample_waves):
        """Test phase similarity calculation."""
        wave1, wave2 = sample_waves
        result = processor.phase_similarity(wave1, wave2)
        
        assert 'similarity' in result
        assert 'shift' in result
        assert 0.0 <= result['similarity'] <= 1.0
        assert 0 <= result['shift'] < processor.config.num_phases

    def test_extract_features(self, processor, sample_waves):
        """Test feature extraction."""
        wave, _ = sample_waves
        
        # Convert to binary tensor
        binary = (wave > processor.config.threshold).float()
        features = processor.extract_features(binary)
        
        # Check feature presence
        assert 'density' in features
        assert 'transitions' in features
        assert 'symmetry' in features
        
        # Check value ranges
        assert 0.0 <= features['density'] <= 1.0
        assert features['transitions'] >= 0.0
        assert 0.0 <= features['symmetry'] <= 1.0

class TestBinaryWaveNetwork:
    """Test suite for BinaryWaveNetwork."""

    def test_initialization(self, network):
        """Test network initialization."""
        assert isinstance(network.encoder, BinaryWaveEncoder)
        assert isinstance(network.processor, BinaryWaveProcessor)
        assert isinstance(network.input_proj, torch.nn.Linear)
        assert isinstance(network.output_proj, torch.nn.Linear)

    def test_forward_pass(self, network):
        """Test forward pass through network."""
        batch_size = 3
        x = torch.randn(batch_size, 10)  # (batch_size, input_size)
        
        # Test without memory
        output, memory = network(x)
        
        assert output.shape == (batch_size, 2)  # (batch_size, output_size)
        assert memory.shape == (batch_size, 4, 4)  # (batch_size, grid_size, grid_size)
        
        # Test with memory
        output2, memory2 = network(x, memory)
        
        assert output2.shape == (batch_size, 2)
        assert memory2.shape == (batch_size, 4, 4)
        
        # Memory should be different after update
        assert not torch.allclose(memory, memory2)

    def test_memory_gating(self, network):
        """Test memory gating mechanism."""
        x = torch.randn(1, 10)
        
        # Initial pass
        _, memory1 = network(x)
        
        # Second pass with same input
        _, memory2 = network(x, memory1)
        
        # Memory should be updated
        assert not torch.allclose(memory1, memory2)
        
        # Convert to binary tensors for comparison
        binary_memory = (memory2 > network.config.threshold).float()
        assert torch.all((binary_memory == 0.0) | (binary_memory == 1.0))

if __name__ == '__main__':
    pytest.main([__file__])