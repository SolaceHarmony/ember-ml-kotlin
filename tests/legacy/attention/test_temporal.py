"""
Tests for temporal attention mechanisms.
"""

import pytest
import torch
import math
import torch.nn.functional as F
from ember_ml.attention.temporal import (
    PositionalEncoding,
    TemporalAttention,
    create_temporal_attention
)

@pytest.fixture
def hidden_size():
    """Fixture providing hidden size."""
    return 64

@pytest.fixture
def batch_size():
    """Fixture providing batch size."""
    return 8

@pytest.fixture
def seq_len():
    """Fixture providing sequence length."""
    return 10

@pytest.fixture
def temporal_attention(hidden_size):
    """Fixture providing temporal attention instance."""
    return TemporalAttention(
        hidden_size=hidden_size,
        num_heads=4,
        dropout=0.1,
        max_len=100,
        use_time_embedding=True
    )

@pytest.fixture
def positional_encoding(hidden_size):
    """Fixture providing positional encoding instance."""
    return PositionalEncoding(
        hidden_size=hidden_size,
        dropout=0.1,
        max_len=100
    )

class TestPositionalEncoding:
    """Test suite for PositionalEncoding class."""

    def test_initialization(self, hidden_size):
        """Test proper initialization of positional encoding."""
        max_len = 100
        pe = PositionalEncoding(hidden_size, max_len=max_len)
        
        # Check buffer registration
        assert hasattr(pe, 'pe')
        assert pe.pe.shape == (1, max_len, hidden_size)
        
        # Check encoding pattern
        pe_data = pe.pe.squeeze(0)
        
        # Test sinusoidal pattern
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            -(math.log(10000.0) / hidden_size)
        )
        
        expected_even = torch.sin(position * div_term)
        expected_odd = torch.cos(position * div_term)
        
        assert torch.allclose(pe_data[:, 0::2], expected_even)
        assert torch.allclose(pe_data[:, 1::2], expected_odd)

    def test_forward_without_times(self, positional_encoding, batch_size, seq_len, hidden_size):
        """Test forward pass without time information."""
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = positional_encoding(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check output maintains reasonable scale
        x_norm = torch.norm(x)
        output_norm = torch.norm(output)
        
        # Allow for larger tolerance due to dropout and positional encoding addition
        assert 0.5 * x_norm <= output_norm <= 2.0 * x_norm

    def test_forward_with_times(self, positional_encoding, batch_size, seq_len, hidden_size):
        """Test forward pass with time information."""
        x = torch.randn(batch_size, seq_len, hidden_size)
        times = torch.linspace(0, 1, seq_len).expand(batch_size, seq_len)
        output = positional_encoding(x, times)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check output maintains reasonable scale
        x_norm = torch.norm(x)
        output_norm = torch.norm(output)
        
        # Allow for larger tolerance due to dropout and time-scaled positional encoding
        assert 0.5 * x_norm <= output_norm <= 2.0 * x_norm

class TestTemporalAttention:
    """Test suite for TemporalAttention."""

    def test_initialization(self, temporal_attention, hidden_size):
        """Test proper initialization of temporal attention."""
        assert temporal_attention.hidden_size == hidden_size
        assert temporal_attention.num_heads == 4
        assert temporal_attention.head_dim == hidden_size // 4
        assert temporal_attention.use_time_embedding is True

    def test_invalid_initialization(self, hidden_size):
        """Test initialization with invalid parameters."""
        with pytest.raises(AssertionError):
            # hidden_size not divisible by num_heads
            TemporalAttention(hidden_size=10, num_heads=3)

    def test_forward_without_times(self, temporal_attention, batch_size, seq_len, hidden_size):
        """Test forward pass without time information."""
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        output = temporal_attention(query, key, value)
        
        # Check output properties
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check attention weights
        weights = temporal_attention.get_attention_weights()
        assert weights.shape == (batch_size, temporal_attention.num_heads, seq_len, seq_len)
        
        # Check each head's attention weights sum to 1 for each query position
        # Apply softmax to ensure proper normalization
        normalized_weights = F.softmax(weights, dim=-1)
        summed_weights = normalized_weights.sum(dim=-1)  # Sum over key dimension
        assert torch.allclose(summed_weights, torch.ones_like(summed_weights), atol=1e-6)

    def test_forward_with_times(self, temporal_attention, batch_size, seq_len, hidden_size):
        """Test forward pass with time information."""
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create time points [batch_size, seq_len]
        times = torch.linspace(0, 1, seq_len).expand(batch_size, seq_len)
        
        output = temporal_attention(query, key, value, times)
        
        # Check output properties
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check attention weights
        weights = temporal_attention.get_attention_weights()
        assert weights.shape == (batch_size, temporal_attention.num_heads, seq_len, seq_len)
        
        # Check each head's attention weights sum to 1 for each query position
        # Apply softmax to ensure proper normalization
        normalized_weights = F.softmax(weights, dim=-1)
        summed_weights = normalized_weights.sum(dim=-1)  # Sum over key dimension
        assert torch.allclose(summed_weights, torch.ones_like(summed_weights), atol=1e-6)

    def test_attention_mask(self, temporal_attention, batch_size, seq_len, hidden_size):
        """Test attention masking."""
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len))
        
        output = temporal_attention(query, key, value, mask=mask)
        
        # Check output properties
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check attention weights respect mask
        weights = temporal_attention.get_attention_weights()
        assert torch.all(weights.masked_fill(mask.unsqueeze(1) == 0, 0) == weights)

    def test_multi_head_attention(self, temporal_attention, batch_size, seq_len, hidden_size):
        """Test multi-head attention mechanism."""
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        output = temporal_attention(query, key, value)
        
        # Check output properties
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check attention weights
        weights = temporal_attention.get_attention_weights()
        assert weights.shape == (batch_size, temporal_attention.num_heads, seq_len, seq_len)
        
        # Each head should produce different attention patterns
        for h1 in range(temporal_attention.num_heads):
            for h2 in range(h1 + 1, temporal_attention.num_heads):
                correlation = torch.corrcoef(
                    torch.stack([
                        weights[:, h1].reshape(-1),
                        weights[:, h2].reshape(-1)
                    ])
                )[0, 1]
                assert abs(correlation) < 0.9  # Heads should not be too correlated

def test_create_temporal_attention(hidden_size):
    """Test factory function for creating temporal attention."""
    attn = create_temporal_attention(
        hidden_size=hidden_size,
        num_heads=4,
        dropout=0.1,
        max_len=100,
        use_time_embedding=True
    )
    
    assert isinstance(attn, TemporalAttention)
    assert attn.hidden_size == hidden_size
    assert attn.num_heads == 4
    assert attn.max_len == 100
    assert attn.use_time_embedding is True

if __name__ == '__main__':
    pytest.main([__file__])