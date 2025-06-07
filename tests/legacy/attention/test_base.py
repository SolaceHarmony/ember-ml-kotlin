"""
Tests for base attention mechanisms.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from ember_ml.attention.base import (
    BaseAttention,
    AttentionLayer,
    MultiHeadAttention,
    AttentionMask,
    AttentionScore
)

@pytest.fixture
def batch_size():
    """Fixture providing batch size."""
    return 4

@pytest.fixture
def seq_length():
    """Fixture providing sequence length."""
    return 10

@pytest.fixture
def hidden_size():
    """Fixture providing hidden dimension."""
    return 32

@pytest.fixture
def num_heads():
    """Fixture providing number of attention heads."""
    return 4

@pytest.fixture
def attention_layer(hidden_size):
    """Fixture providing basic attention layer."""
    return AttentionLayer(
        query_dim=hidden_size,
        key_dim=hidden_size,
        value_dim=hidden_size,
        hidden_dim=hidden_size
    )

@pytest.fixture
def multi_head_attention(hidden_size, num_heads):
    """Fixture providing multi-head attention."""
    return MultiHeadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads
    )

class TestAttentionMask:
    """Test suite for AttentionMask."""

    def test_padding_mask(self, batch_size, seq_length):
        """Test padding mask creation."""
        lengths = torch.randint(1, seq_length + 1, (batch_size,))
        mask = AttentionMask.create_padding_mask(lengths, seq_length)
        
        assert mask.shape == (batch_size, seq_length)
        assert torch.all(mask >= 0)
        assert torch.all(mask <= 1)
        
        # Check mask matches sequence lengths
        for i, length in enumerate(lengths):
            assert torch.all(mask[i, :length] == 1)
            assert torch.all(mask[i, length:] == 0)

    def test_causal_mask(self, seq_length):
        """Test causal mask creation."""
        mask = AttentionMask.create_causal_mask(seq_length)
        
        assert mask.shape == (seq_length, seq_length)
        # Lower triangle should be 1s
        lower_tri = torch.tril(torch.ones(seq_length, seq_length))
        assert torch.all(mask == lower_tri)
        # Upper triangle should be 0s
        assert torch.all(torch.triu(mask, diagonal=1) == 0)

    def test_window_mask(self, seq_length):
        """Test window mask creation."""
        window_size = 2
        mask = AttentionMask.create_window_mask(seq_length, window_size)
        
        assert mask.shape == (seq_length, seq_length)
        assert torch.all(mask >= 0)
        assert torch.all(mask <= 1)
        
        # Check window properties
        for i in range(seq_length):
            start = max(0, i - window_size)
            end = min(seq_length, i + window_size + 1)
            assert torch.all(mask[i, start:end] == 1)
            if start > 0:
                assert torch.all(mask[i, :start] == 0)
            if end < seq_length:
                assert torch.all(mask[i, end:] == 0)

class TestAttentionScore:
    """Test suite for AttentionScore."""

    def test_dot_product(self, batch_size, seq_length, hidden_size):
        """Test dot product attention computation."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        
        scores = AttentionScore.dot_product(query, key)
        
        assert scores.shape == (batch_size, seq_length, seq_length)

    def test_scaled_dot_product(self, batch_size, seq_length, hidden_size):
        """Test scaled dot product attention computation."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        scale = float(hidden_size) ** 0.5
        
        scores = AttentionScore.scaled_dot_product(query, key, scale)
        
        assert scores.shape == (batch_size, seq_length, seq_length)
        
        # Compare with manual computation
        expected = torch.matmul(query, key.transpose(-2, -1)) / scale
        assert torch.allclose(scores, expected)

    def test_additive(self, batch_size, seq_length, hidden_size):
        """Test additive attention computation."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        weight = torch.randn(hidden_size * 2, hidden_size)
        bias = torch.randn(hidden_size)
        
        scores = AttentionScore.additive(query, key, weight, bias)
        
        assert scores.shape == (batch_size, seq_length, seq_length, hidden_size)
        assert torch.all(scores >= -1)
        assert torch.all(scores <= 1)  # Due to tanh

class TestAttentionLayer:
    """Test suite for AttentionLayer."""

    def test_initialization(self, attention_layer, hidden_size):
        """Test attention layer initialization."""
        assert isinstance(attention_layer, nn.Module)
        assert isinstance(attention_layer.query, nn.Linear)
        assert isinstance(attention_layer.key, nn.Linear)
        assert isinstance(attention_layer.value, nn.Linear)
        assert attention_layer.scale == hidden_size ** 0.5

    def test_forward_pass(self, attention_layer, batch_size, seq_length, hidden_size):
        """Test forward processing."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        value = torch.randn(batch_size, seq_length, hidden_size)
        
        output = attention_layer(query, key, value)
        
        # Check output properties
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert not torch.allclose(output, value)  # Should be transformed

    def test_mask_application(self, attention_layer, batch_size, seq_length, hidden_size):
        """Test attention mask application."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        value = torch.randn(batch_size, seq_length, hidden_size)
        
        # Create causal mask
        mask = AttentionMask.create_causal_mask(seq_length)
        mask = mask.expand(batch_size, seq_length, seq_length)
        
        output = attention_layer(query, key, value, mask=mask)
        
        # Check output properties
        assert output.shape == (batch_size, seq_length, hidden_size)
        
        # Check attention weights respect mask
        weights = attention_layer.get_attention_weights()
        assert torch.all(weights.masked_fill(mask == 0, 0) == weights)

class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""

    def test_initialization(self, multi_head_attention, hidden_size, num_heads):
        """Test multi-head attention initialization."""
        assert isinstance(multi_head_attention, nn.Module)
        assert multi_head_attention.embed_dim == hidden_size
        assert multi_head_attention.num_heads == num_heads
        assert multi_head_attention.head_dim == hidden_size // num_heads

    def test_forward_pass(self, multi_head_attention, batch_size, seq_length, hidden_size, num_heads):
        """Test forward processing with multiple heads."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        value = torch.randn(batch_size, seq_length, hidden_size)
        
        output = multi_head_attention(query, key, value)
        
        # Check output properties
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert not torch.allclose(output, value)  # Should be transformed
        
        # Check attention weights
        weights = multi_head_attention.get_attention_weights()
        assert weights.shape == (batch_size, num_heads, seq_length, seq_length)
        
        # Check each head's attention weights sum to 1 for each query position
        # Apply softmax to ensure proper normalization
        normalized_weights = F.softmax(weights, dim=-1)
        summed_weights = normalized_weights.sum(dim=-1)  # Sum over key dimension
        assert torch.allclose(summed_weights, torch.ones_like(summed_weights), atol=1e-6)

    def test_mask_application(self, multi_head_attention, batch_size, seq_length, hidden_size):
        """Test attention mask application."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        value = torch.randn(batch_size, seq_length, hidden_size)
        
        # Create causal mask
        mask = AttentionMask.create_causal_mask(seq_length)
        mask = mask.expand(batch_size, seq_length, seq_length)
        
        output = multi_head_attention(query, key, value, mask=mask)
        
        # Check output properties
        assert output.shape == (batch_size, seq_length, hidden_size)
        
        # Check attention weights respect mask
        weights = multi_head_attention.get_attention_weights()
        assert torch.all(weights.masked_fill(mask.unsqueeze(1) == 0, 0) == weights)

if __name__ == '__main__':
    pytest.main([__file__])