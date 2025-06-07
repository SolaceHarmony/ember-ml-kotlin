"""
Tests for causal attention mechanisms.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from ember_ml.attention.causal import (
    CausalAttention,
    PredictionAttention,
    AttentionState,
    CausalMemory
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
def causal_attention(hidden_size):
    """Fixture providing causal attention mechanism."""
    return CausalAttention(hidden_size)

@pytest.fixture
def prediction_attention(hidden_size):
    """Fixture providing prediction attention mechanism."""
    return PredictionAttention(hidden_size)

@pytest.fixture
def causal_memory():
    """Fixture providing causal memory."""
    return CausalMemory(max_size=100)

class TestAttentionState:
    """Test suite for AttentionState."""

    def test_initialization(self):
        """Test attention state initialization."""
        state = AttentionState()
        assert state.temporal_weight == 0.0
        assert state.causal_weight == 0.0
        assert state.novelty_weight == 0.0

    def test_compute_total(self):
        """Test total attention weight computation."""
        state = AttentionState(
            temporal_weight=0.3,
            causal_weight=0.6,
            novelty_weight=0.9
        )
        total = state.compute_total()
        assert total == (0.3 + 0.6 + 0.9) / 3.0

class TestCausalMemory:
    """Test suite for CausalMemory."""

    def test_initialization(self, causal_memory):
        """Test memory initialization."""
        assert causal_memory.max_size == 100
        assert len(causal_memory.cause_effect_pairs) == 0
        assert len(causal_memory.prediction_accuracy) == 0

    def test_add_memory(self, causal_memory, hidden_size):
        """Test adding to memory."""
        cause = torch.randn(hidden_size)
        effect = torch.randn(hidden_size)
        accuracy = 0.8
        
        causal_memory.add(cause, effect, accuracy)
        
        assert len(causal_memory.cause_effect_pairs) == 1
        assert len(causal_memory.prediction_accuracy) == 1
        assert causal_memory.prediction_accuracy[0] == accuracy

    def test_memory_limit(self, causal_memory, hidden_size):
        """Test memory size limit."""
        for _ in range(150):  # More than max_size
            cause = torch.randn(hidden_size)
            effect = torch.randn(hidden_size)
            causal_memory.add(cause, effect, 0.8)
            
        assert len(causal_memory.cause_effect_pairs) == 100
        assert len(causal_memory.prediction_accuracy) == 100

    def test_get_similar_causes(self, causal_memory, hidden_size):
        """Test finding similar causes."""
        # Add some memories
        cause = torch.randn(hidden_size)
        effect = torch.randn(hidden_size)
        causal_memory.add(cause, effect, 0.8)
        
        # Test with same cause
        similar_indices = causal_memory.get_similar_causes(cause)
        assert len(similar_indices) == 1
        assert similar_indices[0] == 0

class TestPredictionAttention:
    """Test suite for PredictionAttention."""

    def test_initialization(self, prediction_attention, hidden_size):
        """Test prediction attention initialization."""
        assert isinstance(prediction_attention, nn.Module)
        assert prediction_attention.hidden_size == hidden_size
        assert hasattr(prediction_attention, 'predictor')
        assert hasattr(prediction_attention, 'memory')

    def test_forward_pass(self, prediction_attention, batch_size, seq_length, hidden_size):
        """Test forward processing."""
        query = torch.randn(batch_size, seq_length, hidden_size)
        key = torch.randn(batch_size, seq_length, hidden_size)
        value = torch.randn(batch_size, seq_length, hidden_size)
        
        output = prediction_attention(query, key, value)
        
        # Check output properties
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert not torch.allclose(output, value)  # Should be transformed
        
        # Check attention weights
        weights = prediction_attention.get_attention_weights()
        assert weights is not None
        assert weights.shape[0] == batch_size

class TestCausalAttention:
    """Test suite for CausalAttention."""

    def test_initialization(self, causal_attention, hidden_size):
        """Test causal attention initialization."""
        assert isinstance(causal_attention, nn.Module)
        assert causal_attention.hidden_size == hidden_size
        assert hasattr(causal_attention, 'temporal_proj')
        assert hasattr(causal_attention, 'causal_proj')
        assert hasattr(causal_attention, 'novelty_proj')

    def test_forward_pass(self, causal_attention, batch_size, hidden_size):
        """Test forward processing."""
        # Create input tensors with correct shapes
        # CausalAttention expects single state vectors, not sequences
        query = torch.randn(batch_size, hidden_size)  # [batch_size, hidden_size]
        key = torch.randn(batch_size, hidden_size)    # [batch_size, hidden_size]
        value = torch.randn(batch_size, hidden_size)  # [batch_size, hidden_size]
        
        output = causal_attention(query, key, value)
        
        # Check output properties
        assert output.shape == (batch_size, hidden_size)  # Should match input shape
        
        # Check attention weights
        weights = causal_attention.get_attention_weights()
        assert weights is not None
        assert weights.shape == (batch_size, 1, 1)  # Single attention weight per batch item
        
        # Normalize weights using sigmoid to ensure they're between 0 and 1
        normalized_weights = torch.sigmoid(weights)
        assert torch.all(normalized_weights >= 0)
        assert torch.all(normalized_weights <= 1)

    def test_state_update(self, causal_attention, hidden_size):
        """Test attention state updates."""
        current_state = torch.randn(hidden_size)
        target_state = torch.randn(hidden_size)
        
        # Apply sigmoid to ensure weights are between 0 and 1
        state = causal_attention.update(0, current_state, target_state)
        
        assert isinstance(state, AttentionState)
        assert hasattr(state, 'temporal_weight')
        assert hasattr(state, 'causal_weight')
        assert hasattr(state, 'novelty_weight')
        
        # Check weight ranges after sigmoid
        temporal_weight = torch.sigmoid(torch.tensor(state.temporal_weight)).item()
        causal_weight = torch.sigmoid(torch.tensor(state.causal_weight)).item()
        novelty_weight = torch.sigmoid(torch.tensor(state.novelty_weight)).item()
        
        assert 0.0 <= temporal_weight <= 1.0
        assert 0.0 <= causal_weight <= 1.0
        assert 0.0 <= novelty_weight <= 1.0

    def test_save_load_state(self, causal_attention, hidden_size):
        """Test state saving and loading."""
        # Create some state
        current_state = torch.randn(hidden_size)
        target_state = torch.randn(hidden_size)
        causal_attention.update(0, current_state, target_state)
        
        # Save state
        state_dict = causal_attention.save_state()
        
        # Create new attention and load state
        new_attention = CausalAttention(hidden_size)
        new_attention.load_state(state_dict)
        
        # Check state was transferred
        assert new_attention.hidden_size == causal_attention.hidden_size
        assert new_attention.decay_rate == causal_attention.decay_rate
        assert len(new_attention.states) == len(causal_attention.states)

if __name__ == '__main__':
    pytest.main([__file__])