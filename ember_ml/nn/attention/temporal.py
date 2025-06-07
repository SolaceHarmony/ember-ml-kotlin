"""
Temporal attention mechanisms for sequence processing and time-based patterns.
"""

from typing import Optional
import math
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor, zeros, arange, maximum, shape, concatenate, expand_dims, reshape, transpose
from ember_ml.nn.modules import Module
from ember_ml.nn.container import Dropout, Linear, Sequential
from ember_ml.nn.modules.activations import Sigmoid, softmax # Import Sigmoid and softmax
from ember_ml.nn.attention.base import BaseAttention

class PositionalEncoding(Module):
    """Positional encoding for temporal information."""
    
    def __init__(self,
                 hidden_size: int,
                 dropout: float = 0.1,
                 max_len: int = 1000):
        """
        Initialize positional encoding.

        Args:
            hidden_size: Hidden state dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = Dropout(rate=dropout)
        
        # Create positional encoding matrix
        pe = zeros((max_len, hidden_size))
        position = expand_dims(arange(0, max_len), 1)
        position = tensor.cast(position, tensor.float32)
        
        log_term = ops.divide(
            tensor.convert_to_tensor(-math.log(10000.0), tensor.float32),
            tensor.convert_to_tensor(hidden_size, tensor.float32)
        )
        div_term = ops.exp(
            ops.multiply(
                tensor.cast(arange(0, hidden_size, 2), tensor.float32),
                log_term
            )
        )
        
        # Compute sinusoidal pattern
        # Use tensor.slice_update instead of direct indexing
        sin_values = ops.sin(ops.multiply(position, div_term))
        cos_values = ops.cos(ops.multiply(position, div_term))
        
        # Update even indices with sin values
        for i in range(0, hidden_size, 2):
            if i < hidden_size:
                pe = tensor.slice_update(pe, (slice(None), i), sin_values)
        
        # Update odd indices with cos values
        for i in range(1, hidden_size, 2):
            if i < hidden_size:
                pe = tensor.slice_update(pe, (slice(None), i), cos_values)
        
        # Register buffer
        self.register_buffer('pe', expand_dims(pe, 0))
        
    def forward(self,
                x: EmberTensor,
                times: Optional[EmberTensor] = None) -> EmberTensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            times: Optional time points [batch, seq_len]

        Returns:
            Encoded tensor [batch, seq_len, hidden_size]
        """
        seq_len = shape(x)[1]  # Get sequence length from shape
        
        if times is not None:
            # Scale positional encoding by time differences
            time_scale = ops.divide(times, maximum(times))  # Normalize to [0, 1]
            time_scale = expand_dims(time_scale, -1)
            
            # Slice pe to match sequence length
            pe_sliced = tensor.slice_tensor(self.pe, [0, 0, 0], [1, seq_len, -1])
            pe = ops.multiply(pe_sliced, time_scale)
        else:
            # Slice pe to match sequence length
            pe = tensor.slice_tensor(self.pe, [0, 0, 0], [1, seq_len, -1])
            
        # Expand positional encoding to match batch size
        batch_size = shape(x)[0]
        pe = tensor.tile(pe, [batch_size, 1, 1])
        
        return self.dropout(ops.add(x, pe))

class TemporalAttention(BaseAttention):
    """
    Attention mechanism specialized for temporal sequence processing
    with time-aware attention computation.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_heads: int = 1,
                 dropout: float = 0.1,
                 max_len: int = 1000,
                 use_time_embedding: bool = True):
        """
        Initialize temporal attention.

        Args:
            hidden_size: Hidden state dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_len: Maximum sequence length
            use_time_embedding: Whether to use temporal embeddings
        """
        super().__init__()
        remainder = ops.mod(hidden_size, num_heads)
        is_divisible = ops.equal(remainder, 0)
        assert is_divisible, "hidden_size must be divisible by num_heads"
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = ops.floor_divide(hidden_size, num_heads)
        self.max_len = max_len
        self.use_time_embedding = use_time_embedding
        
        # Projections for Q, K, V
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.out_proj = Linear(hidden_size, hidden_size)
        
        # Temporal components
        if use_time_embedding:
            self.time_embedding = PositionalEncoding(
                hidden_size,
                dropout=dropout,
                max_len=max_len
            )
        
        # Time-aware attention components
        combined_size = ops.add(hidden_size, 1)
        self.time_gate = Sequential([
            Linear(combined_size, hidden_size),
            Sigmoid()
        ])
        
        self.dropout = Dropout(dropout)
        self._attention_weights = None
        
    def forward(
            self,
            query: EmberTensor,
            key: EmberTensor,
            value: EmberTensor,
            mask: Optional[EmberTensor] = None,
            times: Optional[EmberTensor] = None) -> EmberTensor:
        """Compute temporal attention.

        Args:
            query: Query tensor [batch, query_len, hidden_size]
            key: Key tensor [batch, key_len, hidden_size]
            value: Value tensor [batch, key_len, hidden_size]
            times: Optional time points [batch, seq_len]
            mask: Optional attention mask [batch, query_len, key_len]

        Returns:
            Attention output [batch, query_len, hidden_size]
        """
        batch_size = shape(query)[0]
        query_len = shape(query)[1]
        key_len = shape(key)[1]
        
        # Add temporal embeddings if enabled
        if self.use_time_embedding and times is not None:
            query = self.time_embedding(query, tensor.slice_tensor(times, [0, 0], [-1, query_len]))
            key = self.time_embedding(key, tensor.slice_tensor(times, [0, 0], [-1, key_len]))
            value = self.time_embedding(value, tensor.slice_tensor(times, [0, 0], [-1, key_len]))
        
        # Project and reshape
        q = self.q_proj(query)
        q = reshape(q, (batch_size, query_len, self.num_heads, self.head_dim))
        q = transpose(q, (0, 2, 1, 3))
        
        k = self.k_proj(key)
        k = reshape(k, (batch_size, key_len, self.num_heads, self.head_dim))
        k = transpose(k, (0, 2, 1, 3))
        
        v = self.v_proj(value)
        v = reshape(v, (batch_size, key_len, self.num_heads, self.head_dim))
        v = transpose(v, (0, 2, 1, 3))
        
        # Compute attention scores
        k_transposed = transpose(k, (0, 1, 3, 2))
        scores = ops.matmul(q, k_transposed)
        
        # Scale scores
        scale_factor = ops.sqrt(tensor.convert_to_tensor(self.head_dim, tensor.float32))
        scores = ops.divide(scores, scale_factor)
        
        # Apply time-based attention if times provided
        if times is not None:
            # Compute time differences [batch, query_len, key_len]
            times_expanded_1 = expand_dims(times, 1)
            times_expanded_2 = expand_dims(times, 2)
            time_diffs = ops.subtract(times_expanded_2, times_expanded_1)
            
            # Project time differences to match query dimensions
            time_diffs = expand_dims(time_diffs, -1)  # [batch, query_len, key_len, 1]
            
            # Reshape query for time gating
            query_expanded = expand_dims(query, 2)
            query_expanded = tensor.tile(query_expanded, [1, 1, key_len, 1])
            
            # Concatenate along feature dimension
            time_features = concatenate([
                query_expanded,
                time_diffs
            ], axis=-1)
            
            # Apply time gating
            time_gates = self.time_gate(time_features)
            
            # Reshape gates to match attention scores
            time_gates = ops.stats.mean(time_gates, axis=-1)
            time_gates = expand_dims(time_gates, 1)
            scores = ops.multiply(scores, time_gates)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = expand_dims(mask, 1)
            mask_condition = ops.equal(mask_expanded, 0)
            # Use a very large negative number instead of float('-inf')
            neg_inf = tensor.convert_to_tensor(-1.0e38, tensor.float32)
            scores = ops.where(mask_condition, neg_inf, scores)
        
        # Apply attention weights
        self._attention_weights = softmax(scores, axis=-1)
        self._attention_weights = self.dropout(self._attention_weights)
        
        # Compute output
        attn_output = ops.matmul(self._attention_weights, v)
        
        # Reshape and project output
        attn_output = transpose(attn_output, (0, 2, 1, 3))
        attn_output = reshape(attn_output, (batch_size, query_len, self.hidden_size))
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def get_attention_weights(self) -> Optional[EmberTensor]:
        """Get last computed attention weights."""
        return self._attention_weights

def create_temporal_attention(hidden_size: int,
                            num_heads: int = 1,
                            dropout: float = 0.1,
                            max_len: int = 1000,
                            use_time_embedding: bool = True) -> TemporalAttention:
    """
    Factory function to create temporal attention mechanism.

    Args:
        hidden_size: Hidden state dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_len: Maximum sequence length
        use_time_embedding: Whether to use temporal embeddings

    Returns:
        Configured temporal attention mechanism
    """
    return TemporalAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
        max_len=max_len,
        use_time_embedding=use_time_embedding
    )