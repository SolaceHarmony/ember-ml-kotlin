"""
Wave Transformer model.

This module provides a transformer model for wave-based neural processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from ember_ml.nn import tensor # Added import

class WaveMultiHeadAttention(nn.Module):
    """
    Multi-head attention for wave-based neural processing.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the wave multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: tensor.convert_to_tensor, 
                key: tensor.convert_to_tensor, 
                value: tensor.convert_to_tensor, 
                mask: Optional[tensor.convert_to_tensor] = None) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = ops.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        output = ops.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        return output

class WaveTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer for wave-based neural processing.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the wave transformer encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.self_attn = WaveMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: tensor.convert_to_tensor, mask: Optional[tensor.convert_to_tensor] = None) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class WaveTransformerEncoder(nn.Module):
    """
    Transformer encoder for wave-based neural processing.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        """
        Initialize the wave transformer encoder.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_layers: Number of encoder layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            WaveTransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: tensor.convert_to_tensor, mask: Optional[tensor.convert_to_tensor] = None) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x

class WaveTransformer(nn.Module):
    """
    Transformer model for wave-based neural processing.
    """
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 num_layers: int, 
                 max_seq_len: int,
                 dropout: float = 0.1):
        """
        Initialize the wave transformer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_layers: Number of encoder layers
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self._init_positional_encoding()
        
        # Encoder
        self.encoder = WaveTransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def _init_positional_encoding(self):
        """
        Initialize positional encoding.
        """
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(torch.log(tensor.convert_to_tensor(10000.0)) / self.d_model))
        
        pos_encoding = torch.zeros(1, self.max_seq_len, self.d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        self.pos_encoding.data = pos_encoding
        
    def forward(self, x: tensor.convert_to_tensor, mask: Optional[tensor.convert_to_tensor] = None) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Encoder
        x = self.encoder(x, mask)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

# Convenience function to create a wave transformer
def create_wave_transformer(d_model: int = 256, 
                           num_heads: int = 8, 
                           d_ff: int = 1024, 
                           num_layers: int = 6, 
                           max_seq_len: int = 1000,
                           dropout: float = 0.1) -> WaveTransformer:
    """
    Create a wave transformer.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of encoder layers
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        
    Returns:
        Wave transformer model
    """
    return WaveTransformer(d_model, num_heads, d_ff, num_layers, max_seq_len, dropout)