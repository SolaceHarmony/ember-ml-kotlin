"""
Wave RNN model.

This module provides an RNN model for wave-based neural processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from ember_ml.nn import tensor # Added import

class WaveGRUCell(nn.Module):
    """
    GRU cell for wave-based neural processing.
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize the wave GRU cell.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            bias: Whether to use bias
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Reset gate
        self.weight_ir = nn.Parameter(tensor.convert_to_tensor(input_size, hidden_size))
        self.weight_hr = nn.Parameter(tensor.convert_to_tensor(hidden_size, hidden_size))
        self.bias_ir = nn.Parameter(tensor.convert_to_tensor(hidden_size)) if bias else None
        self.bias_hr = nn.Parameter(tensor.convert_to_tensor(hidden_size)) if bias else None
        
        # Update gate
        self.weight_iz = nn.Parameter(tensor.convert_to_tensor(input_size, hidden_size))
        self.weight_hz = nn.Parameter(tensor.convert_to_tensor(hidden_size, hidden_size))
        self.bias_iz = nn.Parameter(tensor.convert_to_tensor(hidden_size)) if bias else None
        self.bias_hz = nn.Parameter(tensor.convert_to_tensor(hidden_size)) if bias else None
        
        # Candidate hidden state
        self.weight_in = nn.Parameter(tensor.convert_to_tensor(input_size, hidden_size))
        self.weight_hn = nn.Parameter(tensor.convert_to_tensor(hidden_size, hidden_size))
        self.bias_in = nn.Parameter(tensor.convert_to_tensor(hidden_size)) if bias else None
        self.bias_hn = nn.Parameter(tensor.convert_to_tensor(hidden_size)) if bias else None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Reset parameters.
        """
        nn.init.xavier_uniform_(self.weight_ir)
        nn.init.xavier_uniform_(self.weight_hr)
        nn.init.xavier_uniform_(self.weight_iz)
        nn.init.xavier_uniform_(self.weight_hz)
        nn.init.xavier_uniform_(self.weight_in)
        nn.init.xavier_uniform_(self.weight_hn)
        
        if self.bias:
            nn.init.zeros_(self.bias_ir)
            nn.init.zeros_(self.bias_hr)
            nn.init.zeros_(self.bias_iz)
            nn.init.zeros_(self.bias_hz)
            nn.init.zeros_(self.bias_in)
            nn.init.zeros_(self.bias_hn)
        
    def forward(self, x: tensor.convert_to_tensor, h: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h: Hidden state tensor of shape (batch_size, hidden_size)
            
        Returns:
            New hidden state tensor of shape (batch_size, hidden_size)
        """
        # Reset gate
        r = torch.sigmoid(
            F.linear(x, self.weight_ir, self.bias_ir) +
            F.linear(h, self.weight_hr, self.bias_hr)
        )
        
        # Update gate
        z = torch.sigmoid(
            F.linear(x, self.weight_iz, self.bias_iz) +
            F.linear(h, self.weight_hz, self.bias_hz)
        )
        
        # Candidate hidden state
        n = torch.tanh(
            F.linear(x, self.weight_in, self.bias_in) +
            r * F.linear(h, self.weight_hn, self.bias_hn)
        )
        
        # New hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new

class WaveGRU(nn.Module):
    """
    GRU for wave-based neural processing.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 bias: bool = True, batch_first: bool = True, dropout: float = 0.0,
                 bidirectional: bool = False):
        """
        Initialize the wave GRU.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            num_layers: Number of GRU layers
            bias: Whether to use bias
            batch_first: Whether batch dimension is first
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create GRU cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(WaveGRUCell(layer_input_size, hidden_size, bias))
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None
        
    def forward(self, x: tensor.convert_to_tensor, h: Optional[tensor.convert_to_tensor] = None) -> Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True,
               otherwise (seq_len, batch_size, input_size)
            h: Initial hidden state tensor of shape (num_layers * num_directions, batch_size, hidden_size)
            
        Returns:
            Tuple of (output, hidden_state)
            - output: Output tensor of shape (batch_size, seq_len, hidden_size * num_directions) if batch_first=True,
                     otherwise (seq_len, batch_size, hidden_size * num_directions)
            - hidden_state: Hidden state tensor of shape (num_layers * num_directions, batch_size, hidden_size)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        batch_size, seq_len, _ = x.size()
        
        if h is None:
            h = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, 
                           device=x.device, dtype=x.dtype)
        
        # Reshape hidden state
        h = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, seq_len, self.hidden_size * self.num_directions, 
                             device=x.device, dtype=x.dtype)
        
        # Process each layer
        layer_input = x
        for layer in range(self.num_layers):
            # Process forward direction
            h_forward = h[layer, 0]
            forward_outputs = []
            
            for t in range(seq_len):
                h_forward = self.cells[layer * self.num_directions](layer_input[:, t], h_forward)
                forward_outputs.append(h_forward)
            
            forward_outputs = torch.stack(forward_outputs, dim=1)
            
            # Process backward direction if bidirectional
            if self.bidirectional:
                h_backward = h[layer, 1]
                backward_outputs = []
                
                for t in range(seq_len - 1, -1, -1):
                    h_backward = self.cells[layer * self.num_directions + 1](layer_input[:, t], h_backward)
                    backward_outputs.append(h_backward)
                
                backward_outputs = torch.stack(backward_outputs[::-1], dim=1)
                
                # Concatenate forward and backward outputs
                layer_output = torch.cat([forward_outputs, backward_outputs], dim=2)
                
                # Update hidden state
                h[layer, 0] = h_forward
                h[layer, 1] = h_backward
            else:
                layer_output = forward_outputs
                
                # Update hidden state
                h[layer, 0] = h_forward
            
            # Apply dropout except for the last layer
            if layer < self.num_layers - 1 and self.dropout_layer is not None:
                layer_output = self.dropout_layer(layer_output)
            
            # Set layer output as input for next layer
            layer_input = layer_output
            
            # If this is the last layer, set as final output
            if layer == self.num_layers - 1:
                outputs = layer_output
        
        # Reshape hidden state
        h = h.view(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, h

class WaveRNN(nn.Module):
    """
    RNN model for wave-based neural processing.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_layers: int = 1, rnn_type: str = 'gru', bidirectional: bool = False,
                 dropout: float = 0.0):
        """
        Initialize the wave RNN.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('gru' or 'lstm')
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # RNN layer
        if self.rnn_type == 'gru':
            self.rnn = WaveGRU(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        
    def forward(self, x: tensor.convert_to_tensor, h: Optional[tensor.convert_to_tensor] = None) -> Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h: Initial hidden state tensor
            
        Returns:
            Tuple of (output, hidden_state)
            - output: Output tensor of shape (batch_size, seq_len, output_size)
            - hidden_state: Hidden state tensor
        """
        # RNN forward
        rnn_output, h = self.rnn(x, h)
        
        # Output layer
        output = self.output_layer(rnn_output)
        
        return output, h

# Convenience function to create a wave RNN
def create_wave_rnn(input_size: int, 
                   hidden_size: int, 
                   output_size: int, 
                   num_layers: int = 1, 
                   rnn_type: str = 'gru', 
                   bidirectional: bool = False,
                   dropout: float = 0.0) -> WaveRNN:
    """
    Create a wave RNN.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        num_layers: Number of RNN layers
        rnn_type: Type of RNN ('gru' or 'lstm')
        bidirectional: Whether to use bidirectional RNN
        dropout: Dropout probability
        
    Returns:
        Wave RNN model
    """
    return WaveRNN(input_size, hidden_size, output_size, num_layers, rnn_type, bidirectional, dropout)