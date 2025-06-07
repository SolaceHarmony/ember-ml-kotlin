"""
Test file for the StrideAwareCell and StrideAware classes.

This file demonstrates how to use the StrideAwareCell and StrideAware classes
for multi-timescale processing.
"""

import numpy as np
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.rnn import StrideAwareCell, StrideAware

def test_stride_aware_cell():
    """Test the StrideAwareCell class."""
    print("Testing StrideAwareCell...")
    
    # Create a cell
    cell = StrideAwareCell(
        input_size=10,
        hidden_size=20,
        stride_length=3,
        time_scale_factor=1.5
    )
    
    # Create input
    batch_size = 2
    inputs = tensor.random_normal((batch_size, 10))
    
    # Initialize state
    state = tensor.zeros((batch_size, 20))
    
    # Forward pass
    output, new_state = cell(inputs, state)
    
    print(f"Input shape: {tensor.shape(inputs)}")
    print(f"Output shape: {tensor.shape(output)}")
    print(f"State shape: {tensor.shape(new_state)}")
    
    return output, new_state

def test_stride_aware_layer():
    """Test the StrideAware layer."""
    print("\nTesting StrideAware layer...")
    
    # Create a layer
    layer = StrideAware(
        input_size=10,
        hidden_size=20,
        stride_length=3,
        time_scale_factor=1.5,
        return_sequences=True
    )
    
    # Create input sequence
    batch_size = 2
    seq_length = 5
    inputs = tensor.random_normal((batch_size, seq_length, 10))
    
    # Forward pass
    outputs, final_state = layer(inputs)
    
    print(f"Input shape: {tensor.shape(inputs)}")
    print(f"Output shape: {tensor.shape(outputs)}")
    print(f"Final state shape: {tensor.shape(final_state)}")
    
    # Test with return_sequences=False
    layer_no_seq = StrideAware(
        input_size=10,
        hidden_size=20,
        stride_length=3,
        time_scale_factor=1.5,
        return_sequences=False
    )
    
    outputs_no_seq, final_state_no_seq = layer_no_seq(inputs)
    
    print(f"Output shape (return_sequences=False): {tensor.shape(outputs_no_seq)}")
    
    return outputs, final_state

if __name__ == "__main__":
    print("Testing StrideAware classes for multi-timescale processing...")
    
    # Test cell
    cell_output, cell_state = test_stride_aware_cell()
    
    # Test layer
    layer_output, layer_state = test_stride_aware_layer()
    
    print("\nAll tests completed successfully!")