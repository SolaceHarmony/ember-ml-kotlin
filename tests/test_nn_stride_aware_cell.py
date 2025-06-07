"""
Test the StrideAwareWiredCfCCell and StrideAwareCfC classes.

This module tests the stride-aware cell functionality across all supported backends.
"""

import pytest
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.nn import modules, wirings
from ember_ml.nn.modules.rnn import StrideAwareWiredCfCCell, StrideAwareCfC
from ember_ml.backend import set_backend, get_backend

# Test with all available backends
BACKENDS = ['numpy', 'torch', 'mlx']

@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = get_backend()
    yield original
    # Restore original backend
    if original is not None:
        set_backend(original)
    else:
        # Default to 'numpy' if original is None
        set_backend('numpy')

@pytest.fixture
def wiring():
    """Create a test wiring."""
    return modules.AutoNCP(8, 4)  # Reduced for faster testing

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_stride_aware_cell_creation(backend_name, original_backend, wiring):
    """Test creating a StrideAwareWiredCfCCell."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a cell
        cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=2, time_scale_factor=1.0)
        
        # Verify the cell properties
        assert cell.units == wiring.units
        assert cell.stride_length == 2
        assert cell.time_scale_factor == 1.0
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_stride_aware_cfc_creation(backend_name, original_backend, wiring):
    """Test creating a StrideAwareCfC."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a cell
        cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=2, time_scale_factor=1.0)
        
        # Create a layer
        rnn_layer = StrideAwareCfC(units_or_cell=cell, return_sequences=True)
        
        # Verify the layer properties
        assert rnn_layer.cell == cell
        assert rnn_layer.return_sequences == True
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_stride_aware_cfc_mixed_memory(backend_name, original_backend, wiring):
    """Test creating a StrideAwareCfC with mixed memory."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a cell
        cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=2, time_scale_factor=1.0)
        
        # Create a layer with mixed memory
        rnn_layer = StrideAwareCfC(units_or_cell=cell, return_sequences=True, mixed_memory=True)
        
        # Verify the layer properties
        assert rnn_layer.cell == cell
        assert rnn_layer.return_sequences == True
        assert rnn_layer.mixed_memory == True
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_stride_aware_cell_forward(backend_name, original_backend, wiring):
    """Test the forward pass of a StrideAwareWiredCfCCell."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create input data
        batch_size = 2
        input_dim = wiring.input_dim
        input_data = tensor.random_normal((batch_size, input_dim))
        
        # Create a cell
        cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=2, time_scale_factor=1.0)
        
        # Initialize state
        state = cell.get_initial_state(batch_size=batch_size)
        
        # Forward pass
        output, next_state = cell(input_data, state)
        
        # Verify output shape
        assert output.shape == (batch_size, cell.units)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_stride_aware_cfc_forward(backend_name, original_backend, wiring):
    """Test the forward pass of a StrideAwareCfC."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create input data
        batch_size = 2
        seq_length = 5
        input_dim = wiring.input_dim
        input_data = tensor.random_normal((batch_size, seq_length, input_dim))
        
        # Create a cell
        cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=2, time_scale_factor=1.0)
        
        # Create a layer
        rnn_layer = StrideAwareCfC(units_or_cell=cell, return_sequences=True)
        
        # Forward pass
        output = rnn_layer(input_data)
        
        # Verify output shape
        assert output.shape == (batch_size, seq_length, cell.units)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_serialization(backend_name, original_backend, wiring):
    """Test serialization of StrideAwareWiredCfCCell and StrideAwareCfC."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a cell
        cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=2, time_scale_factor=1.0)
        
        # Get config and recreate
        config = cell.get_config()
        new_cell = StrideAwareWiredCfCCell.from_config(config)
        
        # Verify the new cell
        assert isinstance(new_cell, StrideAwareWiredCfCCell)
        assert new_cell.units == cell.units
        assert new_cell.stride_length == cell.stride_length
        assert new_cell.time_scale_factor == cell.time_scale_factor
        
        # Create a layer
        rnn_layer = StrideAwareCfC(units_or_cell=cell, return_sequences=True)
        
        # Get config and recreate
        config = rnn_layer.get_config()
        new_rnn = StrideAwareCfC.from_config(config)
        
        # Verify the new layer
        assert isinstance(new_rnn, StrideAwareCfC)
        assert new_rnn.return_sequences == rnn_layer.return_sequences
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")