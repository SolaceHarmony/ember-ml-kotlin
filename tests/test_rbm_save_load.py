"""
Test the save_rbm and load_rbm functions.

This module tests the save_rbm and load_rbm functions to ensure they work correctly
across different backends.
"""

import os
import tempfile
import pytest

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.models.rbm import RBMModule
from ember_ml.models.rbm.training import save_rbm, load_rbm


@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = ops.get_backend()
    yield original
    # Ensure original is not None before setting it
    if original is not None:
        ops.set_backend(original)
    else:
        # Default to 'numpy' if original is None
        ops.set_backend('numpy')


@pytest.fixture
def temp_dir():
    """Fixture to create and clean up a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_save_load_rbm(backend_name, original_backend, temp_dir):
    """Test saving and loading an RBM."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create an RBM
        n_visible = 10
        n_hidden = 5
        rbm = RBMModule(n_visible=n_visible, n_hidden=n_hidden)
        
        # Set some custom values to verify they are saved and loaded correctly
        rbm.weights.data = tensor.ones((n_visible, n_hidden))
        rbm.visible_bias.data = tensor.ones(n_visible) * 2
        rbm.hidden_bias.data = tensor.ones(n_hidden) * 3
        
        # Save the RBM
        filepath = os.path.join(temp_dir, "test_rbm.json")
        save_rbm(rbm, filepath)
        
        # Verify the file exists
        assert os.path.exists(filepath)
        
        # Load the RBM
        loaded_rbm = load_rbm(filepath)
        
        # Verify the loaded RBM has the correct parameters
        assert loaded_rbm.n_visible == n_visible
        assert loaded_rbm.n_hidden == n_hidden
        
        # Verify the loaded RBM has the correct weights and biases
        assert ops.all(ops.equal(loaded_rbm.weights.data, tensor.ones((n_visible, n_hidden))))
        assert ops.all(ops.equal(loaded_rbm.visible_bias.data, tensor.ones(n_visible) * 2))
        assert ops.all(ops.equal(loaded_rbm.hidden_bias.data, tensor.ones(n_hidden) * 3))
        
        # Verify the loaded RBM can be used for inference
        input_data = tensor.ones((1, n_visible))
        output = loaded_rbm(input_data)
        
        # Check that the output has the correct shape
        assert tensor.shape(output) == (1, n_hidden)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_save_load_rbm_with_thresholds(backend_name, original_backend, temp_dir):
    """Test saving and loading an RBM with thresholds."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create an RBM
        n_visible = 10
        n_hidden = 5
        rbm = RBMModule(n_visible=n_visible, n_hidden=n_hidden)
        
        # Set some custom values to verify they are saved and loaded correctly
        rbm.weights.data = tensor.ones((n_visible, n_hidden))
        rbm.visible_bias.data = tensor.ones(n_visible) * 2
        rbm.hidden_bias.data = tensor.ones(n_hidden) * 3
        
        # Set thresholds
        rbm.reconstruction_error_threshold = tensor.convert_to_tensor(0.5)
        rbm.free_energy_threshold = tensor.convert_to_tensor(-10.0)
        rbm.n_epochs_trained = tensor.convert_to_tensor(10)
        
        # Save the RBM
        filepath = os.path.join(temp_dir, "test_rbm_with_thresholds.json")
        save_rbm(rbm, filepath)
        
        # Verify the file exists
        assert os.path.exists(filepath)
        
        # Load the RBM
        loaded_rbm = load_rbm(filepath)
        
        # Verify the loaded RBM has the correct parameters
        assert loaded_rbm.n_visible == n_visible
        assert loaded_rbm.n_hidden == n_hidden
        
        # Verify the loaded RBM has the correct weights and biases
        assert ops.all(ops.equal(loaded_rbm.weights.data, tensor.ones((n_visible, n_hidden))))
        assert ops.all(ops.equal(loaded_rbm.visible_bias.data, tensor.ones(n_visible) * 2))
        assert ops.all(ops.equal(loaded_rbm.hidden_bias.data, tensor.ones(n_hidden) * 3))
        
        # Verify the loaded RBM has the correct thresholds
        assert ops.isclose(loaded_rbm.reconstruction_error_threshold, tensor.convert_to_tensor(0.5))
        assert ops.isclose(loaded_rbm.free_energy_threshold, tensor.convert_to_tensor(-10.0))
        assert ops.equal(loaded_rbm.n_epochs_trained, tensor.convert_to_tensor(10))
        
        # Verify the loaded RBM can be used for inference
        input_data = tensor.ones((1, n_visible))
        output = loaded_rbm(input_data)
        
        # Check that the output has the correct shape
        assert tensor.shape(output) == (1, n_hidden)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")