"""
Test the RBM module.

This module tests the RBM module to ensure it works correctly with the backend abstraction layer.
"""

import pytest
import os

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.models.rbm import RBMModule, train_rbm, transform_in_chunks, save_rbm, load_rbm


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
def test_data():
    """Generate test data for RBM training."""
    # Set the backend to numpy for consistent test data
    ops.set_backend('numpy')
    
    # Create random binary data
    n_samples = 100
    n_features = 10
    batch_size = 10
    
    data = tensor.random_bernoulli(
        shape=(n_samples, n_features),
        p=tensor.convert_to_tensor(0.3, dtype=tensor.float32)
    )
    
    # Create batches
    batches = []
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batches.append(tensor.slice(data, [i, 0], [end - i, n_features]))
    
    return data, batches


@pytest.fixture
def trained_rbm(test_data):
    """Create and train an RBM for testing."""
    # Set the backend to numpy for consistent training
    ops.set_backend('numpy')
    
    data, batches = test_data
    
    # Create RBM
    rbm = RBMModule(
        n_visible=10,
        n_hidden=5,
        learning_rate=0.1,
        momentum=0.5,
        weight_decay=0.0001,
        use_binary_states=True
    )
    
    # Train RBM
    train_rbm(
        rbm=rbm,
        data_generator=batches,
        epochs=2,  # Reduced for faster tests
        k=1
    )
    
    return rbm


def test_rbm_training(test_data, trained_rbm, original_backend):
    """Test RBM training."""
    # Set the backend to numpy for consistent testing
    ops.set_backend('numpy')
    
    data, _ = test_data
    rbm = trained_rbm
    
    # Test reconstruction
    reconstructed = rbm.reconstruct(tensor.slice(data, [0, 0], [5, 10]))
    
    # Check that the reconstructed data has the correct shape
    assert tensor.shape(reconstructed) == (5, 10)
    
    # Check that the reconstructed values are between 0 and 1
    assert ops.all(ops.greater_equal(reconstructed, tensor.convert_to_tensor(0)))
    assert ops.all(ops.less_equal(reconstructed, tensor.convert_to_tensor(1)))


def test_rbm_transform(test_data, trained_rbm, original_backend):
    """Test RBM transformation."""
    # Set the backend to numpy for consistent testing
    ops.set_backend('numpy')
    
    _, batches = test_data
    rbm = trained_rbm
    
    # Test transformation
    hidden_probs = transform_in_chunks(rbm, batches[:2])
    
    # Check that the hidden probabilities have the correct shape
    expected_shape = (20, 5)  # 2 batches of 10 samples, 5 hidden units
    assert tensor.shape(hidden_probs) == expected_shape
    
    # Check that the hidden probabilities are between 0 and 1
    assert ops.all(ops.greater_equal(hidden_probs, tensor.convert_to_tensor(0)))
    assert ops.all(ops.less_equal(hidden_probs, tensor.convert_to_tensor(1)))


@pytest.mark.skip(reason="Save/load functionality not fully implemented in NumPy backend")
def test_rbm_save_load(trained_rbm, original_backend, tmp_path):
    """Test RBM save and load."""
    # This test is skipped because the save/load functionality for RBM modules
    # is not fully implemented in the NumPy backend.
    # The original test in test_rbm_module.py also skipped this test.
    pass


def test_rbm_anomaly_detection(test_data, trained_rbm, original_backend):
    """Test RBM anomaly detection."""
    # Set the backend to numpy for consistent testing
    ops.set_backend('numpy')
    
    data, _ = test_data
    rbm = trained_rbm
    
    # Generate normal data (first 10 samples from test data)
    normal_data = tensor.slice(data, [0, 0], [10, 10])
    
    # Generate anomalous data (with higher activation probability)
    anomalous_data = tensor.random_bernoulli(
        shape=(10, 10),
        p=tensor.convert_to_tensor(0.8, dtype=tensor.float32)
    )
    
    # Compute reconstruction errors
    normal_errors = rbm.reconstruction_error(normal_data, per_sample=True)
    anomalous_errors = rbm.reconstruction_error(anomalous_data, per_sample=True)
    
    # Compute mean errors
    mean_normal_error = ops.mean(normal_errors)
    mean_anomalous_error = ops.mean(anomalous_errors)
    
    # Check if anomalous errors are higher on average
    # This is a statistical test, so it might not always pass
    # But it should pass most of the time with the given parameters
    assert ops.greater(mean_anomalous_error, mean_normal_error)