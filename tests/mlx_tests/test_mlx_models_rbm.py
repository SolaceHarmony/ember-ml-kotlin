import pytest
import numpy as np
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.models.rbm import RestrictedBoltzmannMachine
from ember_ml.nn.modules import Module, Parameter # Needed for isinstance checks
from ember_ml.nn.tensor import EmberDType # For dtype conversion

@pytest.fixture(params=['mlx'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')

# Helper function to create dummy data
def create_dummy_rbm_data(shape=(100, 784)):
    """Creates dummy binary data for RBM testing."""
    # RBMs typically work with binary or real-valued data between 0 and 1
    return tensor.random_uniform(shape, minval=0.0, maxval=1.0, dtype=tensor.float32)

# Test cases for RestrictedBoltzmannMachine

def test_rbm_initialization(set_backend_fixture):
    """Test RestrictedBoltzmannMachine initialization."""
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(visible_size=n_visible, hidden_size=n_hidden, device="cpu")
    assert isinstance(rbm, RestrictedBoltzmannMachine)
    assert isinstance(rbm, Module)
    assert rbm.visible_size == n_visible
    assert rbm.hidden_size == n_hidden
    # Check if weights and biases are initialized as Parameters
    assert hasattr(rbm, 'weights') and isinstance(rbm.weights, Parameter)
    assert hasattr(rbm, 'hidden_bias') and isinstance(rbm.hidden_bias, Parameter)
    assert hasattr(rbm, 'visible_bias') and isinstance(rbm.visible_bias, Parameter)
    # Check shapes of initialized parameters
    assert tensor.shape(rbm.weights) == (n_visible, n_hidden)
    assert tensor.shape(rbm.hidden_bias) == (n_hidden,)
    assert tensor.shape(rbm.visible_bias) == (n_visible,)

# Note: Training involves iterative updates and convergence, which is hard to test
# deterministically across backends. This test focuses on basic execution without
# checking for specific convergence criteria.
def test_rbm_train_basic_execution(set_backend_fixture):
    """Test RestrictedBoltzmannMachine train method (basic execution)."""
    n_visible = 10
    n_hidden = 5
    rbm = RestrictedBoltzmannMachine(visible_size=n_visible, hidden_size=n_hidden, device="cpu")
    training_data = create_dummy_rbm_data(shape=(20, n_visible)) # 20 samples
    epochs = 5
    learning_rate = 0.1

    try:
        # Train for a few epochs
        from ember_ml.models.rbm.rbm import train_rbm
        train_rbm(rbm, training_data, num_epochs=epochs, learning_rate=learning_rate)
        # If no exceptions are raised, assume basic execution is successful
        assert True
    except Exception as e:
        pytest.fail(f"RBM train method failed: {e}")

def test_rbm_transform_shape(set_backend_fixture):
    """Test RestrictedBoltzmannMachine transform method shape."""
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(visible_size=n_visible, hidden_size=n_hidden)
    input_data = create_dummy_rbm_data(shape=(10, n_visible)) # 10 samples
    features = rbm.compute_hidden_probabilities(input_data)
    # Transform should output features in the hidden layer dimension
    assert tensor.shape(features) == (10, n_hidden)

def test_rbm_generate_shape(set_backend_fixture):
    """Test RestrictedBoltzmannMachine generate method shape."""
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(visible_size=n_visible, hidden_size=n_hidden)
    n_samples = 10
    generated_samples = rbm.sample(num_samples=n_samples)
    # Generated samples should be in the visible layer dimension
    assert tensor.shape(generated_samples) == (n_samples, n_visible)
    # Check if generated samples are binary (or close to binary for real-valued RBMs)
    # For binary RBMs, values should be 0 or 1. For real-valued, they might be probabilities.
    # Let's check if values are within [0, 1] range for now.
    assert ops.all(ops.greater_equal(generated_samples, tensor.convert_to_tensor(0.0)))
    assert ops.all(ops.less_equal(generated_samples, tensor.convert_to_tensor(1.0)))


def test_rbm_anomaly_score_shape(set_backend_fixture):
    """Test RestrictedBoltzmannMachine anomaly_score method shape."""
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(visible_size=n_visible, hidden_size=n_hidden)
    input_data = create_dummy_rbm_data(shape=(10, n_visible)) # 10 samples
    anomaly_scores = rbm.anomaly_score(input_data)
    # Anomaly score should be a scalar for each sample
    assert tensor.shape(anomaly_scores) == (10,)

def test_rbm_is_anomaly_shape_and_type(set_backend_fixture):
    """Test RestrictedBoltzmannMachine is_anomaly method shape and type."""
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(visible_size=n_visible, hidden_size=n_hidden)
    input_data = create_dummy_rbm_data(shape=(10, n_visible)) # 10 samples
    # is_anomaly requires fitting the anomaly detector first (often part of train or a separate method)
    # Assuming a threshold is set internally or can be passed.
    # For this test, we'll just check the output shape and dtype (boolean)
    # Note: The actual anomaly detection logic is complex and backend-dependent in results.
    # This test focuses on the output structure.
    try:
        is_anomaly_result = rbm.is_anomaly(input_data)
        assert tensor.shape(is_anomaly_result) == (10,)
        # Just check if it's a boolean type
        assert 'bool' in str(is_anomaly_result.dtype).lower()
    except Exception as e:
         # If is_anomaly requires prior fitting or a threshold, it might raise an error.
         # For now, we catch it and fail the test with a specific message.
         pytest.fail(f"RBM is_anomaly method failed (possibly requires prior fitting or threshold): {e}")


# TODO: Add more detailed tests for RBM functionality, including:
# - Checking parameter updates during training (requires access to gradients and optimizer logic)
# - Verifying the correctness of transform, generate, anomaly_score, and is_anomaly outputs
#   for simple, known inputs (if possible and not overly backend-dependent).
# - Testing different RBM configurations (e.g., different activation functions, learning rates).
# - Testing with different data types.