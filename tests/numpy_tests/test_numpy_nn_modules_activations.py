import pytest
import numpy as np # For comparison with known correct results
import math # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import activations as activations_module # Import module for classes
from ember_ml.nn.modules.activations import get_activation # Import helper function
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("mlx")

# Test cases for nn.modules.activations (Module classes and functional)

# --- Module Class Tests ---

def test_relu_module():
    # Test ReLU Module
    relu = activations_module.ReLU()
    x = tensor.convert_to_tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 2.0]])
    result = relu(x)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (ReLU(x) = max(0, x))
    assert ops.allclose(result_np, stats.maximum(0.0, tensor.to_numpy(x)))

def test_tanh_module():
    # Test Tanh Module
    tanh = activations_module.Tanh()
    x = tensor.convert_to_tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 2.0]])
    result = tanh(x)
    assert ops.allclose(tensor.to_numpy(result), nn.modules.activations.tanh(tensor.to_numpy(x)))

def test_sigmoid_module():
    # Test Sigmoid Module
    sigmoid = activations_module.Sigmoid()
    x = tensor.convert_to_tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 2.0]])
    result = sigmoid(x)
    # Use a helper for sigmoid calculation to avoid direct backend calls
    def sigmoid_np(x_np):
        return 1.0 / (1.0 + ops.exp(-x_np))
    assert ops.allclose(tensor.to_numpy(result), sigmoid_np(tensor.to_numpy(x)))

def test_softmax_module():
    # Test Softmax Module
    softmax = activations_module.Softmax(axis=-1)
    x = tensor.convert_to_tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    result = softmax(x)
    # Use numpy.softmax for expected values
    expected_np = ops.exp(tensor.to_numpy(x)) / stats.sum(ops.exp(tensor.to_numpy(x)), axis=-1, keepdims=True)
    assert ops.allclose(tensor.to_numpy(result), expected_np)

def test_dropout_module():
    # Test Dropout Module (training vs inference)
    dropout_rate = 0.5
    dropout = activations_module.Dropout(rate=dropout_rate, seed=42)
    x = tensor.ones((10, 10))

    # During training (dropout should be active)
    result_train = dropout(x, training=True)
    result_train_np = tensor.to_numpy(result_train)
    # Check that some elements are zero and others are scaled
    # Note: NumPy backend might not have a true random dropout implementation
    # This test might need adaptation or skipping for NumPy if it doesn't support random dropout
    # For now, we'll check if it's not identical to the input (if rate > 0)
    if dropout_rate > 0:
        assert not ops.allclose(result_train_np, tensor.to_numpy(x))
    else:
        assert ops.allclose(result_train_np, tensor.to_numpy(x))


    # During inference (dropout should not be active)
    result_eval = dropout(x, training=False)
    result_eval_np = tensor.to_numpy(result_eval)
    assert ops.allclose(result_eval_np, tensor.to_numpy(x)) # Should be equal to input

# Add more test functions for other activation modules:
# test_softplus_module(), test_lecun_tanh_module()


# --- Functional Activation Tests ---

def test_get_activation():
    # Test get_activation helper function
    relu_fn = get_activation("relu")
    tanh_fn = get_activation("tanh")
    sigmoid_fn = get_activation("sigmoid")

    x = tensor.convert_to_tensor([[-1.0, 0.0, 1.0]])

    assert ops.allclose(tensor.to_numpy(relu_fn(x)), stats.maximum(0.0, tensor.to_numpy(x)))
    assert ops.allclose(tensor.to_numpy(tanh_fn(x)), nn.modules.activations.tanh(tensor.to_numpy(x)))
    def sigmoid_np(x_np):
        return 1.0 / (1.0 + ops.exp(-x_np))
    assert ops.allclose(tensor.to_numpy(sigmoid_fn(x)), sigmoid_np(tensor.to_numpy(x)))

    # Test with invalid activation name (should raise AttributeError)
    with pytest.raises(AttributeError):
        get_activation("invalid_activation")

# Add more test functions for other functional activations:
# test_functional_softplus(), test_functional_lecun_tanh()