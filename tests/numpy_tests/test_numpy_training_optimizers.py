import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules # For creating a simple model to test optimizers
from ember_ml.training import Optimizer, SGD, Adam # Import optimizers
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

# --- Simple Model for Testing Optimizers ---
class SimpleModel(modules.Module):
    def __init__(self):
        super().__init__()
        # NumPy backend does not support gradients, so requires_grad should be False
        self.weight = modules.Parameter(tensor.convert_to_tensor(2.0, requires_grad=False))
        self.bias = modules.Parameter(tensor.convert_to_tensor(1.0, requires_grad=False))

    def forward(self, x):
        return ops.add(ops.multiply(x, self.weight), self.bias)

# Test cases for training.optimizers

def test_optimizer_base_initialization():
    # Test base Optimizer initialization
    optimizer = Optimizer(learning_rate=0.01)
    assert isinstance(optimizer, Optimizer)
    assert optimizer.learning_rate == 0.01

def test_sgd_optimizer_initialization():
    # Test SGD Optimizer initialization
    optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    assert isinstance(optimizer, SGD)
    assert optimizer.learning_rate == 0.001
    assert optimizer.momentum == 0.9
    assert optimizer.nesterov is True

def test_adam_optimizer_initialization():
    # Test Adam Optimizer initialization
    optimizer = Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.99, epsilon=1e-6)
    assert isinstance(optimizer, Adam)
    assert optimizer.learning_rate == 0.0001
    assert optimizer.beta_1 == 0.95
    assert optimizer.beta_2 == 0.99
    assert optimizer.epsilon == 1e-6

def test_sgd_optimizer_step():
    # Test SGD optimizer step
    # Note: NumPy backend does not support gradients.
    # This test will be skipped for NumPy.
    pytest.skip("Optimizer step requires gradient calculation, which is not supported by the NumPy backend.")

def test_adam_optimizer_step():
    # Test Adam optimizer step
    # Note: NumPy backend does not support gradients.
    # This test will be skipped for NumPy.
    pytest.skip("Optimizer step requires gradient calculation, which is not supported by the NumPy backend.")


# Add more test functions for other optimizer features:
# test_optimizer_zero_grad(), test_sgd_momentum(), test_adam_with_different_betas(),
# test_optimizer_with_parameter_groups()
# These tests will also likely need to be skipped for NumPy backend.

# Example structure for test_optimizer_zero_grad
# def test_optimizer_zero_grad():
#     # Note: This test requires gradient support.
#     pytest.skip("Zero grad requires gradient calculation, which is not supported by the NumPy backend.")