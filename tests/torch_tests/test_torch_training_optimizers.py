import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules # For creating a simple model to test optimizers
from ember_ml.training import Optimizer, SGD, Adam # Import optimizers
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# --- Simple Model for Testing Optimizers ---
class SimpleModel(modules.Module):
    def __init__(self):
        super().__init__()
        # PyTorch backend supports gradients, requires_grad should be True
        self.weight = modules.Parameter(tensor.convert_to_tensor(2.0, requires_grad=True))
        self.bias = modules.Parameter(tensor.convert_to_tensor(1.0, requires_grad=True))

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
    model = SimpleModel()
    optimizer = SGD(learning_rate=0.1)

    # Manually create gradients for testing the step function
    # dLoss/dweight = -18.0, dLoss/dbias = -6.0 (from test_ops_core)
    gradients = [tensor.convert_to_tensor(-18.0), tensor.convert_to_tensor(-6.0)]
    params = list(model.parameters())

    # Perform one optimization step
    optimizer.step(zip(gradients, params))

    # Check if parameters were updated correctly
    # weight = 2.0 - 0.1 * (-18.0) = 2.0 + 1.8 = 3.8
    # bias = 1.0 - 0.1 * (-6.0) = 1.0 + 0.6 = 1.6
    assert ops.allclose(model.weight.data, 3.8).item()
    assert ops.allclose(model.bias.data, 1.6).item()

def test_adam_optimizer_step():
    # Test Adam optimizer step
    model = SimpleModel()
    optimizer = Adam(learning_rate=0.1)

    # Manually create gradients for testing the step function
    # dLoss/dweight = -18.0, dLoss/dbias = -6.0
    gradients = [tensor.convert_to_tensor(-18.0), tensor.convert_to_tensor(-6.0)]
    params = list(model.parameters())

    # Perform one optimization step
    optimizer.step(zip(gradients, params))

    # Check if parameters were updated (exact values depend on Adam's internal state)
    # We can check that they are not the original values and have changed
    assert not ops.allclose(model.weight.data, 2.0).item()
    assert not ops.allclose(model.bias.data, 1.0).item()

    # Perform another step to check internal state update
    gradients2 = [tensor.convert_to_tensor(-10.0), tensor.convert_to_tensor(-4.0)]
    optimizer.step(zip(gradients2, params))

    assert not ops.allclose(model.weight.data, 3.8).item() # Should have changed from previous step
    assert not ops.allclose(model.bias.data, 1.6).item() # Should have changed from previous step


# Add more test functions for other optimizer features:
# test_optimizer_zero_grad(), test_sgd_momentum(), test_adam_with_different_betas(),
# test_optimizer_with_parameter_groups()

# Example structure for test_optimizer_zero_grad
# def test_optimizer_zero_grad():
#     model = SimpleModel()
#     optimizer = SGD(model.parameters(), learning_rate=0.1) # Assuming optimizer takes parameters in init
#
#     # Manually set some gradients
#     model.weight.grad = tensor.convert_to_tensor(5.0)
#     model.bias.grad = tensor.convert_to_tensor(2.0)
#
#     # Zero gradients
#     optimizer.zero_grad()
#
#     # Check if gradients are zeros
#     assert ops.allclose(model.weight.grad, 0.0).item()
#     assert ops.allclose(model.bias.grad, 0.0).item()