import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules # For creating a simple model to test gradients
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

# Test cases for core ops functions (not in submodules)

def test_gradients():
    # Test gradient calculation for a simple function
    # f(x) = x^2 + 2x + 1
    # df/dx = 2x + 2
    x = tensor.convert_to_tensor(3.0, requires_grad=True)
    
    def func(x):
        return ops.add(ops.add(ops.square(x), ops.multiply(2.0, x)), 1.0)

    # Calculate gradient
    gradient = ops.gradients(func(x), x)

    # Convert to numpy for assertion
    gradient_np = tensor.to_numpy(gradient)

    # Assert correctness (2*3 + 2 = 8)
    assert ops.allclose(gradient_np, 8.0)

    # Test gradient for a simple model
    class SimpleModel(modules.Module):
        def __init__(self):
            super().__init__()
            self.weight = modules.Parameter(tensor.convert_to_tensor(2.0, requires_grad=True))
            self.bias = modules.Parameter(tensor.convert_to_tensor(1.0, requires_grad=True))

        def forward(self, x):
            return ops.add(ops.multiply(x, self.weight), self.bias)

    model = SimpleModel()
    input_data = tensor.convert_to_tensor(3.0)
    
    # Calculate loss (e.g., MSE)
    target = tensor.convert_to_tensor(10.0)
    output = model(input_data)
    loss = ops.mse(target, output) # (10 - (3*2 + 1))^2 = (10 - 7)^2 = 3^2 = 9

    # Calculate gradients of loss with respect to parameters
    gradients = ops.gradients(loss, model.parameters())
    
    # Convert gradients to numpy for assertion
    grad_weight_np = tensor.to_numpy(gradients[0])
    grad_bias_np = tensor.to_numpy(gradients[1])

    # Assert correctness
    # Loss = (target - (input * weight + bias))^2
    # dLoss/dweight = 2 * (target - (input * weight + bias)) * (-input)
    # dLoss/dbias = 2 * (target - (input * weight + bias)) * (-1)
    # At input=3, target=10, weight=2, bias=1:
    # dLoss/dweight = 2 * (10 - (3*2 + 1)) * (-3) = 2 * (10 - 7) * (-3) = 2 * 3 * (-3) = -18
    # dLoss/dbias = 2 * (10 - (3*2 + 1)) * (-1) = 2 * (10 - 7) * (-1) = 2 * 3 * (-1) = -6
    assert ops.allclose(grad_weight_np, -18.0)
    assert ops.allclose(grad_bias_np, -6.0)


def test_eval():
    # Test ops.eval (forces computation)
    # PyTorch backend can be lazy, so eval should force computation.
    x = tensor.random_normal((10, 10))
    # Perform an operation that might be lazy
    y = ops.add(x, 1.0)

    # Calling eval should force the computation
    ops.eval(y)

    # After eval, the tensor should be computed and usable
    assert tensor.shape(y) == (10, 10)
    assert tensor.dtype(y) == tensor.float32 # Assuming default float32

    # Test eval with multiple tensors
    z = ops.multiply(y, 2.0)
    ops.eval(y, z)
    assert tensor.shape(z) == (10, 10)

# Add more test functions for other core ops functions if any are missed