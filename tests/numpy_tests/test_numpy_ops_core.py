import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules # For creating a simple model to test gradients
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

# Test cases for core ops functions (not in submodules)

def test_gradients():
    # Test gradient calculation for a simple function
    # f(x) = x^2 + 2x + 1
    # df/dx = 2x + 2
    x = tensor.convert_to_tensor(3.0, requires_grad=True)
    
    def func(x):
        return ops.add(ops.add(ops.square(x), ops.multiply(2.0, x)), 1.0)

    # Calculate gradient
    # Note: NumPy backend does not support automatic differentiation.
    # This test should either be skipped for NumPy or adapted to test
    # a different aspect if gradients are manually implemented for NumPy.
    # For now, we'll skip this test for NumPy.
    pytest.skip("Gradient calculation is not supported by the NumPy backend.")

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
    # This will also be skipped for NumPy
    pytest.skip("Gradient calculation is not supported by the NumPy backend.")


def test_eval():
    # Test ops.eval (forces computation)
    # NumPy backend is typically not lazy, so eval might not have a significant effect,
    # but the function should still exist and not raise errors.
    x = tensor.random_normal((10, 10))
    # Perform an operation
    y = ops.add(x, 1.0)

    # Calling eval should not raise an error
    ops.eval(y)

    # Check if the tensor is usable after eval
    assert tensor.shape(y) == (10, 10)
    assert tensor.dtype(y) == tensor.float32 # Assuming default float32

    # Test eval with multiple tensors
    z = ops.multiply(y, 2.0)
    ops.eval(y, z)
    assert tensor.shape(z) == (10, 10)

# Add more test functions for other core ops functions if any are missed