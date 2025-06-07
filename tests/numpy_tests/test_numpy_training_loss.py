import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.training.loss import Loss, MSELoss, CrossEntropyLoss # Import loss classes
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

# Test cases for training.loss functions (classes)

def test_mseloss():
    # Test MSELoss class
    loss_fn = MSELoss()
    y_true = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    y_pred = tensor.convert_to_tensor([1.1, 2.1, 3.1])
    result = loss_fn(y_true, y_pred)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (expected MSE is 0.01)
    assert ops.allclose(result_np, 0.01)

    # Test with different shapes (broadcasting)
    y_true_matrix = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred_matrix = tensor.convert_to_tensor([[1.1, 2.1], [3.1, 4.1]])
    result_matrix = loss_fn(y_true_matrix, y_pred_matrix)
    assert ops.allclose(tensor.to_numpy(result_matrix), 0.01)

def test_crossentropy_loss():
    # Test CrossEntropyLoss class
    loss_fn = CrossEntropyLoss(from_logits=False)
    y_true = tensor.convert_to_tensor([1.0, 0.0, 1.0, 0.0])
    y_pred = tensor.convert_to_tensor([0.9, 0.1, 0.8, 0.3]) # Probabilities
    result = loss_fn(y_true, y_pred)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (compare with numpy calculation)
    # Avoid log(0) by clipping predictions
    epsilon = 1e-7
    y_pred_clipped = ops.clip(tensor.to_numpy(y_pred), epsilon, 1.0 - epsilon)
    y_true_np = tensor.to_numpy(y_true)
    expected_np = -stats.mean(y_true_np * np.log(y_pred_clipped) + (1 - y_true_np) * np.log(1 - y_pred_clipped))
    assert ops.allclose(result_np, expected_np)

    # Test with logits
    loss_fn_logits = CrossEntropyLoss(from_logits=True)
    y_pred_logits = tensor.convert_to_tensor([2.2, -2.2, 1.4, -0.8]) # Logits
    result_logits = loss_fn_logits(y_true, y_pred_logits)
    assert ops.allclose(tensor.to_numpy(result_logits), expected_np) # Should be same as probability version

# Add more test functions for other training.loss functions:
# test_mean_absolute_error_loss(), test_categorical_crossentropy_loss(),
# test_sparse_categorical_crossentropy_loss(), test_huber_loss_class(), test_log_cosh_loss_class()

# Example structure for test_mean_absolute_error_loss
# def test_mean_absolute_error_loss():
#     loss_fn = MeanAbsoluteErrorLoss() # Assuming class name
#     y_true = tensor.convert_to_tensor([1.0, 2.0, 3.0])
#     y_pred = tensor.convert_to_tensor([1.1, 2.1, 3.1])
#     result = loss_fn(y_true, y_pred)
#     assert ops.allclose(tensor.to_numpy(result), 0.1)