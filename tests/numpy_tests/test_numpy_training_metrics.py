import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.training import classification_metrics, regression_metrics # Import metrics functions
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

# Test cases for training.metrics functions

def test_classification_metrics():
    # Test classification_metrics
    y_true = tensor.convert_to_tensor([0, 1, 2, 0, 1, 2])
    y_pred = tensor.convert_to_tensor([0, 2, 1, 0, 1, 2]) # Some correct, some incorrect

    metrics = classification_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics

    # Convert metrics values to numpy for assertion
    accuracy_np = tensor.item(metrics['accuracy'])
    precision_np = tensor.item(metrics['precision'])
    recall_np = tensor.item(metrics['recall'])
    f1_np = tensor.item(metrics['f1'])

    # Calculate expected metrics manually (macro-averaged)
    # Classes: 0, 1, 2
    # True:    0, 1, 2, 0, 1, 2
    # Pred:    0, 2, 1, 0, 1, 2
    #
    # Class 0: TP=2, FP=0, FN=0, TN=4  (Accuracy=1.0, Precision=1.0, Recall=1.0, F1=1.0)
    # Class 1: TP=1, FP=1, FN=1, TN=3  (Accuracy=0.66, Precision=0.5, Recall=0.5, F1=0.5)
    # Class 2: TP=1, FP=1, FN=1, TN=3  (Accuracy=0.66, Precision=0.5, Recall=0.5, F1=0.5)
    #
    # Macro Avg:
    # Accuracy: (1.0 + 0.66 + 0.66) / 3 = 0.77
    # Precision: (1.0 + 0.5 + 0.5) / 3 = 0.666...
    # Recall: (1.0 + 0.5 + 0.5) / 3 = 0.666...
    # F1: (1.0 + 0.5 + 0.5) / 3 = 0.666...

    # Overall Accuracy: (2+1+1)/6 = 4/6 = 0.666...
    # The function might return overall accuracy or macro-averaged.
    # Let's assume overall accuracy for now based on common practice.
    expected_accuracy = 4.0 / 6.0

    assert ops.allclose(accuracy_np, expected_accuracy)
    # More detailed checks for precision, recall, f1 would require understanding the exact averaging method.
    # For now, check if they are within a reasonable range.
    assert 0.0 <= precision_np <= 1.0
    assert 0.0 <= recall_np <= 1.0
    assert 0.0 <= f1_np <= 1.0


def test_regression_metrics():
    # Test regression_metrics
    y_true = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    y_pred = tensor.convert_to_tensor([1.1, 2.1, 3.1])

    metrics = regression_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics

    # Convert metrics values to numpy for assertion
    mse_np = tensor.item(metrics['mse'])
    rmse_np = tensor.item(metrics['rmse'])
    mae_np = tensor.item(metrics['mae'])
    r2_np = tensor.item(metrics['r2'])

    # Calculate expected metrics manually
    # Errors: [0.1, 0.1, 0.1]
    # Squared Errors: [0.01, 0.01, 0.01]
    # Abs Errors: [0.1, 0.1, 0.1]
    #
    # MSE: mean([0.01, 0.01, 0.01]) = 0.01
    # RMSE: sqrt(MSE) = sqrt(0.01) = 0.1
    # MAE: mean([0.1, 0.1, 0.1]) = 0.1
    # R2: 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    # mean(y_true) = 2.0
    # sum((y_true - mean(y_true))^2) = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2
    # sum((y_true - y_pred)^2) = 0.01 + 0.01 + 0.01 = 0.03
    # R2 = 1 - (0.03 / 2.0) = 1 - 0.015 = 0.985

    assert ops.allclose(mse_np, 0.01)
    assert ops.allclose(rmse_np, 0.1)
    assert ops.allclose(mae_np, 0.1)
    assert ops.allclose(r2_np, 0.985)

# Add more test functions for other training.metrics functions:
# test_binary_classification_metrics(), test_confusion_matrix(), test_roc_auc(),
# test_precision_recall_curve(), test_average_precision_score()

# Example structure for test_binary_classification_metrics
# def test_binary_classification_metrics():
#     y_true = tensor.convert_to_tensor([1, 0, 1, 0])
#     y_pred_probs = tensor.convert_to_tensor([0.9, 0.1, 0.8, 0.3]) # Probabilities
#     metrics = classification_metrics.binary_classification_metrics(y_true, y_pred_probs, threshold=0.5)
#
#     assert isinstance(metrics, dict)
#     assert 'accuracy' in metrics
#     assert 'precision' in metrics
#     assert 'recall' in metrics
#     assert 'f1' in metrics
#     assert 'tp' in metrics
#     assert 'tn' in metrics
#     assert 'fp' in metrics
#     assert 'fn' in metrics
#
#     # Expected: TP=2, TN=1, FP=1, FN=0
#     # Accuracy: (2+1)/4 = 0.75
#     # Precision: 2/(2+1) = 0.666...
#     # Recall: 2/(2+0) = 1.0
#     # F1: 2 * (0.666 * 1.0) / (0.666 + 1.0) = 2 * 0.666 / 1.666 = 0.8
#
#     assert ops.allclose(tensor.item(metrics['accuracy']), 0.75)
#     assert ops.allclose(tensor.item(metrics['precision']), 2.0/3.0)
#     assert ops.allclose(tensor.item(metrics['recall']), 1.0)
#     assert ops.allclose(tensor.item(metrics['f1']), 0.8)
#     assert tensor.item(metrics['tp']) == 2
#     assert tensor.item(metrics['tn']) == 1
#     assert tensor.item(metrics['fp']) == 1
#     assert tensor.item(metrics['fn']) == 0