import pytest
import numpy as np
import matplotlib.pyplot as plt
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
from ember_ml.utils import visualization

# Attempt to import optional dependencies
try:
    import scipy.signal
    _scipy_available = True
except ImportError:
    _scipy_available = False

try:
    import librosa
    _librosa_available = True
except ImportError:
    _librosa_available = False

try:
    from PIL import Image
    _pillow_available = True
except ImportError:
    _pillow_available = False


@pytest.fixture(params=['mlx'])
def set_backend_fixture(request):
    """Fixture to set the backend for each test."""
    set_backend(request.param)
    yield
    # Optional: Reset to a default backend or the original backend after the test
    # set_backend('numpy')
    plt.close('all') # Close all plots after each test

# Helper function to create dummy data
def create_dummy_wave_data(sample_rate=1000, duration=1.0):
    """Creates a dummy sine wave tensor."""
    t = tensor.linspace(0., duration, int(sample_rate * duration))
    from ember_ml import ops
    wave = 0.5 * ops.sin(2. * ops.pi * 5. * t) + 0.1 * np.random.randn(len(t))
    return tensor.convert_to_tensor(wave, dtype=tensor.float32)

def create_dummy_classification_data():
    """Creates dummy data for classification metrics plots."""
    y_true = tensor.convert_to_tensor([0, 1, 2, 0, 1, 2])
    y_pred = tensor.convert_to_tensor([0, 2, 1, 0, 1, 2])
    return y_true, y_pred

def create_dummy_binary_classification_data():
    """Creates dummy data for binary classification metrics plots."""
    y_true = tensor.convert_to_tensor([0, 1, 0, 1, 0, 1])
    y_score = tensor.convert_to_tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])
    return y_true, y_score

def create_dummy_learning_curve_data():
    """Creates dummy data for learning curve plot."""
    epochs = tensor.arange(1, 11)
    train_scores = tensor.linspace(0.9, 0.95, 10)
    val_scores = tensor.linspace(0.85, 0.92, 10)
    return epochs, train_scores, val_scores

# Test cases for visualization functions

def test_plot_wave(set_backend_fixture):
    """Test plot_wave function."""
    wave_data = create_dummy_wave_data()
    try:
        visualization.plot_wave(wave_data, sample_rate=1000, title="Dummy Waveform")
        # Check if a figure was created
        assert plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_wave failed: {e}")

@pytest.mark.skipif(not _scipy_available, reason="SciPy is not available")
def test_plot_spectrogram(set_backend_fixture):
    """Test plot_spectrogram function."""
    wave_data = create_dummy_wave_data()
    try:
        visualization.plot_spectrogram(wave_data, sample_rate=1000, window_size=256, hop_length=128, title="Dummy Spectrogram")
        # Check if a figure was created
        assert plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_spectrogram failed: {e}")

def test_plot_confusion_matrix(set_backend_fixture):
    """Test plot_confusion_matrix function."""
    y_true, y_pred = create_dummy_classification_data()
    # Need to compute confusion matrix first (assuming a function exists or using numpy)
    # Using numpy for confusion matrix for now, as ops.stats.confusion_matrix is not yet implemented
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    cm = sk_confusion_matrix(y_true, y_pred)
    class_names = ['Class 0', 'Class 1', 'Class 2']
    try:
        visualization.plot_confusion_matrix(cm, class_names, title="Dummy Confusion Matrix")
        # Check if a figure was created
        assert plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_confusion_matrix failed: {e}")

# Note: ROC AUC and Precision-Recall curves require specific metric computations
# which might not be fully implemented in ops.stats yet.
# These tests will check if the plotting functions run with dummy data.
def test_plot_roc_curve(set_backend_fixture):
    """Test plot_roc_curve function."""
    # Dummy data for ROC curve (fpr, tpr, auc)
    fpr = tensor.convert_to_tensor([0.0, 0.1, 0.5, 1.0])
    tpr = tensor.convert_to_tensor([0.0, 0.5, 0.7, 1.0])
    roc_auc = 0.75
    try:
        visualization.plot_roc_curve(fpr, tpr, roc_auc, title="Dummy ROC Curve")
        # Check if a figure was created
        assert plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_roc_curve failed: {e}")

def test_plot_precision_recall_curve(set_backend_fixture):
    """Test plot_precision_recall_curve function."""
    # Dummy data for Precision-Recall curve (precision, recall)
    precision = tensor.convert_to_tensor([1.0, 0.8, 0.6, 0.4, 0.0])
    recall = tensor.convert_to_tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    try:
        visualization.plot_precision_recall_curve(precision, recall, title="Dummy Precision-Recall Curve")
        # Check if a figure was created
        assert plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_precision_recall_curve failed: {e}")

def test_plot_learning_curve(set_backend_fixture):
    """Test plot_learning_curve function."""
    epochs, train_scores, val_scores = create_dummy_learning_curve_data()
    try:
        visualization.plot_learning_curve(train_scores, val_scores, title="Dummy Learning Curve")
        # Check if a figure was created
        assert plt.gcf() is not None
    except Exception as e:
        pytest.fail(f"plot_learning_curve failed: {e}")

@pytest.mark.skipif(not _pillow_available, reason="Pillow is not available")
def test_fig_to_image(set_backend_fixture):
    """Test fig_to_image function."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        img = visualization.fig_to_image(fig)
        assert isinstance(img, Image.Image)
    except Exception as e:
        pytest.fail(f"fig_to_image failed: {e}")

def test_plot_to_numpy(set_backend_fixture):
    """Test plot_to_numpy function."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        result = visualization.plot_to_numpy(fig)
        # For MLX backend, we expect a list of lists, not a TensorLike object
        assert isinstance(result, list)
        # Check dimensions (height, width, channels)
        assert len(result) > 0  # height
        assert len(result[0]) > 0  # width
        assert len(result[0][0]) in [3, 4]  # RGB or RGBA channels
    except Exception as e:
        pytest.fail(f"plot_to_numpy failed: {e}")

# Add comments indicating areas where more detailed tests could be added
# TODO: Add tests for specific plot content and correctness where possible.
# TODO: Add tests for edge cases and invalid inputs for plotting functions.
# TODO: Update tests for plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#       once corresponding metric functions are fully implemented in ops.stats.