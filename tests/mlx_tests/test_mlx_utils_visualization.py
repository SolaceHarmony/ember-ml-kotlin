import pytest

import matplotlib.pyplot as plt # For testing plotting functions

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.utils import visualization # Import visualization utilities
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for utils.visualization functions

def test_plot_wave():
    # Test plot_wave
    wave_data = tensor.convert_to_tensor(ops.sin(tensor.linspace(0, 10, 100)))
    sample_rate = 100
    title = "Sine Wave Plot"

    # plot_wave creates a plot but doesn't return data.
    # We can test that it runs without errors and potentially check if a figure is created.
    try:
        visualization.plot_wave(wave_data, sample_rate, title)
        # Check if a figure was created (requires matplotlib backend that supports figures)
        assert plt.gcf() is not None
        plt.close() # Close the figure to avoid displaying it during tests
    except Exception as e:
        pytest.fail(f"plot_wave raised an exception: {e}")


def test_plot_spectrogram():
    # Test plot_spectrogram
    wave_data = tensor.convert_to_tensor(ops.sin(tensor.linspace(0, 100, 1000)))
    sample_rate = 1000
    window_size = 128
    hop_length = 64
    title = "Spectrogram Plot"

    # plot_spectrogram creates a plot but doesn't return data.
    # It might rely on scipy or librosa, which might not be installed.
    # We can test that it runs without errors if dependencies are met.
    try:
        visualization.plot_spectrogram(wave_data, sample_rate, window_size, hop_length, title)
        assert plt.gcf() is not None
        plt.close()
    except ImportError:
        pytest.skip("Skipping plot_spectrogram test: scipy or librosa not installed")
    except Exception as e:
        pytest.fail(f"plot_spectrogram raised an exception: {e}")


def test_plot_confusion_matrix():
    # Test plot_confusion_matrix
    # Create a simple confusion matrix (NumPy array)
    cm = tensor.convert_to_tensor([[10, 2], [3, 15]])
    class_names = ['Class A', 'Class B']
    title = "Confusion Matrix"

    try:
        visualization.plot_confusion_matrix(cm, class_names, title)
        assert plt.gcf() is not None
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_confusion_matrix raised an exception: {e}")


def test_plot_roc_curve():
    # Test plot_roc_curve
    # Create dummy FPR, TPR, and AUC values (NumPy arrays/float)
    fpr = tensor.convert_to_tensor([0.0, 0.1, 0.5, 1.0])
    tpr = tensor.convert_to_tensor([0.0, 0.5, 0.8, 1.0])
    roc_auc = 0.75
    title = "ROC Curve"

    try:
        visualization.plot_roc_curve(fpr, tpr, roc_auc, title)
        assert plt.gcf() is not None
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_roc_curve raised an exception: {e}")


def test_plot_precision_recall_curve():
    # Test plot_precision_recall_curve
    # Create dummy precision and recall values (NumPy arrays)
    precision = tensor.convert_to_tensor([1.0, 0.8, 0.6, 0.0])
    recall = tensor.convert_to_tensor([0.0, 0.5, 0.8, 1.0])
    title = "Precision-Recall Curve"

    try:
        visualization.plot_precision_recall_curve(precision, recall, title)
        assert plt.gcf() is not None
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_precision_recall_curve raised an exception: {e}")


def test_plot_learning_curve():
    # Test plot_learning_curve
    # Create dummy train and validation scores (lists or NumPy arrays)
    train_scores = [1.0, 0.8, 0.6, 0.4, 0.2]
    val_scores = [1.2, 0.9, 0.7, 0.5, 0.3]
    title = "Learning Curve"

    try:
        visualization.plot_learning_curve(train_scores, val_scores, title)
        assert plt.gcf() is not None
        plt.close()
    except Exception as e:
        pytest.fail(f"plot_learning_curve raised an exception: {e}")


def test_fig_to_image():
    # Test fig_to_image
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig) # Close the figure immediately

    try:
        pil_image = visualization.fig_to_image(fig)
        # Check if the result is a PIL Image
        from PIL import Image
        assert isinstance(pil_image, Image.Image)
    except ImportError:
        pytest.skip("Skipping fig_to_image test: Pillow not installed")
    except Exception as e:
        pytest.fail(f"fig_to_image raised an exception: {e}")


def test_plot_to_numpy():
    # Test plot_to_numpy
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig) # Close the figure immediately

    try:
        result = visualization.plot_to_numpy(fig)
        # Check if the result is a list or array-like
        assert hasattr(result, '__len__')
        # Check shape (should be height x width x channels)
        assert len(result) > 0
        assert len(result[0]) > 0
        assert len(result[0][0]) in [3, 4]  # RGB or RGBA
        
        # For MLX backend, we expect a list of lists, not a TensorLike object
        # This is intentional to avoid unnecessary conversions
        assert isinstance(result, list)
    except Exception as e:
        pytest.fail(f"plot_to_numpy raised an exception: {e}")


# Add more test functions for other visualization utilities if any exist and are testable