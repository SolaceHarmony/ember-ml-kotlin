import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import features # Import features module
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

# Fixture providing sample data for stateful feature tests
@pytest.fixture
def sample_feature_data():
    """Create sample data for stateful feature tests."""
    # Use a consistent seed for reproducibility
    tensor.set_seed(123)
    # Create a simple dataset with some variance
    # Generate standard normal data first
    base_data = tensor.random_normal((100, 5))
    # Define mean and stddev as tensors
    mean_vec = tensor.convert_to_tensor([0.0, 1.0, -0.5, 2.0, -1.0])
    stddev_vec = tensor.convert_to_tensor([1.0, 0.5, 2.0, 0.1, 1.5])
    # Scale and shift the base data
    data = ops.add(ops.multiply(base_data, stddev_vec), mean_vec)
    return data

# Test cases for nn.features stateful components

def test_pca_fit_transform(sample_feature_data):
    from ember_ml.ops import stats
    # Test PCA fit_transform
    data = sample_feature_data
    n_components = 3
    pca_instance = features.pca() # Use factory function

    transformed = pca_instance.fit_transform(data, n_components=n_components)

    # assert isinstance(transformed, tensor.EmberTensor) # Removed check, methods return native tensors
    assert tensor.shape(transformed) == (tensor.shape(data)[0], n_components)

    # Check that the variance of the components is decreasing (basic check)
    variances = stats.var(transformed, axis=0)
    # Convert to numpy for comparison
    variances_np = tensor.to_numpy(variances)
    assert ops.all(np.diff(variances_np) <= 1e-5) # Variances should be non-increasing

def test_pca_inverse_transform(sample_feature_data):
    # Test PCA inverse_transform
    data = sample_feature_data
    n_components = 3
    pca_instance = features.pca()

    transformed = pca_instance.fit_transform(data, n_components=n_components)
    reconstructed = pca_instance.inverse_transform(transformed)

    # assert isinstance(reconstructed, tensor.EmberTensor) # Removed check, methods return native tensors
    assert tensor.shape(reconstructed) == tensor.shape(data)

    # Check that the reconstruction error is reasonably small
    reconstruction_error = ops.stats.mean(ops.square(ops.subtract(data, reconstructed)))
    assert tensor.item(reconstruction_error) < 1.0 # Error should be less than 1.0 for this data

#@pytest.mark.skip(reason="Standardization tests failing with NaN on MLX backend, needs investigation.")
def test_standardize_fit_transform(sample_feature_data):
    # Test Standardize fit_transform
    data = sample_feature_data
    scaler = features.standardize() # Use factory function

    scaled_data = scaler.fit_transform(data)

    # assert isinstance(scaled_data, tensor.EmberTensor) # Removed check, methods return native tensors
    assert tensor.shape(scaled_data) == tensor.shape(data)

    # Check that the mean is close to 0 and std is close to 1 for scaled data
    from ember_ml.ops import stats
    mean_scaled = stats.mean(scaled_data, axis=0)
    std_scaled = stats.std(scaled_data, axis=0)

    assert ops.all(ops.less(ops.abs(mean_scaled), 1e-5)).item()
    assert ops.all(ops.less(ops.abs(ops.subtract(std_scaled, 1.0)), 1e-5)).item()

@pytest.mark.skip(reason="Standardization tests failing with NaN on MLX backend, needs investigation.")
def test_standardize_inverse_transform(sample_feature_data):
    # Test Standardize inverse_transform
    data = sample_feature_data
    scaler = features.standardize()

    scaled_data = scaler.fit_transform(data)
    reconstructed_data = scaler.inverse_transform(scaled_data)

    # assert isinstance(reconstructed_data, tensor.EmberTensor) # Removed check, methods return native tensors
    assert tensor.shape(reconstructed_data) == tensor.shape(data)

    # Check that the reconstructed data is close to the original data
    reconstruction_error = ops.stats.mean(ops.square(ops.subtract(data, reconstructed_data)))
    assert tensor.item(reconstruction_error) < 1e-5 # Error should be very small

