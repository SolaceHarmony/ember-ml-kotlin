import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import modules # Import modules for Sequential and BatchNormalization
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

# Test cases for nn.modules.container components

def test_sequential_initialization():
    # Test Sequential initialization with a list of modules
    layer1 = modules.Dense(input_dim=10, units=20)
    layer2 = modules.ReLU()
    layer3 = modules.Dense(input_dim=20, units=1)
    sequential_model = modules.Sequential([layer1, layer2, layer3])

    assert isinstance(sequential_model, modules.Sequential)
    assert len(sequential_model.layers) == 3
    assert sequential_model.layers[0] == layer1
    assert sequential_model.layers[1] == layer2
    assert sequential_model.layers[2] == layer3


def test_sequential_add_layer():
    # Test adding a layer to Sequential
    sequential_model = modules.Sequential()
    layer1 = modules.Dense(input_dim=10, units=20)
    sequential_model.add(layer1)
    assert len(sequential_model.layers) == 1
    assert sequential_model.layers[0] == layer1

    layer2 = modules.ReLU()
    sequential_model.add(layer2)
    assert len(sequential_model.layers) == 2
    assert sequential_model.layers[1] == layer2


def test_sequential_forward_pass():
    # Test Sequential forward pass
    layer1 = modules.Dense(input_dim=10, units=20)
    layer2 = modules.ReLU()
    layer3 = modules.Dense(input_dim=20, units=1)
    sequential_model = modules.Sequential([layer1, layer2, layer3])

    batch_size = 4
    input_tensor = tensor.random_normal((batch_size, 10))

    output = sequential_model(input_tensor)

    assert isinstance(output, tensor.EmberTensor)
    assert tensor.shape(output) == (batch_size, 1)


def test_sequential_get_config_from_config():
    # Test Sequential get_config and from_config
    layer1 = modules.Dense(input_dim=10, units=20)
    layer2 = modules.ReLU()
    original_model = modules.Sequential([layer1, layer2])

    config = original_model.get_config()
    assert isinstance(config, dict)
    assert 'layers' in config
    assert isinstance(config['layers'], list)
    assert len(config['layers']) == 2
    assert config['class_name'] == 'Sequential'

    # Create new model from config
    new_model = modules.Sequential.from_config(config)
    assert isinstance(new_model, modules.Sequential)
    assert len(new_model.layers) == 2
    assert isinstance(new_model.layers[0], modules.Dense)
    assert isinstance(new_model.layers[1], modules.ReLU)
    # Check config of nested modules
    assert new_model.layers[0].get_config()['units'] == 20


def test_batchnormalization_initialization():
    # Test BatchNormalization initialization
    bn_layer = modules.BatchNormalization()
    assert isinstance(bn_layer, modules.BatchNormalization)
    # Default parameters should be set
    assert bn_layer.axis == -1
    assert bn_layer.momentum == 0.99
    assert bn_layer.epsilon == 0.001
    assert bn_layer.center is True
    assert bn_layer.scale is True


def test_batchnormalization_build():
    # Test BatchNormalization build method
    bn_layer = modules.BatchNormalization()
    input_shape = (4, 10, 20) # (batch, seq, features)
    bn_layer.build(input_shape)

    assert bn_layer.built
    # Parameters should be initialized based on feature dimension (axis -1)
    feature_dim = input_shape[-1]
    assert hasattr(bn_layer, 'gamma')
    assert hasattr(bn_layer, 'beta')
    assert hasattr(bn_layer, 'moving_mean')
    assert hasattr(bn_layer, 'moving_variance')
    assert tensor.shape(bn_layer.gamma.data) == (feature_dim,)
    assert tensor.shape(bn_layer.beta.data) == (feature_dim,)
    assert tensor.shape(bn_layer.moving_mean) == (feature_dim,)
    assert tensor.shape(bn_layer.moving_variance) == (feature_dim,)


def test_batchnormalization_forward_training():
    # Test BatchNormalization forward pass in training mode
    bn_layer = modules.BatchNormalization()
    input_shape = (4, 10, 20)
    bn_layer.build(input_shape)

    input_tensor = tensor.random_normal(input_shape)
    output = bn_layer(input_tensor, training=True)

    assert isinstance(output, tensor.EmberTensor)
    assert tensor.shape(output) == input_shape
    # In training, output should have mean close to 0 and std close to 1 per feature
    mean_output = ops.stats.mean(output, axis=(0, 1)) # Mean over batch and sequence
    std_output = ops.stats.std(output, axis=(0, 1))

    assert ops.all(ops.less(ops.abs(mean_output), 0.1)).item() # Allow some tolerance
    assert ops.all(ops.less(ops.abs(ops.subtract(std_output, 1.0)), 0.1)).item()


def test_batchnormalization_forward_inference():
    # Test BatchNormalization forward pass in inference mode
    bn_layer = modules.BatchNormalization()
    input_shape = (4, 10, 20)
    bn_layer.build(input_shape)

    # Manually set moving averages (simulating training)
    mean_val = tensor.convert_to_tensor(np.random.rand(input_shape[-1]).astype(tensor.float32))
    var_val = tensor.convert_to_tensor(np.random.rand(input_shape[-1]).astype(tensor.float32) + 0.1) # Avoid zero variance
    bn_layer.moving_mean = mean_val
    bn_layer.moving_variance = var_val

    input_tensor = tensor.random_normal(input_shape)
    output = bn_layer(input_tensor, training=False)

    assert isinstance(output, tensor.EmberTensor)
    assert tensor.shape(output) == input_shape
    # In inference, output should be normalized using moving averages
    # Manual calculation of expected output: (input - moving_mean) / sqrt(moving_variance + epsilon) * gamma + beta
    expected_output = ops.add(
        ops.divide(
            ops.subtract(input_tensor, bn_layer.moving_mean),
            ops.sqrt(ops.add(bn_layer.moving_variance, bn_layer.epsilon))
        ),
        bn_layer.beta.data # Use .data for parameter tensor
    )
    # Apply gamma (scale)
    expected_output = ops.multiply(expected_output, bn_layer.gamma.data)

    assert ops.allclose(output, expected_output, atol=1e-5).item() # Allow some tolerance


# Add more test functions for other container module features:
# test_sequential_indexing(), test_sequential_len(), test_batchnormalization_no_center(),
# test_batchnormalization_no_scale(), test_batchnormalization_different_axis()